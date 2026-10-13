"""M5 離線重關聯 —— 用「決定當下拿不到的資訊」回頭修正 chef_id。

為什麼需要這一層(前三輪的教訓):

前三輪都在優化**當下那一刻**的判斷。但廚師中途停下來洗手 30 秒才走到下一台
鏡頭時,那一刻的證據與「一個陌生人剛好在那時出現」在統計上是重疊的
——再多即時證據也分不出來(見 docs/M5_模擬預先登記_逗留_20260825.md R5)。

本模組改變的不是證據,而是**什麼時候做決定**。延後幾分鐘後,有兩件當下
沒有的資訊可用:

  1. **A 後來到底有沒有再出現。** 若 A 消失後 B 才第一次出現,而且 A 從此
     再無蹤影 —— 那 A 與 B 十之八九是同一人。兩者若曾並存,則必然不是。
  2. **現場到底有幾個人。** 排班表說 4 位,系統卻開了 7 個 chef_id,
     那**必定**有 3 個是碎裂 —— 這是不需要任何影像證據的事實。

可行性:spec 的資料流是 M6 → M7(非同步佇列)→ M8 資料庫 → M9 調查前端,
最終產物是給人事後查的紀錄,不是即時警報。所以線上先給暫定編號、
離線精修後再寫入,對使用者沒有差別。這是 MTMC 領域的標準做法。

⚠ 架構相依:離線修正會讓 chef_id **事後改變**。M8/M9 必須能處理
  「這筆事件的 chef_id 後來被更正了」,否則報表會前後不一致。
"""
import math
from collections import defaultdict


class Sighting:
    """一次「某 chef_id 在某鏡頭被看到」的紀錄。重關聯以此為單位。"""

    __slots__ = ("chef_id", "camera_id", "t_sec", "embedding", "zone", "bbox")

    def __init__(self, chef_id, camera_id, t_sec, embedding=None, zone=None, bbox=None):
        self.chef_id = chef_id
        self.camera_id = camera_id
        self.t_sec = t_sec
        self.embedding = embedding
        self.zone = zone
        self.bbox = bbox


class OfflineRefiner:
    """收集 sighting,定期回頭合併「應該是同一人」的 chef_id。

    合併只在**證據 + 硬約束都成立**時發生:
      · 硬約束:兩個 chef_id 的出現時段不得重疊(重疊 = 同時存在 = 必然不同人)
      · 證據:轉場 LLR ≥ min_merge_llr
    """

    def __init__(self, topology, expected_headcount=None,
                 refine_window_s=900.0, min_merge_llr=1.0,
                 use_headcount=True, use_refine=True):
        self.topo = topology
        self.headcount = expected_headcount
        self.window = float(refine_window_s)
        self.min_llr = float(min_merge_llr)
        self.use_headcount = use_headcount
        self.use_refine = use_refine
        self.sightings = []
        self.merged_into = {}          # chef_id -> 合併後的代表 chef_id

    def record(self, chef_id, camera_id, t_sec, embedding=None, zone=None, bbox=None):
        self.sightings.append(Sighting(chef_id, camera_id, t_sec, embedding, zone, bbox))

    def resolve(self, chef_id):
        """查某個暫定 chef_id 最終被歸到哪個代表 id(union-find 的 find)。"""
        seen = set()
        while chef_id in self.merged_into and chef_id not in seen:
            seen.add(chef_id)
            chef_id = self.merged_into[chef_id]
        return chef_id

    def _spans(self):
        """每個 chef_id 的出現區間 [首次, 末次] 與首末筆 sighting。"""
        first, last = {}, {}
        for s in self.sightings:
            cid = self.resolve(s.chef_id)
            if cid not in first or s.t_sec < first[cid].t_sec:
                first[cid] = s
            if cid not in last or s.t_sec > last[cid].t_sec:
                last[cid] = s
        return first, last

    def _pair_llr(self, a_last, b_first):
        """A 的最後一次出現 → B 的第一次出現,這個轉場有多可信。"""
        ok, llr = self.topo.transit_llr(a_last.camera_id, a_last.t_sec,
                                        b_first.camera_id, b_first.t_sec)
        if not ok:
            return None
        if a_last.embedding is not None and b_first.embedding is not None:
            import numpy as np
            llr += self.topo.app_lr.llr(float(np.dot(a_last.embedding, b_first.embedding)))
        llr += self.topo.direction_llr(a_last.camera_id, b_first.camera_id,
                                       a_last.zone, b_first.zone)
        return llr

    def _candidates(self):
        """所有「時段不重疊、且 A 結束在 B 開始之前」的配對,附 LLR。"""
        first, last = self._spans()
        ids = sorted(first)
        out = []
        for a in ids:
            for b in ids:
                if a == b:
                    continue
                # 硬約束:B 必須在 A 結束之後才開始(否則兩者並存,不可能同一人)
                if first[b].t_sec <= last[a].t_sec:
                    continue
                if first[b].t_sec - last[a].t_sec > self.window:
                    continue
                llr = self._pair_llr(last[a], first[b])
                if llr is not None:
                    out.append((llr, a, b))
        out.sort(reverse=True)
        return out

    def refine(self):
        """跑一次離線精修。回傳 (證據合併數, 人數約束強制合併數)。"""
        n_ev = n_hc = 0

        if self.use_refine:                       # G1:證據足夠就合併
            for llr, a, b in self._candidates():
                if llr < self.min_llr:
                    break                          # 已按 LLR 排序,後面只會更低
                ra, rb = self.resolve(a), self.resolve(b)
                if ra != rb:
                    self.merged_into[rb] = ra
                    n_ev += 1

        if self.use_headcount and self.headcount:  # G2:數量超過排班就必定有碎裂
            while len({self.resolve(s.chef_id) for s in self.sightings}) > self.headcount:
                cands = [(l, a, b) for l, a, b in self._candidates()
                         if self.resolve(a) != self.resolve(b)]
                if not cands:
                    break                          # 找不到合法配對 → 寧可不合併
                llr, a, b = cands[0]
                if llr < self.min_llr - 2.0:       # 即使被逼,也有下限:寧缺勿濫
                    break
                self.merged_into[self.resolve(b)] = self.resolve(a)
                n_hc += 1
        return n_ev, n_hc

    def stats(self):
        ids = {self.resolve(s.chef_id) for s in self.sightings}
        return {"sightings": len(self.sightings), "final_ids": len(ids),
                "merges": len(self.merged_into), "expected": self.headcount}


def global_assign(candidate_scores, threshold):
    """G3:同一時間窗內一次解「一對一」指派,取代逐事件貪婪 max()。

    candidate_scores = {track_key: {chef_id: llr}}。
    貪婪的問題:兩條 track 可能都綁到同一個 chef(誤併)。一對一指派禁止這件事。

    回傳 {track_key: chef_id or None}。低於門檻者回 None(開新 chef)。
    """
    import numpy as np
    from scipy.optimize import linear_sum_assignment

    tracks = sorted(candidate_scores)
    chefs = sorted({c for v in candidate_scores.values() for c in v})
    if not tracks or not chefs:
        return {t: None for t in tracks}
    NEG = -1e6
    m = np.full((len(tracks), len(chefs)), NEG)
    for i, t in enumerate(tracks):
        for j, c in enumerate(chefs):
            if c in candidate_scores[t]:
                m[i, j] = candidate_scores[t][c]
    r, c = linear_sum_assignment(-m)
    out = {t: None for t in tracks}
    for i, j in zip(r, c):
        if m[i, j] > NEG and m[i, j] >= threshold:
            out[tracks[i]] = chefs[j]
    return out


def merge_rate_summary(refiner, gt_of_chef):
    """診斷用:合併決策裡有多少是對的。gt_of_chef = {chef_id: 真實身份}。"""
    ok = bad = 0
    for src, dst in refiner.merged_into.items():
        if src in gt_of_chef and dst in gt_of_chef:
            if gt_of_chef[src] == gt_of_chef[dst]:
                ok += 1
            else:
                bad += 1
    return {"correct_merges": ok, "wrong_merges": bad,
            "precision": round(ok / max(ok + bad, 1), 4)}
