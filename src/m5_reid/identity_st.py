"""M5 v2 空間+時序身份管理:時空為主、外觀(DINOv2 可商用)為輔破平手。

擴充 IdentityManager:on_new_track/on_track_lost 帶入 camera_id + bbox + 時間,
用 CameraTopology 的「轉場時間門」先把候選縮到少數,再用外觀 cosine 破平手。
→ 即使外觀弱(DINOv2 Rank-1 僅 11%),時空門也能綁對;且全條可商用。

track_id 各鏡頭不全域唯一 → 內部一律用 (camera_id, track_id) 當 key。
"""
import math

from m5_reid.embedder import l2norm
from m5_reid.identity import ChefIdentity, IdentityManager, MatchResult, cosine
from m5_reid.spatiotemporal import CameraTopology


class SpatioTemporalIdentityManager(IdentityManager):
    def __init__(self, topology, embedder=None, fps=30.0,
                 recently_disappeared_ttl=600, embedding_ema=0.5):
        super().__init__(similarity_threshold=0.0, recently_disappeared_ttl=recently_disappeared_ttl,
                         embedding_ema=embedding_ema, embedder=embedder)
        self.topo = topology
        self.fps = fps
        self.f = topology.fusion
        self._exit = {}         # chef_id -> (camera_id, t_sec, bbox) 離場資訊
        self._cam = {}          # chef_id -> (camera_id, t_sec) 目前所在(active)
        self._pending_exit = {} # (cam, track) -> (cam, t_sec, frame_id, bbox)
        self._world = {}        # chef_id -> (世界座標, 時間)
        self._vel = {}          # chef_id -> 最近觀測到的速度向量(m/s)
        # 診斷:每次綁定決策有幾個候選通過物理可能性檢查。索引 5 代表「5 個以上」。
        # 這是本架構最關鍵的觀測量 —— 若幾乎總是 1,外觀品質對結果沒有影響。
        self._cand_hist = [0] * 6
        self.last_candidates = 0

    @classmethod
    def from_config(cls, topo_cfg, embedder=None, fps=30.0):
        return cls(CameraTopology.from_config(topo_cfg), embedder=embedder, fps=fps)

    def _key(self, track_id, camera_id=None):
        """track_id 各鏡頭不全域唯一 → 一律用 (camera_id, track_id) 當鍵。"""
        return (camera_id, track_id)

    def _forget(self, chef_id):
        """chef 逾 TTL 被永久移除時,連帶清掉時空狀態。

        沒有這個的話 _exit 會隨累計人次單調成長(tick() 只清 self.gone),
        直接違反 spec「連續運作下活躍清單記憶體不無限成長」。
        """
        self._exit.pop(chef_id, None)
        self._cam.pop(chef_id, None)
        self._world.pop(chef_id, None)
        self._vel.pop(chef_id, None)

    def resident_stats(self):
        s = super().resident_stats()
        s.update({"_exit": len(self._exit), "_cam": len(self._cam),
                  "_pending_exit": len(self._pending_exit), "_world": len(self._world),
                  "_vel": len(self._vel)})
        return s

    def _t(self, frame_id, t_sec, camera_id=None):
        """事件時間(秒),已套用該相機的時鐘偏移校正。

        所有跨鏡頭的 Δt 都由這裡產出,所以校正只需做在這一個點。
        未校正的話,鏡頭間 2 秒漂移在 σ=1.5s 下等於 1.33σ,足以把幾乎所有
        真實轉場推出時間窗 —— 見 analyze_gate_capacity.py §4。
        """
        raw = float(t_sec) if t_sec is not None else frame_id / self.fps
        return self.topo.corrected(camera_id, raw) if camera_id is not None else raw

    def on_new_track(self, track_id, *, camera_id=None, frame_id=0, t_sec=None,
                     crop=None, embedding=None, bbox=None, zone=None, world_xy=None,
                     world_v=None):
        self.tick(frame_id)
        t = self._t(frame_id, t_sec, camera_id)
        emb = l2norm(embedding if embedding is not None else self.embedder.extract(crop))
        w_st, w_app = self.f["w_st"], self.f["w_app"]
        llr_mode = self.f["mode"] == "llr"
        thr = self.topo.llr_threshold if llr_mode else self.f["combined_threshold"]

        cands = []          # [(score, chef_id, app_cos)] 所有通過物理可能性檢查的候選

        # (a) recently_disappeared:跨時轉場(不重疊鏡頭的主路徑)
        for cid, chef in self.gone.items():
            ex = self._exit.get(cid)
            if ex is None:
                continue
            app = None
            if llr_mode:
                ok, llr_t = self.topo.transit_llr(ex[0], ex[1], camera_id, t)
                if not ok:
                    continue
                app = cosine(emb, chef.embedding)
                score = llr_t + self.topo.app_lr.llr(app)
                # 位置證據只在**同一台鏡頭**適用:跨鏡頭的影像座標是不同座標系,
                # 沒有外參校正就不可比。這條專門用來收緊 M4 斷軌的重關聯。
                if ex[0] == camera_id and self.topo.pos_lr is not None:
                    score += self.topo.pos_lr.llr(ex[2] if len(ex) > 2 else None,
                                                  bbox, t - ex[1])
                # 方向證據:與 Δt 正交,所以能幫到逗留者(時間軸上救不了的那些)
                score += self.topo.direction_llr(ex[0], camera_id,
                                                 ex[3] if len(ex) > 3 else None, zone)
            else:
                ok, sp = self.topo.transition_gate(ex[0], ex[1], camera_id, t)
                if not ok:
                    continue
                app = cosine(emb, chef.embedding)
                score = w_st * sp + w_app * app
            cands.append((score, cid, app))

        # (b) 該 chef 此刻正被某台「與本鏡頭重疊」的相機看著 → 幾何關聯
        #
        # ⚠ 判斷依據是 chef.track_ids(目前綁定中的所有 (鏡頭, track)),**不是**
        #   _cam 的時間戳。踩過的坑:原本用 `_cam` 並要求 |Δt| ≤ 0.5s,但 _cam 只在
        #   「綁定事件」發生時更新,而「這個人現在正在畫面裡」是一個**持續狀態**,
        #   不是事件。M4 每幀都會輸出當前 active 的 tracks,但 M5 只吃事件,
        #   所以它無從得知誰此刻在畫面上 —— 重疊路徑因此幾乎從不觸發。
        #   全景鏡頭方案第一次跑時碎裂率完全沒降,根因就是這個。
        overlap_pool = {}          # 重疊鏡頭 -> 此刻在該鏡頭裡的 chef 數
        for chef in self.active.values():
            for c, _t in chef.track_ids:
                if c != camera_id and self.topo.is_overlapping(c, camera_id):
                    overlap_pool[c] = overlap_pool.get(c, 0) + 1

        for cid, chef in self.active.items():
            cams = {c for c, _t in chef.track_ids
                    if c != camera_id and self.topo.is_overlapping(c, camera_id)}
            if not cams:
                continue
            # ⚠ 「重疊鏡頭裡有人」不等於「就是這一個人」。若該鏡頭此刻有 K 位廚師,
            #   這條證據只說得出「是那 K 個之中的一個」→ 證據量要除以 K。
            #   沒有這一項的話,多人同時在場時重疊路徑會無差別地給滿分,誤併爆增。
            #   真正要收緊它需要**跨鏡頭的地面平面校正(homography)**,才能比對
            #   「兩台鏡頭看到的是不是地面上同一個點」。本版尚未實作,故只做稀釋。
            app = cosine(emb, chef.embedding)
            if not llr_mode:
                cands.append((w_st * 1.0 + w_app * app, cid, app))
                continue
            if self.topo.ground_lr is not None and world_xy is not None:
                # 有地面校正:直接問「是不是站在同一個位置」,不必稀釋。
                # ⚠ 但要把「上次觀測到現在」的可能移動算進不確定度 —— 否則會拿
                #   舊位置跟新位置比,把對的綁定當成位置對不上而擋掉。
                g = self._world.get(cid)
                if g is None:
                    continue
                score = (self.topo.ground_lr.llr(g[0], world_xy, dt=t - g[1])
                         + self.topo.app_lr.llr(app))
                # 速度證據:位置相同時唯一還能分辨的線索
                if self.topo.vel_lr is not None:
                    score += self.topo.vel_lr.llr(self._vel.get(cid), world_v)
            else:
                # 無校正:只知道「那台鏡頭裡有人」。若該鏡頭此刻有 K 位廚師,
                # 這條證據只說得出「是 K 個之中的一個」→ 證據量除以 K。
                k = max(min(overlap_pool.get(c, 1) for c in cams), 1)
                score = self.f["overlap_llr"] - math.log(k) + self.topo.app_lr.llr(app)
            cands.append((score, cid, app))

        # 候選數是本架構最關鍵的診斷量:若幾乎總是 1,外觀品質根本不影響結果。
        self.last_candidates = len(cands)
        self._cand_hist[min(len(cands), 5)] += 1

        best_score, best_id, _ = max(cands, default=(float("-inf"), None, None))

        gk = self._key(track_id, camera_id)
        if best_id is not None and best_score >= thr:          # 綁到既有廚師
            chef = self.gone.pop(best_id, None) or self.active[best_id]
            chef.state = "active"
            self.active[best_id] = chef
            if gk not in chef.track_ids:
                chef.track_ids.append(gk)
            chef.last_seen = frame_id
            chef.embedding = l2norm(self.ema * chef.embedding + (1 - self.ema) * emb)
            self.track_to_chef[gk] = best_id
            self._cam[best_id] = (camera_id, t)
            if world_xy is not None:
                self._world[best_id] = (world_xy, t)
            if world_v is not None:
                self._vel[best_id] = world_v
            return MatchResult(track_id, best_id, True, round(best_score, 4), frame_id)

        cid = self._next                                        # 開新廚師
        self._next += 1
        self.active[cid] = ChefIdentity(cid, emb, [gk], "active", frame_id, frame_id)
        self.track_to_chef[gk] = cid
        self._cam[cid] = (camera_id, t)
        if world_xy is not None:
            self._world[cid] = (world_xy, t)
        if world_v is not None:
            self._vel[cid] = world_v
        # 無候選時 best_score 是 -inf;回 0.0 表示「沒有任何證據」而非「證據為負」
        shown = 0.0 if best_score == float("-inf") else round(best_score, 4)
        return MatchResult(track_id, cid, False, shown, frame_id)

    def on_track_update(self, track_id, *, camera_id=None, frame_id=0, t_sec=None,
                        world_xy=None, bbox=None, world_v=None):
        """M4 每幀(或每隔幾幀)回報「這條 track 還在,現在在這裡」。

        ⚠ 這個介面是**地面校正的前提**,不是可有可無的優化。
          M5 原本只吃事件(new_track / lost / removed),所以它只知道每位 chef
          **上次綁定時**在哪裡。全景鏡頭的 track 是連續的、中間沒有綁定事件,
          於是比對「30 秒前的位置」與「現在的位置」→ 距離很大 → 強負證據 →
          把對的綁定擋掉(實測碎裂從 0.2% 暴增到 22.7%)。

          這與第五輪踩到的 `_cam` 時間戳問題是同一個結構性缺口:
          **M4 每幀輸出 active tracks,但 M5 只吃事件。**
          第五輪用 chef.track_ids 繞過了「誰在畫面裡」,但「在哪個位置」繞不過。
        """
        cid = self.track_to_chef.get(self._key(track_id, camera_id))
        if cid is None:
            return None
        t = self._t(frame_id, t_sec, camera_id)
        self._cam[cid] = (camera_id, t)
        if world_xy is not None:
            self._world[cid] = (world_xy, t)
        if world_v is not None:
            self._vel[cid] = world_v
        return cid

    def candidate_histogram(self):
        """回傳 {候選數: 次數}。'5' 代表 5 個以上。

        典型廚房情境應該絕大多數落在 0(開新人)與 1(唯一候選)。
        若 2 以上佔比高,表示外觀真的在仲裁,embedder 品質才會影響結果。
        """
        return {i: n for i, n in enumerate(self._cand_hist) if n}

    # ── 離場語意三段式 ────────────────────────────────────────────────
    # M4 的 lost_track 在「短暫遮擋」就會觸發,而 track 被救回時不會再發 new_track
    # (new_track 只認 active_ids - _seen_ids,而 _seen_ids 是累積的)。
    # 若在 lost 就標 gone,該 chef 會永遠卡在 gone、綁定永久遺失。
    # 因此:lost → 只記預備出口;reacquired → 撤銷;removed → 才真的離場。

    def on_track_lost(self, track_id, *, camera_id=None, frame_id=0, t_sec=None,
                      bbox=None, zone=None):
        """進 ByteTrack lost。只記「預備出口」(含離場位置),不動 chef 狀態。

        bbox 是 M4 該 track 最後已知的位置,同鏡頭重關聯要靠它分辨斷軌前後是不是
        同一個人 —— 只有時間的話,同鏡頭有多人時會綁錯(實測誤併率會翻倍)。
        """
        gk = self._key(track_id, camera_id)
        cid = self.track_to_chef.get(gk)
        if cid is None:
            return None
        self._pending_exit[gk] = (camera_id, self._t(frame_id, t_sec, camera_id),
                                  frame_id, bbox, zone)
        return cid

    def on_track_reacquired(self, track_id, *, camera_id=None, frame_id=0, t_sec=None,
                            bbox=None, zone=None):
        """從 lost 找回(人沒走,只是被擋住)→ 撤銷預備出口,chef 維持 active。"""
        gk = self._key(track_id, camera_id)
        self._pending_exit.pop(gk, None)
        cid = self.track_to_chef.get(gk)
        if cid is None:
            return None
        chef = self.active.get(cid)
        if chef is not None:
            chef.last_seen = frame_id
            self._cam[cid] = (camera_id, self._t(frame_id, t_sec, camera_id))
        return cid

    def on_track_removed(self, track_id, *, camera_id=None, frame_id=0, t_sec=None,
                         bbox=None, zone=None):
        """lost buffer 到期 → 真正離場。

        ⚠ 出口時間戳取自 lost_track 當時,不是 removed 當時。用 removed 的話,
        離場時間會系統性延後約 lost_track_buffer/fps,使所有轉場 Δt 偏小,
        直接讓 transition_gate 的時間窗判斷失準。
        """
        gk = self._key(track_id, camera_id)
        pending = self._pending_exit.pop(gk, None)
        if pending is not None:
            exit_cam, exit_t, exit_frame, exit_box, exit_zone = pending
        else:                              # 沒收到 lost 就直接 removed(理論上不該發生)
            exit_cam, exit_t, exit_frame, exit_box, exit_zone = (
                camera_id, self._t(frame_id, t_sec, camera_id), frame_id, bbox, zone)

        cid = self.track_to_chef.pop(gk, None)
        if cid is None:
            return None
        chef = self.active.get(cid)
        if chef is None:
            return None
        if gk in chef.track_ids:
            chef.track_ids.remove(gk)
        chef.last_seen = exit_frame
        if not chef.track_ids:                                  # 無任何鏡頭還看得到 → 離場
            chef.state = "gone"
            self.active.pop(cid, None)
            self.gone[cid] = chef
            self._exit[cid] = (exit_cam, exit_t, exit_box, exit_zone)
            self._cam.pop(cid, None)
        return cid
