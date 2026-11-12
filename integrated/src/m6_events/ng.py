"""團隊 NG 準則(單人):NG1 / NG2 垃圾桶污染,依團隊定義計數與給範圍。

只用規則(M3 偵測 + zone + 時序),不用 VLM。
評估依據:docs/M6_NG準則評估_預先登記_20261027.md(逐次接觸評估,使用者回查修正真值後:
  NG1 R 1.000 / P 1.000(§8.88);NG2 R 0.950 / P 0.864(§8.90);NG1 與 NG2 是獨立事件)。

團隊定義:
  碰垃圾桶:Aoutput1 看到手在垃圾桶區、output0 看到人站在垃圾桶前(前後 0.5 秒內),持續 ≥ 1 秒(trash_event)。
  NG1:第一次接觸污染源開始 → 最後一次接觸結束後 5 秒;5 秒內再接觸合併成一次;
       接觸結束 + grace_s(1 秒)起到結束後 5 秒內,洗手累計 ≥ wash_s(1 秒)→ 不算,否則 NG1。
  NG2:手在污染狀態下(接觸結束 + grace_s 起、洗手累計滿 wash_s 前),每碰一次食材 / 器具 / 容器
       (≥ touch_min_s 0.3 秒,間隔 < touch_gap_s 1 秒合併)算一次;接觸期間與 grace_s 內的碰觸不算。
       範圍 = 最近一次接觸污染源開始 → 這次碰觸結束。
  NG1 與 NG2 是獨立事件,不能用 NG2 判 NG1。

  NG3:手碰到臉,每碰一次算一次;見 m6_events/ng3.py(設定 ng3 節,另需人體姿態,由 run_violations.py 呼叫)。
  NG4(NG4a):洗過的食材 / 廚具直接放桌面,範圍 = 清洗開始 → 放下;見 m6_events/ng4.py(設定 ng4 節)。

範圍外:手機、衛生紙、撿地上東西、擦地等污染源(M3 沒有類別);
        NG4b(放桌面後未清洗即使用,§8.14 無鑑別力)。
"""
from pathlib import Path

import numpy as np

from m6_events.violation import (dilate, intervals, side_series, steps, touch_series, trash_intervals,
                                    zone_any_hand_series)
from common.zones import get_zone
from m6_events.ng4 import detect_ng4


def _episodes(contacts, merge):
    """接觸區間(步)間隔 < merge 步者合併 → [(第一次開始, 最後一次結束, [各次接觸])]。"""
    out = []
    for s, e in contacts:
        if out and s - out[-1][1] < merge:
            out[-1] = (out[-1][0], e, out[-1][2] + [(s, e)])
        else:
            out.append((s, e, [(s, e)]))
    return out


def _touches(touch_any, start, stop, gap, mn):
    """碰觸區間:間隔 < gap 步合併、長度 ≥ mn 步。"""
    merged = []
    for s, e in intervals(touch_any[start:stop]):
        if merged and s - merged[-1][1] < gap:
            merged[-1][1] = e
        else:
            merged.append([s, e])
    return [(s + start, e + start) for s, e in merged if e - s >= mn]


def backtrack_contacts(contacts, food, near, gap, max_len):
    """回推污染起點(§8.22):要丟的東西本身是污染源。

    對每個垃圾桶接觸 (s, e):若 [s − near, s] 內手有碰食材,從最後一次碰觸往前延伸,
    未碰觸連續 ≥ gap 步就停,最多回推 max_len 步。回推後重疊的接觸合併。單位皆為步。
    """
    out = []
    for s, e in contacts:
        lo = max(0, s - near)
        hit = np.flatnonzero(food[lo:s + 1])
        start = s
        if len(hit):
            j = lo + hit[-1]
            start, miss = j, 0
            for i in range(j, max(0, s - max_len) - 1, -1):
                if food[i]:
                    start, miss = i, 0
                else:
                    miss += 1
                    if miss >= gap:
                        break
            start = min(start, s)
        if out and start <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], e))
        else:
            out.append((start, e))
    return out


def series(caches, cfg, root):
    """由各鏡頭 DetCache 算出規則需要的時間序列。caches:{鏡頭: DetCache};root:zone 路徑的基準(integrated/)。"""
    step, cams, hh, te = cfg["step_s"], cfg["cameras"], cfg["hand_hygiene"], cfg["trash_event"]
    zone = lambda v, n: get_zone(Path(root) / cfg["zones"][v], n)
    need = set(cams["close"]["A"] + cams["close"]["B"] + [cams["overview"], te["hand_cam"], te["body_cam"]]
               + hh["close_cams"])
    miss = need - set(caches)
    if miss:
        raise ValueError(f"缺少鏡頭偵測:{sorted(miss)}")
    T = min(len(steps(c)[0]) for c in caches.values())
    contacts = trash_intervals(caches[te["hand_cam"]], caches[te["body_cam"]], T,
                               zone(te["hand_cam"], te["zone"]), zone(te["body_cam"], te["zone"]), te, step)
    wash = np.any([zone_any_hand_series(caches[v], T, zone(v, hh["sink_zone"]), hh["thr_hand"])
                   for v in hh["close_cams"]], 0)
    side = side_series(caches[cams["overview"]], T, **cams["side_x"])
    touch = touch_series(caches, T, side, cams["close"], cfg["touch_classes"], cfg["association"])
    bt = te.get("backtrack", {})
    if bt.get("enable"):
        k = lambda sec: int(round(sec / step))
        food = touch[:, cfg["touch_classes"].index("食材")]
        contacts = backtrack_contacts(contacts, food, k(bt["near_s"]), max(1, k(bt["gap_s"])), k(bt["max_s"]))
    ng4 = []
    if cfg.get("ng4", {}).get("enable"):
        n4 = cfg["ng4"]
        sinks = {v: zone(v, n4.get("sink_zone", hh["sink_zone"])) for v in n4["sink_cams"]}   # §8.65:NG4 用使用者手框範圍
        params = {k: v for k, v in n4.items() if k not in ("enable", "sink_cams", "sink_zone")}
        ng4 = [e for e in detect_ng4(caches, sinks, step, params) if e["ng4"]]
    return dict(T=T, contacts=contacts, wash=wash, touch=touch, ng4=ng4)


def detect_ng(s, cfg, t0=0.0):
    """回傳 NG 事件列表(秒,已加上 t0)。s 為 series() 的結果。"""
    step, hh, ng = cfg["step_s"], cfg["hand_hygiene"], cfg["ng"]
    k = lambda sec: int(round(sec / step))
    need, grace = max(1, k(hh["wash_s"])), k(hh["grace_s"])
    T, wash, touch = s["T"], s["wash"], s["touch"]
    near_wash = dilate(wash, k(ng["ignore_wash_s"])) if ng["ignore_wash_s"] > 0 else np.zeros(T, bool)
    sec = lambda i: round(t0 + i * step, 2)
    eps = _episodes(s["contacts"], k(ng["merge_s"]))
    out = []
    for n, (a, b, parts) in enumerate(eps):
        nxt = eps[n + 1][0] if n + 1 < len(eps) else T
        # NG1:最後一次接觸結束(+ grace)到結束後 ng1_wash_within_s 秒內,洗手累計 < wash_s
        w_end = min(T, b + k(ng["ng1_wash_within_s"]))
        washed5 = wash[min(b + grace, w_end):w_end].sum() >= need
        if not washed5:
            out.append(dict(type="NG1", start=sec(a), end=sec(w_end), n_contacts=len(parts),
                            reason="接觸污染源結束後 5 秒內未洗手"))
        # NG2:髒手期間(到洗手累計滿 wash_s 或下一次污染)每次碰觸算一次
        stop, acc = nxt, 0
        for i in range(min(b + grace, T), nxt):
            acc += int(wash[i])
            if acc >= need:
                stop = i
                break
        for ts, te_ in _touches(touch.any(1), min(b + grace, T), stop, k(ng["touch_gap_s"]), k(ng["touch_min_s"])):
            if near_wash[ts:te_].all():
                continue                                          # 洗手中碰觸(ignore_wash_s > 0 時)
            cls = [cfg["touch_classes"][j] for j in np.flatnonzero(touch[ts:te_].any(0))]
            out.append(dict(type="NG2", start=sec(parts[-1][0]), end=sec(te_), touched=cls,
                            reason="手汙染狀態下接觸食材、餐具或廚具"))
    for e in s.get("ng4", []):                                   # NG4a(series 內已算好,時間從 0 起)
        out.append(dict(type="NG4", start=round(t0 + e["start"], 2), end=round(t0 + e["end"], 2), obj=e["obj"],
                        place=e["place"], status=e["status"], reason=e["reason"]))
    return sorted(out, key=lambda x: (x["start"], x["type"]))
