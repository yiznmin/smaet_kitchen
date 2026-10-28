"""團隊 NG 準則(單人):NG1 / NG2 垃圾桶污染,依團隊定義計數與給範圍。

只用規則(M3 偵測 + zone + 時序),不用 VLM。
評估依據:docs/M6_NG準則評估_預先登記_20261027.md §8.8–8.11(垃圾桶事件層:NG1 R 0.82 / P 0.74;NG2 R 0.90 / P 0.50)。

團隊定義:
  NG1:第一次接觸污染源開始 → 最後一次接觸結束後 5 秒;5 秒內再接觸合併成一次;5 秒內洗手 → 不算。
  NG2:手在污染狀態下(接觸後、洗手前),每接觸一次食材 / 餐具 / 廚具算一次;
       範圍 = 最近一次接觸污染源開始 → 這次接觸結束。

範圍外:手機、衛生紙、撿地上東西、擦地等污染源(M3 沒有類別);NG3(吃東西,規則不成立);NG4(樣本不足,未定案)。
"""
from pathlib import Path

import numpy as np

from m6_events.violation import (dilate, intervals, side_series, steps, touch_series, trash_intervals,
                                    zone_any_hand_series)
from common.zones import get_zone


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
    return dict(T=T, contacts=contacts, wash=wash, touch=touch)


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
    return sorted(out, key=lambda x: (x["start"], x["type"]))
