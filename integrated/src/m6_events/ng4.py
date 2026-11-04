"""團隊 NG4a 觸發(單人):清洗物件 → 放下 → 判放置位置。

團隊定義:洗過的食材 / 廚具直接放桌面 = NG4(範圍 清洗開始 → 放桌面);放砧板 / 容器 = 合格。
只用規則(M3 偵測 + 洗手台 zone + 時序),不用 VLM。
預先登記:docs/M6_NG準則評估_預先登記_20261027.md §8.28

流程(每類物件分開,單位皆為步,每步 0.1 秒):
  1. 清洗:B 近拍鏡頭中,物件框中心在洗手台 zone、且與「中心在洗手台 zone 的手框」重疊(手拿著它在水槽)。
     遲滯平滑後,間隔 < wash_gap_s 合併;累計 ≥ wash_min_s 才算一次清洗。
  2. 放下:清洗結束後 max_gap_s 內,任一近拍鏡頭出現「新的」同類物件框(清洗最後 base_s 秒洗手台外沒有的,IoU < new_iou),
     不在洗手台 zone,且符合下列其一並持續 ≥ put_min_s:
       - 在砧板 / 容器上(IoA ≥ sup_ioa;手還按著也算,放上砧板常馬上切)
       - 不在砧板 / 容器上、手已離開(避免拿在半空中被當成放下)
  3. 位置:放下起點後 judge_s 秒內,各鏡頭「在支撐物上」或「在桌面」成立比例 ≥ vote 者投票;
     只有支撐物 → 砧板或容器(合格);只有桌面 → 桌面(NG4 確定);兩者都有或都沒有 → 無法判斷(NG4 待確認)。
範圍外:NG4b(放桌面後未清洗即使用,§8.14 無鑑別力)。
"""
import numpy as np

from m3.classes import NAME_TO_CLASS
from m6_events.association import ioa
from m6_events.smoothing import hysteresis
from m6_events.violation import HAND, at, in_poly, intervals, steps

DEFAULT = dict(
    objects=["食材", "刀具", "砧板", "鍋鏟", "鍋子", "容器"],
    supports=["砧板", "容器"],
    views=["Aoutput1", "Aoutput2", "Aoutput3", "Boutput1", "Boutput2"],
    thr_obj=0.4, thr_hand=0.2, smooth=[3, 2, 3],
    wash_gap_s=1.0, wash_min_s=1.0,
    base_s=1.0, new_iou=0.3, sup_ioa=0.3,
    max_gap_s=30.0, put_min_s=1.0, judge_s=1.5, vote=0.5,
)


def _ctr(b):
    return np.c_[(b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2]


def _iou(a, b):
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    ar = lambda z: (z[:, 2] - z[:, 0]) * (z[:, 3] - z[:, 1])
    return inter / np.maximum(ar(a)[:, None] + ar(b)[None] - inter, 1e-6)


def _outside(b, poly):
    return b if poly is None or not len(b) else b[~in_poly(_ctr(b), poly)]


class _View:
    """單一鏡頭的逐步偵測存取。"""

    def __init__(self, cache, sink, p):
        self.c, self.idx, self.sink, self.p = cache, steps(cache), sink, p

    def get(self, i, cls_name):
        xy, cf, cl = at(self.c, self.idx, i)
        b = xy[(cl == NAME_TO_CLASS[cls_name]) & (cf >= self.p["thr_obj"])]
        h = xy[(cl == HAND) & (cf >= self.p["thr_hand"])]
        return b, h, xy, cf, cl


def wash_series(views, T, p):
    """[T, 物件類別] 原始清洗訊號(手拿著物件在水槽)。"""
    raw = np.zeros((T, len(p["objects"])), bool)
    for v in views.values():
        if v.sink is None:
            continue
        for i in range(T):
            xy, cf, cl = at(v.c, v.idx, i)
            h = xy[(cl == HAND) & (cf >= p["thr_hand"])]
            if not len(h):
                continue
            h = h[in_poly(_ctr(h), v.sink)]
            if not len(h):
                continue
            for j, name in enumerate(p["objects"]):
                b = xy[(cl == NAME_TO_CLASS[name]) & (cf >= p["thr_obj"])]
                if len(b):
                    b = b[in_poly(_ctr(b), v.sink)]
                    if len(b) and (ioa(b, h) > 0).any():
                        raw[i, j] = True
    return raw


def wash_segments(raw_j, p, step):
    k = lambda s: max(1, int(round(s / step)))
    # wash_smooth = False(§8.30):M3 在水槽裡的食材框時有時無,不平滑、直接累計原始成立步數
    sm = hysteresis(raw_j, *p["smooth"]) if p.get("wash_smooth", True) else np.asarray(raw_j, bool)
    segs = []
    for s, e in intervals(sm):
        if segs and s - segs[-1][1] < k(p["wash_gap_s"]):
            segs[-1][1] = e
        else:
            segs.append([s, e])
    return [(s, e) for s, e in segs if sm[s:e].sum() >= k(p["wash_min_s"])]


def _place_flags(v, i, name, base, p, prev=None):
    """回傳 (在支撐物上, 在桌面, 這步的新框)。

    prev:前 still_s 秒那一步的新框;stationary_ok(§8.36)時,和 prev 幾乎重合(IoU ≥ still_iou)的框
    即使手還碰著也算「放好了」(手扶著 / 放下後手停在旁邊)。
    """
    b, h, xy, cf, cl = v.get(i, name)
    b = _outside(b, v.sink)
    if len(b) and len(base):
        b = b[~(_iou(b, base) >= p["new_iou"]).any(1)]
    if not len(b):
        return False, False, b
    sup_ids = [NAME_TO_CLASS[n] for n in p["supports"]]
    s = xy[np.isin(cl, sup_ids) & (cf >= p["thr_obj"])]
    if name in p["supports"] and len(s):
        s = s[~(_iou(s, b) > 0.95).any(1)]                     # 支撐物類別本身:排除自己那個框
    on = (ioa(b, s) >= p["sup_ioa"]).any(1) if len(s) else np.zeros(len(b), bool)
    held = (ioa(b, h) > 0).any(1) if len(h) else np.zeros(len(b), bool)
    if p.get("stationary_ok") and prev is not None and len(prev):
        held &= ~(_iou(b, prev) >= p.get("still_iou", 0.8)).any(1)
    return bool(on.any()), bool((~on & ~held).any()), b


def _baseline(v, name, ws, we, p, k):
    """舊框:清洗最後 base_s 秒洗手台外的框;base_mode = static(§8.36)時,只留清洗開始前也在同位置的框(靜止物)。"""
    cat = lambda r: np.concatenate([_outside(v.get(i, name)[0], v.sink) for i in r]) if len(r) else np.zeros((0, 4))
    base = cat(range(max(ws, we - k(p["base_s"])), we))
    if p.get("base_mode", "last") == "static" and len(base):
        pre = cat(range(max(0, ws - k(p["base_s"])), ws))
        base = base[(_iou(base, pre) >= 0.5).any(1)] if len(pre) else base[:0]
    return base


def find_put(views, T, name, ws, we, p, step):
    """清洗 [ws, we) 之後的放下時間與位置;找不到回傳 None。"""
    k = lambda s: max(1, int(round(s / step)))
    lo, hi = we, min(T, we + k(p["max_gap_s"]) + k(p["put_min_s"]) + k(p["judge_s"])
                     + int(round(p.get("judge_offset_s", 0.0) / step)))
    if hi - lo < k(p["put_min_s"]):
        return None
    flags, lag = {}, k(p.get("still_s", 0.5))
    for vn, v in views.items():
        base = _baseline(v, name, ws, we, p, k)
        f, boxes = [], []
        for n, i in enumerate(range(lo, hi)):
            a, c, b = _place_flags(v, i, name, base, p, boxes[n - lag] if n >= lag else None)
            f.append((a, c))
            boxes.append(b)
        flags[vn] = np.array(f, bool).reshape(-1, 2)
    t = None
    for vn, f in flags.items():
        sm = hysteresis(f[:, 0] | f[:, 1], *p["smooth"])
        for s, e in intervals(sm):
            if s <= k(p["max_gap_s"]) and e - s >= k(p["put_min_s"]):
                t = s if t is None else min(t, s)
                break
    if t is None:
        return None
    sup, cnt = [], []
    off = int(round(p.get("judge_offset_s", 0.0) / step))      # §8.32:放下起點常早於真正放好,判位置往後移
    for vn, f in flags.items():
        w = f[t + off:t + off + k(p["judge_s"])]
        if len(w) and w[:, 0].mean() >= p["vote"]:
            sup.append(vn)
        if len(w) and w[:, 1].mean() >= p["vote"]:
            cnt.append(vn)
    if p.get("decide", "strict") == "majority":                # §8.32:兩邊都有時比鏡頭數,同數才算無法判斷
        place = "砧板或容器" if len(sup) > len(cnt) else "桌面" if len(cnt) > len(sup) else "無法判斷"
    else:
        place = "砧板或容器" if sup and not cnt else "桌面" if cnt and not sup else "無法判斷"
    return dict(put=lo + t, place=place, sup_views=sup, counter_views=cnt)


def detect_ng4(caches, sinks, step=0.1, params=None, t0=0.0):
    """caches:{鏡頭: DetCache};sinks:{鏡頭: 洗手台多邊形}(沒有的鏡頭不用給)。

    回傳每一次「清洗 → 放下」事件(含合格的),ng4 欄位表示是否違規;時間為秒(已加 t0)。
    """
    p = dict(DEFAULT, **(params or {}))
    views = {v: _View(caches[v], sinks.get(v), p) for v in p["views"] if v in caches}
    T = min(len(steps(c.c)[0]) for c in views.values())
    raw = wash_series(views, T, p)
    k = lambda s: max(1, int(round(s / step)))
    sec = lambda i: round(t0 + i * step, 2)
    out = []
    for j, name in enumerate(p["objects"]):
        for ws, we in wash_segments(raw[:, j], p, step):
            r = find_put(views, T, name, ws, we, p, step)
            if r is None:
                continue
            ng = r["place"] != "砧板或容器"
            out.append(dict(type="NG4", obj=name, start=sec(ws), wash_end=sec(we), put=sec(r["put"]),
                            end=sec(r["put"] + k(p["put_min_s"])), place=r["place"], ng4=ng,
                            status=("確定" if r["place"] == "桌面" else "待確認") if ng else "",
                            sup_views="、".join(r["sup_views"]), counter_views="、".join(r["counter_views"]),
                            reason="清洗後放桌面" if r["place"] == "桌面" else
                            "清洗後放置位置無法判斷" if ng else "清洗後放砧板 / 容器(合格)"))
    return sorted(out, key=lambda x: x["start"])
