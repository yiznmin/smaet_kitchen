"""M6 違規判定(單人):觸發 → 洗手 → 碰食材 / 器具。

輸入全部是 M3 偵測(DetCache 快取,每 0.1 秒一步)+ zone + 規則設定(configs/violation_rules.yaml)。
不用大模型、不用 SAM。單人情境:不分 chef_id,所有手與人框都視為同一個人。

零件:
  touch_series      人—物關聯 V 規則 → 角色分工融合 → 遲滯平滑(同 M6 / HY3 的 V 分支)
  wash_series       洗手台 zone:手中心在 zone(hand)或人框蓋住 zone(body)
  trash_intervals   垃圾桶事件(Aoutput1 手在 zone 且 output0 人框蓋住 zone)
  HandHygieneFSM    髒手 → 先洗手 = 合格;先碰觸 = 違規;都沒有 = 未解除

預先登記:docs/M6_違規判定_單人_預先登記_20261021.md
"""
from dataclasses import dataclass, field

import cv2
import numpy as np

from m3.classes import NAME_TO_CLASS
from m6_events.association import _above, frame_features_r1, ioa
from m6_events.smoothing import hysteresis

PERSON, HAND = NAME_TO_CLASS["人"], NAME_TO_CLASS["手"]
X_UPPER = 0.3                       # 沒偵測到手時,物件與人上半身 IoA 的退路門檻(同 M6)


# ── 基本工具 ─────────────────────────────────────────────────────

def steps(cache):
    """快取的時間步:回傳 (幀號, 每步偵測的起訖索引)。"""
    stride = int(cache.meta["cache_stride"])
    fids = np.arange(0, int(cache.meta["max_fid"]) + 1, stride)
    return fids, np.searchsorted(cache._fid, fids, "left"), np.searchsorted(cache._fid, fids, "right")


def at(cache, idx, i):
    """第 i 步的 (框, 信心, 類別);超出片段長度時取最後一步。"""
    _, lo, hi = idx
    j = min(i, len(lo) - 1)
    return cache._xyxy[lo[j]:hi[j]].astype(float), cache._conf[lo[j]:hi[j]], cache._cls[lo[j]:hi[j]]


def owned_hands(xy, cf, cl, thr_hand, thr_person=0.3, hand_in_person=0.5):
    """手框以 IoA 指派給人框;回傳 (手框, 所屬人框)。"""
    persons = xy[(cl == PERSON) & _above(cf, thr_person)]
    hands = xy[(cl == HAND) & _above(cf, thr_hand)]
    if not len(hands) or not len(persons):
        return np.zeros((0, 4)), np.zeros((0, 4))
    m = ioa(hands, persons)
    best = m.argmax(1)
    ok = m[np.arange(len(hands)), best] >= hand_in_person
    return hands[ok], persons[best[ok]]


def in_poly(pts, poly):
    p = np.asarray(poly, np.float32).reshape(-1, 1, 2)
    return np.array([cv2.pointPolygonTest(p, (float(x), float(y)), False) >= 0 for x, y in pts], bool)


def dilate(x, k):
    """±k 步內有成立即成立。"""
    return np.convolve(np.asarray(x, int), np.ones(2 * k + 1, int), "same") > 0


def intervals(mask):
    """布林序列 → [(起, 訖(不含)), ...]。"""
    m = np.concatenate([[0], np.asarray(mask, int), [0]])
    d = np.diff(m)
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


# ── 零件 ─────────────────────────────────────────────────────────

def zone_hand_series(cache, T, poly, thr_hand):
    """手中心在 zone 內。"""
    idx = steps(cache)
    out = np.zeros(T, bool)
    for i in range(T):
        hands, _ = owned_hands(*at(cache, idx, i), thr_hand)
        if len(hands):
            ctr = np.c_[(hands[:, 0] + hands[:, 2]) / 2, (hands[:, 1] + hands[:, 3]) / 2]
            out[i] = in_poly(ctr, poly).any()
    return out


def zone_body_series(cache, T, poly, x, thr_person=0.3):
    """有人框蓋住 zone(zone 外接框落在人框內的比例 ≥ x)。"""
    idx = steps(cache)
    p = np.asarray(poly, float)
    zb = np.array([[p[:, 0].min(), p[:, 1].min(), p[:, 0].max(), p[:, 1].max()]])
    out = np.zeros(T, bool)
    for i in range(T):
        xy, cf, cl = at(cache, idx, i)
        pb = xy[(cl == PERSON) & _above(cf, thr_person)]
        out[i] = len(pb) > 0 and bool((ioa(zb, pb) >= x).any())
    return out


def side_series(overview, T, x_a, x_b, thr_person=0.3):
    """總覽鏡頭信心最高人框的腳點 x → A / B / M(沒人時沿用前一步)。"""
    idx = steps(overview)
    out, last = np.empty(T, dtype="<U1"), "M"
    for i in range(T):
        xy, cf, cl = at(overview, idx, i)
        k = (cl == PERSON) & _above(cf, thr_person)
        if k.any():
            b = xy[np.flatnonzero(k)[cf[k].argmax()]]
            cx = (b[0] + b[2]) / 2
            last = "A" if cx < x_a else ("B" if cx > x_b else "M")
        out[i] = last
    return out


def touch_series(caches, T, side, close, classes, a):
    """人—物關聯 V 規則 → 角色分工融合 → 平滑。回傳 [T, n_cls] 布林。

    caches:{鏡頭: DetCache}(近拍鏡頭);a:association 設定。
    """
    class_ids = [NAME_TO_CLASS[c] for c in classes]
    raws = {}
    for cam in close["A"] + close["B"]:
        c = caches[cam]
        idx = steps(c)
        rows = []
        for i in range(T):
            xy, cf, cl = at(c, idx, i)
            pxy, pcf, pcl = at(c, idx, i - a["lag"]) if i >= a["lag"] else (xy[:0], cf[:0], cl[:0])
            r = frame_features_r1(xy, cf, cl, pxy, pcf, pcl, class_ids, thr_person=a["thr_person"],
                                  thr_hand=a["thr_hand"], thr_obj=a["thr_obj"],
                                  hand_in_person=a["hand_in_person"], iou_moves=[a["iou_move"]])
            rows.append((r["mv_gap"][0] <= a["d"]) | (r["upper"] >= X_UPPER))
        raws[cam] = np.stack(rows)
    fused = np.zeros((T, len(classes)), bool)
    for s, cams in {"A": close["A"], "B": close["B"], "M": close["A"] + close["B"]}.items():
        sel = side == s
        if sel.any():
            fused[sel] = sum(raws[c][sel].astype(int) for c in cams) >= a["role_k"]
    n, k_on = a["win"]
    return np.stack([hysteresis(fused[:, j], n, k_on, a["k_off"]) for j in range(len(classes))], 1)


def trash_intervals(hand_cache, body_cache, T, hand_poly, body_poly, t, step):
    """垃圾桶事件:兩台 ±sync 秒內同時成立 → 平滑 → 連續 ≥ min_s 秒。回傳事件區間(步)。"""
    k = int(round(t["sync_s"] / step))
    raw = (dilate(zone_hand_series(hand_cache, T, hand_poly, t["thr_hand"]), k)
           & dilate(zone_body_series(body_cache, T, body_poly, t["body_ioa"]), k))
    sm = hysteresis(raw, *t["smooth"])
    need = max(1, int(round(t["min_s"] / step)))
    return [(s, e) for s, e in intervals(sm) if e - s >= need]


# ── 狀態機 ───────────────────────────────────────────────────────

@dataclass
class Verdict:
    result: str                     # 合格 / 違規 / 未解除
    dirty_from: int                 # 髒手開始(步)
    at: int = None                  # 判定時刻(步)
    touched: list = field(default_factory=list)      # 違規時碰到的類別
    wash_steps: int = 0


class HandHygieneFSM:
    """髒手期間:洗手累計 ≥ wash_s → 合格;洗手前碰到食材 / 器具 → 違規;都沒有 → 未解除。"""

    def __init__(self, wash_s, step, classes):
        self.need = max(1, int(round(wash_s / step)))
        self.classes = classes

    def run(self, dirty_from, wash, touch):
        washed = 0
        for i in range(max(dirty_from, 0), len(wash)):
            washed += int(wash[i])
            if washed >= self.need:
                return Verdict("合格", dirty_from, i, wash_steps=washed)
            if touch[i].any():
                return Verdict("違規", dirty_from, i, [self.classes[j] for j in np.flatnonzero(touch[i])], washed)
        return Verdict("未解除", dirty_from, None, wash_steps=washed)
