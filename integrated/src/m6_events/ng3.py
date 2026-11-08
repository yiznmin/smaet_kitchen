"""團隊 NG3(單人):手碰到臉,每次碰臉算一次。

定義(使用者 2026-11-05):手碰到臉就算違規(含拿食物入口、用手臂 / 手背擦臉;摸頭髮、隔著衛生紙擦鼻子不算)。
規則(登記 §8.84–8.85 選定、§8.91 人工檢視後 R 0.764 / P 0.656):
  1. 臉:ViTPose 關鍵點,鼻 / 眼 / 耳(0–4)中分數 ≥ kp_thr 者取平均當臉中心;肩寬 = 左右肩(5、6)距離。
  2. 手:M3 手框(信心 ≥ thr_hand);手框到臉中心的最短距離 ≤ r × 肩寬 → 這一步碰臉。
  3. 鏡頭:views 任一台成立即成立(看得到臉的 output0、Aoutput1)。
  4. 連續 ≥ min_s 秒才算一次碰臉;兩次間隔 < merge_s 秒合併。範圍 = 碰臉開始 → 結束。
輸入:各鏡頭 M3 偵測快取 + 同鏡頭姿態快取(npz:step、box、kp[N, 17, 3];見 m6_events/pose.py)。
"""
import numpy as np

from m6_events.violation import HAND, at, intervals, steps

DEFAULT = dict(views=["output0", "Aoutput1"], r=0.5, kp_thr=0.3, thr_hand=0.2, min_s=0.3, merge_s=2.0, pose_every=2)


def faces(pz, i, kp_thr):
    """姿態快取第 i 步的 [(臉中心 xy, 肩寬)];該步沒有關鍵點回傳 []。"""
    out = []
    for k in pz["kp"][pz["step"] == i]:
        head = k[:5][k[:5, 2] >= kp_thr]
        if len(head) and k[5, 2] >= kp_thr and k[6, 2] >= kp_thr:
            sw = float(np.linalg.norm(k[5, :2] - k[6, :2]))
            if sw >= 1:
                out.append((head[:, :2].mean(0), sw))
    return out


def hand_face_dist(H, fc):
    """各手框到臉中心的最短距離(像素;臉中心在框內為 0)。"""
    if not len(H):
        return np.zeros(0)
    return np.hypot(np.clip(fc[0], H[:, 0], H[:, 2]) - fc[0], np.clip(fc[1], H[:, 1], H[:, 3]) - fc[1])


def face_touch_raw(dc, pz, T, p):
    """單一鏡頭逐步碰臉(bool[T])。姿態每 pose_every 步一幀,成立時延續到下一個姿態幀。"""
    idx = steps(dc)
    out = np.zeros(T, bool)
    for i in np.unique(pz["step"]):
        i = int(i)
        if i >= T:
            continue
        fs = faces(pz, i, p["kp_thr"])
        if not fs:
            continue
        xy, cf, cl = at(dc, idx, i)
        H = xy[(cl == HAND) & (cf >= p["thr_hand"])]
        if any((hand_face_dist(H, fc) <= p["r"] * sw).any() for fc, sw in fs):
            out[i:min(T, i + p["pose_every"])] = True
    return out


def touch_segments(raw, step, min_s, merge_s):
    """連續 ≥ min_s 秒的段落,間隔 < merge_s 秒合併。回傳 [(開始步, 結束步)]。"""
    gap, mn = int(round(merge_s / step)), max(1, int(round(min_s / step)))
    merged = []
    for s, e in intervals(raw):
        if e - s < mn:
            continue
        if merged and s - merged[-1][1] < gap:
            merged[-1][1] = e
        else:
            merged.append([s, e])
    return [(s, e) for s, e in merged]


def detect_ng3(caches, poses, step=0.1, params=None, t0=0.0, trace=None):
    """caches:{鏡頭: DetCache};poses:{鏡頭: 姿態 npz(dict-like)}。只用兩者都有的 views 鏡頭。

    回傳 NG3 事件(秒,已加 t0)。trace(選用,dict):填入 T、各鏡頭 raw、合併 raw、段落,供報告影片顯示。
    """
    p = dict(DEFAULT, **(params or {}))
    vs = [v for v in p["views"] if v in caches and v in poses]
    if not vs:
        return []
    T = min(len(steps(caches[v])[0]) for v in vs)
    raws = {v: face_touch_raw(caches[v], poses[v], T, p) for v in vs}
    raw = np.any(list(raws.values()), 0)
    segs = touch_segments(raw, step, p["min_s"], p["merge_s"])
    if trace is not None:
        trace.update(T=T, raws=raws, raw=raw, segs=segs, views=vs)
    sec = lambda i: round(t0 + i * step, 2)
    return [dict(type="NG3", start=sec(s), end=sec(e), views="、".join(v for v in vs if raws[v][s:e].any()),
                 reason="手碰到臉") for s, e in segs]
