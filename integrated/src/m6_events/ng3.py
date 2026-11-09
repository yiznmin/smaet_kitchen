"""團隊 NG3(單人):手碰到臉,每次碰臉算一次。

定義(使用者 2026-11-05):手碰到臉就算違規(含拿食物入口、用手臂 / 手背擦臉;摸頭髮、隔著衛生紙擦鼻子不算)。
規則(登記 §8.84–8.85 選定、§8.91 人工檢視後 R 0.764 / P 0.656):
  1. 臉:ViTPose 關鍵點,鼻 / 眼 / 耳(0–4)中分數 ≥ kp_thr 者取平均當臉中心;肩寬 = 左右肩(5、6)距離。
  2. 手:M3 手框(信心 ≥ thr_hand);手框到臉中心的最短距離 ≤ r × 肩寬 → 這一步碰臉。
  3. 鏡頭(fusion,§8.94):any = views 任一台成立即成立(看得到臉的 output0、Aoutput1);
     veto = 任一台成立且沒有任一台「看得到臉與手、距離 > veto_r × 肩寬」;agree = 看得到臉與手的鏡頭都要成立。
     veto2 / agree2(§8.96):同上,但只有「看清楚」(看到臉且手框 ≥ 2 個)的鏡頭能否決(碰臉的手被擋住時不誤否決)。
     clear_conf(§8.102):「看清楚」另要求信心第二高的手框 ≥ clear_conf(0 = 不限)。
  5. (§8.104)wrist_r > 0:手框到同幀任一人手腕關鍵點(9、10)≤ wrist_r × 肩寬才算手(排除牆面 / 地板誤偵);
     face_front:臉中心須含鼻子或眼睛(0–2),只看到耳朵(後腦勺)不算看到臉。
  6. (§8.106)short_r > 0:段落內最長連續成立 < short_s 秒(短暫觸發)時,各鏡頭成立步的最短距離須 ≤ short_r × 肩寬,
     否則視為手快速經過臉旁、不算。
  4. 長度 ≤ gap_fill_s 的空隙先補起來(§8.94);連續 ≥ min_s 秒才算一次碰臉;兩次間隔 < merge_s 秒合併。
輸入:各鏡頭 M3 偵測快取 + 同鏡頭姿態快取(npz:step、box、kp[N, 17, 3];見 m6_events/pose.py)。
"""
import numpy as np

from m6_events.violation import HAND, at, intervals, steps

DEFAULT = dict(views=["output0", "Aoutput1"], r=0.5, kp_thr=0.3, thr_hand=0.2, min_s=0.3, merge_s=2.0, pose_every=2,
               gap_fill_s=0.0, fusion="any", veto_r=1.0, clear_conf=0.0, wrist_r=0.0, face_front=False,
               short_s=0.3, short_r=0.0)


def faces(pz, i, kp_thr, front=False):
    """姿態快取第 i 步的 [(臉中心 xy, 肩寬)];該步沒有關鍵點回傳 []。front:須看到鼻子或眼睛。"""
    out = []
    for k in pz["kp"][pz["step"] == i]:
        head = k[:5][k[:5, 2] >= kp_thr]
        if front and not (k[:3, 2] >= kp_thr).any():
            continue
        if len(head) and k[5, 2] >= kp_thr and k[6, 2] >= kp_thr:
            sw = float(np.linalg.norm(k[5, :2] - k[6, :2]))
            if sw >= 1:
                out.append((head[:, :2].mean(0), sw))
    return out


def wrists(pz, i, kp_thr):
    """姿態快取第 i 步的 [(手腕 xy, 肩寬)](左右手腕 9、10,分數 ≥ kp_thr;肩寬要看得到)。"""
    out = []
    for k in pz["kp"][pz["step"] == i]:
        if k[5, 2] < kp_thr or k[6, 2] < kp_thr:
            continue
        sw = float(np.linalg.norm(k[5, :2] - k[6, :2]))
        if sw < 1:
            continue
        out += [(k[j, :2], sw) for j in (9, 10) if k[j, 2] >= kp_thr]
    return out


def hand_face_dist(H, fc):
    """各手框到臉中心的最短距離(像素;臉中心在框內為 0)。"""
    if not len(H):
        return np.zeros(0)
    return np.hypot(np.clip(fc[0], H[:, 0], H[:, 2]) - fc[0], np.clip(fc[1], H[:, 1], H[:, 3]) - fc[1])


def face_ratio_nh(dc, pz, T, p):
    """單一鏡頭逐步 (d, nh, c2):d = 手到臉中心最短距離 / 肩寬(沒看到臉或手 = nan);nh = 手框數(沒看到臉 = 0);
    c2 = 信心第二高的手框信心(不足 2 個 = 0)。姿態每 pose_every 步一幀,延續到下一幀。"""
    idx = steps(dc)
    out, nh, c2 = np.full(T, np.nan), np.zeros(T, int), np.zeros(T)
    for i in np.unique(pz["step"]):
        i = int(i)
        if i >= T:
            continue
        fs = faces(pz, i, p["kp_thr"], p.get("face_front", False))
        if not fs:
            continue
        xy, cf, cl = at(dc, idx, i)
        m = (cl == HAND) & (cf >= p["thr_hand"])
        if p.get("wrist_r", 0) > 0:                                  # 手框要有手腕關鍵點佐證
            ws = wrists(pz, i, p["kp_thr"])
            ok = np.zeros(len(m), bool)
            for w, sw in ws:
                ok |= hand_face_dist(xy, w) / sw <= p["wrist_r"]
            m &= ok
        H = xy[m]
        nh[i:min(T, i + p["pose_every"])] = len(H)
        if len(H) >= 2:
            c2[i:min(T, i + p["pose_every"])] = np.sort(cf[m])[-2]
        if not len(H):
            continue
        out[i:min(T, i + p["pose_every"])] = min((hand_face_dist(H, fc) / sw).min() for fc, sw in fs)
    return out, nh, c2


def face_ratio(dc, pz, T, p):
    """單一鏡頭逐步「手到臉中心最短距離 / 肩寬」(float[T];沒看到臉或手 = nan)。"""
    return face_ratio_nh(dc, pz, T, p)[0]


def face_touch_raw(dc, pz, T, p):
    """單一鏡頭逐步碰臉(bool[T]):手到臉中心 ≤ r × 肩寬。"""
    with np.errstate(invalid="ignore"):
        return face_ratio(dc, pz, T, p) <= p["r"]


def fuse(ratios, p, nhs=None, c2s=None):
    """多鏡頭融合(§8.94、§8.96、§8.102)。ratios:{鏡頭: float[T]};nhs:{鏡頭: int[T]} 手框數;
    c2s:{鏡頭: float[T]} 第二高手框信心(veto2 / agree2 需要)。"""
    R = np.array(list(ratios.values()))
    f = p.get("fusion", "any")
    with np.errstate(invalid="ignore"):
        near = (R <= p["r"]).any(0)
        if f in ("veto2", "agree2"):                            # 只有看清楚(手框 ≥ 2)的鏡頭能否決
            clear = np.array([nhs[v] for v in ratios]) >= 2
            if p.get("clear_conf", 0) > 0:
                clear &= np.array([c2s[v] for v in ratios]) >= p["clear_conf"]
            R = np.where(clear, R, np.nan)
            f = f[:-1]
        if f == "veto":
            return near & ~(R > p["veto_r"]).any(0)
        if f == "agree":
            return near & ~(R > p["r"]).any(0)
    return near


def fill_gaps(raw, k):
    """長度 ≤ k 步、前後都成立的空隙補成成立。"""
    raw = np.asarray(raw, bool).copy()
    if k <= 0:
        return raw
    on = np.flatnonzero(raw)
    for a, b in zip(on[:-1], on[1:]):
        if 1 < b - a <= k + 1:
            raw[a:b] = True
    return raw


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


def keep_short(segs, raw, ratios, step, p):
    """§8.106:短暫觸發段落須貼得更近(最短距離 ≤ short_r)才保留。"""
    if not p.get("short_r", 0):
        return segs
    k = int(round(p.get("short_s", 0.3) / step))
    R = np.array(list(ratios.values()))
    out = []
    for s, e in segs:
        if max(b - a for a, b in intervals(raw[s:e])) < k:
            with np.errstate(invalid="ignore"):
                near = np.where(R[:, s:e] <= p["r"], R[:, s:e], np.nan)
            if not np.isfinite(near).any() or np.nanmin(near) > p["short_r"]:
                continue
        out.append((s, e))
    return out


def detect_ng3(caches, poses, step=0.1, params=None, t0=0.0, trace=None):
    """caches:{鏡頭: DetCache};poses:{鏡頭: 姿態 npz(dict-like)}。只用兩者都有的 views 鏡頭。

    回傳 NG3 事件(秒,已加 t0)。trace(選用,dict):填入 T、各鏡頭 raw、合併 raw、段落,供報告影片顯示。
    """
    p = dict(DEFAULT, **(params or {}))
    vs = [v for v in p["views"] if v in caches and v in poses]
    if not vs:
        return []
    T = min(len(steps(caches[v])[0]) for v in vs)
    rn = {v: face_ratio_nh(caches[v], poses[v], T, p) for v in vs}
    ratios, nhs, c2s = {v: rn[v][0] for v in vs}, {v: rn[v][1] for v in vs}, {v: rn[v][2] for v in vs}
    with np.errstate(invalid="ignore"):
        raws = {v: ratios[v] <= p["r"] for v in vs}
    raw = fill_gaps(fuse(ratios, p, nhs, c2s), int(round(p.get("gap_fill_s", 0) / step)))
    segs = keep_short(touch_segments(raw, step, p["min_s"], p["merge_s"]), raw, ratios, step, p)
    if trace is not None:
        trace.update(T=T, raws=raws, raw=raw, segs=segs, views=vs, ratios=ratios)
    sec = lambda i: round(t0 + i * step, 2)
    return [dict(type="NG3", start=sec(s), end=sec(e), views="、".join(v for v in vs if raws[v][s:e].any()),
                 reason="手碰到臉") for s, e in segs]
