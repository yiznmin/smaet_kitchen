"""NG3 新定義「手碰到臉」評估(登記 §8.47)。

正例 = 團隊 NG3 片段 + 無NG 吃東西事件;反例 = 其他類 NG 片段 + 其他類無NG 事件(排除 YH2003 05_17)。
規則 face_touch:指定鏡頭任一台連續 ≥ T 秒「手框中心在人框頭部區」。參與者留一選參數。

用法:
    python scripts/eval_ng3_face.py            # 全部片段
    python scripts/eval_ng3_face.py --smoke 3  # 只跑前 3 個片段(冒煙)
"""
import argparse
import csv
import os
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "integrated/src"))
sys.path.insert(0, str(ROOT / "scripts"))

import eval_ng_clips as E                                                 # noqa: E402
from m4_track.det_cache import DetCache                                   # noqa: E402
from m6_events.violation import HAND, PERSON, at, intervals, steps       # noqa: E402

OUT = ROOT / "results/m6/ng/ng3_face"
STEP = 0.1
GAP_S = 2.0                                                # 碰臉間隔 < 2 秒合併成一次
VIEWS = {"face": ("output0", "Aoutput1"), "all": None}     # face = 看得到臉的鏡頭
# (views, thr_hand, head_frac, T 秒),第一組 = 預設
GRID = [(v, h, f, t) for v in VIEWS for h in (0.3, 0.2) for f in (0.25, 0.20, 0.35) for t in (0.3, 0.5, 1.0)]


def tag(g):
    return "{}_h{}_f{}_T{}".format(*g)


def head_series(c, n, thr_hand, head_frac):
    """每一步是否有手框中心落在人框頭部區。"""
    idx = steps(c)
    out = np.zeros(n, bool)
    for i in range(n):
        xy, cf, cl = at(c, idx, i)
        P, H = xy[(cl == PERSON) & (cf >= 0.3)], xy[(cl == HAND) & (cf >= thr_hand)]
        if not (len(P) and len(H)):
            continue
        cx, cy = (H[:, 0] + H[:, 2]) / 2, (H[:, 1] + H[:, 3]) / 2
        top = P[:, 1] + head_frac * (P[:, 3] - P[:, 1])
        out[i] = bool(((cx[:, None] >= P[None, :, 0]) & (cx[:, None] <= P[None, :, 2])
                       & (cy[:, None] >= P[None, :, 1]) & (cy[:, None] <= top[None])).any())
    return out


def touches(raw, T_s):
    """碰臉次數:連續 ≥ T 秒的段落,間隔 < GAP_S 合併。"""
    gap, mn = int(round(GAP_S / STEP)), max(1, int(round(T_s / STEP)))
    segs = [[s, e] for s, e in intervals(raw) if e - s >= mn]
    merged = []
    for s, e in segs:
        if merged and s - merged[-1][1] < gap:
            merged[-1][1] = e
        else:
            merged.append([s, e])
    return merged


def clip_row(c):
    cs = {v: DetCache(p) for v, p in c["caches"].items()}
    n = min(len(steps(x)[0]) for x in cs.values())
    cache = {}
    for v, x in cs.items():
        for h, f in {(g[1], g[2]) for g in GRID}:
            cache[(v, h, f)] = head_series(x, n, h, f)
    gt = int(c["ng"] == "NG3" or (c["ng"] == "無NG" and c["cat"] == "吃東西"))
    row = dict(key=c["key"], pid=c["pid"], ng=c["ng"], cat=c["cat"], note=c["note"], dur_s=c["dur"],
               views=len(cs), y=gt)
    for g in GRID:
        vs = [v for v in cs if VIEWS[g[0]] is None or v in VIEWS[g[0]]]
        raw = np.any([cache[(v, g[1], g[2])] for v in vs], 0) if vs else np.zeros(n, bool)
        k = touches(raw, g[3])
        row[tag(g)] = int(bool(k)) if vs else None
        row["n_" + tag(g)] = len(k)
        if g == GRID[0]:
            row["touch_s"] = ";".join(f"{a * STEP:.1f}-{b * STEP:.1f}" for a, b in k)
    return row


# ── §8.49 第 2 輪:ViTPose 臉部定位 ─────────────────────────────────
OUT2 = ROOT / "results/m6/ng/ng3_face_r2"
POSE_EV = ROOT / "results/m7/events/pose"                 # 事件片段既有快取(M7 §9.2)
POSE_NG = ROOT / "results/m6/ng/ng3_face/pose"            # NG 片段新算
POSE_VIEWS = ("output0", "Aoutput1")
POSE_MODEL = "usyd-community/vitpose-base-simple"
POSE_EVERY = 2                                            # 每 2 步(0.2 秒)一幀
KP_THR, HAND_THR2, BOTH_TOL = 0.3, 0.2, 5                 # 關鍵點分數、手框信心、both 容許 ±0.5 秒
# (r, src, cams, T 秒),第一組 = 預設
GRID2 = [(r, s, cm, t) for r in (0.5, 0.3, 0.8) for s in ("either", "hand", "wrist")
         for cm in ("any", "both") for t in (0.3, 0.5)]


def tag2(g):
    return "r{}_{}_{}_T{}".format(*g)


def pose_path(c, v):
    if c["key"].startswith("ev_"):
        return POSE_EV / f"{Path(c['caches'][v]).stem}.npz"
    return POSE_NG / f"{c['key']}_{v}.npz"


def short_path(p):
    """OpenCV 在 Windows 開不了含中文的路徑:在同一磁碟建 ASCII 檔名的硬連結(不複製內容)。"""
    p = Path(p)
    if not p.exists() and (p.parent / "NG" / p.name).exists():   # 團隊片段已移到 NG/ 子資料夾
        p = p.parent / "NG" / p.name
    if str(p).isascii():
        return str(p)
    tmp = POSE_NG / "_tmp_link.mp4"
    tmp.unlink(missing_ok=True)
    os.link(p, tmp)
    return str(tmp)


def run_pose(clips):
    """NG 片段缺的關鍵點快取用 ViTPose 補算(做法同 scripts/m7_event_pose.py)。"""
    todo = [(c, v) for c in clips for v in POSE_VIEWS if v in c["caches"] and not pose_path(c, v).exists()]
    print(f"ViTPose 待跑 {len(todo)} 支", flush=True)
    if not todo:
        return
    import cv2
    import torch
    from transformers import AutoProcessor, VitPoseForPoseEstimation
    proc = AutoProcessor.from_pretrained(POSE_MODEL)
    model = VitPoseForPoseEstimation.from_pretrained(POSE_MODEL).to("cuda").eval()
    POSE_NG.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for n, (c, v) in enumerate(todo, 1):
        dc = DetCache(c["caches"][v])
        fids, lo, hi = steps(dc)
        want = {int(fids[i]): i for i in range(0, len(fids), POSE_EVERY)}
        cap = cv2.VideoCapture(short_path(c["files"][v]))
        if not cap.isOpened():
            raise RuntimeError(f"開不了影片:{c['files'][v]}")
        rec_s, rec_b, rec_k, fid = [], [], [], 0
        while want:
            ok, f = cap.read()
            if not ok:
                break
            i = want.pop(fid, None)
            fid += 1
            if i is None:
                continue
            cl, cf, xy = dc._cls[lo[i]:hi[i]], dc._conf[lo[i]:hi[i]], dc._xyxy[lo[i]:hi[i]].astype(float)
            pb = xy[(cl == PERSON) & (cf >= 0.3)]
            if not len(pb):
                continue
            xywh = np.c_[pb[:, :2], pb[:, 2:] - pb[:, :2]]
            inp = proc(f[:, :, ::-1].copy(), boxes=[xywh], return_tensors="pt").to("cuda")
            with torch.inference_mode():
                out = model(**inp)
            res = proc.post_process_pose_estimation(out, boxes=[xywh])[0]
            for b, r in zip(pb, res):
                rec_s.append(i)
                rec_b.append(b)
                rec_k.append(np.c_[r["keypoints"].cpu().numpy(), r["scores"].cpu().numpy()])
        cap.release()
        np.savez_compressed(pose_path(c, v), step=np.array(rec_s, np.int32),
                            box=np.array(rec_b, np.float32).reshape(-1, 4),
                            kp=np.array(rec_k, np.float32).reshape(-1, 17, 3))
        if n % 20 == 0 or n == len(todo):
            print(f"  ViTPose {n}/{len(todo)}  {time.time() - t0:.0f} 秒", flush=True)


def face_series(dc, pz, n):
    """回傳 {(r, src): bool[n]}:該步是否有手框 / 手腕在臉中心 r × 肩寬內。關鍵點每 0.2 秒一幀,下一步沿用。"""
    idx = steps(dc)
    out = {(r, s): np.zeros(n, bool) for r in {g[0] for g in GRID2} for s in ("hand", "wrist")}
    st, kp = pz["step"], pz["kp"]
    for i in np.unique(st):
        if i >= n:
            continue
        K = kp[st == i]
        xy, cf, cl = at(dc, idx, int(i))
        H = xy[(cl == HAND) & (cf >= HAND_THR2)]
        for k in K:
            head = k[:5][k[:5, 2] >= KP_THR]
            if not len(head) or k[5, 2] < KP_THR or k[6, 2] < KP_THR:
                continue
            fc, s = head[:, :2].mean(0), np.linalg.norm(k[5, :2] - k[6, :2])
            if s < 1:
                continue
            dh = (np.hypot(np.clip(fc[0], H[:, 0], H[:, 2]) - fc[0], np.clip(fc[1], H[:, 1], H[:, 3]) - fc[1]).min()
                  if len(H) else np.inf)
            wr = k[[9, 10]][k[[9, 10], 2] >= KP_THR]
            dw = np.hypot(*(wr[:, :2] - fc).T).min() if len(wr) else np.inf
            for r in {g[0] for g in GRID2}:
                for src, d in (("hand", dh), ("wrist", dw)):
                    if d <= r * s:
                        out[(r, src)][i:min(n, i + POSE_EVERY)] = True
    return out


def dilate(x, k):
    if not x.any():
        return x
    c = np.concatenate([[0], np.cumsum(x.astype(int))])
    i = np.arange(len(x))
    return (c[np.minimum(len(x), i + k + 1)] - c[np.maximum(0, i - k)]) > 0   # 前後 k 步內有任一成立


def clip_row2(c):
    gt = int(c["ng"] == "NG3" or (c["ng"] == "無NG" and c["cat"] == "吃東西"))
    row = dict(key=c["key"], pid=c["pid"], ng=c["ng"], cat=c["cat"], note=c["note"], dur_s=c["dur"], y=gt)
    vs = [v for v in POSE_VIEWS if v in c["caches"] and pose_path(c, v).exists()]
    row["views"] = len(vs)
    if not vs:
        for g in GRID2:
            row[tag2(g)] = None
        return row
    dcs = {v: DetCache(c["caches"][v]) for v in vs}
    n = min(len(steps(x)[0]) for x in dcs.values())
    fs = {v: face_series(dcs[v], np.load(pose_path(c, v)), n) for v in vs}
    for g in GRID2:
        r, src, cm, T_s = g
        per = [fs[v][(r, "hand")] | fs[v][(r, "wrist")] if src == "either" else fs[v][(r, src)] for v in vs]
        if cm == "any":
            raw = np.any(per, 0)
        elif len(per) == 2:
            raw = (per[0] & dilate(per[1], BOTH_TOL)) | (per[1] & dilate(per[0], BOTH_TOL))
        else:
            raw = np.zeros(n, bool)                       # both 需要兩台都在
        k = touches(raw, T_s)
        row[tag2(g)] = int(bool(k))
        row["n_" + tag2(g)] = len(k)
        if g == GRID2[0]:
            row["touch_s"] = ";".join(f"{a * STEP:.1f}-{b * STEP:.1f}" for a, b in k)
    return row


def prf(rows, col):
    v = [(bool(r[col]), bool(r["y"])) for r in rows if r[col] is not None]
    tp = sum(p and y for p, y in v)
    fp = sum(p and not y for p, y in v)
    fn = sum(y and not p for p, y in v)
    R, P = tp / max(tp + fn, 1), tp / max(tp + fp, 1)
    return dict(tp=tp, fp=fp, fn=fn, tn=len(v) - tp - fp - fn, n=len(v), R=round(R, 3), P=round(P, 3),
                F1=round(2 * P * R / max(P + R, 1e-9), 3))


def write(name, data):
    with open(OUT / f"{name}.csv", "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(dict.fromkeys(k for r in data for k in r)))
        w.writeheader()
        w.writerows(data)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", type=int, default=0)
    ap.add_argument("--pose", action="store_true", help="§8.49 第 2 輪:ViTPose 臉部定位")
    args = ap.parse_args()
    global OUT
    t0 = time.time()
    clips = [c for c in E.load_clips().values() if not c["key"].startswith(E.EXCLUDE)
             and not c["key"].startswith("ev_" + E.EXCLUDE[0])]
    if args.smoke:
        clips = clips[:args.smoke]
    if args.pose:
        OUT = OUT2
        run_pose(clips)
        grid, row_fn, tag_fn = GRID2, clip_row2, tag2
    else:
        grid, row_fn, tag_fn = GRID, clip_row, tag
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"片段 {len(clips)}:正例 {sum(c['ng'] == 'NG3' or c['cat'] == '吃東西' and c['ng'] == '無NG' for c in clips)}",
          flush=True)
    rows = []
    for i, c in enumerate(clips, 1):
        rows.append(row_fn(c))
        if i % 20 == 0:
            print(f"  {i}/{len(clips)}  {time.time() - t0:.0f} 秒", flush=True)
    cols = [tag_fn(g) for g in grid]

    # 參與者留一
    pids = sorted({r["pid"] for r in rows})
    folds = []
    for pid in pids:
        tr = [r for r in rows if r["pid"] != pid]
        best = cols[0]
        for col in cols[1:]:
            if prf(tr, col)["F1"] > prf(tr, best)["F1"]:
                best = col
        te = [r for r in rows if r["pid"] == pid]
        for r in te:
            r["cv"], r["cv_param"] = r[best], best
        folds.append(dict(pid=pid, chosen=best, n_pos=sum(r["y"] for r in te), **prf(te, best)))
    cv = prf(rows, "cv")
    summ = [dict(setting="LOPO", **cv,
                 R_ci=E.wilson(cv["tp"], cv["tp"] + cv["fn"]), P_ci=E.wilson(cv["tp"], cv["tp"] + cv["fp"]))]
    summ += [dict(setting=col, **prf(rows, col)) for col in cols]
    # 分項:正例來源的召回、反例各類的誤觸發
    for name, sel in (("正例_NG3片段", lambda r: r["ng"] == "NG3"),
                      ("正例_無NG吃東西", lambda r: r["ng"] == "無NG" and r["cat"] == "吃東西")):
        s = [r for r in rows if sel(r)]
        summ.append(dict(setting=f"LOPO 召回 {name}", n=len(s), tp=sum(bool(r["cv"]) for r in s)))
    for key, cnt in Counter((r["ng"], r["cat"]) for r in rows if not r["y"]).items():
        s = [r for r in rows if not r["y"] and (r["ng"], r["cat"]) == key]
        summ.append(dict(setting=f"LOPO 誤觸發 {key[0]}_{key[1]}", n=cnt, fp=sum(bool(r["cv"]) for r in s)))
    write("per_clip", rows)
    write("summary", summ)
    write("folds", folds)
    print(f"LOPO {cv}  R_ci {summ[0]['R_ci']}  P_ci {summ[0]['P_ci']}")
    best = max(summ[1:len(cols) + 1], key=lambda d: d["F1"])
    print(f"預設 {summ[1]}\n最佳(全部資料,僅參考){best}")
    print(Counter(f["chosen"] for f in folds).most_common(5))
    for s in summ[len(cols) + 1:]:
        print(" ", s)
    print(f"輸出 {OUT},{time.time() - t0:.0f} 秒")


if __name__ == "__main__":
    main()
