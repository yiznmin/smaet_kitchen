"""NG3 新定義「手碰到臉」評估(登記 §8.47)。

正例 = 團隊 NG3 片段 + 無NG 吃東西事件;反例 = 其他類 NG 片段 + 其他類無NG 事件(排除 YH2003 05_17)。
規則 face_touch:指定鏡頭任一台連續 ≥ T 秒「手框中心在人框頭部區」。參與者留一選參數。

用法:
    python scripts/eval_ng3_face.py            # 全部片段
    python scripts/eval_ng3_face.py --smoke 3  # 只跑前 3 個片段(冒煙)
"""
import argparse
import csv
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
    args = ap.parse_args()
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    clips = [c for c in E.load_clips().values() if not c["key"].startswith(E.EXCLUDE)
             and not c["key"].startswith("ev_" + E.EXCLUDE[0])]
    if args.smoke:
        clips = clips[:args.smoke]
    print(f"片段 {len(clips)}:正例 {sum(c['ng'] == 'NG3' or c['cat'] == '吃東西' and c['ng'] == '無NG' for c in clips)}",
          flush=True)
    rows = []
    for i, c in enumerate(clips, 1):
        rows.append(clip_row(c))
        if i % 20 == 0:
            print(f"  {i}/{len(clips)}  {time.time() - t0:.0f} 秒", flush=True)
    cols = [tag(g) for g in GRID]

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
                 R_ci=tuple(map(float, E.wilson(cv["tp"], cv["tp"] + cv["fn"]))), P_ci=tuple(map(float, E.wilson(cv["tp"], cv["tp"] + cv["fp"]))))]
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
