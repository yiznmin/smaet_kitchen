"""NG4 逐次比對:在團隊 NG4 每一次的前後 30 秒視窗內,用規則找出每一次 NG4(清洗後放桌面 / 放桌面後未清洗即使用)。

預先登記:docs/M6_NG準則評估_預先登記_20261027.md §8.13
- 片段:data/ng4_win/<視窗>/<鏡頭>.mp4;快取:results/det_cache/ng4_win_s3/<視窗>/<鏡頭>.npz(不進 git)
- 輸出:results/m6/ng/ng4/detections.csv、truth.csv、summary.csv(含團隊標註,不進 git)

用法:
    python scripts/eval_ng4_instances.py --prepare
    python scripts/eval_ng4_instances.py
"""
import argparse
import csv
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import eval_ng_clips as E                                                 # noqa: E402
from fetch_ng_windows import FPS, VID_ROOT, cache                         # noqa: E402
from m3.classes import NAME_TO_CLASS                                      # noqa: E402
from m4_track.det_cache import DetCache                                   # noqa: E402
from m6_events.association import ioa                                     # noqa: E402
from m6_events.smoothing import hysteresis                                # noqa: E402
from m6_events.violation import HAND, at, in_poly, intervals, steps       # noqa: E402

CUT = ROOT / "data/ng4_win"
CACHE = ROOT / "results/det_cache/ng4_win_s3"
OUT = ROOT / "results/m6/ng/ng4"
VIEWS = ["Aoutput1", "Aoutput2", "Aoutput3", "Boutput1", "Boutput2"]
PAD_S, AFTER_WASH_S, HELD_S, MATCH_S = 30.0, 30.0, 0.5, 2.0
OBJ = ["食材", "刀具", "砧板", "鍋鏟", "鍋子", "容器"]
SUP = ["砧板", "容器"]
OBJ_THR, SUP_IOA, HAND_THR = 0.4, 0.3, 0.2
GRID = [(1.0, 1.0), (0.5, 1.0), (2.0, 1.0), (1.0, 2.0), (0.5, 2.0), (2.0, 2.0)]
STEP = 0.1


def truth():
    out = []
    for r in csv.DictReader(open(E.NG_DIR / "ng_clips_index.csv", encoding="utf-8-sig")):
        if r["ng_type"] != "NG4" or r["view"] != "Aoutput1":
            continue
        out.append(dict(session=r["session_id"], kind="NG4a" if "清洗後" in r["reason"] else "NG4b",
                        start=float(r["clip_start_sec"]), end=float(r["clip_end_sec"]), reason=r["reason"],
                        note=r["note"]))
    return out


def windows(gt):
    by = defaultdict(list)
    for g in gt:
        by[g["session"]].append([max(0.0, g["start"] - PAD_S), g["end"] + PAD_S])
    out = []
    for s, ws in by.items():
        ws.sort()
        merged = [ws[0]]
        for a, b in ws[1:]:
            if a <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], b)
            else:
                merged.append([a, b])
        for a, b in merged:
            pid, sess = s.split("/")[1:]
            f0 = int(a * FPS)
            out.append(dict(key=f"{pid}_{sess}_{f0}", session=s, f0=f0, t0=f0 / FPS, dur=b - f0 / FPS))
    return out


def prepare(wins):
    for w in wins:
        src = VID_ROOT / w["session"].split("/", 1)[1]
        for v in VIEWS:
            out = CUT / w["key"] / f"{v}.mp4"
            if out.exists() or (CACHE / w["key"] / f"{v}.npz").exists() or not (src / f"{v}.mp4").exists():
                continue
            out.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{(w['f0'] - 0.5) / FPS:.4f}", "-i", str(src / f"{v}.mp4"),
                            "-t", f"{w['dur']:.3f}", "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
                            "-r", "30", str(out)], check=True)
        cache([CUT / w["key"] / f"{v}.mp4" for v in VIEWS], VIEWS, CACHE / w["key"])


def centers(b):
    return np.c_[(b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2]


def series(cs, sink):
    """wash / placed / held:[T, 物件類別] 布林。"""
    T = min(len(steps(c)[0]) for c in cs.values())
    ids = [NAME_TO_CLASS[n] for n in OBJ]
    sup_ids = [NAME_TO_CLASS[n] for n in SUP]
    wash, placed, held = (np.zeros((T, len(OBJ)), bool) for _ in range(3))
    for v, c in cs.items():
        idx = steps(c)
        for i in range(T):
            xy, cf, cl = at(c, idx, i)
            h = xy[(cl == HAND) & (cf >= HAND_THR)]
            hand_in_sink = v in sink and len(h) and in_poly(centers(h), sink[v]).any()
            sup = xy[np.isin(cl, sup_ids) & (cf >= OBJ_THR)]
            for j, oid in enumerate(ids):
                b = xy[(cl == oid) & (cf >= OBJ_THR)]
                if not len(b):
                    continue
                if v in sink:
                    ins = in_poly(centers(b), sink[v])
                    if hand_in_sink and ins.any():
                        wash[i, j] = True
                    b = b[~ins]
                    if not len(b):
                        continue
                touch = ioa(b, h).max(1) > 0 if len(h) else np.zeros(len(b), bool)
                held[i, j] |= bool(touch.any())
                free = b[~touch]
                if len(free) and len(sup):
                    s2 = sup[~np.all(sup == free[:, None], -1).any(0)] if OBJ[j] in SUP else sup
                    if len(s2):
                        free = free[ioa(free, s2).max(1) < SUP_IOA]
                placed[i, j] |= len(free) > 0
    placed = np.stack([hysteresis(placed[:, j], 3, 2, 3) for j in range(len(OBJ))], 1)
    return T, wash, placed, held


def detect(T, wash, placed, held, w, p):
    k = lambda s: max(1, int(round(s / STEP)))
    det = []
    for j, name in enumerate(OBJ):
        pl = [(s, e) for s, e in intervals(placed[:, j]) if e - s >= k(p)]
        # NG4a:清洗 ≥ w 秒,結束後 30 秒內放桌面 ≥ p 秒
        for ws, we in intervals(wash[:, j]):
            if we - ws < k(w):
                continue
            nxt = [(s, e) for s, e in pl if we <= s <= we + k(AFTER_WASH_S)]
            if nxt:
                det.append(dict(kind="NG4a", obj=name, start=ws, end=nxt[0][0] + k(p)))
        # NG4b:放桌面 ≥ p 秒 → 拿起 ≥ 0.5 秒,中間沒洗
        for s, e in pl:
            hs = [(a, b) for a, b in intervals(held[:, j]) if a >= s + k(p) and b - a >= k(HELD_S)]
            if hs and not wash[s:hs[0][0], j].any():
                det.append(dict(kind="NG4b", obj=name, start=s, end=hs[0][0]))
    return det


def overlap(a0, a1, b0, b1):
    return min(a1, b1) + MATCH_S >= max(a0, b0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepare", action="store_true")
    args = ap.parse_args()
    gt = truth()
    wins = windows(gt)
    print(f"NG4 真值 {len(gt)} 次、視窗 {len(wins)} 個", flush=True)
    if args.prepare:
        prepare(wins)
        print("全部完成", flush=True)
        return
    cfg = yaml.safe_load((E.INTEG / "configs/violation_rules.yaml").read_text(encoding="utf-8"))
    sink = {v: E.get_zone(E.INTEG / cfg["zones"][v], cfg["hand_hygiene"]["sink_zone"]) for v in ("Boutput1", "Boutput2")}
    dets, hit_gt = [], defaultdict(set)
    for w in wins:
        cs = {v: DetCache(CACHE / w["key"] / f"{v}.npz") for v in VIEWS}
        T, wash, placed, held = series(cs, sink)
        mine = [g for g in gt if g["session"] == w["session"] and w["t0"] <= g["start"] <= w["t0"] + w["dur"]]
        for gp in GRID:
            for d in detect(T, wash, placed, held, *gp):
                a, b = w["t0"] + d["start"] * STEP, w["t0"] + d["end"] * STEP
                m = [gt.index(g) for g in mine if overlap(a, b, g["start"], g["end"])]
                mk = [gt.index(g) for g in mine if overlap(a, b, g["start"], g["end"]) and g["kind"] == d["kind"]]
                dets.append(dict(grid=f"w{gp[0]}_p{gp[1]}", window=w["key"], kind=d["kind"], obj=d["obj"],
                                 start=round(a, 1), end=round(b, 1), match_any=int(bool(m)), match_kind=int(bool(mk))))
                hit_gt[(gp, "any")].update(m)
                hit_gt[(gp, "kind")].update(mk)
    OUT.mkdir(parents=True, exist_ok=True)
    E.OUT = OUT
    E.write_csv("detections", dets)
    E.write_csv("truth", [dict(g, idx=i, **{f"hit_w{gp[0]}_p{gp[1]}": int(i in hit_gt[(gp, 'any')]) for gp in GRID})
                          for i, g in enumerate(gt)])
    summ = []
    for gp in GRID:
        tag = f"w{gp[0]}_p{gp[1]}"
        d = [x for x in dets if x["grid"] == tag]
        row = dict(grid=tag, n_det=len(d), det_NG4a=sum(x["kind"] == "NG4a" for x in d),
                   det_NG4b=sum(x["kind"] == "NG4b" for x in d))
        for kind in ("NG4a", "NG4b", "全部"):
            idx = [i for i, g in enumerate(gt) if kind == "全部" or g["kind"] == kind]
            k_any = sum(i in hit_gt[(gp, "any")] for i in idx)
            k_kind = sum(i in hit_gt[(gp, "kind")] for i in idx)
            row[f"R_{kind}"] = f"{k_any}/{len(idx)}"
            row[f"R_{kind}_同類"] = f"{k_kind}/{len(idx)}"
        tp = sum(x["match_any"] for x in d)
        row["P"] = round(tp / max(len(d), 1), 3)
        row["P_ci"] = "-".join(map(str, E.wilson(tp, len(d))))
        lo, hi = E.wilson(sum(i in hit_gt[(gp, "any")] for i in range(len(gt))), len(gt))
        row["R_全部_ci"] = f"{lo}-{hi}"
        summ.append(row)
        print(f"  {tag:10s} 偵測 {len(d):3d}(a {row['det_NG4a']}、b {row['det_NG4b']})  "
              f"R 全部 {row['R_全部']}(同類 {row['R_全部_同類']})  NG4a {row['R_NG4a']}  NG4b {row['R_NG4b']}  "
              f"P {row['P']}", flush=True)
    E.write_csv("summary", summ)


if __name__ == "__main__":
    main()
