"""M7 按類別混用(HY3)的逐類結果、錯誤拆解與錯誤影像。

預先登記:docs/M7_模型評估_預先登記_20261020.md §10(結果 §10.1);本腳本只整理,不改設定。
設定取 results/m7/stage_a3/selected.json(主真值、F1 最大點)。

輸出(results/m7/hy3/,本機;EPFL 影像不進 git):
  per_class.csv     V、HY3-SAM 2.1、HY3-EdgeTAM 的逐類指標(逐步、逐段、F1@τ)與 pooled
  fn_breakdown.csv  漏報時間步卡在哪一關(該側近拍鏡頭合計)
  fp_breakdown.csv  誤報時間步當下,真值裡人在做什麼
  v_errors_compare.csv  M6 §6.3 的 39 段錯誤在 HY3 下的狀態
  errors/<類別>.jpg、errors/errors.csv  HY3-SAM 2.1 每類最長 3 段誤報、3 段漏報

用法:
    python scripts/m7_hy3_report.py
"""
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import eval_m6_epfl as E                                                  # noqa: E402
import eval_m6_round1 as R1                                               # noqa: E402
import eval_m7_stage_a as A                                               # noqa: E402
import render_m6_errors as RE                                             # noqa: E402
from common.video_io import imwrite_unicode                               # noqa: E402
from m6_assoc.gt import NOUN_TO_CLASS, SCORED, build_gt, load_actions     # noqa: E402
from m6_assoc.metrics import (boundary_mask, intervals_from_mask,         # noqa: E402
                              mask_from_intervals)
from m6_assoc.roles import load_roles, role_fuse                          # noqa: E402

CACHE = ROOT / "results/det_cache/epfl_allcls_armA_s3"
ANN = ROOT / "data/epfl_ann/YH2040/2023_09_12_09_50_41/annotations/actions_annotations.xlsx"
SEL = ROOT / "results/m7/stage_a3/selected.json"
OUT = ROOT / "results/m7/hy3"
ARMS = ["V", "HY3-sam2.1_b+", "HY3-edgetam"]
MAIN = "HY3-sam2.1_b+"
NON_CONTACT = ("Wait", "Squat", "Read")


def build_raws(arm, cfg, F, feats, cams, T):
    if arm == "V":
        return {c: R1.raw("V", F[R1.fkey(cfg)][c], cfg) for c in cams}
    kind, model = arm.split("-", 1)
    mr = A.mask_raws(feats[model], cams, T, cfg)
    cols = [SCORED.index(x) for x in A.HYBRID[kind]]
    out = {}
    for c in cams:
        r = R1.raw("V", F[R1.fkey(cfg)][c], cfg).copy()
        r[:, cols] = mr[c][:, cols] | (F[R1.fkey(cfg)][c]["upper"][:, cols] >= R1.X_UPPER)
        out[c] = r
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    roles = load_roles(ROOT / "configs/camera_roles.epfl.json")
    close = roles["close"]
    cams = close["A"] + close["B"]
    acts = load_actions(ANN)
    gt, _ = build_gt(acts)
    t, step, F = R1.load_r1(str(CACHE), cams)
    T = len(t)
    sel = json.loads(SEL.read_text(encoding="utf-8"))["selected"]
    feats = {m: dict(np.load(ROOT / f"results/m7/stage_a/feat_{m}_x.npz")) for m in ("sam2.1_b+", "edgetam")}
    side = feats["edgetam"]["side"]

    preds, raws, cfgs = {}, {}, {}
    for arm in ARMS:
        cfg = sel[arm]["main"]["best_f1"]["cfg"]
        cfgs[arm] = cfg
        raws[arm] = build_raws(arm, cfg, F, feats, cams, T)
        preds[arm] = E.smooth_all(role_fuse(raws[arm], side, close, cfg["role_k"]), cfg)

    # 1. 逐類指標
    rows = []
    for arm in ARMS:
        res = E.score(t, step, preds[arm], gt)
        for cls in SCORED + ["pooled"]:
            r = res["pooled"] if cls == "pooled" else res["per_class"][cls]
            ts, sg = r["timestep"], r["segment"]
            rows.append(dict(arm=arm, cls=cls, ts_P=ts["P"], ts_R=ts["R"], ts_F1=ts["F1"],
                             tp=ts["tp"], fp=ts["fp"], fn=ts["fn"],
                             seg_P=sg["P"], seg_R=sg["R"], seg_F1=sg["F1"],
                             gt_segs=sg["gt_n"], gt_hit=sg["gt_hit"], pred_segs=sg["pred_n"], pred_ok=sg["pred_ok"],
                             **{f"f1@{k}": v["F1"] for k, v in r["f1_at"].items()}))
    with open(OUT / "per_class.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("逐類(逐步 P / R / F1)")
    for arm in ARMS:
        print(f"  {arm:14s} " + "  ".join(f"{r['cls']} {r['ts_P']:.2f}/{r['ts_R']:.2f}/{r['ts_F1']:.2f}"
                                          for r in rows if r["arm"] == arm))

    # 2. 錯誤拆解(HY3-SAM 2.1)
    cfg, p, rw = cfgs[MAIN], preds[MAIN], raws[MAIN]
    f = {c: F[R1.fkey(cfg)][c] for c in cams}
    act_cls = [[] for _ in range(T)]                 # 每步真值動作(類別或名詞)
    for a in acts:
        if a["verb"] in NON_CONTACT or not a["noun"]:
            continue
        lab = NOUN_TO_CLASS.get(a["noun"], a["noun"])
        for i in np.flatnonzero((t >= a["start"]) & (t < a["end"])):
            act_cls[i].append(lab)
    fn_rows, fp_rows = [], []
    for i, cls in enumerate(SCORED):
        g = mask_from_intervals(t, gt[cls]["union"])
        care = ~boundary_mask(t, gt[cls]["union"], E.TOL)
        fn_c, fp_c = Counter(), Counter()
        for s in np.flatnonzero(g & ~p[:, i] & care):
            cs = close[side[s]] if side[s] in ("A", "B") else cams
            if any(rw[c][s, i] for c in cs):
                fn_c["單鏡頭有成立,融合 / 平滑沒過"] += 1
            elif any(f[c]["gap"][s, i] <= cfg["d"] for c in cs):
                fn_c["在手邊但判為靜止"] += 1
            elif any(f[c]["present"][s, i] > 0 for c in cs):
                fn_c["有偵測到但不在手邊"] += 1
            else:
                fn_c["近拍鏡頭沒偵測到"] += 1
        for s in np.flatnonzero(~g & p[:, i] & care):
            others = [x for x in act_cls[s] if x != cls]
            if any(x in SCORED for x in others):
                fp_c["人正在碰其他計分類別"] += 1
            elif others:
                fp_c["人正在碰非計分物件(手、水、冰箱…)"] += 1
            else:
                fp_c["真值沒有任何動作"] += 1
        n_fn, n_fp = sum(fn_c.values()), sum(fp_c.values())
        fn_rows.append(dict(cls=cls, n=n_fn, **{k: round(v / max(n_fn, 1), 3) for k, v in fn_c.items()}))
        fp_rows.append(dict(cls=cls, n=n_fp, **{k: round(v / max(n_fp, 1), 3) for k, v in fp_c.items()}))
    for name, rr in (("fn_breakdown.csv", fn_rows), ("fp_breakdown.csv", fp_rows)):
        keys = list(dict.fromkeys(k for r in rr for k in r))
        with open(OUT / name, "w", newline="", encoding="utf-8-sig") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            w.writerows(rr)
        print(f"\n{name}")
        for r in rr:
            print("  " + "  ".join(f"{k} {v}" for k, v in r.items()))

    # 3. M6 的 39 段錯誤在 HY3 下的狀態
    v_err = list(csv.DictReader(open(ROOT / "results/m6/errors/errors.csv", encoding="utf-8-sig")))
    cmp_rows = []
    for r in v_err:
        m = (t >= float(r["start"])) & (t <= float(r["end"]))
        frac = float(p[m, SCORED.index(r["cls"])].mean())
        fixed = frac <= 0.2 if r["kind"] == "FP" else frac >= 0.5
        cmp_rows.append(dict(r, hy3_on_frac=round(frac, 2), fixed=int(fixed)))
    with open(OUT / "v_errors_compare.csv", "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(cmp_rows[0]))
        w.writeheader()
        w.writerows(cmp_rows)
    for kind in ("FP", "FN"):
        k = [r for r in cmp_rows if r["kind"] == kind]
        print(f"\nM6 的 {kind} {len(k)} 段,HY3 修好 {sum(r['fixed'] for r in k)} 段:" +
              "、".join(f"{r['cls']} {r['start']}" for r in k if r["fixed"]))

    # 4. HY3 錯誤影像
    edir = OUT / "errors"
    edir.mkdir(exist_ok=True)
    rcfg = dict(cfg, iou_move=cfg["iou_move"])
    err_rows = []
    for i, cls in enumerate(SCORED):
        g = mask_from_intervals(t, gt[cls]["union"])
        cells = []
        for kind, mask in (("FP", p[:, i] & ~g), ("FN", g & ~p[:, i])):
            ivs = sorted(intervals_from_mask(t, mask, step), key=lambda v: v[0] - v[1])[:3]
            for s, e in ivs:
                k = int(round((s + e) / 2 / step))
                sd = side[k]
                group = close[sd] if sd in ("A", "B") else cams
                fired = [c for c in group if rw[c][k, i]]
                cam = (fired[0] if fired else group[0]) if kind == "FP" else (group[0] if sd in ("A", "B") else "Aoutput2")
                now = [f"{a['verb']} {a['noun']}" for a in acts
                       if a["start"] <= t[k] < a["end"] and a["verb"] not in NON_CONTACT]
                cells.append(RE.draw_cell(cam, k, cls, rcfg, kind, (s, e), now))
                err_rows.append(dict(cls=cls, kind=kind, start=round(s, 1), end=round(e, 1), dur=round(e - s, 1),
                                     cam=cam, side=sd, via="SAM" if cls in A.HYBRID["HY3"] else "V",
                                     gt_actions="; ".join(now)))
            while len(cells) % 3:
                cells.append(np.zeros((RE.CELL_H + RE.HEAD, RE.CELL_W, 3), np.uint8))
        if cells:
            grid = np.vstack([np.hstack(cells[r:r + 3]) for r in range(0, len(cells), 3)])
            imwrite_unicode(edir / f"{cls}.jpg", grid)
    with open(edir / "errors.csv", "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(err_rows[0]))
        w.writeheader()
        w.writerows(err_rows)
    print(f"\n輸出 {OUT}")


if __name__ == "__main__":
    main()
