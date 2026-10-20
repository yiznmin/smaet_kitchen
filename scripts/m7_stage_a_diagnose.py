"""M7 階段 A 結果拆解(看完結果後的檢討,不改評估)。

1. M6 §6.3 的 39 段錯誤:各組選定設定下,誤報段還有多少比例被判有接觸、漏報段有多少比例被補回
2. 每類:V 抓到但 A 沒抓到的真值時間步,A 是卡在哪一關(候選 / 接觸 / 在動),以及
   「物件遮罩扣掉手之後是空的」的比例(手遮罩把手上物件一起吃掉 → mask_iou = 1 → 被判靜止)

用法:
    python scripts/m7_stage_a_diagnose.py
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
from m6_assoc.gt import SCORED, build_gt, load_actions                    # noqa: E402
from m6_assoc.metrics import mask_from_intervals                          # noqa: E402
from m6_assoc.roles import load_roles, role_fuse                          # noqa: E402

CACHE = ROOT / "results/det_cache/epfl_allcls_armA_s3"
ANN = ROOT / "data/epfl_ann/YH2040/2023_09_12_09_50_41/annotations/actions_annotations.xlsx"
OUT = ROOT / "results/m7/stage_a"


def main():
    roles = load_roles(ROOT / "configs/camera_roles.epfl.json")
    close = roles["close"]
    cams = close["A"] + close["B"]
    gt, _ = build_gt(load_actions(ANN))
    t, step, F = R1.load_r1(str(CACHE), cams)
    T = len(t)
    sel = json.loads((OUT / "selected.json").read_text(encoding="utf-8"))["selected"]
    models = [a[2:] for a in sel if a.startswith("A-")]
    feats = {m: dict(np.load(OUT / f"feat_{m}.npz")) for m in models}
    side = feats[models[0]]["side"]

    preds, raws_all = {}, {}
    for arm in sel:
        cfg = sel[arm]["main"]["best_f1"]["cfg"]
        if arm == "V":
            raws = {c: R1.raw("V", F[R1.fkey(cfg)][c], cfg) for c in cams}
        else:
            mr = A.mask_raws(feats[arm[2:]], cams, T, cfg)
            raws = {c: mr[c] | (F[R1.fkey(cfg)][c]["upper"] >= R1.X_UPPER) for c in cams}
        raws_all[arm] = raws
        preds[arm] = E.smooth_all(role_fuse(raws, side, close, cfg["role_k"]), cfg)

    # 1. 39 段錯誤
    rows = list(csv.DictReader(open(ROOT / "results/m6/errors/errors.csv", encoding="utf-8-sig")))
    seg_out = []
    for r in rows:
        i = SCORED.index(r["cls"])
        m = (t >= float(r["start"])) & (t <= float(r["end"]))
        rec = dict(r)
        for arm, p in preds.items():
            rec[arm] = round(float(p[m, i].mean()), 2)
        seg_out.append(rec)
    with open(OUT / "error_segments.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(seg_out[0]))
        w.writeheader()
        w.writerows(seg_out)
    print("39 段錯誤(誤報段:仍判有接觸的比例,越低越好;漏報段:被補回的比例,越高越好)")
    for kind in ("FP", "FN"):
        for arm in preds:
            v = [s[arm] for s in seg_out if s["kind"] == kind]
            fixed = sum((x <= 0.2) if kind == "FP" else (x >= 0.5) for x in v)
            print(f"  {kind} {arm:14s} 平均 {np.mean(v):.2f}  修好 {fixed}/{len(v)}")
    occl = [s for s in seg_out if s["kind"] == "FP" and s["cls"] in ("刀具", "砧板", "鍋鏟", "鍋子")
            and s["start"] not in ("255.2",)]
    print("  遮擋誤報 9 段:", {arm: sum(s[arm] <= 0.2 for s in occl) for arm in preds}, f"/ {len(occl)}")

    # 2. V 抓到、A 沒抓到:卡在哪一關(單鏡頭單幀,該側近拍)
    print("\nV 單幀成立、A 單幀不成立的真值步(該側近拍鏡頭合計),A 卡在哪一關")
    diag = {}
    for m in models:
        cfg = sel[f"A-{m}"]["main"]["best_f1"]["cfg"]
        C, P = feats[m]["cand"], feats[m]["pair"]
        diag[m] = {}
        for ci_cls, cls in enumerate(SCORED):
            g = mask_from_intervals(t, gt[cls]["union"])
            cnt = Counter()
            for ci, cam in enumerate(cams):
                v = raws_all["V"][cam][:, ci_cls] & ~raws_all[f"A-{m}"][cam][:, ci_cls] & g
                for s in np.flatnonzero(v):
                    cs = np.flatnonzero((C[:, 0] == ci) & (C[:, 1] == s) & (C[:, 2] == ci_cls)
                                        & (C[:, 3] > np.float64(np.float32(cfg["thr_obj"]))))
                    if not len(cs):
                        cnt["沒有候選(門檻 / d 較嚴)"] += 1
                        continue
                    pp = P[np.isin(P[:, 0].astype(int), cs)]
                    pp = pp[(pp[:, 1] > np.float64(np.float32(cfg["thr_hand"]))) & (pp[:, 2] <= cfg["d"])]
                    if not len(pp):
                        cnt["沒有候選(門檻 / d 較嚴)"] += 1
                    elif not (pp[:, 3 + A.R_IDX[cfg["r"]]] > 0).any():
                        cnt["遮罩沒接觸"] += 1
                    elif (C[cs, 4] >= 1.0).all():
                        cnt["判靜止(遮罩 IoU = 1,含扣手後為空)"] += 1
                    else:
                        cnt["判靜止(遮罩 IoU ≥ m)"] += 1
            n = sum(cnt.values())
            diag[m][cls] = dict(n=n, **{k: round(v / max(n, 1), 3) for k, v in cnt.items()})
            print(f"  {m:10s} {cls}:{n:5d} 步  " + "  ".join(f"{k} {v / max(n, 1):.0%}" for k, v in cnt.items()))
    (OUT / "diagnose.json").write_text(json.dumps(diag, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
