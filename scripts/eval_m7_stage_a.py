"""M7 階段 A 評估:遮罩版接觸 + 在動,與 M6 的 V 比較。

預先登記:docs/M7_模型評估_預先登記_20261020.md §3
先跑 scripts/m7_stage_a_features.py 產生 results/m7/stage_a/feat_<model>.npz。

單幀判定(每台鏡頭、每類):
  V   :M6 修正第 1 輪(對照組,必須重現 0.725)
  A-* :存在一個候選物件,conf > thr_obj、mask_iou < m,且有一隻手 conf > thr_hand、間距 ≤ d、
        膨脹 r 後與物件遮罩重疊 > 0;另加上半身退路(同 V)
之後同 V:角色分工融合(role_k)→ 遲滯平滑 → 逐步 F1 最大 / Recall 優先點。

第 2 輪(§9):AM-* = 接觸用框間距(同 V)、在動用遮罩;AX-* = AM + 遮罩互斥扣除(feat_<model>_x.npz)

用法:
    python scripts/eval_m7_stage_a.py --arms V A-edgetam A-sam2.1_b+
    python scripts/eval_m7_stage_a.py --arms V AM-edgetam AM-sam2.1_b+ AX-edgetam AX-sam2.1_b+ --out-dir results/m7/stage_a2
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import eval_m6_epfl as E                                                  # noqa: E402
import eval_m6_round1 as R1                                               # noqa: E402
import eval_m6_round2 as R2                                               # noqa: E402
from m6_assoc.gt import SCORED, build_gt, load_actions, union             # noqa: E402
from m6_assoc.roles import load_roles, role_fuse                          # noqa: E402

M_GRID = [0.7, 0.8, 0.9]
R_IDX = {0.0: 0, 0.01: 1, 0.02: 2}
# 參考真值:只算 M3 學過的名詞(容器只算 Bowl、Salad_bowl;抹布只算 Towel)
NOT_LEARNED = {"Plate", "Box", "Bottle", "Tissue", "Sponge"}
# §10 按類別混用:這些類別走 AX(遮罩 + 互斥扣除),其餘走 V
HYBRID = {"HY1": ["容器"], "HY3": ["容器", "刀具", "鍋鏟"]}


def learned_gt(gt):
    out = {}
    for c, v in gt.items():
        segs = [s for s in v["segments"] if s[3] not in NOT_LEARNED]
        out[c] = {"segments": segs, "union": union([(s, e) for s, e, _, _ in segs])}
    return out


def mask_raws(feat, cams, T, cfg):
    """回傳 {cam: [T, n_cls]}:遮罩版單幀判定(不含上半身退路)。"""
    C, P = feat["cand"], feat["pair"]
    ok_c = (C[:, 3] > np.float64(np.float32(cfg["thr_obj"]))) & (C[:, 4] < cfg["m"])
    pc = P[:, 0].astype(int)
    ok_p = ok_c[pc] & (P[:, 1] > np.float64(np.float32(cfg["thr_hand"]))) & (P[:, 2] <= cfg["d"])
    if "r" in cfg:                                   # A:遮罩接觸;AM / AX(第 2 輪)不看遮罩接觸
        ok_p &= P[:, 3 + R_IDX[cfg["r"]]] > 0
    hit = C[np.unique(pc[ok_p])]
    out = {c: np.zeros((T, len(SCORED)), bool) for c in cams}
    for ci, cam in enumerate(cams):
        h = hit[hit[:, 0] == ci]
        out[cam][h[:, 1].astype(int), h[:, 2].astype(int)] = True
    return out


def a_cfgs(with_r=True):
    grid = dict(thr_obj=R1.FRAME_GRID["thr_obj"], thr_hand=R1.FRAME_GRID["thr_hand"], d=R1.D_GRID, m=M_GRID)
    if with_r:
        grid["r"] = list(R_IDX)
    frame = E.product(grid)
    return [dict(f, **sm, role_k=k) for f in frame for sm in R1.SMOOTH for k in R1.ROLE_K]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=["V", "A-edgetam", "A-sam2.1_b+"],
                    help="V;A-<model>(第 1 輪);AM-<model>、AX-<model>(第 2 輪)")
    ap.add_argument("--cache-dir", default=str(ROOT / "results/det_cache/epfl_allcls_armA_s3"))
    ap.add_argument("--ann", default=str(ROOT / "data/epfl_ann/YH2040/2023_09_12_09_50_41/annotations/actions_annotations.xlsx"))
    ap.add_argument("--feat-dir", default=str(ROOT / "results/m7/stage_a"))
    ap.add_argument("--out-dir", default=str(ROOT / "results/m7/stage_a"))
    args = ap.parse_args()
    out = Path(args.out_dir)
    t0 = time.time()

    roles = load_roles(ROOT / "configs/camera_roles.epfl.json")
    close = roles["close"]
    cams = close["A"] + close["B"]
    gt, _ = build_gt(load_actions(args.ann))
    gts = {"main": gt, "learned": learned_gt(gt)}
    print("讀 M6 特徵(V 對照組與上半身退路)…")
    t, step, F = R1.load_r1(args.cache_dir, cams)
    T = len(t)

    arms = args.arms
    def feat_name(arm):
        kind, model = arm.split("-", 1)
        return f"feat_{model}{'_x' if kind in ('AX', *HYBRID) else ''}"
    feats = {feat_name(a): dict(np.load(Path(args.feat_dir) / f"{feat_name(a)}.npz")) for a in arms if a != "V"}
    side = next(iter(feats.values()))["side"]
    for m, f in feats.items():
        assert (f["side"] == side).all() and np.allclose(f["t"], t), f"{m} 的時間軸 / 定位與 M6 不一致"

    selected, detail, grid_rows = {}, {}, []
    for arm in arms:
        for gname, g in gts.items():
            rows = []
            if arm == "V":
                cfgs = [dict(c, role_k=k) for c in R1.arm_cfgs("V") for k in R1.ROLE_K]
            elif arm.split("-")[0] in HYBRID:
                cfgs = [dict(c, iou_move=q) for c in a_cfgs(with_r=False) for q in R1.IOU_MOVES]
            else:
                cfgs = a_cfgs(with_r=arm.startswith("A-"))
            cache = {}
            for cfg in cfgs:
                key = tuple(cfg[k] for k in ("thr_obj", "thr_hand", "d", "m", "r", "iou_move") if k in cfg)
                if key not in cache:
                    if arm == "V":
                        raws = {c: R1.raw("V", F[R1.fkey(cfg)][c], cfg) for c in cams}
                    elif arm.split("-")[0] in HYBRID:
                        mr = mask_raws(feats[feat_name(arm)], cams, T, cfg)
                        cols = [SCORED.index(x) for x in HYBRID[arm.split("-")[0]]]
                        raws = {}
                        for c in cams:
                            r = R1.raw("V", F[R1.fkey(cfg)][c], cfg).copy()
                            r[:, cols] = mr[c][:, cols] | (F[R1.fkey(cfg)][c]["upper"][:, cols] >= R1.X_UPPER)
                            raws[c] = r
                    else:
                        mr = mask_raws(feats[feat_name(arm)], cams, T, cfg)
                        raws = {c: mr[c] | (F[R1.fkey(cfg)][c]["upper"] >= R1.X_UPPER) for c in cams}
                    cache[key] = raws
                fused = role_fuse(cache[key], side, close, cfg["role_k"])
                res = E.score(t, step, E.smooth_all(fused, cfg), g)
                rows.append(dict(cfg=cfg, summary=E.flat(res), result=res))
                grid_rows.append([arm, gname, json.dumps(cfg)] +
                                 [round(E.flat(res)[k], 4) for k in E.SUMMARY_KEYS])
            sel = {"best_f1": R2.pick_f1(rows), "recall_first": R2.pick_recall(rows)}
            selected.setdefault(arm, {})[gname] = {
                k: (dict(cfg=v["cfg"], summary=v["summary"],
                         per_class={c: v["result"]["per_class"][c]["timestep"] for c in SCORED})
                    if v else None) for k, v in sel.items()}
            print(f"[{arm} / {gname}] {len(rows)} 組,{time.time() - t0:.0f} 秒", flush=True)

    with open(out / "grid.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["arm", "gt", "cfg"] + E.SUMMARY_KEYS)
        w.writerows(grid_rows)
    meta = {m: json.loads(str(f["meta"])) for m, f in feats.items()}
    (out / "selected.json").write_text(json.dumps(dict(selected=selected, feat_meta=meta),
                                                  ensure_ascii=False, indent=2), encoding="utf-8")

    for gname in gts:
        print(f"\n真值:{gname}")
        for arm in arms:
            for key in ("best_f1", "recall_first"):
                v = selected[arm][gname][key]
                if v is None:
                    print(f"  {arm:14s} {key:12s} 沒有設定達逐步 P ≥ {R2.P_FLOOR}")
                    continue
                s = v["summary"]
                print(f"  {arm:14s} {key:12s} 逐步 P {s['timestep_P']:.3f} R {s['timestep_R']:.3f} "
                      f"F1 {s['timestep_F1']:.3f} | 逐段 P {s['segment_P']:.3f} R {s['segment_R']:.3f}  "
                      f"{json.dumps(v['cfg'])}")
            pc = selected[arm][gname]["best_f1"]["per_class"]
            print("      逐類 P/R " + "  ".join(f"{c} {x['P']:.2f}/{x['R']:.2f}" for c, x in pc.items()))
    print(f"\n輸出 {out},{time.time() - t0:.0f} 秒")


if __name__ == "__main__":
    main()
