"""M6 修正第 1 輪:一手一物(N)、只算有在動的物件(V)、兩者並用(NV)。

預先登記:docs/M6_修正第1輪_預先登記_20261019.md
設定沿用第一輪最好的「角色分工(自動定位)」:位置門檻凍結自 results/m6/dev_yh2040_roles/selected_roles.json。
主要判讀改為**逐步 F1**(第一輪 §9.4:逐段 P 會被碎片化區間拉高)。

用法:
    python scripts/eval_m6_round1.py --cache-dir results/det_cache/epfl_allcls_armA_s3 \\
        --ann data/epfl_ann/YH2040/2023_09_12_09_50_41/annotations/actions_annotations.xlsx \\
        --side-labels configs/epfl_side_labels/YH2040_2023_09_12_09_50_41.csv \\
        --roles-selected results/m6/dev_yh2040_roles/selected_roles.json \\
        --out-dir results/m6/dev_yh2040_r1
"""
import argparse
import csv
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import eval_m6_epfl as E                                                  # noqa: E402
import eval_m6_roles as R                                                 # noqa: E402
from m3.classes import NAME_TO_CLASS                                      # noqa: E402
from m4_track.det_cache import DetCache                                   # noqa: E402
from m6_assoc.association import frame_features, frame_features_r1        # noqa: E402
from m6_assoc.gt import SCORED, build_gt, load_actions                    # noqa: E402
from m6_assoc.roles import auto_side, gate_gt, load_roles, load_side_labels, role_fuse  # noqa: E402

# ── 預先登記的網格 ───────────────────────────────────────────────
FIXED = dict(thr_person=0.3, hand_in_person=0.5)
FRAME_GRID = dict(thr_obj=[0.4, 0.5], thr_hand=[0.2, 0.3])
D_GRID = [0.0, 0.05, 0.1]
IOU_MOVES = [0.7, 0.8, 0.9]
X_UPPER = 0.3
X_PERSON = [0.1, 0.3, 0.5]
LAG = 5                     # 動態比較的間隔(步;stride 3 → 0.5 秒)
SMOOTH = E.product(E.SMOOTH_GRID)
ROLE_K = [1, 2]
ARMS = ["orig", "N", "V", "NV", "base_present", "base_person"]


def load_r1(cache_dir, cams):
    """回傳 t, step, F;F[(thr_obj, thr_hand)][cam] = dict(各特徵陣列,第 0 維為時間)。"""
    class_ids = [NAME_TO_CLASS[c] for c in SCORED]
    F, t_ref, step = {}, None, None
    for cam in cams:
        c = DetCache(Path(cache_dir) / f"{cam}.npz", expect={"classes": "all", "complete": True})
        fps, stride = float(c.meta["video_meta"]["fps"]), int(c.meta["cache_stride"])
        fids = np.arange(0, int(c.meta["max_fid"]) + 1, stride)
        if t_ref is None:
            t_ref, step = fids / fps, stride / fps
        lo = np.searchsorted(c._fid, fids, side="left")
        hi = np.searchsorted(c._fid, fids, side="right")
        for to, th in itertools.product(FRAME_GRID["thr_obj"], FRAME_GRID["thr_hand"]):
            rows = []
            for i, (a, b) in enumerate(zip(lo, hi)):
                pa, pb = (lo[i - LAG], hi[i - LAG]) if i >= LAG else (a, a)
                r1 = frame_features_r1(c._xyxy[a:b], c._conf[a:b], c._cls[a:b],
                                       c._xyxy[pa:pb], c._conf[pa:pb], c._cls[pa:pb], class_ids,
                                       thr_obj=to, thr_hand=th, iou_moves=IOU_MOVES, **FIXED)
                base = frame_features(c._xyxy[a:b], c._conf[a:b], c._cls[a:b], class_ids,
                                      thr_obj=to, thr_hand=th, **FIXED)
                r1["present"], r1["person_ioa"] = base[:, 0], base[:, 1]
                rows.append(r1)
            F.setdefault((to, th), {})[cam] = {k: np.stack([r[k] for r in rows]) for k in rows[0]}
        print(f"  {cam}:{len(fids):,} 步")
    n = len(t_ref)
    return t_ref[:n], step, F


def raw(arm, f, cfg):
    """單幀判定 [T, n_cls]。"""
    if arm == "base_present":
        return f["present"] > 0
    if arm == "base_person":
        return f["person_ioa"] >= cfg["x_person"]
    g = {"orig": f["gap"], "N": f["nn_gap"]}.get(arm)
    if g is None:
        q = IOU_MOVES.index(cfg["iou_move"])
        g = (f["mv_gap"] if arm == "V" else f["nnmv_gap"])[:, q, :]
    by_hand = g <= cfg["d"]
    return by_hand | (f["upper"] >= X_UPPER)


def arm_cfgs(arm):
    frame = E.product(FRAME_GRID)
    extra = {"orig": dict(d=D_GRID), "N": dict(d=D_GRID),
             "V": dict(d=D_GRID, iou_move=IOU_MOVES), "NV": dict(d=D_GRID, iou_move=IOU_MOVES),
             "base_present": {}, "base_person": dict(x_person=X_PERSON)}[arm]
    if arm == "base_present":
        frame = E.product(dict(thr_obj=FRAME_GRID["thr_obj"]))
    return [dict(fc, **mc, **sm) for fc in frame for mc in E.product(extra) for sm in SMOOTH]


def fkey(cfg):
    return (cfg["thr_obj"], cfg.get("thr_hand", FRAME_GRID["thr_hand"][0]))


def pick(rows):
    """主要判讀:逐步 F1 最大;同分取逐步 R 大者。"""
    return max(rows, key=lambda r: (r["summary"]["timestep_F1"], r["summary"]["timestep_R"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--ann", required=True)
    ap.add_argument("--side-labels", required=True)
    ap.add_argument("--roles-selected", required=True, help="第一輪角色分工的 selected_roles.json(取自動定位門檻)")
    ap.add_argument("--roles", default=str(ROOT / "configs" / "camera_roles.epfl.json"))
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--frozen", default=None)
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    roles = load_roles(args.roles)
    close, locator = roles["close"], roles["locator"]
    cams = close["A"] + close["B"]
    gt, _ = build_gt(load_actions(args.ann))
    print("讀快取、算單幀特徵…")
    t, step, F = load_r1(args.cache_dir, cams)
    oracle = load_side_labels(args.side_labels, t)
    auto_x = json.loads(Path(args.roles_selected).read_text(encoding="utf-8"))["auto_x"]
    side = auto_side(R.foot_x(args.cache_dir, locator, t), **auto_x)
    gated = {s: gate_gt(gt, t, step, oracle == s) for s in close}
    frozen = json.loads(Path(args.frozen).read_text(encoding="utf-8")) if args.frozen else None

    selected, detail, grid_rows = {}, {}, []
    for arm in ARMS:
        cfgs_sys = ([frozen[arm]["role_auto"]["cfg"]] if frozen else
                    [dict(c, role_k=k) for c in arm_cfgs(arm) for k in ROLE_K])
        rows = []
        for cfg in cfgs_sys:
            raws = {c: raw(arm, F[fkey(cfg)][c], cfg) for c in cams}
            res = E.score(t, step, E.smooth_all(role_fuse(raws, side, close, cfg["role_k"]), cfg), gt)
            rows.append(dict(cfg=cfg, summary=E.flat(res), result=res))
            grid_rows.append([arm, "role_auto", json.dumps(cfg)] + [round(E.flat(res)[k], 4)
                                                                   for k in E.SUMMARY_KEYS])
        best = pick(rows)
        selected.setdefault(arm, {})["role_auto"] = dict(cfg=best["cfg"], summary=best["summary"])
        detail.setdefault(arm, {})["role_auto"] = best["result"]

        cfgs_gate = ([frozen[arm]["close_gated"]["cfg"]] if frozen else arm_cfgs(arm))
        rows = []
        for cfg in cfgs_gate:
            per_cam = {}
            for s, cs in close.items():
                for c in cs:
                    per_cam[c] = E.score(t, step, E.smooth_all(raw(arm, F[fkey(cfg)][c], cfg), cfg),
                                         gated[s])
            rows.append(dict(cfg=cfg, summary=E.summarize(per_cam), per_cam=per_cam))
        best = pick(rows)
        selected[arm]["close_gated"] = dict(cfg=best["cfg"], summary=best["summary"])
        detail[arm]["close_gated"] = best["per_cam"]
        print(f"[{arm}] 完成")

    with open(out / "grid.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["arm", "system", "cfg"] + E.SUMMARY_KEYS)
        w.writerows(grid_rows)
    (out / "selected.json").write_text(json.dumps(selected, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "metrics.json").write_text(json.dumps(dict(
        meta=dict(cache_dir=args.cache_dir, ann=args.ann, auto_x=auto_x, lag_steps=LAG,
                  frame_grid=FRAME_GRID, d=D_GRID, iou_moves=IOU_MOVES, x_upper=X_UPPER,
                  frozen=args.frozen, seconds=round(time.time() - t0, 1)),
        selected=selected, detail=detail), ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n結果(主要判讀 = 角色分工(自動)的逐步 F1)")
    for arm in ARMS:
        for key in ("role_auto", "close_gated"):
            s, c = selected[arm][key]["summary"], selected[arm][key]["cfg"]
            print(f"  {arm:13s} {key:11s} 逐步 P {s['timestep_P']:.3f} R {s['timestep_R']:.3f} "
                  f"F1 {s['timestep_F1']:.3f} | 逐段 P {s['segment_P']:.3f} R {s['segment_R']:.3f} | "
                  f"F1@0.25 {s['f1@0.25']:.3f}  {json.dumps(c)}")
    print("\n逐類(角色分工自動;逐步 P / R)")
    for arm in ("orig", "N", "V", "NV"):
        pc = detail[arm]["role_auto"]["per_class"]
        print(f"  {arm:5s} " + "  ".join(f"{c} {v['timestep']['P']:.2f}/{v['timestep']['R']:.2f}"
                                         for c, v in pc.items()))
    print(f"\n輸出 {out},{time.time() - t0:.0f} 秒")


if __name__ == "__main__":
    main()
