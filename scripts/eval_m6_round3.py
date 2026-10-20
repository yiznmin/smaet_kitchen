"""M6 修正第 3 輪:「在動」改用同一個物件的前後對應(VT)。

預先登記:docs/M6_修正第3輪_預先登記_20261020.md
V :修正第 1 輪,motion IoU = 與 0.5 秒前「任一個」同類框的最大 IoU(對照組)
VT:逐類別貪婪 IoU 追蹤(association.track_motion),motion IoU = 與「自己」0.5 秒前的框比

用法:
    python scripts/eval_m6_round3.py --cache-dir results/det_cache/epfl_allcls_armA_s3 \\
        --ann data/epfl_ann/YH2040/2023_09_12_09_50_41/annotations/actions_annotations.xlsx \\
        --roles-selected results/m6/dev_yh2040_roles/selected_roles.json \\
        --out-dir results/m6/dev_yh2040_r3
"""
import argparse
import csv
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import eval_m6_epfl as E                                                  # noqa: E402
import eval_m6_roles as R                                                 # noqa: E402
import eval_m6_round1 as R1                                               # noqa: E402
import eval_m6_round2 as R2                                               # noqa: E402
from m3.classes import NAME_TO_CLASS                                      # noqa: E402
from m4_track.det_cache import DetCache                                   # noqa: E402
from m6_assoc.association import _above, frame_features, frame_features_r1, track_motion  # noqa: E402
from m6_assoc.gt import SCORED, build_gt, load_actions                    # noqa: E402
from m6_assoc.metrics import mask_from_intervals                          # noqa: E402
from m6_assoc.roles import auto_side, load_roles, role_fuse               # noqa: E402

ARMS = ["V", "VT"]
CONTAINER_NOUNS = ("Bowl", "Salad_bowl", "Plate", "Box", "Bottle")


def load_tracked(cache_dir, cams):
    """同 R1.load_r1 的 F,但 mv_gap 由追蹤版 motion 算。回傳 t, step, F。"""
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
        for to in R1.FRAME_GRID["thr_obj"]:
            frames = []
            for a, b in zip(lo, hi):
                keep = np.isin(c._cls[a:b], class_ids) & _above(c._conf[a:b], to)
                frames.append((c._xyxy[a:b][keep].astype(np.float64), c._cls[a:b][keep]))
            motion = track_motion(frames, class_ids, lag=R1.LAG)
            for th in R1.FRAME_GRID["thr_hand"]:
                rows = []
                for i, (a, b) in enumerate(zip(lo, hi)):
                    pa, pb = (lo[i - R1.LAG], hi[i - R1.LAG]) if i >= R1.LAG else (a, a)
                    r = frame_features_r1(
                        c._xyxy[a:b], c._conf[a:b], c._cls[a:b],
                        c._xyxy[pa:pb], c._conf[pa:pb], c._cls[pa:pb], class_ids,
                        thr_obj=to, thr_hand=th, iou_moves=R1.IOU_MOVES, motion=motion[i], **R1.FIXED)
                    base = frame_features(c._xyxy[a:b], c._conf[a:b], c._cls[a:b], class_ids,
                                          thr_obj=to, thr_hand=th, **R1.FIXED)
                    r["present"], r["person_ioa"] = base[:, 0], base[:, 1]
                    rows.append(r)
                F.setdefault((to, th), {})[cam] = {k: np.stack([r[k] for r in rows]) for k in rows[0]}
        print(f"  {cam}:{len(fids):,} 步")
    return t_ref, step, F


def container_breakdown(t, F, cams, close, side, gt, acts, cfg):
    """容器真值時間步上,該側近拍鏡頭的情況分佈(與修正第 2 輪 §6.2 同定義)。"""
    ci = SCORED.index("容器")
    q = R1.IOU_MOVES.index(cfg["iou_move"])
    f = {c: F[R1.fkey(cfg)][c] for c in cams}
    noun_at = [None] * len(t)
    for a in acts:
        if a["noun"] in CONTAINER_NOUNS and a["verb"] not in ("Wait", "Squat", "Read"):
            for i in np.flatnonzero((t >= a["start"]) & (t < a["end"])):
                noun_at[i] = f"{a['verb']} {a['noun']}"
    g = mask_from_intervals(t, gt["容器"]["union"])
    cat, carry = Counter(), Counter()
    for i in np.flatnonzero(g):
        cs = close[side[i]] if side[i] in ("A", "B") else cams
        mv = any(f[c]["mv_gap"][i, q, ci] <= cfg["d"] for c in cs)
        near = any(f[c]["gap"][i, ci] <= cfg["d"] for c in cs)
        pres = any(f[c]["present"][i, ci] > 0 for c in cs)
        k = "抓到" if mv else ("在手邊但判為靜止" if near else ("有偵測到但不在手邊" if pres else "沒偵測到"))
        cat[k] += 1
        if noun_at[i] == "Carry Bottle":
            carry[k] += 1
    n = int(g.sum())
    return dict(n=n, **{k: round(v / n, 4) for k, v in cat.items()},
                carry_bottle={k: round(v / max(sum(carry.values()), 1), 4) for k, v in carry.items()})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--ann", required=True)
    ap.add_argument("--roles-selected", required=True)
    ap.add_argument("--roles", default=str(ROOT / "configs" / "camera_roles.epfl.json"))
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    roles = load_roles(args.roles)
    close, locator = roles["close"], roles["locator"]
    cams = close["A"] + close["B"]
    acts = load_actions(args.ann)
    gt, _ = build_gt(acts)
    print("讀快取:V(修正第 1 輪特徵)…")
    t, step, FV = R1.load_r1(args.cache_dir, cams)
    print("讀快取:VT(追蹤版 motion)…")
    _, _, FT = load_tracked(args.cache_dir, cams)
    auto_x = json.loads(Path(args.roles_selected).read_text(encoding="utf-8"))["auto_x"]
    side = auto_side(R.foot_x(args.cache_dir, locator, t), **auto_x)

    selected, detail, grid_rows = {}, {}, []
    for arm, F in (("V", FV), ("VT", FT)):
        rows = []
        for cfg in (dict(c, role_k=k) for c in R1.arm_cfgs("V") for k in R1.ROLE_K):
            raws = {c: R1.raw("V", F[R1.fkey(cfg)][c], cfg) for c in cams}
            res = E.score(t, step, E.smooth_all(role_fuse(raws, side, close, cfg["role_k"]), cfg), gt)
            rows.append(dict(cfg=cfg, summary=E.flat(res), result=res))
            grid_rows.append([arm, json.dumps(cfg)] + [round(E.flat(res)[k], 4) for k in E.SUMMARY_KEYS])
        sel = {"best_f1": R2.pick_f1(rows), "recall_first": R2.pick_recall(rows)}
        selected[arm] = {k: (dict(cfg=v["cfg"], summary=v["summary"]) if v else None) for k, v in sel.items()}
        detail[arm] = {k: (v["result"] if v else None) for k, v in sel.items()}
        selected[arm]["container_breakdown"] = container_breakdown(
            t, F, cams, close, side, gt, acts, sel["best_f1"]["cfg"])
        print(f"[{arm}] {len(rows)} 組完成")

    with open(out / "grid.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["arm", "cfg"] + E.SUMMARY_KEYS)
        w.writerows(grid_rows)
    (out / "selected.json").write_text(json.dumps(selected, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "metrics.json").write_text(json.dumps(dict(
        meta=dict(cache_dir=args.cache_dir, ann=args.ann, auto_x=auto_x, lag_steps=R1.LAG,
                  link_iou=0.3, max_gap=3, seconds=round(time.time() - t0, 1)),
        selected=selected, detail=detail), ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n結果(角色分工自動定位)")
    for arm in ARMS:
        for key in ("best_f1", "recall_first"):
            v = selected[arm][key]
            if v is None:
                print(f"  {arm:3s} {key:12s} 沒有設定達逐步 P ≥ {R2.P_FLOOR}")
                continue
            s = v["summary"]
            print(f"  {arm:3s} {key:12s} 逐步 P {s['timestep_P']:.3f} R {s['timestep_R']:.3f} "
                  f"F1 {s['timestep_F1']:.3f} | 逐段 P {s['segment_P']:.3f} R {s['segment_R']:.3f}  "
                  f"{json.dumps(v['cfg'])}")
        print(f"      容器拆解 {selected[arm]['container_breakdown']}")
    print("\n逐類(best_f1;逐步 P / R)")
    for arm in ARMS:
        pc = detail[arm]["best_f1"]["per_class"]
        print(f"  {arm:3s} " + "  ".join(f"{c} {v['timestep']['P']:.2f}/{v['timestep']['R']:.2f}"
                                         for c, v in pc.items()))
    print(f"\n輸出 {out},{time.time() - t0:.0f} 秒")


if __name__ == "__main__":
    main()
