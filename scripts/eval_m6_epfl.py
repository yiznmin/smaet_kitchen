"""M6 人—物關聯在 EPFL 上的評估(開發集調參 / 保留集凍結參數)。

輸入:全類別偵測快取(scripts/cache_detections.py --all-classes)+ EPFL 動作標註。
流程:單幀特徵(association.frame_features)→ 判定 → 遲滯平滑 →(融合)→ 對照 GT 計分。

三種方法用同一套平滑網格與同一條選擇規則,公平比較:
  m6           :手的距離 + 上半身退路
  base_present :畫面裡有這類物件就算接觸
  base_person  :物件與人框 IoA(不用手)

選擇規則(預先登記):以 9 個鏡頭的中位數為準,
  在「逐段 Precision ≥ p_floor」的設定裡取逐段 Recall 最大;同分取逐時間步 F1 最大。
  沒有任何設定達 p_floor → 取逐段 F1 最大並標記。

用法:
    # 開發集:掃網格、選操作點
    python scripts/eval_m6_epfl.py --cache-dir results/det_cache/epfl_allcls_armA_s3 \\
        --ann data/epfl_ann/YH2040/2023_09_12_09_50_41/annotations/actions_annotations.xlsx \\
        --out-dir results/m6/dev_yh2040
    # 保留集:用開發集凍結的設定
    python scripts/eval_m6_epfl.py ... --frozen results/m6/dev_yh2040/selected.json
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

from m3.classes import NAME_TO_CLASS                                    # noqa: E402
from m4_track.det_cache import DetCache                                 # noqa: E402
from m6_assoc.association import EVIDENCE, decide, frame_features      # noqa: E402
from m6_assoc.gt import SCORED, build_gt, load_actions, print_report   # noqa: E402
from m6_assoc.metrics import evaluate, intervals_from_mask             # noqa: E402
from m6_assoc.smoothing import fuse_k_of_n, hysteresis                 # noqa: E402

CAMS = ["Aoutput0", "Aoutput1", "Aoutput2", "Aoutput3",
        "Boutput0", "Boutput1", "Boutput2", "Boutput3", "output0"]

# ── 預先登記的網格 ───────────────────────────────────────────────
FIXED = dict(thr_person=0.3, hand_in_person=0.5)
FRAME_GRID = dict(thr_obj=[0.2, 0.3, 0.4], thr_hand=[0.2, 0.3])
M6_GRID = dict(d=[0.0, 0.05, 0.1, 0.2], x_upper=[0.3])
PERSON_GRID = dict(x_person=[0.1, 0.3, 0.5])
SMOOTH_GRID = dict(win=[(1, 1), (3, 2), (5, 3)], k_off=[3, 5, 10])
FUSE_K = [1, 2, 3]
TOL, COVER, P_FLOOR = 0.5, 0.5, 0.60


def product(grid):
    keys = list(grid)
    return [dict(zip(keys, v)) for v in itertools.product(*grid.values())]


def load_features(cache_dir, cams):
    """回傳 (t, step, feats, meta),feats[(thr_obj, thr_hand)][cam] = [T, n_cls, 4]。"""
    class_ids = [NAME_TO_CLASS[c] for c in SCORED]
    feats, t_ref, metas = {}, None, {}
    for cam in cams:
        cache = DetCache(Path(cache_dir) / f"{cam}.npz", expect={"classes": "all", "complete": True})
        m = cache.meta
        metas[cam] = {k: m.get(k) for k in ("weights_id", "cache_thr", "cache_stride",
                                             "resolution", "n_scanned", "max_fid")}
        fps, stride = float(m["video_meta"]["fps"]), int(m["cache_stride"])
        fids = np.arange(0, int(m["max_fid"]) + 1, stride)
        t = fids / fps
        if t_ref is None:
            t_ref, step = t, stride / fps
        n = min(len(t_ref), len(t))
        t_ref = t_ref[:n]
        lo = np.searchsorted(cache._fid, fids, side="left")
        hi = np.searchsorted(cache._fid, fids, side="right")
        for fc in product(FRAME_GRID):
            arr = np.stack([frame_features(cache._xyxy[a:b], cache._conf[a:b], cache._cls[a:b],
                                           class_ids, thr_obj=fc["thr_obj"], thr_hand=fc["thr_hand"],
                                           **FIXED)
                            for a, b in zip(lo, hi)])
            feats.setdefault((fc["thr_obj"], fc["thr_hand"]), {})[cam] = arr
        print(f"  {cam}:{len(fids):,} 步")
    n = len(t_ref)
    for k in feats:
        for cam in feats[k]:
            feats[k][cam] = feats[k][cam][:n]
    return t_ref, step, feats, metas


def raw_masks(method, feat, cfg):
    """feat [T, n_cls, 4] → 單幀布林 [T, n_cls](與證據)。"""
    if method == "m6":
        return decide(feat, cfg["d"], cfg["x_upper"])
    if method == "base_present":
        return feat[..., 0] > 0, None
    return feat[..., 1] >= cfg["x_person"], None


def smooth_all(raw, sm):
    n, k_on = sm["win"]
    return np.stack([hysteresis(raw[:, i], n, k_on, sm["k_off"]) for i in range(raw.shape[1])], 1)


def score(t, step, pred, gt):
    return evaluate(t, {c: pred[:, i] for i, c in enumerate(SCORED)}, gt, step, TOL, COVER)


SUMMARY_KEYS = ["segment_P", "segment_R", "segment_F1", "timestep_P", "timestep_R", "timestep_F1",
                "f1@0.1", "f1@0.25", "f1@0.5"]


def flat(res):
    """單一結果的 pooled 指標攤平成 SUMMARY_KEYS。"""
    p = res["pooled"]
    out = {f"{a}_{b}": p[a][b] for a in ("segment", "timestep") for b in ("P", "R", "F1")}
    out.update({f"f1@{tau}": p["f1_at"][tau]["F1"] for tau in p["f1_at"]})
    return out


def summarize(per_cam):
    """9 鏡頭 pooled 指標的中位數。"""
    flats = [flat(r) for r in per_cam.values()]
    return {k: float(np.median([f[k] for f in flats])) for k in SUMMARY_KEYS}


def top_cams(per_cam, n=3):
    """事後挑的參考值:依逐段 F1 取最好的 n 台,報平均(⚠ 事後挑選,只當參考)。"""
    ranked = sorted(per_cam, key=lambda c: flat(per_cam[c])["segment_F1"], reverse=True)[:n]
    return dict(cams=ranked, **{k: float(np.mean([flat(per_cam[c])[k] for c in ranked]))
                                for k in SUMMARY_KEYS})


def select(rows):
    ok = [r for r in rows if r["summary"]["segment_P"] >= P_FLOOR]
    if ok:
        best = max(ok, key=lambda r: (r["summary"]["segment_R"], r["summary"]["timestep_F1"]))
        return best, True
    return max(rows, key=lambda r: r["summary"]["segment_F1"]), False


def configs_for(method):
    grids = {"m6": M6_GRID, "base_present": {}, "base_person": PERSON_GRID}[method]
    frame = product(FRAME_GRID) if method != "base_present" else product(dict(thr_obj=FRAME_GRID["thr_obj"]))
    out = []
    for fc in frame:
        for mc in product(grids):
            for sm in product(SMOOTH_GRID):
                out.append(dict(fc, **mc, **sm))
    return out


def feat_key(cfg):
    return (cfg["thr_obj"], cfg.get("thr_hand", FRAME_GRID["thr_hand"][0]))


def run_single_view(method, cfgs, t, step, feats, gt, cams):
    rows = []
    for cfg in cfgs:
        per_cam = {}
        for cam in cams:
            raw, _ = raw_masks(method, feats[feat_key(cfg)][cam], cfg)
            per_cam[cam] = score(t, step, smooth_all(raw, cfg), gt)
        rows.append(dict(method=method, mode="single", cfg=cfg, summary=summarize(per_cam),
                         per_cam=per_cam))
    return rows


def run_fused(method, cfgs, t, step, feats, gt, cams):
    rows = []
    for cfg in cfgs:
        raws = np.stack([raw_masks(method, feats[feat_key(cfg)][c], cfg)[0] for c in cams])
        for k in FUSE_K:
            fused = smooth_all(fuse_k_of_n(raws, k), cfg)
            res = score(t, step, fused, gt)
            rows.append(dict(method=method, mode="fused", cfg=dict(cfg, fuse_k=k),
                             summary=flat(res),
                             result=res))
    return rows


def write_associations(path, t, step, feats, cfg, cams):
    """選定的 M6 設定 → 每鏡頭的關聯區間(含每段的證據組成)。"""
    with open(path, "w", encoding="utf-8") as f:
        for cam in cams:
            raw, ev = raw_masks("m6", feats[feat_key(cfg)][cam], cfg)
            pred = smooth_all(raw, cfg)
            for i, cls in enumerate(SCORED):
                for s, e in intervals_from_mask(t, pred[:, i], step):
                    sel = (t >= s) & (t < e)
                    kinds = ev[sel, i]
                    f.write(json.dumps(dict(
                        cam=cam, obj_class=cls, t_start=round(s, 2), t_end=round(e, 2),
                        evidence={EVIDENCE[k]: int((kinds == k).sum()) for k in (1, 2)},
                        n_steps=int(sel.sum())), ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--ann", required=True, help="actions_annotations.xlsx")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--cams", nargs="+", default=CAMS)
    ap.add_argument("--frozen", default=None, help="開發集輸出的 selected.json;給了就不掃網格")
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    gt, rep = build_gt(load_actions(args.ann))
    print_report(gt, rep)
    print("讀快取、算單幀特徵…")
    t, step, feats, metas = load_features(args.cache_dir, args.cams)

    methods = ["m6", "base_present", "base_person"]
    if args.frozen:
        frozen = json.loads(Path(args.frozen).read_text(encoding="utf-8"))
        cfgs = {m: [frozen["single"][m]["cfg"]] for m in methods}
        fcfgs = {m: [{k: v for k, v in frozen["fused"][m]["cfg"].items() if k != "fuse_k"}]
                 for m in methods}
    else:
        cfgs = {m: configs_for(m) for m in methods}
        fcfgs = cfgs

    single, fused, selected = {}, {}, {"single": {}, "fused": {}}
    for m in methods:
        print(f"[{m}] 單視角 {len(cfgs[m])} 組…")
        single[m] = run_single_view(m, cfgs[m], t, step, feats, gt, args.cams)
        best, ok = select(single[m])
        selected["single"][m] = dict(cfg=best["cfg"], summary=best["summary"], meets_p_floor=ok)
        print(f"[{m}] 融合…")
        fused[m] = run_fused(m, fcfgs[m], t, step, feats, gt, args.cams)
        if args.frozen:
            k = frozen["fused"][m]["cfg"]["fuse_k"]
            fused[m] = [r for r in fused[m] if r["cfg"]["fuse_k"] == k]
        best_f, ok_f = select(fused[m])
        selected["fused"][m] = dict(cfg=best_f["cfg"], summary=best_f["summary"], meets_p_floor=ok_f)

    # 網格總表
    with open(out / "grid.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["method", "mode", "cfg"] + SUMMARY_KEYS)
        for rows in list(single.values()) + list(fused.values()):
            for r in rows:
                s = r["summary"]
                w.writerow([r["method"], r["mode"], json.dumps(r["cfg"])] +
                           [round(s[k], 4) for k in SUMMARY_KEYS])

    # 選定設定的逐鏡頭、逐類別明細
    detail = {"single": {}, "fused": {}}
    for m in methods:
        r = next(x for x in single[m] if x["cfg"] == selected["single"][m]["cfg"])
        detail["single"][m] = r["per_cam"]
        selected["single"][m]["top3_reference"] = top_cams(r["per_cam"])
        r = next(x for x in fused[m] if x["cfg"] == selected["fused"][m]["cfg"])
        detail["fused"][m] = r["result"]
    sel_m6 = selected["single"]["m6"]["cfg"]
    write_associations(out / "associations.jsonl", t, step, feats, sel_m6, args.cams)

    meta = dict(cache_dir=args.cache_dir, ann=args.ann, cams=args.cams, cache_meta=metas,
                frozen=args.frozen, fixed=FIXED, tol=TOL, cover=COVER, p_floor=P_FLOOR,
                gt_report={k: (dict(v) if hasattr(v, "items") else v) for k, v in rep.items()},
                gt_counts={c: dict(segments=len(gt[c]["segments"]), union=len(gt[c]["union"]),
                                   seconds=round(sum(e - s for s, e in gt[c]["union"]), 1))
                           for c in SCORED},
                seconds=round(time.time() - t0, 1))
    (out / "selected.json").write_text(json.dumps(selected, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "metrics.json").write_text(json.dumps(dict(meta=meta, selected=selected, detail=detail),
                                                 ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n選定操作點(單視角 = 9 鏡頭中位數;融合 = 單一結果)")
    for mode in ("single", "fused"):
        for m in methods:
            s, c = selected[mode][m]["summary"], selected[mode][m]["cfg"]
            flag = "" if selected[mode][m]["meets_p_floor"] else "  ⚠ 未達 P 下限"
            print(f"  {mode:6s} {m:13s} 逐段 P {s['segment_P']:.3f} R {s['segment_R']:.3f}  "
                  f"逐步 P {s['timestep_P']:.3f} R {s['timestep_R']:.3f} F1 {s['timestep_F1']:.3f}  "
                  f"F1@0.25 {s['f1@0.25']:.3f}{flag}  {json.dumps(c)}")
    print("\n單視角逐鏡頭(選定設定;逐段 P / R)")
    for m in methods:
        pc = detail["single"][m]
        print(f"  {m:13s} " + "  ".join(f"{c}:{flat(pc[c])['segment_P']:.2f}/{flat(pc[c])['segment_R']:.2f}"
                                        for c in args.cams))
        t3 = selected["single"][m]["top3_reference"]
        print(f"  {'':13s} 事後最好 3 台 {t3['cams']}:逐段 P {t3['segment_P']:.3f} R {t3['segment_R']:.3f}")
    print(f"\n輸出 {out},{time.time() - t0:.0f} 秒")


if __name__ == "__main__":
    main()
