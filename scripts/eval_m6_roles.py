"""M6 依鏡頭角色分工的評估(補充登記 §8.1;觀察紀錄 docs/EPFL_鏡頭配置觀察_20261019.md)。

兩項分析,三種方法(m6 / base_present / base_person)用同一網格、同一選擇規則:

  1. 近拍鏡頭 × 分側真值:A 近拍只對「人在 A 側」的真值計分,B 近拍只對 B 側。
     → 鏡頭看得到時,規則本身好不好(與「看不到」分開)
  2. 角色分工系統:依位置挑該側近拍組、組內至少 k 台成立 → 平滑 → 對完整真值計分。
     位置來源兩種:人工標記(上限)、自動(output0 腳點 x)

用法:
    python scripts/eval_m6_roles.py --cache-dir results/det_cache/epfl_allcls_armA_s3 \\
        --ann data/epfl_ann/YH2040/2023_09_12_09_50_41/annotations/actions_annotations.xlsx \\
        --side-labels configs/epfl_side_labels/YH2040_2023_09_12_09_50_41.csv \\
        --out-dir results/m6/dev_yh2040_roles
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import eval_m6_epfl as E                                                 # noqa: E402
from m3.classes import NAME_TO_CLASS                                     # noqa: E402
from m4_track.det_cache import DetCache                                  # noqa: E402
from m6_assoc.gt import build_gt, load_actions                           # noqa: E402
from m6_assoc.roles import (auto_side, gate_gt, load_roles,              # noqa: E402
                            load_side_labels, role_fuse)

ROLE_K = [1, 2]


def foot_x(cache_dir, cam, t, fps=30.0, thr=0.3):
    """定位鏡頭上每個時間步信心最高的人框的腳點 x;沒有人 → NaN。"""
    c = DetCache(Path(cache_dir) / f"{cam}.npz", expect={"classes": "all", "complete": True})
    out = np.full(len(t), np.nan)
    person = NAME_TO_CLASS["人"]
    for i, fid in enumerate(np.round(t * fps).astype(int)):
        lo = np.searchsorted(c._fid, fid, side="left")
        hi = np.searchsorted(c._fid, fid, side="right")
        cls, conf, xy = c._cls[lo:hi], c._conf[lo:hi], c._xyxy[lo:hi]
        keep = (cls == person) & (conf > np.float32(thr).astype(conf.dtype))
        if keep.any():
            j = np.flatnonzero(keep)[conf[keep].argmax()]
            out[i] = (xy[j, 0] + xy[j, 2]) / 2
    return out


def pick_auto_thresholds(fx, oracle):
    """在開發集上取門檻:A 側腳點的第 95 百分位、B 側的第 5 百分位(中間留給 M)。"""
    a = fx[(oracle == "A") & ~np.isnan(fx)]
    b = fx[(oracle == "B") & ~np.isnan(fx)]
    x_a, x_b = float(np.percentile(a, 95)), float(np.percentile(b, 5))
    if x_a >= x_b:                       # 兩側分佈重疊:取中點,不留 M
        x_a = x_b = (x_a + x_b) / 2
    return dict(x_a=round(x_a, 1), x_b=round(x_b, 1))


def run_close_gated(method, cfgs, t, step, feats, gt, side, close):
    rows = []
    for cfg in cfgs:
        per_cam = {}
        for s, cams in close.items():
            g = gate_gt(gt, t, step, side == s)
            for cam in cams:
                raw, _ = E.raw_masks(method, feats[E.feat_key(cfg)][cam], cfg)
                per_cam[cam] = E.score(t, step, E.smooth_all(raw, cfg), g)
        rows.append(dict(cfg=cfg, summary=E.summarize(per_cam), per_cam=per_cam))
    return rows


def run_role_system(method, cfgs, t, step, feats, gt, side, close):
    rows = []
    cams = close["A"] + close["B"]
    for cfg in cfgs:
        raws = {c: E.raw_masks(method, feats[E.feat_key(cfg)][c], cfg)[0] for c in cams}
        for k in ROLE_K:
            res = E.score(t, step, E.smooth_all(role_fuse(raws, side, close, k), cfg), gt)
            rows.append(dict(cfg=dict(cfg, role_k=k), summary=E.flat(res), result=res))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--ann", required=True)
    ap.add_argument("--side-labels", required=True)
    ap.add_argument("--roles", default=str(ROOT / "configs" / "camera_roles.epfl.json"))
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--frozen", default=None, help="開發集輸出的 selected_roles.json")
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    roles = load_roles(args.roles)
    close, locator = roles["close"], roles["locator"]
    cams = close["A"] + close["B"]
    gt, _ = build_gt(load_actions(args.ann))
    print("讀快取、算單幀特徵…")
    t, step, feats, _ = E.load_features(args.cache_dir, cams)

    oracle = load_side_labels(args.side_labels, t)
    fx = foot_x(args.cache_dir, locator, t)
    frozen = json.loads(Path(args.frozen).read_text(encoding="utf-8")) if args.frozen else None
    thr = frozen["auto_x"] if frozen else pick_auto_thresholds(fx, oracle)
    auto = auto_side(fx, **thr)
    agree = float((auto == oracle).mean())
    print(f"自動定位門檻 {thr};與人工標記一致 {agree:.3f};"
          f"定位鏡頭沒偵測到人 {np.isnan(fx).mean():.3f}")

    methods = ["m6", "base_present", "base_person"]
    selected, detail = {"auto_x": thr, "auto_agree": agree}, {}
    for m in methods:
        cfgs = ([frozen[m]["close_gated"]["cfg"]] if frozen else E.configs_for(m))
        rows = run_close_gated(m, cfgs, t, step, feats, gt, oracle, close)
        best, ok = E.select(rows)
        selected.setdefault(m, {})["close_gated"] = dict(cfg=best["cfg"], summary=best["summary"],
                                                         meets_p_floor=ok)
        detail.setdefault(m, {})["close_gated"] = best["per_cam"]
        for src, side in (("oracle", oracle), ("auto", auto)):
            if frozen:
                fc = dict(frozen[m][f"role_{src}"]["cfg"])
                k = fc.pop("role_k")
                rows = [r for r in run_role_system(m, [fc], t, step, feats, gt, side, close)
                        if r["cfg"]["role_k"] == k]
            else:
                rows = run_role_system(m, cfgs, t, step, feats, gt, side, close)
            best, ok = E.select(rows)
            selected[m][f"role_{src}"] = dict(cfg=best["cfg"], summary=best["summary"], meets_p_floor=ok)
            detail[m][f"role_{src}"] = best["result"]
        print(f"[{m}] 完成")

    (out / "selected_roles.json").write_text(json.dumps(selected, ensure_ascii=False, indent=2),
                                             encoding="utf-8")
    (out / "metrics_roles.json").write_text(json.dumps(dict(
        meta=dict(cache_dir=args.cache_dir, ann=args.ann, side_labels=args.side_labels,
                  roles=roles, frozen=args.frozen, role_k=ROLE_K,
                  seconds=round(time.time() - t0, 1)),
        selected=selected, detail=detail), ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n結果(近拍 × 分側真值 = 5 台近拍中位數;角色分工 = 單一系統結果)")
    for m in methods:
        for key in ("close_gated", "role_oracle", "role_auto"):
            s, c = selected[m][key]["summary"], selected[m][key]["cfg"]
            flag = "" if selected[m][key]["meets_p_floor"] else "  ⚠ 未達 P 下限"
            print(f"  {m:13s} {key:12s} 逐段 P {s['segment_P']:.3f} R {s['segment_R']:.3f}  "
                  f"逐步 F1 {s['timestep_F1']:.3f}  F1@0.25 {s['f1@0.25']:.3f}{flag}  {json.dumps(c)}")
    print("\n近拍逐鏡頭(m6,分側真值;逐段 P / R)")
    for cam, r in detail["m6"]["close_gated"].items():
        f = E.flat(r)
        print(f"  {cam}: {f['segment_P']:.2f} / {f['segment_R']:.2f}")
    print(f"\n輸出 {out},{time.time() - t0:.0f} 秒")


if __name__ == "__main__":
    main()
