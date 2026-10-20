"""M6 修正第 2 輪:V + 延續規則(VS)。

預先登記:docs/M6_修正第2輪_預先登記_20261020.md
V:只算在動的物件(修正第 1 輪最好)。VS:最近 S 秒內曾「在動且在手邊」的類別,
之後即使靜止,只要仍在手邊(任一該類物件與手的間距 ≤ d)就延續關聯。

用法:
    python scripts/eval_m6_round2.py --cache-dir results/det_cache/epfl_allcls_armA_s3 \\
        --ann data/epfl_ann/YH2040/2023_09_12_09_50_41/annotations/actions_annotations.xlsx \\
        --side-labels configs/epfl_side_labels/YH2040_2023_09_12_09_50_41.csv \\
        --roles-selected results/m6/dev_yh2040_roles/selected_roles.json \\
        --out-dir results/m6/dev_yh2040_r2
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
import eval_m6_roles as R                                                 # noqa: E402
import eval_m6_round1 as R1                                               # noqa: E402
from m6_assoc.gt import SCORED, build_gt, load_actions                    # noqa: E402
from m6_assoc.roles import auto_side, gate_gt, load_roles, load_side_labels, role_fuse  # noqa: E402

STICKY_S = [1.0, 2.0, 5.0]          # 秒
ARMS = ["V", "VS"]
P_FLOOR = 0.60


def recent_any(x, n):
    """x: [T, C] 布林 → 最近 n 步(含本步)內是否曾為真。"""
    c = np.cumsum(np.vstack([np.zeros((1, x.shape[1]), int), x.astype(int)]), 0)
    lo = np.clip(np.arange(len(x)) + 1 - n, 0, None)
    return (c[np.arange(len(x)) + 1] - c[lo]) > 0


def raw(arm, f, cfg, step):
    v = R1.raw("V", f, cfg)
    if arm == "V":
        return v
    near = f["gap"] <= cfg["d"]
    return v | (near & recent_any(v, int(round(cfg["sticky_s"] / step))))


def arm_cfgs(arm):
    base = R1.arm_cfgs("V")
    if arm == "V":
        return base
    return [dict(c, sticky_s=s) for c in base for s in STICKY_S]


def pick_f1(rows):
    return max(rows, key=lambda r: (r["summary"]["timestep_F1"], r["summary"]["timestep_R"]))


def pick_recall(rows):
    """Recall 優先的操作點:逐步 P ≥ 0.60 中取逐步 R 最大;沒有則 None。"""
    ok = [r for r in rows if r["summary"]["timestep_P"] >= P_FLOOR]
    return max(ok, key=lambda r: (r["summary"]["timestep_R"], r["summary"]["timestep_F1"])) if ok else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--ann", required=True)
    ap.add_argument("--side-labels", required=True)
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
    gt, _ = build_gt(load_actions(args.ann))
    print("讀快取、算單幀特徵…")
    t, step, F = R1.load_r1(args.cache_dir, cams)
    oracle = load_side_labels(args.side_labels, t)
    auto_x = json.loads(Path(args.roles_selected).read_text(encoding="utf-8"))["auto_x"]
    side = auto_side(R.foot_x(args.cache_dir, locator, t), **auto_x)
    gated = {s: gate_gt(gt, t, step, oracle == s) for s in close}

    selected, detail, grid_rows = {}, {}, []
    for arm in ARMS:
        rows = []
        for cfg in (dict(c, role_k=k) for c in arm_cfgs(arm) for k in R1.ROLE_K):
            raws = {c: raw(arm, F[R1.fkey(cfg)][c], cfg, step) for c in cams}
            res = E.score(t, step, E.smooth_all(role_fuse(raws, side, close, cfg["role_k"]), cfg), gt)
            rows.append(dict(cfg=cfg, summary=E.flat(res), result=res))
            grid_rows.append([arm, json.dumps(cfg)] + [round(E.flat(res)[k], 4) for k in E.SUMMARY_KEYS])
        sel = {"best_f1": pick_f1(rows), "recall_first": pick_recall(rows)}
        selected[arm] = {k: (dict(cfg=v["cfg"], summary=v["summary"]) if v else None) for k, v in sel.items()}
        detail[arm] = {k: (v["result"] if v else None) for k, v in sel.items()}

        best = sel["best_f1"]["cfg"]                 # 診斷:同設定的近拍 × 分側真值
        per_cam = {c: E.score(t, step, E.smooth_all(raw(arm, F[R1.fkey(best)][c], best, step), best),
                              gated[s]) for s, cs in close.items() for c in cs}
        selected[arm]["close_gated_same_cfg"] = E.summarize(per_cam)
        print(f"[{arm}] {len(rows)} 組完成")

    with open(out / "grid.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["arm", "cfg"] + E.SUMMARY_KEYS)
        w.writerows(grid_rows)
    (out / "selected.json").write_text(json.dumps(selected, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "metrics.json").write_text(json.dumps(dict(
        meta=dict(cache_dir=args.cache_dir, ann=args.ann, auto_x=auto_x, sticky_s=STICKY_S,
                  p_floor=P_FLOOR, seconds=round(time.time() - t0, 1)),
        selected=selected, detail=detail), ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n結果(角色分工自動定位)")
    for arm in ARMS:
        for key in ("best_f1", "recall_first"):
            v = selected[arm][key]
            if v is None:
                print(f"  {arm:3s} {key:12s} 沒有設定達逐步 P ≥ {P_FLOOR}")
                continue
            s = v["summary"]
            print(f"  {arm:3s} {key:12s} 逐步 P {s['timestep_P']:.3f} R {s['timestep_R']:.3f} "
                  f"F1 {s['timestep_F1']:.3f} | 逐段 P {s['segment_P']:.3f} R {s['segment_R']:.3f}  "
                  f"{json.dumps(v['cfg'])}")
    print("\n逐類(best_f1;逐步 P / R)")
    for arm in ARMS:
        pc = detail[arm]["best_f1"]["per_class"]
        print(f"  {arm:3s} " + "  ".join(f"{c} {v['timestep']['P']:.2f}/{v['timestep']['R']:.2f}"
                                         for c, v in pc.items()))
    print(f"\n輸出 {out},{time.time() - t0:.0f} 秒")


if __name__ == "__main__":
    main()
