"""算逐片段獨立推論的指標(預先登記 §5)。自己不實作判定,全部走 `common.clip_metrics`。

輸出:
  results/clips/metrics/<clip_id>/s<stride>/per_frame.csv   ← 主要產物,所有率由它算出
  results/clips/metrics/<clip_id>/s<stride>/clip_metrics.json
  results/clips/agg/per_clip.csv / by_level_stride.{csv,md} / by_gap_band.csv /
                    by_handoff_type.csv / stride_delta.csv

⚠ 彙總一律**彙集逐幀列**再算一次(分母才對),另報逐片段中位數。
⚠ 排他率在獨佔片段上回 None → 報表印「不可量測」,不可印 100%。
"""
import argparse
import csv
import json
import statistics as st
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from common.clip_metrics import UNMEASURABLE, clip_metrics, wilson   # noqa: E402
from eval_m4m5_chirla import load_gt                                 # noqa: E402


def load_tracks(run_dir):
    """{(鏡頭, fid): [(track_id, bbox 或 None)]} 與 {(鏡頭, track_id): chef_id}。

    ⚠ 空框(x1 為空)代表這一幀沒配到偵測、用 Kalman 預測 —— 保留下來但標成 None,
      它不參與配對,但要算進「空轉」。
    """
    tracks, chefs = defaultdict(list), {}
    with open(Path(run_dir) / "tracks.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            cam, fid, tid = r["camera_id"], int(r["video_fid"]), int(r["track_id"])
            box = (None if r["x1"] == ""
                   else tuple(float(r[k]) for k in ("x1", "y1", "x2", "y2")))
            tracks[(cam, fid)].append((tid, box))
            if r.get("chef_id"):
                chefs[(cam, tid)] = int(r["chef_id"])
    return tracks, chefs


def annotate(rows, m):
    """把三個率的逐格判定寫回 per_frame.csv —— 彙總時直接彙集這些旗標。"""
    for r in rows:
        b = r["gt_present"] and r["matched"] and r["chef_id"] != ""
        r["in_B"] = int(bool(b))
        r["is_continuity"] = int(b and r["chef_id"] == m["c0"]) if b else ""
        r["is_majority"] = int(b and r["chef_id"] == m["c_star"]) if b else ""
        r["is_exclusive"] = ("" if (not b or m["p_exclusive"] is None)
                             else int(not r["others_same_chef"]))
    return rows


def pooled(rows, num_key, den_key=None):
    n = sum(1 for r in rows if r[den_key]) if den_key else len(rows)
    k = sum(1 for r in rows if r[den_key] and r[num_key]) if den_key else sum(
        1 for r in rows if r[num_key])
    return (round(k / n, 4) if n else None), k, n


def fmt(v, ok=True):
    return UNMEASURABLE if (v is None or not ok) else f"{v * 100:.2f}%"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--manifest", default="results/clips/clip_manifest.json")
    ap.add_argument("--run-root", default="results/clips/runs")
    ap.add_argument("--strides", nargs="+", type=int, default=[5, 1])
    ap.add_argument("--out-dir", default="results/clips")
    args = ap.parse_args()

    man = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    gts, per_clip, all_rows, missing = {}, [], [], []

    for clip in man["clips"]:
        if clip["seq"] not in gts:
            gts[clip["seq"]] = load_gt(args.root, clip["seq"])
        for stride in args.strides:
            run_dir = Path(args.run_root) / clip["clip_id"] / f"s{stride}"
            if not (run_dir / "tracks.csv").exists():
                missing.append(f"{clip['clip_id']} s{stride}")
                continue
            tracks, chefs = load_tracks(run_dir)
            rows, m = clip_metrics(clip, stride, gts[clip["seq"]], tracks, chefs)
            rows = annotate(rows, m)
            out = Path(args.out_dir) / "metrics" / clip["clip_id"] / f"s{stride}"
            out.mkdir(parents=True, exist_ok=True)
            with open(out / "per_frame.csv", "w", encoding="utf-8-sig", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows(rows)
            (out / "clip_metrics.json").write_text(
                json.dumps(m, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
            per_clip.append(m)
            all_rows += rows

    if missing:
        print(f"⚠ 缺 {len(missing)} 份執行結果:", missing[:5])

    agg_dir = Path(args.out_dir) / "agg"
    agg_dir.mkdir(parents=True, exist_ok=True)

    # ── 逐片段表 ──
    scalar = [k for k in per_clip[0] if not isinstance(per_clip[0][k], (list, dict))]
    keys = sorted({k for m in per_clip for k in m if k in scalar or
                   k not in ("chef_switch_detail",)})
    keys = [k for k in keys if not isinstance(per_clip[0].get(k), (list, dict))]
    with open(agg_dir / "per_clip.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for m in per_clip:
            w.writerow({k: m.get(k) for k in keys})

    # ── 依規則 × stride 彙總(彙集逐幀列) ──
    groups = defaultdict(list)
    rule_of = {c["clip_id"]: c["rule"] for c in man["clips"]}
    excl_of = {c["clip_id"]: c["exclusive"] for c in man["clips"]}
    for r in all_rows:
        groups[(rule_of[r["clip_id"]], r["stride"])].append(r)

    lines = ["# 逐片段獨立推論:彙總", "",
             "⚠ **每段從零開始、只給該段指名的鏡頭** —— 與交付版(7 台、整段、共用身份狀態)"
             "問的是不同問題,**數字不可與 `results/levels/B*/` 並列**。", "",
             "⚠ stride 1 的記憶只有 stride 5 的五分之一(TTL 100 秒 → 20 秒、"
             "跟丟緩衝 5 秒 → 1 秒,兩者都以迴圈計數)。", ""]
    agg_rows = []
    for (rule, stride), rows in sorted(groups.items()):
        clips_here = [m for m in per_clip
                      if rule_of[m["clip_id"]] == rule and m["stride"] == stride]
        rec, k_rec, n_T = pooled(rows, "matched", "gt_present")
        ious = [r["iou"] for r in rows if r["gt_present"]]
        cont, k_c, n_B = pooled(rows, "is_continuity", "in_B")
        maj, k_m, _ = pooled(rows, "is_majority", "in_B")
        non_excl = [r for r in rows if not excl_of[r["clip_id"]]]
        exc, k_e, n_e = pooled(non_excl, "is_exclusive", "in_B") if non_excl else (None, 0, 0)
        n_box = sum(r["n_tracks"] for r in rows)
        n_gh = sum(r["n_ghost"] for r in rows)
        row = dict(rule=rule, stride=stride, n_clips=len(clips_here),
                   n_cells=len(rows), n_target_cells=n_T, n_bound_cells=n_B,
                   recall_target=rec, recall_ci=wilson(k_rec, n_T),
                   mean_iou=round(sum(ious) / len(ious), 4) if ious else None,
                   median_iou=round(st.median(ious), 4) if ious else None,
                   bound_rate=round(n_B / n_T, 4) if n_T else None,
                   p_continuity=cont, p_continuity_ci=wilson(k_c, n_B),
                   p_majority=maj, p_majority_ci=wilson(k_m, n_B),
                   p_exclusive=exc, n_exclusive_cells=n_e,
                   chef_switches=sum(m["chef_switches"] for m in clips_here),
                   n_chef_ids_median=st.median([m["n_distinct_chef_ids"] for m in clips_here]),
                   ghost_box_rate=round(n_gh / n_box, 4) if n_box else None,
                   n_boxes=n_box,
                   median_clip_recall=round(st.median(
                       [m["recall_target"] for m in clips_here if m["recall_target"] is not None]), 4)
                   if clips_here else None)
        agg_rows.append(row)
        lines += [f"## {rule} · stride {stride}"
                  f"(片段 {row['n_clips']} 段,取樣格 {row['n_cells']},"
                  f"目標在場 {n_T},有綁定 {n_B})", "",
                  "| 指標 | 值 |", "|---|---|",
                  f"| 目標召回率 | {fmt(rec)} {row['recall_ci']} |",
                  f"| IoU 平均 / 中位 | {row['mean_iou']} / {row['median_iou']} |",
                  f"| 綁定率 | {fmt(row['bound_rate'])} |",
                  f"| **連續率** | {fmt(cont)} {row['p_continuity_ci']} |",
                  f"| **多數率** | {fmt(maj)} {row['p_majority_ci']} |",
                  f"| **排他率** | {fmt(exc, n_e > 0)} |",
                  f"| 編號切換次數 | {row['chef_switches']} |",
                  f"| 每段編號數(中位) | {row['n_chef_ids_median']} |",
                  f"| 誤偵框比例 | {fmt(row['ghost_box_rate'])}(框 {n_box}) |", ""]

    with open(agg_dir / "by_level_stride.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(agg_rows[0]))
        w.writeheader()
        w.writerows(agg_rows)

    # ── L2:接回率(依間隔帶) ──
    l2 = [m for m in per_clip if m["rule"] == "L2"]
    l2_rows = []
    for stride in args.strides:
        for band in ("within_buffer", "beyond_buffer"):
            g = [m for m in l2 if m["stride"] == stride and m.get("gap_band") == band]
            if not g:
                continue
            for k in (5, 1):
                np_ = sum(m[f"l2_n_pairs_k{k}"] for m in g)
                ok = sum(round((m[f"l2_p_reacquire_k{k}"] or 0) * m[f"l2_n_pairs_k{k}"])
                         for m in g)
                l2_rows.append(dict(stride=stride, gap_band=band, k=k, n_clips=len(g),
                                    n_pairs=np_, n_pairs_dropped=sum(
                                        m[f"l2_n_pairs_dropped_k{k}"] for m in g),
                                    p_reacquire=round(ok / np_, 4) if np_ else None,
                                    ci=wilson(ok, np_)))
    if l2_rows:
        with open(agg_dir / "by_gap_band.csv", "w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(l2_rows[0]))
            w.writeheader()
            w.writerows(l2_rows)
        lines += ["## L2:離開再回來有沒有接回同一個編號", "",
                  "| stride | 間隔帶 | k | 段 | 可用間隔對 | 排除 | 接回率 |",
                  "|---|---|---|---|---|---|---|"]
        for r in l2_rows:
            lines.append(f"| {r['stride']} | {r['gap_band']} | {r['k']} | {r['n_clips']} | "
                         f"{r['n_pairs']} | {r['n_pairs_dropped']} | "
                         f"{fmt(r['p_reacquire'])} {r['ci']} |")
        lines.append("")

    # ── L3:跨鏡頭一致(L3S / L3T 分開) ──
    l3_rows = []
    for stride in args.strides:
        for rule in ("L3S", "L3T"):
            g = [m for m in per_clip if m["rule"] == rule and m["stride"] == stride]
            if not g:
                continue
            npair = sum(m["l3_n_pairs_usable"] for m in g)
            ok = sum(round((m["l3_p_handoff"] or 0) * m["l3_n_pairs_usable"]) for m in g)
            nsim = sum(m["l3_n_simul_frames"] for m in g)
            sok = sum(round((m["l3_p_simul_agree"] or 0) * m["l3_n_simul_frames"]) for m in g)
            l3_rows.append(dict(stride=stride, rule=rule, n_clips=len(g),
                                n_pairs=npair,
                                n_pairs_dropped=sum(m["l3_n_pairs_dropped"] for m in g),
                                p_handoff=round(ok / npair, 4) if npair else None,
                                handoff_ci=wilson(ok, npair),
                                n_simul_frames=nsim,
                                p_simul_agree=round(sok / nsim, 4) if nsim else None,
                                simul_ci=wilson(sok, nsim)))
    if l3_rows:
        with open(agg_dir / "by_handoff_type.csv", "w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(l3_rows[0]))
            w.writeheader()
            w.writerows(l3_rows)
        lines += ["## L3:跨鏡頭有沒有給同一個編號(L3S / L3T 永遠分開報)", "",
                  "| stride | 類型 | 段 | 可用鏡頭對 | 排除 | 一致率 | 同時出現幀 | 同時一致率 |",
                  "|---|---|---|---|---|---|---|---|"]
        for r in l3_rows:
            lines.append(f"| {r['stride']} | {r['rule']} | {r['n_clips']} | {r['n_pairs']} | "
                         f"{r['n_pairs_dropped']} | {fmt(r['p_handoff'])} {r['handoff_ci']} | "
                         f"{r['n_simul_frames']} | {fmt(r['p_simul_agree'])} {r['simul_ci']} |")
        lines.append("")

    # ── stride 差異 ──
    if len(args.strides) > 1:
        a, b = args.strides[0], args.strides[1]
        d_rows = []
        by = {(r["rule"], r["stride"]): r for r in agg_rows}
        for rule in sorted({r["rule"] for r in agg_rows}):
            x, y = by.get((rule, a)), by.get((rule, b))
            if not x or not y:
                continue
            d_rows.append(dict(rule=rule, **{
                f"{k}_s{a}": x[k] for k in ("recall_target", "mean_iou", "p_continuity",
                                            "p_majority", "ghost_box_rate")},
                **{f"{k}_s{b}": y[k] for k in ("recall_target", "mean_iou", "p_continuity",
                                               "p_majority", "ghost_box_rate")},
                **{f"d_{k}": (None if x[k] is None or y[k] is None else round(y[k] - x[k], 4))
                   for k in ("recall_target", "mean_iou", "p_continuity",
                             "p_majority", "ghost_box_rate")}))
        with open(agg_dir / "stride_delta.csv", "w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(d_rows[0]))
            w.writeheader()
            w.writerows(d_rows)

    (agg_dir / "by_level_stride.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"已存 {agg_dir}")
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
