"""六個子集的評估與彙總。判定全部走 `common.clip_metrics`,這裡只做「片段 → 執行」的對應與彙集。

預先登記:docs/六子集測試_預先登記_20261013.md §4。

與 `eval_clips.py` 的差別:
  · 片段以 clip_id 為單位、執行以 run_id 為單位(多人 L1 一個窗多個目標共用一次執行)。
  · 依「範圍(單人 / 多人)× 規則 × 序列群(全部 / 評估集 / 推導集)」彙總。
  · 彙總方式相同:彙集逐格的旗標再算一次,不是把各片段的率平均。

輸出(results/six_subsets/<profile>/eval/):
  per_clip.csv   每筆片段 × 目標一列
  summary.csv    各子集的彙總
以及 results/six_subsets/summary_all.csv / summary_all.md(各組設定並排)。

用法:
    python scripts/six_subsets_eval.py --profiles delivery single_person plan_strict plan_share
"""
import argparse
import csv
import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from common.clip_metrics import clip_metrics, wilson       # noqa: E402
from eval_clips import annotate, load_tracks               # noqa: E402
from eval_m4m5_chirla import load_gt                       # noqa: E402

RULES = ("L1", "L2", "L3S", "L3T")


def rate(k, n):
    return None if not n else round(k / n, 4)


def summarize(clips, rows_of, ms):
    """clips:這一群的片段;rows_of / ms:clip_id → 逐格列 / 純量。"""
    rows = [r for c in clips for r in rows_of[c["clip_id"]]]
    mm = [ms[c["clip_id"]] for c in clips]
    n_t = sum(r["gt_present"] for r in rows)
    k_rec = sum(1 for r in rows if r["gt_present"] and r["matched"])
    b = [r for r in rows if r["in_B"]]
    k_c = sum(1 for r in b if r["is_continuity"])
    k_m = sum(1 for r in b if r["is_majority"])
    eb = [r for r in b if r["is_exclusive"] != ""]
    k_e = sum(1 for r in eb if r["is_exclusive"])
    n_box, n_gh = sum(r["n_tracks"] for r in rows), sum(r["n_ghost"] for r in rows)
    out = dict(n_clips=len(clips), n_target_cells=n_t, n_bound_cells=len(b),
               recall=rate(k_rec, n_t), bound_rate=rate(len(b), n_t),
               continuity=rate(k_c, len(b)), continuity_k=k_c, continuity_ci=wilson(k_c, len(b)),
               majority=rate(k_m, len(b)),
               exclusive=rate(k_e, len(eb)), exclusive_k=k_e, exclusive_n=len(eb),
               exclusive_ci=wilson(k_e, len(eb)),
               chef_switches=sum(m["chef_switches"] for m in mm),
               chef_ids_median=st.median([m["n_distinct_chef_ids"] for m in mm]) if mm else None,
               ghost_box_rate=rate(n_gh, n_box))
    for k in (5, 1):       # L2 再取得
        npair = sum(m.get(f"l2_n_pairs_k{k}", 0) for m in mm)
        ok = sum(round((m.get(f"l2_p_reacquire_k{k}") or 0) * m.get(f"l2_n_pairs_k{k}", 0))
                 for m in mm)
        out.update({f"reacq_k{k}": rate(ok, npair), f"reacq_k{k}_k": ok, f"reacq_k{k}_n": npair})
    npair = sum(m.get("l3_n_pairs_usable", 0) for m in mm)     # L3 跨鏡頭
    ok = sum(round((m.get("l3_p_handoff") or 0) * m.get("l3_n_pairs_usable", 0)) for m in mm)
    nsim = sum(m.get("l3_n_simul_frames", 0) for m in mm)
    sok = sum(round((m.get("l3_p_simul_agree") or 0) * m.get("l3_n_simul_frames", 0)) for m in mm)
    out.update(handoff=rate(ok, npair), handoff_k=ok, handoff_n=npair,
               handoff_dropped=sum(m.get("l3_n_pairs_dropped", 0) for m in mm),
               handoff_ci=wilson(ok, npair),
               simul=rate(sok, nsim), simul_k=sok, simul_n=nsim)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="D:/yizhen/CHIRLA/CHIRLA_data/CHIRLA")
    ap.add_argument("--manifest", default="results/six_subsets/manifest.json")
    ap.add_argument("--out-root", default="results/six_subsets")
    ap.add_argument("--profiles", nargs="+",
                    default=["delivery", "single_person", "plan_strict", "plan_share"])
    args = ap.parse_args()

    man = json.loads((ROOT / args.manifest).read_text(encoding="utf-8"))
    out_root = ROOT / args.out_root
    gts, all_rows = {}, []
    for profile in args.profiles:
        rows_of, ms, missing, cache = {}, {}, [], {}
        for clip in man["clips"]:
            run_dir = out_root / profile / "runs" / clip["run_id"]
            if not (run_dir / "tracks.csv").exists():
                missing.append(clip["clip_id"])
                continue
            if clip["seq"] not in gts:
                gts[clip["seq"]] = load_gt(args.root, clip["seq"])
            if clip["run_id"] not in cache:
                cache = {clip["run_id"]: load_tracks(run_dir)}     # 只留最近一個窗,省記憶體
            tracks, chefs = cache[clip["run_id"]]
            rows, m = clip_metrics(clip, 5, gts[clip["seq"]], tracks, chefs)
            rows_of[clip["clip_id"]] = annotate(rows, m)
            ms[clip["clip_id"]] = m
        done = [c for c in man["clips"] if c["clip_id"] in ms]
        print(f"{profile}:評估 {len(done)} / {len(man['clips'])} 筆,缺 {len(missing)} 筆",
              flush=True)

        ev = out_root / profile / "eval"
        ev.mkdir(parents=True, exist_ok=True)
        keys = ["clip_id", "scope", "rule", "seq", "seq_group", "cameras", "gt_id", "duration_s",
                "exclusive", "n_other_ids", "n_target_cells", "recall_target", "n_B",
                "p_continuity", "p_majority", "p_exclusive", "chef_switches",
                "n_distinct_chef_ids", "ghost_box_rate", "l2_p_reacquire_k5", "l2_n_pairs_k5",
                "l3_p_handoff", "l3_n_pairs_usable", "l3_n_pairs_dropped",
                "l3_p_simul_agree", "l3_n_simul_frames"]
        with open(ev / "per_clip.csv", "w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            w.writeheader()
            for c in done:
                w.writerow({**ms[c["clip_id"]], "scope": c["scope"], "seq_group": c["seq_group"]})

        groups = defaultdict(list)
        for c in done:
            groups[(c["scope"], c["rule"], "all")].append(c)
            groups[(c["scope"], c["rule"], c["seq_group"])].append(c)
        prof_rows = []
        for scope in ("single", "multi"):
            for rule in RULES:
                for sg in ("all", "eval", "derive"):
                    g = groups.get((scope, rule, sg))
                    if g:
                        prof_rows.append(dict(profile=profile, scope=scope, rule=rule,
                                              seq_group=sg, n_missing=len(missing),
                                              **summarize(g, rows_of, ms)))
        with open(ev / "summary.csv", "w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(prof_rows[0]))
            w.writeheader()
            w.writerows(prof_rows)
        all_rows += prof_rows

    with open(out_root / "summary_all.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0]))
        w.writeheader()
        w.writerows(all_rows)

    def pct(v):
        return "—" if v is None else f"{v * 100:.1f}%"

    def frac(r, key):
        return "—" if not r[f"{key}_n"] else f"{pct(r[key])} ({r[f'{key}_k']}/{r[f'{key}_n']})"

    lines = ["# 六個子集 × 各組設定(逐片段獨立推論,stride 5)", ""]
    by = {(r["profile"], r["scope"], r["rule"], r["seq_group"]): r for r in all_rows}
    for sg, title in (("all", "全部序列"), ("eval", "只含評估集序列"), ("derive", "只含推導集序列")):
        lines += [f"## {title}", ""]
        for scope in ("single", "multi"):
            for rule in RULES:
                rs = [by.get((p, scope, rule, sg)) for p in args.profiles]
                if not any(rs):
                    continue
                r0 = next(r for r in rs if r)
                lines += [f"### {'單人' if scope == 'single' else '多人'} {rule}"
                          f"({r0['n_clips']} 筆,目標在場 {r0['n_target_cells']} 格)", "",
                          "| 設定 | 召回 | 連續率 | 排他率 | 切換 | 誤偵框 | L2 再取得 k5 | L3 跨鏡頭一致 | L3 同時幀一致 |",
                          "|---|---|---|---|---|---|---|---|---|"]
                for p, r in zip(args.profiles, rs):
                    if not r:
                        continue
                    lines.append(
                        f"| {p} | {pct(r['recall'])} | "
                        f"{pct(r['continuity'])} ({r['continuity_k']}/{r['n_bound_cells']}) | "
                        f"{frac(r, 'exclusive')} | {r['chef_switches']} | "
                        f"{pct(r['ghost_box_rate'])} | {frac(r, 'reacq_k5')} | "
                        f"{frac(r, 'handoff')} | {frac(r, 'simul')} |")
                lines.append("")
    (out_root / "summary_all.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
