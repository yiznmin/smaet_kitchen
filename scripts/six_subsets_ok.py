"""六個子集裡「沒有問題」的片段:挑出來、渲染檢視影片、寫成清單,與 problem_clips 對照看。

「沒有問題」= `six_subsets_eval.py::problem_flags` 沒有任何旗標(門檻與問題片段相同)。

挑法(先寫死,不看畫面挑):
  · 每個子集取沒有問題的片段中,目標在場格數最多的前 N 段(預設 5;不足 N 段就全取)。
    挑格數多的是因為畫面內容較多、較值得看;同分依 clip_id 排序。
  · 影片用 `--profile` 那一組設定(預設 single_person,與 problem_clips 相同)重放。

輸出(--out-dir,預設 integrated/ok_clips):
  <範圍>_<規則>/<clip_id>_s5.mp4
  正常片段清單.md
  ok_clips_all.csv     四組設定下所有沒有問題的片段

用法:
    python scripts/six_subsets_ok.py
"""
import argparse
import csv
import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from common.clip_metrics import clip_metrics                       # noqa: E402
from eval_clips import annotate, load_tracks                       # noqa: E402
from eval_m4m5_chirla import load_gt                               # noqa: E402
from render_clip_video import render                               # noqa: E402
from six_subsets_problems import NAME, PROFILES, RULES, brief, pct, read_per_clip   # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="D:/yizhen/CHIRLA/CHIRLA_data/CHIRLA")
    ap.add_argument("--manifest", default="results/six_subsets/manifest.json")
    ap.add_argument("--results-root", default="results/six_subsets")
    ap.add_argument("--profile", default="single_person")
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--out-dir", default="integrated/ok_clips")
    args = ap.parse_args()

    out_root, out_dir = ROOT / args.results_root, ROOT / args.out_dir
    man = json.loads((ROOT / args.manifest).read_text(encoding="utf-8"))
    clips = {c["clip_id"]: c for c in man["clips"]}
    per = {p: read_per_clip(out_root, p) for p in PROFILES}
    main_rows = per[args.profile]
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "ok_clips_all.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        keys = ["scope", "rule", "seq", "seq_group", "clip_id", "cameras", "gt_id", "duration_s",
                "n_target_cells", "recall_target", "p_continuity", "p_exclusive",
                "l2_p_reacquire_k5", "l3_p_handoff", "l3_p_simul_agree", "n_distinct_chef_ids"]
        w.writerow(["profile"] + keys)
        for p in PROFILES:
            for r in per[p].values():
                if not r["problem_flags"]:
                    w.writerow([p] + [r[k] for k in keys])

    picked, total_ok = {}, {}
    for scope in ("single", "multi"):
        for rule in RULES:
            ok = sorted((r for r in main_rows.values()
                         if r["scope"] == scope and r["rule"] == rule and not r["problem_flags"]),
                        key=lambda r: (-int(r["n_target_cells"]), r["clip_id"]))
            total_ok[(scope, rule)] = len(ok)
            picked[(scope, rule)] = ok[:args.top]
    n_pick = sum(len(v) for v in picked.values())
    print(f"要渲染 {n_pick} 段(設定 {args.profile})", flush=True)

    work = out_dir / "_work"
    rargs = SimpleNamespace(width=640, crf=26, encoder="ffmpeg", font="C:/Windows/Fonts/msjh.ttc")
    gts, done, i = {}, {}, 0
    for (scope, rule), rows in picked.items():
        sub = out_dir / f"{scope}_{rule}"
        for r in rows:
            i += 1
            sub.mkdir(parents=True, exist_ok=True)
            clip = clips[r["clip_id"]]
            run_src = out_root / args.profile / "runs" / clip["run_id"]
            run_dst = work / "runs" / clip["clip_id"] / "s5"
            met_dst = work / "metrics" / clip["clip_id"] / "s5"
            run_dst.mkdir(parents=True, exist_ok=True)
            met_dst.mkdir(parents=True, exist_ok=True)
            for name in ("tracks.csv", "run_meta.json"):
                shutil.copy2(run_src / name, run_dst / name)
            if clip["seq"] not in gts:
                gts = {clip["seq"]: load_gt(args.root, clip["seq"])}
            tracks, chefs = load_tracks(run_dst)
            prow, m = clip_metrics(clip, 5, gts[clip["seq"]], tracks, chefs)
            prow = annotate(prow, m)
            with open(met_dst / "per_frame.csv", "w", encoding="utf-8-sig", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(prow[0]))
                w.writeheader()
                w.writerows(prow)
            info = render(clip, 5, work / "runs", work / "metrics", sub, rargs)
            done[clip["clip_id"]] = info
            print(f"[{i:>3}/{n_pick}] {scope} {rule} {clip['clip_id']}  "
                  f"{info['frames']} 幀 {info['bytes'] / 1e6:.2f} MB", flush=True)
    shutil.rmtree(work, ignore_errors=True)

    L = ["# 正常片段清單(六個子集測試,2026-10-13 的結果)", "",
         "- 這個資料夾收的是**測試結果沒有問題的片段**的檢視影片,與 `../problem_clips/` 對照看。**內部檢查用,不可交付。**",
         f"- 影片用 `{args.profile}` 設定、COCO 預訓偵測的執行結果重放(不是重跑),每秒 6 幀。",
         "- 「沒有問題」= 沒有任何問題旗標,判定與 `../problem_clips/問題片段清單.md` 的旗標表相同"
         "(召回 ≥ 80%、連續率 ≥ 90%、多人排他率 ≥ 90%、L2 有接回、L3 各台編號一致且同時幀一致率 ≥ 90%)。",
         f"- 每個子集取沒有問題的片段中,目標在場格數最多的前 {args.top} 段;不足就全取。",
         "- 四組設定下所有沒有問題的片段在 `ok_clips_all.csv`。",
         "- 畫面標示的意思見 `../problem_clips/問題片段清單.md`「怎麼看影片」。", "",
         "## 各子集沒有問題的筆數", "",
         "| 子集 | 總筆數 | " + " | ".join(f"`{p}`" for p in PROFILES) + " | 本資料夾渲染 |",
         "|---|---|" + "---|" * (len(PROFILES) + 1)]
    for scope in ("single", "multi"):
        for rule in RULES:
            tot = sum(1 for r in main_rows.values() if r["scope"] == scope and r["rule"] == rule)
            cnt = [sum(1 for r in per[p].values()
                       if r["scope"] == scope and r["rule"] == rule and not r["problem_flags"])
                   for p in PROFILES]
            L.append(f"| {NAME[scope]} {rule} | {tot} | " + " | ".join(map(str, cnt))
                     + f" | {len(picked[(scope, rule)])} |")
    L += ["", "⚠ 多人 L3 沒有問題的片段極少(`single_person` 下 L3S 4/448、L3T 1/59),"
          "這裡的影片是**少數例外**,不代表多人跨鏡頭的一般表現。", ""]
    for (scope, rule), rows in picked.items():
        if not rows:
            L += [f"## {NAME[scope]} {rule}", "", f"`{args.profile}` 下沒有任何沒問題的片段。", ""]
            continue
        L += [f"## {NAME[scope]} {rule}({len(rows)} 段,資料夾 `{scope}_{rule}/`)", "",
              f"| # | 影片 | 鏡頭 | 目標 | 秒數 | `{args.profile}` 的數字 | 其他三組的連續 / 排他 / 跨鏡頭 |",
              "|---|---|---|---|---|---|---|"]
        for j, r in enumerate(rows, 1):
            others = []
            for p in PROFILES:
                if p == args.profile:
                    continue
                o = per[p][r["clip_id"]]
                others.append(f"`{p}` {pct(o['p_continuity'])} / {pct(o['p_exclusive'])} / "
                              f"{pct(o['l3_p_handoff'])}")
            cams = r["cameras"].replace("camera_", "c")
            grp = "(推導集)" if r["seq_group"] == "derive" else ""
            L.append(f"| {j} | `{r['clip_id']}_s5.mp4` | {r['seq']}{grp} {cams} | {r['gt_id']} | "
                     f"{float(r['duration_s']):.1f} | {brief(r)} | " + ";".join(others) + " |")
        L.append("")
    (out_dir / "正常片段清單.md").write_text("\n".join(L), encoding="utf-8")
    (out_dir / "render_summary.json").write_text(json.dumps(done, ensure_ascii=False, indent=1),
                                                 encoding="utf-8")
    bad = [k for k, v in done.items() if v["frames"] != v["expected_frames"]]
    print(f"完成:{len(done)} 支影片;幀數不符 {len(bad)} 支;清單 {out_dir / '正常片段清單.md'}")


if __name__ == "__main__":
    main()
