"""六個子集的問題片段:挑出來、渲染檢視影片、寫成一份清單,放在同一個資料夾給人看。

問題旗標由 `six_subsets_eval.py::problem_flags` 判定(門檻只用來挑片段,不是驗收門檻)。

挑法(先寫死,不看畫面挑):
  · 單人四個子集:有任一問題旗標的**全部**渲染。
  · 多人四個子集:幾乎每一段都有問題,所以各取**嚴重度最高的前 N 段**(預設 10)。
    嚴重度 = 各項指標低於門檻的缺口相加;同分依 clip_id 排序。
  · 影片一律用 `--profile` 指定的那一組設定(預設 single_person)重放;
    清單裡同時列出同一段在四組設定下的數字。

影片是**重放**不是重跑:框與編號讀自該次執行的 tracks.csv,逐幀對錯讀自同一套
clip_metrics(與指標同一份資料)。畫面上綠框是標註的目標,彩色框是系統框(同編號同色)。

輸出(--out-dir,預設 integrated/problem_clips;資料夾名用英文是因為 ffmpeg 在中文路徑會失敗):
  <範圍>_<規則>/<clip_id>_s5.mp4
  問題片段清單.md      渲染的片段逐段列出問題與數字
  problem_clips_all.csv 四組設定下所有有問題旗標的片段(未渲染的也在)

用法:
    python scripts/six_subsets_problems.py
"""
import argparse
import csv
import json
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from common.clip_metrics import clip_metrics                 # noqa: E402
from eval_clips import annotate, load_tracks                 # noqa: E402
from eval_m4m5_chirla import load_gt                         # noqa: E402
from render_clip_video import render                         # noqa: E402

PROFILES = ["delivery", "single_person", "plan_strict", "plan_share"]
RULES = ("L1", "L2", "L3S", "L3T")
NAME = {"single": "單人", "multi": "多人"}


def read_per_clip(out_root, profile):
    with open(out_root / profile / "eval" / "per_clip.csv", encoding="utf-8-sig") as f:
        return {r["clip_id"]: r for r in csv.DictReader(f)}


def pct(v):
    return "—" if v in ("", None) else f"{float(v) * 100:.0f}%"


def brief(r):
    """一段在某組設定下的幾個關鍵數字,給清單用。"""
    out = [f"召回 {pct(r['recall_target'])}", f"連續 {pct(r['p_continuity'])}"]
    if r["p_exclusive"] != "":
        out.append(f"排他 {pct(r['p_exclusive'])}")
    if r["l2_n_pairs_k5"] not in ("", "0"):
        out.append(f"再取得 {pct(r['l2_p_reacquire_k5'])}")
    if r["l3_n_pairs_usable"] not in ("", "0"):
        out.append(f"跨鏡頭 {pct(r['l3_p_handoff'])}")
    if r["l3_n_simul_frames"] not in ("", "0"):
        out.append(f"同時幀 {pct(r['l3_p_simul_agree'])}")
    out.append(f"編號數 {r['n_distinct_chef_ids']}")
    return "、".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="D:/yizhen/CHIRLA/CHIRLA_data/CHIRLA")
    ap.add_argument("--manifest", default="results/six_subsets/manifest.json")
    ap.add_argument("--results-root", default="results/six_subsets")
    ap.add_argument("--profile", default="single_person", help="影片用哪一組設定重放")
    ap.add_argument("--top-multi", type=int, default=10)
    ap.add_argument("--out-dir", default="integrated/problem_clips")
    ap.add_argument("--no-render", action="store_true")
    args = ap.parse_args()

    out_root, out_dir = ROOT / args.results_root, ROOT / args.out_dir
    man = json.loads((ROOT / args.manifest).read_text(encoding="utf-8"))
    clips = {c["clip_id"]: c for c in man["clips"]}
    per = {p: read_per_clip(out_root, p) for p in PROFILES}
    main_rows = per[args.profile]

    # ── 四組設定下所有有問題旗標的片段 ──
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "problem_clips_all.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["profile", "scope", "rule", "seq", "seq_group", "clip_id", "cameras", "gt_id",
                    "duration_s", "problem_flags", "severity", "recall_target", "p_continuity",
                    "p_exclusive", "l2_p_reacquire_k5", "l3_p_handoff", "l3_p_simul_agree",
                    "n_distinct_chef_ids", "chef_switches", "ghost_box_rate"])
        for p in PROFILES:
            for r in per[p].values():
                if r["problem_flags"]:
                    w.writerow([p] + [r[k] for k in (
                        "scope", "rule", "seq", "seq_group", "clip_id", "cameras", "gt_id",
                        "duration_s", "problem_flags", "severity", "recall_target", "p_continuity",
                        "p_exclusive", "l2_p_reacquire_k5", "l3_p_handoff", "l3_p_simul_agree",
                        "n_distinct_chef_ids", "chef_switches", "ghost_box_rate")])

    # ── 挑要渲染的 ──
    picked = defaultdict(list)
    for scope in ("single", "multi"):
        for rule in RULES:
            bad = sorted((r for r in main_rows.values()
                          if r["scope"] == scope and r["rule"] == rule and r["problem_flags"]),
                         key=lambda r: (-float(r["severity"]), r["clip_id"]))
            picked[(scope, rule)] = bad if scope == "single" else bad[:args.top_multi]
    n_pick = sum(len(v) for v in picked.values())
    print(f"要渲染 {n_pick} 段(設定 {args.profile})", flush=True)

    # ── 渲染 ──
    work = out_dir / "_work"
    rargs = SimpleNamespace(width=640, crf=26, encoder="ffmpeg", font="C:/Windows/Fonts/msjh.ttc")
    gts, done, i = {}, {}, 0
    for (scope, rule), rows in picked.items():
        sub = out_dir / f"{scope}_{rule}"
        for r in rows:
            i += 1
            clip = clips[r["clip_id"]]
            if args.no_render:
                continue
            sub.mkdir(parents=True, exist_ok=True)
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

    # ── 清單 ──
    L = ["# 問題片段清單(六個子集測試,2026-10-13)", "",
         "- 這個資料夾收的是**測試結果有問題的片段**的檢視影片,給人看畫面找原因用。**內部檢查用,不可交付。**",
         f"- 影片用 `{args.profile}` 設定的執行結果重放(不是重跑),每秒 6 幀。",
         "- 完整的問題片段名單(四組設定、含沒有渲染的)在 `problem_clips_all.csv`。",
         "- 指標的完整表格在 `../六子集測試結果.md`。", "",
         "## 怎麼看影片", "",
         "| 畫面上的東西 | 意思 |", "|---|---|",
         "| 綠色細框、`GT person N` | 人工標註的目標 |",
         "| 彩色粗框、`chef N` | 系統框;同一個編號在各鏡頭同一個顏色 |",
         "| 左上 `OK` | 這一幀目標的編號等於他在這段裡第一個編號 |",
         "| 左上 `WRONG ID` | 目標的編號變了 |",
         "| 左上 `MISS` | 目標有標註但系統沒框到 |",
         "| 左上 `unbound` | 框到了但還沒有編號 |",
         "| 第二行 `SHARED ID with person N` | 這一幀有另一個真人戴著同一個編號(多人片段才有) |",
         "| 沒有綠框對應的彩色框 | 可能是誤偵(倒影等),也可能是畫面裡的其他人 |", "",
         "## 問題旗標", "",
         "| 旗標 | 判定 |", "|---|---|",
         "| 漏偵 | 目標召回率 < 80% |",
         "| 完全沒有編號 | 目標從頭到尾沒有被綁到任何編號 |",
         "| 編號中途改變 | 連續率 < 90% |",
         "| 與別人共用編號 | 排他率 < 90%(多人) |",
         "| 離開再回來沒接回 | L2 的間隔前後不是同一個編號 |",
         "| 離開前或回來後沒有編號 | L2 的間隔有一側沒有任何綁定格,無法判定 |",
         "| 跨鏡頭編號不一致 | L3 有鏡頭對的多數編號不同 |",
         "| 某台鏡頭沒有編號 | L3 有一台鏡頭上目標從未拿到編號 |",
         "| 同時幀編號不一致 | L3S 同時幀一致率 < 90% |", "",
         "門檻只用來挑片段,不是驗收門檻。", "",
         "## 各子集有問題的筆數", "",
         "| 子集 | 總筆數 | " + " | ".join(f"`{p}`" for p in PROFILES) + " | 本資料夾渲染 |",
         "|---|---|" + "---|" * (len(PROFILES) + 1)]
    for scope in ("single", "multi"):
        for rule in RULES:
            tot = sum(1 for r in main_rows.values() if r["scope"] == scope and r["rule"] == rule)
            cnt = [sum(1 for r in per[p].values()
                       if r["scope"] == scope and r["rule"] == rule and r["problem_flags"])
                   for p in PROFILES]
            L.append(f"| {NAME[scope]} {rule} | {tot} | " + " | ".join(map(str, cnt))
                     + f" | {len(picked[(scope, rule)])} |")
    L += ["", f"- 單人:`{args.profile}` 下有問題的全部渲染。",
          f"- 多人:幾乎每段都有問題,各子集只渲染嚴重度最高的 {args.top_multi} 段"
          "(嚴重度 = 各項指標低於門檻的缺口相加)。", ""]

    for (scope, rule), rows in picked.items():
        if not rows:
            continue
        flags = Counter(f for r in rows for f in r["problem_flags"].split("、"))
        L += [f"## {NAME[scope]} {rule}({len(rows)} 段,資料夾 `{scope}_{rule}/`)", "",
              "這批片段的問題分布:" + "、".join(f"{k} {v} 段" for k, v in flags.most_common()), "",
              f"| # | 影片 | 鏡頭 | 目標 | 秒數 | 問題 | `{args.profile}` 的數字 | 其他三組的連續 / 排他 / 跨鏡頭 |",
              "|---|---|---|---|---|---|---|---|"]
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
                     f"{float(r['duration_s']):.1f} | {r['problem_flags']} | {brief(r)} | "
                     + ";".join(others) + " |")
        L.append("")
    (out_dir / "問題片段清單.md").write_text("\n".join(L), encoding="utf-8")
    (out_dir / "render_summary.json").write_text(
        json.dumps(done, ensure_ascii=False, indent=1), encoding="utf-8")
    bad = [k for k, v in done.items() if v["frames"] != v["expected_frames"]]
    print(f"完成:{len(done)} 支影片;幀數不符 {len(bad)} 支;清單 {out_dir / '問題片段清單.md'}")


if __name__ == "__main__":
    main()
