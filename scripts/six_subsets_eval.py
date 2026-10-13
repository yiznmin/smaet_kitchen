"""六個子集的評估與彙總。判定全部走 `common.clip_metrics`,這裡只做「片段 → 執行」的對應與彙集。

預先登記:docs/六子集測試_預先登記_20261013.md §4;完整指標清單沿用
docs/M4_外觀模型_L2單鏡頭再取得_結果彙整_20260929.md §3 的四類(身份 / 偵測與定位 / 代價 / 跨鏡頭)。

與 `eval_clips.py` 的差別:
  · 片段以 clip_id 為單位、執行以 run_id 為單位(多人 L1 一個窗多個目標共用一次執行)。
  · 依「範圍(單人 / 多人)× 規則 × 序列群(全部 / 評估集 / 推導集)」彙總。
  · 彙總方式相同:彙集逐格的旗標再算一次,不是把各片段的率平均。
  · 「框」與「track」層的統計(誤偵框、誤偵 track)以 run_id 去重 ——
    多人 L1 同一個窗有多個目標,不去重會把同一批框算好幾次。

輸出(results/six_subsets/<profile>/eval/):
  per_clip.csv   每筆片段 × 目標一列(含問題旗標)
  summary.csv    各子集的彙總
以及 results/six_subsets/ 的 summary_all.csv / summary_all.md(精簡並排)、
metrics_full.md(完整指標,附 95% 區間)。

用法:
    python scripts/six_subsets_eval.py
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
MIRROR = {"camera_5", "camera_6"}      # 對著玻璃面,倒影會被框成人
PATHS = ("重疊", "重疊+連結", "轉場連結", "未建模")
# 問題旗標的門檻(只用來挑片段給人看,不是驗收門檻)
TH = dict(recall=0.80, continuity=0.90, exclusive=0.90, simul=0.90)


def problem_flags(m):
    """這一筆片段有哪些問題。回傳旗標清單與嚴重度(各項缺口相加,越大越糟)。"""
    flags, sev = [], 0.0

    def add(name, deficit):
        nonlocal sev
        flags.append(name)
        sev += max(deficit, 0.0)

    if m["recall_target"] is not None and m["recall_target"] < TH["recall"]:
        add("漏偵", TH["recall"] - m["recall_target"])
    if m["n_B"] == 0:
        add("完全沒有編號", 1.0)
    if m["p_continuity"] is not None and m["p_continuity"] < TH["continuity"]:
        add("編號中途改變", TH["continuity"] - m["p_continuity"])
    if m.get("p_exclusive") is not None and m["p_exclusive"] < TH["exclusive"]:
        add("與別人共用編號", TH["exclusive"] - m["p_exclusive"])
    if m.get("l2_n_pairs_k5") and m["l2_p_reacquire_k5"] < 1.0:
        add("離開再回來沒接回", 1.0 - m["l2_p_reacquire_k5"])
    if m.get("l2_n_pairs_dropped_k5"):
        add("離開前或回來後沒有編號", 0.5)
    if m.get("l3_n_pairs_usable") and m["l3_p_handoff"] < 1.0:
        add("跨鏡頭編號不一致", 1.0 - m["l3_p_handoff"])
    if m.get("l3_n_pairs_dropped"):
        add("某台鏡頭沒有編號", 0.5)
    if m.get("l3_n_simul_frames") and m["l3_p_simul_agree"] < TH["simul"]:
        add("同時幀編號不一致", TH["simul"] - m["l3_p_simul_agree"])
    return flags, round(sev, 4)


def rate(k, n):
    return None if not n else round(k / n, 4)


def summarize(clips, rows_of, ms):
    """clips:這一群的片段;rows_of / ms:clip_id → 逐格列 / 純量。"""
    rows = [r for c in clips for r in rows_of[c["clip_id"]]]
    mm = [ms[c["clip_id"]] for c in clips]
    tgt = [r for r in rows if r["gt_present"]]
    hit = [r for r in tgt if r["matched"]]
    b = [r for r in rows if r["in_B"]]
    k_c = sum(1 for r in b if r["is_continuity"])
    k_m = sum(1 for r in b if r["is_majority"])
    eb = [r for r in b if r["is_exclusive"] != ""]
    k_e = sum(1 for r in eb if r["is_exclusive"])
    k_iou75 = sum(1 for r in tgt if r["iou"] >= 0.75)
    # 框與 track 層:以 run_id 去重(同一個窗只算一次)
    seen, uniq = set(), []
    for c in clips:
        if c["run_id"] not in seen:
            seen.add(c["run_id"])
            uniq.append(c)
    urows = [r for c in uniq for r in rows_of[c["clip_id"]]]
    um = [ms[c["clip_id"]] for c in uniq]
    n_box, n_gh = sum(r["n_tracks"] for r in urows), sum(r["n_ghost"] for r in urows)
    n_trk = sum(m["n_tracks_in_window"] for m in um)
    n_gtrk = sum(m["n_ghost_tracks"] for m in um)
    k_shared = sum(round((m["p_chef_shared_with_other_box"] or 0) * m["n_B"]) for m in mm)
    ious = sorted(r["iou"] for r in tgt)
    q = (lambda p: round(ious[min(len(ious) - 1, int(p * len(ious)))], 4)) if ious else (lambda p: None)
    # 切換率的分母:目標有編號的格子裡,同一段同一台鏡頭沿時間相鄰的格對數
    per_cam = defaultdict(int)
    for r in b:
        per_cam[(r["clip_id"], r["camera_id"])] += 1
    n_adj = sum(n - 1 for n in per_cam.values())
    n_csw = sum(m["chef_switches"] for m in mm)
    n_tsw = sum(m["track_switches"] for m in mm)
    out = dict(
        n_clips=len(clips), n_runs=len(uniq), n_cells=len(rows),
        visibility=rate(len(tgt), len(rows)), iou_p10=q(0.10), iou_p90=q(0.90),
        n_adjacent=n_adj, chef_switch_rate=rate(n_csw, n_adj),
        track_switch_rate=rate(n_tsw, n_adj),
        l2_false_success=sum(1 for m in mm if m.get("l2_false_success")),
        n_target_cells=len(tgt), n_matched=len(hit), n_bound_cells=len(b),
        # 偵測與定位
        recall=rate(len(hit), len(tgt)), recall_ci=wilson(len(hit), len(tgt)),
        bound_rate=rate(len(b), len(tgt)), bound_ci=wilson(len(b), len(tgt)),
        mean_iou_matched=round(sum(r["iou"] for r in hit) / len(hit), 4) if hit else None,
        median_iou=round(st.median([r["iou"] for r in tgt]), 4) if tgt else None,
        iou75=rate(k_iou75, len(tgt)), iou75_k=k_iou75, iou75_ci=wilson(k_iou75, len(tgt)),
        # 身份
        continuity=rate(k_c, len(b)), continuity_k=k_c, continuity_ci=wilson(k_c, len(b)),
        majority=rate(k_m, len(b)), majority_k=k_m, majority_ci=wilson(k_m, len(b)),
        exclusive=rate(k_e, len(eb)), exclusive_k=k_e, exclusive_n=len(eb),
        exclusive_ci=wilson(k_e, len(eb)),
        chef_switches=sum(m["chef_switches"] for m in mm),
        track_switches=sum(m["track_switches"] for m in mm),
        n_chef_ids=sum(m["n_distinct_chef_ids"] for m in mm),
        chef_ids_median=st.median([m["n_distinct_chef_ids"] for m in mm]) if mm else None,
        n_clips_one_id=sum(1 for m in mm if m["n_distinct_chef_ids"] == 1),
        n_clips_perfect=sum(1 for m in mm if m["p_continuity"] == 1.0),
        n_clips_unbound=sum(1 for m in mm if m["n_B"] == 0),
        chefs_before_target_median=st.median(
            [m["n_chefs_created_before_target_bound"] for m in mm]) if mm else None,
        # 代價
        n_boxes=n_box, n_ghost_boxes=n_gh, ghost_box_rate=rate(n_gh, n_box),
        ghost_box_ci=wilson(n_gh, n_box),
        n_tracks=n_trk, n_ghost_tracks=n_gtrk, ghost_track_rate=rate(n_gtrk, n_trk),
        ghost_track_ci=wilson(n_gtrk, n_trk),
        shared_with_other_box=rate(k_shared, len(b)), shared_with_other_box_k=k_shared,
        n_problem_clips=sum(1 for m in mm if m["problem_flags"]))
    for k in (5, 1):       # L2 再取得
        npair = sum(m.get(f"l2_n_pairs_k{k}", 0) for m in mm)
        ok = sum(round((m.get(f"l2_p_reacquire_k{k}") or 0) * m.get(f"l2_n_pairs_k{k}", 0))
                 for m in mm)
        out.update({f"reacq_k{k}": rate(ok, npair), f"reacq_k{k}_k": ok, f"reacq_k{k}_n": npair,
                    f"reacq_k{k}_ci": wilson(ok, npair),
                    f"reacq_k{k}_dropped": sum(m.get(f"l2_n_pairs_dropped_k{k}", 0) for m in mm)})
    npair = sum(m.get("l3_n_pairs_usable", 0) for m in mm)     # L3 跨鏡頭
    ok = sum(round((m.get("l3_p_handoff") or 0) * m.get("l3_n_pairs_usable", 0)) for m in mm)
    nsim = sum(m.get("l3_n_simul_frames", 0) for m in mm)
    sok = sum(round((m.get("l3_p_simul_agree") or 0) * m.get("l3_n_simul_frames", 0)) for m in mm)
    out.update(handoff=rate(ok, npair), handoff_k=ok, handoff_n=npair,
               handoff_dropped=sum(m.get("l3_n_pairs_dropped", 0) for m in mm),
               handoff_ci=wilson(ok, npair),
               simul=rate(sok, nsim), simul_k=sok, simul_n=nsim, simul_ci=wilson(sok, nsim))
    return out


def breakdowns(clips, rows_of, ms, topo):
    """三種分解:L2 依間隔帶、L3 依鏡頭對路徑、各子集依是否牽涉倒影鏡頭。回傳 {鍵: (分子, 分母)}。"""
    ov, lk = topo
    out = defaultdict(lambda: [0, 0])

    def add(key, k, n):
        out[key][0] += k
        out[key][1] += n

    for c in clips:
        m, sc = ms[c["clip_id"]], c["scope"]
        mir = "含倒影鏡頭" if m["mirror"] else "不含倒影鏡頭"
        b = [r for r in rows_of[c["clip_id"]] if r["in_B"]]
        add(("mirror", sc, c["rule"], mir, "n_clips"), 1, 1)
        add(("mirror", sc, c["rule"], mir, "continuity"), sum(1 for r in b if r["is_continuity"]), len(b))
        eb = [r for r in b if r["is_exclusive"] != ""]
        add(("mirror", sc, c["rule"], mir, "exclusive"), sum(1 for r in eb if r["is_exclusive"]), len(eb))
        tg = [r for r in rows_of[c["clip_id"]] if r["gt_present"]]
        add(("mirror", sc, c["rule"], mir, "recall"), sum(1 for r in tg if r["matched"]), len(tg))
        if c["level"] == "L2" and m.get("l2_n_pairs_k5"):
            add(("gap", sc, m["gap_band"], "reacq"),
                round(m["l2_p_reacquire_k5"] * m["l2_n_pairs_k5"]), m["l2_n_pairs_k5"])
            add(("gap", sc, m["gap_band"], "n_clips"), 1, 1)
            add(("mirror", sc, c["rule"], mir, "reacq"),
                round(m["l2_p_reacquire_k5"] * m["l2_n_pairs_k5"]), m["l2_n_pairs_k5"])
        if c["level"] == "L3":
            per = m.get("l3_chef_per_camera") or {}
            cams = sorted(c["cameras"])
            for i, a in enumerate(cams):
                for bb in cams[i + 1:]:
                    key = frozenset((a, bb))
                    path = ("重疊+連結" if key in ov and key in lk else "重疊" if key in ov
                            else "轉場連結" if key in lk else "未建模")
                    add(("path", sc, c["rule"], path, "pairs"), 1, 1)
                    if per.get(a) is None or per.get(bb) is None:
                        add(("path", sc, c["rule"], path, "dropped"), 1, 1)
                        continue
                    add(("path", sc, c["rule"], path, "handoff"), int(per[a] == per[bb]), 1)
                    add(("mirror", sc, c["rule"], mir, "handoff"), int(per[a] == per[bb]), 1)
    return {k: tuple(v) for k, v in out.items()}


def kn(d, key):
    k, n = d.get(key, (0, 0))
    return "—" if not n else f"{k / n * 100:.1f}%({k:,}/{n:,})"


def pct(v):
    return "—" if v is None else f"{v * 100:.1f}%"


def cell(r, key, k=None, n=None, ci=None):
    """『率(分子/分母)[區間]』;分母為 0 印「不可量測」。"""
    if r is None:
        return "—"
    nn = r[n] if n else None
    if n and not nn:
        return "不可量測"
    s = pct(r[key])
    if k and n:
        s += f"({r[k]:,}/{nn:,})"
    if ci and r[ci][0] is not None:
        s += f"<br>[{r[ci][0]:.3f}, {r[ci][1]:.3f}]"
    return s


FULL_ROWS = (
    ("**身份**", None),
    ("連續率(主指標)", lambda r: cell(r, "continuity", "continuity_k", "n_bound_cells", "continuity_ci")),
    ("多數率(輔助)", lambda r: cell(r, "majority", "majority_k", "n_bound_cells", "majority_ci")),
    ("排他率", lambda r: cell(r, "exclusive", "exclusive_k", "exclusive_n", "exclusive_ci")),
    ("編號切換總次數", lambda r: f"{r['chef_switches']:,}"),
    ("編號切換率(次數 / 相鄰格對)", lambda r: cell(r, "chef_switch_rate", "chef_switches", "n_adjacent")),
    ("不同編號個數(理想值 = 筆數)", lambda r: f"{r['n_chef_ids']:,}(理想 {r['n_clips']:,})"),
    ("全程只有一個編號的筆數", lambda r: f"{r['n_clips_one_id']:,}/{r['n_clips']:,}"),
    ("連續率 100% 的筆數", lambda r: f"{r['n_clips_perfect']:,}/{r['n_clips']:,}"),
    ("目標拿到編號前已開的編號數(中位)", lambda r: f"{r['chefs_before_target_median']:g}"),
    ("L2 再取得率 k=5", lambda r: cell(r, "reacq_k5", "reacq_k5_k", "reacq_k5_n", "reacq_k5_ci")),
    ("L2 再取得率 k=1", lambda r: cell(r, "reacq_k1", "reacq_k1_k", "reacq_k1_n", "reacq_k1_ci")),
    ("L2 假成功段數(再取得成功但結尾編號已不同)", lambda r: f"{r['l2_false_success']}" if r["reacq_k5_n"] else "—"),
    ("L2 無法判定的間隔對", lambda r: f"{r['reacq_k5_dropped']}" if r["reacq_k5_n"] or r["reacq_k5_dropped"] else "—"),
    ("**跨鏡頭(L3)**", None),
    ("跨鏡頭一致率", lambda r: cell(r, "handoff", "handoff_k", "handoff_n", "handoff_ci")),
    ("排除的鏡頭對(某台沒有編號)", lambda r: f"{r['handoff_dropped']}" if r["handoff_n"] or r["handoff_dropped"] else "—"),
    ("同時幀一致率", lambda r: cell(r, "simul", "simul_k", "simul_n", "simul_ci")),
    ("**偵測與定位**", None),
    ("目標召回率", lambda r: cell(r, "recall", "n_matched", "n_target_cells", "recall_ci")),
    ("綁定率", lambda r: cell(r, "bound_rate", "n_bound_cells", "n_target_cells", "bound_ci")),
    ("完全沒有編號的筆數", lambda r: f"{r['n_clips_unbound']:,}/{r['n_clips']:,}"),
    ("IoU 平均(配對成功者)", lambda r: "—" if r["mean_iou_matched"] is None else f"{r['mean_iou_matched']:.4f}"),
    ("IoU 中位(全部目標格)", lambda r: "—" if r["median_iou"] is None else f"{r['median_iou']:.4f}"),
    ("IoU 第 10 / 第 90 百分位", lambda r: "—" if r["iou_p10"] is None else f"{r['iou_p10']:.4f} / {r['iou_p90']:.4f}"),
    ("IoU ≥ 0.75 比例", lambda r: cell(r, "iou75", "iou75_k", "n_target_cells", "iou75_ci")),
    ("track 切換總次數", lambda r: f"{r['track_switches']:,}"),
    ("track 切換率(次數 / 相鄰格對)", lambda r: cell(r, "track_switch_rate", "track_switches", "n_adjacent")),
    ("目標可見比例(有標註框的格 / 取樣格)", lambda r: cell(r, "visibility", "n_target_cells", "n_cells")),
    ("**代價**", None),
    ("誤偵框比例", lambda r: cell(r, "ghost_box_rate", "n_ghost_boxes", "n_boxes", "ghost_box_ci")),
    ("誤偵 track 比例", lambda r: cell(r, "ghost_track_rate", "n_ghost_tracks", "n_tracks", "ghost_track_ci")),
    ("系統框總數", lambda r: f"{r['n_boxes']:,}"),
    ("編號同時出現在另一個框上的比例", lambda r: cell(r, "shared_with_other_box", "shared_with_other_box_k", "n_bound_cells")),
    ("**問題片段**", None),
    ("有任一問題旗標的筆數", lambda r: f"{r['n_problem_clips']:,}/{r['n_clips']:,}"),
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="D:/yizhen/CHIRLA/CHIRLA_data/CHIRLA")
    ap.add_argument("--manifest", default="results/six_subsets/manifest.json")
    ap.add_argument("--out-root", default="results/six_subsets")
    ap.add_argument("--eval-topology", default="configs/fix_grid/base_overlapfix.yaml",
                    help="只用來把 L3 的鏡頭對分成 重疊 / 連結 / 未建模,四組設定共用同一份")
    ap.add_argument("--profiles", nargs="+",
                    default=["delivery", "single_person", "plan_strict", "plan_share"])
    args = ap.parse_args()

    man = json.loads((ROOT / args.manifest).read_text(encoding="utf-8"))
    out_root = ROOT / args.out_root
    gts, all_rows, extra = {}, [], {}
    import yaml
    tp = yaml.safe_load((ROOT / args.eval_topology).read_text(encoding="utf-8"))["camera_topology"]
    topo = ({frozenset(p) for p in tp.get("overlapping", [])},
            {frozenset((l["from"], l["to"])) for l in tp.get("links", [])})
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
            bound = sorted((r for r in rows_of[clip["clip_id"]] if r["in_B"]),
                           key=lambda r: (r["video_fid"], r["camera_id"]))
            m["last_chef"] = bound[-1]["chef_id"] if bound else ""
            # 假成功:間隔前後的多數編號相同(算再取得成功),但片段結尾已不是第一個編號
            m["l2_false_success"] = int(bool(
                m.get("l2_n_pairs_k5") and m["l2_p_reacquire_k5"] == 1.0
                and bound and bound[-1]["chef_id"] != m["c0"]))
            m["mirror"] = int(bool(set(clip["cameras"]) & MIRROR))
            flags, sev = problem_flags(m)
            m.update(scope=clip["scope"], seq_group=clip["seq_group"], run_id=clip["run_id"],
                     problem_flags="、".join(flags), severity=sev)
            ms[clip["clip_id"]] = m
        done = [c for c in man["clips"] if c["clip_id"] in ms]
        print(f"{profile}:評估 {len(done)} / {len(man['clips'])} 筆,缺 {len(missing)} 筆",
              flush=True)

        ev = out_root / profile / "eval"
        ev.mkdir(parents=True, exist_ok=True)
        keys = ["clip_id", "run_id", "scope", "rule", "seq", "seq_group", "cameras", "gt_id",
                "duration_s", "exclusive", "n_other_ids", "problem_flags", "severity",
                "n_target_cells", "n_matched", "recall_target", "n_B", "bound_rate",
                "mean_iou_matched", "median_iou", "frac_iou_ge_075",
                "mirror", "c0", "c_star", "last_chef", "l2_false_success", "p_continuity", "p_majority", "p_exclusive",
                "chef_switches", "track_switches", "n_distinct_chef_ids", "n_track_ids",
                "n_chefs_created_before_target_bound",
                "n_boxes", "n_ghost_boxes", "ghost_box_rate", "n_tracks_in_window",
                "n_ghost_tracks", "ghost_track_rate", "p_chef_shared_with_other_box",
                "l2_p_reacquire_k5", "l2_n_pairs_k5", "l2_n_pairs_dropped_k5",
                "l2_p_reacquire_k1", "l2_n_pairs_k1", "gaps_s", "gap_band",
                "l3_p_handoff", "l3_n_pairs_usable", "l3_n_pairs_dropped", "l3_chef_per_camera",
                "l3_p_simul_agree", "l3_n_simul_frames"]
        with open(ev / "per_clip.csv", "w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            w.writeheader()
            for c in done:
                w.writerow(ms[c["clip_id"]])

        extra[profile] = breakdowns(done, rows_of, ms, topo)
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
    by = {(r["profile"], r["scope"], r["rule"], r["seq_group"]): r for r in all_rows}
    name = {"single": "單人", "multi": "多人"}

    # ── 精簡並排 ──
    def frac(r, key):
        return "—" if not r[f"{key}_n"] else f"{pct(r[key])} ({r[f'{key}_k']}/{r[f'{key}_n']})"

    lines = ["# 六個子集 × 各組設定(逐片段獨立推論,stride 5)", ""]
    for sg, title in (("all", "全部序列"), ("eval", "只含評估集序列"), ("derive", "只含推導集序列")):
        lines += [f"## {title}", ""]
        for scope in ("single", "multi"):
            for rule in RULES:
                rs = [by.get((p, scope, rule, sg)) for p in args.profiles]
                if not any(rs):
                    continue
                r0 = next(r for r in rs if r)
                lines += [f"### {name[scope]} {rule}"
                          f"({r0['n_clips']} 筆,目標在場 {r0['n_target_cells']} 格)", "",
                          "| 設定 | 召回 | 連續率 | 排他率 | 切換 | 誤偵框 | L2 再取得 k5 | L3 跨鏡頭一致 | L3 同時幀一致 |",
                          "|---|---|---|---|---|---|---|---|---|"]
                for p, r in zip(args.profiles, rs):
                    if r:
                        lines.append(
                            f"| {p} | {pct(r['recall'])} | "
                            f"{pct(r['continuity'])} ({r['continuity_k']}/{r['n_bound_cells']}) | "
                            f"{frac(r, 'exclusive')} | {r['chef_switches']} | "
                            f"{pct(r['ghost_box_rate'])} | {frac(r, 'reacq_k5')} | "
                            f"{frac(r, 'handoff')} | {frac(r, 'simul')} |")
                lines.append("")
    (out_root / "summary_all.md").write_text("\n".join(lines), encoding="utf-8")

    # ── 完整指標 ──
    full = ["# 六個子集 × 各組設定:完整指標", "",
            "- 逐片段獨立推論,stride 5;率一律彙集逐格後計算,方括號為 Wilson 95% 區間。",
            "- 誤偵框與誤偵 track 以時間窗去重(多人 L1 同一個窗有多個目標)。",
            "- 指標定義見 `src/common/clip_metrics.py` 檔頭與 "
            "`docs/六子集測試_預先登記_20261013.md` §4。", ""]
    for sg, title in (("all", "全部序列"), ("eval", "只含評估集序列"),
                      ("derive", "只含推導集序列")):
        full += [f"## {title}", ""]
        for scope in ("single", "multi"):
            for rule in RULES:
                rs = [by.get((p, scope, rule, sg)) for p in args.profiles]
                if not any(rs):
                    continue
                r0 = next(r for r in rs if r)
                full += [f"### {name[scope]} {rule}({r0['n_clips']:,} 筆、{r0['n_runs']:,} 個時間窗、"
                         f"目標在場 {r0['n_target_cells']:,} 格)", "",
                         "| 指標 | " + " | ".join(f"`{p}`" for p in args.profiles) + " |",
                         "|---" * (len(args.profiles) + 1) + "|"]
                for label, fn in FULL_ROWS:
                    if fn is None:
                        full.append(f"| {label} |" + " |" * len(args.profiles))
                        continue
                    vals = [fn(r) if r else "—" for r in rs]
                    # ⚠ clip_metrics.ghost_stats 的「誤偵 track」只看有沒有配到**目標**,
                    #   多人片段裡其他真人的 track 也會被算成誤偵 → 多人不報這一列。
                    if scope == "multi" and label == "誤偵 track 比例":
                        vals = ["不適用"] * len(vals)
                    if all(v in ("—", "不可量測") for v in vals) and "L2" in label + "" \
                            and rule != "L2":
                        continue
                    if rule not in ("L3S", "L3T") and label in (
                            "跨鏡頭一致率", "排除的鏡頭對(某台沒有編號)", "同時幀一致率"):
                        continue
                    full.append(f"| {label} | " + " | ".join(vals) + " |")
                full.append("")
    P = args.profiles
    hdr = " | ".join(f"`{p}`" for p in P)
    sep = "|---" * (len(P) + 1) + "|"
    full += ["## 分解(全部序列)", "",
             "### L2 再取得率依間隔長短", "",
             "- within_buffer:離開不超過 5 秒(M4 的跟丟緩衝內);beyond_buffer:超過 5 秒。", "",
             f"| 範圍 | 間隔帶 | 段數 | {hdr} |", "|---|---|---" + "|---" * len(P) + "|"]
    for scope in ("single", "multi"):
        for band in ("within_buffer", "beyond_buffer"):
            n = extra[P[0]].get(("gap", scope, band, "n_clips"), (0, 0))[1]
            if n:
                full.append(f"| {name[scope]} | {band} | {n} | "
                            + " | ".join(kn(extra[p], ("gap", scope, band, "reacq")) for p in P) + " |")
    full += ["", "### L3 跨鏡頭一致率依鏡頭對的路徑", "",
             f"- 路徑依 `{args.eval_topology}` 分類,四組設定共用同一份分類"
             "(`delivery` 自己的拓撲沒有列入其中四個重疊對)。",
             "- 重疊:兩台視野重疊;轉場連結:拓撲有列出走過去的時間;未建模:兩者皆無。", "",
             f"| 範圍 | 規則 | 路徑 | 鏡頭對 | 排除 | {hdr} |", "|---|---|---|---|---" + "|---" * len(P) + "|"]
    for scope in ("single", "multi"):
        for rule in ("L3S", "L3T"):
            for path in PATHS:
                n = extra[P[0]].get(("path", scope, rule, path, "pairs"), (0, 0))[1]
                if n:
                    dr = "/".join(str(extra[p].get(("path", scope, rule, path, "dropped"), (0, 0))[1]) for p in P)
                    full.append(f"| {name[scope]} | {rule} | {path} | {n} | {dr} | "
                                + " | ".join(kn(extra[p], ("path", scope, rule, path, "handoff")) for p in P) + " |")
    full += ["", "### 依是否牽涉玻璃倒影鏡頭(camera_5、camera_6)", ""]
    for metric, title in (("recall", "目標召回率"), ("continuity", "連續率"), ("exclusive", "排他率"),
                          ("reacq", "L2 再取得率"), ("handoff", "L3 跨鏡頭一致率")):
        full += [f"#### {title}", "", f"| 子集 | 倒影 | 筆數 | {hdr} |", "|---|---|---" + "|---" * len(P) + "|"]
        for scope in ("single", "multi"):
            for rule in RULES:
                for mir in ("不含倒影鏡頭", "含倒影鏡頭"):
                    n = extra[P[0]].get(("mirror", scope, rule, mir, "n_clips"), (0, 0))[1]
                    vals = [kn(extra[p], ("mirror", scope, rule, mir, metric)) for p in P]
                    if n and any(v != "—" for v in vals):
                        full.append(f"| {name[scope]} {rule} | {mir} | {n} | " + " | ".join(vals) + " |")
        full.append("")
    (out_root / "metrics_full.md").write_text("\n".join(full), encoding="utf-8")
    print(f"已存 {out_root / 'summary_all.md'}、{out_root / 'metrics_full.md'}")


if __name__ == "__main__":
    main()
