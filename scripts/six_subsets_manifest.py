"""六個子集(單人 / 多人 × L1 / L2 / L3)的片段清單 —— 只讀 CSV 與標註。

預先登記:docs/六子集測試_預先登記_20261013.md §2。

與 `build_clip_manifest.py` 的差別(那一支是 9/23 預先登記的產物,不改它):
  · 六份 CSV 全收。單人 L1 用**完整** `level1_single_cam_continuous.csv`(178 段),
    不是抽樣的 `level1_few_sample.csv`(34 段)。
  · 多人 L1 一個窗裡每個人都是目標 → 一個窗產生多筆「片段 × 目標」,共用同一次執行。
    所以每筆多一個 `run_id`(= 序列 + 鏡頭 + 窗),執行以 run_id 為單位、評估以 clip_id 為單位。
  · 影片資料夾名稱可指定(2026-09-28 資料集把 videos/ 改名)。
  · 自檢不過的片段**剔除並列出原因**,不中止 —— 多人清單有 807 列,不該因為一列有問題整份作廢。

對齊規則、鏡頭取法、`exclusive` 由標註重算,全部沿用 build_clip_manifest 的函式,不另寫。

用法:
    python scripts/six_subsets_manifest.py --out results/six_subsets/manifest.json
"""
import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from build_clip_manifest import GRID, FPS, gt_facts, rule_of, topo_subset   # noqa: E402
from eval_m4m5_chirla import load_gt                                        # noqa: E402
from levels_from_csv import align_in, cam_spans, spans                      # noqa: E402

# (範圍, 等級, 檔名)
CSV_FILES = (("single", "L1", "level1_single_cam_continuous.csv"),
             ("single", "L2", "level2_single_cam_reappear.csv"),
             ("single", "L3", "level3_cross_camera.csv"),
             ("multi", "L1", "multi_level1_single_cam_continuous.csv"),
             ("multi", "L2", "multi_level2_single_cam_reappear.csv"),
             ("multi", "L3", "multi_level3_cross_camera.csv"))
DERIVE_SEQS = ("seq_000", "seq_001", "seq_002")   # 拓撲時間參數與平面圖 σ 的推導集


def read_csv(path):
    with open(path, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def cams_of(level, row):
    if level in ("L1", "L2"):
        return [row["camera"]]
    return sorted({c for c, _a, _b in cam_spans(row["segments"])})


def targets_of(scope, level, row):
    """多人 L1 的窗內每個人都是目標;其餘是 person_id 那一位。"""
    if scope == "multi" and level == "L1":
        return [int(p) for p in row["person_ids"].split()]
    return [int(row["person_id"])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="D:/yizhen/CHIRLA/CHIRLA_data/CHIRLA")
    ap.add_argument("--csv-dir", default="D:/yizhen/CHIRLA/CHIRLA_data/CHIRLA")
    ap.add_argument("--video-dir", default="clips_singal_person_result")
    ap.add_argument("--topology", default="configs/fix_grid/base.yaml")
    ap.add_argument("--det-cache-root", default="results/det_cache/coco_nano")
    ap.add_argument("--out", default="results/six_subsets/manifest.json")
    args = ap.parse_args()

    with open(ROOT / args.topology, encoding="utf-8") as f:
        topo = (yaml.safe_load(f) or {}).get("camera_topology", {})
    gts, clips, dropped, vids_cache = {}, [], [], {}

    def videos_for(seq, cams):
        out = {}
        for c in cams:
            if (seq, c) not in vids_cache:
                hits = sorted((Path(args.root) / args.video_dir / seq).glob(f"{c}_*.avi"))
                vids_cache[(seq, c)] = str(hits[0]).replace("\\", "/") if len(hits) == 1 else None
            out[c] = vids_cache[(seq, c)]
        return out

    for scope, level, name in CSV_FILES:
        for i, r in enumerate(read_csv(Path(args.csv_dir) / name)):
            seq, cams = r["seq"], cams_of(level, r)
            csv_lo, csv_hi = int(r["start_frame"]), int(r["end_frame"])
            lo, hi = align_in(csv_lo, csv_hi, GRID)
            rule = rule_of(level, r)
            cam_tag = "+".join(c.replace("camera_", "c") for c in cams)
            run_id = f"{seq}-{cam_tag}-{lo - 1:06d}-{hi - 1:06d}"
            tag = f"{scope}:{name}#{i}"
            if hi <= lo:
                dropped.append(f"{tag}:對齊後窗長度 <= 0")
                continue
            if seq not in gts:
                gts[seq] = load_gt(args.root, seq)
            vids = videos_for(seq, cams)
            if any(v is None for v in vids.values()):
                dropped.append(f"{tag}:影片不是剛好 1 支")
                continue
            if any(not (ROOT / args.det_cache_root / seq / f"{c}.npz").exists() for c in cams):
                dropped.append(f"{tag}:缺偵測快取")
                continue
            for gid in targets_of(scope, level, r):
                facts = gt_facts(gts[seq], cams, gid, lo, hi)
                empty = [c for c, n in facts["target_cells_per_cam"].items() if n == 0]
                if empty:
                    dropped.append(f"{tag} 目標 {gid}:在 {empty} 的窗內沒有任何標註框")
                    continue
                clip = dict(
                    clip_id=f"{scope[0]}{rule}-{seq}-{cam_tag}-{gid}-{lo - 1:06d}",
                    run_id=run_id, scope=scope, level=level, rule=rule, seq=seq,
                    seq_group="derive" if seq in DERIVE_SEQS else "eval",
                    cameras=cams, gt_id=gid,
                    start_frame=lo, end_frame=hi, start_fid=lo - 1, end_fid=hi - 1,
                    duration_s=round((hi - lo) / FPS, 3),
                    csv_file=name, csv_index=i,
                    csv_start_frame=csv_lo, csv_end_frame=csv_hi,
                    videos=vids, det_cache=f"{args.det_cache_root}/{seq}",
                    **facts, **topo_subset(topo, cams))
                if level == "L2":
                    ap_ = spans(r["appearances"])
                    gaps = [round((b[0] - a[1]) / FPS, 3) for a, b in zip(ap_, ap_[1:])]
                    clip.update(appearances=[list(x) for x in ap_], gaps_s=gaps,
                                gap_band=("within_buffer" if gaps and max(gaps) <= 5.0
                                          else "beyond_buffer"))
                if level == "L3":
                    if rule == "L3?":
                        dropped.append(f"{tag}:未知的 handoff_type")
                        continue
                    clip.update(segments=[list(x) for x in cam_spans(r["segments"])],
                                handoff_type=r["handoff_type"])
                clips.append(clip)

    dup = [k for k, n in Counter(c["clip_id"] for c in clips).items() if n > 1]
    if dup:      # 同一人、同一組鏡頭、同一個起點出現兩次 → 只留第一筆
        seen, keep = set(), []
        for c in clips:
            if c["clip_id"] in seen:
                dropped.append(f"{c['scope']}:{c['csv_file']}#{c['csv_index']}:clip_id 重複")
                continue
            seen.add(c["clip_id"])
            keep.append(c)
        clips = keep

    counts = Counter((c["scope"], c["rule"]) for c in clips)
    runs = {c["run_id"] for c in clips}
    out = dict(source=[f for _s, _l, f in CSV_FILES], root=args.root, video_dir=args.video_dir,
               topology=args.topology, grid=GRID, fps=FPS, argv=sys.argv[1:],
               counts={f"{s}/{r}": n for (s, r), n in sorted(counts.items())},
               n_clips=len(clips), n_runs=len(runs), dropped=dropped, clips=clips)
    path = ROOT / args.out
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print("片段 × 目標:", out["counts"])
    print(f"共 {len(clips)} 筆、{len(runs)} 次獨立執行;剔除 {len(dropped)} 筆")
    print("獨佔 / 非獨佔:", Counter((c["scope"], c["exclusive"]) for c in clips))
    print("推導集 / 評估集:", Counter((c["scope"], c["seq_group"]) for c in clips))
    for d in dropped[:15]:
        print("  剔除:", d)
    print(f"已存 {path}")


if __name__ == "__main__":
    main()
