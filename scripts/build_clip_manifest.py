"""把三份 Level CSV 轉成「逐片段獨立推論」的片段清單(只讀標註,不看任何追蹤輸出)。

預先登記:`docs/難度分級_逐片段獨立推論_預先登記_20260923.md` §2。

## 與 9/17 的 B 套差在哪

B 套是「整段序列跑一次,再截時間窗算指標」;本輪是「每段自己跑一次,只給該段指名的鏡頭」。
所以這份清單除了窗的範圍,還要記錄**執行需要的東西**:影片路徑、偵測快取路徑、
該段實際拿得到的拓撲子集大小。

## ⚠ 定死的規則(跑之前)

- **只讀 CSV 與標註。** 難度必須是影片的性質,不能是受測系統的函數
  (理由同 `chirla_select_levels.py` 檔頭)。
- **兩個 stride 共用同一個窗**:對齊到 `(frame − 1) % 5 == 0`(= lcm(1,5)),窗只縮不擴。
  不這樣做的話 stride 1 與 stride 5 量的是不同片段,差異表就沒有意義。
- **L3 的鏡頭取自 `segments` 欄**,不是 `camera_order`、也不是 `brief_cams`
  (那兩欄只記錄下來對照)。
- `exclusive` / `n_other_ids` / 目標在各鏡頭的在場幀數**一律由標註重算**,不信 CSV 欄位。

用法:
    python scripts/build_clip_manifest.py --root "D:/yizhen/CHIRLA/CHIRLA_data/CHIRLA" \
        --csv-dir "D:/yizhen/CHIRLA/CHIRLA_data/CHIRLA" \
        --det-cache-root results/det_cache/coco_nano \
        --out results/clips/clip_manifest.json
"""
import argparse
import csv
import json
import subprocess
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from eval_m4m5_chirla import load_gt                       # noqa: E402
from levels_from_csv import align_in, cam_spans, spans     # noqa: E402

FPS = 30.0
GRID = 5                    # lcm(1, 5):兩個 stride 必須用同一個窗
CSV_FILES = {"L1": "level1_few_sample.csv",
             "L2": "level2_single_cam_reappear.csv",
             "L3": "level3_cross_camera.csv"}


def read_csv(path):
    with open(path, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def sampled(lo, hi, stride):
    """窗內的取樣真值幀(1-based)。與 chirla_select_levels.sampled_frames 同一條規則。"""
    first = lo + ((1 - lo) % stride)
    return range(first, hi + 1, stride)


def cams_of(level, row):
    if level in ("L1", "L2"):
        return [row["camera"]]
    return sorted({c for c, _a, _b in cam_spans(row["segments"])})


def rule_of(level, row):
    if level != "L3":
        return level
    return {"overlapping": "L3S", "sequential": "L3T"}.get(row["handoff_type"], "L3?")


def videos_for(root, seq, cams):
    """回傳 {鏡頭: 影片路徑}。缺任何一台就是錯誤,不靜默略過。"""
    out = {}
    for c in cams:
        hits = sorted((Path(root) / "videos" / seq).glob(f"{c}_*.avi"))
        if len(hits) != 1:
            raise SystemExit(f"{seq}/{c}:找到 {len(hits)} 支影片,預期剛好 1 支")
        out[c] = str(hits[0]).replace("\\", "/")
    return out


def gt_facts(gt, cams, gid, lo, hi):
    """由標註重算窗內事實。分母用取樣格(stride 5 的格點,兩個 stride 共用)。"""
    frames = list(sampled(lo, hi, GRID))
    per_cam, others = {}, set()
    for c in cams:
        g = gt.get(c, {})
        n_target = 0
        for f in frames:
            ids = {i for i, _b in g.get(f, [])}
            n_target += int(gid in ids)
            others |= ids - {gid}
        per_cam[c] = n_target
    return dict(n_sampled_cells=len(frames) * len(cams),
                n_target_cells=sum(per_cam.values()),
                target_cells_per_cam=per_cam,
                exclusive=not others, n_other_ids=len(others),
                other_ids=sorted(others))


def topo_subset(topo, cams):
    """這段實際拿得到多少拓撲:2 台的片段看不到中間那台,原本 A→X→B 會變 A→B。"""
    s = set(cams)
    links = [l for l in topo.get("links", [])
             if l["from"] in s and l["to"] in s]
    ov = [p for p in topo.get("overlapping", []) if set(p) <= s]
    return dict(n_links_in_subset=len(links),
                n_overlapping_pairs_in_subset=len(ov),
                n_pairs=len(list(combinations(sorted(s), 2))),
                links_in_subset=[[l["from"], l["to"]] for l in links],
                overlapping_in_subset=[sorted(p) for p in ov])


def build(args):
    with open(args.topology, encoding="utf-8") as f:
        topo = (yaml.safe_load(f) or {}).get("camera_topology", {})
    gts, clips, bad = {}, [], []

    for level, name in CSV_FILES.items():
        for i, r in enumerate(read_csv(Path(args.csv_dir) / name)):
            seq, gid = r["seq"], int(r["person_id"])
            cams = cams_of(level, r)
            csv_lo, csv_hi = int(r["start_frame"]), int(r["end_frame"])
            lo, hi = align_in(csv_lo, csv_hi, GRID)
            if hi <= lo:
                bad.append(f"{level}#{i}:對齊後窗長度 <= 0")
                continue
            if seq not in gts:
                gts[seq] = load_gt(args.root, seq)
            facts = gt_facts(gts[seq], cams, gid, lo, hi)
            vids = videos_for(args.root, seq, cams)
            rule = rule_of(level, r)
            cam_tag = "+".join(c.replace("camera_", "c") for c in cams)
            clip = dict(
                clip_id=f"{rule}-{seq}-{cam_tag}-{gid}-{lo - 1:06d}",
                level=level, rule=rule, seq=seq, cameras=cams, gt_id=gid,
                start_frame=lo, end_frame=hi,               # 1-based 標註幀
                start_fid=lo - 1, end_fid=hi - 1,           # 0-based,tracks.csv 的鍵
                t_start_s=round((lo - 1) / FPS, 3), t_end_s=round((hi - 1) / FPS, 3),
                duration_s=round((hi - lo) / FPS, 3),
                csv_start_frame=csv_lo, csv_end_frame=csv_hi,
                shift_start=lo - csv_lo, shift_end=csv_hi - hi,
                n_loops={s: (hi - 1 - (lo - 1)) // s + 1 for s in (1, 5)},
                videos=vids,
                det_cache=f"{args.det_cache_root}/{seq}",
                csv_row=dict(level=level, index=i, **r),
                **facts, **topo_subset(topo, cams))
            if level == "L2":
                ap = spans(r["appearances"])
                gaps = [round((b[0] - a[1]) / FPS, 3) for a, b in zip(ap, ap[1:])]
                clip.update(appearances=[list(x) for x in ap], gaps_s=gaps,
                            gap_band=("within_buffer" if gaps and max(gaps) <= 5.0
                                      else "beyond_buffer"))
            if level == "L3":
                clip.update(segments=[list(x) for x in cam_spans(r["segments"])],
                            camera_order=r["camera_order"],
                            brief_cams=r["brief_cams"],
                            handoff_type=r["handoff_type"])
            clips.append(clip)

    # ── 自檢(V1 的一部分,先在這裡擋掉) ──
    ids = Counter(c["clip_id"] for c in clips)
    bad += [f"{k}:clip_id 重複 {n} 次" for k, n in ids.items() if n > 1]
    for c in clips:
        if (c["start_frame"] - 1) % GRID:
            bad.append(f"{c['clip_id']}:起始幀不在共用格點上")
        for cam, n in c["target_cells_per_cam"].items():
            if n == 0:
                bad.append(f"{c['clip_id']}:目標在 {cam} 的窗內沒有任何標註框")
        for cam in c["cameras"]:
            if not Path(f"{c['det_cache']}/{cam}.npz").exists():
                bad.append(f"{c['clip_id']}:缺偵測快取 {cam}")
        if c["rule"] == "L3?":
            bad.append(f"{c['clip_id']}:未知的 handoff_type")

    sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                         text=True, cwd=ROOT).stdout.strip()
    counts = Counter(c["rule"] for c in clips)
    out = dict(source=CSV_FILES, csv_dir=str(args.csv_dir), root=str(args.root),
               topology=str(args.topology), grid=GRID, fps=FPS, git_sha=sha,
               argv=sys.argv[1:], counts=dict(counts), n_clips=len(clips),
               selfcheck_failures=bad, clips=clips)

    print("每個規則的段數:", dict(counts))
    print("對齊位移(起點/終點, 幀):",
          Counter((c["shift_start"], c["shift_end"]) for c in clips))
    print("獨佔的段:", sum(c["exclusive"] for c in clips), "/", len(clips))
    print("拓撲子集有連結的段:", sum(c["n_links_in_subset"] > 0 for c in clips),
          ";有重疊對的段:", sum(c["n_overlapping_pairs_in_subset"] > 0 for c in clips))

    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"已存 {path}")
    if bad:
        print(f"[FAIL] 自檢 {len(bad)} 項:")
        for b in bad[:20]:
            print(f"  - {b}")
        return 1
    print("[OK] 自檢全部通過")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="CHIRLA 根目錄(讀標註與影片)")
    ap.add_argument("--csv-dir", required=True, help="三份 Level CSV 所在目錄")
    ap.add_argument("--topology", default="configs/fix_grid/base.yaml")
    ap.add_argument("--det-cache-root", default="results/det_cache/coco_nano")
    ap.add_argument("--out", default="results/clips/clip_manifest.json")
    return build(ap.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
