"""逐片段獨立推論的執行網格:67 段 × {stride 5, stride 1},每段一個獨立 process。

預先登記 `docs/難度分級_逐片段獨立推論_預先登記_20260923.md` §4。

## 為什麼是 subprocess 而不是 import 進來跑

`m5_track_video.py` 的主迴圈是順序敏感的(tick → 心跳 → 事件 → 存圖),
而且「每段從零開始」的語意就是**一個新 process**。呼叫同一支腳本能保證
片段執行與交付執行走的是**同一份程式碼**;自己重寫迴圈必然會漂移
(`src/common/draw_tracks.py` 與 `render_level_video.py` 的存在都是這個教訓)。

## 共用參數 = 交付格 m5_cbiou,逐字相同

只有 `--videos` / `--cameras` / `--stride` / `--start-fid` / `--end-fid` / `--out` 隨片段變。
⚠ **不給 `--track-gt`**:片段的 track 編號與整段不同,那份對照表不適用
(`make_track_gt.py` 檔頭已經寫了這件事)。

用法:
    python scripts/run_clip_grid.py --manifest results/clips/clip_manifest.json --dry-run
    python scripts/run_clip_grid.py --manifest results/clips/clip_manifest.json
"""
import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = str(ROOT / ".venv" / "Scripts" / "python.exe")

COMMON = ["--topology", "configs/fix_grid/base.yaml",
          "--tracker", "configs/tracker_rf_cbiou.yaml",
          "--variant", "nano", "--thr", "0.10", "--person-cls", "1",
          "--embedder", "none", "--fps", "30.0", "--max-frames", "-1"]


def cmd_for(clip, stride, out_root, ttl=600, tracker=None):
    out = Path(out_root) / clip["clip_id"] / f"s{stride}" / "chef_events.jsonl"
    cmd = [PY, "scripts/m5_track_video.py",
           "--videos", *[clip["videos"][c] for c in clip["cameras"]],
           "--cameras", *clip["cameras"], *COMMON,
           "--ttl", str(ttl),
           "--stride", str(stride),
           "--start-fid", str(clip["start_fid"]),
           "--end-fid", str(clip["end_fid"]),
           "--det-cache", clip["det_cache"],
           "--clip-id", clip["clip_id"],
           "--out", str(out).replace("\\", "/")]
    if tracker:                       # §4.1 秒數對齊的次要對照(只給 L2)
        i = cmd.index("--tracker")
        cmd[i + 1] = tracker
    return cmd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="results/clips/clip_manifest.json")
    ap.add_argument("--out-root", default="results/clips/runs")
    ap.add_argument("--strides", nargs="+", type=int, default=[5, 1])
    ap.add_argument("--clips", nargs="+", default=None, help="只跑這些 clip_id")
    ap.add_argument("--rules", nargs="+", default=None, help="只跑這些規則(L1/L2/L3S/L3T)")
    ap.add_argument("--ttl-mode", choices=("loops", "seconds"), default="loops",
                    help="loops = 出貨設定(主網格);seconds = §4.1 次要對照")
    ap.add_argument("--index", default="results/clips/agg/run_index.csv")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    man = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    clips = man["clips"]
    if args.clips:
        clips = [c for c in clips if c["clip_id"] in set(args.clips)]
    if args.rules:
        clips = [c for c in clips if c["rule"] in set(args.rules)]
    sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                         text=True, cwd=ROOT).stdout.strip()

    jobs = []
    for c in clips:
        for s in args.strides:
            ttl, tracker = 600, None
            if args.ttl_mode == "seconds" and s != 5:
                # 秒數對齊:TTL 100 秒、跟丟緩衝 5 秒,與 stride 5 相同
                ttl, tracker = 600 * 5 // s, f"configs/tracker_rf_cbiou_s{s}.yaml"
            jobs.append((c, s, ttl, tracker))

    print(f"片段 {len(clips)} 段 × stride {args.strides} → {len(jobs)} 次執行"
          f"(ttl-mode={args.ttl_mode})")
    print("共用參數:", " ".join(COMMON))
    if args.dry_run:
        for c, s, ttl, tracker in jobs:
            print(" ".join(cmd_for(c, s, args.out_root, ttl, tracker)))
        return 0

    rows, t0 = [], time.time()
    for i, (c, s, ttl, tracker) in enumerate(jobs, 1):
        out_dir = Path(args.out_root) / c["clip_id"] / f"s{s}"
        out_dir.mkdir(parents=True, exist_ok=True)
        cmd = cmd_for(c, s, args.out_root, ttl, tracker)
        t = time.time()
        r = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT,
                           encoding="utf-8", errors="replace")
        (out_dir / "stdout.log").write_text((r.stdout or "") + (r.stderr or ""),
                                            encoding="utf-8")
        rows.append(dict(clip_id=c["clip_id"], rule=c["rule"], stride=s, ttl=ttl,
                         tracker=tracker or "configs/tracker_rf_cbiou.yaml",
                         returncode=r.returncode, wall_s=round(time.time() - t, 2),
                         git_sha=sha, argv=" ".join(cmd)))
        flag = "" if r.returncode == 0 else "  ← 失敗"
        print(f"[{i:>3}/{len(jobs)}] {c['clip_id']:<34} s{s} "
              f"{rows[-1]['wall_s']:>6.2f}s{flag}")

    idx = Path(args.index)
    idx.parent.mkdir(parents=True, exist_ok=True)
    with open(idx, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    bad = [r for r in rows if r["returncode"]]
    print(f"\n總耗時 {time.time() - t0:.0f} 秒;失敗 {len(bad)} 次;已存 {idx}")
    for r in bad[:10]:
        print("  失敗:", r["clip_id"], "s%d" % r["stride"])
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
