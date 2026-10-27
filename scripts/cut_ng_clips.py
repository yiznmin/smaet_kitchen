"""依團隊 ng_clips_index.csv 的起訖時間,自己從 EPFL 原始影片切 NG 片段(train 場次 + 團隊檔案讀不到的)。

預先登記:docs/M6_NG準則評估_預先登記_20261027.md §8.2
- 影片:data/epfl_sessions/<參與者>/<場次>/<鏡頭>.mp4(缺的會下載;不進 git)
- 片段:data/ng_clips_cut/<片段>/<鏡頭>.mp4(起點對齊整數幀;不進 git)
- 快取:results/det_cache/ng_clips_s3/<片段>/<鏡頭>.npz(和團隊片段同一個資料夾結構)
片段名稱 = <參與者>_<場次>_<ngNNN>,與 cache_ng_clips.py 相同。

用法:
    python scripts/cut_ng_clips.py
"""
import csv
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from fetch_sample import extract_one, read_central_directory   # noqa: E402
from fetch_ng_windows import FPS, URL, VID_ROOT, VIEWS, cache  # noqa: E402

NG_DIR = Path("D:/yizhen/NG_clips")
CUT = ROOT / "data/ng_clips_cut"
CACHE = ROOT / "results/det_cache/ng_clips_s3"
EXCLUDE = {"test/YH2003/2023_05_17_09_08_58"}


def readable(p):
    return p.exists() and cv2.VideoCapture(str(p)).isOpened()


def main():
    clips = defaultdict(dict)
    team_ok = defaultdict(set)
    for r in csv.DictReader(open(NG_DIR / "ng_clips_index.csv", encoding="utf-8-sig")):
        if r["session_id"] in EXCLUDE:
            continue
        pid, sess = r["session_id"].split("/")[1:]
        no = re.search(r"_(ng\d+)_", r["clip_file"]).group(1)
        k = f"{pid}_{sess}_{no}"
        clips[k] = dict(session=r["session_id"], start=float(r["clip_start_sec"]), end=float(r["clip_end_sec"]))
        if readable(NG_DIR / r["clip_file"]):
            team_ok[k].add(r["view"])
    # 團隊影片 7 台都可讀的片段不用切
    todo = {k: c for k, c in clips.items() if not set(VIEWS) <= team_ok[k]}
    sessions = sorted({c["session"] for c in todo.values()})
    print(f"片段 {len(clips)}、要自己切 {len(todo)}、涉及場次 {len(sessions)}", flush=True)

    entries = None
    for s in sessions:
        for v in VIEWS:
            dest = VID_ROOT / s.split("/", 1)[1] / f"{v}.mp4"
            if dest.exists():
                continue
            if entries is None:
                entries = {e["name"]: e for e in read_central_directory(URL)[1]}
            e = entries.get(f"Public_release_videos/{s}/videos/{v}.mp4")
            if e is None:
                print(f"  找不到 {s} {v}", flush=True)
                continue
            n = extract_one(URL, e, dest)
            print(f"  下載 {s} {v} {n / 1e6:.0f} MB", flush=True)

        for k, c in sorted(todo.items()):
            if c["session"] != s:
                continue
            f0 = int(c["start"] * FPS)                   # 起點對齊整數幀(同 fetch_ng_windows.py)
            for v in VIEWS:
                src, out = VID_ROOT / s.split("/", 1)[1] / f"{v}.mp4", CUT / k / f"{v}.mp4"
                if out.exists() or not src.exists() or (CACHE / k / f"{v}.npz").exists():
                    continue
                out.parent.mkdir(parents=True, exist_ok=True)
                subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{(f0 - 0.5) / FPS:.4f}", "-i", str(src),
                                "-t", f"{c['end'] - f0 / FPS:.3f}", "-an", "-c:v", "libx264", "-preset", "veryfast",
                                "-crf", "18", "-r", "30", str(out)], check=True)
            vs = [v for v in VIEWS if v not in team_ok[k]]
            cache([CUT / k / f"{v}.mp4" for v in vs], vs, CACHE / k)
    print("全部完成", flush=True)


if __name__ == "__main__":
    main()
