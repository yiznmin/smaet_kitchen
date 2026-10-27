"""團隊 NG 片段(D:/yizhen/NG_clips)的 M3 全類別偵測快取。

預先登記:docs/M6_NG準則評估_預先登記_20261027.md §8.1
快取存 results/det_cache/ng_clips_s3/<參與者>_<場次>_<ngNNN>/<鏡頭>.npz(不進 git)
只快取讀得到的影片;已存在的跳過。

用法:
    python scripts/cache_ng_clips.py
"""
import csv
import re
import subprocess
from collections import defaultdict
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
NG_DIR = Path("D:/yizhen/NG_clips")
OUT = ROOT / "results/det_cache/ng_clips_s3"
WEIGHTS = ROOT / "integrated/weights/m3_nano_r2_armA/checkpoint_best_total.pth"


def clip_key(row):
    pid, sess = row["session_id"].split("/")[1:]
    no = re.search(r"_(ng\d+)_", row["clip_file"]).group(1)
    return f"{pid}_{sess}_{no}"


def main():
    jobs = defaultdict(list)
    for r in csv.DictReader(open(NG_DIR / "ng_clips_index.csv", encoding="utf-8-sig")):
        p = NG_DIR / r["clip_file"]
        if p.exists() and cv2.VideoCapture(str(p)).isOpened():
            jobs[clip_key(r)].append((p, r["view"]))
    print(f"片段 {len(jobs)}、影片 {sum(map(len, jobs.values()))}", flush=True)
    for k, vs in sorted(jobs.items()):
        out = OUT / k
        todo = [(p, v) for p, v in vs if not (out / f"{v}.npz").exists()]
        if not todo:
            continue
        cmd = [str(ROOT / ".venv/Scripts/python"), str(ROOT / "scripts/cache_detections.py"), "--all-classes",
               "--weights", str(WEIGHTS), "--out-dir", str(out), "--stride", "3", "--spot-check", "5",
               "--videos", *[str(p) for p, _ in todo], "--cameras", *[v for _, v in todo]]
        r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")
        ok = all((out / f"{v}.npz").exists() for _, v in todo)
        print(f"  {k}:{len(todo)} 台 {'完成' if ok else '失敗 ' + r.stderr[-300:]}", flush=True)
    print("全部完成", flush=True)


if __name__ == "__main__":
    main()
