"""從清單尾端往前預先下載場次影片,與 cut_ng_clips.py 平行跑以縮短下載時間。

先寫 .part 再改名,避免 cut_ng_clips.py 讀到寫一半的檔案。影片不進 git。

用法:
    python scripts/prefetch_sessions.py --workers 2 train/YH2043/2023_10_20_09_05_38 ...
"""
import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from fetch_sample import extract_one, read_central_directory  # noqa: E402
from fetch_ng_windows import URL, VID_ROOT, VIEWS             # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sessions", nargs="+")
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()
    entries = {e["name"]: e for e in read_central_directory(URL)[1]}
    jobs = [(s, v) for s in args.sessions for v in VIEWS
            if not (VID_ROOT / s.split("/", 1)[1] / f"{v}.mp4").exists()]
    print(f"要下載 {len(jobs)} 支", flush=True)

    def one(job):
        s, v = job
        dest = VID_ROOT / s.split("/", 1)[1] / f"{v}.mp4"
        if dest.exists():
            return
        part = dest.with_suffix(".part")
        n = extract_one(URL, entries[f"Public_release_videos/{s}/videos/{v}.mp4"], part)
        if not dest.exists():
            os.replace(part, dest)
        else:
            part.unlink()
        print(f"  預載 {s} {v} {n / 1e6:.0f} MB", flush=True)

    with ThreadPoolExecutor(args.workers) as ex:
        list(ex.map(one, jobs))
    print("全部完成", flush=True)


if __name__ == "__main__":
    main()
