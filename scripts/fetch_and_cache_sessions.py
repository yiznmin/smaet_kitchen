"""違規先後順序測試:下載 EPFL 完整場次(7 台)並建立前 N 分鐘的 M3 全類別偵測快取。

預先登記:docs/M6_違規先後順序_預先登記_20261027.md
影片存 data/epfl_sessions/<參與者>/<場次>/<鏡頭>.mp4(本機,不進 git;EPFL CC BY-NC 4.0 僅內部驗證)
快取存 results/det_cache/epfl_sessions_s3/<參與者>_<場次>/<鏡頭>.npz
已存在的影片 / 快取會跳過;偵測 CUDA 偶發錯誤時每台重試最多 3 次。

用法:
    python scripts/fetch_and_cache_sessions.py --minutes 20
"""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from fetch_sample import extract_one, read_central_directory   # noqa: E402

URL = "https://zenodo.org/records/15535461/files/Public_release_videos.zip?download=1"
SESSIONS = {                                      # 開發 5 場 + 測試 3 場(登記 §2)
    "dev": ["train/YH2018/2023_05_31_09_07_02", "train/YH2025/2023_08_04_09_01_53",
            "train/YH2043/2023_10_09_09_06_48", "train/YH2002/2023_12_11_16_11_15",
            "train/YH2040/2023_08_25_09_12_12"],
    "test": ["test/YH2029/2023_06_12_15_05_02", "test/YH2011/2023_07_10_15_03_05",
             "test/YH2004/2023_07_12_14_02_36"],
}
VIEWS = ["output0", "Aoutput0", "Aoutput1", "Aoutput2", "Aoutput3", "Boutput1", "Boutput2"]
WEIGHTS = ROOT / "integrated/weights/m3_nano_r2_armA/checkpoint_best_total.pth"
VID_ROOT = ROOT / "data/epfl_sessions"
CACHE_ROOT = ROOT / "results/det_cache/epfl_sessions_s3"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=20)
    ap.add_argument("--skip-download", action="store_true")
    args = ap.parse_args()
    sessions = SESSIONS["dev"] + SESSIONS["test"]

    if not args.skip_download:
        print("讀取 zip central directory …", flush=True)
        entries = {e["name"]: e for e in read_central_directory(URL)[1]}
        for s in sessions:
            for v in VIEWS:
                dest = VID_ROOT / s.split("/", 1)[1] / f"{v}.mp4"
                if dest.exists():
                    continue
                e = entries.get(f"Public_release_videos/{s}/videos/{v}.mp4")
                if e is None:
                    print(f"  找不到 {s} {v}", flush=True)
                    continue
                n = extract_one(URL, e, dest)
                print(f"  下載 {s} {v} {n / 1e6:.0f} MB", flush=True)

    max_frames = int(args.minutes * 60 * 30)
    py = str(ROOT / ".venv/Scripts/python")
    for s in sessions:
        sid = s.split("/", 1)[1]
        out = CACHE_ROOT / sid.replace("/", "_")
        for v in VIEWS:
            vid = VID_ROOT / sid / f"{v}.mp4"
            if (out / f"{v}.npz").exists() or not vid.exists():
                continue
            for attempt in range(3):
                cmd = [py, str(ROOT / "scripts/cache_detections.py"), "--all-classes", "--weights", str(WEIGHTS),
                       "--out-dir", str(out), "--stride", "3", "--max-frames", str(max_frames),
                       "--spot-check", "5", "--videos", str(vid), "--cameras", v]
                r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")
                ok = r.returncode == 0 and (out / f"{v}.npz").exists()
                print(f"  偵測 {sid} {v} 第 {attempt + 1} 次:{'完成' if ok else '失敗'} "
                      f"{r.stdout.strip().splitlines()[-1] if r.stdout.strip() else ''} {'' if ok else r.stderr[-300:]}",
                      flush=True)
                if ok:
                    break
    print("全部完成", flush=True)


if __name__ == "__main__":
    main()
