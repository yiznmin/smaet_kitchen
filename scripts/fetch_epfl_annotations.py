"""
下載 EPFL 某場次的動作標註 + 中繼資料(不下載整包)。

來源有兩個 zip,都用 HTTP Range 只抓需要的檔:
  - 標註包 Zenodo 15551913 `Public_release_pose.zip`(12.6 GB):
      annotations/action_annotations.xlsx(Start/End 秒、Verbs、Nouns、Confusion)
      annotations/activity_annotations.json(高階活動段)
  - 影片包 Zenodo 15535461 `Public_release_videos.zip`(192 GB):
      meta_data/*(camera_matrix.json、timestamps.txt…)、IMUs/*(刀、鍋鏟的 IMU)

授權:EPFL 為 CC BY-NC 4.0(禁商用),僅內部驗證,不可交付、不可進報告;須引用 arXiv 2506.01608。

用法:
  python scripts/fetch_epfl_annotations.py                       # 預設場次 = configs/benchmark.yaml 的 default_sample
  python scripts/fetch_epfl_annotations.py --session YH2040/2023_09_12_09_50_41 --pose
"""
import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from fetch_sample import read_central_directory, extract_one   # noqa: E402

POSE_ZIP = "https://zenodo.org/records/15551913/files/Public_release_pose.zip?download=1"


def main():
    cfg = yaml.safe_load((ROOT / "configs" / "benchmark.yaml").read_text(encoding="utf-8"))
    video_zip = cfg["dataset"]["zip_url"]
    # Public_release_videos/train/YH2040/2023_09_12_09_50_41/videos/Boutput0.mp4 -> YH2040/2023_09_12_09_50_41
    default = "/".join(cfg["dataset"]["default_sample"].split("/")[2:4])

    ap = argparse.ArgumentParser()
    ap.add_argument("--session", default=default, help="<participant>/<session>")
    ap.add_argument("--out", default=str(ROOT / "data" / "epfl_ann"))
    ap.add_argument("--pose", action="store_true", help="一併下載 pose_3d(數百 MB)")
    args = ap.parse_args()

    out = Path(args.out) / args.session
    jobs = [(POSE_ZIP, "annotations/"), (video_zip, "meta_data/"), (video_zip, "IMUs/")]
    if args.pose:
        jobs.append((POSE_ZIP, "pose_3d/"))

    cache = {}
    for url, sub in jobs:
        if url not in cache:
            print(f"讀取 central directory:{url}")
            cache[url] = read_central_directory(url)[1]
        hits = [e for e in cache[url]
                if f"/{args.session}/{sub}" in e["name"] and not e["name"].endswith("/")]
        if not hits:
            print(f"  ⚠ 找不到 {args.session}/{sub}")
        for e in hits:
            rel = e["name"].split(f"/{args.session}/", 1)[1]
            n = extract_one(url, e, out / rel)
            print(f"  {rel}  {n/1e6:.2f} MB")
    print(f"完成 → {out}")


if __name__ == "__main__":
    main()
