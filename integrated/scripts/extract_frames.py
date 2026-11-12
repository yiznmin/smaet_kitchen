"""從影片抽出要標註的畫面(訓練 M3 用)。

每支影片均勻切成 n 段,每段挑一張有動作的畫面(避開空景與重複畫面),存成 jpg。
建議每台鏡頭、不同時段 / 不同廚師的影片都抽一些,讓訓練資料涵蓋各種角度與情境。

用法:
  python scripts/extract_frames.py --videos <a.mp4> <b.mp4> --n 50 --out datasets/kitchen/images
輸出檔名:<影片檔名>_<幀號>.jpg
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from common.video_io import imwrite_unicode, iter_frames, video_meta   # noqa: E402
from m2_motion.detector import MotionDetector                          # noqa: E402


def extract(video, n, out):
    total = video_meta(video)["nb_frames"]
    seg = max(1, total // n)
    det = MotionDetector()
    saved, done_seg = 0, -1
    for fid, _t, frame in iter_frames(str(video)):
        r = det.process(frame)            # 每幀都要餵,動態判斷才連續
        s = fid // seg
        if s == done_seg or fid < 60 or not r["has_motion"]:   # 前 60 幀是動態偵測暖機
            continue
        if imwrite_unicode(out / f"{Path(video).stem}_{fid:06d}.jpg", frame):
            saved += 1
            done_seg = s
        if saved >= n:
            break
    return saved


def main():
    ap = argparse.ArgumentParser(description="從影片抽出要標註的畫面")
    ap.add_argument("--videos", nargs="+", required=True)
    ap.add_argument("--n", type=int, default=50, help="每支影片抽幾張")
    ap.add_argument("--out", required=True, help="輸出資料夾")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for v in args.videos:
        print(f"{v}:抽出 {extract(v, args.n, out)} 張")
    print(f"輸出:{out}")


if __name__ == "__main__":
    main()
