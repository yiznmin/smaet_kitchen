"""
從 9 個鏡頭視角各抽幀,準備 M3 多視角標註素材。
每個視角:優先挑有動作的幀,全片均勻分散(不雷同)。

輸出:data/m3_finetune/mv_images/<view>_f<fid>.jpg(view 前綴,方便追來源)
用法:
  python scripts/extract_multiview_frames.py            # 每視角 10 張
  python scripts/extract_multiview_frames.py --per 12
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from common.video_io import iter_frames, video_meta, imwrite_unicode   # noqa: E402
from m2_motion.detector import MotionDetector                          # noqa: E402

EPFL = ROOT / "data" / "epfl"
OUT = ROOT / "data" / "m3_finetune" / "mv_images"
VIEWS = ["Aoutput0", "Aoutput1", "Aoutput2", "Aoutput3",
         "Boutput0", "Boutput1", "Boutput2", "Boutput3", "output0"]


def extract_one(view, per):
    video = EPFL / f"{view}.mp4"
    if not video.exists():
        print(f"  跳過(找不到){view}")
        return 0
    total = video_meta(video)["nb_frames"]
    seg = max(1, total // per)
    det = MotionDetector()
    saved, next_seg, picked = 0, 0, False
    for fid, ts, frame in iter_frames(video):
        r = det.process(frame)
        cur = fid // seg
        if cur > next_seg:
            next_seg, picked = cur, False
        if not picked and r["has_motion"] and fid > 60:
            fp = OUT / f"{view}_f{fid:05d}.jpg"
            if imwrite_unicode(fp, frame):
                saved += 1
                picked = True
        if saved >= per:
            break
    print(f"  {view}: {saved} 張")
    return saved


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per", type=int, default=10, help="每視角抽幾張")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"每視角抽 {args.per} 張(有動作、分散):")
    tot = sum(extract_one(v, args.per) for v in VIEWS)
    print(f"\n完成:共 {tot} 張 → {OUT.relative_to(ROOT).as_posix()}/")


if __name__ == "__main__":
    main()
