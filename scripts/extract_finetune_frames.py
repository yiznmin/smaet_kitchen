"""
從 EPFL 影片抽幀,準備 M3 微調 bootstrap 的標註素材。

策略:優先挑「有動作」的幀(M2 觸發),並在全片均勻分散(涵蓋不同時間/動作),
避免全是相似或空景。輸出 jpg 供標註。

輸出:data/m3_finetune/images/frame_<fid>.jpg(data/ 已 gitignore)
用法:
  python scripts/extract_finetune_frames.py            # 預設抽 ~40 張
  python scripts/extract_finetune_frames.py --n 60
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from common.video_io import iter_frames, video_meta, imwrite_unicode   # noqa: E402
from m2_motion.detector import MotionDetector                          # noqa: E402

VIDEO = ROOT / "data" / "epfl" / "Boutput0.mp4"
OUT = ROOT / "data" / "m3_finetune" / "images"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40, help="要抽幾張")
    ap.add_argument("--video", default=str(VIDEO))
    args = ap.parse_args()

    video = Path(args.video)
    meta = video_meta(video)
    total = meta["nb_frames"]
    # 均勻分散的目標幀號(全片切 n 段,各段挑一張「有動作」的)
    seg = max(1, total // args.n)
    print(f"影片 {total} 幀,分 {args.n} 段(每段 ~{seg} 幀),各挑一張有動作的")

    det = MotionDetector()
    OUT.mkdir(parents=True, exist_ok=True)

    saved = []
    next_seg = 0                      # 下一個要填的段
    picked_in_seg = False
    for fid, ts, frame in iter_frames(video):
        r = det.process(frame)
        cur_seg = fid // seg
        if cur_seg > next_seg:        # 進到新段,重置
            next_seg = cur_seg
            picked_in_seg = False
        if not picked_in_seg and r["has_motion"] and fid > 60:   # 過了暖機、有動作
            fp = OUT / f"frame_{fid:05d}.jpg"
            if imwrite_unicode(fp, frame):
                saved.append(fp.name)
                picked_in_seg = True
        if len(saved) >= args.n:
            break

    print(f"完成:抽出 {len(saved)} 張 → {OUT.relative_to(ROOT).as_posix()}/")
    print("前幾張:", saved[:6])


if __name__ == "__main__":
    main()
