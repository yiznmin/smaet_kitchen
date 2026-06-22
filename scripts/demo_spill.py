"""
示範:把一塊合成污漬貼到 EPFL 地板上,展示靜態線把它框起來。

EPFL 影片本身沒有污漬,這裡合成一塊「扁平、落地靜止」的污漬(含水/油反光),
貼在地板的安靜區域(遠離人),讓靜態線偵測 → 框出來。
紅框 = 偵測到的污漬;綠框 = 其他靜態殘留(多為人腿,待分類器拒絕)。

用法:
  python scripts/demo_spill.py data/epfl/Boutput0.mp4
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from common.video_io import iter_frames, imwrite_unicode    # noqa: E402
from m2_motion.static_line import StaticTargetDetector       # noqa: E402

CONFIG = ROOT / "configs" / "benchmark.yaml"
OUT = ROOT / "results" / "m2" / "static_line" / "spill_demo"
CROP_DIR = OUT / "crops"

EPFL_FLOOR = [[330, 715], [1080, 715], [950, 380], [470, 380]]
SPILL_CENTER = (560, 600)     # 地板上、遠離人的安靜位置
SPILL_AXES = (55, 24)         # 扁平橢圓(寬 > 高 → 通過長寬比過濾)
SPILL_START = 55              # 第幾個取樣幀開始出現污漬(讓慢速基準先建立乾淨地板)


def paste_spill(frame):
    """在地板貼一塊扁平污漬:暗色主體 + 一小塊高光(水/油反光)。"""
    f = frame.copy()
    ov = f.copy()
    cv2.ellipse(ov, SPILL_CENTER, SPILL_AXES, 0, 0, 360, (70, 62, 55), -1)
    cv2.ellipse(ov, (SPILL_CENTER[0] - 16, SPILL_CENTER[1] - 6), (14, 7), 0, 0, 360,
                (195, 195, 205), -1)
    return cv2.addWeighted(ov, 0.85, f, 0.15, 0)


def main():
    video = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data" / "epfl" / "Boutput0.mp4"
    if not video.exists():
        sys.exit(f"找不到影片 {video}")
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    s = cfg["m2_static_line"]
    det = StaticTargetDetector(
        baseline_alpha=s["baseline_alpha"], baseline_diff_thresh=s["baseline_diff_thresh"],
        motion_thresh=s["motion_thresh"], blur_ksize=s["blur_ksize"],
        min_blob_area=s["min_blob_area"], max_blob_area=s["max_blob_area"],
        crop_size=s["crop_size"], warmup_frames=s["warmup_frames"],
        motion_exclusion_margin=s["motion_exclusion_margin"], max_aspect_ratio=s["max_aspect_ratio"],
        persistence_frames=s["persistence_frames"], match_max_dist=s["match_max_dist"],
        track_max_miss=s["track_max_miss"],
        density_radius=s["density_radius"], density_max_neighbors=s["density_max_neighbors"],
        floor_zone=EPFL_FLOOR)

    OUT.mkdir(parents=True, exist_ok=True)
    CROP_DIR.mkdir(parents=True, exist_ok=True)
    crops_saved = []
    floor_pts = np.array(EPFL_FLOOR, np.int32)
    # 污漬實際範圍(用 blob 框是否落在此處判定,而非 256 裁切窗)
    sx1, sy1 = SPILL_CENTER[0] - SPILL_AXES[0], SPILL_CENTER[1] - SPILL_AXES[1]
    sx2, sy2 = SPILL_CENTER[0] + SPILL_AXES[0], SPILL_CENTER[1] + SPILL_AXES[1]

    def is_spill(c):
        bx1, by1, bx2, by2 = c["blob_bbox"]      # 用 blob 本身,不是裁切窗
        return bx1 <= sx2 and bx2 >= sx1 and by1 <= sy2 and by2 >= sy1

    saved = []
    k = 0
    for fid, ts, frame in iter_frames(video, stride=10):
        k += 1
        spill_present = k >= SPILL_START
        if spill_present:
            frame = paste_spill(frame)
        r = det.process(frame)

        # 只在污漬已貼上、且 blob 真的落在污漬處時才算偵測到
        if spill_present and any(is_spill(c) for c in r["crops"]):
            vis = frame.copy()
            cv2.polylines(vis, [floor_pts], True, (255, 255, 0), 2)        # 青=地板 ROI
            cv2.circle(vis, SPILL_CENTER, 4, (0, 0, 255), -1)
            for i, c in enumerate(r["crops"]):
                x1, y1, x2, y2 = c["bbox"]
                spill = is_spill(c)
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255) if spill else (0, 255, 0),
                              3 if spill else 2)
                # 存「會被送進分類器」的原始裁切小圖(污漬 / 其他 各自標名)
                if len(saved) == 0:        # 只存第一個命中幀的所有 crop
                    tag = "spill" if spill else "other"
                    cfp = CROP_DIR / f"crop_f{fid}_{tag}_{i}.png"
                    if imwrite_unicode(cfp, c["crop"]):
                        crops_saved.append(cfp.relative_to(ROOT).as_posix())
            cv2.putText(vis, "RED=spill detected   GREEN=other static (legs)",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            fp = OUT / f"spill_f{fid}.png"
            if imwrite_unicode(fp, vis):
                saved.append(fp.relative_to(ROOT).as_posix())
            if len(saved) >= 6:
                break

    print(f"已存污漬偵測全圖 {len(saved)} 張:{saved}")
    print(f"已存送分類器的原始裁切小圖 {len(crops_saved)} 張:{crops_saved}")


if __name__ == "__main__":
    main()
