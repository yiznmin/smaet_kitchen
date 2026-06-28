"""
做一支 demo 影片:把合成老鼠貼進 EPFL 真廚房(竄動 + 走停),並疊上分支 B 偵測框。

誠實聲明:老鼠為**程式合成**(非真老鼠影像)。用途:
  - 視覺展示「老鼠出現 → 被偵測線抓到」的端到端效果
  - 幾何過濾的動態壓力測試(竄動、走停)
不可用於訓練分類器辨識真老鼠(那需要真實老鼠影像)。

輸出:results/m2/mouse_demo.mp4
用法: python scripts/make_mouse_video.py data/epfl/Boutput0.mp4
"""
import shutil
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from common.video_io import iter_frames, video_meta       # noqa: E402
from m2_motion.branch_b import SmallTargetDetector         # noqa: E402
from m2_motion.detector import MotionDetector              # noqa: E402

CONFIG = ROOT / "configs" / "benchmark.yaml"
DEST = ROOT / "results" / "m2" / "mouse_demo.mp4"
N_FRAMES = 900            # 取前 900 幀(~30s @30fps)
MOUSE_START = 120         # 老鼠第幾幀進場(讓 MOG2 背景先建立)


def mouse_pos(f):
    """竄動路徑:沿地板走走停停。回傳 (cx, cy) 或 None(未進場/離場)。
    決定性:快走 ~12 幀 → 停 ~8 幀,沿地板由左往右再折返。"""
    if f < MOUSE_START:
        return None
    t = f - MOUSE_START
    cycle = 20                      # 每 20 幀一個「走12+停8」週期
    seg = t // cycle
    in_cycle = t % cycle
    moved_segs = seg                # 已完成的位移段數
    # 每個週期前 12 幀在走(每幀 9px),後 8 幀停
    walked = min(in_cycle, 12) * 9 + moved_segs * 12 * 9
    span = 760
    pos = walked % (2 * span)
    x = 360 + (pos if pos <= span else 2 * span - pos)      # 在 [360,1120] 來回
    y = 640 + int(28 * np.sin(t * 0.08))                    # 沿地板上下緩擺
    return int(x), int(y)


def draw_mouse(img, cx, cy, moving_right):
    """畫一隻簡單但像樣的老鼠:身體橢圓 + 頭 + 尾巴。"""
    body = (88, 92, 104)            # 灰褐(BGR)
    d = 1 if moving_right else -1
    cv2.ellipse(img, (cx, cy), (11, 7), 0, 0, 360, body, -1)            # 身體
    cv2.circle(img, (cx + d * 10, cy - 1), 5, body, -1)                 # 頭
    cv2.circle(img, (cx + d * 12, cy - 5), 2, (70, 74, 86), -1)         # 耳
    tail = [(cx - d * 11, cy), (cx - d * 22, cy + 4), (cx - d * 30, cy - 2)]
    cv2.polylines(img, [np.array(tail, np.int32)], False, (120, 124, 138), 1)  # 尾
    cv2.circle(img, (cx + d * 13, cy - 1), 1, (40, 40, 40), -1)         # 眼


def make_det(cfg):
    b = cfg["m2_branch_b"]
    return SmallTargetDetector(
        min_blob_area=b["min_blob_area"], large_blob_area=b["large_blob_area"],
        exclusion_margin=b["exclusion_margin"], crop_size=b["crop_size"],
        max_crops_per_frame=b["max_crops_per_frame"], mog2_history=b["mog2_history"],
        mog2_var_threshold=b["mog2_var_threshold"], warmup_frames=b["warmup_frames"],
        persistence_frames=b["persistence_frames"], match_max_dist=b["match_max_dist"],
        track_max_miss=b["track_max_miss"],
        density_radius=b["density_radius"], density_max_neighbors=b["density_max_neighbors"])


def main():
    video = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data" / "epfl" / "Boutput0.mp4"
    if not video.exists():
        sys.exit(f"找不到影片 {video}")
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    meta = video_meta(video)
    W, H = meta["width"], meta["height"]
    fps = meta.get("fps", 30) or 30

    det = make_det(cfg)
    m = cfg["m2_motion"]
    branch_a = MotionDetector(m["diff_threshold"], m["min_motion_ratio"],
                              m["blur_ksize"], m["use_reference_frame"])   # 分支 A:框人/大動作
    tmp = Path(tempfile.gettempdir()) / "mouse_demo_tmp.mp4"     # ASCII 暫存(VideoWriter 不吃中文路徑)
    vw = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (W, H))

    prev_x = None
    detected_frames = present_frames = 0
    for fid, ts, frame in iter_frames(video):
        if fid >= N_FRAMES:
            break
        pos = mouse_pos(fid)
        if pos is not None:
            cx, cy = pos
            draw_mouse(frame, cx, cy, moving_right=(prev_x is None or cx >= prev_x))
            prev_x = cx

        # 先在「乾淨畫面」上跑兩條線(不可先畫框,否則框線會干擾 MOG2)
        ra = branch_a.process(frame)
        r = det.process(frame)

        # 處理完才畫:分支 A 青色大框(框人/大動作 → 送 M3)
        if ra["has_motion"] and ra["bbox"]:
            ax1, ay1, ax2, ay2 = ra["bbox"]
            cv2.rectangle(frame, (ax1, ay1), (ax2, ay2), (255, 255, 0), 2)
            cv2.putText(frame, "Branch A: motion -> M3 (person)", (ax1, max(20, ay1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)

        # 分支 B 疊框:命中老鼠的=綠色,其餘候選=細灰
        got = False
        for c in r["crops"]:
            bx1, by1, bx2, by2 = c["blob_bbox"]
            x1, y1, x2, y2 = c["bbox"]
            hit = pos is not None and bx1 <= pos[0] <= bx2 and by1 <= pos[1] <= by2
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0) if hit else (170, 170, 170),
                          3 if hit else 1)
            got = got or hit
        if pos is not None and fid >= MOUSE_START + cfg["m2_branch_b"]["persistence_frames"]:
            present_frames += 1
            detected_frames += int(got)

        tag = "synthetic mouse (demo)"
        if pos is not None:
            cv2.putText(frame, f"{tag}  -  {'DETECTED' if got else '...'}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0) if got else (0, 255, 255), 2)
        vw.write(frame)

    vw.release()
    DEST.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(tmp), str(DEST))               # shutil 可處理中文目的路徑
    recall = detected_frames / present_frames if present_frames else 0
    print(f"完成:{DEST.relative_to(ROOT).as_posix()}")
    print(f"老鼠在場且過 persistence 的幀:{present_frames},其中被偵測到:{detected_frames}（{recall:.1%}）")


if __name__ == "__main__":
    main()
