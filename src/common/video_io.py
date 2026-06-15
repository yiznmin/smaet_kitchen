"""
共用影像 I/O(M1 的離線版:只負責讀檔解碼,RTSP/斷線重連等留待 M1 線上版)。

提供 iter_frames():逐幀產出 (frame_id, timestamp_s, frame_bgr)。
注意:Windows 上含中文路徑時,cv2.VideoCapture 多數情況可讀,
      但寫檔請用 imencode + write_bytes(見 imwrite_unicode)。
"""
from pathlib import Path

import cv2
import numpy as np


def iter_frames(path, stride=1):
    """逐幀讀取影片。stride>1 時每隔 stride 幀取一張(模擬降取樣)。"""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"無法開啟影片: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    fid = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if fid % stride == 0:
                yield fid, fid / fps, frame
            fid += 1
    finally:
        cap.release()


def video_meta(path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"無法開啟影片: {path}")
    meta = dict(
        width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        fps=round(cap.get(cv2.CAP_PROP_FPS), 3),
        nb_frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    )
    cap.release()
    return meta


def imwrite_unicode(path, img):
    """cv2.imwrite 無法處理非 ASCII(中文)路徑,改用 imencode 寫 bytes。"""
    path = Path(path)
    ext = path.suffix or ".png"
    ok, buf = cv2.imencode(ext, img)
    if ok:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(buf.tobytes())
    return ok
