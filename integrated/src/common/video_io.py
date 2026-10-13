"""
共用影像 I/O(M1 的離線版:只負責讀檔解碼,RTSP/斷線重連等留待 M1 線上版)。

提供 iter_frames():逐幀產出 (frame_id, timestamp_s, frame_bgr)。
注意:Windows 上含中文路徑時,cv2.VideoCapture 多數情況可讀,
      但寫檔請用 imencode + write_bytes(見 imwrite_unicode)。
"""
from pathlib import Path

import cv2
import numpy as np


def iter_frames(path, stride=1, start=0, end=None):
    """逐幀讀取影片。stride>1 時每隔 stride 幀取一張(模擬降取樣)。

    start / end(2026-09-23 加,為了逐片段獨立推論):
      start  從這個 fid 開始(含)。**必須是 stride 的倍數** —— 取樣格點要與全長執行相同
             (GT 幀 = fid + 1),否則片段的 fid 落在 {0, stride, 2·stride, …} 之外,
             既對不上標註也對不上已提交的數字。
      end    到這個 fid 為止(**含**)。None = 讀到影片結束。

    ⚠ start=0、end=None 時與 2026-09-22 之前**逐位相同**(V0b 驗)。
    ⚠ 快轉一律用 grab() 順序前進,**不用 CAP_PROP_POS_FRAMES** ——
      這些影片跳轉不可靠,幀號一偏,整支影片的標註就錯位
      (見 scripts/render_level_video.py 檔頭)。
    """
    if start % stride:
        raise ValueError(f"start={start} 不在 stride={stride} 的格點上")
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"無法開啟影片: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    fid = 0
    try:
        while fid < start:                 # 快轉:只解多工不解碼
            if not cap.grab():
                return
            fid += 1
        while True:
            if end is not None and fid > end:
                break
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
