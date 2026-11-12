"""影片讀取(M1):逐幀解碼影片檔。

  iter_frames(path, stride, start, end)  逐幀產出 (幀號, 秒數, BGR 影像);stride > 1 時每隔 stride 幀取一張
  video_meta(path)                       幀率、總幀數、解析度
  imwrite_unicode(path, img)             寫入含中文的路徑(Windows 上 cv2.imwrite 不支援)

只支援影片檔,不支援 RTSP 串流。"""
from pathlib import Path

import cv2
import numpy as np


def iter_frames(path, stride=1, start=0, end=None):
    """逐幀讀取影片。stride>1 時每隔 stride 幀取一張(模擬降取樣)。

    start / end(只處理影片的一段):
      start  從這個 fid 開始(含)。**必須是 stride 的倍數** —— 取樣格點要與從頭跑時相同,
             否則取到的幀號落在 {0, stride, 2·stride, …} 之外,結果與從頭跑對不上。
      end    到這個 fid 為止(**含**)。None = 讀到影片結束。

    ⚠ 快轉一律用 grab() 順序前進,**不用 CAP_PROP_POS_FRAMES** ——
      部分影片編碼跳轉不精確,幀號一偏,後面所有幀都會錯位。
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
