"""
M2 動態偵測:用背景差分(cv2.absdiff)判斷畫面是否有變動。

設計重點(對應說明文件 M2):
  - 極低 CPU 成本(灰階 + 模糊 + 差分 + 閾值),作為 M3(YOLO)的「省電開關」。
  - has_motion 為 True 時才需要把影格送進 M3。
  - 兩種比較模式:與前一幀比(預設)或與固定參考幀比(use_reference_frame)。

不使用任何模型,純 OpenCV/CPU。
"""
import cv2
import numpy as np


class MotionDetector:
    def __init__(self, diff_threshold=25, min_motion_ratio=0.002,
                 blur_ksize=5, use_reference_frame=False):
        self.diff_threshold = diff_threshold
        self.min_motion_ratio = min_motion_ratio
        self.blur_ksize = blur_ksize if blur_ksize % 2 == 1 else blur_ksize + 1
        self.use_reference_frame = use_reference_frame
        self._prev = None        # 前一幀(灰階模糊後)
        self._ref = None         # 固定參考幀

    def _preprocess(self, frame_bgr):
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        return cv2.GaussianBlur(gray, (self.blur_ksize, self.blur_ksize), 0)

    def process(self, frame_bgr, return_mask=False):
        """回傳 dict:has_motion、motion_ratio、bbox(動態區域,可能 None)。
        return_mask=True 時額外回傳 mask(M2 實際偵測到的變化遮罩,供驗證視覺化)。"""
        cur = self._preprocess(frame_bgr)
        baseline = self._ref if self.use_reference_frame else self._prev

        if baseline is None:
            # 第一幀:建立基準,視為無動態
            if self.use_reference_frame:
                self._ref = cur
            self._prev = cur
            out = dict(has_motion=False, motion_ratio=0.0, bbox=None)
            if return_mask:
                out["mask"] = np.zeros_like(cur)
            return out

        diff = cv2.absdiff(cur, baseline)
        _, thresh = cv2.threshold(diff, self.diff_threshold, 255, cv2.THRESH_BINARY)
        thresh = cv2.dilate(thresh, None, iterations=2)

        motion_pixels = int(cv2.countNonZero(thresh))
        ratio = motion_pixels / thresh.size
        has_motion = ratio > self.min_motion_ratio

        bbox = None
        if has_motion:
            ys, xs = np.where(thresh > 0)
            if xs.size:
                bbox = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]

        self._prev = cur
        out = dict(has_motion=has_motion, motion_ratio=round(ratio, 5), bbox=bbox)
        if return_mask:
            out["mask"] = thresh
        return out
