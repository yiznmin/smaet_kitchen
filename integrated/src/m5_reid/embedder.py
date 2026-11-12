"""M5 外觀特徵抽取:BaseEmbedder 介面 + ColorHistogramEmbedder(純 OpenCV 顏色直方圖,可商用)。

可插拔介面:要換外觀模型時,實作一個 BaseEmbedder 子類即可,IdentityManager 不用改。
注意:常見的 Re-ID 預訓權重(Market-1501、MSMT17、DukeMTMC 等資料集)多為研究限定、不可商用。
"""
import numpy as np


def l2norm(v):
    v = np.asarray(v, dtype=np.float32)
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


class BaseEmbedder:
    """外觀特徵抽取介面。extract() 回傳 L2-normalized 1D 向量。"""
    dim = 0

    def extract(self, crop_bgr) -> np.ndarray:
        raise NotImplementedError


class ColorHistogramEmbedder(BaseEmbedder):
    """HSV 顏色直方圖當外觀特徵(佔位)。

    對「衣服顏色不同的人」可分辨,足以跑通並驗證整條 Re-ID 邏輯;
    但**非姿態/光照強健**,不代表最終精度——正式精度靠日後換入的外觀模型。
    """
    def __init__(self, h_bins=16, s_bins=16):
        self.h_bins, self.s_bins = h_bins, s_bins
        self.dim = h_bins * s_bins

    def extract(self, crop_bgr) -> np.ndarray:
        import cv2
        hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None,
                            [self.h_bins, self.s_bins], [0, 180, 0, 256])
        return l2norm(hist.flatten())
