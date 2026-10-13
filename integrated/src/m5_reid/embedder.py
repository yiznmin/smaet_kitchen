"""M5 外觀特徵抽取:BaseEmbedder 介面 + ColorHistogramEmbedder(佔位,授權乾淨)。

⚠ spec 指定 OSNet(512 維外觀特徵);但其常見預訓權重在 Market-1501/MSMT17/DukeMTMC
   等資料集上訓練 = **研究限定、不可商用**(見 docs/M5_身份管理設計與驗證.md 的授權分析)。
   因此本模組採「可插拔介面」:
   - 先用純 OpenCV 的**顏色直方圖**當佔位特徵(自足、可商用、能跑通整條 Re-ID 流程)。
   - 待取得**可商用授權或自訓**的外觀模型,再實作一個 BaseEmbedder 子類換入,
     IdentityManager 完全不動(只換 embedder)。
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
