"""OSNet 外觀特徵 embedder(**僅供驗證對照,不可出貨**)。

torchreid 程式碼為 MIT,但 `osnet_x1_0` 等**預訓權重是在 Market-1501/MSMT17/DukeMTMC
等研究限定資料集上訓練**,不可用於商業出貨(見 docs/M5_身份管理設計與驗證.md 授權分析)。
此類別只用來當「Re-ID 專用模型的準確度上限」對照;出貨請改用 DINOv2 或自訓模型。

依賴 torchreid(`pip install torchreid`)+ torch。
"""
import numpy as np

from m5_reid.embedder import BaseEmbedder, l2norm


class OSNetEmbedder(BaseEmbedder):
    def __init__(self, model_name="osnet_x1_0", device=None, weights=""):
        import torch
        from torchreid.utils import FeatureExtractor
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.extractor = FeatureExtractor(model_name=model_name,
                                          model_path=weights or "", device=self.device)
        self.dim = 512

    def _rgb(self, crop_bgr):
        import cv2
        return cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)     # torchreid 內部自帶前處理

    def extract(self, crop_bgr) -> np.ndarray:
        feat = self.extractor([self._rgb(crop_bgr)])          # [1, 512]
        return l2norm(feat[0].cpu().numpy())

    def extract_batch(self, crops):
        feats = self.extractor([self._rgb(c) for c in crops]).cpu().numpy()
        return np.stack([l2norm(f) for f in feats])
