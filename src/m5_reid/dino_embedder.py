"""DINOv2 外觀特徵 embedder(**可商用**:code + weights 皆 Apache-2.0)。

Meta DINOv2 主模型(https://github.com/facebookresearch/dinov2)程式碼與權重皆 Apache-2.0
(CC-BY-NC 只適用其 Cell-DINO 生物延伸,不影響此處)。→ 可放進出貨產品。
DINOv2 是通用視覺特徵,非 Re-ID 專用;實際 Re-ID 準確度由 reid_eval_market1501 驗證。

依賴 torch(推理)。在有 torch 的環境(Colab 或本機 inference env)才 import。
"""
import numpy as np

from m5_reid.embedder import BaseEmbedder, l2norm

_DIM = {"dinov2_vits14": 384, "dinov2_vitb14": 768,
        "dinov2_vitl14": 1024, "dinov2_vitg14": 1536}


class DINOv2Embedder(BaseEmbedder):
    def __init__(self, model_name="dinov2_vits14", device=None, image_size=224):
        import torch
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = torch.hub.load("facebookresearch/dinov2", model_name)
        self.model = self.model.to(self.device).eval()
        self.dim = _DIM.get(model_name, 384)
        # image_size 需為 patch(14)的倍數
        self.image_size = (image_size // 14) * 14
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def _pre(self, crop_bgr):
        import cv2
        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (self.image_size, self.image_size))
        x = (rgb.astype(np.float32) / 255.0 - self.mean) / self.std
        return x.transpose(2, 0, 1)                          # CHW

    def extract(self, crop_bgr) -> np.ndarray:
        x = self.torch.from_numpy(self._pre(crop_bgr)).unsqueeze(0).to(self.device)
        with self.torch.no_grad():
            feat = self.model(x)                             # [1, dim](CLS token)
        return l2norm(feat[0].float().cpu().numpy())

    def extract_batch(self, crops):
        """批次抽特徵(加速 Re-ID 評估用)。回傳 [N, dim] 各列 L2-normalized。"""
        xs = np.stack([self._pre(c) for c in crops])
        t = self.torch.from_numpy(xs).to(self.device)
        with self.torch.no_grad():
            feats = self.model(t).float().cpu().numpy()
        return np.stack([l2norm(f) for f in feats])
