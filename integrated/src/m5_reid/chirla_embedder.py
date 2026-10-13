"""把 CHIRLA 訓練出來的 Re-ID 模型包成 BaseEmbedder,讓它能直接插回 M5。

這是「補上 M5 缺的那塊」的落地接口。M5 的 IdentityManager 只呼叫 `extract(crop)`
(見 `identity_st.py:75`),所以只要符合契約,訓練好的模型就能取代 DINOv2 —— 而且
因為 CHIRLA 是 CC-BY-4.0,**這是第一個可以出貨的外觀模型**。

契約(`src/m5_reid/embedder.py:21-24`):
  · 輸入 = 單張 BGR HxWx3 uint8(OpenCV 慣例,未縮放未正規化)
  · 輸出 = **L2-normalized 的 1D ndarray**,長度 == self.dim
  · 另實作 extract_batch(crops) -> [N, dim],評估腳本走這支

⚠ 用了它之後**必須重新校準 `evidence.py` 的 AppearanceLR.MEASURED`** ——
  那裡目前只有 dinov2/osnet 兩組 EPFL 實測的 (mu_same, mu_diff)。沒有新模型的那組,
  `AppearanceLR.measured()` 會直接 raise。校準數字由
  `reid_eval_epfl.cross_view_consistency()` 產生。這是最容易漏掉的一步。

⚠ 權重來源會決定能不能出貨,所以 checkpoint 裡存了 `provenance` 字串,
  載入時會印出來 —— 避免哪天忘了手上這顆是 arm S 還是 arm R。
"""
import numpy as np

from m5_reid.embedder import BaseEmbedder, l2norm

_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class ChirlaEmbedder(BaseEmbedder):
    """載入 train_reid_chirla.py 產出的 best.pth。"""

    def __init__(self, ckpt, device=None, size=(256, 128), quiet=False):
        import torch                        # lazy import,與其他 embedder 一致
        self._torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.h, self.w = size

        blob = torch.load(ckpt, map_location="cpu", weights_only=False)
        self.dim = int(blob.get("dim", 2048))
        self.provenance = blob.get("provenance", "(未記錄)")
        self.arm = blob.get("arm", "?")
        n_classes = len(blob.get("pid_map") or {}) or 1

        # 重建與訓練時完全相同的結構(build_model 在訓練腳本裡,這裡複製最小必要部分)
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
        from train_reid_chirla import build_model
        # pretrained=False:只要結構,權重下一行就被 checkpoint 蓋掉。
        # 不這樣做的話每次載模型都會白下載 100MB 的 ImageNet 權重再丟掉。
        self.model, _ = build_model(n_classes, arm="S", weights=None,
                                    device=self.device, pretrained=False)
        self.model.load_state_dict(blob["model"])
        self.model.eval()

        if not quiet:
            print(f"ChirlaEmbedder dim={self.dim} arm={self.arm} device={self.device}")
            print(f"  權重來源:{self.provenance}")
            if self.arm == "R":
                print("  ⚠ arm R 的起始權重是研究限定的外部 Re-ID 權重,**不可出貨**")

    def _pre(self, crop_bgr):
        import cv2
        img = cv2.resize(crop_bgr, (self.w, self.h))
        x = img[:, :, ::-1].astype(np.float32) / 255.0      # BGR → RGB
        x = (x - _MEAN) / _STD
        return np.ascontiguousarray(x.transpose(2, 0, 1))

    def extract(self, crop_bgr):
        return self.extract_batch([crop_bgr])[0]

    def extract_batch(self, crops, batch_size=64):
        torch = self._torch
        if not len(crops):
            return np.zeros((0, self.dim), dtype=np.float32)
        out = []
        with torch.no_grad():
            for i in range(0, len(crops), batch_size):
                chunk = np.stack([self._pre(c) for c in crops[i:i + batch_size]])
                f = self.model(torch.from_numpy(chunk).to(self.device))
                out.append(f.cpu().numpy().astype(np.float32))
        feats = np.concatenate(out, 0)
        # 契約要求逐列 L2-normalized —— evaluate_cmc_map 靠內積當 cosine,
        # 少了這一步所有相似度都是錯的。
        return np.stack([l2norm(v) for v in feats])
