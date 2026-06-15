"""
M2 分支 B 分類器:對「已被背景相減定位 + 裁切放大」的小目標 crop 做 binary 確認
(是不是目標,如老鼠)。對應 docs/M2_雙分支重構設計.md §5。

模型族:MobileNetV3 / EfficientNet-lite(Apache/BSD,授權乾淨),匯出 ONNX,
與 M3 共用 ONNX Runtime 後端(不引入新框架)。

注意:目前尚無訓練好的權重(資料缺口見設計文件 §5.3 — 需蒐集老鼠正樣本 +
蒸氣/光影等困難負樣本)。無模型時 available=False,predict() 回 (None, 0.0),
讓管線仍可先驗證「定位/裁切」是否正確,日後補上權重即可啟用確認層。
"""
from pathlib import Path

import cv2
import numpy as np

try:
    import onnxruntime as ort
except Exception:          # onnxruntime 未安裝時不阻塞(分支 B 定位仍可跑)
    ort = None


class TargetClassifier:
    def __init__(self, onnx_path=None, input_size=224,
                 mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225),
                 target_index=1, threshold=0.5, providers=None):
        self.input_size = int(input_size)
        self.mean = np.array(mean, dtype=np.float32).reshape(1, 3, 1, 1)
        self.std = np.array(std, dtype=np.float32).reshape(1, 3, 1, 1)
        self.target_index = target_index
        self.threshold = threshold
        self.session = None
        self._input_name = None

        if onnx_path and ort is not None and Path(onnx_path).exists():
            if providers is None:
                avail = ort.get_available_providers()
                providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider")
                             if p in avail] or None
            self.session = ort.InferenceSession(str(onnx_path), providers=providers)
            self._input_name = self.session.get_inputs()[0].name

    @property
    def available(self):
        return self.session is not None

    def _preprocess(self, crop_bgr):
        img = cv2.resize(crop_bgr, (self.input_size, self.input_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1)[None]            # HWC → NCHW
        return ((img - self.mean) / self.std).astype(np.float32)

    def predict(self, crop_bgr):
        """回傳 (is_target, confidence)。
        有模型:is_target 為 bool、confidence 為目標類別機率。
        無模型:回 (None, 0.0) —— 由呼叫端視為「待確認」。"""
        if not self.available:
            return None, 0.0
        logits = self.session.run(None, {self._input_name: self._preprocess(crop_bgr)})[0][0]
        prob = _softmax(np.asarray(logits, dtype=np.float32))
        conf = float(prob[self.target_index])
        return conf >= self.threshold, conf


def _softmax(v):
    v = v - np.max(v)
    e = np.exp(v)
    return e / np.sum(e)
