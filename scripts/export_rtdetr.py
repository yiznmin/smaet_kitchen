"""
匯出 RT-DETRv2(Apache-2.0)成 ONNX,透過 HuggingFace transformers。
- 變體:r18 / r34(COCO,來自 PekingU)。
- wrapper:pixel_values [N,3,640,640] → (logits, pred_boxes),動態 batch,opset 17。
產出 models/rtdetrv2_{r18,r34}.onnx。
"""
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
MODELS.mkdir(exist_ok=True)

try:
    from transformers import RTDetrV2ForObjectDetection as RTD
except Exception:
    from transformers import RTDetrForObjectDetection as RTD

import sys

VARIANTS = [
    ("rtdetrv2_r18", "PekingU/rtdetr_v2_r18vd"),
    ("rtdetrv2_r34", "PekingU/rtdetr_v2_r34vd"),
]
IMG = int(sys.argv[1]) if len(sys.argv) > 1 else 640
SUFFIX = f"_{IMG}" if len(sys.argv) > 1 else ""


class Wrap(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, pixel_values):
        out = self.model(pixel_values=pixel_values)
        return out.logits, out.pred_boxes


for name, hf_id in VARIANTS:
    print(f"\n=== 匯出 {name} ({hf_id}) ===")
    try:
        model = RTD.from_pretrained(hf_id).eval()
        wrap = Wrap(model).eval()
        dummy = torch.randn(1, 3, IMG, IMG)
        dest = MODELS / f"{name}{SUFFIX}.onnx"
        torch.onnx.export(
            wrap, dummy, str(dest),
            opset_version=17, input_names=["pixel_values"],
            output_names=["logits", "pred_boxes"],
            dynamic_axes={"pixel_values": {0: "batch"},
                          "logits": {0: "batch"}, "pred_boxes": {0: "batch"}},
            do_constant_folding=True, dynamo=False,
        )
        print(f"  ✓ {name} -> {dest.relative_to(ROOT).as_posix()} ({dest.stat().st_size/1e6:.1f} MB)")
    except Exception as e:
        print(f"  ✗ {name} 失敗: {repr(e)[:200]}")

print("\n完成。")
