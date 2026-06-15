"""
匯出 D-FINE(Apache-2.0)成 ONNX,透過 HuggingFace transformers(免 clone repo)。
- 變體:nano / small / medium(COCO 權重,來自 ustc-community)。
- 包一層 wrapper:輸入 pixel_values [N,3,640,640] → 輸出 (logits, pred_boxes)。
- 動態 batch 軸,opset 17,dynamo=False(舊版匯出器較穩)。
產出 models/dfine_{n,s,m}.onnx。
"""
from pathlib import Path

import torch
import torch.nn as nn
from transformers import DFineForObjectDetection

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
MODELS.mkdir(exist_ok=True)

import sys

VARIANTS = [
    ("dfine_n", "ustc-community/dfine-nano-coco"),
    ("dfine_s", "ustc-community/dfine-small-coco"),
    ("dfine_m", "ustc-community/dfine-medium-coco"),
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
        model = DFineForObjectDetection.from_pretrained(hf_id)
        model.eval()
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
