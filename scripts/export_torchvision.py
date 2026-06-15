"""
匯出一個 torchvision 真實偵測模型成 ONNX,用來驗證 M3 harness 跑「真實模型」沒問題。

說明:
  - 預設 SSDlite(MobileNetV3),真實偵測架構、BSD-3 授權(可商用)。
  - 用隨機權重(weights=None):測速只在乎運算量與記憶體,不需要真權重,也免下載。
  - ⚠ 僅作流程驗證,非本專案出貨候選(出貨候選為 YOLOX/D-FINE/RF-DETR 等,見 docs/模型選型評估.md)。

用法:
  python scripts/export_torchvision.py
"""
from pathlib import Path

import torch
from torchvision.models.detection import ssdlite320_mobilenet_v3_large

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "models" / "ssdlite_mobilenetv3.onnx"
OUT.parent.mkdir(parents=True, exist_ok=True)

model = ssdlite320_mobilenet_v3_large(weights=None, weights_backbone=None)
model.eval()

# SSDlite 預設輸入 320x320;以一張 dummy 影像匯出,batch 維設為動態
dummy = torch.randn(1, 3, 320, 320)

print("匯出 ONNX 中 …")
torch.onnx.export(
    model, dummy, str(OUT),
    opset_version=17,
    input_names=["images"],
    output_names=["boxes", "scores", "labels"],
    dynamic_axes={"images": {0: "batch"}},
    do_constant_folding=True,
    dynamo=False,                 # 用穩定的舊版(TorchScript)匯出器,偵測模型相容性較佳
)
print(f"完成:{OUT.relative_to(ROOT).as_posix()}")

# 印出實際的輸入名稱與形狀,供 harness 對應
import onnx
m = onnx.load(str(OUT))
for i in m.graph.input:
    dims = [d.dim_param or d.dim_value for d in i.type.tensor_type.shape.dim]
    print(f"  input: {i.name}  shape={dims}")
