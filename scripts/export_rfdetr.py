"""
匯出 RF-DETR(Apache-2.0)的 nano/small/medium 三個變體成 ONNX。
- dynamic_batch=True:可測 batch 1/4/8。
- 各變體用其原生解析度(nano 384 / small 512 / medium 576)。
- simplify=False:避免額外相依(onnxsim)。
產出搬到 models/<name>.onnx。
⚠ 不含 XL/2XL(那是 PML 授權,不可商用)。
"""
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
MODELS.mkdir(exist_ok=True)

from rfdetr import RFDETRMedium, RFDETRNano, RFDETRSmall   # noqa: E402

# 可選:指定統一解析度(須為 56 的倍數,如 672)做公平比較;不給則用各變體原生尺寸
SIZE = int(sys.argv[1]) if len(sys.argv) > 1 else None
SUFFIX = f"_{SIZE}" if SIZE else ""

VARIANTS = [("rfdetr_nano", RFDETRNano), ("rfdetr_small", RFDETRSmall),
            ("rfdetr_medium", RFDETRMedium)]

for name, cls in VARIANTS:
    print(f"\n=== 匯出 {name} ===")
    out_dir = MODELS / f"_export_{name}"
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        # 指定 resolution 時在「建構」階段設定(位置編碼一開始就是該尺寸),
        # 避免 forward 內插 → 規避 aten::_upsample_bicubic2d_aa 無法匯出的問題。
        model = cls(resolution=SIZE) if SIZE else cls()
        model.export(output_dir=str(out_dir), simplify=False,
                     dynamic_batch=True, opset_version=17, verbose=False)
        # 找出產生的 .onnx,搬到 models/<name>.onnx
        onnxs = sorted(out_dir.glob("*.onnx"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not onnxs:
            print(f"  ✗ {name}: 匯出後找不到 .onnx")
            continue
        dest = MODELS / f"{name}{SUFFIX}.onnx"
        shutil.copy(onnxs[0], dest)
        print(f"  ✓ {name} -> {dest.relative_to(ROOT).as_posix()} ({dest.stat().st_size/1e6:.1f} MB)")
    except Exception as e:
        print(f"  ✗ {name} 匯出失敗: {repr(e)[:200]}")

print("\n完成。")
