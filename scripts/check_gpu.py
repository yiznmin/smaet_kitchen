"""
診斷 onnxruntime 的 CUDA Execution Provider 是否「真的能用」。

available_providers() 列出 CUDA 只代表「編進去了」,不代表 DLL 載得到。
本腳本建一個極小模型,強制要求 CUDAExecutionProvider,印出 session 實際採用的 provider
與任何載入錯誤,藉此判斷 GPU 環境是否就緒。
"""
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from common.cuda_env import enable_cuda_dlls   # noqa: E402

added = enable_cuda_dlls(verbose=True)          # 掛載 pip 安裝的 CUDA/cuDNN DLL
print(f"掛載 {len(added)} 個 CUDA DLL 目錄\n")

import onnxruntime as ort   # noqa: E402  必須在掛載 DLL 之後

print("onnxruntime", ort.__version__)
print("available providers:", ort.get_available_providers())

# 建一個極小模型:Y = X @ W  (1x8 @ 8x8)
X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 8])
Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 8])
W = helper.make_tensor("W", TensorProto.FLOAT, [8, 8],
                       np.random.RandomState(0).randn(64).astype(np.float32).tobytes(), raw=True)
node = helper.make_node("MatMul", ["X", "W"], ["Y"])
graph = helper.make_graph([node], "tiny", [X], [Y], initializer=[W])
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
model.ir_version = 9
onnx.save(model, "tiny.onnx")

print("\n--- 嘗試用 CUDAExecutionProvider 建 session ---")
try:
    so = ort.SessionOptions()
    sess = ort.InferenceSession("tiny.onnx",
                                providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    used = sess.get_providers()
    print("session 實際使用的 providers:", used)
    out = sess.run(None, {"X": np.ones((1, 8), np.float32)})
    print("推論成功,輸出 shape:", out[0].shape)
    if "CUDAExecutionProvider" in used:
        print("\n✅ GPU(CUDA)可用!")
    else:
        print("\n⚠ CUDA 未被採用,已 fallback 到 CPU(CUDA/cuDNN DLL 可能缺失)")
except Exception as e:
    print("\n❌ 建立 CUDA session 失敗:")
    print(repr(e))
