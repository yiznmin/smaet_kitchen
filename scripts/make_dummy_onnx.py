"""
產生一個「合成的偵測風格 CNN」ONNX,用來驗證 M3 測速 harness(非真實模型,不代表任何模型速度)。

特性:
  - 輸入 [N,3,H,W],N/H/W 皆為動態軸 → 可測 batch 1/4/8 與 imgsz 640/1280
  - 4 層 stride-2 卷積堆疊(下採樣 16x),足以在 GPU 上產生可量測的運算與記憶體佔用
僅供驗證 harness 的計時/記憶體/OOM 邏輯。
"""
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "models" / "_dummy_det.onnx"
OUT.parent.mkdir(parents=True, exist_ok=True)

rng = np.random.RandomState(0)
chans = [3, 24, 48, 96, 96]
nodes, inits = [], []
prev = "input"
for i in range(4):
    w = rng.randn(chans[i + 1], chans[i], 3, 3).astype(np.float32) * 0.1
    wn = f"w{i}"
    inits.append(helper.make_tensor(wn, TensorProto.FLOAT, list(w.shape), w.tobytes(), raw=True))
    nodes.append(helper.make_node("Conv", [prev, wn], [f"c{i}"],
                                  kernel_shape=[3, 3], strides=[2, 2], pads=[1, 1, 1, 1]))
    nodes.append(helper.make_node("Relu", [f"c{i}"], [f"r{i}"]))
    prev = f"r{i}"

inp = helper.make_tensor_value_info("input", TensorProto.FLOAT, ["N", 3, "H", "W"])
out = helper.make_tensor_value_info(prev, TensorProto.FLOAT, ["N", 96, "H16", "W16"])
graph = helper.make_graph(nodes, "dummy_det", [inp], [out], initializer=inits)
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
model.ir_version = 9
onnx.checker.check_model(model)
onnx.save(model, str(OUT))
print(f"已產生合成模型:{OUT.relative_to(ROOT).as_posix()}")
