"""
M3 物件偵測「速度/資源」benchmark(模型無關)。

對 configs/benchmark.yaml 列出的每個 ONNX 模型,在 imgsz × batch 組合下量測:
  - 延遲 latency(mean / p95,毫秒,每個 batch 一次推論)
  - 吞吐 throughput(images/sec)
  - GPU 記憶體峰值(MB,相對 baseline 的增量;用 pynvml,fallback 無)
  - 可支援鏡頭數(throughput ÷ target_fps)
失敗(OOM / 固定shape衝突 / 缺檔)該列標記並繼續,不中斷整體。

輸出:results/m3/m3_speed_table.csv + results/m3/m3_speed.md(含執行環境)。

⚠ 數字僅在「實際部署的目標主機」上才可作選型依據(說明文件 C.4 #4)。
   本開發機 RTX 3050 4GB 的結果僅供流程驗證。

用法:
  python scripts/bench_m3_speed.py
  python scripts/bench_m3_speed.py --device cpu     # 強制 CPU
"""
import argparse
import csv
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from common.cuda_env import enable_cuda_dlls   # noqa: E402

CUDA_DIRS = enable_cuda_dlls()                  # 必須在 import onnxruntime 前
import onnxruntime as ort                       # noqa: E402

CONFIG = ROOT / "configs" / "benchmark.yaml"
OUT = ROOT / "results" / "m3"


# ---------- GPU 記憶體量測 ----------
class GpuMem:
    def __init__(self):
        self.ok = False
        try:
            import pynvml
            pynvml.nvmlInit()
            self.h = pynvml.nvmlDeviceGetHandleByIndex(0)
            self.pynvml = pynvml
            self.name = pynvml.nvmlDeviceGetName(self.h)
            if isinstance(self.name, bytes):
                self.name = self.name.decode()
            self.ok = True
        except Exception:
            self.name = "unknown"

    def used_mb(self):
        if not self.ok:
            return None
        return self.pynvml.nvmlDeviceGetMemoryInfo(self.h).used / 1024 / 1024


def env_info(providers, gpu):
    di = ort.get_device()
    return {
        "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "onnxruntime": ort.__version__,
        "ort_device": di,
        "providers": providers,
        "gpu": gpu.name,
        "cuda_dll_dirs": len(CUDA_DIRS),
    }


def peek_input_shape(onnx_path):
    """讀 ONNX 第一個輸入的 [batch, c, h, w];動態軸回傳 None。"""
    import onnx
    m = onnx.load(str(onnx_path))
    dims = m.graph.input[0].type.tensor_type.shape.dim
    vals = [(d.dim_value if d.HasField("dim_value") else None) for d in dims]
    vals = (vals + [None, None, None, None])[:4]
    return {"batch": vals[0], "c": vals[1], "h": vals[2], "w": vals[3]}


def bench_one(sess, in_name, batch, h, w, warmup, iters, gpu, base_global):
    """回傳 (latency_mean_ms, p95_ms, throughput_fps, gpu_peak_mb) 或丟出例外。
    gpu_peak_mb = 本程序在此組合下用掉的 VRAM 峰值(相對啟動前 baseline,含 CUDA context)。"""
    x = np.random.rand(batch, 3, h, w).astype(np.float32)
    peak = base_global or 0

    def sample():
        nonlocal peak
        u = gpu.used_mb()
        if u is not None and u > peak:
            peak = u

    for _ in range(warmup):       # 暖機期間記憶體就會配置,須一起取峰值
        sess.run(None, {in_name: x})
        sample()

    lat = []
    for _ in range(iters):
        t0 = time.perf_counter()
        sess.run(None, {in_name: x})
        lat.append((time.perf_counter() - t0) * 1000)
        sample()
    mean = statistics.mean(lat)
    p95 = sorted(lat)[max(0, int(len(lat) * 0.95) - 1)]
    throughput = batch * 1000.0 / mean
    gpu_peak = round(peak - base_global, 1) if (base_global is not None) else None
    return round(mean, 3), round(p95, 3), round(throughput, 1), gpu_peak


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", choices=["cuda", "cpu"], default=None)
    ap.add_argument("--onnx", help="只測這一個 onnx 檔(驗證 harness 用)")
    ap.add_argument("--name", default="adhoc", help="搭配 --onnx 的顯示名稱")
    ap.add_argument("--imgsz", type=int, nargs="+",
                    help="覆寫解析度清單(模型輸入尺寸固定時用,如 --imgsz 320)")
    ap.add_argument("--config", help="改用其他設定檔(預設 configs/benchmark.yaml)")
    ap.add_argument("--tag", default="", help="輸出檔名後綴(避免覆蓋,如 fair672)")
    args = ap.parse_args()

    cfg_path = Path(args.config) if args.config else CONFIG
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    m3 = cfg["m3_speed"]
    if args.onnx:
        m3["models"] = [dict(name=args.name, license="(test)",
                             shippable=False, onnx=args.onnx)]
    if args.imgsz:
        m3["imgsz"] = args.imgsz
    device = args.device or m3.get("device", "cuda")
    providers = (["CUDAExecutionProvider", "CPUExecutionProvider"]
                 if device == "cuda" else ["CPUExecutionProvider"])

    gpu = GpuMem()
    OUT.mkdir(parents=True, exist_ok=True)

    base_global = gpu.used_mb()          # 啟動前 GPU 已用量(桌面等),作為記憶體量測基準
    rows = []
    print(f"providers 要求: {providers}; GPU baseline {base_global} MB")
    def mkrow(name, lic, ship, size, batch, provider="", status=""):
        return dict(model=name, license=lic, shippable=ship, imgsz=size, batch=batch,
                    latency_ms_mean="", latency_ms_p95="", throughput_fps="",
                    gpu_mem_mb="", max_cameras="", provider=provider, status=status)

    for m in m3["models"]:
        onnx_path = ROOT / m["onnx"]
        if not onnx_path.exists():
            rows.append(mkrow(m["name"], m["license"], m["shippable"], "-", "-", status="MISSING_ONNX"))
            print(f"  [MISSING_ONNX] {m['name']}: {m['onnx']}")
            continue
        # session 每個模型建一次(動態軸可重用於不同 batch/尺寸)
        try:
            sess = ort.InferenceSession(str(onnx_path), providers=providers)
            in_name = sess.get_inputs()[0].name
            sess_provider = sess.get_providers()[0]
        except Exception as e:
            rows.append(mkrow(m["name"], m["license"], m["shippable"], "-", "-", status="LOAD_ERROR"))
            print(f"  [LOAD_ERROR] {m['name']}: {repr(e)[:120]}")
            continue

        # 依模型輸入決定有效尺寸/批次:固定→用模型自己的;動態→套 config
        shp = peek_input_shape(onnx_path)
        eff_sizes = [(shp["h"], shp["w"])] if (shp["h"] and shp["w"]) else [(s, s) for s in m3["imgsz"]]
        eff_batches = [shp["batch"]] if isinstance(shp["batch"], int) and shp["batch"] else m3["batch_sizes"]

        for (h, w) in eff_sizes:
            size_label = f"{w}" if h == w else f"{w}x{h}"
            for batch in eff_batches:
                row = mkrow(m["name"], m["license"], m["shippable"], size_label, batch, sess_provider)
                try:
                    mean, p95, thr, gmem = bench_one(
                        sess, in_name, batch, h, w,
                        m3["warmup_iters"], m3["timed_iters"], gpu, base_global)
                    row.update(latency_ms_mean=mean, latency_ms_p95=p95,
                               throughput_fps=thr, gpu_mem_mb=gmem,
                               max_cameras=int(thr // m3["target_fps"]), status="OK")
                    print(f"  [OK] {m['name']:14} {size_label:>9} b{batch} "
                          f"-> {mean}ms/p95 {p95}ms, {thr} img/s, gpu+{gmem}MB, {sess_provider}")
                except Exception as e:
                    msg = repr(e)
                    row["status"] = "OOM" if "memory" in msg.lower() or "oom" in msg.lower() else "ERROR"
                    print(f"  [{row['status']}] {m['name']} {size_label} b{batch}: {msg[:110]}")
                rows.append(row)

    # 寫 CSV
    tag = f"_{args.tag}" if args.tag else ""
    csv_path = OUT / f"m3_speed_table{tag}.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # 寫報告
    env = env_info(providers, gpu)
    ok_rows = [r for r in rows if r["status"] == "OK"]
    md = [
        "# M3 物件偵測速度 benchmark",
        "",
        "## 執行環境(數據來源)",
        f"- 日期:{env['date']}",
        f"- GPU:**{env['gpu']}**",
        f"- onnxruntime:{env['onnxruntime']}(device={env['ort_device']})",
        f"- 要求 providers:{env['providers']}",
        f"- 掛載的 CUDA DLL 目錄數:{env['cuda_dll_dirs']}",
        "",
        "> ⚠ 本機若為 RTX 3050(開發機),數字**僅供流程驗證,非選型依據**;",
        "> 最終選型須於目標部署主機重跑(說明文件 C.4 #4)。",
        "",
        "## 結果摘要",
        f"- 總組合:{len(rows)};成功 {len(ok_rows)};"
        f"失敗 {len(rows) - len(ok_rows)}(OOM/ERROR/缺檔)。",
        "- 完整數據見 `m3_speed_table.csv`。",
        "",
        "| model | license | imgsz | batch | lat_mean(ms) | p95 | img/s | gpu(MB) | 鏡頭數 | provider |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        if r["status"] == "OK":
            md.append(f"| {r['model']} | {r['license']} | {r['imgsz']} | {r['batch']} | "
                      f"{r['latency_ms_mean']} | {r['latency_ms_p95']} | {r['throughput_fps']} | "
                      f"{r['gpu_mem_mb']} | {r['max_cameras']} | {r['provider']} |")
        else:
            md.append(f"| {r['model']} | {r['license']} | {r['imgsz']} | {r['batch']} | "
                      f"— | — | — | — | — | **{r['status']}** |")
    md_path = OUT / f"m3_speed{tag}.md"
    md_path.write_text("\n".join(md), encoding="utf-8")
    print(f"\n報告:{md_path.relative_to(ROOT).as_posix()}")
    print(f"數據:{csv_path.relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    main()
