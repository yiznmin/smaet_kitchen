"""遠端環境自檢 —— 開始訓練前先確認這台機器跑得動。

在遠端跑一次,把輸出貼回來就知道還缺什麼。每一項都對應一個「不滿足就會卡住」的
實際需求,不是把有的套件列一列而已。

⚠ 特別檢查 Python 版本:本專案已經吃過虧 —— Python 3.14 上 YOLOX / DAMO-YOLO /
  LW-DETR 三個模型直接匯不出來(2021 年的程式碼與 numpy 2.3 / torch 2.10 不相容)。
  Re-ID 生態的框架(torchreid、fast-reid)多半也是 2019~2022 年的碼,同一個風險。

用法:python scripts/check_reid_env.py [--data-dir /打算放 CHIRLA 的路徑]
"""
import argparse
import importlib
import platform
import shutil
import sys

WARN, FAIL = [], []


def ok(name, detail=""):
    print(f"  ✅ {name}" + (f"  —— {detail}" if detail else ""))


def warn(name, detail="", fix=""):
    print(f"  ⚠ {name}" + (f"  —— {detail}" if detail else ""))
    if fix:
        print(f"       → {fix}")
    WARN.append(name)


def fail(name, detail="", fix=""):
    print(f"  ❌ {name}" + (f"  —— {detail}" if detail else ""))
    if fix:
        print(f"       → {fix}")
    FAIL.append(name)


def check_python():
    print("\n[1] Python 版本")
    v = sys.version_info
    s = f"{v.major}.{v.minor}.{v.micro} ({platform.machine()})"
    if v[:2] in ((3, 10), (3, 11), (3, 12)):
        ok(f"Python {s}", "Re-ID 生態的相容甜蜜點")
    elif v[:2] >= (3, 13):
        fail(f"Python {s} 太新",
             "本專案在 3.14 上已經有三個模型匯不出來;Re-ID 框架多為 2019~2022 年的碼",
             "建 3.11 環境:conda create -n reid python=3.11 / python3.11 -m venv .venv")
    else:
        warn(f"Python {s} 偏舊", "torch 新版可能不支援")


def check_pkg(mod, why, pip_name=None, required=True):
    try:
        m = importlib.import_module(mod)
        ver = getattr(m, "__version__", "?")
        ok(f"{mod} {ver}", why)
        return m
    except ImportError:
        (fail if required else warn)(
            f"缺 {mod}", why, f"pip install {pip_name or mod}")
        return None


def check_torch():
    print("\n[2] PyTorch 與 GPU")
    torch = check_pkg("torch", "訓練", "torch(依 CUDA 版本挑 wheel:pytorch.org)")
    if torch is None:
        return None
    check_pkg("torchvision", "ResNet50 骨幹與 ImageNet 預訓權重")

    if not torch.cuda.is_available():
        fail("CUDA 不可用", f"torch {torch.__version__} 看不到 GPU",
             "確認裝的是 CUDA 版 torch(檔名帶 +cu12x)而不是 +cpu;"
             "並確認 nvidia-smi 正常")
        return torch

    n = torch.cuda.device_count()
    for i in range(n):
        p = torch.cuda.get_device_properties(i)
        gb = p.total_memory / 1e9
        detail = f"{p.name}, {gb:.1f} GB VRAM, CC {p.major}.{p.minor}"
        # ResNet50 + last-stride-1 + 256x128 + batch 32,含梯度約需 6~8 GB
        if gb >= 10:
            ok(f"GPU {i}", detail + " —— 充裕,可用預設 --P 8 --K 4")
        elif gb >= 6:
            warn(f"GPU {i}", detail + " —— 勉強",
                 "batch 調小:--P 6 --K 4,或加 --freeze-until 5")
        else:
            fail(f"GPU {i}", detail + " —— VRAM 不足",
                 "--P 4 --K 4 且 --freeze-until 6;仍不行就要換機器")
    print(f"       torch {torch.__version__} / CUDA {torch.version.cuda}")
    return torch


def check_gpu_forward(torch):
    """能列出 GPU ≠ 真的算得動。實際跑一次 forward+backward。"""
    print("\n[3] GPU 實跑(列得出來不代表算得動)")
    if torch is None or not torch.cuda.is_available():
        warn("跳過", "沒有可用的 CUDA")
        return
    try:
        import torch.nn as nn
        from torchvision.models import resnet50
        m = resnet50(weights=None).cuda()
        x = torch.randn(8, 3, 256, 128, device="cuda")
        loss = nn.functional.cross_entropy(m(x), torch.zeros(8, dtype=torch.long, device="cuda"))
        loss.backward()
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated() / 1e9
        ok("ResNet50 forward+backward 成功", f"batch 8 峰值 {peak:.2f} GB "
           f"→ batch 32 推估約 {peak*4:.1f} GB")
    except Exception as e:
        fail("GPU 實跑失敗", f"{type(e).__name__}: {e}",
             "常見原因:CUDA 與驅動版本不合、VRAM 被其他人佔用")


def check_rest():
    print("\n[4] 其他依賴")
    check_pkg("cv2", "影像讀寫。⚠ 中文路徑要用 imdecode(np.fromfile(...)) 不能用 imread",
              "opencv-python")
    check_pkg("numpy", "指標計算")
    check_pkg("scipy", "匈牙利配對(M5 指標)")
    check_pkg("h5py", "匯出 CHIRLA 官方要求的 HDF5 embedding 格式", required=True)
    check_pkg("yaml", "讀 config", "pyyaml")
    check_pkg("huggingface_hub", "下載 CHIRLA(gated,需先 login)",
              "huggingface_hub[cli]", required=False)


def check_disk(path):
    print(f"\n[5] 磁碟空間({path})")
    try:
        u = shutil.disk_usage(path)
    except OSError as e:
        fail("讀不到磁碟資訊", str(e))
        return
    free = u.free / 1e9
    # CHIRLA 10.9GB + 解壓/快取 + checkpoint(每個 ~100MB)+ 匯出的 embedding
    need = 40
    detail = f"可用 {free:.1f} GB / 總計 {u.total/1e9:.1f} GB"
    if free >= need:
        ok("空間充裕", detail + f"(CHIRLA 10.9GB + 工作空間,建議留 {need} GB)")
    elif free >= 20:
        warn("空間勉強", detail, "先只下載 benchmark/ 與 annotations/,不要下載 videos/")
    else:
        fail("空間不足", detail, f"至少要 {need} GB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=".", help="打算放 CHIRLA 的路徑")
    args = ap.parse_args()

    print("=" * 70)
    print("CHIRLA Re-ID 訓練環境自檢")
    print("=" * 70)
    print(f"  {platform.platform()}")

    check_python()
    torch = check_torch()
    check_gpu_forward(torch)
    check_rest()
    check_disk(args.data_dir)

    print("\n" + "=" * 70)
    if FAIL:
        print(f"❌ {len(FAIL)} 項必須解決:{FAIL}")
        print("   解決前不要開始訓練 —— 環境問題在訓練跑到一半才爆最浪費時間。")
        return 1
    if WARN:
        print(f"⚠ 可以開始,但有 {len(WARN)} 項要注意:{WARN}")
        return 0
    print("✅ 環境齊備,可以開始。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
