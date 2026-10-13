"""
讓 onnxruntime-gpu 找得到由 pip 安裝(nvidia-*-cu12 wheels)的 CUDA / cuDNN DLL。

這些 wheel 把 DLL 裝在 site-packages/nvidia/<component>/bin(Windows),
ORT 預設不會去那裡找,因此須在 import/建立 session 前用 os.add_dll_directory 掛載。

用法:在使用 onnxruntime 前先 `from common.cuda_env import enable_cuda_dlls; enable_cuda_dlls()`
"""
import os
import sys
from pathlib import Path


def enable_cuda_dlls(verbose=False):
    """掃描所有 nvidia-*-cu12 套件的 DLL 目錄,加入 Windows DLL 搜尋路徑。回傳加入的目錄清單。"""
    added = []
    if sys.platform != "win32":
        return added
    for site in sys.path:
        nvidia = Path(site) / "nvidia"
        if not nvidia.is_dir():
            continue
        # 每個元件目錄下找 bin / lib 裡的 .dll
        for comp in nvidia.iterdir():
            for sub in ("bin", "lib"):
                d = comp / sub
                if d.is_dir() and any(d.glob("*.dll")):
                    try:
                        os.add_dll_directory(str(d))
                        os.environ["PATH"] = str(d) + os.pathsep + os.environ.get("PATH", "")
                        added.append(str(d))
                        if verbose:
                            print(f"  + DLL dir: {d}")
                    except OSError:
                        pass
    return added


if __name__ == "__main__":
    dirs = enable_cuda_dlls(verbose=True)
    print(f"共加入 {len(dirs)} 個 DLL 目錄")
