"""用 M3 微調權重對 CHIRLA 10 個序列 × 7 台建立偵測快取(每序列一個 process,平行跑)。

登記:docs/六子集測試_預先登記_20261013.md §12。

只是 `cache_detections.py` 的外層:那一支的 `--chirla-root` 模式找的是 `videos/` 資料夾
(2026-09-28 已改名),所以這裡改用它的 `--videos` 模式逐序列呼叫。偵測本身一行都沒有另寫。
"""
import glob
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = str(ROOT / ".venv" / "Scripts" / "python.exe")
DATA = "D:/yizhen/CHIRLA/CHIRLA_data/CHIRLA/clips_singal_person_result"
WEIGHTS = "integrated/weights/m3_nano_r2_armA/checkpoint_best_total.pth"
OUT = "results/det_cache/ft_armA_s5"
SEQS = ["seq_000", "seq_001", "seq_002", "seq_004", "seq_006", "seq_007", "seq_020",
        "seq_024", "seq_025", "seq_026"]


def one(seq):
    vids, cams = [], []
    for i in range(1, 8):
        hit = sorted(glob.glob(f"{DATA}/{seq}/camera_{i}_*.avi"))
        if hit:
            vids.append(hit[0])
            cams.append(f"camera_{i}")
    cmd = [PY, "scripts/cache_detections.py", "--videos", *vids, "--cameras", *cams,
           "--out-dir", f"{OUT}/{seq}", "--weights", WEIGHTS, "--stride", "5",
           "--thr", "0.05", "--spot-check", "50", "--spot-thr", "0.3"]
    t = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT, encoding="utf-8",
                       errors="replace",
                       env={**__import__("os").environ, "PYTHONIOENCODING": "utf-8"})
    log = ROOT / OUT / seq / "cache.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text((r.stdout or "") + (r.stderr or ""), encoding="utf-8")
    tail = (r.stdout or "").strip().splitlines()[-1:] or [""]
    print(f"  {seq} rc={r.returncode} {time.time() - t:.0f}s {len(cams)} 台 | {tail[0]}", flush=True)
    return r.returncode


def main():
    with ThreadPoolExecutor(3) as ex:
        rcs = list(ex.map(one, SEQS))
    return 1 if any(rcs) else 0


if __name__ == "__main__":
    sys.exit(main())
