"""六個子集的執行:每個獨立窗 × 每組設定各跑一次 `integrated/run_pipeline.py`。

預先登記:docs/六子集測試_預先登記_20261013.md §3。

- 每次執行是一個**新 process**(逐片段獨立推論的語意就是從零開始)。
- 以 `run_id` 為單位:多人 L1 一個窗有多個目標,只需要跑一次。
- 平行度只影響速度:各次執行互不共用狀態,輸出寫在各自的資料夾。
- 已有 `run_meta.json` 的執行會跳過(中斷後可接續);`--force` 重跑。

用法:
    python scripts/six_subsets_run.py --profiles delivery single_person plan_strict plan_share
"""
import argparse
import csv
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = str(ROOT / ".venv" / "Scripts" / "python.exe")
PIPELINE = str(ROOT / "integrated" / "run_pipeline.py")


def job(profile, run, out_root, force, weights=None, det_root=None):
    out = out_root / profile / "runs" / run["run_id"]
    if (out / "run_meta.json").exists() and not force:
        return dict(profile=profile, run_id=run["run_id"], returncode=0, wall_s=0.0, skipped=True)
    out.mkdir(parents=True, exist_ok=True)
    cmd = [PY, PIPELINE, "--profile", profile, "--seq", run["seq"],
           "--videos", *[run["videos"][c] for c in run["cameras"]],
           "--cameras", *run["cameras"],
           "--det-cache", str(ROOT / (f"{det_root}/{run['seq']}" if det_root
                                      else run["det_cache"])),
           "--start-fid", str(run["start_fid"]), "--end-fid", str(run["end_fid"]),
           "--out", str(out)]
    if weights:          # M3 微調權重(2026-10-13):偵測快取必須是同一顆權重建的,DetCache 會核對
        cmd += ["--weights", weights]
    t = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace",
                       env={**__import__("os").environ, "PYTHONIOENCODING": "utf-8"})
    (out / "stdout.log").write_text((r.stdout or "") + (r.stderr or ""), encoding="utf-8")
    return dict(profile=profile, run_id=run["run_id"], returncode=r.returncode,
                wall_s=round(time.time() - t, 2), skipped=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="results/six_subsets/manifest.json")
    ap.add_argument("--out-root", default="results/six_subsets")
    ap.add_argument("--profiles", nargs="+",
                    default=["delivery", "single_person", "plan_strict", "plan_share"])
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 個窗(試跑用)")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--weights", default=None,
                    help="M3 微調權重(相對於 integrated/);不給 = COCO 預訓")
    ap.add_argument("--det-cache-root", default=None,
                    help="偵測快取根目錄(其下為 <seq>/<camera>.npz);不給 = 片段清單記的那一份")
    args = ap.parse_args()

    man = json.loads((ROOT / args.manifest).read_text(encoding="utf-8"))
    runs = {}
    for c in man["clips"]:
        runs.setdefault(c["run_id"], {k: c[k] for k in (
            "run_id", "seq", "cameras", "videos", "det_cache", "start_fid", "end_fid")})
    runs = sorted(runs.values(), key=lambda r: r["run_id"])
    if args.limit:
        runs = runs[:args.limit]
    out_root = ROOT / args.out_root
    jobs = [(p, r) for p in args.profiles for r in runs]
    print(f"{len(runs)} 個窗 × {len(args.profiles)} 組設定 = {len(jobs)} 次執行,"
          f"平行 {args.workers}", flush=True)

    rows, t0 = [], time.time()
    with ThreadPoolExecutor(args.workers) as ex:
        for i, res in enumerate(ex.map(lambda a: job(a[0], a[1], out_root, args.force, args.weights,
                                                       args.det_cache_root), jobs), 1):
            rows.append(res)
            if res["returncode"]:
                print(f"  失敗:{res['profile']} {res['run_id']}", flush=True)
            if i % 200 == 0 or i == len(jobs):
                print(f"[{i}/{len(jobs)}] {time.time() - t0:.0f}s", flush=True)

    for p in args.profiles:
        mine = [r for r in rows if r["profile"] == p]
        with open(out_root / p / "run_index.csv", "w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(mine[0]))
            w.writeheader()
            w.writerows(mine)
    bad = [r for r in rows if r["returncode"]]
    print(f"總耗時 {time.time() - t0:.0f} 秒;失敗 {len(bad)} 次", flush=True)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
