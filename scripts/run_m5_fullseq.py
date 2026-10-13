"""多人全長序列的 M5 執行:7 個評估序列 × 7 台,跑完接 eval_m4m5_chirla.py。

`run_m5_fullseq.sh` 的 Python 版(2026-10-13)。共用參數**逐字相同**,只多兩件事:
  · `--topology` 支援 `{seq}` 佔位 —— 平面圖投影含各序列的畫面對齊,每序列一份拓撲
  · 輸出根目錄可指定(原腳本寫死 results/m5_topofix)

⚠ 評估端的拓撲(--eval-topology)只用來做「重疊 / 轉場」路徑分類,
  各格必須用同一份,差異才全部來自系統端。

用法:
    python scripts/run_m5_fullseq.py --name P1 \
        --topology "configs/plan_grid/P1/{seq}.yaml" --out-root results/m5_plan/full
"""
import argparse
import glob
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = str(ROOT / ".venv" / "Scripts" / "python.exe")
SEQS = ["seq_004", "seq_006", "seq_007", "seq_020", "seq_024", "seq_025", "seq_026"]
CAMS = [f"camera_{i}" for i in range(1, 8)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--topology", required=True, help="可含 {seq} 佔位")
    ap.add_argument("--embedder", default="dinov2")
    ap.add_argument("--out-root", default="results/m5_plan/full")
    ap.add_argument("--data-root", default="D:/yizhen/CHIRLA/CHIRLA_data/CHIRLA")
    ap.add_argument("--eval-topology", default="configs/fix_grid/base_overlapfix.yaml")
    ap.add_argument("--seqs", nargs="+", default=SEQS)
    ap.add_argument("--skip-eval", action="store_true")
    args = ap.parse_args()

    out = Path(args.out_root) / args.name
    for seq in args.seqs:
        vids, cams = [], []
        for c in CAMS:
            hit = sorted(glob.glob(f"{args.data_root}/clips_singal_person_result/{seq}/{c}_*.avi"))
            if hit:
                vids.append(hit[0].replace("\\", "/"))
                cams.append(c)
        cmd = [PY, "scripts/m5_track_video.py", "--videos", *vids, "--cameras", *cams,
               "--topology", args.topology.replace("{seq}", seq),
               "--tracker", "configs/tracker_rf_cbiou.yaml",
               "--variant", "nano", "--thr", "0.10", "--person-cls", "1",
               "--embedder", args.embedder, "--fps", "30.0",
               "--max-frames", "-1", "--ttl", "600", "--stride", "5",
               "--det-cache", f"results/det_cache/coco_nano/{seq}",
               "--out", f"{out.as_posix()}/{seq}/chef_events.jsonl"]
        t = time.time()
        (out / seq).mkdir(parents=True, exist_ok=True)
        r = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT,
                           encoding="utf-8", errors="replace")
        (out / seq / "stdout.log").write_text((r.stdout or "") + (r.stderr or ""),
                                              encoding="utf-8")
        print(f"  {args.name} {seq} {'完成' if r.returncode == 0 else '失敗'} "
              f"({time.time() - t:.0f}s)", flush=True)
        if r.returncode:
            print((r.stderr or "")[-1500:])
            return 1
    if args.skip_eval:
        return 0
    cmd = [PY, "scripts/eval_m4m5_chirla.py", "--root", args.data_root,
           "--run-dir", *[f"{out.as_posix()}/{s}" for s in args.seqs],
           "--seq", *args.seqs, "--topology", args.eval_topology, "--out", out.as_posix()]
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT,
                       encoding="utf-8", errors="replace")
    print("\n".join((r.stdout or "").splitlines()[-3:]))
    if r.returncode:
        print((r.stderr or "")[-1500:])
    return r.returncode


if __name__ == "__main__":
    sys.exit(main())
