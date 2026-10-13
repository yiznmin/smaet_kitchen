"""四組設定在 7 個評估序列全長上的執行(走 integrated/run_pipeline.py),跑完接 eval_m4m5_chirla.py。

登記:docs/六子集測試_預先登記_20261013.md §10。

與 `run_m5_fullseq.py` 的差別:那一支呼叫舊的 `m5_track_video.py`,M4 固定 C-BIoU;
這一支呼叫整合管線,M4 / M5 / 拓撲全部由 profile 決定 —— 量的是「整合版照設定檔跑」的結果。

⚠ 評估端的拓撲(--eval-topology)只用來把決策分成重疊 / 轉場兩條路徑,四組共用同一份。

用法:
    python scripts/full_seq_run.py
"""
import argparse
import glob
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = str(ROOT / ".venv" / "Scripts" / "python.exe")
SEQS = ["seq_004", "seq_006", "seq_007", "seq_020", "seq_024", "seq_025", "seq_026"]
CAMS = [f"camera_{i}" for i in range(1, 8)]


def run_one(profile, seq, out_root, data_root, weights=None, det_root="results/det_cache/coco_nano"):
    out = out_root / profile / seq
    if (out / "run_meta.json").exists():
        return profile, seq, 0, 0.0
    out.mkdir(parents=True, exist_ok=True)
    vids, cams = [], []
    for c in CAMS:
        hit = sorted(glob.glob(f"{data_root}/clips_singal_person_result/{seq}/{c}_*.avi"))
        if hit:
            vids.append(hit[0].replace("\\", "/"))
            cams.append(c)
    cmd = [PY, str(ROOT / "integrated" / "run_pipeline.py"), "--profile", profile, "--seq", seq,
           "--videos", *vids, "--cameras", *cams,
           "--det-cache", str(ROOT / det_root / seq),
           "--out", str(out)]
    if weights:
        cmd += ["--weights", weights]
    t = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace",
                       env={**__import__("os").environ, "PYTHONIOENCODING": "utf-8"})
    (out / "stdout.log").write_text((r.stdout or "") + (r.stderr or ""), encoding="utf-8")
    return profile, seq, r.returncode, round(time.time() - t, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profiles", nargs="+",
                    default=["delivery", "single_person", "plan_strict", "plan_share"])
    ap.add_argument("--out-root", default="results/full_seq_integrated")
    ap.add_argument("--data-root", default="D:/yizhen/CHIRLA/CHIRLA_data/CHIRLA")
    ap.add_argument("--eval-topology", default="configs/fix_grid/base_overlapfix.yaml")
    ap.add_argument("--workers", type=int, default=7)
    ap.add_argument("--weights", default=None, help="M3 微調權重(相對於 integrated/)")
    ap.add_argument("--det-cache-root", default="results/det_cache/coco_nano")
    args = ap.parse_args()
    out_root = ROOT / args.out_root
    jobs = [(p, s) for p in args.profiles for s in SEQS]
    bad = 0
    with ThreadPoolExecutor(args.workers) as ex:
        for p, s, rc, sec in ex.map(lambda a: run_one(a[0], a[1], out_root, args.data_root, args.weights,
                                                 args.det_cache_root), jobs):
            print(f"  {p} {s} {'完成' if rc == 0 else '失敗'} ({sec}s)", flush=True)
            bad += int(rc != 0)
    if bad:
        return 1
    for p in args.profiles:
        cmd = [PY, "scripts/eval_m4m5_chirla.py", "--root", args.data_root,
               "--run-dir", *[f"{args.out_root}/{p}/{s}" for s in SEQS], "--seq", *SEQS,
               "--topology", args.eval_topology, "--out", f"{args.out_root}/{p}"]
        r = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT, encoding="utf-8",
                           errors="replace",
                           env={**__import__("os").environ, "PYTHONIOENCODING": "utf-8"})
        print(p, "評估", "完成" if r.returncode == 0 else "失敗", flush=True)
        if r.returncode:
            print((r.stdout or "")[-800:], (r.stderr or "")[-800:])
    return 0


if __name__ == "__main__":
    sys.exit(main())
