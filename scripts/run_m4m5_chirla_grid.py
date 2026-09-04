"""跑 `docs/CHIRLA_M4M5驗證_預先登記_20260903.md` §5 的 2×2 受控網格。

**為什麼需要這支而不是手打指令**:4 格 × 7 個評估序列 = 28 次執行。
§5 要求「除標示的因素外,所有設定逐字相同」—— 手打 28 次幾乎一定會有一格不一樣,
而且那種錯誤事後看輸出看不出來。這裡把共用參數寫成單一來源,
只有三個欄位(weights / person_cls / embedder)隨格子改變。

網格(§5):

    偵測器  COCO(RF-DETR nano 預訓,person_cls=1)
            FT  (我們微調的 11 類,person_cls=0)
    外觀    none / dinov2

⚠ **評估集固定,不得挑選**(§3)。算力不足時照 §5.2 **先寫死的順序**
  從尾端移除序列,用 `--drop N` 指定移除幾個,腳本會印出實際跑了哪些。
  不得依「哪個序列結果比較好看」挑選。

⚠ `chirla_armS` 不納入(§5 已寫死):9/1 測得其跨相機 d′ = 0.118,
  低於 dinov2 的 0.250,且系統層差距落在雜訊帶內。

用法:
    python scripts/run_m4m5_chirla_grid.py --root <CHIRLA根> --dry-run
    python scripts/run_m4m5_chirla_grid.py --root <CHIRLA根>
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# §3 定死的評估集,順序即 §5.2 的保留優先序(尾端先被移除)
EVAL_SEQS = ["seq_004", "seq_006", "seq_007", "seq_020", "seq_024", "seq_025", "seq_026"]
DROP_ORDER = ["seq_026", "seq_025", "seq_024", "seq_020", "seq_007", "seq_006"]

CAMERAS = [f"camera_{i}" for i in range(1, 8)]

# 只有這三個欄位隨格子改變 —— 其餘一律共用,由 COMMON 提供
CELLS = {
    "coco_none":  dict(weights=None, person_cls=1, embedder="none"),
    "coco_dino":  dict(weights=None, person_cls=1, embedder="dinov2"),
    "ft_none":    dict(weights="model_result/nano/checkpoint_best_regular.pth",
                       person_cls=0, embedder="none"),
    "ft_dino":    dict(weights="model_result/nano/checkpoint_best_regular.pth",
                       person_cls=0, embedder="dinov2"),
}

COMMON = dict(topology="configs/camera_topology.chirla.yaml",
              variant="nano", thr=0.3, stride=5, fps=30.0, ttl=600)


def videos_for(root, seq):
    d = Path(root) / "videos" / seq
    out = []
    for cam in CAMERAS:
        hits = sorted(d.glob(cam + "_*.avi"))
        if not hits:
            raise SystemExit(f"{seq} 缺 {cam} 的影片")
        out.append(str(hits[0]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out-root", default="results/chirla_m4m5")
    ap.add_argument("--drop", type=int, default=0,
                    help="照 §5.2 的固定順序從尾端移除幾個序列(算力不足時用)")
    ap.add_argument("--cells", nargs="+", default=list(CELLS),
                    help="只跑指定的格子(續跑用)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    seqs = [s for s in EVAL_SEQS if s not in DROP_ORDER[:args.drop]]
    print("=" * 78)
    print("CHIRLA M4/M5 2×2 受控網格")
    print("=" * 78)
    print(f"  評估序列({len(seqs)}/{len(EVAL_SEQS)}):{', '.join(seqs)}")
    if args.drop:
        print(f"  ⚠ 依 §5.2 的固定順序移除:{', '.join(DROP_ORDER[:args.drop])}")
    print(f"  格子:{', '.join(args.cells)}")
    print(f"  共用參數:{COMMON}")

    ft = Path(CELLS["ft_none"]["weights"])
    if any(c.startswith("ft") for c in args.cells) and not ft.exists():
        raise SystemExit(f"缺 {ft} —— 先跑 `git lfs pull`(見 handoff/遠端操作手冊.md 步驟 0.5)")

    jobs, t0 = [], time.time()
    for cell in args.cells:
        cfg = CELLS[cell]
        for seq in seqs:
            out = Path(args.out_root) / cell / seq
            cmd = [sys.executable, str(ROOT / "scripts" / "m5_track_video.py"),
                   "--videos", *videos_for(args.root, seq),
                   "--cameras", *CAMERAS,
                   "--topology", COMMON["topology"],
                   "--variant", COMMON["variant"],
                   "--thr", str(COMMON["thr"]),
                   "--stride", str(COMMON["stride"]),
                   "--fps", str(COMMON["fps"]),
                   "--ttl", str(COMMON["ttl"]),
                   "--max-frames", "-1",
                   "--person-cls", str(cfg["person_cls"]),
                   "--embedder", cfg["embedder"],
                   "--out", str(out / "chef_events.jsonl")]
            if cfg["weights"]:
                cmd += ["--weights", cfg["weights"]]
            jobs.append((cell, seq, out, cmd))

    print(f"\n  共 {len(jobs)} 次執行\n")
    if args.dry_run:
        for cell, seq, _out, cmd in jobs:
            print(f"  [{cell}/{seq}] " + " ".join(
                x if " " not in x else f'"{x}"' for x in cmd[1:]))
        return 0

    done = []
    for i, (cell, seq, out, cmd) in enumerate(jobs, 1):
        if (out / "run_meta.json").exists():
            print(f"[{i}/{len(jobs)}] {cell}/{seq}  已存在,跳過")
            done.append((cell, seq, "skipped"))
            continue
        t = time.time()
        print(f"[{i}/{len(jobs)}] {cell}/{seq}  跑…", flush=True)
        r = subprocess.run(cmd, capture_output=True, text=True,
                           encoding="utf-8", errors="replace")
        dt = time.time() - t
        if r.returncode != 0:
            print(f"    失敗(exit {r.returncode}):{(r.stderr or '')[-600:]}")
            done.append((cell, seq, f"fail:{r.returncode}"))
            continue
        meta = json.loads((out / "run_meta.json").read_text(encoding="utf-8"))
        print(f"    {dt/60:.1f} 分鐘 · {meta['n_loops']:,} 迴圈 · "
              f"{meta['ms_per_loop']['p50']:.0f} ms/迴圈 · chef {meta['total_chefs']}")
        done.append((cell, seq, "ok"))

    print(f"\n  總計 {(time.time()-t0)/3600:.2f} 小時")
    print(f"  {sum(1 for _,_,s in done if s=='ok')} 成功 / "
          f"{sum(1 for _,_,s in done if s=='skipped')} 跳過 / "
          f"{sum(1 for _,_,s in done if s.startswith('fail'))} 失敗")

    # 自檢:四格的 run_meta 除了三個網格因素外必須逐字相同(§5 的核心要求)
    print("\n  自檢:各格參數是否只差在網格因素")
    GRID_KEYS = {"weights", "person_cls", "embedder", "camera_video_map", "videos",
                 "n_det", "n_det_per_cam", "ms_per_loop", "wall_seconds", "n_loops",
                 "n_loops_full", "per_cam_loops", "coverage", "truncated",
                 "total_chefs", "candidate_histogram", "resident_final"}
    base, mismatch = None, []
    for cell in args.cells:
        for seq in seqs:
            p = Path(args.out_root) / cell / seq / "run_meta.json"
            if not p.exists():
                continue
            m = {k: v for k, v in json.loads(p.read_text(encoding="utf-8")).items()
                 if k not in GRID_KEYS}
            if base is None:
                base = (f"{cell}/{seq}", m)
            elif m != base[1]:
                diff = {k for k in set(m) | set(base[1]) if m.get(k) != base[1].get(k)}
                mismatch.append((f"{cell}/{seq}", sorted(diff)))
    if mismatch:
        print(f"    [FAIL] {len(mismatch)} 個執行的非網格參數與 {base[0]} 不同:")
        for name, d in mismatch[:5]:
            print(f"      {name}: {d}")
        return 1
    print(f"    [OK] 所有執行的非網格參數與 {base[0] if base else '—'} 一致")
    return 0


if __name__ == "__main__":
    sys.exit(main())
