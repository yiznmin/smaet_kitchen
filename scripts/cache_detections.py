"""把 RF-DETR 的「人」偵測結果快取下來,之後換追蹤參數不必重跑偵測。

## 為什麼

層 0(追蹤碎裂)的受控網格要對**同一批偵測**換門檻、換 stride 重跑追蹤。
CHIRLA 評估集七序列共 483,229 幀,直接跑每一格都要重做偵測(上界約 4.4 小時)。
快取一次之後,每組追蹤參數只剩追蹤本身的時間。

## ⚠ 三件事

1. **直接呼叫 `m5_track_video.detect_person`**,不另寫一份偵測 —— 兩份實作
   只要有一個前處理細節不同,快取就不等價。
2. **以低門檻快取**(預設 0.05),事後才過濾。等價性的理由與 float32 比較的細節
   見 `src/m4_track/det_cache.py` 檔頭。
3. **每支影片快取完都抽幀做逐位比對**(直接用 `--spot-thr` 偵測 vs 快取過濾到同一門檻)。
   不相同就 exit 1 —— 邏輯等價不保證數值等價(例如 GPU 非決定性)。

可續跑:輸出已存在、且 meta 相符並標記完整的影片會跳過。

用法:
    # CHIRLA 評估集(遠端)
    python scripts/cache_detections.py --chirla-root "D:/.../CHIRLA" --out-dir results/det_cache/coco_nano

    # 任意影片(本機 EPFL 驗證用)
    python scripts/cache_detections.py --videos a.mp4 b.mp4 --cameras cam1 cam2 \\
        --out-dir results/det_cache/epfl --max-frames 125
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from common.video_io import iter_frames, video_meta                  # noqa: E402
from m4_track.det_cache import DetCache, save, weights_id            # noqa: E402


def _same(a, b):
    """兩個 Detections 是否逐位相同(含 dtype 與順序)。"""
    if len(a) != len(b):
        return False
    if len(a) == 0:
        return True
    for x, y in ((a.xyxy, b.xyxy), (a.confidence, b.confidence), (a.class_id, b.class_id)):
        if x.dtype != y.dtype or not np.array_equal(x, y):
            return False
    return True


def cache_one(model, detect_person, video, out_path, *, variant, weights, person_cls,
              thr, stride, max_frames, spot_check, spot_thr):
    expect = dict(variant=variant, weights_id=weights_id(weights), person_cls=person_cls,
                  cache_thr=thr, cache_stride=stride, complete=max_frames < 0)
    if Path(out_path).exists():
        try:
            DetCache(out_path, expect=expect)
            print(f"  ⏭ 已存在且相符,跳過:{out_path}")
            return True
        except (ValueError, KeyError) as e:
            print(f"  ↻ 既有快取不符({e}),重做:{out_path}")

    vm = video_meta(video)
    total = vm["nb_frames"] // stride if max_frames < 0 else min(max_frames, vm["nb_frames"])
    fids, xy, cf, cl = [], [], [], []
    n, last, t0 = 0, -1, time.time()
    for fid, _t, frame in iter_frames(video, stride=stride):
        if max_frames >= 0 and n >= max_frames:
            break
        det = detect_person(model, frame, thr, person_cls)
        if len(det):
            fids.append(np.full(len(det), fid, dtype=np.int64))
            xy.append(det.xyxy)
            cf.append(det.confidence)
            cl.append(det.class_id)
        n, last = n + 1, fid
        if n % 1000 == 0:
            rate = n / (time.time() - t0)
            print(f"    {n:>7,}/{total:,} 幀  {rate:5.1f} 幀/秒  "
                  f"剩約 {(total - n) / max(rate, 1e-9) / 60:.0f} 分")

    save(out_path,
         np.concatenate(fids) if fids else np.zeros(0, dtype=np.int64),
         np.concatenate(xy) if xy else np.zeros((0, 4), dtype=np.float32),
         np.concatenate(cf) if cf else np.zeros(0, dtype=np.float32),
         np.concatenate(cl) if cl else np.zeros(0, dtype=np.int64),
         dict(expect, n_scanned=n, max_fid=last, video=str(video), video_meta=vm,
              n_det=int(sum(len(a) for a in cf)), seconds=round(time.time() - t0, 1)))
    print(f"  ✓ {out_path}  {n:,} 幀、{sum(len(a) for a in cf):,} 個框、"
          f"{time.time() - t0:.0f} 秒")

    if spot_check <= 0:
        return True
    # ⚠ 抽幀逐位比對:直接偵測@spot_thr vs 快取過濾到 spot_thr
    cache = DetCache(out_path, expect=expect)
    bad = checked = 0
    for fid, _t, frame in iter_frames(video, stride=stride):
        if checked >= spot_check:
            break
        direct = detect_person(model, frame, spot_thr, person_cls)
        if not _same(direct, cache.get(fid, spot_thr)):
            bad += 1
            if bad <= 3:
                print(f"    ✗ fid={fid}:直接 {len(direct)} 框 vs 快取 {len(cache.get(fid, spot_thr))} 框")
        checked += 1
    ok = bad == 0
    print(f"  {'✅' if ok else '❌'} 抽 {checked} 幀逐位比對(門檻 {spot_thr}):不同 {bad} 幀")
    return ok


def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--chirla-root", help="CHIRLA 根目錄;跑評估集全部序列 × 七台")
    src.add_argument("--videos", nargs="+")
    ap.add_argument("--cameras", nargs="+", help="與 --videos 一一對應")
    ap.add_argument("--seqs", nargs="+", default=None, help="只跑指定序列(CHIRLA)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--variant", default="nano")
    ap.add_argument("--weights", default=None)
    ap.add_argument("--person-cls", type=int, default=None)
    ap.add_argument("--thr", type=float, default=0.05,
                    help="快取門檻。之後事後過濾只能往上不能往下")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--max-frames", type=int, default=-1, help="-1 = 整支影片")
    ap.add_argument("--spot-check", type=int, default=50, help="抽幾幀做逐位比對;0 = 不做")
    ap.add_argument("--spot-thr", type=float, default=0.3)
    args = ap.parse_args()

    from m5_track_video import (PERSON_CLS_COCO, PERSON_CLS_FINETUNED,
                                detect_person, load_model)
    person_cls = (args.person_cls if args.person_cls is not None
                  else (PERSON_CLS_FINETUNED if args.weights else PERSON_CLS_COCO))
    if args.spot_thr < args.thr:
        raise SystemExit("--spot-thr 不可低於 --thr")

    jobs = []
    out_dir = Path(args.out_dir)
    if args.chirla_root:
        from run_m4m5_chirla_grid import CAMERAS, EVAL_SEQS, videos_for
        for seq in (args.seqs or EVAL_SEQS):
            for cam, v in zip(CAMERAS, videos_for(args.chirla_root, seq)):
                jobs.append((f"{seq}/{cam}", v, out_dir / seq / f"{cam}.npz"))
    else:
        if not args.cameras or len(args.cameras) != len(args.videos):
            raise SystemExit("--cameras 必須與 --videos 一一對應")
        for cam, v in zip(args.cameras, args.videos):
            jobs.append((cam, v, out_dir / f"{cam}.npz"))

    print(f"模型 {args.variant}{'/微調' if args.weights else '/COCO 預訓'},"
          f"person_cls={person_cls},快取門檻 {args.thr},stride {args.stride},共 {len(jobs)} 支")
    model = load_model(args.variant, args.weights)
    all_ok, t0 = True, time.time()
    for i, (name, video, out_path) in enumerate(jobs, 1):
        print(f"\n[{i}/{len(jobs)}] {name}")
        all_ok &= cache_one(model, detect_person, video, out_path,
                            variant=args.variant, weights=args.weights,
                            person_cls=person_cls, thr=args.thr, stride=args.stride,
                            max_frames=args.max_frames, spot_check=args.spot_check,
                            spot_thr=args.spot_thr)
    print(f"\n完成,{(time.time() - t0) / 60:.1f} 分鐘。"
          f"{'全部逐位比對通過' if all_ok else '⚠ 有影片逐位比對失敗 —— 快取不可用'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
