"""層 0 B 類斷點診斷 —— 把 2026-09-14 預先登記 §8.8.1 / §8.8.2 的分析做成可重現腳本。

B 類 = 同一真人、同一鏡頭的相鄰兩條 track,後一條出生時前一條還在 lost 緩衝內。
9/14 遠端以唯讀 python 臨時分析得到兩個關鍵數字,但**沒有提交腳本**。這支回答同樣兩件事:

  §8.8.1 斷點期間偵測器有沒有給框?   → 9/14:追蹤器看得到的幀 72.4% 有 ≥0.30 的正確框
  §8.8.2 為什麼沒接上?               → 9/14:舊 track 最後真實框 vs 新 track 第一框 IoU 中位 0.199,
                                         < 0.2 佔 50.4%,≥ 0.5 佔 16.9%

之後每一輪換配對方法,都用它看**剩下的** B 落在哪個 IoU 區間:
放大框的方法(C-BIoU)該吃掉 < 0.2 那段,修漂移的方法(OC-SORT)該吃掉 ≥ 0.5 那段。
只看 B 總數降了多少,分不出方法是不是照它宣稱的機制在作用。

## 口徑

  - 間隔:預設 (跟丟事件的影格, 新 track 出生的影格),兩端都不含
    (§8.8.1「37.5% 間隔內沒有追蹤器看得到的幀,多為新 track 在下一次更新就出生」
     只有兩端都不含才會出現);`--interval lost-inclusive` 含跟丟那一幀
  - 真人在畫面裡:該幀 GT 在這台鏡頭有這個 id(⚠ GT 1-based、影格 0-based)
  - 有框:快取裡與 GT 框 IoU ≥ 0.5(全專案評估標準)的偵測,記最高分
  - 追蹤器看得到的幀:影格是 stride 的倍數
  - 最後真實框:前一條 track 在跟丟之前、confidence 不是 None 的最後一列(排除卡爾曼預測框)
  - ⚠ IoU 是**代理量**:追蹤器實際比的是卡爾曼預測框,不是最後真實框
  - ⚠ 9/14 可能是從 `--dump-dir` 的 CSV(座標小數 1 位)算的;本工具在記憶體算未四捨五入的框,
    IoU 的第三位小數可能不同(§8.3 的同類精度差)

## 驗收

以 9/14 `base` 的設定跑,必須重現:B 683、看得到的幀 ≥0.30 佔 72.4%、IoU 中位 0.199、
< 0.2 佔 50.4%、≥ 0.5 佔 16.9%。重現不出來先換 `--interval` 再查,不要直接拿來比新方法。

用法(旗標與 eval_m4_chirla.py 相同,兩支跑的是同一個 tracker):
    python scripts/diag_m4_breaks.py --cache-dir results/det_cache/coco_nano --root "$R" --seqs $S \\
        --thr 0.30 --stride 5 --lost-buffer-seconds 5 --label base
"""
import argparse
import json
import statistics as st
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from m4_track.det_cache import DetCache                                   # noqa: E402
from eval_m4_chirla import (add_common_args, classify_pairs, load_setup,  # noqa: E402
                            run_camera, tracker_config)
from eval_m4m5_chirla import IOU_THR, iou, load_gt, match_tracks          # noqa: E402

SCORE_BINS = (("無框", None), ("0.05~0.10", 0.10), ("0.10~0.25", 0.25),
              ("0.25~0.30", 0.30), ("≥ 0.30", float("inf")))
IOU_BINS = (("< 0.2", 0.2), ("0.2~0.5", 0.5), ("0.5~0.8", 0.8), ("≥ 0.8", float("inf")))


def score_bin(best):
    if best is None:
        return SCORE_BINS[0][0]
    return next(name for name, hi in SCORE_BINS[1:] if best < hi)


def iou_bin(v):
    return next(name for name, hi in IOU_BINS if v < hi)


def best_score(cache, fid, gt_box):
    """該幀快取裡與真值框 IoU ≥ 0.5 的偵測最高分;沒有就 None。"""
    det = cache.get(fid, float(cache.meta["cache_thr"]))
    best = None
    for box, conf in zip(det.xyxy, det.confidence if len(det) else []):
        if iou(tuple(map(float, box)), gt_box) >= IOU_THR:
            c = float(conf)
            best = c if best is None else max(best, c)
    return best


def analyse_break(d, cam, gt, cache, rows, stride, interval):
    lo, hi = d["lost_fid"], d["new_fid"]
    frames = range(lo + 1 if interval == "open" else lo, hi)
    per_frame = []                                         # (看得到?, 分數區間)
    for fid in frames:
        boxes = [b for g, b in gt.get(cam, {}).get(fid + 1, []) if g == d["gid"]]
        if not boxes:
            continue                                       # 真人不在這台鏡頭
        per_frame.append((fid % stride == 0, score_bin(best_score(cache, fid, boxes[0]))))

    prev_real = [r for r in rows[(cam, d["prev_tid"])] if r["fid"] < lo and r["conf"] is not None]
    new_first = rows[(cam, d["new_tid"])][0]
    link = None
    if prev_real and prev_real[-1]["bbox"] and new_first["bbox"]:
        old = prev_real[-1]["bbox"]
        link = dict(iou=iou(old, new_first["bbox"]), gap_frames=hi - lo,
                    new_score=new_first["conf"], old_width=old[2] - old[0])
    return per_frame, link


def break_summary(per_frame, visible_only):
    fr = [b for vis, b in per_frame if vis or not visible_only]
    if not fr:
        return "間隔內沒有人在畫面的幀"
    if "≥ 0.30" in fr:
        return "有 ≥ 0.30 的框"
    if all(b == "無框" for b in fr):
        return "整段無框"
    return "有框但都 < 0.30"


def print_table(title, counter, names, total):
    print(f"\n  {title}")
    for name in names:
        n = counter.get(name, 0)
        print(f"    {name:<22}{n:>8,}{n / total if total else float('nan'):>9.1%}")


def main():
    ap = argparse.ArgumentParser()
    add_common_args(ap)
    ap.add_argument("--root", required=True, help="CHIRLA 根目錄(要真值)")
    ap.add_argument("--interval", choices=("open", "lost-inclusive"), default="open")
    args = ap.parse_args()
    if not args.seqs:
        raise SystemExit("需要 --seqs")

    expect, base_tcfg = load_setup(args)
    from run_m4m5_chirla_grid import CAMERAS
    cams = args.cameras or CAMERAS
    t0 = time.time()

    frames_all, frames_vis = Counter(), Counter()
    breaks_all, breaks_vis = Counter(), Counter()
    links, n_b, n_no_link = [], 0, 0
    for seq in args.seqs:
        gt = load_gt(args.root, seq)
        tracks, events, caches = [], [], {}
        for cam in cams:
            cache = caches[cam] = DetCache(Path(args.cache_dir) / seq / f"{cam}.npz", expect=expect)
            tcfg, _ = tracker_config(base_tcfg, args.stride, float(cache.meta["video_meta"]["fps"]),
                                     args.lost_buffer_seconds)
            tr, ev, _ = run_camera(cache, cam, thr=args.thr, stride=args.stride,
                                   tcfg=tcfg, max_loops=-1)
            tracks += tr
            events += ev
        track_gt, *_ = match_tracks(gt, [t for t in tracks if t["bbox"] is not None])
        rows = defaultdict(list)
        for t in tracks:
            rows[(t["cam"], t["tid"])].append(t)

        for cam, _gap, d in classify_pairs(events, track_gt, detail=True).get("B", []):
            n_b += 1
            per_frame, link = analyse_break(d, cam, gt, caches[cam], rows, args.stride, args.interval)
            for vis, b in per_frame:
                frames_all[b] += 1
                frames_vis[b] += vis
            breaks_all[break_summary(per_frame, visible_only=False)] += 1
            breaks_vis[break_summary(per_frame, visible_only=True)] += 1
            if link is None:
                n_no_link += 1
            else:
                links.append(dict(link, seq=seq, cam=cam, **d))
        print(f"  {seq}: 累計 B {n_b}")

    names = [n for n, _ in SCORE_BINS]
    brk = ["有 ≥ 0.30 的框", "有框但都 < 0.30", "整段無框", "間隔內沒有人在畫面的幀"]
    print(f"\n[{args.label}] backend={base_tcfg.get('backend', 'bytetrack')}  B 類斷點 {n_b}"
          f"  間隔口徑 {args.interval}")
    print("\n§8.8.1 斷點期間偵測器有沒有給框")
    print_table("全部幀(快取逐幀)", frames_all, names, sum(frames_all.values()))
    print_table("以斷點計(全部幀)", breaks_all, brk, n_b)
    print_table("只算追蹤器看得到的幀", frames_vis, names, sum(frames_vis.values()))
    print_table("以斷點計(看得到的幀)", breaks_vis, brk, n_b)

    ious = [x["iou"] for x in links]
    iou_ct = Counter(iou_bin(v) for v in ious)
    print("\n§8.8.2 舊 track 最後真實框 vs 新 track 第一框")
    print_table("IoU", iou_ct, [n for n, _ in IOU_BINS], len(ious))

    def med(key):
        v = [x[key] for x in links if x[key] is not None]
        return round(st.median(v), 3) if v else None

    scores = [x["new_score"] for x in links if x["new_score"] is not None]
    stats = dict(iou_median=med("iou"), gap_frames_median=med("gap_frames"),
                 new_score_median=med("new_score"),
                 new_score_p10=round(float(np.percentile(scores, 10)), 3) if scores else None,
                 old_width_median=med("old_width"), n_without_real_box=n_no_link)
    print(f"\n  {stats}")

    summary = dict(label=args.label, backend=base_tcfg.get("backend", "bytetrack"),
                   thr=args.thr, stride=args.stride, interval=args.interval, seqs=args.seqs,
                   n_breaks=n_b, frames_all=dict(frames_all), frames_visible=dict(frames_vis),
                   breaks_all=dict(breaks_all), breaks_visible=dict(breaks_vis),
                   iou_bins=dict(iou_ct), **stats,
                   breaks=[{k: (round(v, 4) if isinstance(v, float) else v) for k, v in x.items()}
                           for x in links],
                   seconds=round(time.time() - t0, 1))
    out = Path(args.out or ROOT / "results" / "m4_layer0" / f"diag_{args.label}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已存 {out}({summary['seconds']} 秒)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
