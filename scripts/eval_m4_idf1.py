"""M4 單鏡頭身份一致性:IDF1 與 ID 切換次數(2026-09-15)。

## 為什麼

M4 的最終工作是「同一個人在同一台鏡頭裡保持同一個 track 編號」。`eval_m4_chirla.py` 量的是
誤偵率、召回、斷點類型等中間指標;修法輪次(`docs/M4_修法輪次_20260915.md` §2)的主判準要用
**單鏡頭 IDF1 與 ID 切換次數**。

## 做法

直接呼叫 `trackers.eval` 的 `compute_identity_metrics` / `compute_clear_metrics`(套件聲明與 TrackEval 對齊),
不自己寫公式。每個 (序列, 鏡頭) 是一個獨立的評估單元:

  評估幀 = 追蹤器實際處理的幀(0, stride, 2·stride, … 到該鏡頭標註最後一幀)
  真值   = 該幀的標註框(1-based 幀號 = video_fid + 1;distractor 負號取絕對值,與 load_gt 相同)
  預測   = tracks.csv 該幀的 track 框
  相似度 = IoU,門檻 0.5

所有單元以套件的 aggregate 函式合計。

## 自檢(不通過就中止)

  1. 套件文件的範例:IDF1 應為 0.8333
  2. 把真值本身當成追蹤輸出:IDF1 = 1、IDSW = 0

用法:
    .venv/Scripts/python.exe scripts/eval_m4_idf1.py --root <CHIRLA根> \\
        --tracks-dir results/m4_5cam/cbiou_dump --seqs seq_004 ... \\
        --cameras camera_1 camera_2 camera_3 camera_4 camera_7 --stride 5 --label cbiou --out results/m4_5cam/idf1_cbiou.json
"""
import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from eval_m4m5_chirla import iou, load_gt                                          # noqa: E402
from trackers.eval.clear import aggregate_clear_metrics, compute_clear_metrics     # noqa: E402
from trackers.eval.identity import aggregate_identity_metrics, compute_identity_metrics   # noqa: E402


def sim_matrix(gb, tb):
    m = np.zeros((len(gb), len(tb)))
    for i, g in enumerate(gb):
        for j, t in enumerate(tb):
            m[i, j] = iou(g, t)
    return m


def unit_inputs(gt_cam, trk_by_fid, stride):
    """回傳一個 (序列, 鏡頭) 單元的 gt_ids、tracker_ids、similarity 三個逐幀清單。"""
    last = max(gt_cam) if gt_cam else 0          # 標註幀號是 1-based
    gids, tids, sims = [], [], []
    for fid in range(0, last, stride):
        g = gt_cam.get(fid + 1, [])
        t = trk_by_fid.get(fid, [])
        gids.append(np.array([x[0] for x in g], dtype=int))
        tids.append(np.array([x[0] for x in t], dtype=int))
        sims.append(sim_matrix([x[1] for x in g], [x[1] for x in t]))
    return gids, tids, sims


def selfcheck():
    ex = compute_identity_metrics(
        [np.array([0, 1])] * 3,
        [np.array([10, 20]), np.array([10, 30]), np.array([10, 30])],
        [np.array([[0.9, 0.1], [0.1, 0.8]]), np.array([[0.85, 0.1], [0.1, 0.75]]),
         np.array([[0.8, 0.1], [0.1, 0.7]])])
    if abs(ex["IDF1"] - 5 / 6) > 1e-9:
        raise SystemExit(f"[FAIL] 套件範例 IDF1 {ex['IDF1']} ≠ 0.8333")
    return ex["IDF1"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--tracks-dir", required=True)
    ap.add_argument("--seqs", nargs="+", required=True)
    ap.add_argument("--cameras", nargs="+", required=True)
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--label", default="run")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    print(f"自檢 1:套件範例 IDF1 = {selfcheck():.4f} [OK]")

    idm, clm, idm_gt, clm_gt, per_unit = [], [], [], [], {}
    for seq in args.seqs:
        gt = load_gt(args.root, seq)
        trk = defaultdict(lambda: defaultdict(list))
        with open(Path(args.tracks_dir) / seq / "tracks.csv", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r["camera_id"] in args.cameras:
                    trk[r["camera_id"]][int(r["video_fid"])].append(
                        (int(r["track_id"]), (float(r["x1"]), float(r["y1"]), float(r["x2"]), float(r["y2"]))))
        for cam in args.cameras:
            if cam not in gt:
                raise SystemExit(f"[FAIL] {seq} {cam} 沒有標註檔")
            gids, tids, sims = unit_inputs(gt[cam], trk[cam], args.stride)
            i_m = compute_identity_metrics(gids, tids, sims)
            c_m = compute_clear_metrics(gids, tids, sims)
            idm.append(i_m)
            clm.append(c_m)
            per_unit[f"{seq}|{cam}"] = dict(IDF1=i_m["IDF1"], IDSW=c_m["IDSW"], MOTA=c_m["MOTA"])
            # 自檢 2:真值當追蹤輸出
            gt_as_trk = {fid: [(g, b) for g, b in gt[cam].get(fid + 1, [])] for fid in range(0, max(gt[cam] or [0]), args.stride)}
            g2, t2, s2 = unit_inputs(gt[cam], gt_as_trk, args.stride)
            idm_gt.append(compute_identity_metrics(g2, t2, s2))
            clm_gt.append(compute_clear_metrics(g2, t2, s2))

    agg_gt_i, agg_gt_c = aggregate_identity_metrics(idm_gt), aggregate_clear_metrics(clm_gt)
    if abs(agg_gt_i["IDF1"] - 1.0) > 1e-9 or agg_gt_c["IDSW"] != 0:
        raise SystemExit(f"[FAIL] 真值當追蹤輸出:IDF1 {agg_gt_i['IDF1']}、IDSW {agg_gt_c['IDSW']}")
    print("自檢 2:真值當追蹤輸出 IDF1 = 1.0000、IDSW = 0 [OK]")

    ai, ac = aggregate_identity_metrics(idm), aggregate_clear_metrics(clm)
    print(f"\n[{args.label}] 單元 {len(per_unit)}(序列 × 鏡頭)")
    print(f"   IDF1 {ai['IDF1']:.4f}(IDR {ai['IDR']:.4f} / IDP {ai['IDP']:.4f})")
    print(f"   IDSW {ac['IDSW']}   MOTA {ac['MOTA']:.4f}   CLR_TP {ac['CLR_TP']}  CLR_FN {ac['CLR_FN']}  CLR_FP {ac['CLR_FP']}")
    by_cam = defaultdict(lambda: [[], []])
    for k, i_m, c_m in zip(per_unit, idm, clm):
        by_cam[k.split("|")[1]][0].append(i_m)
        by_cam[k.split("|")[1]][1].append(c_m)
    print("   各鏡頭:")
    cam_out = {}
    for cam in args.cameras:
        a1, a2 = aggregate_identity_metrics(by_cam[cam][0]), aggregate_clear_metrics(by_cam[cam][1])
        cam_out[cam] = dict(IDF1=a1["IDF1"], IDSW=a2["IDSW"], MOTA=a2["MOTA"])
        print(f"     {cam:<10} IDF1 {a1['IDF1']:.4f}  IDSW {a2['IDSW']:>5}  MOTA {a2['MOTA']:.4f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    keep = ("IDF1", "IDR", "IDP", "IDTP", "IDFN", "IDFP")
    keepc = ("MOTA", "MOTP", "IDSW", "CLR_TP", "CLR_FN", "CLR_FP", "MT", "PT", "ML", "Frag")
    out.write_text(json.dumps(dict(
        label=args.label, args=vars(args),
        identity={k: ai[k] for k in keep if k in ai}, clear={k: ac[k] for k in keepc if k in ac},
        per_camera=cam_out, per_unit=per_unit,
        selfcheck=dict(package_example_idf1=5 / 6, gt_as_tracker_idf1=agg_gt_i["IDF1"], gt_as_tracker_idsw=agg_gt_c["IDSW"])),
        ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print(f"\n已存 {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
