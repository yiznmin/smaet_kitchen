"""M4 誤偵 track 與混人 track 的組成診斷 —— 探索性,非預先登記(2026-09-15)。

## 為什麼

`eval_m4_chirla.py` 的「誤偵」是一條規則:track 與標註框 IoU ≥ 0.5(逐幀匈牙利配對)
的幀不到一半。符合這條規則的可能是性質完全不同的東西,修法也不同:

  ① 幾乎不碰到人   —— 真的不是人(椅子、螢幕、反光)→ 靜止/非人過濾
  ② 貼著人但框不準 —— 真人被遮擋或只框到一部分 → 改善框品質,過濾會砍掉真人
  ③ 部分配對       —— 有一段是人、一段不是 → 追蹤器的問題
  ④ 重複框         —— 框在真人身上(IoU ≥ 0.5),但那個人已被另一條 track 配走
                     → 同一個人被兩條 track 框住,屬於 A 類斷點的同一家族

混人 track(第二個真人 ≥ 3 票)另外量:換人時兩個人的標註框靠多近。

## 分類規則(執行前寫死,依序判定,第一個符合的為準)

  ③ 配對率 ≥ PARTIAL_RATE
  ④ 「IoU ≥ 0.5 卻沒配到」的幀佔比 ≥ DUP_SHARE
  ② 「與某個人框 IoU ≥ NEAR_IOU」的幀佔比 ≥ NEAR_SHARE
  ① 其餘

⚠ 門檻是切分用的,不是結論 —— 每一類都另外輸出完整分布,可以換門檻重算。

## 口徑

誤偵與混人的判定**直接 import `eval_m4m5_chirla.match_tracks`**;逐幀配對另外重算一次
以取得每一幀的配對對象,並自檢兩者的票數逐條相同,不同就中止。

用法:
    python scripts/diag_m4_ghosts.py --root <CHIRLA根> \\
        --tracks-dir results/m5_step3/m5_cbiou/m4dump --seqs seq_004 ... \\
        --track-gt-dir results/m5_step3/m5_cbiou/track_gt --out results/m4_ghosts/cbiou.json
"""
import argparse
import csv
import json
import statistics as st
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from eval_m4m5_chirla import IOU_THR, iou, load_gt, match_tracks   # noqa: E402

PARTIAL_RATE = 0.2
DUP_SHARE = 0.5
NEAR_IOU = 0.1
NEAR_SHARE = 0.5
STATIC_RATIO = 0.25     # 框中心移動範圍 < 0.25 × 框高中位數 → 靜止
CATS = ("① 幾乎不碰到人", "② 貼著人但框不準", "③ 部分配對", "④ 重複框")


def load_rows(path):
    with open(path, encoding="utf-8") as f:
        return [dict(fid=int(r["video_fid"]), cam=r["camera_id"], tid=int(r["track_id"]),
                     bbox=(float(r["x1"]), float(r["y1"]), float(r["x2"]), float(r["y2"])),
                     conf=float(r["conf"]))
                for r in csv.DictReader(f)]


def per_frame_assign(gt, rows):
    """與 match_tracks 相同的逐幀匈牙利配對,但保留每一幀的結果。

    回傳 info[(cam,tid)] = [(fid, 最大 IoU, 最大 IoU 的真人, 匈牙利配到的真人或 None)] 與 votes。
    """
    from scipy.optimize import linear_sum_assignment
    per_frame = defaultdict(list)
    for t in rows:
        per_frame[(t["cam"], t["fid"])].append(t)
    info, votes = defaultdict(list), defaultdict(Counter)
    for (cam, fid), ts in per_frame.items():
        gts = gt.get(cam, {}).get(fid + 1, [])      # ⚠ GT 是 1-based
        if not gts:
            for t in ts:
                info[(cam, t["tid"])].append((fid, 0.0, None, None))
            continue
        m = np.zeros((len(ts), len(gts)))
        for i, t in enumerate(ts):
            for j, (_g, gb) in enumerate(gts):
                m[i, j] = iou(t["bbox"], gb)
        r, c = linear_sum_assignment(-m)
        got = {}
        for i, j in zip(r, c):
            if m[i, j] >= IOU_THR:
                got[i] = gts[j][0]
                votes[(cam, ts[i]["tid"])][gts[j][0]] += 1
        for i, t in enumerate(ts):
            j = int(m[i].argmax())
            info[(cam, t["tid"])].append((fid, float(m[i, j]), gts[j][0], got.get(i)))
    return info, votes


def features(frames, boxes, confs):
    n = len(frames)
    rate = sum(1 for x in frames if x[3] is not None) / n
    dup = sum(1 for x in frames if x[1] >= IOU_THR and x[3] is None) / n
    near = sum(1 for x in frames if x[1] >= NEAR_IOU) / n
    if rate >= PARTIAL_RATE:
        cat = CATS[2]
    elif dup >= DUP_SHARE:
        cat = CATS[3]
    elif near >= NEAR_SHARE:
        cat = CATS[1]
    else:
        cat = CATS[0]
    hs = [b[3] - b[1] for b in boxes]
    cx = [(b[0] + b[2]) / 2 for b in boxes]
    cy = [(b[1] + b[3]) / 2 for b in boxes]
    h_med = st.median(hs)
    ext = max(max(cx) - min(cx), max(cy) - min(cy)) / h_med if h_med > 0 else 0.0
    return dict(cat=cat, n=n, match_rate=rate, dup_share=dup, near_share=near,
                iou_median=st.median(x[1] for x in frames),
                conf_mean=sum(confs) / n, h_med=h_med,
                w_med=st.median(b[2] - b[0] for b in boxes),
                move_ratio=ext, static=ext < STATIC_RATIO)


def mixed_switch(frames, gt, cam):
    """第一次換人時,前後兩位真人的標註框 IoU 與中心距離(÷ 平均框高)。"""
    seq = [(x[0], x[3]) for x in sorted(frames) if x[3] is not None]
    sw = [(f0, a, f1, b) for (f0, a), (f1, b) in zip(seq, seq[1:]) if a != b]
    if not sw:
        return None
    f0, a, f1, b = sw[0]
    by = dict(gt.get(cam, {}).get(f1 + 1, []))
    if a not in by or b not in by:
        return dict(n_switch=len(sw), gap_updates=None, gt_iou=None, dist_ratio=None,
                    both_visible=False)
    ba, bb = by[a], by[b]
    hm = ((ba[3] - ba[1]) + (bb[3] - bb[1])) / 2
    d = np.hypot((ba[0] + ba[2] - bb[0] - bb[2]) / 2, (ba[1] + ba[3] - bb[1] - bb[3]) / 2)
    return dict(n_switch=len(sw), gap_updates=None, gt_iou=iou(ba, bb),
                dist_ratio=float(d / hm) if hm > 0 else None, both_visible=True)


def q(vals):
    vals = sorted(v for v in vals if v is not None)
    if not vals:
        return None
    p = lambda x: vals[min(len(vals) - 1, int(round(x * (len(vals) - 1))))]
    return dict(n=len(vals), p25=round(p(0.25), 3), median=round(p(0.5), 3), p75=round(p(0.75), 3))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--tracks-dir", required=True)
    ap.add_argument("--seqs", nargs="+", required=True)
    ap.add_argument("--track-gt-dir", default=None, help="若給,自檢誤偵集合與既有真值表一致")
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    ghosts, reals, mixed = [], [], []
    for seq in args.seqs:
        gt = load_gt(args.root, seq)
        rows = load_rows(Path(args.tracks_dir) / seq / "tracks.csv")
        track_gt, _s, _m, _t, votes_ref = match_tracks(gt, rows, return_votes=True)
        info, votes = per_frame_assign(gt, rows)

        # 自檢 1:逐幀重算的票數與 match_tracks 逐條相同
        a = {k: dict(v) for k, v in votes_ref.items() if v}
        b = {k: dict(v) for k, v in votes.items() if v}
        if a != b:
            raise SystemExit(f"[FAIL] {seq} 逐幀配對與 match_tracks 票數不同")
        # 自檢 2:誤偵集合與既有真值表一致
        if args.track_gt_dir:
            ref = json.loads((Path(args.track_gt_dir) / f"{seq}.json").read_text(encoding="utf-8"))
            mine = {f"{c}|{t}": g for (c, t), g in track_gt.items() if g is not None}
            if mine != ref:
                raise SystemExit(f"[FAIL] {seq} 真值表與 {args.track_gt_dir} 不同")
        # 自檢 3:每條 track 的相鄰 fid 間距都是 stride 的倍數
        by_tr = defaultdict(list)
        for r in rows:
            by_tr[(r["cam"], r["tid"])].append(r)
        for k, rs in by_tr.items():
            fs = sorted(r["fid"] for r in rs)
            if any((y - x) % args.stride for x, y in zip(fs, fs[1:])):
                raise SystemExit(f"[FAIL] {seq} {k} 的幀間距不是 stride {args.stride} 的倍數")

        for k, rs in by_tr.items():
            rs.sort(key=lambda r: r["fid"])
            fr = sorted(info[k])
            feat = features(fr, [r["bbox"] for r in rs], [r["conf"] for r in rs])
            feat.update(seq=seq, cam=k[0], tid=k[1])
            gid = track_gt[k]
            if gid is None:
                ghosts.append(feat)
            else:
                reals.append(feat)
                top = votes_ref[k].most_common(2)
                if len(top) >= 2 and top[1][1] >= 3:
                    sw = mixed_switch(fr, gt, k[0])
                    mixed.append(dict(seq=seq, cam=k[0], tid=k[1], **(sw or dict(n_switch=0))))

    sec = args.stride / args.fps
    n_all = len(ghosts) + len(reals)
    print(f"track 共 {n_all:,};誤偵 {len(ghosts):,}({len(ghosts) / n_all:.2%});真人 {len(reals):,};"
          f"混人 {len(mixed):,}")
    print("自檢:逐幀配對 = match_tracks [OK]" + ("、真值表一致 [OK]" if args.track_gt_dir else "")
          + f"、幀間距 = stride {args.stride} 的倍數 [OK]")

    print(f"\n1. 誤偵 track 的組成(規則:③ 配對率 ≥ {PARTIAL_RATE};④ IoU≥0.5 未配到的幀 ≥ {DUP_SHARE};"
          f"② IoU≥{NEAR_IOU} 的幀 ≥ {NEAR_SHARE};① 其餘)")
    cc = Counter(g["cat"] for g in ghosts)
    for c in CATS:
        print(f"   {c:<12} {cc[c]:>5,}({cc[c] / len(ghosts):.1%})")

    print("\n2. 各類特徵(中位數;真人 track 為對照)")
    print(f"   {'類別':<14}{'條數':>6}{'長度(秒)':>10}{'平均信心':>9}{'框高':>7}{'框寬':>7}"
          f"{'移動比':>8}{'靜止佔比':>9}{'IoU中位':>9}")
    groups = [(c, [g for g in ghosts if g["cat"] == c]) for c in CATS] + [("真人 track", reals)]
    feat_out = {}
    for name, gs in groups:
        if not gs:
            continue
        row = dict(n=len(gs),
                   seconds=q([g["n"] * sec for g in gs]), conf_mean=q([g["conf_mean"] for g in gs]),
                   h_med=q([g["h_med"] for g in gs]), w_med=q([g["w_med"] for g in gs]),
                   move_ratio=q([g["move_ratio"] for g in gs]),
                   static_share=round(sum(g["static"] for g in gs) / len(gs), 4),
                   iou_median=q([g["iou_median"] for g in gs]))
        feat_out[name] = row
        print(f"   {name:<14}{len(gs):>6}{row['seconds']['median']:>10.2f}{row['conf_mean']['median']:>9.3f}"
              f"{row['h_med']['median']:>7.0f}{row['w_med']['median']:>7.0f}"
              f"{row['move_ratio']['median']:>8.2f}{row['static_share']:>9.1%}{row['iou_median']['median']:>9.3f}")

    print("\n3. 各鏡頭:誤偵 track 數與組成")
    cams = sorted({g["cam"] for g in ghosts} | {r["cam"] for r in reals})
    cam_out = {}
    print(f"   {'鏡頭':<10}{'track':>7}{'誤偵':>6}{'誤偵率':>8}" + "".join(f"{c[:1]:>7}" for c in CATS))
    for cam in cams:
        g = [x for x in ghosts if x["cam"] == cam]
        n = len(g) + sum(1 for r in reals if r["cam"] == cam)
        c = Counter(x["cat"] for x in g)
        cam_out[cam] = dict(n_tracks=n, n_ghost=len(g), cats={k: c[k] for k in CATS})
        print(f"   {cam:<10}{n:>7}{len(g):>6}{len(g) / n if n else 0:>8.1%}"
              + "".join(f"{c[k]:>7}" for k in CATS))

    print("\n4. 混人 track:第一次換人時,兩位真人的標註框")
    sw = [m for m in mixed if m.get("n_switch")]
    vis = [m for m in sw if m.get("both_visible")]
    print(f"   有換人紀錄 {len(sw)} / {len(mixed)};換人那一幀兩人都有標註 {len(vis)}")
    if vis:
        ov = sum(1 for m in vis if m["gt_iou"] > 0)
        print(f"   兩人框互相重疊(IoU > 0){ov}({ov / len(vis):.1%});"
              f"兩人框 IoU 中位 {q([m['gt_iou'] for m in vis])['median']};"
              f"中心距離 ÷ 框高 中位 {q([m['dist_ratio'] for m in vis])['median']}")
    print(f"   換人次數分布:{dict(sorted(Counter(m['n_switch'] for m in mixed).items()))}")

    print("\n5. 假說對照(執行前寫下)")
    one = [g for g in ghosts if g["cat"] == CATS[0]]
    h1a = cc[CATS[0]] / len(ghosts)
    h1b = sum(g["static"] for g in one) / len(one) if one else 0.0
    print(f"   H1 ① 佔多數且多為靜止:① 佔 {h1a:.1%}、① 中靜止 {h1b:.1%} → "
          f"{'成立' if h1a > 0.5 and h1b > 0.5 else '不成立'}")
    for cam in ("camera_5", "camera_6"):
        c = cam_out.get(cam, {}).get("cats", {})
        top = max(c, key=c.get) if c else None
        print(f"   H2 {cam} 以 ① 為主:最大類 = {top} → {'成立' if top == CATS[0] else '不成立'}")
    h3 = cc[CATS[1]] / len(ghosts)
    print(f"   H3 ② 佔 ≥ 20%:② 佔 {h3:.1%} → {'成立' if h3 >= 0.2 else '不成立'}"
          f"(另報:④ 重複框 {cc[CATS[3]] / len(ghosts):.1%},也是框在真人身上)")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(
        args=vars(args), rules=dict(PARTIAL_RATE=PARTIAL_RATE, DUP_SHARE=DUP_SHARE, NEAR_IOU=NEAR_IOU,
                                    NEAR_SHARE=NEAR_SHARE, STATIC_RATIO=STATIC_RATIO),
        n_tracks=n_all, n_ghost=len(ghosts), n_real=len(reals), n_mixed=len(mixed),
        ghost_cats={c: cc[c] for c in CATS}, features=feat_out, per_camera=cam_out,
        mixed=dict(n_switch=len(sw), both_visible=len(vis),
                   gt_overlap=sum(1 for m in vis if m["gt_iou"] > 0),
                   gt_iou=q([m["gt_iou"] for m in vis]), dist_ratio=q([m["dist_ratio"] for m in vis]),
                   switch_hist=dict(Counter(m["n_switch"] for m in mixed)))),
        ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已存 {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
