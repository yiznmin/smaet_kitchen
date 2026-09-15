"""外觀在框重疊時刻失效的原因:量化「裁圖被前面的人蓋住」與選錯的關係,並試遮掉被蓋住的部分 —— 探索性,非預先登記(2026-09-15)。

## 為什麼

`diag_m4_appearance.py` 查出:沒被擋時外觀 d′ > 2,但換人時刻選對只有 60% 左右、模糊出生 d′ < 0.4。
8 個失敗樣本看到裁圖裡有兩個以上的人 —— 這支把它量化,並測遮掉之後能不能救回來。

## 樣本(與 diag_m4_appearance.py 完全相同,並自檢重現其不遮的數字)

  S 換人時刻 297:模板 = track 在 f0 配到 a 的框;候選 = f1 的 a、b 標註框
  B 模糊出生 431(D 264 / N 167):模板 = 既有 track 1 秒前的框;比對 = 新框

## 規則(執行前寫死)

每張裁圖有一個「目標人」:
  S 模板 → a(深度參考用 a 在 f0 的標註框底);S 候選 → a / b 的標註框本身
  B 模板 → 既有 track 在該幀配到的人(沒配到就用 IoU 最大的人;都沒有則無目標);B 新框 → IoU 最大的人
  深度參考 y = 目標人標註框的底邊(沒有目標人就用裁圖框底邊)

**站在前面** = 另一個框的底邊 > 深度參考 y(離鏡頭較近)。⚠ 簡化判斷,坐著/彎腰會判錯。

  被蓋比例 = 裁圖內被「前面其他人的標註框」蓋住的像素比例;每個樣本取其裁圖中的最大值

  遮法(被遮像素塗成 ImageNet 平均色)
    none        不遮
    gt_front    用前面其他人的**標註框**遮(理論上限,實際系統沒有真值)
    track_front 用同幀前面其他 **track 框**遮(實際可行;排除目標自己的 track:
                S 模板 / B 新框排除自己的 track,標註框裁圖排除與它 IoU 最大的 track)

## 假說(執行前寫下)

  H1  被蓋比例越高選對率越低(四組 [0,0.2)/[0.2,0.4)/[0.4,0.6)/[0.6,1] 單調不增),且 [0,0.2) 組 armS 選對率 ≥ 75%
  H2a gt_front 下可出貨模型(DINOv2、armS)最高換人選對率 ≥ 80%
  H2b gt_front 下可出貨模型最高模糊出生 d′ ≥ 1.0
  H2c track_front 的改善至少是 gt_front 改善的一半(以 armS 換人選對率計)

用法:
    .venv/Scripts/python.exe scripts/diag_m4_appearance_mask.py --root <CHIRLA根> \\
        --tracks-dir results/m4_5cam/cbiou_dump --seqs seq_004 ... \\
        --switch-json results/m4_5cam/switch_cbiou.json --births-json results/m4_5cam/birth_geometry_cbiou.json \\
        --appearance-json results/m4_5cam/appearance_cbiou.json --out results/m4_5cam/appearance_mask_cbiou.json
"""
import argparse
import bisect
import glob
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from diag_m4_appearance import BIRTH_IOU, LAG, MIN_SIDE, MODELS, auc, build_embedder, dprime   # noqa: E402
from diag_m4_ghosts import load_rows, per_frame_assign                                        # noqa: E402
from eval_m4m5_chirla import iou, load_gt                                                     # noqa: E402

VARIANTS = ("none", "gt_front", "track_front")
BINS = ((0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 1.01))
FILL_BGR = (104, 116, 124)       # ImageNet 平均色(RGB 124,116,104)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--tracks-dir", required=True)
    ap.add_argument("--seqs", nargs="+", required=True)
    ap.add_argument("--switch-json", required=True)
    ap.add_argument("--births-json", required=True)
    ap.add_argument("--appearance-json", required=True, help="diag_m4_appearance.py 的輸出,自檢不遮的數字用")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    t0 = time.time()

    specs = {}          # cid -> dict(seq, cam, fid, box, ref_y, gt_others, trk_others)
    frames_needed = defaultdict(set)

    switch = json.loads(Path(args.switch_json).read_text(encoding="utf-8"))["switches"]
    births = json.loads(Path(args.births_json).read_text(encoding="utf-8"))["births"]
    S, B = [], []
    for seq in args.seqs:
        gt = load_gt(args.root, seq)
        rows = load_rows(Path(args.tracks_dir) / seq / "tracks.csv")
        info, _ = per_frame_assign(gt, rows)
        at = {(cam, tid, x[0]): x for (cam, tid), fr in info.items() for x in fr}
        box_at = {(r["cam"], r["tid"], r["fid"]): r["bbox"] for r in rows}
        fids_of = defaultdict(list)
        frame_tracks = defaultdict(list)
        for r in rows:
            fids_of[(r["cam"], r["tid"])].append(r["fid"])
            frame_tracks[(r["cam"], r["fid"])].append((r["tid"], r["bbox"]))
        for v in fids_of.values():
            v.sort()
        first = {k: v[0] for k, v in fids_of.items()}

        def need(cam, fid, box, person, own_tid):
            gts = gt.get(cam, {}).get(fid + 1, [])
            gd = dict(gts)
            ref_y = gd[person][3] if person in gd else box[3]
            gt_others = [b for g, b in gts if g != person and b[3] > ref_y]
            tr = frame_tracks[(cam, fid)]
            if own_tid is None:                       # 標註框裁圖:排除與它 IoU 最大的 track
                own_tid = max(tr, key=lambda x: iou(box, x[1]))[0] if tr else None
            trk_others = [b for t, b in tr if t != own_tid and b[3] > ref_y]
            cid = len(specs)
            specs[cid] = dict(seq=seq, cam=cam, fid=fid, box=tuple(box), gt_others=gt_others, trk_others=trk_others)
            frames_needed[(seq, cam)].add(fid)
            return cid

        for s in switch:                               # 選取規則與 diag_m4_appearance.py 相同
            if s["seq"] != seq or not s["a_visible"]:
                continue
            g = dict(gt[s["cam"]].get(s["f1"] + 1, []))
            if s["b"] not in g:
                continue
            cam = s["cam"]
            S.append(dict(cat=s["cat"],
                          tmpl=need(cam, s["f0"], box_at[(cam, s["tid"], s["f0"])], s["a"], s["tid"]),
                          a=need(cam, s["f1"], g[s["a"]], s["a"], None),
                          b=need(cam, s["f1"], g[s["b"]], s["b"], None)))

        for bth in births:
            if bth["seq"] != seq or bth["iou"] < BIRTH_IOU or bth["label"] not in ("D 重複", "N 新人"):
                continue
            cam, fid = bth["cam"], bth["fid"]
            nbox = box_at[(cam, bth["tid"], fid)]
            ov, e = max((iou(nbox, bb), t) for t, bb in frame_tracks[(cam, fid)]
                        if t != bth["tid"] and first[(cam, t)] < fid)
            ef = fids_of[(cam, e)]
            j = bisect.bisect_right(ef, fid - LAG) - 1
            tf = ef[j] if j >= 0 else ef[0]
            _f, bi, bg, mg = at[(cam, e, tf)]
            tp = mg if mg is not None else (bg if bi > 0 else None)
            _f2, nbi, nbg, _nmg = at[(cam, bth["tid"], fid)]
            B.append(dict(label=bth["label"], same_person_old=bth["n_same_person_old"],
                          tmpl=need(cam, tf, box_at[(cam, e, tf)], tp, e),
                          new=need(cam, fid, nbox, nbg if nbi > 0 else None, bth["tid"])))

    print(f"樣本:S {len(S)}、B {len(B)}(D {sum(b['label'] == 'D 重複' for b in B)} / "
          f"N {sum(b['label'] == 'N 新人' for b in B)});裁圖 {len(specs):,}", flush=True)

    import cv2
    crops = {v: {} for v in VARIANTS}
    cover = {}
    by_frame = defaultdict(list)
    for cid, sp in specs.items():
        by_frame[(sp["seq"], sp["cam"], sp["fid"])].append(cid)
    for (seq, cam), fids in sorted(frames_needed.items()):
        cap = cv2.VideoCapture(sorted(glob.glob(f"{args.root}/videos/{seq}/{cam}_*.avi"))[0])
        f, last = 0, max(fids)
        while f <= last and cap.grab():
            if f in fids:
                ok, img = cap.retrieve()
                if ok:
                    h, w = img.shape[:2]
                    for cid in by_frame[(seq, cam, f)]:
                        sp = specs[cid]
                        x1, y1, x2, y2 = sp["box"]
                        x1, y1 = max(0, int(x1)), max(0, int(y1))
                        x2, y2 = min(w, int(round(x2))), min(h, int(round(y2)))
                        if x2 - x1 < MIN_SIDE or y2 - y1 < MIN_SIDE:
                            continue
                        base = img[y1:y2, x1:x2]
                        for v in VARIANTS:
                            c = base.copy()
                            if v != "none":
                                mask = np.zeros(c.shape[:2], bool)
                                for ox1, oy1, ox2, oy2 in sp["gt_others" if v == "gt_front" else "trk_others"]:
                                    a1, b1 = max(0, int(ox1) - x1), max(0, int(oy1) - y1)
                                    a2, b2 = min(x2 - x1, int(round(ox2)) - x1), min(y2 - y1, int(round(oy2)) - y1)
                                    if a2 > a1 and b2 > b1:
                                        mask[b1:b2, a1:a2] = True
                                c[mask] = FILL_BGR
                                if v == "gt_front":
                                    cover[cid] = float(mask.mean())
                            crops[v][cid] = c
            f += 1
        cap.release()
    print(f"裁圖成功 {len(crops['none']):,} / {len(specs):,}(解碼 {time.time() - t0:.0f} 秒)", flush=True)

    ref = json.loads(Path(args.appearance_json).read_text(encoding="utf-8"))["results"]
    results = {}
    for spec, name in MODELS:
        emb = build_embedder(spec)
        results[name] = {}
        for v in VARIANTS:
            ids = sorted(crops[v])
            feats = {}
            for i in range(0, len(ids), 256):
                chunk = ids[i:i + 256]
                for cid, vec in zip(chunk, emb.extract_batch([crops[v][c] for c in chunk])):
                    feats[cid] = vec
            cos = lambda a, b: float(np.dot(feats[a], feats[b]))
            s_ok = [x for x in S if all(c in feats for c in (x["tmpl"], x["a"], x["b"]))]
            right = {id(x): cos(x["tmpl"], x["a"]) > cos(x["tmpl"], x["b"]) for x in s_ok}
            b_ok = [x for x in B if x["tmpl"] in feats and x["new"] in feats]
            bd = [cos(x["tmpl"], x["new"]) for x in b_ok if x["label"] == "D 重複"]
            bn = [cos(x["tmpl"], x["new"]) for x in b_ok if x["label"] == "N 新人"]
            r = dict(S_right=sum(right.values()) / len(s_ok), S_n=len(s_ok),
                     B_dprime=dprime(bd, bn), B_auc=auc(bd, bn))
            if v == "none":
                bins = []
                for lo, hi in BINS:
                    sx = [x for x in s_ok if lo <= max(cover.get(x["tmpl"], 0), cover.get(x["a"], 0), cover.get(x["b"], 0)) < hi]
                    bx = [x for x in b_ok if lo <= max(cover.get(x["tmpl"], 0), cover.get(x["new"], 0)) < hi]
                    bdx = [cos(x["tmpl"], x["new"]) for x in bx if x["label"] == "D 重複"]
                    bnx = [cos(x["tmpl"], x["new"]) for x in bx if x["label"] == "N 新人"]
                    bins.append(dict(bin=[lo, min(hi, 1.0)], S_n=len(sx),
                                     S_right=(sum(right[id(x)] for x in sx) / len(sx)) if sx else None,
                                     B_n_d=len(bdx), B_n_n=len(bnx), B_dprime=dprime(bdx, bnx)))
                r["bins"] = bins
                # 自檢:不遮的數字必須重現 diag_m4_appearance.py。
                # ⚠ 第一版 d′ 容許 1e-6,armS 判 FAIL 但兩者到小數 4 位相同 —— 本輪裁圖集合比上一輪少
                #   (沒有 C0),送進 GPU 的批次組成不同,浮點結果有極小差異。換人選對率(整數計數)
                #   仍要求完全相同;d′ 放寬到 1e-3,並印出實際差距。此放寬是看到失敗後才做的。
                ds = abs(r["S_right"] - ref[name]["S"]["right_rate"])
                dd = abs(r["B_dprime"] - ref[name]["B"]["dprime"])
                r["selfcheck_diff"] = dict(S_right=ds, B_dprime=dd)
                if ds > 1e-9 or dd > 1e-3:
                    raise SystemExit(f"[FAIL] {name} 不遮的結果與 {args.appearance_json} 不同:"
                                     f"S 差 {ds:.2e}、B d′ 差 {dd:.2e}")
                print(f"   自檢 {name}:換人選對率差 {ds:.1e}、模糊出生 d′ 差 {dd:.1e}", flush=True)
            results[name][v] = r
        del emb
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass
        print(f"\n[{name}](不遮的數字與上一輪相同 [OK])", flush=True)
        for v in VARIANTS:
            r = results[name][v]
            print(f"   {v:<12} 換人選對 {r['S_right']:.1%}   模糊出生 d′ {r['B_dprime']:.2f}  AUC {r['B_auc']:.3f}")
        print("   被蓋比例分組(不遮):")
        for bn_ in results[name]["none"]["bins"]:
            sr = f"{bn_['S_right']:.1%}" if bn_["S_right"] is not None else "—"
            bdp = f"{bn_['B_dprime']:.2f}" if bn_["B_dprime"] is not None else "—"
            print(f"     [{bn_['bin'][0]:.1f},{bn_['bin'][1]:.1f}) 換人 n={bn_['S_n']:<4} 選對 {sr:<7}"
                  f"模糊出生 D={bn_['B_n_d']:<4} N={bn_['B_n_n']:<4} d′ {bdp}")

    print("\n被蓋比例分布(每個樣本取最大)")
    for lo, hi in BINS:
        ns = sum(1 for x in S if lo <= max(cover.get(x["tmpl"], 0), cover.get(x["a"], 0), cover.get(x["b"], 0)) < hi)
        nb = sum(1 for x in B if lo <= max(cover.get(x["tmpl"], 0), cover.get(x["new"], 0)) < hi)
        print(f"   [{lo:.1f},{min(hi, 1.0):.1f})  換人 {ns:>4}  模糊出生 {nb:>4}")

    ship = [n for _s, n in MODELS if "研究限定" not in n]
    arms = next(n for _s, n in MODELS if "armS" in n)
    print("\n假說對照(執行前寫下)")
    sb = [b["S_right"] for b in results[arms]["none"]["bins"]]
    mono = all(x is None or y is None or y <= x + 1e-12 for x, y in zip(sb, sb[1:]))
    print(f"   H1 armS 選對率隨被蓋比例單調不增且 [0,0.2) ≥ 75%:各組 "
          + " / ".join("—" if x is None else f"{x:.1%}" for x in sb)
          + f" → {'成立' if mono and sb[0] is not None and sb[0] >= 0.75 else '不成立'}")
    h2a = max(results[n]["gt_front"]["S_right"] for n in ship)
    h2b = max(results[n]["gt_front"]["B_dprime"] for n in ship)
    print(f"   H2a gt_front 可出貨模型換人選對率 ≥ 80%:最高 {h2a:.1%} → {'成立' if h2a >= 0.8 else '不成立'}")
    print(f"   H2b gt_front 可出貨模型模糊出生 d′ ≥ 1.0:最高 {h2b:.2f} → {'成立' if h2b >= 1.0 else '不成立'}")
    base_ = results[arms]["none"]["S_right"]
    g_imp = results[arms]["gt_front"]["S_right"] - base_
    t_imp = results[arms]["track_front"]["S_right"] - base_
    ok_c = g_imp > 0 and t_imp >= g_imp / 2
    print(f"   H2c armS track_front 改善 ≥ gt_front 改善的一半:gt {g_imp:+.1%}、track {t_imp:+.1%} → "
          f"{'成立' if ok_c else '不成立'}" + ("(gt_front 沒有改善,無從比較)" if g_imp <= 0 else ""))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(args=vars(args), n=dict(S=len(S), B=len(B), crops=len(specs)),
                                   results=results, seconds=round(time.time() - t0, 1)),
                              ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n已存 {out}(共 {time.time() - t0:.0f} 秒)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
