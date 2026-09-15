"""同一台鏡頭、幾秒內的外觀,能不能分開框重疊的人 —— 探索性診斷,非預先登記(2026-09-15)。

## 為什麼

§11–§12(`docs/M4_誤偵組成診斷_20260915.md`)查出:框重疊時,追蹤器只有框的位置/大小/重疊/運動可用,
分不出誰是誰;`rf_cbiou` 關聯完全不看外觀。既有外觀 d′(`chirla_armS` 0.118、DINOv2 0.25)是
**跨鏡頭、跨時間**量的,不能套到「同一台鏡頭、幾秒內」。這支直接在出問題的時刻量。

## 三組比對(規則執行前寫死;真值只用來選樣本與評分)

  S 換人時刻   `diag_m4_switch.py` 的換人事件中「a 在換人那一幀還有標註」的那些。
               模板 = 這條 track 最後配到 a 那一幀(f0)的 track 框;
               候選 = f1 那一幀 a 的標註框、b 的標註框。
               外觀選對 = cos(模板, a) > cos(模板, b)

  B 模糊出生   `diag_m4_birth_geometry.py` 中出生時與既有框重疊 ≥ 0.3、標記為 D 重複 或 N 新人 的出生。
               E = 出生幀與新框重疊最大的較早 track;模板 = E 在「出生幀 − 30 幀」或之前最近一幀的框;
               比對 = cos(模板, 新框)。重複應該像、真的新人應該不像

  C0 容易對照  同一台鏡頭、真人 track、框與同幀其他 track 重疊都 < 0.1(沒被擋):
               錨點 = X 在幀 t;同人 = X 在 t + 30 附近;不同人 = 同幀(t + 30 附近)另一條不同真人的 track。
               固定種子 0 抽 600 個錨點

## 指標

  d′ = (同人平均 − 不同人平均) / sqrt((同人變異 + 不同人變異) / 2);AUC
  S:外觀選對率。B:D 與 N 的 d′、AUC,以及「誤殺 N ≤ 5% 時能抓到的 D 比例」

## 假說(執行前寫下)

  H0 容易對照 C0 至少一個模型 d′ ≥ 2(模型本身在單鏡頭短時間有能力)
  H1 換人時刻 S 最好的模型選對率 ≥ 80%
  H2 模糊出生 B 至少一個模型 d′ ≥ 1.0

模型:DINOv2 vits14(可出貨)、chirla armS(可出貨)、chirla armR(⚠ 研究限定,只當上限對照)。

用法(在背景跑,需解碼影片):
    .venv/Scripts/python.exe scripts/diag_m4_appearance.py --root <CHIRLA根> \\
        --tracks-dir results/m4_5cam/cbiou_dump --seqs seq_004 ... \\
        --switch-json results/m4_5cam/switch_cbiou.json --births-json results/m4_5cam/birth_geometry_cbiou.json \\
        --out results/m4_5cam/appearance_cbiou.json
"""
import argparse
import bisect
import glob
import json
import math
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from diag_m4_ghosts import load_rows                     # noqa: E402
from eval_m4m5_chirla import iou, load_gt, match_tracks   # noqa: E402

BIRTH_IOU = 0.3
LAG = 30                 # 1 秒 × 30fps
FREE_IOU = 0.1
N_CONTROL = 600
MIN_SIDE = 8             # 裁圖最短邊(px)
MODELS = (("dinov2", "DINOv2 vits14(可出貨)"),
          ("chirla:model_result/reid/armS_reid_multi_camera/best.pth", "chirla armS(可出貨)"),
          ("chirla:model_result/reid/armR_reid_multi_camera/best.pth", "chirla armR(⚠ 研究限定)"))


def dprime(same, diff):
    if len(same) < 2 or len(diff) < 2:
        return None
    s, d = np.asarray(same), np.asarray(diff)
    return float((s.mean() - d.mean()) / math.sqrt((s.var() + d.var()) / 2 + 1e-12))


def auc(pos, neg):
    if not pos or not neg:
        return None
    allv = sorted([(v, 1) for v in pos] + [(v, 0) for v in neg])
    ranks, i = {}, 0
    r = np.empty(len(allv))
    while i < len(allv):
        j = i
        while j < len(allv) and allv[j][0] == allv[i][0]:
            j += 1
        r[i:j] = (i + j + 1) / 2
        i = j
    rp = sum(r[k] for k, (_v, y) in enumerate(allv) if y == 1)
    return float((rp - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def build_embedder(spec):
    if spec == "dinov2":
        from m5_reid.dino_embedder import DINOv2Embedder
        return DINOv2Embedder()
    from m5_reid.chirla_embedder import ChirlaEmbedder
    return ChirlaEmbedder(str(ROOT / spec.split(":", 1)[1]), quiet=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--tracks-dir", required=True)
    ap.add_argument("--seqs", nargs="+", required=True)
    ap.add_argument("--switch-json", required=True)
    ap.add_argument("--births-json", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    t0 = time.time()
    rng = random.Random(0)

    crops_needed = defaultdict(set)      # (seq, cam) -> {fid}
    specs = {}                           # crop id -> (seq, cam, fid, bbox)

    def need(seq, cam, fid, box):
        cid = len(specs)
        specs[cid] = (seq, cam, fid, tuple(box))
        crops_needed[(seq, cam)].add(fid)
        return cid

    switch = json.loads(Path(args.switch_json).read_text(encoding="utf-8"))["switches"]
    births = json.loads(Path(args.births_json).read_text(encoding="utf-8"))["births"]

    S, B, C0 = [], [], []
    for seq in args.seqs:
        gt = load_gt(args.root, seq)
        rows = load_rows(Path(args.tracks_dir) / seq / "tracks.csv")
        track_gt, *_ = match_tracks(gt, rows)
        box_at = {(r["cam"], r["tid"], r["fid"]): r["bbox"] for r in rows}
        fids_of = defaultdict(list)
        frame_tracks = defaultdict(list)
        for r in rows:
            fids_of[(r["cam"], r["tid"])].append(r["fid"])
            frame_tracks[(r["cam"], r["fid"])].append((r["tid"], r["bbox"]))
        for v in fids_of.values():
            v.sort()
        first = {k: v[0] for k, v in fids_of.items()}

        # S 換人時刻
        for s in switch:
            if s["seq"] != seq or not s["a_visible"]:
                continue
            g = dict(gt[s["cam"]].get(s["f1"] + 1, []))
            if s["b"] not in g:
                continue
            S.append(dict(cat=s["cat"], tmpl=need(seq, s["cam"], s["f0"], box_at[(s["cam"], s["tid"], s["f0"])]),
                          a=need(seq, s["cam"], s["f1"], g[s["a"]]), b=need(seq, s["cam"], s["f1"], g[s["b"]])))

        # B 模糊出生
        for bth in births:
            if bth["seq"] != seq or bth["iou"] < BIRTH_IOU or bth["label"] not in ("D 重複", "N 新人"):
                continue
            cam, fid = bth["cam"], bth["fid"]
            nbox = box_at[(cam, bth["tid"], fid)]
            cand = [(iou(nbox, bb), t) for t, bb in frame_tracks[(cam, fid)]
                    if t != bth["tid"] and first[(cam, t)] < fid]
            ov, e = max(cand)
            if abs(ov - bth["iou"]) > 1e-9:
                raise SystemExit(f"[FAIL] {seq} {cam} t{bth['tid']} 出生重疊與 birth_geometry 不一致")
            ef = fids_of[(cam, e)]
            j = bisect.bisect_right(ef, fid - LAG) - 1
            tf = ef[j] if j >= 0 else ef[0]
            B.append(dict(label=bth["label"], same_person_old=bth["n_same_person_old"],
                          tmpl=need(seq, cam, tf, box_at[(cam, e, tf)]), new=need(seq, cam, fid, nbox)))

        # C0 容易對照:錨點候選
        def free(cam, fid, tid, box):
            return all(iou(box, bb) < FREE_IOU for t, bb in frame_tracks[(cam, fid)] if t != tid)

        anchors = []
        for (cam, tid), ff in fids_of.items():
            if track_gt[(cam, tid)] is None:
                continue
            for f in ff[::6]:
                anchors.append((cam, tid, f))
        rng.shuffle(anchors)
        got = 0
        for cam, tid, f in anchors:
            if got >= N_CONTROL // len(args.seqs):
                break
            ff = fids_of[(cam, tid)]
            k = bisect.bisect_left(ff, f + LAG)
            if k >= len(ff) or ff[k] - f > LAG + 10:
                continue
            f2 = ff[k]
            ba, bp = box_at[(cam, tid, f)], box_at[(cam, tid, f2)]
            if not (free(cam, f, tid, ba) and free(cam, f2, tid, bp)):
                continue
            others = [(t, bb) for t, bb in frame_tracks[(cam, f2)]
                      if t != tid and track_gt[(cam, t)] not in (None, track_gt[(cam, tid)]) and free(cam, f2, t, bb)]
            if not others:
                continue
            ot, ob = rng.choice(sorted(others))
            C0.append(dict(tmpl=need(seq, cam, f, ba), same=need(seq, cam, f2, bp), diff=need(seq, cam, f2, ob)))
            got += 1

    print(f"樣本:S 換人 {len(S)}、B 模糊出生 {len(B)}(D {sum(b['label'] == 'D 重複' for b in B)} / "
          f"N {sum(b['label'] == 'N 新人' for b in B)})、C0 對照 {len(C0)};裁圖 {len(specs):,} 張"
          f"、影片 {len(crops_needed)} 支", flush=True)

    # 解碼:逐支影片循序 grab,只在需要的幀 retrieve(幀號與 common.video_io.iter_frames 相同)
    import cv2
    crops = {}
    by_frame = defaultdict(list)
    for cid, (seq, cam, fid, box) in specs.items():
        by_frame[(seq, cam, fid)].append(cid)
    for vi, ((seq, cam), fids) in enumerate(sorted(crops_needed.items())):
        vid = sorted(glob.glob(f"{args.root}/videos/{seq}/{cam}_*.avi"))[0]
        cap = cv2.VideoCapture(vid)
        last, f = max(fids), 0
        while f <= last:
            if not cap.grab():
                break
            if f in fids:
                ok, img = cap.retrieve()
                if ok:
                    h, w = img.shape[:2]
                    for cid in by_frame[(seq, cam, f)]:
                        x1, y1, x2, y2 = specs[cid][3]
                        x1, y1 = max(0, int(x1)), max(0, int(y1))
                        x2, y2 = min(w, int(round(x2))), min(h, int(round(y2)))
                        if x2 - x1 >= MIN_SIDE and y2 - y1 >= MIN_SIDE:
                            crops[cid] = img[y1:y2, x1:x2].copy()
            f += 1
        cap.release()
        print(f"   解碼 {vi + 1}/{len(crops_needed)} {seq} {cam}(累計 {time.time() - t0:.0f} 秒)", flush=True)
    print(f"裁圖成功 {len(crops):,} / {len(specs):,}", flush=True)

    results = {}
    for spec, name in MODELS:
        emb = build_embedder(spec)
        ids = sorted(crops)
        feats = {}
        for i in range(0, len(ids), 256):
            chunk = ids[i:i + 256]
            for cid, v in zip(chunk, emb.extract_batch([crops[c] for c in chunk])):
                feats[cid] = v
        cos = lambda a, b: float(np.dot(feats[a], feats[b]))
        ok = lambda *c: all(x in feats for x in c)

        c_same = [cos(x["tmpl"], x["same"]) for x in C0 if ok(x["tmpl"], x["same"], x["diff"])]
        c_diff = [cos(x["tmpl"], x["diff"]) for x in C0 if ok(x["tmpl"], x["same"], x["diff"])]
        s_ok = [x for x in S if ok(x["tmpl"], x["a"], x["b"])]
        s_right = [cos(x["tmpl"], x["a"]) > cos(x["tmpl"], x["b"]) for x in s_ok]
        s_by = defaultdict(list)
        for x, r in zip(s_ok, s_right):
            s_by[x["cat"]].append(r)
        b_ok = [x for x in B if ok(x["tmpl"], x["new"])]
        bd = [cos(x["tmpl"], x["new"]) for x in b_ok if x["label"] == "D 重複"]
        bn = [cos(x["tmpl"], x["new"]) for x in b_ok if x["label"] == "N 新人"]
        bn_real = [cos(x["tmpl"], x["new"]) for x in b_ok if x["label"] == "N 新人" and not x["same_person_old"]]
        # 誤殺 N ≤ 5% 時能抓到的 D:門檻取 N 相似度的 95 百分位
        thr = float(np.quantile(bn, 0.95)) if bn else None
        d_at5 = (sum(v > thr for v in bd) / len(bd)) if bd and thr is not None else None
        results[name] = dict(
            C0=dict(n=len(c_same), same_mean=float(np.mean(c_same)), diff_mean=float(np.mean(c_diff)),
                    dprime=dprime(c_same, c_diff), auc=auc(c_same, c_diff)),
            S=dict(n=len(s_ok), right_rate=sum(s_right) / len(s_ok) if s_ok else None,
                   by_cat={k: dict(n=len(v), right_rate=sum(v) / len(v)) for k, v in s_by.items()}),
            B=dict(n_d=len(bd), n_n=len(bn), d_mean=float(np.mean(bd)), n_mean=float(np.mean(bn)),
                   dprime=dprime(bd, bn), auc=auc(bd, bn), dprime_real_posthoc=dprime(bd, bn_real),
                   thr_n95=thr, d_recall_at_n_kill_5=d_at5))
        r = results[name]
        print(f"\n[{name}]", flush=True)
        print(f"   C0 容易對照 n={r['C0']['n']}:同人 {r['C0']['same_mean']:.3f} / 不同人 {r['C0']['diff_mean']:.3f}"
              f"  d′ {r['C0']['dprime']:.2f}  AUC {r['C0']['auc']:.3f}")
        print(f"   S 換人時刻 n={r['S']['n']}:外觀選對 {r['S']['right_rate']:.1%}  "
              + "、".join(f"{k} {v['right_rate']:.0%}({v['n']})" for k, v in r["S"]["by_cat"].items()))
        print(f"   B 模糊出生 D={r['B']['n_d']} N={r['B']['n_n']}:D {r['B']['d_mean']:.3f} / N {r['B']['n_mean']:.3f}"
              f"  d′ {r['B']['dprime']:.2f}  AUC {r['B']['auc']:.3f}  誤殺 N ≤ 5% 時抓到 D {r['B']['d_recall_at_n_kill_5']:.1%}"
              f"(事後口徑 d′ {r['B']['dprime_real_posthoc']:.2f})")
        del emb
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass

    print("\n假說對照(執行前寫下)")
    h0 = max(v["C0"]["dprime"] for v in results.values())
    h1 = max(v["S"]["right_rate"] for v in results.values())
    h2 = max(v["B"]["dprime"] for v in results.values())
    print(f"   H0 容易對照至少一個模型 d′ ≥ 2:最高 {h0:.2f} → {'成立' if h0 >= 2 else '不成立'}")
    print(f"   H1 換人時刻最好的模型選對率 ≥ 80%:最高 {h1:.1%} → {'成立' if h1 >= 0.8 else '不成立'}")
    print(f"   H2 模糊出生至少一個模型 d′ ≥ 1.0:最高 {h2:.2f} → {'成立' if h2 >= 1.0 else '不成立'}")
    print("   ⚠ armR 為研究限定,只當上限對照;可出貨的是 DINOv2 與 armS")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(args=vars(args), rules=dict(BIRTH_IOU=BIRTH_IOU, LAG=LAG, FREE_IOU=FREE_IOU,
                                                               N_CONTROL=N_CONTROL, MIN_SIDE=MIN_SIDE),
                                   n=dict(S=len(S), B=len(B), C0=len(C0), crops=len(specs), crops_ok=len(crops)),
                                   results=results, seconds=round(time.time() - t0, 1)),
                              ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n已存 {out}(共 {time.time() - t0:.0f} 秒)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
