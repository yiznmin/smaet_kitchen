"""從 CHIRLA 的**推導集**估各鏡頭對的地面單應性,供 F4 CrossViewLR 使用。

## 為什麼這件事做得成

CHIRLA 沒有相機標定參數,所以 `GroundPlaneLR` 一直不生效,重疊路徑只能用
`overlap_llr = 5.0` 這個常數 —— 而它**在數學上不可能拒絕任何候選**
(5.0 − log(k) + app < 1.609 需要 k > 29.7,而單台鏡頭同時最多 9 人)。
2026-09-04 實測後果:重疊路徑誤併 **80.80%**、正確率 **16.49%**(≈1/6,與隨機無異)。

**但我們有逐幀真值身份。** 同一時刻兩台看到同一個人 → 那就是一組地面對應點
(取 bbox 底邊中點)。夠多組就能用 RANSAC 解出 H。

⚠ 這**不是**度量標定 —— 沒有公尺、沒有外參。它只回答
「這兩個觀測落在地面的同一點嗎」,而那正是我們要的。

## ⚠ 三個方法論陷阱,都已擋住

**1. 只能用推導集。** `docs/CHIRLA_M4M5驗證_預先登記_20260903.md` §3 定死:
   拓撲的每一個參數只能從 seq_000/001/002 估,評估集不得參與。
   本腳本**硬性拒絕**推導集以外的序列,`--derivation` 改不了這件事。

**2. σ 必須用留出樣本估,不能用擬合用的那些點。**
   拿擬合殘差當 σ 會系統性低估(那些點正是被 H 最佳化過的)→ σ 太小 →
   LLR 過度自信 → **把真正的同一人判成位置對不上**。
   本專案在 `GroundPlaneLR` 上踩過這個形狀的坑(第六輪:真的那位只拿到 +0.20 nats)。

**3. σ 是 pairwise 殘差,不可再乘 √2。**
   它是直接從「兩台觀測的差」量出來的,已含兩台的誤差。
   `GroundPlaneLR` 的 σ 是單台標定誤差所以要乘,這裡不是。
   本專案已在 GroundPlaneLR(第六輪)與 VelocityLR(第七輪)各踩一次,不要第三次。

用法:
    python scripts/chirla_build_crossview.py --root "D:/.../CHIRLA" \\
        --topology configs/camera_topology.chirla.yaml \\
        --out configs/camera_topology.chirla_f4.yaml
"""
import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import yaml

# 預先登記 §3 定死的推導集。⚠ 不提供 CLI 覆寫 —— 這是紅線不是預設值。
DERIVATION = ("seq_000", "seq_001", "seq_002")
# 「兩邊都拍得清楚」的門檻,沿用 chirla_overlap_stats.py(Re-ID 慣用輸入的一半)
GOOD_W, GOOD_H = 50, 120


def phys(name):
    return "_".join(name.split("_")[:2])


def foot(b):
    """bbox 底邊中點 —— 站立的人與地面的接觸點。"""
    return ((float(b[0]) + float(b[2])) * 0.5, float(b[3]))


def big(b):
    return (float(b[2]) - float(b[0])) >= GOOD_W and (float(b[3]) - float(b[1])) >= GOOD_H


def collect(root):
    """回傳 {(cam_a, cam_b): [(foot_a, foot_b), ...]},只掃推導集。"""
    aroot = Path(root) / "annotations"
    if not aroot.exists():
        raise SystemExit(f"找不到 {aroot}")
    pairs = defaultdict(list)
    n_seq = 0
    for seq in sorted(p for p in aroot.iterdir() if p.is_dir()):
        if seq.name not in DERIVATION:          # ⚠ 紅線:評估集不得參與
            continue
        n_seq += 1
        seen = defaultdict(dict)                # (frame, id) -> cam -> bbox
        for f in sorted(seq.glob("*.json")):
            cam = phys(f.stem)
            for fr, dets in json.loads(f.read_text(encoding="utf-8")).items():
                for o in dets:
                    seen[(int(fr), abs(int(o["id"])))][cam] = o["BboxP"]
        for _k, v in seen.items():
            cams = sorted(c for c in v if big(v[c]))
            for i, a in enumerate(cams):
                for b in cams[i + 1:]:
                    pairs[(a, b)].append((foot(v[a]), foot(v[b])))
    if n_seq == 0:
        raise SystemExit(f"推導集 {DERIVATION} 一個都沒找到 —— 資料路徑對嗎?")
    print(f"  掃了推導集 {n_seq} 個序列(定死為 {DERIVATION})")
    return pairs


def fit(pts, rng, holdout=0.3, ransac_px=25.0):
    """擬合 H 並用**留出樣本**估 σ。回傳 (H, sigma_px, area_px2, 診斷)。"""
    src = np.array([p[0] for p in pts], dtype=np.float64)
    dst = np.array([p[1] for p in pts], dtype=np.float64)
    idx = rng.permutation(len(pts))
    n_ho = max(20, int(len(pts) * holdout))
    if len(pts) - n_ho < 30:                    # 擬合樣本太少,不做
        return None
    ho, tr = idx[:n_ho], idx[n_ho:]

    H, mask = cv2.findHomography(src[tr], dst[tr], cv2.RANSAC, ransac_px)
    if H is None:
        return None
    inl = int(mask.sum()) if mask is not None else 0
    if inl < 30:
        return None

    # ⚠ σ 用留出樣本 —— 見檔頭陷阱 2。
    proj = cv2.perspectiveTransform(src[ho].reshape(-1, 1, 2), H).reshape(-1, 2)
    res = np.hypot(*(proj - dst[ho]).T)
    # 用中位數推 σ 而不是平均:留出樣本裡仍會有標註錯誤與非地面點(被抱起來的
    # 東西、坐著的人),平均會被它們拉走。1.4826 是 MAD→σ 的一致性常數。
    sigma = float(np.median(res)) / 1.1774      # median of Rayleigh ≈ 1.1774σ
    sigma = max(sigma, 3.0)                     # 下限:腳點量化誤差至少幾個像素

    # 「不同人」時腳點的散布範圍 = 目標鏡頭裡實際觀測到的腳點凸包面積
    hull = cv2.convexHull(dst.astype(np.float32))
    area = float(cv2.contourArea(hull))
    if area < 1000.0:                           # 退化(全擠在一條線上)
        return None
    return H, sigma, area, dict(n=len(pts), n_fit=len(tr), n_holdout=n_ho,
                                inliers=inl, inlier_rate=round(inl / len(tr), 3),
                                res_median_px=round(float(np.median(res)), 1),
                                res_p90_px=round(float(np.percentile(res, 90)), 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--topology", default="configs/camera_topology.chirla.yaml")
    ap.add_argument("--out", default="configs/camera_topology.chirla_f4.yaml")
    ap.add_argument("--min-pairs", type=int, default=200,
                    help="少於這麼多組對應就不估 —— 3 點解出來的 H 是垃圾")
    ap.add_argument("--ransac-px", type=float, default=25.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    print("=" * 76)
    print("F4:從推導集估各鏡頭對的地面單應性")
    print("=" * 76)
    pairs = collect(args.root)
    rng = np.random.default_rng(args.seed)

    print(f"\n  {'鏡頭對':<26}{'對應點':>8}{'內點率':>8}{'殘差中位':>10}"
          f"{'σ':>8}{'共視區':>11}{'同人LLR':>9}{'3σ處':>8}")
    print("  " + "-" * 90)
    out, skipped = {}, []
    for (a, b), pts in sorted(pairs.items(), key=lambda kv: -len(kv[1])):
        if len(pts) < args.min_pairs:
            skipped.append((a, b, len(pts), "對應點太少"))
            continue
        r = fit(pts, rng, ransac_px=args.ransac_px)
        if r is None:
            skipped.append((a, b, len(pts), "擬合失敗/退化"))
            continue
        H, sigma, area, diag = r
        s = min(sigma, math.sqrt(area / 12.0))
        at0 = math.log(area) - math.log(2 * math.pi * s * s)
        at3 = at0 - (3 * sigma) ** 2 / (2 * s * s)
        out[f"{a}|{b}"] = {"H": [[float(x) for x in row] for row in H],
                           "sigma_px": round(sigma, 2),
                           "area_px2": round(area, 1), "diag": diag}
        print(f"  {a+' + '+b:<26}{len(pts):>8}{diag['inlier_rate']:>8.2f}"
              f"{diag['res_median_px']:>9.1f}px{sigma:>8.1f}{area:>11.0f}"
              f"{at0:>+9.2f}{max(at3,-8.0):>+8.2f}")

    if skipped:
        print("\n  未建立(這些鏡頭對會**自動退回舊的常數路徑**,不是錯誤):")
        for a, b, n, why in skipped:
            print(f"    {a} + {b:<12} {n:>6} 組 —— {why}")

    cfg = yaml.safe_load(Path(args.topology).read_text(encoding="utf-8"))
    ct = cfg["camera_topology"]
    ct.setdefault("fusion", {})["cross_view"] = {
        "enabled": True, "clip": 8.0, "speed_px_per_s": 0.0,
        "pairs": {k: {kk: vv for kk, vv in v.items() if kk != "diag"}
                  for k, v in out.items()}}
    hdr = (f"# CHIRLA 拓撲 + F4 跨鏡頭單應性({Path(__file__).name} 產生)\n"
           f"#\n"
           f"# ⚠ 單應性**只從推導集 {'/'.join(DERIVATION)} 估**,評估集完全沒有參與。\n"
           f"#   腳本硬性拒絕推導集以外的序列,見 DERIVATION 常數。\n"
           f"# ⚠ σ 用**留出樣本**估(30%),不是擬合殘差 —— 後者會系統性低估,\n"
           f"#   使 LLR 過度自信而把真正的同一人判成位置對不上。\n"
           f"# ⚠ σ 是 pairwise 殘差,**不可再乘 √2**(已含兩台鏡頭的誤差)。\n"
           f"#\n"
           f"# 建立了 {len(out)} 對,未建立 {len(skipped)} 對(自動退回常數路徑)。\n")
    Path(args.out).write_text(hdr + yaml.safe_dump(cfg, allow_unicode=True,
                                                   sort_keys=False, width=200),
                              encoding="utf-8")
    diag_p = Path("results/m5_reid/crossview_fit.json")
    diag_p.parent.mkdir(parents=True, exist_ok=True)
    diag_p.write_text(json.dumps(
        {"pairs": {k: v["diag"] for k, v in out.items()},
         "skipped": [{"a": a, "b": b, "n": n, "why": w} for a, b, n, w in skipped],
         "args": vars(args), "derivation": list(DERIVATION)},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  已寫出 {args.out}")
    print(f"  診斷 {diag_p}")
    if not out:
        print("\n  ❌ 一對都沒建立 —— F4 開了也等於沒開,先查資料路徑與 --min-pairs")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
