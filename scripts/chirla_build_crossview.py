"""從 CHIRLA 的**推導集**估 F4(逐對地面單應性)與 F5(轉場出入口位置)的參數。

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

## F5 一併估:轉場的出入口位置

轉場路徑上唯一的實質證據是 Δt,而 2026-09-05 的量化診斷顯示
**3 條連結有 2 條在任何 Δt 都過不了門檻**(要 pdf > 0.24/秒,實際峰值 0.198)。
碎裂 54.42% 就是這麼來的。補的證據是「你是從通往這裡的那個門出去的嗎」,
同樣只從推導集學。⚠ σ 的處理與 F4 **不同**,理由見 `fit_place` 的說明。

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

import sys

import cv2
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

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


def transit_places(root, window_s=60.0, merge_gap=60):
    """回傳 {(A,B): [(exit_foot_A, enter_foot_B), ...]},只掃推導集。

    轉場的判定與 `chirla_build_topology.py` 的 `transitions()` 一致
    (同一身份、前一段在 A 結束、下一段在 B 開始、Δt 在窗內),
    差別只在這裡**同時記下兩端的腳點**。
    """
    aroot = Path(root) / "annotations"
    out = defaultdict(list)
    for seq in sorted(p for p in aroot.iterdir() if p.is_dir()):
        if seq.name not in DERIVATION:              # ⚠ 紅線
            continue
        # id -> cam -> {frame: bbox}
        per = defaultdict(lambda: defaultdict(dict))
        for f in sorted(seq.glob("*.json")):
            cam = phys(f.stem)
            for fr, dets in json.loads(f.read_text(encoding="utf-8")).items():
                for o in dets:
                    per[abs(int(o["id"]))][cam][int(fr)] = o["BboxP"]
        for _ident, by_cam in per.items():
            segs = []
            for cam, fb in by_cam.items():
                frs = sorted(fb)
                s = prev = frs[0]
                for x in frs[1:]:
                    if x - prev > merge_gap:
                        segs.append((s, prev, cam)); s = x
                    prev = x
                segs.append((s, prev, cam))
            segs.sort()
            for (s1, e1, c1), (s2, e2, c2) in zip(segs, segs[1:]):
                if c1 == c2 or s2 <= e1:
                    continue
                if not (0 < (s2 - e1) / 30.0 <= window_s):
                    continue
                out[(c1, c2)].append((foot(by_cam[c1][e1]), foot(by_cam[c2][s2])))
    return out


def fit_place(pts, min_n):
    """擬合退場/入場的二維高斯 + 共視區面積。

    ⚠ 這裡**不用留出樣本**(與 F4 的單應性不同),理由要講清楚:
      高斯的 μ/Σ 是閉式估計,不是最佳化出來的,不存在「擬合過的點殘差偏小」
      那種問題。真正的風險是 **n 小時 Σ 本身不準**,所以改用**預測共變異數**
      Σ_pred = Σ·(1 + 1/n) —— 把「我們對 Σ 的不確定度」也算進去。
      n 大時它趨近 Σ,n 小時它自動放寬(= 證據自動變弱),方向是安全的。
    """
    if len(pts) < min_n:
        return None
    ex = np.array([p[0] for p in pts], dtype=float)
    en = np.array([p[1] for p in pts], dtype=float)

    def g(a):
        n = len(a)
        mu = a.mean(axis=0)
        cov = np.cov(a.T, ddof=1) if n > 2 else np.eye(2) * 900.0
        return mu, np.asarray(cov, dtype=float).reshape(2, 2) * (1.0 + 1.0 / n)

    mu_e, cov_e = g(ex)
    mu_n, cov_n = g(en)
    # 「不同人」的散布範圍:該鏡頭裡實際觀察到的腳點凸包
    def area(a):
        if len(a) < 3:
            return 1280.0 * 720.0 * 0.3
        return max(float(cv2.contourArea(cv2.convexHull(a.astype(np.float32)))), 1e4)
    return (mu_e, cov_e, area(ex), mu_n, cov_n, area(en), len(pts))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--topology", default="configs/camera_topology.chirla.yaml")
    ap.add_argument("--out", default="configs/camera_topology.chirla_f4.yaml")
    ap.add_argument("--min-pairs", type=int, default=200,
                    help="少於這麼多組對應就不估 —— 3 點解出來的 H 是垃圾")
    ap.add_argument("--ransac-px", type=float, default=25.0)
    ap.add_argument("--seed", type=int, default=0)
    # F5
    ap.add_argument("--tp-min", type=int, default=15,
                    help="一條連結至少要有這麼多次轉場才估出入口位置")
    ap.add_argument("--tp-window", type=float, default=60.0)
    ap.add_argument("--tp-clip", type=float, default=6.0)
    ap.add_argument("--p-offdoor", type=float, default=0.10,
                    help="「這次沒走常走的那個門」的比例。它決定證據的下界 log(ε)")
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

    # ── F5:轉場的出入口位置 ──────────────────────────────────────────
    print("\n" + "=" * 76)
    print("F5:從推導集估各連結的出入口位置(TransitPlaceLR)")
    print("=" * 76)
    tp_raw = transit_places(args.root, window_s=args.tp_window)
    tp_out, tp_skip = {}, []
    print(f"\n  {'連結':<26}{'轉場數':>8}{'退場σ':>10}{'入場σ':>10}"
          f"{'門口LLR':>10}{'400px外':>10}")
    print("  " + "-" * 74)
    for (a, b), pts in sorted(tp_raw.items(), key=lambda kv: -len(kv[1])):
        r = fit_place(pts, args.tp_min)
        if r is None:
            tp_skip.append((a, b, len(pts)))
            continue
        mu_e, cov_e, ar_f, mu_n, cov_n, ar_t, n = r
        tp_out[f"{a}>{b}"] = {
            "mu_exit": [float(x) for x in mu_e],
            "cov_exit": [[float(x) for x in row] for row in cov_e],
            "area_from": round(float(ar_f), 1),
            "mu_enter": [float(x) for x in mu_n],
            "cov_enter": [[float(x) for x in row] for row in cov_n],
            "area_to": round(float(ar_t), 1), "n": n}
        from m5_reid.evidence import TransitPlaceLR
        m = TransitPlaceLR(mu_e, cov_e, ar_f, mu_n, cov_n, ar_t,
                           clip=args.tp_clip, p_offdoor=args.p_offdoor)
        at0 = m._term(tuple(mu_e), m.mu_e, m.ic_e, m.ld_e, m.log_area_from)
        at4 = m._term((mu_e[0] - 400, mu_e[1]), m.mu_e, m.ic_e, m.ld_e, m.log_area_from)
        se = math.sqrt(max(cov_e[0][0], cov_e[1][1]))
        sn = math.sqrt(max(cov_n[0][0], cov_n[1][1]))
        print(f"  {a+'→'+b:<26}{n:>8}{se:>9.0f}px{sn:>9.0f}px"
              f"{at0:>+10.2f}{at4:>+10.2f}")
    if tp_skip:
        print("\n  未建立(該連結加 0,行為與基線相同):")
        for a, b, n in tp_skip:
            print(f"    {a} → {b:<12} 只有 {n} 次轉場(需要 {args.tp_min})")

    cfg = yaml.safe_load(Path(args.topology).read_text(encoding="utf-8"))
    ct = cfg["camera_topology"]
    if tp_out:
        ct.setdefault("fusion", {})["transit_place"] = {
            "enabled": True, "clip": args.tp_clip, "p_offdoor": args.p_offdoor,
            "links": {k: {kk: vv for kk, vv in v.items() if kk != "n"}
                      for k, v in tp_out.items()}}
        # ⚠ 同一份資訊不可算兩次 —— CameraTopology 也會在建構時擋,這裡先關掉。
        ct["fusion"].setdefault("direction", {})["enabled"] = False
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
         "transit_place": {k: v["n"] for k, v in tp_out.items()},
         "transit_place_skipped": [{"a": a, "b": b, "n": n} for a, b, n in tp_skip],
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
