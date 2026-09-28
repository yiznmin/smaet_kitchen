"""從**人工點選**的跨鏡頭地面對應點估單應性與 σ。

## 這支在回答什麼

F4 `CrossViewLR` 是沒有外觀模型時**唯一**能回答「重疊視野裡的是**哪一個**人」的證據,
而它的鑑別力由 σ(該鏡頭對的殘差尺度)決定。現行 6 對的 H 是從
「同一時刻、同一個人的**框底中點** + 真值身份」擬合的(`chirla_build_crossview.py:77-102`),
但框底中點**不是真的腳** —— 腳被工作台擋住、人坐著、抱東西,底邊都會跑掉。
結果 3 對的 σ 大到證據沒有鑑別力(74.69 / 37.10 / 16.91 px)。

本支改吃**手點的靜態場景特徵**(門檻、地磚交點、設備螺栓),量它能把 σ 壓到多少。
手點還有兩個附帶好處:**完全不需要真值身份**(真實廚房影片第一天就能做),
以及消除現行 H 的真值洩漏。

## ⚠ 為什麼不重用 `chirla_build_crossview.py` 的 `fit()`

那支有三道硬門檻:`len(pts) - n_ho < 30`(實際需 ≥50 點)、RANSAC 內點 `< 30`、
凸包 `area < 1000`。手點的 8~12 點根本進不去。**而那三道門檻正是它檔頭
「陷阱 2:σ 必須用留出樣本估,不能用擬合用的那些點」的執行機制** ——
為了小樣本把它們參數化放寬,會摧毀既有腳本的方法論保證。所以新寫。

## σ 的估計方式:自由度校正,不是 leave-one-out

⚠ **本檔第一版用 LOO 估 σ,自檢抓到它系統性高估,已改。** 實測(合成資料、
每格 200 次抽樣、注入已知雜訊,見 `--selftest`):

| 估計器 | N=8 | N=12 | N=20 | N=30 |
|---|---|---|---|---|
| LOO `median(res)/1.1774` | **+57%** | +27% | +14% | +7% |
| 全量殘差 RMS,除以 2N | **−31%** | −20% | −12% | −8% |
| **全量殘差 RMS,除以 (2N−8)** | **−3%** | −2% | −1% | −1% |

兩個偏誤的方向相反,原因不同:

- **LOO 高估**:單應性有 **8 個自由度**,N 組點只有 2N 條方程 → 平均槓桿 `h = 8/2N`
  (N=8 時高達 0.5)。抽掉一點會讓 `H_i` 明顯變差,LOO 殘差被 `1/(1−h)` 放大。
  LOO 估的是「N−1 點擬合」的誤差,小樣本下那比「N 點擬合」差很多。
- **除以 2N 低估**:殘差被最佳化過 —— 這就是陷阱 2 說的過度自信。

**除以 `2N − 8` 是這兩者之間唯一誠實的做法**:8 是單應性的自由度,`2N` 是純量方程數。
這不是「拿擬合點算殘差」那個陷阱 —— 陷阱在於除以 n 而不是 n−p,自由度校正正是它的解。
它同時是三者中**變異最小**的,所以在 8~12 點時也能支撐判準。

## LOO 保留下來做別的事:抓標歪的點

LOO 不適合估 σ,但它**很擅長指出哪一組標錯了**(自檢裡故意把一組偏 50px,
它的 LOO 殘差 72.1 對其他組的 3~27,一眼就看得出來)。所以逐組 LOO 殘差照印,
當作「這一組要不要重點」的依據。

## 三個沿用既有口徑的決定

1. **σ 是「每軸」的 σ**,與 `chirla_build_crossview.py:127` 的
   `median(res)/1.1774`(Rayleigh 中位數,res 是 2D 距離)同一個尺度,
   所以可以直接和 f4.yaml 的 74.69 比。`CrossViewLR` 的 `2πσ²` 也是這個約定。
2. **下限 `max(σ, 3.0)`**,與 `fit()` L128 同值同理由(點選量化誤差至少幾像素)。
   參照 `calibration.py:140-148` 的 `suggest_sigma` floor:「避免標定點太少時
   殘差假性偏低(過度自信)」。
3. ⚠⚠ **σ 不可以再乘 √2。** 它是直接從「兩台觀測的差」量出來的,**已含兩台的誤差**。
   `GroundPlaneLR` 的 σ 是單台標定誤差所以要乘,這裡不是。本專案已在
   `GroundPlaneLR`(第六輪)與 `VelocityLR`(第七輪)各踩過一次,不要有第三次。

## ⚠ 不用 RANSAC,改用最小平方

現行 `ransac_px=25.0` 對手點太鬆(手點誤差應該只有幾像素),而 8 點時 RANSAC
反而可能踢掉好點、留下壞點。改成最小平方,由 LOO 殘差自己暴露壞點。

## ⚠ area_px2 沿用 f4.yaml 的舊值,不用手點的凸包

舊 area 是「推導集裡實際觀測到的所有腳點」的凸包(`chirla_build_crossview.py:131-132`),
語意是「**不同人**時腳點散布的範圍」= 場景屬性。手點 8~12 點的凸包會嚴重低估它。
沿用舊值也讓新舊版的 LLR 直接可比(`log(area)` 在 `evidence.py` 裡是常數增益項)。
輸出會註明 area 的出處不是手點。

用法:
    python scripts/fit_pair_calib.py results/m5_reid/pair_calib/points.json
    python scripts/fit_pair_calib.py <points.json> --out results/m5_reid/pair_calib/fit.json
    python scripts/fit_pair_calib.py --selftest      # 合成資料自檢,不需任何輸入
"""
import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from m5_reid.calibration import _max_collinear          # noqa: E402

SIGMA_FLOOR_PX = 3.0        # 與 chirla_build_crossview.py:128 同值同理由
H_DOF = 8                   # 單應性的自由度(3x3 齊次,尺度自由)
RAYLEIGH_MEDIAN = 1.1774    # median(Rayleigh) ≈ 1.1774·σ,僅用於與舊口徑對照

# configs/fix_grid/f4.yaml 的既有值,用來對照「壓下來多少」。
# ⚠ 只是對照用的常數快照,不是真相來源 —— 若 f4.yaml 改了要同步。
F4_BASELINE = {
    "camera_2|camera_3": dict(sigma_px=16.91, area_px2=266408.2),
    "camera_1|camera_3": dict(sigma_px=8.36, area_px2=166166.0),
    "camera_2|camera_5": dict(sigma_px=74.69, area_px2=72989.8),
    "camera_4|camera_5": dict(sigma_px=37.10, area_px2=45867.5),
    "camera_6|camera_7": dict(sigma_px=8.71, area_px2=8270.5),
    "camera_1|camera_2": dict(sigma_px=7.99, area_px2=11278.2),
}


def _fit_h(src, dst, what=""):
    """最小平方擬合單應性。⚠ 不用 RANSAC —— 見檔頭。"""
    H, _ = cv2.findHomography(np.asarray(src, np.float64),
                              np.asarray(dst, np.float64), method=0)
    if H is None or not np.all(np.isfinite(H)):
        raise ValueError(f"無法求出 homography{what} —— 點是否共線或重複?")
    return H


def _project(H, pts):
    pts = np.asarray(pts, np.float64).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(pts, H).reshape(-1, 2)


def _check_not_collinear(pts, name):
    """⚠ 四點共線時 cv2 仍算得出一個矩陣,然後在真實資料上吐出荒謬座標。

    這是本專案反覆出現的「靜默失效」型態,明確擋掉。
    判準與 `calibration.py:55-58` 相同(借它的 `_max_collinear`,不重寫)。
    """
    if _max_collinear(np.asarray(pts, np.float64)) >= len(pts) - 0.5:
        raise ValueError(
            f"{name} 的點全部共線 —— 需要散開的點(不要三點以上落在同一條"
            "地板接縫上)。挑地面上分散的四角,近處遠處都要有。")


def _chi2_quantile(p, nu):
    """χ² 分位數的 Wilson–Hilferty 近似 —— 避免為了一個信賴區間相依 scipy。

    χ²_p,ν ≈ ν·(1 − 2/(9ν) + z_p·√(2/(9ν)))³,ν ≥ 3 時誤差在百分之幾以內,
    對「σ 大概落在哪個區間」這個用途足夠。
    """
    z = {0.025: -1.959964, 0.975: 1.959964}[p]
    t = 1.0 - 2.0 / (9.0 * nu) + z * math.sqrt(2.0 / (9.0 * nu))
    return nu * max(t, 1e-6) ** 3


def fit_pair(src, dst, loo=True):
    """回傳 (H, sigma_px, loo_res, info)。

    σ:全量殘差 RMS 除以自由度 `2N − 8`(見檔頭,實測近乎無偏且變異最小)。
    loo_res:leave-one-out 殘差,**只用來抓標歪的點**,不用來估 σ。

    loo=False 時跳過 LOO(回傳全 nan)。σ 完全不受影響 —— 自檢要跑上千次
    抽樣時 N 折擬合是純粹的浪費(第一版就是這樣慢到逾時)。
    """
    src = np.asarray(src, dtype=np.float64).reshape(-1, 2)
    dst = np.asarray(dst, dtype=np.float64).reshape(-1, 2)
    if len(src) != len(dst):
        raise ValueError(f"兩邊點數不一致:{len(src)} vs {len(dst)}")
    n = len(src)
    # 2N − 8 > 0 需要 N ≥ 5;LOO 每輪用 N−1 ≥ 4 組擬合也要求 N ≥ 5。
    if n < 5:
        raise ValueError(
            f"至少需要 5 組點(自由度 2N−8 要為正,LOO 每輪也要 N−1 ≥ 4),"
            f"目前只有 {n} 組。建議點 8~12 組。")
    _check_not_collinear(src, "影像 A")
    _check_not_collinear(dst, "影像 B")

    H = _fit_h(src, dst)
    res = np.hypot(*(_project(H, src) - dst).T)          # 每組的 2D 殘差(px)
    nu = 2 * n - H_DOF                                   # 純量方程數 − 自由度
    sigma_raw = math.sqrt(float((res ** 2).sum()) / nu)
    sigma = max(sigma_raw, SIGMA_FLOOR_PX)

    # 95% 信賴區間:σ̂²·ν/σ² ~ χ²_ν
    lo = sigma_raw * math.sqrt(nu / _chi2_quantile(0.975, nu))
    hi = sigma_raw * math.sqrt(nu / _chi2_quantile(0.025, nu))

    # LOO 殘差 —— 只為了指出哪一組要重點。
    loo_res = np.full(n, np.nan)
    if loo:
        for i in range(n):
            keep = [j for j in range(n) if j != i]
            try:
                _check_not_collinear(src[keep], f"影像 A(抽掉第 {i + 1} 組後)")
                _check_not_collinear(dst[keep], f"影像 B(抽掉第 {i + 1} 組後)")
                H_i = _fit_h(src[keep], dst[keep])
                loo_res[i] = float(np.hypot(*(_project(H_i, src[i:i + 1])[0] - dst[i])))
            except ValueError:
                pass                     # 抽掉它就退化 → 它是撐住幾何的關鍵點,留 nan
    return H, sigma, loo_res, dict(
        n=n, dof=nu, sigma_raw=round(sigma_raw, 4),
        floor_hit=bool(sigma_raw < SIGMA_FLOOR_PX),
        ci95=[round(lo, 2), round(hi, 2)],
        fit_res_px=[round(float(v), 3) for v in res],
        fit_res_median_px=round(float(np.median(res)), 3),
        fit_res_max_px=round(float(res.max()), 3))


def peak_llr(area_px2, sigma_px):
    """LLR 在殘差為 0 時的峰值項:log A − log(2πσ²)。

    這是「σ 壓下來到底值多少 nats」的直接答案。見 `evidence.py` 的 CrossViewLR:
        LLR = log A − log(2πσ²) − d²/(2σ²)
    """
    return math.log(area_px2) - math.log(2 * math.pi * sigma_px * sigma_px)


def _selftest():
    """合成資料自檢。σ 估得對不對,只有這個方法驗得出來。"""
    ok = True

    def chk(cond, name, detail=""):
        nonlocal ok
        print(f"  {'OK  ' if cond else 'FAIL'} {name}" + (f"  -- {detail}" if detail else ""))
        if not cond:
            ok = False

    H_true = np.array([[0.90, 0.05, 30.0],
                       [-0.08, 1.10, -20.0],
                       [1.2e-4, 3.0e-5, 1.0]], dtype=np.float64)

    def draw(rng, n, noise):
        src = np.column_stack([rng.uniform(80, 1000, n), rng.uniform(240, 700, n)])
        return src, _project(H_true, src) + rng.normal(0.0, noise, (n, 2))

    print("[1] σ 無偏性(每格 120 次抽樣,注入已知雜訊)")
    for n in (8, 12, 20):
        for noise in (3.0, 8.0, 15.0):
            rng = np.random.default_rng(1000 + n * 10 + int(noise))
            est = []
            for _ in range(120):
                s, d = draw(rng, n, noise)
                # loo=False:這一格只要 σ,N 折擬合是純浪費(第一版就這樣逾時)
                est.append(fit_pair(s, d, loo=False)[3]["sigma_raw"])
            bias = np.mean(est) / noise - 1
            chk(abs(bias) < 0.10,
                f"N={n:>2} 注入 σ={noise:>4.1f} → 平均 {np.mean(est):.2f}",
                f"偏誤 {bias:+.1%}(要求 |偏誤| < 10%)")

    print("[2] 兩個錯誤的估計器,方向相反 —— 這是改設計的依據")
    rng = np.random.default_rng(7)
    naive, loo_est, good = [], [], []
    for _ in range(120):
        s, d = draw(rng, 8, 8.0)
        H, _sig, loo, info = fit_pair(s, d)
        res = np.array(info["fit_res_px"])
        naive.append(math.sqrt((res ** 2).sum() / (2 * len(s))))   # 除以 2N
        loo_est.append(float(np.nanmedian(loo)) / RAYLEIGH_MEDIAN)
        good.append(info["sigma_raw"])
    chk(np.mean(naive) < 8.0 * 0.85,
        "除以 2N 會低估(陷阱 2 說的過度自信)", f"{np.mean(naive):.2f} vs 注入 8.0")
    chk(np.mean(loo_est) > 8.0 * 1.25,
        "LOO 會高估(槓桿 h=8/2N,N=8 時達 0.5)", f"{np.mean(loo_est):.2f} vs 注入 8.0")
    chk(abs(np.mean(good) / 8.0 - 1) < 0.10,
        "除以 2N−8 落在兩者之間且近乎無偏", f"{np.mean(good):.2f} vs 注入 8.0")
    chk(np.std(good) < np.std(loo_est),
        "而且變異比 LOO 小", f"std {np.std(good):.2f} < {np.std(loo_est):.2f}")

    print("[3] 信賴區間要蓋住真值")
    rng = np.random.default_rng(11)
    hit = 0
    for _ in range(200):
        s, d = draw(rng, 10, 8.0)
        lo, hi = fit_pair(s, d, loo=False)[3]["ci95"]
        hit += (lo <= 8.0 <= hi)
    chk(hit / 200 >= 0.88, f"95% 區間實測覆蓋率 {hit / 200:.0%}", "近似法,允許略低於 95%")

    print("[4] LOO 仍要抓得出標歪的那一組")
    rng = np.random.default_rng(3)
    s, d = draw(rng, 10, 2.0)
    d[3] += np.array([50.0, -50.0])
    _H, _s2, loo, _i = fit_pair(s, d)
    chk(int(np.nanargmax(loo)) == 3, "LOO 殘差最大的正是被標偏的第 4 組",
        f"逐組 = {np.round(loo, 1).tolist()}")

    print("[5] 防呆")
    try:
        fit_pair(s[:4], d[:4]); chk(False, "N=4 應該拋錯")
    except ValueError as e:
        chk("至少需要 5 組" in str(e), "N=4 明確拋錯", str(e)[:34])
    col = np.column_stack([np.linspace(0, 500, 6), np.linspace(0, 500, 6)])
    try:
        fit_pair(col, _project(H_true, col)); chk(False, "共線應該拋錯")
    except ValueError as e:
        chk("共線" in str(e), "共線明確拋錯", str(e)[:34])

    print("[6] 峰值 LLR 的換算")
    area = F4_BASELINE["camera_2|camera_5"]["area_px2"]
    old, new = peak_llr(area, 74.69), peak_llr(area, 8.0)
    chk(new > old, f"σ 74.69 → 8.0 時峰值 {old:+.2f} → {new:+.2f} nats",
        f"增加 {new - old:+.2f}(門檻 1.6094)")

    print("\n" + ("全部通過。" if ok else "**有項目失敗。**"))
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("points", nargs="?", help="make_pair_calib_html.py 匯出的 JSON")
    ap.add_argument("--out", default=None)
    ap.add_argument("--selftest", action="store_true", help="合成資料自檢,不需輸入")
    args = ap.parse_args()

    if args.selftest:
        return _selftest()
    if not args.points:
        ap.error("要給 points.json,或用 --selftest")

    d = json.loads(Path(args.points).read_text(encoding="utf-8"))
    a, b = d["pair"]
    pts = d["points"]
    H, sigma, loo, info = fit_pair([p[a] for p in pts], [p[b] for p in pts])

    key = f"{a}|{b}"
    base = F4_BASELINE.get(key) or F4_BASELINE.get(f"{b}|{a}")
    area = base["area_px2"] if base else None

    print(f"[{key}]  seq={d.get('seq')}  frame={d.get('frame')}  "
          f"{info['n']} 組手點(自由度 2N−8 = {info['dof']})")
    print()
    print("  逐組殘差(px)—— LOO 那欄是用來抓標歪的點,不是用來估 σ:")
    print("     組   擬合殘差    LOO 殘差")
    worst = int(np.nanargmax(loo)) if not np.all(np.isnan(loo)) else -1
    for i, (r, l) in enumerate(zip(info["fit_res_px"], loo)):
        ls = "  退化(關鍵點)" if math.isnan(l) else f"{l:>9.2f}"
        print(f"    {i + 1:>3}  {r:>9.2f}  {ls}"
              + ("   ← LOO 最大,建議重點這一組" if i == worst else ""))
    print()
    print(f"  **σ = {sigma:.2f} px**  95% 區間 [{info['ci95'][0]}, {info['ci95'][1]}]")
    if info["floor_hit"]:
        print(f"    ⚠ 已被下限 3.0 咬住(原始估計 {info['sigma_raw']:.2f})—— "
              "不採信比點選量化誤差還小的值")

    out = dict(pair=[a, b], seq=d.get("seq"), frame=d.get("frame"),
               source="manual_static_features",
               sigma_estimator="full-fit RMS / (2N-8)",
               n_points=info["n"], dof=info["dof"],
               sigma_px=round(sigma, 2), sigma_raw_px=info["sigma_raw"],
               sigma_ci95_px=info["ci95"], floor_hit=info["floor_hit"],
               fit_residuals_px=info["fit_res_px"],
               loo_residuals_px=[None if math.isnan(v) else round(float(v), 3)
                                 for v in loo],
               H=[[float(v) for v in row] for row in H],
               area_px2=area,
               area_provenance=("沿用 configs/fix_grid/f4.yaml —— 它是推導集觀測腳點的"
                                "凸包(場景屬性),**不是**手點的凸包" if area else None))

    if base:
        old_s = base["sigma_px"]
        drop = (old_s - sigma) / old_s
        sigma_max = math.sqrt(area / 12.0)
        old_p, new_p = peak_llr(area, old_s), peak_llr(area, sigma)
        verdict = "成功" if sigma <= 10 else "部分成功" if sigma <= 20 else "**失敗**"
        print()
        print(f"  對照 f4.yaml:σ {old_s:.2f} → {sigma:.2f} px({drop:+.1%})")
        print(f"  sigma_max = sqrt(area/12) = {sigma_max:.1f} px → 新 σ "
              f"{'未被' if sigma < sigma_max else '**被**'}這個上限咬到")
        print()
        print("  證據強度(LLR 峰值項 log A − log 2πσ²,門檻 1.6094):")
        print(f"    現行 f4  {old_p:+.2f} nats")
        print(f"    手點     {new_p:+.2f} nats   ({new_p - old_p:+.2f})")
        print()
        print(f"  判準(預先登記:≤10 成功 / 10~20 部分 / >20 失敗)→ **{verdict}**")
        out.update(f4_sigma_px=old_s, sigma_drop=round(drop, 4),
                   sigma_max_px=round(sigma_max, 1),
                   peak_llr_f4=round(old_p, 4), peak_llr_manual=round(new_p, 4),
                   verdict=verdict)

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(
            json.dumps(out, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8")
        print(f"\n已存 {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
