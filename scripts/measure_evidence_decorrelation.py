"""量「連續觀測多久才帶來新資訊」—— P1 累積投票的取樣間隔必須由這支決定。

## 為什麼非量不可

2026-09-13 的診斷:M5 在 track **出現那一瞬間**做一次綁定決策,之後不再重評。
但一條 track 中位活 26 個迴圈、p90 活 1742 個 —— **我們丟掉了 96% 的機會**。

冠軍法(AI City Track 1,IDF1 95.36)的作法是逐幀指派 + **滑動窗多數決**。

⚠ **但不可以把連續幀的 LLR 直接相加。** 人在 1/30 秒內幾乎沒動,
第二幀的位置證據幾乎不帶新資訊。直接累加 26 幀 × 1.12 nats = 29 nats
是**假的自信** —— 與專案在 `VelocityLR`(第七輪 OU 去相關)踩過的坑同一形狀,
也與 `GroundPlaneLR`(第六輪)的 √2 錯誤同一類。

所以取樣間隔**必須從資料量**,不得用猜的。

## 判準:什麼時候算「帶來新資訊」

位置證據是 `LLR ∝ −d²/(2σ²)`,其中 σ 是該鏡頭對的殘差尺度。
兩次觀測要能提供**不同**的證據,人至少要移動超過 σ ——
否則兩次算出來的 d 幾乎一樣,第二票只是把第一票再投一次。

所以量:**腳點位移超過 σ 需要多久**(σ 掃過實際擬合出來的範圍)。

⚠ 這是**保守**的判準:位移超過 σ 只代表「證據會變」,不代表「完全獨立」。
  真正的獨立需要位移遠大於人與人的典型間距。報告時要照這樣寫,
  不得把「超過 σ」講成「獨立」。

用法:
    python scripts/measure_evidence_decorrelation.py --tracks results/m5_full9/tracks.csv
    python scripts/measure_evidence_decorrelation.py --tracks <CHIRLA 的 tracks.csv> --fps 30 --stride 5
"""
import argparse
import csv
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

import numpy as np

# 9/12 實際擬合出來的六對單應性的 σ(px),見 configs/camera_topology.chirla_f4.yaml
FITTED_SIGMAS = [8.0, 8.4, 8.7, 16.9, 37.1, 74.7]


def foot(r):
    return ((float(r["x1"]) + float(r["x2"])) * 0.5, float(r["y2"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracks", required=True, help="m5_track_video.py 產出的 tracks.csv")
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--stride", type=int, default=5, help="跑的時候幾幀取一次")
    ap.add_argument("--min-len", type=int, default=10, help="太短的 track 沒有投票空間,略過")
    ap.add_argument("--out", default="results/m5_reid/evidence_decorrelation.json")
    args = ap.parse_args()

    sec_per_loop = args.stride / args.fps
    rows = list(csv.DictReader(open(args.tracks, encoding="utf-8")))
    by_track = defaultdict(dict)                 # (cam, tid) -> {loop_i: foot}
    for r in rows:
        by_track[(r["camera_id"], r["track_id"])][int(r["loop_i"])] = foot(r)

    tracks = {k: v for k, v in by_track.items() if len(v) >= args.min_len}
    lens = sorted(len(v) for v in tracks.values())
    print("=" * 78)
    print("證據去相關量測 —— 決定累積投票的取樣間隔")
    print("=" * 78)
    print(f"\n  來源 {args.tracks}")
    print(f"  {len(by_track)} 條 track,其中 {len(tracks)} 條長度 ≥ {args.min_len}")
    if not tracks:
        raise SystemExit("沒有夠長的 track —— 投票這件事在這份資料上無從量起")
    print(f"  長度:中位 {st.median(lens):.0f} / p90 {lens[int(len(lens) * .9)]} / 最長 {lens[-1]} 個迴圈")

    # 位移 vs Δloop 曲線
    max_lag = min(200, lens[-1] - 1)
    lags = sorted(set([1, 2, 3, 5, 8, 12, 20, 30, 50, 80, 120, 200]) & set(range(1, max_lag + 1)))
    curve = {}
    for lag in lags:
        d = []
        for v in tracks.values():
            ks = sorted(v)
            for i in ks:
                if i + lag in v:
                    a, b = v[i], v[i + lag]
                    d.append(float(np.hypot(a[0] - b[0], a[1] - b[1])))
        if len(d) >= 30:
            curve[lag] = dict(n=len(d), median=float(np.median(d)),
                              p25=float(np.percentile(d, 25)),
                              p75=float(np.percentile(d, 75)))

    print(f"\n  腳點位移 vs 時間差(單位 px;{sec_per_loop:.2f} 秒/迴圈)")
    print(f"    {'Δ迴圈':>7}{'Δ秒':>8}{'位移中位':>10}{'p25':>9}{'p75':>9}{'樣本':>9}")
    print("    " + "-" * 52)
    for lag, c in curve.items():
        print(f"    {lag:>7}{lag * sec_per_loop:>8.2f}{c['median']:>10.1f}"
              f"{c['p25']:>9.1f}{c['p75']:>9.1f}{c['n']:>9}")

    # 對每個實際擬合出來的 σ,找「位移中位超過 σ」所需的 Δ
    print(f"\n  位移中位超過 σ 需要多久?(σ 取 9/12 實際擬合的六對)")
    print(f"    {'σ(px)':>8}{'需要Δ迴圈':>11}{'需要Δ秒':>11}  說明")
    print("    " + "-" * 60)
    need = {}
    for sig in FITTED_SIGMAS:
        hit = next((lag for lag, c in curve.items() if c["median"] >= sig), None)
        need[sig] = hit
        if hit is None:
            print(f"    {sig:>8.1f}{'>' + str(max(curve)):>11}{'':>11}  "
                  f"⚠ 掃到 Δ={max(curve)} 都沒超過 —— 這一對的證據幾乎不會變")
        else:
            print(f"    {sig:>8.1f}{hit:>11}{hit * sec_per_loop:>11.2f}  "
                  f"{'取樣間隔至少要這麼大' if hit > 1 else '每迴圈都算新資訊'}")

    # 給 P1 的建議取樣間隔:取「多數 σ 都能超過」的那個 Δ
    ok = [v for v in need.values() if v is not None]
    rec = int(np.median(ok)) if ok else None
    print("\n  " + "─" * 74)
    if rec:
        print(f"  建議取樣間隔 ≥ **{rec} 個迴圈**({rec * sec_per_loop:.2f} 秒)")
        vote = st.median(lens) / rec
        print(f"  → 中位長度 {st.median(lens):.0f} 的 track 可投 **{vote:.1f} 票**")
        if vote < 3:
            print("  ⚠ 票數 < 3 —— 多數決在這份資料上幾乎沒有空間,P1 的預期效益要下修")
    else:
        print("  ⚠ 沒有任何 σ 在掃描範圍內被超過 —— 位置證據在這份資料上幾乎是常數,")
        print("    重複投票不會帶來新資訊。P1 必須改用別的證據軸(外觀/候選集變化)。")
    print("  " + "─" * 74)
    print("\n  ⚠ 「位移 > σ」是**保守**判準:它只代表證據會變,不代表兩次觀測獨立。")
    print("    真正的獨立需要位移遠大於人與人的典型間距。不得把它講成「獨立」。")

    p = Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({
        "source": args.tracks, "sec_per_loop": sec_per_loop,
        "n_tracks": len(by_track), "n_usable": len(tracks),
        "len_median": st.median(lens), "len_p90": lens[int(len(lens) * .9)],
        "displacement_curve": curve,
        "lag_to_exceed_sigma": {str(k): v for k, v in need.items()},
        "recommended_stride_loops": rec, "args": vars(args),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已存 {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
