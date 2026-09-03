"""外觀品質 × 佈局 的 2×2 —— 直接預測 CHIRLA 訓練出來的模型值多少。

## 為什麼之前的結論不夠

我們量過「把外觀從實測水準降到零,碎裂率一個小數點都沒動」,結論是
「外觀對本架構幾乎沒有影響」。但那個實驗有兩個限制:

1. **只往下掃,沒往上掃。** γ 從 0 到 1 是讓外觀**變差**。而 CHIRLA 要做的是
   讓它**變好** —— 那個方向從來沒量過。
2. **是在有全景鏡頭的世界量的。** 那種世界一切走重疊路徑,而重疊路徑有地面校正
   這條強證據壓著,外觀根本沒有舞台。

⚠ 2026-09-01 發現:**地面校正與軌跡只在重疊路徑生效,跨時轉場路徑完全沒有**
  (`docs/M5_模擬預先登記_幾何世界_20260901.md` §9.3)。
  所以**沒有全景鏡頭時,轉場路徑上能用的只剩轉場時間 + 外觀** ——
  外觀在那裡的權重可能完全不同。

## 2×2

| | 有全景鏡頭(重疊主導) | 無全景鏡頭(轉場主導) |
|---|---|---|
| **DINOv2 品質**(同人 0.490 / 不同人 0.465) | 基準 | ? |
| **OSNet 品質**(0.618 / 0.488) | ? | ? |

世界端與系統端的外觀參數**同步設定**(世界產生什麼品質的特徵,系統就用對應的
實測分布去判斷)—— 否則量到的是模型失配而不是外觀價值。

**這直接回答:如果 CHIRLA 訓練出 OSNet 等級的模型,對整套系統值多少?**
而且答案很可能**取決於佈局** —— 那本身就是給業主的重要建議。

用法:
    python scripts/sweep_appearance_value.py --chefs 4 --reps 8 --hours 1.0
"""
import argparse
import json
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from m5_reid import metrics                                   # noqa: E402
from m5_reid.evidence import AppearanceLR                     # noqa: E402
from m5_sim import world as W                                 # noqa: E402
from sim_m5_montecarlo import _base_topo, run_once            # noqa: E402

# 世界端的外觀品質。數字來自 EPFL 449 crops 的實測(evidence.AppearanceLR.MEASURED)
QUALITY = {
    "dinov2": dict(app_mu_same=0.490, app_sigma_same=0.10, app_mu_diff=0.465),
    "osnet": dict(app_mu_same=0.618, app_sigma_same=0.12, app_mu_diff=0.488),
}

LAYOUTS = {
    # 有全景鏡頭:一切走重疊路徑,地面校正+軌跡都在線上
    "有全景鏡頭": dict(master_camera="master", master_fragment_rate=0.05),
    # 無全景鏡頭:主要走轉場路徑 —— 地面校正與軌跡在那條路徑上**不生效**
    "無全景鏡頭": dict(master_camera=None),
}


def build_topo(profile, with_master):
    t = _base_topo()
    t.fusion.update({
        "unknown_path": {"enabled": False},
        "same_camera": {"enabled": True, "tau_break_s": 2.0, "max_gap_s": 15.0},
        "position": {"enabled": True}, "direction": {"enabled": False},
        "ground_plane": {"enabled": True, "sigma_m": 0.1, "area_m2": 30.0, "clip": 8.0},
        "velocity": {"enabled": True, "window_s": 1.0, "max_speed_mps": 1.5, "clip": 6.0},
        # ⚠ 系統端的外觀分布要與世界端同步 —— 不然量到的是模型失配不是外觀價值
        "appearance_profile": profile,
    })
    t._build_evidence()
    if with_master:
        t.overlapping |= {frozenset(("master", c)) for c in ("cam1", "cam2", "cam3")}
        t.cameras.setdefault("master", {})
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chefs", type=int, default=4)
    ap.add_argument("--reps", type=int, default=8)
    ap.add_argument("--hours", type=float, default=1.0)
    ap.add_argument("--out", default=str(ROOT / "results" / "m5_reid" / "appearance_value.json"))
    args = ap.parse_args()

    print("=" * 76)
    print("外觀品質 × 佈局 —— 外觀到底值多少,取決於候選集有多大")
    print("=" * 76)
    print("\n  外觀證據的理論上限(AppearanceLR,單位 nats,門檻 1.61):")
    for name in QUALITY:
        a = AppearanceLR(**AppearanceLR.MEASURED[name])
        q = QUALITY[name]
        print(f"    {name:<8} 同人 {q['app_mu_same']:.3f} / 不同人 {q['app_mu_diff']:.3f}"
              f"  → 典型同人 {a.llr(q['app_mu_same']):+.2f}"
              f" / 最大 {a.llr(0.95):+.2f}")

    print(f"\n  跑法:{args.reps} 次重複 × {args.hours} 小時 × {args.chefs} 位廚師\n")
    print(f"  {'佈局':<14}{'外觀':<10}{'轉場':>8}{'碎裂':>9}{'誤併':>9}"
          f"{'IDF1':>8}{'候選>1':>9}")
    print("  " + "-" * 68)

    out = {}
    for lay_name, lay in LAYOUTS.items():
        for prof, qual in QUALITY.items():
            brks, fms, idfs, ntr, multi, tot = [], [], [], 0, 0, 0
            for rep in range(args.reps):
                cfg = W.WorldConfig(n_chefs=args.chefs, duration_s=args.hours * 3600,
                                    seed=2000 + rep, calib_sigma_m=0.1,
                                    tau_v_s=3.0, vel_window_s=1.0, **qual, **lay)
                topo = build_topo(prof, lay["master_camera"] is not None)
                recs, _tags = run_once(cfg, topo)
                s = metrics.summarize(recs, expected_headcount=args.chefs)
                if s["p_break"] is not None:
                    brks.append(s["p_break"]); ntr += s["n_transitions"]
                if s["p_false_merge"] is not None:
                    fms.append(s["p_false_merge"])
                if s["idf1"] is not None:
                    idfs.append(s["idf1"])
            brk = st.mean(brks) if brks else None
            fm = st.mean(fms) if fms else None
            key = f"{lay_name}|{prof}"
            out[key] = dict(layout=lay_name, profile=prof, n_transitions=ntr,
                            p_break=brk, p_false_merge=fm,
                            idf1=st.mean(idfs) if idfs else None)
            print(f"  {lay_name:<14}{prof:<10}{ntr:>8}{brk:>8.1%}{fm:>9.1%}"
                  f"{(st.mean(idfs) if idfs else 0):>8.3f}{'—':>9}")

    print("\n" + "─" * 76)
    print("外觀從 DINOv2 換成 OSNet 品質,買到多少?")
    print("─" * 76)
    for lay_name in LAYOUTS:
        d = out.get(f"{lay_name}|dinov2")
        o = out.get(f"{lay_name}|osnet")
        if not (d and o):
            continue
        db = (o["p_break"] - d["p_break"]) * 100
        dm = (o["p_false_merge"] - d["p_false_merge"]) * 100
        print(f"\n  {lay_name}:")
        print(f"    碎裂 {d['p_break']:.1%} → {o['p_break']:.1%}  ({db:+.1f} pp)")
        print(f"    誤併 {d['p_false_merge']:.1%} → {o['p_false_merge']:.1%}  ({dm:+.1f} pp)")
        if abs(dm) < 1.0 and abs(db) < 1.0:
            print("    → 幾乎沒差 —— 外觀在這個佈局下沒有舞台")
        elif dm < -3:
            print("    → **顯著改善** —— 外觀在這個佈局下值得投資")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(
        {"quality": QUALITY, "results": out, "args": vars(args)},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已存 {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
