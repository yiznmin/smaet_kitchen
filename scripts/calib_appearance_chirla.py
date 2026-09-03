"""用 CHIRLA 量出 `AppearanceLR.MEASURED` 需要的四個數字。

**為什麼不是照交接單用 `reid_eval_epfl.cross_view_consistency()`**:
那條路需要 EPFL 同步幀(`data/m5_reid_epfl/`)與 RF-DETR 偵測權重
(`model_result/nano/checkpoint_best_regular.pth`),兩者都被 `.gitignore` 排除,
不在這台機器上。而且就算有,EPFL 有 **session confound** —— 一個場次一人,
同一人的 crop 全部同衣服同光照 → same 的平均偏高、可分性偏樂觀
(見 `src/m5_reid/evidence.py` 的 AppearanceLR docstring)。

CHIRLA 反而更貼近 M5 的部署情境:7 台**非重疊**鏡頭、橫跨 7 個月、同一人會換衣服。
所以這裡直接從已匯出的 embedding 量跨相機的 cosine 分布。

量法(刻意與 M5 的用法對齊,不是隨便取所有配對):
  · 只用 `id >= 0` 的樣本 —— 負號是 distractor(unknown identity),
    把它們當成一個個「身份」會污染 diff 分布。
  · **只取跨實體相機的配對**。M5 問的是「cam2 這個人是不是剛從 cam1 走掉的那位」,
    同一台相機內的相似度再高也用不到,算進來會讓 same 的平均虛高。
    實體相機 = 目錄名的 `camera_N` 前綴(後面那串是錄影時間戳,不是不同相機)。
  · 配對方式沿用官方 `--per-subset`:query 的 test_k 對 gallery 的 train_k。

⚠ 這是**換了校準來源**(EPFL → CHIRLA),不是把 EPFL 的數字重算一次。
  報告裡不可與 `MEASURED["dinov2"]` / `MEASURED["osnet"]` 那兩組直接比大小,
  它們量的是不同資料集、不同 confound 的分布。

用法:
    python scripts/calib_appearance_chirla.py --method chirla_armS --scenario reid_long_term
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


def load_h5(p):
    import h5py
    with h5py.File(p, "r") as f:
        return f["embeddings"][:], f["ids"][:], f["paths"][:].astype(str)


def phys_cam(path):
    """從路徑取實體相機。目錄名帶錄影時間戳(camera_3_2023-06-02-11_14_26),
    同一台相機在 10 個序列裡有 10 個名字,不去掉時間戳會把 7 台數成 58 台。"""
    for part in path.replace("\\", "/").split("/"):
        if part.startswith("camera_"):
            return "_".join(part.split("_")[:2])
    return "?"


def subset_of(path):
    parts = path.replace("\\", "/").split("/")
    for i, p in enumerate(parts):
        if p in ("train", "test") and i + 1 < len(parts) \
                and parts[i + 1].startswith(("train_", "test_")):
            return parts[i + 1]
    return "unknown"


def collect(gallery_h5, query_h5):
    """回傳 (same_cos, diff_cos):跨實體相機、非 distractor 的配對 cosine。"""
    gf, gid, gpath = load_h5(gallery_h5)
    qf, qid, qpath = load_h5(query_h5)
    gcam = np.array([phys_cam(p) for p in gpath])
    qcam = np.array([phys_cam(p) for p in qpath])
    gsub = np.array([subset_of(p) for p in gpath])
    qsub = np.array([subset_of(p) for p in qpath])

    same, diff = [], []
    per_subset = {}
    for qs in sorted(set(qsub)):
        gs = qs.replace("test_", "train_")
        qi = np.where((qsub == qs) & (qid >= 0))[0]
        gi = np.where((gsub == gs) & (gid >= 0))[0]
        if not len(qi) or not len(gi):
            continue
        sims = qf[qi] @ gf[gi].T                      # 特徵已 L2-normalized
        cross = qcam[qi][:, None] != gcam[gi][None, :]   # 只留跨實體相機
        ident = qid[qi][:, None] == gid[gi][None, :]
        s = sims[cross & ident]
        d = sims[cross & ~ident]
        per_subset[qs] = dict(n_same=int(s.size), n_diff=int(d.size),
                              mu_same=float(s.mean()) if s.size else None,
                              mu_diff=float(d.mean()) if d.size else None)
        same.append(s)
        diff.append(d)
    return np.concatenate(same) if same else np.array([]), \
        np.concatenate(diff) if diff else np.array([]), per_subset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb-root", default="embeddings")
    ap.add_argument("--method", default="chirla_armS")
    ap.add_argument("--model", default="resnet50_bnneck")
    ap.add_argument("--scenario", default="reid_long_term")
    ap.add_argument("--out", default=str(ROOT / "results" / "m5_reid" /
                                        "appearance_calib_chirla.json"))
    args = ap.parse_args()

    d = Path(args.emb_root) / args.method / args.model
    g = d / f"{args.scenario}_gallery_embeddings.h5"
    q = d / f"{args.scenario}_query_embeddings.h5"
    for f in (g, q):
        if not f.exists():
            raise SystemExit(f"找不到 {f} —— 先跑 export_chirla_embeddings.py")

    same, diff, per_subset = collect(g, q)
    if not same.size or not diff.size:
        # 2026-09-03 實測發現:CHIRLA 四個 scenario 只有兩個是真的跨相機。
        # long_term 與 reappearance 的官方 subset 配對(train_k vs test_k)
        # **兩邊是同一台實體相機** —— 它們量的是「同一台相機、跨月份換衣服」與
        # 「同一台相機、離開後再出現」,不是跨鏡頭。所以在那兩個 scenario 上
        # 校準跨相機分布結構上不可能有樣本,不是程式壞了。
        cams = {k: v for k, v in per_subset.items()}
        raise SystemExit(
            f"{args.scenario} 取不到跨實體相機的配對。\n"
            "  這個 scenario 的官方 subset 配對兩邊是同一台相機"
            "(long_term / reappearance 都是這樣),量的不是跨鏡頭。\n"
            "  M5 需要的跨相機分布請改用 --scenario reid_multi_camera "
            "或 reid_multi_camera_long_term。\n"
            f"  逐 subset 樣本數:{cams}")

    raw = dict(mu_same=round(float(same.mean()), 4),
               sigma_same_raw=round(float(same.std()), 4),
               mu_diff=round(float(diff.mean()), 4),
               sigma_diff_raw=round(float(diff.std()), 4))

    # ⚠ 用**合併 σ**(兩個分布共用同一個 σ),不是各自的 σ。三個理由:
    #  1. σ_same ≠ σ_diff 時高斯 LLR 是**二次式**,在遠離平均處會爆掉 ——
    #     實測 armR0 的 σ 只有 0.037 而 μ≈0.87,`max_abs_llr(0,1)` 掃到 cos=0
    #     時算出 27.58 nats(判定門檻才 1.61,時間證據峰值才 ~5 nats)。
    #     那是高斯尾部的外插,不是真的證據,`evidence.py` 的 docstring 早警告過。
    #     更糟的是 σ_same > σ_diff 時 `llr(mu_same)` 會是**負的** ——
    #     「剛好等於同人平均的 cosine 反而變成不同人的證據」,自相矛盾。
    #  2. 世界端 `WorldConfig` 只吃一個 `app_sigma_same`,系統端若用兩個 σ,
    #     量到的會是模型失配而不是外觀價值(sweep_appearance_value.py 的註解
    #     明講「系統端的外觀分布要與世界端同步」)。
    #  3. 現有的 dinov2(0.10/0.10)與 osnet(0.12/0.12)本來就是對稱的,
    #     用合併 σ 才能與它們並列。實測驗證:合併 σ 下
    #     `llr(mu_same) = d'^2 / 2`,代回 EPFL 的數字剛好還原文件記載的
    #     dinov2 +0.03 / osnet +0.59 —— 公式是對的。
    sigma = float(np.sqrt((same.var() + diff.var()) / 2))
    stats = dict(mu_same=raw["mu_same"], sigma_same=round(sigma, 4),
                 mu_diff=raw["mu_diff"], sigma_diff=round(sigma, 4))
    delta = stats["mu_same"] - stats["mu_diff"]
    d_prime = delta / sigma

    print("=" * 74)
    print(f"AppearanceLR 校準 · {args.method} · {args.scenario} · CHIRLA 跨實體相機")
    print("=" * 74)
    print(f"  同人配對 {same.size:,} 組   不同人配對 {diff.size:,} 組")
    print(f"  同人   cos {raw['mu_same']:.4f} ± {raw['sigma_same_raw']:.4f}")
    print(f"  不同人 cos {raw['mu_diff']:.4f} ± {raw['sigma_diff_raw']:.4f}")
    print(f"  可分距離 {delta:+.4f}   合併 σ {sigma:.4f}   **d' = {d_prime:.3f}**")

    from m5_reid.evidence import AppearanceLR
    lr = AppearanceLR(**stats)
    print(f"\n  {lr.describe()}")
    print(f"  典型同人加分 {lr.llr(stats['mu_same']):+.4f} nats(= d'^2/2)")
    print("  參照:判定門檻 1.61、時間證據峰值約 5 nats;")
    print("        EPFL 實測 dinov2 d'=0.25(+0.03 nats)、osnet d'=1.08(+0.59 nats)")
    print("        ⚠ 那兩組量自 EPFL(一場次一人,同衣服同光照,偏樂觀),")
    print("          與 CHIRLA 跨相機跨月份不是同一個難度,**不可直接比大小**。")

    print(f"\n  逐 subset(檢查有沒有某一格主導整體平均):")
    print(f"    {'subset':<10}{'同人組數':>10}{'不同人組數':>12}{'mu_same':>10}{'mu_diff':>10}")
    for k, v in sorted(per_subset.items()):
        ms = f"{v['mu_same']:.4f}" if v["mu_same"] is not None else "-"
        md = f"{v['mu_diff']:.4f}" if v["mu_diff"] is not None else "-"
        print(f"    {k:<10}{v['n_same']:>10,}{v['n_diff']:>12,}{ms:>10}{md:>10}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    blob = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {}
    blob[f"{args.method}:{args.scenario}"] = dict(
        **stats, **raw, delta=round(delta, 4), d_prime=round(d_prime, 4),
        n_same=int(same.size), n_diff=int(diff.size),
        source="CHIRLA 跨實體相機 gallery×query(closed-set,排除 distractor)",
        note="sigma_same/sigma_diff 為合併 σ;各自的原始 σ 見 sigma_*_raw",
        per_subset=per_subset)
    out.write_text(json.dumps(blob, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  → {out}")
    print("\n貼進 src/m5_reid/evidence.py 的 AppearanceLR.MEASURED:")
    print(f'    "{args.method}": dict(mu_same={stats["mu_same"]}, '
          f'sigma_same={stats["sigma_same"]}, mu_diff={stats["mu_diff"]}, '
          f'sigma_diff={stats["sigma_diff"]}),')
    return 0


if __name__ == "__main__":
    sys.exit(main())
