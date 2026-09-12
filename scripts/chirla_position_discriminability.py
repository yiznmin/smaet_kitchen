"""地面位置到底能不能區分身份 —— F4 失敗的兩個可能中,先查決定性的那個。

## 為什麼這支最該先跑

2026-09-12 的修復網格:F4 沒過自己的主判準(重疊路徑正確率要 16.49% → >50%,
實際掉到 **6.07%**),誤併只從 80.80% 降到 77.77%。

失敗有兩個可能,而那一輪的設計**分不開**:

  (a) 單應性不夠準 —— 六對裡三對的 σ 大到證據沒有鑑別力
      (camera_2+5 的 σ=74.7px,同人 LLR 只有 +0.73,**同一個人也推不過門**)
  (b) 位置本身不足以區分身份 —— 多人同時在場時,兩個人站在相近的地面位置
      完全正常。幾何一致性是**必要條件不是充分條件**。

⚠ **(b) 有決定性:若 (b) 成立,把單應性修準也沒用**,F4 這條路就該放棄,
  力氣轉到誤偵過濾(誤偵綁定率 32.55%,9/12 的分析指出那才是主要病灶)。
  所以先查 (b),而且它只要讀 annotations,幾分鐘。

## 量什麼

模擬 M5 重疊路徑實際在做的事:候選此刻在鏡頭 A 被看著,把他的腳點經 H_AB
投到 B,跟 B 上這條新 track 的腳點比。

對每一個**真實對應**(身份 i 同時出現在 A 與 B):

    d_self  = |proj_A(i) − foot_B(i)|              ← 應該 ≈ σ
    d_other = |proj_A(j) − foot_B(i)|   j ≠ i      ← 別人的投影離 i 多遠

  · 「可混淆的競爭者」= d_other < 3σ 的 j 有幾個
  · **決策餘裕** = LLR(d_self) − max_j LLR(d_other) —— 這直接決定 M5 會不會選對

若餘裕中位數 < 門檻 1.609,表示**即使單應性完美,位置也選不出對的那個**。

⚠ σ 與共視區直接讀 `configs/camera_topology.chirla_f4.yaml`(9/12 實際跑的那份),
  不重新擬合 —— 這樣量到的就是 F4 當時面對的局面。

用法:
    python scripts/chirla_position_discriminability.py --root "D:/.../CHIRLA"
    python scripts/chirla_position_discriminability.py --root "..." --split eval
"""
import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from m5_reid.evidence import CrossViewLR                      # noqa: E402

# 沿用 docs/CHIRLA_M4M5驗證_預先登記_20260903.md §3 定死的切分
DERIVATION = ("seq_000", "seq_001", "seq_002")
EVALUATION = ("seq_004", "seq_006", "seq_007", "seq_020",
              "seq_024", "seq_025", "seq_026")
GOOD_W, GOOD_H = 50, 120        # 沿用 chirla_overlap_stats.py 的「拍得清楚」門檻


def phys(name):
    return "_".join(name.split("_")[:2])


def big(b):
    return ((float(b[2]) - float(b[0])) >= GOOD_W
            and (float(b[3]) - float(b[1])) >= GOOD_H)


def build_models(topology):
    """讀 9/12 實際跑的那份拓撲,**不重新擬合**。兩個方向都建。"""
    cfg = yaml.safe_load(Path(topology).read_text(encoding="utf-8"))
    fusion = cfg["camera_topology"]["fusion"]
    cv = fusion.get("cross_view") or {}
    pairs = cv.get("pairs") or {}
    if not pairs:
        raise SystemExit(f"{topology} 裡沒有 cross_view.pairs"
                         " —— 先跑 chirla_build_crossview.py")
    clip = float(cv.get("clip", 8.0))
    models = {}
    for key, m in pairs.items():
        a, b = key.split("|")
        H = np.asarray(m["H"], dtype=float).reshape(3, 3)
        sig, area = float(m["sigma_px"]), float(m["area_px2"])
        models[(a, b)] = CrossViewLR(H, sig, area, clip=clip)
        try:
            models[(b, a)] = CrossViewLR(np.linalg.inv(H), sig, area, clip=clip)
        except np.linalg.LinAlgError:
            pass                       # H 退化 → 只留單向
    thr = math.log(float(fusion["cost_false_merge_over_break"]))
    return models, thr


def scan(aroot, seqs, models):
    """回傳 {(a,b): 統計}。只看「A 有 ≥2 人」的幀 —— 沒有別人就不構成考驗。"""
    per_pair = defaultdict(lambda: dict(n=0, comp=[], margin=[],
                                        self_d=[], best_other=[]))
    for seq in sorted(p for p in aroot.iterdir() if p.is_dir()):
        if seq.name not in seqs:
            continue
        fr_cam = defaultdict(lambda: defaultdict(dict))   # frame -> cam -> {id: bbox}
        for f in sorted(seq.glob("*.json")):
            cam = phys(f.stem)
            for fr, dets in json.loads(f.read_text(encoding="utf-8")).items():
                for o in dets:
                    if big(o["BboxP"]):
                        fr_cam[int(fr)][cam][abs(int(o["id"]))] = o["BboxP"]
        for _fr, by_cam in fr_cam.items():
            for (a, b), mdl in models.items():
                if a not in by_cam or b not in by_cam or len(by_cam[a]) < 2:
                    continue
                proj = {}
                for i, bb in by_cam[a].items():
                    q = mdl.project(mdl.foot(bb))
                    if q is not None:
                        proj[i] = q
                for i, bb_b in by_cam[b].items():
                    if i not in proj:
                        continue
                    fb = mdl.foot(bb_b)
                    d_self = math.hypot(proj[i][0] - fb[0], proj[i][1] - fb[1])
                    others = [math.hypot(proj[j][0] - fb[0], proj[j][1] - fb[1])
                              for j in proj if j != i]
                    if not others:
                        continue
                    d_best = min(others)
                    s = per_pair[(a, b)]
                    s["n"] += 1
                    s["comp"].append(sum(1 for d in others if d < 3 * mdl.sigma))
                    s["self_d"].append(d_self)
                    s["best_other"].append(d_best)
                    # ⚠ M5 實際比的是 LLR 不是距離,所以餘裕要用 LLR 算
                    s["margin"].append(mdl.llr_at(d_self) - mdl.llr_at(d_best))
    return per_pair


def report(sp, per_pair, models, thr):
    label = "推導集" if sp == "deriv" else "評估集"
    print("=" * 88)
    print(f"地面位置的鑑別力 —— {sp}({label})")
    print("=" * 88)
    print(f"\n  {'鏡頭對':<24}{'樣本':>8}{'σ':>7}{'可混淆者':>10}"
          f"{'自身殘差':>10}{'最近他人':>10}{'決策餘裕':>11}{'餘裕過門檻':>11}")
    print("  " + "-" * 86)
    agg = dict(n=0, comp=[], margin=[])
    for (a, b), s in sorted(per_pair.items(), key=lambda kv: -kv[1]["n"]):
        if s["n"] < 50:
            continue
        mg = np.array(s["margin"])
        agg["n"] += s["n"]
        agg["comp"] += s["comp"]
        agg["margin"] += s["margin"]
        print(f"  {a + '→' + b:<24}{s['n']:>8}{models[(a, b)].sigma:>7.1f}"
              f"{np.median(s['comp']):>10.1f}{np.median(s['self_d']):>9.0f}px"
              f"{np.median(s['best_other']):>9.0f}px{np.median(mg):>+11.2f}"
              f"{(mg >= thr).mean():>11.1%}")
    if not agg["n"]:
        print("  (沒有足夠樣本)")
        return {"_all": {"n": 0}}
    mg = np.array(agg["margin"])
    print("  " + "-" * 86)
    print(f"  {'合計':<24}{agg['n']:>8}{'':>7}{np.median(agg['comp']):>10.1f}"
          f"{'':>10}{'':>10}{np.median(mg):>+11.2f}{(mg >= thr).mean():>11.1%}")
    print(f"\n  判讀(門檻 {thr:.3f} nats):")
    print(f"    · 真實對應中,**{(mg >= thr).mean():.1%}** 的決策餘裕足以選對")
    print(f"    · 可混淆競爭者(落在 3σ 內的別人)中位 {np.median(agg['comp']):.1f} 人")
    print("    → " + ("位置**不足以**區分身份 —— 把單應性修準也沒用,(b) 成立"
                      if np.median(mg) < thr else
                      "位置**足以**區分 —— F4 失敗應歸因於單應性品質 (a)"))
    out = {f"{a}|{b}": dict(n=s["n"],
                            comp_median=float(np.median(s["comp"])),
                            margin_median=float(np.median(s["margin"])),
                            margin_pass_rate=float((np.array(s["margin"]) >= thr).mean()))
           for (a, b), s in per_pair.items() if s["n"] >= 50}
    out["_all"] = dict(n=agg["n"], comp_median=float(np.median(agg["comp"])),
                       margin_median=float(np.median(mg)),
                       margin_pass_rate=float((mg >= thr).mean()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--topology",
                    default=str(ROOT / "configs" / "camera_topology.chirla_f4.yaml"))
    ap.add_argument("--split", choices=["deriv", "eval", "both"], default="both")
    ap.add_argument("--out",
                    default=str(ROOT / "results" / "m5_reid" / "position_discrim.json"))
    args = ap.parse_args()

    models, thr = build_models(args.topology)
    aroot = Path(args.root) / "annotations"
    if not aroot.exists():
        raise SystemExit(f"找不到 {aroot}")

    splits = {"deriv": DERIVATION, "eval": EVALUATION}
    out = {}
    for sp in (["deriv", "eval"] if args.split == "both" else [args.split]):
        out[sp] = report(sp, scan(aroot, splits[sp], models), models, thr)
        print()

    print("⚠ 這是**診斷**不是擬合。評估集的數字只可用於解釋 F4 為何失敗,")
    print("  **不得**從中挑任何參數 —— 那會污染 9/3 §3 定死的切分。")
    p = Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"results": out, "threshold": thr, "args": vars(args)},
                            ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已存 {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
