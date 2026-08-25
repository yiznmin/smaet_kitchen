"""M5 蒙地卡羅可行性包絡 —— 回答「這個架構在什麼條件下可用、超出範圍會怎麼壞」。

解析式(analyze_gate_capacity.py)只能回答「假設全部成立時的碎裂率」。
真實條件的偏差 —— σ 估錯、繞路、逗留、時鐘漂移、M4 斷軌 —— 只能靠模擬量。

三個反循環論證的措施:
  1. 世界模型(src/m5_sim/world.py)用**對數常態 + 逗留 + 繞路**產生真實轉場,
     系統用它自己以為的高斯/混合模型判斷。兩者刻意不同族、不共用參數,
     且 world.py 不 import m5_reid.evidence。
  2. 五組消融同吃一個世界:全開新 / 純外觀 / 純時空 / 完整 / oracle。
     完整系統必須明顯落在(純外觀、純時空)與 oracle 之間,否則模擬器本身有問題。
  3. 兩個世界參數從真實資料校準(EPFL 實測外觀分布、M4 實跑軌跡),報告中標明。

報告方式是**倒過來寫**:不是「在我們的假設下 IDF1=0.9」,而是
「IDF1 ≥ 0.9 需要:σ̂ 在真值的 X 倍內、廚師 ≤ N 人、逗留率 ≤ Y%」。
這個陳述可被業主的真實測試否證,而且不需要對方相信我們的先驗。

用法:
  python scripts/sim_m5_montecarlo.py                    # 消融 + 主要 sweep
  python scripts/sim_m5_montecarlo.py --reps 50          # 提高重複次數
  python scripts/sim_m5_montecarlo.py --quick            # 快速煙霧測試
"""
import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from m5_reid import metrics                                        # noqa: E402
from m5_reid.evidence import AppearanceLR, HistogramParzenTransit  # noqa: E402
from m5_reid.identity_st import SpatioTemporalIdentityManager      # noqa: E402
from m5_reid.spatiotemporal import CameraTopology                  # noqa: E402
from m5_sim import world as W                                      # noqa: E402

TOPO_PATH = ROOT / "configs" / "camera_topology.yaml"

# 世界的真實轉場(對數常態)。系統 config 寫的是 μ=4.0/σ=1.5,而這組參數的
# 實際 mean≈4.25、std≈1.53 —— 刻意讓「正確」不等於「config 寫的」。
TRUE_MEDIAN, TRUE_LOG_SIGMA = 4.0, 0.35


def _base_topo():
    return CameraTopology.from_yaml(TOPO_PATH)


def _with_fusion(**over):
    t = _base_topo()
    t.fusion.update(over)
    t._build_evidence()
    return t


def run_once(wcfg, topo, ablation="full", ttl_s=600.0):
    """跑一次模擬,回傳 records = [(gt_id, pred_id, matched, is_transition)]。"""
    events = W.generate(wcfg, topo.links, topo.all_cameras())

    if ablation == "all_new":                      # 基線:從不綁定
        recs, seen, tags = [], set(), []
        for kind, gt, cam, t, emb, tag in events:
            if kind == "enter":
                recs.append((gt, f"new{len(recs)}", False, gt in seen))
                tags.append(tag)
                seen.add(gt)
        return recs, tags

    m = SpatioTemporalIdentityManager(topo, fps=1.0, recently_disappeared_ttl=ttl_s)
    recs, tags, seen, tid = [], [], set(), {}
    for kind, gt, cam, t, emb, tag in events:
        f = int(t)
        if kind == "enter":
            tid[(gt, cam)] = tid.get((gt, cam), 0) + 1
            key = tid[(gt, cam)] * 1000 + gt
            r = m.on_new_track(key, camera_id=cam, frame_id=f, t_sec=t, embedding=emb)
            recs.append((gt, r.chef_id, r.matched, gt in seen))
            tags.append(tag)
            seen.add(gt)
        else:
            key = tid.get((gt, cam), 1) * 1000 + gt
            m.on_track_lost(key, camera_id=cam, frame_id=f, t_sec=t)
            m.on_track_removed(key, camera_id=cam, frame_id=f + 1, t_sec=t + 1.0)
    return recs, tags


def decompose_failures(recs, tags):
    """把碎裂拆成成因 —— 只知道「25% 碎裂」不夠,要知道碎在哪才知道修什麼。

    絕不把碎裂與誤併相加成單一「錯誤率」:兩者對 M6 的後果完全不同。
    """
    from collections import defaultdict
    last_pred, stat = {}, defaultdict(lambda: [0, 0, 0])   # tag -> [總數, 碎裂, 誤併]
    for (gt, pred, matched, is_tr), tag in zip(recs, tags):
        if is_tr:
            s = stat[tag]
            s[0] += 1
            if not matched:
                s[1] += 1
            elif last_pred.get(gt) != pred:
                s[2] += 1
        last_pred[gt] = pred
    return {k: dict(n=v[0], breaks=v[1], merges=v[2]) for k, v in stat.items()}


def _oracle_topo(wcfg, base):
    """oracle:系統知道真實的轉場分布(用世界抽出的樣本做 Parzen 估計)。

    這同時示範了 scripts/calibrate_topology.py 未來要走的路 —— 從實測 Δt 估分布,
    而不是猜 μ/σ。oracle 與完整系統的差距 = 「參數失準的代價」。
    """
    rng = np.random.RandomState(12345)
    samples = [W.sample_transit(wcfg, rng) for _ in range(400)]
    t = copy.deepcopy(base)
    for k in list(t.transits):
        t.transits[k] = HistogramParzenTransit(samples, fallback=base.transits[k])
    return t


def ablation_suite(wcfg, reps, budget_break, budget_fm):
    """五組消融同吃一個世界。完整系統必須落在(純外觀/純時空)與 oracle 之間。"""
    base = _base_topo()
    st_only = _with_fusion()
    st_only.app_lr = AppearanceLR.uninformative()
    variants = {
        "(a) 全開新(基線)": ("all_new", base),
        "(b) 純外觀": ("app_only", base),
        "(c) 純時空": ("full", st_only),
        "(d) 完整系統": ("full", base),
        "(e) oracle(知道真分布)": ("full", None),      # None → 每 rep 重建
    }
    print("=" * 78)
    print("§1 消融對照 —— 五組系統吃同一個世界")
    print("=" * 78)
    print(f"    {wcfg.n_chefs} 位廚師 / {wcfg.duration_s/3600:.1f} 小時 / "
          f"逗留 {wcfg.p_loiter:.0%} / 繞路 {wcfg.p_detour:.0%} / "
          f"M4 斷軌 {wcfg.m4_fragment_rate:.0%} / {reps} 次重複")
    print()
    print(f"  {'系統':<24}{'IDF1':>8}{'碎裂率':>10}{'誤併率':>10}{'ID切換':>9}{'碎裂倍數':>10}")
    print("  " + "-" * 74)
    results, decomp = {}, {}
    for name, (mode, topo) in variants.items():
        accum = []
        for r in range(reps):
            w = copy.deepcopy(wcfg)
            w.rng = np.random.RandomState(1000 + r)
            tp = _oracle_topo(w, base) if topo is None else topo
            if mode == "app_only":
                tp = copy.deepcopy(base)
                for k in list(tp.transits):      # 時間完全不提供資訊
                    tp.transits[k] = _FlatTransit()
            rc, tg = run_once(w, tp, "all_new" if mode == "all_new" else "full")
            accum.append(metrics.summarize(rc))
            if name == "(d) 完整系統":
                for k, v in decompose_failures(rc, tg).items():
                    d = decomp.setdefault(k, [0, 0, 0])
                    d[0] += v["n"]; d[1] += v["breaks"]; d[2] += v["merges"]
        f1 = np.mean([a["idf1"] for a in accum if a["idf1"] is not None])
        pb = np.mean([a["p_break"] for a in accum if a["p_break"] is not None])
        fm = np.mean([a["p_false_merge"] for a in accum if a["p_false_merge"] is not None])
        sw = np.mean([a["id_switches"] for a in accum])
        fr = np.mean([a["fragmentation"]["mean"] for a in accum])
        results[name] = dict(idf1=f1, p_break=pb, p_false_merge=fm, switches=sw, frag=fr)
        print(f"  {name:<24}{f1:>8.3f}{pb*100:>9.1f}%{fm*100:>9.1f}%{sw:>9.1f}{fr:>10.2f}")
    print()
    full, app, st, orc = (results["(d) 完整系統"], results["(b) 純外觀"],
                          results["(c) 純時空"], results["(e) oracle(知道真分布)"])
    sane = full["idf1"] >= max(app["idf1"], st["idf1"]) - 0.02 and full["idf1"] <= orc["idf1"] + 0.02
    print(f"  健全性檢查:完整系統({full['idf1']:.3f})應介於"
          f" max(純外觀 {app['idf1']:.3f}, 純時空 {st['idf1']:.3f}) 與 oracle {orc['idf1']:.3f} 之間"
          f" → {'✅' if sane else '❌ 模擬器本身可能有問題'}")
    print(f"  M5 模組的價值 = 完整({full['idf1']:.3f}) − 全開新"
          f"({results['(a) 全開新(基線)']['idf1']:.3f}) = "
          f"{full['idf1']-results['(a) 全開新(基線)']['idf1']:+.3f} IDF1")
    print(f"  參數失準的代價 = oracle({orc['idf1']:.3f}) − 完整({full['idf1']:.3f}) = "
          f"{orc['idf1']-full['idf1']:+.3f} IDF1")
    print(f"  誰在扛:純時空 {st['idf1']:.3f} vs 純外觀 {app['idf1']:.3f}"
          f" → {'時空' if st['idf1'] > app['idf1'] else '外觀'}為主")

    print()
    print("  失效分解(完整系統)—— 碎裂到底碎在哪:")
    print(f"    {'成因':<10}{'轉場數':>8}{'碎裂':>8}{'碎裂率':>10}{'誤併':>8}{'誤併率':>10}")
    print("    " + "-" * 56)
    order = ["normal", "loiter", "detour", "fragment"]
    label = {"normal": "正常轉場", "loiter": "中途逗留", "detour": "繞路",
             "fragment": "M4 斷軌"}
    for k in order:
        if k not in decomp:
            continue
        n, b, mg = decomp[k]
        print(f"    {label[k]:<10}{n:>8}{b:>8}{b/max(n,1)*100:>9.1f}%"
              f"{mg:>8}{mg/max(n,1)*100:>9.1f}%")
    tot = sum(v[0] for v in decomp.values())
    tb = sum(v[1] for v in decomp.values())
    if tb:
        print()
        print("    碎裂的組成:", "、".join(
            f"{label[k]} {decomp[k][1]/tb*100:.0f}%" for k in order
            if k in decomp and decomp[k][1]))
    return results, sane, decomp


class _FlatTransit:
    """時間完全不提供資訊(純外觀消融用):任何物理可能的 Δt 都同樣可信。"""
    hard_min_ratio = 0.0

    def hard_min(self):
        return 0.0

    def logpdf(self, dt):
        from m5_reid.evidence import NEG_INF
        return -np.log(600.0) if dt > 0 else NEG_INF

    def describe(self):
        return "Flat(無時間資訊)"


def sweep(wcfg, reps, budget_break, budget_fm, quick=False):
    """掃描失配維度,找斷點。輸出「達標需要什麼條件」而非單一數字。"""
    base = _base_topo()
    print()
    print("=" * 78)
    print("§2 失配掃描 —— 系統模型與真實世界差多少才會壞")
    print("=" * 78)

    def run(**over):
        w = copy.deepcopy(wcfg)
        for k, v in over.items():
            setattr(w, k, v)
        pb, fm, f1 = [], [], []
        for r in range(reps):
            w2 = copy.deepcopy(w)
            w2.rng = np.random.RandomState(2000 + r)
            s = metrics.summarize(run_once(w2, base)[0])
            if s["p_break"] is not None:
                pb.append(s["p_break"]); fm.append(s["p_false_merge"]); f1.append(s["idf1"])
        return (np.mean(f1) if f1 else 0, np.mean(pb) if pb else 1, np.mean(fm) if fm else 0)

    envelope = {}
    sweeps = [
        ("廚師人數", "n_chefs", [2, 3, 4, 5, 6, 8] if not quick else [2, 5]),
        ("逗留比例", "p_loiter", [0.0, 0.15, 0.30, 0.50] if not quick else [0.0, 0.5]),
        ("繞路比例", "p_detour", [0.0, 0.05, 0.15, 0.30] if not quick else [0.0, 0.3]),
        ("轉場中位數失配(×)", "transit_median_s",
         [2.0, 3.0, 4.0, 6.0, 8.0] if not quick else [2.0, 8.0]),
        ("真實變異 log σ", "transit_log_sigma",
         [0.15, 0.35, 0.60, 0.90] if not quick else [0.15, 0.9]),
        ("M4 斷軌率", "m4_fragment_rate", [0.0, 0.05, 0.15, 0.30] if not quick else [0.0, 0.3]),
        ("同制服退化 γ", "gamma_uniform", [0.0, 0.5, 0.9, 1.0] if not quick else [0.0, 1.0]),
    ]
    for label, attr, values in sweeps:
        print(f"\n  {label}")
        print(f"    {'值':>10}{'IDF1':>9}{'碎裂率':>10}{'誤併率':>10}  達標")
        ok_values = []
        for v in values:
            f1, pb, fm = run(**{attr: v})
            ok = pb <= budget_break and fm <= budget_fm
            if ok:
                ok_values.append(v)
            print(f"    {v:>10}{f1:>9.3f}{pb*100:>9.1f}%{fm*100:>9.1f}%  "
                  f"{'✅' if ok else '✗'}")
        envelope[label] = ok_values
    return envelope


def compare_fixes(wcfg, reps, budget_break, budget_fm):
    """三個結構缺口修法的逐項效果 —— 網格與判準見 docs/M5_模擬預先登記_20260825.md。

    ⚠ 本函式只執行預先登記的網格,不做登記外的調參。
    """
    print()
    print("=" * 78)
    print("§4 結構缺口修法(依 docs/M5_模擬預先登記_20260825.md 的登記網格)")
    print("=" * 78)

    def build(**over):
        t = _base_topo()
        for k, v in over.items():
            if isinstance(v, dict):
                t.fusion[k] = {**(t.fusion.get(k) or {}), **v}
            else:
                t.fusion[k] = v
        t._build_evidence()
        return t

    OFF_F1 = {"unknown_path": {"enabled": False}}
    OFF_F3 = {"same_camera": {"enabled": False}}
    variants = [
        ("基準(上一輪,三個都關)", build(**OFF_F1, **OFF_F3)),
        ("+F1 繞路 logprior=-2", build(**OFF_F3)),
        ("+F3 同鏡頭重關聯", build(**OFF_F1)),
        ("+F1+F3", build()),
    ]
    for lp in [-3.0, -1.0]:
        variants.append((f"+F1(logprior={lp:+.0f})+F3",
                         build(unknown_path={"logprior": lp})))
    for tau in [45.0, 90.0]:
        variants.append((f"+F1+F3, τ_逗留={tau:.0f}(登記值)",
                         build(tau_loiter_s=tau)))
    for gap in [5.0]:
        variants.append((f"+F1+F3, 同鏡頭上限={gap:.0f}s",
                         build(same_camera={"max_gap_s": gap})))

    print(f"  {'設定':<28}{'碎裂率':>9}{'誤併率':>9}{'正常轉場碎裂':>13}{'等級':>6}")
    print("  " + "-" * 70)
    rows = []
    for name, tp in variants:
        pb, fm, normal_b = [], [], []
        for r in range(reps):
            w = copy.deepcopy(wcfg)
            w.rng = np.random.RandomState(3000 + r)
            rc, tg = run_once(w, tp)
            s = metrics.summarize(rc)
            if s["p_break"] is None:
                continue
            pb.append(s["p_break"]); fm.append(s["p_false_merge"])
            d = decompose_failures(rc, tg).get("normal", {"n": 0, "breaks": 0})
            normal_b.append(d["breaks"] / max(d["n"], 1))
        b, m, nb = np.mean(pb), np.mean(fm), np.mean(normal_b)
        grade = ("A" if b <= budget_break and m <= budget_fm else
                 "B" if b <= 0.10 and m <= 0.03 else "C")
        rows.append((name, b, m, nb, grade))
        print(f"  {name:<28}{b*100:>8.1f}%{m*100:>8.1f}%{nb*100:>12.1f}%{grade:>6}")
    print()
    print("  等級定義(預先登記 §4):A 碎裂≤5% 且誤併≤1% / B 碎裂≤10% 且誤併≤3% / C 其餘")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--hours", type=float, default=2.0)
    ap.add_argument("--chefs", type=int, default=4)
    ap.add_argument("--budget-break", type=float, default=0.05)
    ap.add_argument("--budget-false-merge", type=float, default=0.01)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--fixes", action="store_true",
                    help="只跑結構缺口修法的對照(預先登記 §3 網格)")
    ap.add_argument("--out", default=str(ROOT / "results" / "m5_reid" / "sim_envelope.json"))
    args = ap.parse_args()
    if args.quick:
        args.reps, args.hours = 5, 0.5

    defects = W.m4_defect_rates()
    wcfg = W.WorldConfig(n_chefs=args.chefs, duration_s=args.hours * 3600,
                         transit_median_s=TRUE_MEDIAN, transit_log_sigma=TRUE_LOG_SIGMA,
                         **{k: v for k, v in defects.items() if k != "source"})

    print("=" * 78)
    print("M5 蒙地卡羅可行性包絡")
    print("=" * 78)
    print(f"  世界轉場分布 : 對數常態(中位數 {TRUE_MEDIAN}s, logσ {TRUE_LOG_SIGMA})"
          f" + 逗留 + 繞路   ← 系統以為是高斯/混合")
    print(f"  外觀分布     : EPFL 449 crops 實測(same 0.490 / diff 0.465)  ← 非臆測")
    print(f"  M4 缺陷率    : {defects['source']}")
    print(f"  預算         : 碎裂 ≤ {args.budget_break:.0%} / 誤併 ≤ {args.budget_false_merge:.0%}")
    print()

    if args.fixes:
        compare_fixes(wcfg, args.reps, args.budget_break, args.budget_false_merge)
        sys.exit(0)
    results, sane, decomp = ablation_suite(wcfg, args.reps, args.budget_break, args.budget_false_merge)
    envelope = sweep(wcfg, args.reps, args.budget_break, args.budget_false_merge, args.quick)

    print()
    print("=" * 78)
    print("§3 作業包絡(可被業主的真實測試否證的陳述)")
    print("=" * 78)
    print(f"  在碎裂 ≤ {args.budget_break:.0%}、誤併 ≤ {args.budget_false_merge:.0%} 的前提下,")
    print("  本系統的適用範圍是:")
    for label, vals in envelope.items():
        if not vals:
            print(f"    · {label:<22} ✗ 沒有任何測試值達標")
        else:
            print(f"    · {label:<22} {vals}")
    print()
    print("  ⚠ 超出上述範圍的表現不在保證範圍。這份包絡由模擬推得,")
    print("     必須由業主的真實測試驗證或否證。")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "world": {"transit": "lognormal", "median_s": TRUE_MEDIAN,
                  "log_sigma": TRUE_LOG_SIGMA, "m4_defects": defects["source"]},
        "budgets": {"p_break": args.budget_break, "p_false_merge": args.budget_false_merge},
        "reps": args.reps, "hours": args.hours,
        "ablation": {k: {kk: float(vv) for kk, vv in v.items()} for k, v in results.items()},
        "envelope": {k: [float(x) for x in v] for k, v in envelope.items()},
        "sanity_check_passed": bool(sane),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  已存 {out}")
    sys.exit(0 if sane else 1)


if __name__ == "__main__":
    main()
