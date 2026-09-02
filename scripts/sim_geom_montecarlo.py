"""幾何世界的雙臂對照:有死角(GAP) vs 全覆蓋(FULL)。

預先登記:`docs/M5_模擬預先登記_幾何世界_20260901.md`(commit 9009ac6,跑之前提交)。
⚠ 跑完只追加結果,不修改該文件的 §2~§7。

**這是配對比較**:同一個 seed 在兩臂下產生**完全相同的行走軌跡**
(world_geom 把動作與觀測分成兩條隨機數流),所以量到的差異只來自鏡頭幾何。

⚠ 拓撲由 `estimate_topology()` 從幾何**估**出來(照業主會做的 距離/0.9 換算),
  **絕不把世界的真實轉場分布餵給系統** —— 那就回到自問自答。

用法:
    python scripts/sim_geom_montecarlo.py --coverage-only
    python scripts/sim_geom_montecarlo.py --chefs 4 8 --reps 10 --hours 1.5
"""
import argparse
import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from m5_reid import metrics                                      # noqa: E402
from m5_reid.identity_st import SpatioTemporalIdentityManager    # noqa: E402
from m5_reid.spatiotemporal import CameraTopology                # noqa: E402
from m5_sim import world_geom as G                               # noqa: E402

# 第七輪的最佳設定。兩臂**完全相同**,只有世界的鏡頭幾何不同。
FUSION = {
    "mode": "llr", "background_arrival_hz": 1 / 600.0,
    "cost_false_merge_over_break": 5.0, "transit_model": "loiter",
    "p_loiter": 0.15, "tau_loiter_s": 20.0,
    "unknown_path": {"enabled": False},
    "same_camera": {"enabled": True, "tau_break_s": 2.0, "max_gap_s": 15.0},
    "position": {"enabled": True}, "direction": {"enabled": False},
    # ⚠ 模擬路徑不經 homography,sigma_m 必須顯式給,否則
    #   spatiotemporal.py:171-185 會判「啟用但無 σ」並靜默停用。
    "ground_plane": {"enabled": True, "sigma_m": 0.1, "area_m2": 30.0, "clip": 8.0},
    "velocity": {"enabled": True, "window_s": 1.0, "max_speed_mps": 1.5, "clip": 6.0},
}

BUDGET = {"p_break": 0.05, "p_false_merge": 0.01}


def build_topo(cfg):
    links, overlapping = G.estimate_topology(cfg)
    return CameraTopology(links=links, overlapping=overlapping, fusion=dict(FUSION),
                          cameras={c.name: {} for c in cfg.cameras}), links, overlapping


def run_once(cfg, topo):
    """跑一次,回傳 (records, 實測轉場時間)。records 格式同 metrics.binding_outcomes。"""
    events = G.generate(cfg)
    m = SpatioTemporalIdentityManager(topo, fps=1.0, recently_disappeared_ttl=600.0)
    recs, tid, seen = [], {}, set()
    for kind, gt, cam, t, emb, tag, box, zn, wxy, wv in events:
        f = int(t)
        if kind == "update":
            m.on_track_update(tid.get((gt, cam), 1) * 1000 + gt, camera_id=cam,
                              frame_id=f, t_sec=t, world_xy=wxy, world_v=wv)
            continue
        if kind == "enter":
            tid[(gt, cam)] = tid.get((gt, cam), 0) + 1
            key = tid[(gt, cam)] * 1000 + gt
            r = m.on_new_track(key, camera_id=cam, frame_id=f, t_sec=t, embedding=emb,
                               bbox=box, zone=zn, world_xy=wxy, world_v=wv)
            recs.append((gt, r.chef_id, r.matched, gt in seen))
            seen.add(gt)
        else:
            key = tid.get((gt, cam), 1) * 1000 + gt
            m.on_track_lost(key, camera_id=cam, frame_id=f, t_sec=t, bbox=box, zone=zn)
            m.on_track_removed(key, camera_id=cam, frame_id=f + 1, t_sec=t + 1.0,
                               bbox=box, zone=zn)
    return recs, G.measure_transits(events)


def grade(brk, fm):
    if brk is None or fm is None:
        return "—"
    if brk <= 0.05 and fm <= 0.01:
        return "A"
    if brk <= 0.10 and fm <= 0.03:
        return "B"
    return "C"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=["GAP", "FULL"])
    ap.add_argument("--chefs", nargs="+", type=int, default=[4, 8])
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--hours", type=float, default=1.5)
    ap.add_argument("--coverage-only", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "results" / "m5_sim" / "geom_arms.json"))
    args = ap.parse_args()

    print("=" * 78)
    print("幾何世界雙臂對照:有死角 vs 全覆蓋")
    print("=" * 78)
    print("預先登記 docs/M5_模擬預先登記_幾何世界_20260901.md(commit 9009ac6)\n")

    # ── 覆蓋率(實算,不引用註解裡的數字)──
    print(f"  {'佈局':<7}{'視角':>7}{'射程':>8}{'死角':>9}{'單鏡頭':>9}{'重疊':>9}"
          f"{'連結':>7}{'重疊對':>8}")
    print("  " + "-" * 66)
    cov_all, topo_all = {}, {}
    for arm in args.arms:
        cfg = G.WorldConfig(layout=arm)
        cov = cfg.coverage(250)
        topo, links, overlapping = build_topo(cfg)
        cov_all[arm], topo_all[arm] = cov, (len(links), len(overlapping))
        lay = G.LAYOUTS[arm]
        print(f"  {arm:<7}{lay['fov_deg']:>6.0f}°{lay['range_m']:>7.0f}m"
              f"{cov['blind']:>8.1%}{cov['single']:>9.1%}{cov['multi']:>9.1%}"
              f"{len(links):>7}{len(overlapping):>8}")
    if args.coverage_only:
        return 0

    # ── 雙臂 × 人數 ──
    print(f"\n  跑法:{args.reps} 次重複 × {args.hours} 小時 × "
          f"{len(args.arms)} 臂 × {len(args.chefs)} 種人數")
    print("  ⚠ 配對比較:同 seed 在兩臂下軌跡完全相同,差異只來自鏡頭幾何\n")
    print(f"  {'佈局':<7}{'人數':>5}{'轉場':>8}{'碎裂':>9}{'誤併':>9}"
          f"{'IDF1':>8}{'加權成本':>10}{'等級':>6}")
    print("  " + "-" * 64)

    out, ratios = {}, defaultdict(list)
    for arm in args.arms:
        for n in args.chefs:
            brks, fms, idfs, ntr = [], [], [], 0
            for rep in range(args.reps):
                cfg = G.WorldConfig(n_chefs=n, duration_s=args.hours * 3600,
                                    layout=arm, seed=1000 + rep,
                                    calib_sigma_m=0.1, tau_v_s=3.0, vel_window_s=1.0)
                topo, links, _ov = build_topo(cfg)
                recs, real = run_once(cfg, topo)
                s = metrics.summarize(recs, expected_headcount=n)
                if s["p_break"] is not None:
                    brks.append(s["p_break"])
                    ntr += s["n_transitions"]
                if s["p_false_merge"] is not None:
                    fms.append(s["p_false_merge"])
                if s["idf1"] is not None:
                    idfs.append(s["idf1"])
                # 預測 3:實測轉場時間 vs 由距離估的 mean_s
                est = {(l["from"], l["to"]): l["mean_s"] for l in links}
                for k, v in real.items():
                    if k in est and est[k] > 0 and len(v) >= 5:
                        ratios[arm].append(st.median(v) / est[k])

            brk = st.mean(brks) if brks else None
            fm = st.mean(fms) if fms else None
            cost = (brk + 5 * fm) * 100 if brk is not None and fm is not None else None
            out[f"{arm}_{n}"] = dict(arm=arm, chefs=n, n_transitions=ntr,
                                     p_break=brk, p_false_merge=fm,
                                     idf1=st.mean(idfs) if idfs else None,
                                     weighted_cost=cost, grade=grade(brk, fm))
            print(f"  {arm:<7}{n:>5}{ntr:>8}{brk:>8.1%}{fm:>9.1%}"
                  f"{(st.mean(idfs) if idfs else 0):>8.3f}{cost:>10.1f}"
                  f"{grade(brk, fm):>6}")

    # ── 對帳預先登記的預測 ──
    print("\n" + "─" * 78)
    print("對帳 §6 的五個預測")
    print("─" * 78)
    for n in args.chefs:
        g, f = out.get(f"GAP_{n}"), out.get(f"FULL_{n}")
        if not (g and f):
            continue
        db = (g["p_break"] - f["p_break"]) * 100
        dm = (g["p_false_merge"] - f["p_false_merge"]) * 100
        print(f"\n  {n} 人:GAP − FULL")
        print(f"    碎裂 {db:+.1f} 個百分點 → 預測 1(GAP 較高,差 ≥5pp)"
              f"{'  ✅ 成立' if db >= 5 else ('  ⚠ 方向對但幅度不足' if db > 0 else '  ❌ 否證')}")
        print(f"    誤併 {dm:+.1f} 個百分點 → 預測 2(GAP ≤ FULL)"
              f"{'  ✅ 成立' if dm <= 0 else '  ❌ **否證** —— 死角是雙重壞處'}")
        if abs(db) < 2 and abs(dm) < 2:
            print("    ⚠ 兩者都在雜訊帶內 → 觸發 §7 失敗條件:佈局參數不夠極端")

    for arm in args.arms:
        if ratios[arm]:
            r = sorted(ratios[arm])
            med = r[len(r) // 2]
            print(f"\n  {arm} 實測轉場 / 距離估計 = {med:.2f} 倍"
                  f"(n={len(r)})→ 預測 3(1.5~3 倍)"
                  f"{'  ✅ 成立' if 1.5 <= med <= 3.0 else '  ❌ 落在區間外'}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(
        {"coverage": cov_all, "topology": {k: dict(zip(("links", "overlapping"), v))
                                           for k, v in topo_all.items()},
         "results": out, "budget": BUDGET,
         "transit_ratio": {a: sorted(v)[len(v) // 2] for a, v in ratios.items() if v},
         "args": vars(args)},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已存 {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
