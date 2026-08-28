"""誤併發生時,兩人相距多遠?—— 檢驗第七輪的預測 1。

預先登記 docs/M5_模擬預先登記_軌跡_20260828.md §7 預測 1:
「軌跡證據的效益集中在『兩人相距 < 0.5 m』那一類,對其他情境幾乎無影響。
  若整體改善但近距離那類沒改善,表示我把證據接錯地方了。」

作法:跑同一組世界(相同 seed),軌跡證據關/開各一次,對每次誤併算出
「進場者的真實位置」與「被誤綁的那位廚師最後已知位置」之間的距離,
再比較兩次的距離分布。若軌跡真的在攻擊近距離模糊,開啟後**近距離那一類的
誤併次數應大幅下降**,而遠距離那一類幾乎不變。

用法:python scripts/diag_merge_distance.py
"""
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from m5_reid.identity_st import SpatioTemporalIdentityManager   # noqa: E402
from m5_sim import world as W                                   # noqa: E402
sys.path.insert(0, str(ROOT / "scripts"))
from sim_m5_montecarlo import _base_topo                        # noqa: E402

BINS = [(0.0, 0.5), (0.5, 1.5), (1.5, 3.0), (3.0, float("inf"))]


def bin_of(d):
    for lo, hi in BINS:
        if lo <= d < hi:
            return f"{lo:.1f}~{hi:.1f}m" if hi != float("inf") else f">{lo:.1f}m"
    return "?"


def run(wcfg, topo):
    """回傳 (誤併距離的分箱計數, 誤併總數, 轉場總數)。"""
    events = W.generate(wcfg, topo.links, topo.all_cameras(), topo.link_zones)
    m = SpatioTemporalIdentityManager(topo, fps=1.0, recently_disappeared_ttl=600.0)

    recs, seen, tid = [], set(), {}
    last_xy = {}                       # gt_id → 最後已知真實位置
    at_bind = []                       # 每次轉場綁定時的 (gt, pred, matched, 各人位置快照)
    for kind, gt, cam, t, emb, tag, box, zn, wxy, wv in events:
        f = int(t)
        if wxy is not None:
            last_xy[gt] = wxy
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
            at_bind.append((gt, r.chef_id, r.matched, gt in seen, wxy, dict(last_xy)))
            seen.add(gt)
        else:
            key = tid.get((gt, cam), 1) * 1000 + gt
            m.on_track_lost(key, camera_id=cam, frame_id=f, t_sec=t, bbox=box, zone=zn)
            m.on_track_removed(key, camera_id=cam, frame_id=f + 1, t_sec=t + 1.0,
                               bbox=box, zone=zn)

    # 用與 metrics.binding_outcomes 相同的歸屬判準找出誤併
    votes = defaultdict(Counter)
    for gt, pred, _m, _t in recs:
        votes[pred][gt] += 1
    owner = {p: v.most_common(1)[0][0] for p, v in votes.items()}

    hist, n_fm, n_tr = Counter(), 0, 0
    last_pred = {}
    for gt, pred, matched, is_tr, wxy, snap in at_bind:
        if is_tr:
            n_tr += 1
            bad = matched and (owner.get(pred) != gt or
                               (last_pred.get(gt) not in (None, pred)))
            if bad:
                n_fm += 1
                # 被誤綁到誰身上 → 那位的最後已知位置
                victim = owner.get(pred)
                v_xy = snap.get(victim)
                if wxy is not None and v_xy is not None and victim != gt:
                    hist[bin_of(math.hypot(wxy[0] - v_xy[0], wxy[1] - v_xy[1]))] += 1
                else:
                    hist["無位置"] += 1
        last_pred[gt] = pred
    return hist, n_fm, n_tr


def make_topo(velocity_on, window_s=1.0):
    """與 sim_m5_montecarlo 第七輪網格完全相同的設定,只切換 velocity。"""
    t = _base_topo()                       # 用同一個基底,避免兩支腳本各說各話
    t.fusion["unknown_path"] = {"enabled": False}
    t.fusion["same_camera"] = {"enabled": True, "tau_break_s": 2.0, "max_gap_s": 15.0}
    t.fusion["position"] = {"enabled": True}
    t.fusion["direction"] = {"enabled": False}
    t.fusion["ground_plane"] = {"enabled": True, "sigma_m": 0.1, "area_m2": 30.0,
                                "clip": 8.0}
    t.fusion["velocity"] = {"enabled": bool(velocity_on), "window_s": window_s,
                            "max_speed_mps": 1.5, "clip": 6.0}
    t._build_evidence()
    t.overlapping |= {frozenset(("master", c)) for c in ["cam1", "cam2", "cam3"]}
    t.cameras.setdefault("master", {})
    return t


def main():
    reps, hours = 10, 1.5
    print("=" * 72)
    print("誤併發生時兩人的距離分布 —— 檢驗第七輪預測 1")
    print("=" * 72)
    print("(同一組 seed,只切換軌跡證據開/關,所以差異只來自軌跡)\n")

    results = {}
    for label, vel_on in [("軌跡關", False), ("軌跡開 窗=1.0s", True)]:
        total, n_fm, n_tr = Counter(), 0, 0
        for rep in range(reps):
            wcfg = W.WorldConfig(n_chefs=4, duration_s=hours * 3600, seed=1000 + rep,
                                 master_camera="master", master_fragment_rate=0.05,
                                 calib_sigma_m=0.1, tau_v_s=3.0, vel_window_s=1.0)
            h, fm, tr = run(wcfg, make_topo(vel_on))
            total += h
            n_fm += fm
            n_tr += tr
        results[label] = (total, n_fm, n_tr)

    keys = [f"{lo:.1f}~{hi:.1f}m" if hi != float("inf") else f">{lo:.1f}m"
            for lo, hi in BINS] + ["無位置"]
    print(f"  {'兩人距離':<12}{'軌跡關':>10}{'軌跡開':>10}{'減少':>10}{'減少比例':>10}")
    print("  " + "-" * 52)
    for k in keys:
        a = results["軌跡關"][0].get(k, 0)
        b = results["軌跡開 窗=1.0s"][0].get(k, 0)
        pct = f"{(a - b) / a * 100:>9.0f}%" if a else "         -"
        print(f"  {k:<12}{a:>10}{b:>10}{a - b:>10}{pct}")
    print("  " + "-" * 52)
    for label in results:
        h, fm, tr = results[label]
        print(f"  {label:<14}誤併 {fm:>4} / 轉場 {tr:>5} = {fm / tr * 100:.1f}%")

    print("\n判讀:")
    a0 = results["軌跡關"][0].get("0.0~0.5m", 0)
    b0 = results["軌跡開 窗=1.0s"][0].get("0.0~0.5m", 0)
    far_a = sum(results["軌跡關"][0].get(k, 0) for k in keys[1:-1])
    far_b = sum(results["軌跡開 窗=1.0s"][0].get(k, 0) for k in keys[1:-1])
    near_cut = (a0 - b0) / a0 * 100 if a0 else 0
    far_cut = (far_a - far_b) / far_a * 100 if far_a else 0
    print(f"  近距離(<0.5m)減少 {near_cut:.0f}%,其餘距離減少 {far_cut:.0f}%")
    if near_cut > far_cut + 10:
        print("  → 預測 1 成立:效益確實集中在近距離模糊那一類。")
    elif abs(near_cut - far_cut) <= 10:
        print("  → 預測 1 **不成立**:效益是全面性的,不專屬近距離。")
        print("    這代表軌跡證據不只在補位置的盲點,而是獨立的一條證據軸。")
    else:
        print("  → 預測 1 **反向**:遠距離改善比近距離多 —— 證據可能接錯地方。")


if __name__ == "__main__":
    main()
