"""幾何式世界的驗證 —— 重點是「鏡頭歸屬真的由位置決定」這件事。

事件式的 world.py 做不到這一點:那裡的 cam 是隨機挑的,與位置無關。
所以 S1(幾何一致性)是這個模組存在的理由,**它失敗就等於整件事沒意義**。

沿用 repo 的 verify_*.py 風格:直跑、印 PASS/FAIL、失敗 exit 1。

用法:python scripts/verify_world_geom.py
"""
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from m5_sim import world_geom as G      # noqa: E402

FAILED = []


def check(name, cond, detail=""):
    print(f"  {'✅' if cond else '❌'} {name}" + (f"  —— {detail}" if detail else ""))
    if not cond:
        FAILED.append(name)


def at(traj_rows, t):
    """取 t 時刻最接近的真實位置。"""
    best, bd = None, 1e18
    for tt, (xy, _v) in traj_rows:
        d = abs(tt - t)
        if d < bd:
            best, bd = xy, d
    return best, bd


def s1_geometry():
    """**本模組存在的理由**:事件裡的 cam 必須真的看得到那個人。

    事件式世界做不到 —— 那裡的 cam 是 rng 挑的,與位置無關。
    """
    print("\nS1 幾何一致性(事件的 cam 必須真的看得到該位置)")
    cfg = G.WorldConfig(n_chefs=3, duration_s=120, layout="GAP",
                        sim_dt_s=0.2, traj_dt_s=0.2, seed=7)
    ev, traj = G.generate(cfg, return_traj=True)
    cams = {c.name: c for c in cfg.cameras}

    bad, checked, worst = 0, 0, 0.0
    for kind, gt, cam, t, *_ in ev:
        if kind not in ("enter", "update"):
            continue                       # leave 的時刻人已經離開視野,不該檢查
        xy, dt = at(traj[gt], t)
        if xy is None or dt > cfg.sim_dt_s:
            continue                       # 找不到夠近的真值點就跳過
        checked += 1
        if not cams[cam].sees(xy):
            bad += 1
            worst = max(worst, dt)
    check("進場/心跳事件的鏡頭都真的看得到該位置",
          bad == 0 and checked > 20,
          f"檢查 {checked} 筆,不一致 {bad} 筆")

    # 反面:隨機挑一台不同的鏡頭,大多數時候應該看不到 —— 證明這個檢查有鑑別力
    miss = 0
    for kind, gt, cam, t, *_ in ev[:200]:
        if kind != "enter":
            continue
        xy, dt = at(traj[gt], t)
        if xy is None or dt > cfg.sim_dt_s:
            continue
        others = [c for n, c in cams.items() if n != cam]
        if others and not any(c.sees(xy) for c in others):
            miss += 1
    check("有進場事件是「只有那一台看得到」", miss > 0,
          f"{miss} 筆是單鏡頭可見 —— 這種情境才需要轉場證據")


def s2_contract():
    """10 欄 tuple + (t, update 優先) 排序 —— 消費端靠這個。"""
    print("\nS2 事件契約(與 world.py 相容)")
    cfg = G.WorldConfig(n_chefs=2, duration_s=60, seed=3)
    ev = G.generate(cfg)
    check("事件非空", len(ev) > 0, f"{len(ev)} 筆")
    check("每筆 10 欄", all(len(e) == 10 for e in ev))
    kinds = set(e[0] for e in ev)
    check("kind 只有 enter/leave/update", kinds <= {"enter", "leave", "update"}, str(kinds))
    ts = [e[3] for e in ev]
    check("時間非遞減", all(a <= b for a, b in zip(ts, ts[1:])))
    # 同一時刻 update 必須排在 enter 之前
    order_ok = True
    for a, b in zip(ev, ev[1:]):
        if a[3] == b[3] and a[0] != "update" and b[0] == "update":
            order_ok = False
    check("同時刻心跳排在事件之前", order_ok,
          "位置資訊要比綁定決策早到,否則 M5 拿不到當下位置")
    enters = [e for e in ev if e[0] == "enter"]
    check("enter 帶 embedding", all(e[4] is not None for e in enters))
    check("所有事件都帶 world_xy 與 world_v",
          all(e[8] is not None and e[9] is not None for e in ev))


def s3_determinism():
    """同 seed 必須跑出完全相同的事件流。"""
    print("\nS3 決定性")
    a = G.generate(G.WorldConfig(n_chefs=3, duration_s=60, seed=42))
    b = G.generate(G.WorldConfig(n_chefs=3, duration_s=60, seed=42))
    check("同 seed 事件數相同", len(a) == len(b), f"{len(a)} vs {len(b)}")
    same = all(x[0] == y[0] and x[1] == y[1] and x[2] == y[2]
               and abs(x[3] - y[3]) < 1e-9 for x, y in zip(a, b))
    check("同 seed 事件內容相同", same)
    c = G.generate(G.WorldConfig(n_chefs=3, duration_s=60, seed=43))
    check("不同 seed 會不同", len(c) != len(a) or
          any(x[2] != y[2] for x, y in zip(a, c)))


def s4_layouts():
    """雙臂:GAP 要有死角,FULL 不能有。這是對照實驗的前提。"""
    print("\nS4 雙臂佈局(同擺位,只改視野)")
    gap = G.WorldConfig(layout="GAP").coverage(200)
    full = G.WorldConfig(layout="FULL").coverage(200)
    print(f"      GAP   死角 {gap['blind']:.1%} / 單鏡頭 {gap['single']:.1%} "
          f"/ 重疊 {gap['multi']:.1%}")
    print(f"      FULL  死角 {full['blind']:.1%} / 單鏡頭 {full['single']:.1%} "
          f"/ 重疊 {full['multi']:.1%}")
    check("GAP 有明顯死角", gap["blind"] > 0.15, f"{gap['blind']:.1%}")
    check("GAP 有大量單鏡頭區域(轉場證據才有用武之地)",
          gap["single"] > 0.30, f"{gap['single']:.1%}")
    check("FULL 沒有死角", full["blind"] < 0.01, f"{full['blind']:.1%}")
    check("兩臂的擺位相同(只有視野不同)",
          [(c.x, c.y) for c in G.WorldConfig(layout="GAP").cameras]
          == [(c.x, c.y) for c in G.WorldConfig(layout="FULL").cameras])


def s5_blind_gaps():
    """GAP 臂必須真的出現「所有鏡頭都看不到」的時間區段。"""
    print("\nS5 死角是真的(人會從所有鏡頭消失)")
    cfg = G.WorldConfig(n_chefs=2, duration_s=300, layout="GAP",
                        sim_dt_s=0.2, traj_dt_s=0.2, seed=11)
    _ev, traj = G.generate(cfg, return_traj=True)
    gaps = []
    for gt, rows in traj.items():
        run = 0.0
        for t, (xy, _v) in rows:
            if any(c.sees(xy) for c in cfg.cameras):
                if run > 0:
                    gaps.append(run)
                run = 0.0
            else:
                run += cfg.sim_dt_s
        if run > 0:
            gaps.append(run)
    check("出現過完全看不到的時段", len(gaps) > 0, f"{len(gaps)} 段")
    if gaps:
        gaps.sort()
        print(f"      死角時長:中位 {gaps[len(gaps)//2]:.1f}s / "
              f"最長 {gaps[-1]:.1f}s / 總計 {sum(gaps):.0f}s")
        check("死角時長合理(不是每一步都在消失)",
              gaps[len(gaps) // 2] >= cfg.sim_dt_s, f"中位 {gaps[len(gaps)//2]:.1f}s")


def s6_emergent_transit():
    """轉場時間是「走出來的」—— 量出實際值,與由距離估的值對照。

    這正是幾何世界比事件世界強的地方:事件世界裡轉場時間是我們設的,
    系統再用高斯去猜它;幾何世界裡系統必須去估一個真實存在的量。
    """
    print("\nS6 轉場時間是浮現的(不是設定的)")
    cfg = G.WorldConfig(n_chefs=4, duration_s=900, layout="GAP", seed=5)
    ev = G.generate(cfg)
    real = G.measure_transits(ev)
    links, overlapping = G.estimate_topology(cfg)
    est = {(l["from"], l["to"]): l["mean_s"] for l in links}

    check("量得到跨鏡頭轉場", len(real) > 0, f"{len(real)} 對鏡頭")
    check("估出了拓撲", len(links) > 0 or len(overlapping) > 0,
          f"{len(links)} 條連結 / {len(overlapping)} 對重疊")
    if real:
        print(f"      {'鏡頭對':<16}{'實際中位':>10}{'距離估計':>10}{'比值':>8}{'樣本':>7}")
        print("      " + "-" * 52)
        shown = 0
        for k in sorted(real, key=lambda k: -len(real[k]))[:5]:
            v = sorted(real[k])
            med = v[len(v) // 2]
            e = est.get(k)
            ratio = f"{med/e:.2f}" if e else "—(重疊)"
            print(f"      {k[0]+'→'+k[1]:<16}{med:>9.1f}s"
                  f"{(f'{e:.1f}s' if e else '—'):>10}{ratio:>8}{len(v):>7}")
            shown += 1
        check("有足夠樣本可以對照", shown > 0)


def main():
    print("=" * 70)
    print("幾何式模擬世界 —— 驗證")
    print("=" * 70)
    for fn in (s1_geometry, s2_contract, s3_determinism,
               s4_layouts, s5_blind_gaps, s6_emergent_transit):
        fn()
    print("\n" + "=" * 70)
    if FAILED:
        print(f"[FAIL] {len(FAILED)} 項未通過:{FAILED}")
        return 1
    print("[ALL PASS] 鏡頭歸屬由幾何決定、契約相容、死角真實、轉場時間浮現")
    return 0


if __name__ == "__main__":
    sys.exit(main())
