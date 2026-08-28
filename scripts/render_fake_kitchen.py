"""把模擬中的假廚房畫出來 —— 假廚師走動,系統即時綁身分。

模擬器(src/m5_sim/world.py)本來就在造一個 10×6 公尺的假廚房:廚師以 0.9 m/s
帶慣性走動、會中途逗留、會繞路、鏡頭會斷軌。缺的只是**畫面** —— 七輪實驗的
證據全是數字,沒人看得出系統到底在做什麼。

這支腳本補上那個畫面。重點在於它**不是動畫示意圖**:
  · 廚師的位置來自真正的模擬世界(OU 速度過程)
  · 身分是**真正的 SpatioTemporalIdentityManager 當場判的**,不是預先算好的
  · 判錯就會在畫面上看到 —— 這正是重點

畫法:
  實心顏色 = 這個人**真正是誰**(ground truth)
  外圈顏色 = **系統認為**他是誰
  兩色相同 → 綁對了;兩色不同 → 系統認錯人(誤併),一眼就看得出來

用法:
    python scripts/render_fake_kitchen.py --chefs 4 --seconds 240 --gif
"""
import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt              # noqa: E402
from matplotlib.patches import Circle, FancyBboxPatch   # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from m5_reid.identity_st import SpatioTemporalIdentityManager   # noqa: E402
from m5_sim import world as W                                   # noqa: E402
from sim_m5_montecarlo import _base_topo                        # noqa: E402

# 經 dataviz validator 驗過的類別色
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
           "#e87ba4", "#4a3aa7", "#008300", "#e34948"]
INK, MUTED, GRID = "#1a1a19", "#6b6a66", "#e4e3df"
FLOOR, WALL, BAD = "#f6f5f2", "#c9c7c0", "#d13438"


for _f in ("Microsoft JhengHei", "Noto Sans TC", "PingFang TC", "SimHei"):
    if _f in {x.name for x in matplotlib.font_manager.fontManager.ttflist}:
        plt.rcParams["font.family"] = _f
        break
else:
    print("⚠ 找不到中文字型,圖上的中文會變成方框")
plt.rcParams["axes.unicode_minus"] = False


def color_of(i):
    return PALETTE[int(i) % len(PALETTE)] if i is not None else "#b9b7b1"


def build_topo():
    """第七輪的最佳設定:全景鏡頭 + 地面校正 + 軌跡證據。"""
    t = _base_topo()
    t.fusion["unknown_path"] = {"enabled": False}
    t.fusion["same_camera"] = {"enabled": True, "tau_break_s": 2.0, "max_gap_s": 15.0}
    t.fusion["position"] = {"enabled": True}
    t.fusion["direction"] = {"enabled": False}
    t.fusion["ground_plane"] = {"enabled": True, "sigma_m": 0.1,
                                "area_m2": 30.0, "clip": 8.0}
    t.fusion["velocity"] = {"enabled": True, "window_s": 1.0,
                            "max_speed_mps": 1.5, "clip": 6.0}
    t._build_evidence()
    t.overlapping |= {frozenset(("master", c)) for c in ["cam1", "cam2", "cam3"]}
    t.cameras.setdefault("master", {})
    return t


def simulate(cfg, topo):
    """跑模擬 + 真的餵給 M5,回傳每個時刻的狀態供繪圖。

    回傳 (frames, stats):
      frames = [(t, {gt: (xy, cam, chef_id)}, 事件說明)]
      stats  = 綁定/碎裂/誤併的累計
    """
    events, traj = W.generate(cfg, topo.links, topo.all_cameras(), {}, return_traj=True)
    m = SpatioTemporalIdentityManager(topo, fps=1.0, recently_disappeared_ttl=600.0)

    # 先跑一遍事件流,記下「每個 gt 在每個時刻被系統指派成哪個 chef_id」
    assign = defaultdict(list)          # gt -> [(t, chef_id)]
    where = defaultdict(list)           # gt -> [(t, cam)]
    recs, tid, seen = [], {}, set()
    notes = []                          # (t, 文字) 畫面上的事件註記
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
            assign[gt].append((t, r.chef_id))
            where[gt].append((t, cam))
            if gt in seen and cam != "master":
                notes.append((t, f"{cam} ← "
                              + ("接回 chef %d" % r.chef_id if r.matched
                                 else "開新身份 chef %d" % r.chef_id)))
            seen.add(gt)
        else:
            key = tid.get((gt, cam), 1) * 1000 + gt
            m.on_track_lost(key, camera_id=cam, frame_id=f, t_sec=t, bbox=box, zone=zn)
            m.on_track_removed(key, camera_id=cam, frame_id=f + 1, t_sec=t + 1.0,
                               bbox=box, zone=zn)

    # 每個 chef_id 實際屬於誰(多數決)—— 與 metrics.binding_outcomes 同一套判準
    votes = defaultdict(Counter)
    for gt, pred, _mm, _tt in recs:
        votes[pred][gt] += 1
    owner = {p: v.most_common(1)[0][0] for p, v in votes.items()}

    n_tr = fm = brk = 0
    last = {}
    for gt, pred, matched, is_tr in recs:
        if is_tr:
            n_tr += 1
            if not matched:
                brk += 1
            elif owner.get(pred) != gt or last.get(gt) not in (None, pred):
                fm += 1
        last[gt] = pred
    stats = dict(n_transitions=n_tr, breaks=brk, false_merges=fm, owner=owner)
    return traj, assign, where, notes, stats


def at(series, t, default=None):
    """取 t 時刻之前最後一筆(階梯保持)。"""
    val = default
    for tt, v in series:
        if tt <= t:
            val = v
        else:
            break
    return val


def render(cfg, traj, assign, where, notes, stats, out_dir, fps=4, seconds=None):
    kw, kh = cfg.kitchen_m
    t_end = seconds if seconds is not None else cfg.duration_s
    n = int(t_end * fps)
    out_dir.mkdir(parents=True, exist_ok=True)
    owner = stats["owner"]
    tails = defaultdict(list)

    for i in range(n):
        t = i / fps
        fig, (ax, panel) = plt.subplots(
            1, 2, figsize=(12.4, 5.4), gridspec_kw={"width_ratios": [2.5, 1]})
        fig.patch.set_facecolor("white")

        # ── 廚房平面 ──
        ax.add_patch(FancyBboxPatch((-0.38, -0.38), kw + 0.76, kh + 0.76,
                                    boxstyle="round,pad=0,rounding_size=0.18",
                                    fc=FLOOR, ec=WALL, lw=2.5, zorder=1))
        for gx in range(1, int(kw)):
            ax.plot([gx, gx], [0, kh], color=GRID, lw=0.6, zorder=2)
        for gy in range(1, int(kh)):
            ax.plot([0, kw], [gy, gy], color=GRID, lw=0.6, zorder=2)

        n_wrong = 0
        for gt in sorted(traj):
            pt = at(traj[gt], t)
            if pt is None:
                continue
            (x, y), _v = pt
            cid = at(assign[gt], t)
            cam = at(where[gt], t)
            tails[gt].append((x, y))
            tails[gt] = tails[gt][-14:]

            xs = [p[0] for p in tails[gt]]
            ys = [p[1] for p in tails[gt]]
            ax.plot(xs, ys, color=color_of(gt), lw=1.6, alpha=0.32, zorder=3)

            # 實心 = 真實身份;外圈 = 系統認為的身份
            sys_gt = owner.get(cid)
            wrong = cid is not None and sys_gt is not None and sys_gt != gt
            n_wrong += wrong
            ax.add_patch(Circle((x, y), 0.30, fc=color_of(gt),
                                ec=BAD if wrong else color_of(sys_gt),
                                lw=4.5 if wrong else 3.5, zorder=5))
            ax.text(x, y - 0.62, f"chef {cid}" if cid else "—",
                    ha="center", fontsize=9.5, color=BAD if wrong else INK,
                    fontweight="bold" if wrong else "normal", zorder=6)
            ax.text(x, y + 0.50, cam or "", ha="center", fontsize=8,
                    color=MUTED, zorder=6)

        ax.set_xlim(-0.7, kw + 0.7)
        ax.set_ylim(-1.1, kh + 0.9)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(f"假廚房 {kw:.0f} × {kh:.0f} m     t = {t:6.1f} s",
                     fontsize=13, fontweight="bold", color=INK, loc="left", pad=10)

        # ── 右側面板 ──
        panel.axis("off")
        panel.set_xlim(0, 1)
        panel.set_ylim(0, 1)
        panel.text(0, .975, "系統當下的判斷", fontsize=12, fontweight="bold", color=INK)
        rows = []
        for gt in sorted(traj):
            cid = at(assign[gt], t)
            if cid is None:
                continue
            sys_gt = owner.get(cid)
            rows.append((gt, cid, sys_gt is not None and sys_gt != gt))
        # ⚠ 行距必須依人數縮放。固定 0.075 在 8 人時名單會壓到下一段標題,
        #   而 8 人正是最該看的情境(誤併隨人數上升)。
        gap = min(.072, .50 / max(len(rows), 1))
        y = .90
        for gt, cid, wrong in rows:
            panel.add_patch(Circle((.045, y), .019, fc=color_of(gt),
                                   ec=BAD if wrong else color_of(owner.get(cid)),
                                   lw=2.4, transform=panel.transAxes, zorder=5))
            panel.text(.11, y, f"真實 #{gt}", fontsize=9.5, va="center", color=MUTED)
            panel.text(.42, y, "→", fontsize=9.5, va="center", color=MUTED)
            panel.text(.52, y, f"chef {cid}", fontsize=10, va="center",
                       color=BAD if wrong else INK,
                       fontweight="bold" if wrong else "normal")
            if wrong:
                panel.text(.83, y, "認錯", fontsize=9, va="center",
                           color=BAD, fontweight="bold")
            y -= gap

        y -= .05
        panel.text(0, y, "最近的綁定決策", fontsize=11, fontweight="bold", color=INK)
        y -= .055
        for tt, msg in [x for x in notes if t - 6 <= x[0] <= t][-3:]:
            panel.text(0, y, f"{tt:6.1f}s   {msg}", fontsize=9, color=MUTED)
            y -= .05

        done = [x for x in notes if x[0] <= t]
        panel.text(0, .165, f"累計:轉場綁定 {len(done)} 次 · "
                   f"全程碎裂 {stats['breaks']} · 誤併 {stats['false_merges']}",
                   fontsize=9.5, color=MUTED)
        panel.text(0, .075, "說明", fontsize=10, fontweight="bold", color=INK)
        panel.text(0, .028, "實心 = 真的是誰   外圈 = 系統認為是誰",
                   fontsize=8.5, color=MUTED)
        panel.text(0, -.015, "兩色不同 = 系統認錯人(誤併)", fontsize=8.5, color=BAD)

        fig.tight_layout()
        fig.savefig(out_dir / f"k{i:05d}.png", dpi=110)
        plt.close(fig)
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chefs", type=int, default=4)
    ap.add_argument("--seconds", type=float, default=180.0)
    ap.add_argument("--fps", type=int, default=4)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--interval", type=float, default=25.0,
                    help="平均多久轉場一次(秒)。預設 60 太慢,看不到綁定")
    ap.add_argument("--out", default=str(ROOT / "results" / "m5_sim" / "kitchen"))
    ap.add_argument("--gif", action="store_true", help="另存 GIF")
    args = ap.parse_args()

    cfg = W.WorldConfig(
        n_chefs=args.chefs, duration_s=args.seconds, seed=args.seed,
        transition_interval_s=args.interval,
        master_camera="master", master_fragment_rate=0.05,
        calib_sigma_m=0.1, tau_v_s=3.0, vel_window_s=1.0,
        traj_dt_s=1.0 / args.fps)          # 視覺化才開,見 world.py 的警告

    topo = build_topo()
    traj, assign, where, notes, stats = simulate(cfg, topo)
    print(f"模擬 {args.chefs} 位廚師 / {args.seconds:.0f} 秒:"
          f"轉場 {stats['n_transitions']} 次、"
          f"碎裂 {stats['breaks']}、誤併 {stats['false_merges']}")

    out = Path(args.out)
    n = render(cfg, traj, assign, where, notes, stats, out,
               fps=args.fps, seconds=args.seconds)
    print(f"已產生 {n} 張 → {out}")

    if args.gif:
        from PIL import Image
        imgs = [Image.open(p) for p in sorted(out.glob("k*.png"))]
        gif = out.parent / "fake_kitchen.gif"
        imgs[0].save(gif, save_all=True, append_images=imgs[1:],
                     duration=int(1000 / args.fps), loop=0, optimize=True)
        print(f"已產生 {gif}({gif.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
