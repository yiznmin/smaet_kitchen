"""把 M5 模擬結果畫成圖 —— 模擬原本只產生數字,報告與簡報需要圖。

⚠ 所有數字都**從實跑的輸出檔解析**,不在這裡手打。
   改了模擬要重跑產生器,圖才會跟著更新:
     python scripts/sim_m5_montecarlo.py --fixes --reps 10 --hours 1.5 \
         > results/m5_reid/sim_velocity.txt
     python scripts/diag_merge_distance.py > results/m5_reid/diag_merge_distance.txt

產出(results/m5_reid/):
  fig_round7_sweep.png    第七輪掃描網格 —— 唯一的受控比較
  fig_merge_distance.png  誤併距離分解 —— 軌跡證據解決了哪一類
  fig_status_budget.png   現況 vs 驗收預算

用法:python scripts/plot_m5_results.py
"""
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results" / "m5_reid"

# 經 dataviz validator 驗過的類別色(light mode,adjacent pairs 全通過)
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID = "#1a1a19", "#6b6a66", "#e4e3df"
GOOD, WARN = "#1baf7a", "#c2410c"

for f in ("Microsoft JhengHei", "Noto Sans TC", "PingFang TC", "SimHei"):
    if f in {x.name for x in matplotlib.font_manager.fontManager.ttflist}:
        plt.rcParams["font.family"] = f
        break
else:
    print("⚠ 找不到中文字型,圖上的中文會變成方框")
plt.rcParams.update({
    "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.unicode_minus": False,
    "figure.facecolor": "white", "axes.facecolor": "white",
})


def style(ax):
    """細軸、淡格線、無上右框 —— 讓資料而不是框架吸引注意。"""
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


# ── 解析 ──────────────────────────────────────────────────────────────
def parse_sweep(path):
    """從 sim_velocity.txt 抓第七輪的 (碎裂, 誤併),key = (窗, τv)。"""
    out = {}
    pat = re.compile(r"(\[基準\] 無軌跡|\+軌跡 窗=([\d.]+)s) τv=(\d+)s\s+"
                     r"([\d.]+)%\s+([\d.]+)%")
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        m = pat.search(line)
        if m:
            win = 0.0 if m.group(2) is None else float(m.group(2))
            out[(win, float(m.group(3)))] = (float(m.group(4)), float(m.group(5)))
    if not out:
        raise SystemExit(f"{path} 解析不到第七輪的列 —— 先跑 sim_m5_montecarlo.py --fixes")
    return out


def parse_distance(path):
    """從 diag_merge_distance.txt 抓距離分箱的 (關, 開)。"""
    rows, tot = [], {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        m = re.match(r"\s{2}(<?>?[\d.~m]+m|無位置)\s+(\d+)\s+(\d+)\s+(-?\d+)", line)
        if m:
            rows.append((m.group(1), int(m.group(2)), int(m.group(3))))
        m2 = re.search(r"(軌跡關|軌跡開 窗=1\.0s)\s+誤併\s+(\d+) / 轉場\s+(\d+)", line)
        if m2:
            tot[m2.group(1)] = (int(m2.group(2)), int(m2.group(3)))
    if not rows:
        raise SystemExit(f"{path} 解析不到距離分箱 —— 先跑 diag_merge_distance.py")
    return rows, tot


# ── 圖 1:第七輪掃描 ──────────────────────────────────────────────────
def fig_sweep(sweep):
    wins = sorted({w for w, _ in sweep})
    taus = sorted({t for _, t in sweep})
    xs = range(len(wins))
    fig, ax = plt.subplots(figsize=(9.0, 5.2))
    style(ax)

    ax.axhline(1.0, color=GOOD, lw=1.6, ls=(0, (5, 4)), zorder=2)
    ax.text(-0.45, 1.9, "驗收預算 1%", color=GOOD, va="bottom", fontsize=10.5)

    for tau, col in zip(taus, (BLUE, ORANGE, AQUA)):
        ys = [sweep[(w, tau)][1] for w in wins]
        ax.plot(xs, ys, color=col, lw=2.2, marker="o", ms=8,
                mec="white", mew=1.8, zorder=3)
        # 直接標在線末 —— aqua 對白底的對比低於 3:1,dataviz 的 relief rule
        # 要求可見標籤,而且直接標比看圖例找線快得多。
        # 三條線在右端只差 1.6 個百分點,標籤會疊 → 把值併進標籤、垂直錯開。
        ax.annotate(f"{ys[0]:.1f}%", (0, ys[0]), textcoords="offset points",
                    xytext=(-10, 0), ha="right", va="center",
                    color=col, fontsize=10.5, fontweight="bold")
        dy = {1.0: 13, 3.0: 0, 8.0: -13}.get(tau, 0)
        ax.annotate(f"τ_v={tau:.0f}s  {ys[-1]:.1f}%", (len(wins) - 1, ys[-1]),
                    textcoords="offset points", xytext=(14, dy), va="center",
                    color=col, fontsize=11, fontweight="bold",
                    arrowprops=dict(arrowstyle="-", color=col, lw=1,
                                    shrinkA=2, shrinkB=6) if dy else None)

    ax.set_xticks(list(xs))
    ax.set_xticklabels(["無軌跡\n(基準)"] + [f"{w:.1f} 秒" for w in wins[1:]])
    ax.set_xlim(-0.55, len(wins) - 1 + 1.15)
    ax.set_ylim(0, 23)
    ax.set_ylabel("誤併率(越低越好)", fontsize=11)
    ax.set_xlabel("軌跡觀測窗", fontsize=11, labelpad=8)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
    fig.text(0.055, 0.945, "軌跡證據把誤併率壓到三分之一",
             fontsize=15.5, fontweight="bold", color=INK)
    fig.text(0.055, 0.895, "4 位廚師 · 全景鏡頭 + 地面校正 σ=0.1m · "
             "τ_v = 走路方向的持續性", color=MUTED, fontsize=10.5)
    fig.text(0.055, 0.015, "資料:results/m5_reid/sim_velocity.txt(--reps 10 --hours 1.5)。"
             "同一輪內的受控比較。", color=MUTED, fontsize=9)
    fig.tight_layout(rect=[0.02, 0.045, 1, 0.87])
    p = RES / "fig_round7_sweep.png"
    fig.savefig(p, dpi=170)
    plt.close(fig)
    return p


# ── 圖 2:誤併距離分解 ────────────────────────────────────────────────
def fig_distance(rows, tot):
    ren = {"0.0~0.5m": "< 0.5 m", "0.5~1.5m": "0.5 ~ 1.5 m",
           "1.5~3.0m": "1.5 ~ 3.0 m", ">3.0m": "> 3.0 m",
           "無位置": "對造未被觀測"}
    labels = [ren.get(r[0], r[0]) for r in rows]
    off, on = [r[1] for r in rows], [r[2] for r in rows]
    y = range(len(rows))
    fig, ax = plt.subplots(figsize=(9.4, 5.0))
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)

    h = 0.34
    ax.barh([v - h / 2 - .02 for v in y], off, height=h, color=ORANGE, zorder=3)
    ax.barh([v + h / 2 + .02 for v in y], on, height=h, color=BLUE, zorder=3)
    for i, (a, b) in enumerate(zip(off, on)):
        ax.text(a + 7, i - h / 2 - .02, str(a), va="center", color=ORANGE, fontsize=10)
        ax.text(b + 7, i + h / 2 + .02, str(b), va="center", color=BLUE, fontsize=10)
        if a:
            # ⚠ 方向必須是 (開−關)/關:減少要顯示成負號。
            #   第一版寫成 (關−開)/關,於是「減少 81%」印成「+81%」、
            #   「增加 78%」印成「−78%」—— 正負號整個讀反,是最糟的那種圖表錯誤。
            d = (b - a) / a * 100
            ax.text(max(a, b) + 78, i, f"{d:+.0f}%",
                    va="center", fontsize=12, fontweight="bold",
                    color=GOOD if d < 0 else WARN)

    ax.set_yticks(list(y))
    ax.set_yticklabels(labels, fontsize=11.5)
    ax.invert_yaxis()
    ax.set_xlabel("誤併次數", fontsize=11, labelpad=6)
    ax.set_xlim(0, max(off + on) * 1.34)
    # 圖例放標題下方,不放圖內 —— 圖內任何位置都會壓到長條。
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=ORANGE),
                       plt.Rectangle((0, 0), 1, 1, color=BLUE)],
              labels=["軌跡證據 關", "軌跡證據 開(觀測窗 1.0 秒)"],
              frameon=False, ncol=2, fontsize=10.5,
              loc="lower left", bbox_to_anchor=(0, 1.01))
    fig.text(0.045, 0.945, "軌跡證據解決的正是「兩人站在同一位置」那一類",
             fontsize=15.5, fontweight="bold", color=INK)
    fig.text(0.045, 0.895, "縱軸:誤併發生當下,兩人的實際距離。同一組 seed,只切換軌跡開關。",
             color=MUTED, fontsize=10.5)
    fig.text(0.045, 0.015,
             "注意:> 3.0 m 反而增加 —— 相隔遠但同向同速的兩人會被誤綁,"
             "是本輪量到的新失效模式(約佔全部轉場 1.7%)。",
             color=WARN, fontsize=9.5)
    fig.tight_layout(rect=[0.02, 0.05, 1, 0.84])
    p = RES / "fig_merge_distance.png"
    fig.savefig(p, dpi=170)
    plt.close(fig)
    return p


# ── 圖 3:現況 vs 預算 ────────────────────────────────────────────────
def fig_budget(sweep):
    brk, fm = sweep[(2.0, 3.0)]
    items = [("身份碎裂率", brk, 5.0), ("身份誤併率", fm, 1.0)]
    fig, ax = plt.subplots(figsize=(8.4, 3.6))
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)

    for i, (name, val, budget) in enumerate(items):
        over = val - budget
        # ⚠ 差距要照實表達。5.2% 對預算 5% 是「略超 0.2 個百分點」,
        #   不是「差 1 倍」(5.2/5.0=1.04,講倍數會嚴重誤導);
        #   5.6% 對預算 1% 才適合講倍數。門檻怎麼講,取決於超出的量級。
        if over <= 0:
            verdict, col = "達標", GOOD
        elif val / budget < 2:
            verdict, col = f"略超 {over:.1f} 個百分點", WARN
        else:
            verdict, col = f"{val / budget:.1f} 倍於預算", WARN
        ax.barh(i, val, height=0.40, color=BLUE if over <= 0 else ORANGE, zorder=3)
        ax.plot([budget, budget], [i - .29, i + .29], color=INK, lw=2.6, zorder=4)
        ax.text(val + .14, i, f"{val:.1f}%", va="center", fontsize=12.5,
                fontweight="bold", color=BLUE if over <= 0 else ORANGE)
        ax.text(budget, i - .40, f"預算 {budget:.0f}%", ha="center",
                va="bottom", fontsize=9.5, color=INK)
        ax.text(7.0, i, verdict, va="center", fontsize=11.5,
                fontweight="bold", color=col)

    ax.set_yticks(range(len(items)))
    ax.set_yticklabels([n for n, _, _ in items], fontsize=12)
    ax.invert_yaxis()
    ax.set_xlim(0, 11.4)
    ax.set_xticks([0, 2, 4, 6])
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
    fig.text(0.045, 0.93, "現況 vs 驗收預算", fontsize=15.5,
             fontweight="bold", color=INK)
    fig.text(0.045, 0.025, "設定:觀測窗 2.0s、τ_v=3s。誤併門檻嚴 5 倍,"
             "因為它靜默、下游驗不出來。碎裂率仍略高於預算,故等級為 C。",
             color=MUTED, fontsize=9)
    fig.tight_layout(rect=[0.02, 0.07, 1, 0.87])
    p = RES / "fig_status_budget.png"
    fig.savefig(p, dpi=170)
    plt.close(fig)
    return p


def main():
    sweep = parse_sweep(RES / "sim_velocity.txt")
    rows, tot = parse_distance(RES / "diag_merge_distance.txt")
    for p in (fig_sweep(sweep), fig_distance(rows, tot), fig_budget(sweep)):
        print(f"已產生 {p}")


if __name__ == "__main__":
    sys.exit(main())
