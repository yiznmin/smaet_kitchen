"""A. 用時間分辨重複 track 與真的新人:新 track 出生後等 W 秒,期間消失的就當沒出現過 —— 探索性,非預先登記(2026-09-15)。

## 為什麼

`diag_m4_birth_geometry.py` 查出只看出生時刻的框幾何分不開(後方另一個人的局部框與同一人的局部框長得一樣)。
觀察到重複 track 通常很短 —— 這支量「等幾秒」能分開多少、代價多大。

## 規則(執行前寫死)

出生標記直接讀 `diag_m4_birth_geometry.py` 的輸出(D 重複 / N 新人 / G 不碰人 / U 其他;真值只用來評分)。
track 存活長度 = tracks.csv 中最後一幀 − 第一幀(事後才知道,**代表系統要晚 W 秒才決定**)。

  只看時間:存活 < W 秒 → 丟掉
  時間 + 幾何:存活 < W 秒 **且** 出生時與既有框重疊 ≥ 0.3 → 丟掉

  W ∈ {0.5, 1, 1.5, 2, 3, 5} 秒

目標:丟掉 D ≥ 70% 且丟掉 N ≤ 5%。
假說 HA1:某個 W 只看時間就達標。HA2:同一個 W 下,時間 + 幾何丟掉的 N 少於只看時間。

另報(執行前沒寫死、看到上一輪結果後才加的口徑,只供解讀):N 扣掉 151 個「同一幀已有同一人舊 track」後的丟掉比例。

用法:
    .venv/Scripts/python.exe scripts/diag_m4_birth_time.py --root <CHIRLA根> \\
        --tracks-dir results/m4_5cam/cbiou_dump --seqs seq_004 ... \\
        --births-json results/m4_5cam/birth_geometry_cbiou.json --out results/m4_5cam/birth_time_cbiou.json
"""
import argparse
import json
import statistics as st
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from diag_m4_ghosts import load_rows, per_frame_assign   # noqa: E402
from eval_m4m5_chirla import load_gt                    # noqa: E402

WAITS = (0.5, 1.0, 1.5, 2.0, 3.0, 5.0)
GEO_IOU = 0.3
FPS = 30.0
TARGET_D = 0.70
TARGET_N = 0.05
LABELS = ("D 重複", "N 新人", "G 不碰人", "U 其他")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--tracks-dir", required=True)
    ap.add_argument("--seqs", nargs="+", required=True)
    ap.add_argument("--births-json", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    births = json.loads(Path(args.births_json).read_text(encoding="utf-8"))["births"]
    life, matched = {}, {}
    total_matched = 0
    for seq in args.seqs:
        gt = load_gt(args.root, seq)
        rows = load_rows(Path(args.tracks_dir) / seq / "tracks.csv")
        span = {}
        for r in rows:
            k = (r["cam"], r["tid"])
            lo, hi = span.get(k, (r["fid"], r["fid"]))
            span[k] = (min(lo, r["fid"]), max(hi, r["fid"]))
        info, _ = per_frame_assign(gt, rows)
        for k, (lo, hi) in span.items():
            life[(seq,) + k] = (hi - lo) / FPS
            m = sum(1 for x in info[k] if x[3] is not None)
            matched[(seq,) + k] = m
            total_matched += m

    for b in births:
        k = (b["seq"], b["cam"], b["tid"])
        if k not in life:
            raise SystemExit(f"[FAIL] 出生紀錄 {k} 在 tracks.csv 找不到")
        b["life_s"] = life[k]
        b["matched_frames"] = matched[k]
    if len(births) != len(life):
        raise SystemExit(f"[FAIL] 出生數 {len(births)} ≠ track 數 {len(life)}")

    lab = Counter(b["label"] for b in births)
    n_real = [b for b in births if b["label"] == LABELS[1] and not b["n_same_person_old"]]
    print(f"出生 {len(births):,}(與 track 數相同 [OK]);真人配對幀合計 {total_matched:,}")
    print("\n1. 存活長度(秒)中位 / 25~75 百分位")
    for k in LABELS:
        v = sorted(b["life_s"] for b in births if b["label"] == k)
        print(f"   {k:<8} {len(v):>5}  中位 {st.median(v):.2f}  ({v[len(v) // 4]:.2f} ~ {v[3 * len(v) // 4]:.2f})")

    rows_out = []
    print(f"\n2. 規則表(目標:丟掉 D ≥ {TARGET_D:.0%} 且丟掉 N ≤ {TARGET_N:.0%})")
    print(f"   {'規則':<22}{'丟掉 D':>13}{'丟掉 N':>13}{'N(事後口徑)':>14}{'丟掉 G':>9}{'丟掉的真人幀':>14}{'達標':>5}")
    for mode in ("時間", "時間+幾何"):
        for w in WAITS:
            def drop(b):
                short = b["life_s"] < w
                return short and (mode == "時間" or b["iou"] >= GEO_IOU)
            dropped = [b for b in births if drop(b)]
            h = Counter(b["label"] for b in dropped)
            rd = h[LABELS[0]] / lab[LABELS[0]]
            rn = h[LABELS[1]] / lab[LABELS[1]]
            rn_real = sum(1 for b in n_real if drop(b)) / len(n_real)
            lost = sum(b["matched_frames"] for b in dropped if b["label"] != LABELS[0])
            ok = rd >= TARGET_D and rn <= TARGET_N
            rows_out.append(dict(mode=mode, wait_s=w, hits={k: h[k] for k in LABELS}, drop_d=rd, drop_n=rn,
                                 drop_n_real_posthoc=rn_real, lost_matched_frames=lost,
                                 lost_share=lost / total_matched, ok=ok))
            name = f"{mode} < {w}s" + (f" 且重疊≥{GEO_IOU}" if mode != "時間" else "")
            print(f"   {name:<22}{h[LABELS[0]]:>5}({rd:>6.1%}){h[LABELS[1]]:>5}({rn:>6.1%}){rn_real:>13.1%}"
                  f"{h[LABELS[2]]:>9}{lost:>7}({lost / total_matched:>5.2%}){'是' if ok else '':>5}")

    print("\n3. 假說對照(執行前寫下)")
    t_ok = [r for r in rows_out if r["mode"] == "時間" and r["ok"]]
    print(f"   HA1 只看時間有 W 達標:{'成立(W = ' + ', '.join(str(r['wait_s']) for r in t_ok) + ')' if t_ok else '不成立'}")
    better = all(g["drop_n"] <= t["drop_n"] for t, g in zip(
        [r for r in rows_out if r["mode"] == "時間"], [r for r in rows_out if r["mode"] == "時間+幾何"]))
    strictly = any(g["drop_n"] < t["drop_n"] for t, g in zip(
        [r for r in rows_out if r["mode"] == "時間"], [r for r in rows_out if r["mode"] == "時間+幾何"]))
    print(f"   HA2 同一個 W 下,時間+幾何丟掉的 N 較少:{'成立' if better and strictly else '不成立'}"
          "(⚠ 丟掉的 D 也會跟著變少,要一起看)")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(args=vars(args), rules=dict(WAITS=WAITS, GEO_IOU=GEO_IOU, TARGET_D=TARGET_D,
                                                                TARGET_N=TARGET_N),
                                   labels={k: lab[k] for k in LABELS}, total_matched_frames=total_matched,
                                   table=rows_out), ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n已存 {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
