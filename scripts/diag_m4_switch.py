"""C. 混人 track:換人那一刻發生了什麼 —— 探索性,非預先登記(2026-09-15)。

## 為什麼

5 台 `cbiou` 有 297 條混人 track(第二個真人 ≥ 3 票);`diag_m4_ghosts.py` 查出換人時兩人框 95% 互相重疊。
要決定修法,得先知道換人時是「人不見了、track 跳到旁邊的人」,還是「兩個人都在、追蹤器配錯」。

## 規則(執行前寫死)

混人判定與 `eval_m4_chirla.py` 相同(整體真值不為空、第二名 ≥ 3 票)。
沿 track 的匈牙利配對序列(與 `match_tracks` 相同),相鄰兩次配對的人不同 = 一次換人 a → b
(f0 = 最後配到 a 的幀、f1 = 第一次配到 b 的幀)。每次換人依序分類:

  互換             另一條 track Y 在 f0 前 2 秒內配過 b,且在 f1 後 2 秒內配到 a(兩條 track 把人交換)
  a 已看不到       f1 那一幀 a 在這台鏡頭沒有標註(完全被擋或離開)
  a 還在、被接走   a 在 f1 有標註,且 f1 後 2 秒內有另一條 track 配到 a
  a 還在、沒人追   其餘

另報:b 在 f0 前 2 秒內是否已被另一條 track 配到(被搶走);f1 − f0 的幀數。

假說 HC1:> 50% 的換人發生時 a 還看得到(「a 還在」兩類 + 互換)。HC2:互換 < 30%。

用法:
    .venv/Scripts/python.exe scripts/diag_m4_switch.py --root <CHIRLA根> \\
        --tracks-dir results/m4_5cam/cbiou_dump --seqs seq_004 ... --out results/m4_5cam/switch_cbiou.json
"""
import argparse
import json
import statistics as st
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from diag_m4_ghosts import load_rows, per_frame_assign   # noqa: E402
from eval_m4m5_chirla import load_gt, match_tracks      # noqa: E402

WIN = 60          # 2 秒 × 30fps
CATS = ("互換", "a 已看不到", "a 還在、被接走", "a 還在、沒人追")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--tracks-dir", required=True)
    ap.add_argument("--seqs", nargs="+", required=True)
    ap.add_argument("--expect-mixed", type=int, default=None, help="自檢:混人 track 數應等於此值")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    switches, n_mixed = [], 0
    for seq in args.seqs:
        gt = load_gt(args.root, seq)
        rows = load_rows(Path(args.tracks_dir) / seq / "tracks.csv")
        track_gt, _s, _m, _t, votes = match_tracks(gt, rows, return_votes=True)
        info, votes2 = per_frame_assign(gt, rows)
        if {k: dict(v) for k, v in votes.items() if v} != {k: dict(v) for k, v in votes2.items() if v}:
            raise SystemExit(f"[FAIL] {seq} 逐幀配對與 match_tracks 不同")
        # (cam, 人) -> [(幀, track)] 所有配對
        by_person = defaultdict(list)
        for (cam, tid), fr in info.items():
            for fid, _bi, _bg, mg in fr:
                if mg is not None:
                    by_person[(cam, mg)].append((fid, tid))

        def tracks_on(cam, g, lo, hi, exclude):
            return {t for f, t in by_person[(cam, g)] if lo <= f <= hi and t != exclude}

        for (cam, tid), fr in info.items():
            gid = track_gt[(cam, tid)]
            if gid is None:
                continue
            top = votes[(cam, tid)].most_common(2)
            if not (len(top) >= 2 and top[1][1] >= 3):
                continue
            n_mixed += 1
            seqm = [(x[0], x[3]) for x in sorted(fr) if x[3] is not None]
            for (f0, a), (f1, b) in zip(seqm, seqm[1:]):
                if a == b:
                    continue
                after_a = tracks_on(cam, a, f1, f1 + WIN, tid)
                before_b = tracks_on(cam, b, f0 - WIN, f0, tid)
                a_vis = a in dict(gt.get(cam, {}).get(f1 + 1, []))
                if after_a & before_b:
                    cat = CATS[0]
                elif not a_vis:
                    cat = CATS[1]
                elif after_a:
                    cat = CATS[2]
                else:
                    cat = CATS[3]
                switches.append(dict(seq=seq, cam=cam, tid=tid, f0=f0, f1=f1, a=a, b=b, cat=cat,
                                     gap=f1 - f0, b_taken=bool(before_b), a_visible=a_vis))

    if args.expect_mixed is not None and n_mixed != args.expect_mixed:
        raise SystemExit(f"[FAIL] 混人 track {n_mixed} ≠ 預期 {args.expect_mixed}")
    n = len(switches)
    print(f"混人 track {n_mixed}" + (" (與預期相同 [OK])" if args.expect_mixed is not None else "")
          + f";換人事件 {n}")
    firsts = {}
    for s in switches:
        k = (s["seq"], s["cam"], s["tid"])
        if k not in firsts or s["f1"] < firsts[k]["f1"]:
            firsts[k] = s
    for name, group in (("全部換人事件", switches), ("每條 track 第一次換人", list(firsts.values()))):
        c = Counter(s["cat"] for s in group)
        m = len(group)
        print(f"\n{name}({m})")
        for k in CATS:
            print(f"   {k:<12} {c[k]:>4}({c[k] / m:.1%})")
        bt = sum(s["b_taken"] for s in group)
        gaps = sorted(s["gap"] for s in group)
        print(f"   b 在換人前已被另一條 track 追著(被搶走){bt}({bt / m:.1%})")
        print(f"   f1 − f0 幀數:中位 {st.median(gaps)}、25~75 百分位 {gaps[m // 4]} ~ {gaps[3 * m // 4]}"
              f"(stride 5 → 5 表示相鄰兩次更新就換)")

    c = Counter(s["cat"] for s in switches)
    vis = (c[CATS[0]] + c[CATS[2]] + c[CATS[3]]) / n
    print("\n假說對照(執行前寫下,以全部換人事件計)")
    print(f"   HC1 換人時 a 還看得到 > 50%:{vis:.1%} → {'成立' if vis > 0.5 else '不成立'}")
    print(f"   HC2 互換 < 30%:{c[CATS[0]] / n:.1%} → {'成立' if c[CATS[0]] / n < 0.3 else '不成立'}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(args=vars(args), rules=dict(WIN=WIN), n_mixed=n_mixed, n_switches=n,
                                   cats={k: c[k] for k in CATS},
                                   first_cats=dict(Counter(s["cat"] for s in firsts.values())),
                                   switches=switches), ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n已存 {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
