"""②「貼著人但框不準」的 track 出現時,同一個人是否已有另一條 track 在追 —— 探索性診斷,非預先登記(2026-09-15)。

## 為什麼

`diag_m4_boxes.py` 查出 ② 主要發生在遮擋與多人擠在一起時。修法取決於一個還沒量的問題:

  重複   同一台鏡頭、同一幀,已經有另一條 track 配到同一個人 → 同一個人被多開一條
         → 修法在追蹤器:不要在已被追蹤的人身上再開一條
  唯一   沒有別的 track 在追這個人 → ② 是他被擋住時唯一的 track
         → 修法在關聯:擋完之後要接回原本的 track,否則就是斷點

## 量什麼(規則執行前寫死)

對每條 ② track 的每一幀「與某人框 IoU ≥ 0.1」的幀,取 IoU 最大的那個人 g:
同一幀是否有**另一條** track 以匈牙利配對配到 g(與 `match_tracks` 同一個配對)。

  track 層級:這種幀佔比 ≥ 50% → 重複;否則 → 唯一
  出生時刻:第一個「與某人框 IoU ≥ 0.1」的幀是否已有另一條 track 配到那個人(M5 在 track 出生時就要判斷)
  唯一的那些:取 ② 期間多數幀的 g;同一台鏡頭上 g 的真人 track
    是否在 ② 開始前 5 秒內結束(前面剛斷)、是否在 ② 結束後 5 秒內開始(後面重開)

用法:
    .venv/Scripts/python.exe scripts/diag_m4_dup.py --root <CHIRLA根> \\
        --tracks-dir results/m4_5cam/cbiou_dump --seqs seq_004 ... \\
        --ghosts-json results/m4_5cam/ghosts_cbiou.json --boxes-json results/m4_5cam/boxes_cbiou.json \\
        --out results/m4_5cam/dup_cbiou.json
"""
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from diag_m4_ghosts import CATS, NEAR_IOU, features, load_rows, per_frame_assign   # noqa: E402
from eval_m4m5_chirla import load_gt, match_tracks                             # noqa: E402

DUP_SHARE = 0.5
WINDOW_S = 5.0
FPS = 30.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--tracks-dir", required=True)
    ap.add_argument("--seqs", nargs="+", required=True)
    ap.add_argument("--ghosts-json", required=True)
    ap.add_argument("--boxes-json", default=None, help="diag_m4_boxes.py 的輸出,用來依框的關係分組")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    win = int(WINDOW_S * FPS)

    kind_of = {}
    if args.boxes_json:
        for t in json.loads(Path(args.boxes_json).read_text(encoding="utf-8"))["tracks"]:
            kind_of[(t["seq"], t["cam"], t["tid"])] = t["kind"]

    out_tracks = []
    for seq in args.seqs:
        gt = load_gt(args.root, seq)
        rows = load_rows(Path(args.tracks_dir) / seq / "tracks.csv")
        track_gt, *_ = match_tracks(gt, rows)
        info, _ = per_frame_assign(gt, rows)
        by = defaultdict(list)
        for r in rows:
            by[(r["cam"], r["tid"])].append(r)

        matched_by = defaultdict(dict)          # (cam, fid) -> {真人: 配到他的 track}
        for (cam, tid), fr in info.items():
            for fid, _bi, _bg, mg in fr:
                if mg is not None:
                    matched_by[(cam, fid)][mg] = tid
        span = {k: (min(r["fid"] for r in rs), max(r["fid"] for r in rs)) for k, rs in by.items()}

        for (cam, tid), rs in by.items():
            if track_gt[(cam, tid)] is not None:
                continue
            fr = sorted(info[(cam, tid)])
            if features(fr, [r["bbox"] for r in rs], [r["conf"] for r in rs])["cat"] != CATS[1]:
                continue
            near = [x for x in fr if x[1] >= NEAR_IOU]
            dup = []
            for fid, _bi, bg, _mg in near:
                o = matched_by[(cam, fid)].get(bg)
                dup.append(o is not None and o != tid)
            share = sum(dup) / len(near)
            g = Counter(x[2] for x in near).most_common(1)[0][0]
            f0, f1 = span[(cam, tid)]
            real_g = [k for k in by if k[0] == cam and k != (cam, tid) and track_gt[k] == g]
            before = any(f0 - win <= span[k][1] < f0 for k in real_g)
            after = any(f1 < span[k][0] <= f1 + win for k in real_g)
            out_tracks.append(dict(seq=seq, cam=cam, tid=tid, n_near=len(near), dup_share=share,
                                   dup=share >= DUP_SHARE, dup_at_first=dup[0], person=g,
                                   real_ended_before=before, real_started_after=after,
                                   kind=kind_of.get((seq, cam, tid))))

    ref = json.loads(Path(args.ghosts_json).read_text(encoding="utf-8"))["ghost_cats"][CATS[1]]
    if len(out_tracks) != ref:
        raise SystemExit(f"[FAIL] ② 條數 {len(out_tracks)} ≠ {args.ghosts_json} 的 {ref}")
    if args.boxes_json and any(t["kind"] is None for t in out_tracks):
        raise SystemExit("[FAIL] 有 ② track 在 boxes-json 找不到")
    n = len(out_tracks)
    print(f"② 貼著人但框不準:{n} 條(與 {args.ghosts_json} 相同 [OK])")

    nd = sum(t["dup"] for t in out_tracks)
    print(f"\n1. 同一幀是否已有另一條 track 配到同一個人(≥ {DUP_SHARE:.0%} 的幀 → 重複)")
    print(f"   重複 {nd}({nd / n:.1%})  唯一 {n - nd}({(n - nd) / n:.1%})")
    bins = Counter("0%" if t["dup_share"] == 0 else "100%" if t["dup_share"] == 1
                   else "(0,50%)" if t["dup_share"] < 0.5 else "[50%,100%)" for t in out_tracks)
    print("   佔比分布:" + "、".join(f"{b} {bins[b]}" for b in ("0%", "(0,50%)", "[50%,100%)", "100%")))
    nf = sum(t["dup_at_first"] for t in out_tracks)
    print(f"   出生時刻已有另一條在追:{nf}({nf / n:.1%})")

    print("\n2. 依框的關係分組(重複 / 條數)")
    kinds = sorted({t["kind"] for t in out_tracks}, key=str)
    for k in kinds:
        ts = [t for t in out_tracks if t["kind"] == k]
        print(f"   {str(k):<10} {sum(t['dup'] for t in ts):>3} / {len(ts):<3}({sum(t['dup'] for t in ts) / len(ts):.1%})")

    uniq = [t for t in out_tracks if not t["dup"]]
    print(f"\n3. 唯一的 {len(uniq)} 條:同一台鏡頭上那個人的真人 track")
    if uniq:
        b = sum(t["real_ended_before"] for t in uniq)
        a = sum(t["real_started_after"] for t in uniq)
        both = sum(t["real_ended_before"] and t["real_started_after"] for t in uniq)
        print(f"   ② 開始前 {WINDOW_S:.0f} 秒內剛結束 {b}({b / len(uniq):.1%});"
              f"② 結束後 {WINDOW_S:.0f} 秒內重新開始 {a}({a / len(uniq):.1%});兩者皆是 {both}({both / len(uniq):.1%})")

    print("\n4. 假說對照(執行前寫下)")
    print(f"   大多數是重複 track:重複 {nd / n:.1%} → {'成立' if nd / n > 0.5 else '不成立'}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(
        args=vars(args), rules=dict(DUP_SHARE=DUP_SHARE, WINDOW_S=WINDOW_S, NEAR_IOU=NEAR_IOU),
        n=n, dup=nd, unique=n - nd, dup_at_first=nf, share_bins=dict(bins),
        unique_real_ended_before=sum(t["real_ended_before"] for t in uniq),
        unique_real_started_after=sum(t["real_started_after"] for t in uniq),
        tracks=out_tracks), ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n已存 {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
