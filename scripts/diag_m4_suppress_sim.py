"""B. 接受取捨:新框與既有框重疊 ≥ T 就先不開新 track —— 離線近似模擬,探索性,非預先登記(2026-09-15)。

## 為什麼

`diag_m4_birth_geometry.py` 查出重疊 ≥ 0.4 最接近目標(抓到重複 58.1%、誤殺真的新人 3.9%)。
但「誤殺」的真實代價還沒量:被擋下的人之後會不會在不重疊時重新出生、少追多久。

## 模擬規則(執行前寫死)

對每條 track,出生幀與同一幀出生較早的 track 最大重疊 ≥ T → 擋下出生;
之後在它存活期間,**第一個「與同一幀其他較早出生的 track 最大重疊 < T」的幀**才出生(延後出生);
整段都 ≥ T → 整條不開。延後出生之前的幀視為沒有被追蹤。

⚠ **這是離線近似**:真的改追蹤器後,被擋下的偵測可能被併進別的 track、或改變之後的關聯,
結果會與這裡不同。要修的話必須正式改程式重跑。

  T ∈ {0.3, 0.4, 0.5}

指標(出生標記讀 `diag_m4_birth_geometry.py`;真值只用來評分):
  D 重複整條不開的比例、N 新人被延後 / 整條不開的數量、
  損失的真人配對幀 ÷ 全部真人配對幀(只算 N / U / G 的 track,D 的幀本來就是多的)、
  M5 少判斷的次數(= 整條不開的 track 數)

假說 HB:T = 0.4 時,D 整條不開 ≥ 50%,且損失的真人配對幀 ≤ 1%。

用法:
    .venv/Scripts/python.exe scripts/diag_m4_suppress_sim.py --root <CHIRLA根> \\
        --tracks-dir results/m4_5cam/cbiou_dump --seqs seq_004 ... \\
        --births-json results/m4_5cam/birth_geometry_cbiou.json --out results/m4_5cam/suppress_sim_cbiou.json
"""
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from diag_m4_ghosts import load_rows, per_frame_assign   # noqa: E402
from eval_m4m5_chirla import iou, load_gt               # noqa: E402

THRS = (0.3, 0.4, 0.5)
LABELS = ("D 重複", "N 新人", "G 不碰人", "U 其他")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--tracks-dir", required=True)
    ap.add_argument("--seqs", nargs="+", required=True)
    ap.add_argument("--births-json", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    label_of = {(b["seq"], b["cam"], b["tid"]): b for b in
                json.loads(Path(args.births_json).read_text(encoding="utf-8"))["births"]}

    # 每條 track:依幀排序的 (幀, 與同幀較早出生 track 的最大重疊, 這一幀有沒有配到人)
    per_track = {}
    total_matched = 0
    for seq in args.seqs:
        gt = load_gt(args.root, seq)
        rows = load_rows(Path(args.tracks_dir) / seq / "tracks.csv")
        info, _ = per_frame_assign(gt, rows)
        first = {}
        frame_tracks = defaultdict(list)
        for r in rows:
            k = (r["cam"], r["tid"])
            first[k] = min(first.get(k, r["fid"]), r["fid"])
            frame_tracks[(r["cam"], r["fid"])].append((r["tid"], r["bbox"]))
        mframe = {(cam, tid, x[0]): x[3] is not None for (cam, tid), fr in info.items() for x in fr}
        seq_tracks = defaultdict(list)
        for (cam, fid), ts in frame_tracks.items():
            for tid, box in ts:
                ov = max((iou(box, bb) for t, bb in ts if t != tid and first[(cam, t)] < first[(cam, tid)]),
                         default=0.0)
                m = mframe[(cam, tid, fid)]
                total_matched += m
                seq_tracks[(cam, tid)].append((fid, ov, m))
        for k, v in seq_tracks.items():
            if (seq,) + k not in label_of:
                raise SystemExit(f"[FAIL] {(seq,) + k} 不在出生紀錄裡")
            per_track[(seq,) + k] = sorted(v)
    if len(per_track) != len(label_of):
        raise SystemExit(f"[FAIL] track 數 {len(per_track)} ≠ 出生紀錄 {len(label_of)}")
    # 自檢:出生幀的重疊要等於 birth_geometry 算的(那邊的「較早」是與出生幀比,這裡是與這條 track 的出生比,出生幀上相同)
    bad = sum(1 for k, v in per_track.items() if abs(v[0][1] - label_of[k]["iou"]) > 1e-9)
    if bad:
        raise SystemExit(f"[FAIL] {bad} 條 track 的出生幀重疊與 birth_geometry 不一致")

    lab = Counter(b["label"] for b in label_of.values())
    print(f"track {len(per_track):,}(與出生紀錄相同、出生幀重疊一致 [OK]);真人配對幀合計 {total_matched:,}")
    out_rows = []
    print(f"\n{'T':>5}{'擋下出生':>9}{'D 整條不開':>14}{'N 延後':>8}{'N 整條不開':>11}"
          f"{'損失真人幀':>16}{'M5 少判斷':>10}")
    for T in THRS:
        blocked, never, delayed = Counter(), Counter(), Counter()
        lost = 0
        for k, v in per_track.items():
            if v[0][1] < T:
                continue
            lb = label_of[k]["label"]
            blocked[lb] += 1
            j = next((i for i, x in enumerate(v) if x[1] < T), None)
            lost_here = sum(x[2] for x in (v if j is None else v[:j]))
            if j is None:
                never[lb] += 1
            else:
                delayed[lb] += 1
            if lb != LABELS[0]:
                lost += lost_here
        rd = never[LABELS[0]] / lab[LABELS[0]]
        out_rows.append(dict(T=T, blocked={x: blocked[x] for x in LABELS}, never={x: never[x] for x in LABELS},
                             delayed={x: delayed[x] for x in LABELS}, d_never_rate=rd,
                             lost_matched_frames=lost, lost_share=lost / total_matched,
                             m5_fewer_decisions=sum(never.values())))
        print(f"{T:>5}{sum(blocked.values()):>9}{never[LABELS[0]]:>6}({rd:>6.1%}){delayed[LABELS[1]]:>8}"
              f"{never[LABELS[1]]:>11}{lost:>8}({lost / total_matched:>6.2%}){sum(never.values()):>10}")

    r4 = next(r for r in out_rows if r["T"] == 0.4)
    print("\n假說對照(執行前寫下)")
    print(f"   HB T=0.4:D 整條不開 {r4['d_never_rate']:.1%}、損失真人幀 {r4['lost_share']:.2%} → "
          f"{'成立' if r4['d_never_rate'] >= 0.5 and r4['lost_share'] <= 0.01 else '不成立'}")
    print("   ⚠ 離線近似:真的改追蹤器後結果會不同")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(args=vars(args), labels={k: lab[k] for k in LABELS},
                                   total_matched_frames=total_matched, table=out_rows),
                              ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n已存 {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
