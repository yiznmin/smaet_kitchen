"""新 track 出生時,只用框的幾何關係能不能分辨「重複 track」與「真的新人」—— 探索性診斷,非預先登記(2026-09-15)。

## 為什麼

`diag_m4_dup.py` 查出:② 框不準的誤偵 85.8% 是同一個人被多開一條 track,74.5% 在出生當下就已有人在追。
修法可以放在「新 track 出生」這一刻 —— 但那時判斷「是不是同一個人」用的是真值,**實際系統沒有真值**。
這支量:只看新框與既有 track 框的幾何關係,能分到多準。

## 標記(只用來評分,不參與判斷;執行前寫死)

出生幀(`track_events.csv` 的 new_track 幀,自檢等於 tracks.csv 第一列),新 track N 與人框的配對:

  D 重複   N 與某人 g 的框 IoU ≥ 0.1,且同一幀已有**另一條** track 以匈牙利配對配到 g
  N 新人   N 自己配到某人,而且沒有別的 track 配到那個人
  G 不碰人 N 與任何人框 IoU < 0.1
  U 其他   與某人框 IoU ≥ 0.1,但 N 沒配到、也沒有別的 track 配到那個人

## 幾何特徵(不用真值)

同一台鏡頭、同一幀、**出生比 N 早**的 track E:
  包含比例 = max_E 交集(N,E) / N 面積
  重疊     = max_E IoU(N,E)

## 判定目標(執行前寫死)

  可用 = 抓到 D ≥ 70%,且誤殺 N ≤ 5%
  假說 H1:在誤殺 N ≤ 5% 的門檻中,「包含比例」能抓到的 D 多於「重疊」

用法:
    .venv/Scripts/python.exe scripts/diag_m4_birth_geometry.py --root <CHIRLA根> \\
        --tracks-dir results/m4_5cam/cbiou_dump --seqs seq_004 ... --out results/m4_5cam/birth_geometry_cbiou.json
"""
import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from diag_m4_ghosts import NEAR_IOU, load_rows, per_frame_assign   # noqa: E402
from eval_m4m5_chirla import iou, load_gt, match_tracks            # noqa: E402

CONT_THR = (0.5, 0.6, 0.7, 0.8, 0.9, 0.95)
IOU_THR_LIST = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7)
TARGET_D = 0.70
TARGET_N = 0.05
LABELS = ("D 重複", "N 新人", "G 不碰人", "U 其他")


def area(b):
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def inter(a, b):
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    return w * h if w > 0 and h > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--tracks-dir", required=True)
    ap.add_argument("--seqs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    births = []
    for seq in args.seqs:
        gt = load_gt(args.root, seq)
        rows = load_rows(Path(args.tracks_dir) / seq / "tracks.csv")
        track_gt, *_ = match_tracks(gt, rows)
        info, _ = per_frame_assign(gt, rows)

        with open(Path(args.tracks_dir) / seq / "track_events.csv", encoding="utf-8") as f:
            born = {(r["camera_id"], int(r["track_id"])): int(r["video_fid"])
                    for r in csv.DictReader(f) if r["kind"] == "new_track"}
        first = {}
        frame_tracks = defaultdict(list)
        for r in rows:
            k = (r["cam"], r["tid"])
            first[k] = min(first.get(k, r["fid"]), r["fid"])
            frame_tracks[(r["cam"], r["fid"])].append((r["tid"], r["bbox"]))
        if born != first:
            raise SystemExit(f"[FAIL] {seq} new_track 事件幀與 tracks.csv 第一列不一致")

        at = {}                                  # (cam, tid, fid) -> (最大 IoU, 那個人, 配到的人)
        matched_at = defaultdict(dict)           # (cam, fid) -> {人: track}
        for (cam, tid), fr in info.items():
            for fid, bi, bg, mg in fr:
                at[(cam, tid, fid)] = (bi, bg, mg)
                if mg is not None:
                    matched_at[(cam, fid)][mg] = tid

        for (cam, tid), b in born.items():
            box = next(bb for t, bb in frame_tracks[(cam, b)] if t == tid)
            bi, bg, mg = at[(cam, tid, b)]
            other = matched_at[(cam, b)].get(bg) if bg is not None else None
            if bi >= NEAR_IOU and other is not None and other != tid:
                label = LABELS[0]
            elif mg is not None:
                label = LABELS[1]
            elif bi < NEAR_IOU:
                label = LABELS[2]
            else:
                label = LABELS[3]
            existing = [(t, bb) for t, bb in frame_tracks[(cam, b)] if t != tid and first[(cam, t)] < b]
            a = area(box)
            cont = max((inter(box, bb) / a for _t, bb in existing), default=0.0) if a > 0 else 0.0
            ov = max((iou(box, bb) for _t, bb in existing), default=0.0)
            # N 新人裡,同一幀是否有「整體真值為同一人」的舊 track(新框比舊框配得好 → 其實舊框才是多的)
            same_person_old = (label == LABELS[1]
                               and any(track_gt[(cam, t)] == mg for t, _bb in existing))
            births.append(dict(seq=seq, cam=cam, tid=tid, fid=b, label=label, cont=cont, iou=ov,
                               n_existing=len(existing), ghost=track_gt[(cam, tid)] is None,
                               n_same_person_old=same_person_old))

    n = len(births)
    lab = Counter(x["label"] for x in births)
    print(f"新 track 出生共 {n:,}(= M5 要判斷的次數);new_track 事件幀 = tracks.csv 第一列 [OK]")
    print("\n1. 出生時刻的標記(只用來評分)")
    for k in LABELS:
        print(f"   {k:<8} {lab[k]:>5}({lab[k] / n:.1%})")
    nsp = sum(x["n_same_person_old"] for x in births)
    print(f"   ⚠ N 新人中,同一幀已有整體真值為同一人的舊 track:{nsp}(新框配得比舊框好,舊框被判為沒配到)")
    print(f"   出生時同一幀沒有任何較早的 track:{sum(x['n_existing'] == 0 for x in births)}")

    def evaluate(feat, thr):
        hit = Counter(x["label"] for x in births if x[feat] >= thr)
        return {k: hit[k] for k in LABELS}

    print(f"\n2. 規則表(目標:抓到 D ≥ {TARGET_D:.0%} 且誤殺 N ≤ {TARGET_N:.0%})")
    print(f"   {'規則':<16}{'抓到 D':>14}{'誤殺 N':>14}{'擋掉 G':>12}{'擋掉 U':>12}{'達標':>6}")
    table = []
    for feat, name, thrs in (("cont", "包含比例", CONT_THR), ("iou", "重疊 IoU", IOU_THR_LIST)):
        for t in thrs:
            h = evaluate(feat, t)
            rd = h[LABELS[0]] / lab[LABELS[0]] if lab[LABELS[0]] else 0.0
            rn = h[LABELS[1]] / lab[LABELS[1]] if lab[LABELS[1]] else 0.0
            ok = rd >= TARGET_D and rn <= TARGET_N
            table.append(dict(feature=feat, thr=t, hits=h, recall_d=rd, kill_n=rn, ok=ok))
            print(f"   {name + ' ≥ ' + str(t):<16}{h[LABELS[0]]:>6}({rd:>6.1%}){h[LABELS[1]]:>6}({rn:>6.1%})"
                  f"{h[LABELS[2]]:>6}({h[LABELS[2]] / max(lab[LABELS[2]], 1):>5.0%}){h[LABELS[3]]:>6}({h[LABELS[3]] / max(lab[LABELS[3]], 1):>5.0%})"
                  f"{'是' if ok else '':>6}")

    print("\n3. 假說對照(執行前寫下)")
    best = {}
    for feat in ("cont", "iou"):
        ok = [r for r in table if r["feature"] == feat and r["kill_n"] <= TARGET_N]
        best[feat] = max(ok, key=lambda r: r["recall_d"]) if ok else None
    for feat, name in (("cont", "包含比例"), ("iou", "重疊 IoU")):
        b = best[feat]
        print(f"   {name}:誤殺 N ≤ {TARGET_N:.0%} 時最多抓到 D "
              + (f"{b['recall_d']:.1%}(門檻 {b['thr']},誤殺 {b['kill_n']:.1%})" if b else "—(沒有門檻能讓誤殺 ≤ 5%)"))
    rc = best["cont"]["recall_d"] if best["cont"] else -1
    ri = best["iou"]["recall_d"] if best["iou"] else -1
    print(f"   H1 包含比例優於重疊:{'成立' if rc > ri else '不成立'}")
    print(f"   有任何規則達標:{'是' if any(r['ok'] for r in table) else '否'}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(
        args=vars(args), rules=dict(NEAR_IOU=NEAR_IOU, TARGET_D=TARGET_D, TARGET_N=TARGET_N),
        n_births=n, labels={k: lab[k] for k in LABELS}, n_same_person_old=nsp,
        table=table, births=births), ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n已存 {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
