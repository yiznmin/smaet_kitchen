"""量「一個編號跨時間被幾個不同真人共用」—— 使用者說的「滾雪球」。

## 為什麼需要這支

既有工具只量得到**同一瞬間**的衝突:
`resident_stats()["same_cam_conflicts"]` = 一位 chef 在同一台鏡頭上同時綁著兩條 track。
但實際看到的現象是**跨時間累積**的:`chef 21` 在 seq_026 的綁定,
當下對到 **9 個不同真人**(2 號 23 次、7 號 29 次、9 號 23 次…)。

這兩件事不一樣,而且**同鏡頭互斥(F2)只擋得到前者**。這支把後者量出來,
才能判斷「加上互斥規則」到底能不能阻止編號被越來越多人共用。

## 口徑

- 「這條 track 是誰」用 `track_gt`(`make_track_gt.py` 產出,整段序列 IoU 多數決),
  **與誤併率同一套判定** —— 這樣兩個數字可以並排讀。
- ⚠ 與 `diag_chef_label_check.py` 的**逐幀 IoU** 判定不同口徑,兩者數字不會一樣,不可混用。
- ⚠ chef_id **必須加序列命名空間**:每個序列是獨立的 process,`_next` 都從 1 開始,
  不加的話 seq_004 的 21 號會和 seq_026 的 21 號被當成同一個(2026-09-13 踩過這個坑)。
- 沒有配到真人的 track(誤偵)單獨計數,不計入「幾個真人」。

用法:
    python scripts/diag_chef_sharing.py --run-root results/m5_step3/m5_cbiou \\
        --track-gt-dir results/m5_step3/m5_cbiou/track_gt \\
        --seqs seq_004 seq_006 --label base --out results/levels/chef_sharing_base.json
"""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def load_track_gt(path):
    """{"<camera>|<track_id>": gt_id} → {(camera, track_id): gt_id}。"""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return {(k.split("|")[0], int(k.split("|")[1])): v for k, v in raw.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root", required=True, help="含 <seq>/chef_events.jsonl")
    ap.add_argument("--track-gt-dir", required=True, help="含 <seq>.json")
    ap.add_argument("--seqs", nargs="+", required=True)
    ap.add_argument("--label", default="run")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    people = defaultdict(set)        # (seq, chef) -> {真人}
    binds = Counter()                # (seq, chef) -> 綁定次數
    ghost_binds = Counter()          # (seq, chef) -> 綁到誤偵 track 的次數
    per_seq = {}

    for seq in args.seqs:
        tgt = load_track_gt(Path(args.track_gt_dir) / f"{seq}.json")
        n_ev = n_ghost = 0
        for line in (Path(args.run_root) / seq / "chef_events.jsonl").read_text(
                encoding="utf-8").splitlines():
            if not line.strip():
                continue
            e = json.loads(line)
            key = (seq, e["chef_id"])           # ⚠ 加序列命名空間
            binds[key] += 1
            n_ev += 1
            gid = tgt.get((e["camera_id"], e["track_id"]))
            if gid is None:                     # 誤偵 track
                ghost_binds[key] += 1
                n_ghost += 1
            else:
                people[key].add(gid)
        chefs = {k for k in binds if k[0] == seq}
        multi = {k for k in chefs if len(people[k]) >= 2}
        per_seq[seq] = dict(
            n_bindings=n_ev, n_ghost_bindings=n_ghost, n_chefs=len(chefs),
            n_chefs_multi_person=len(multi),
            max_people_per_chef=max((len(people[k]) for k in chefs), default=0))

    hist = Counter(len(people[k]) for k in binds)          # 幾個真人 -> 幾個編號
    worst = sorted(binds, key=lambda k: (-len(people[k]), -binds[k]))[:5]

    n_chefs = len(binds)
    n_multi = sum(1 for k in binds if len(people[k]) >= 2)
    # 「被共用的編號吃掉多少綁定」—— 只數編號個數會低估影響,那些編號往往也是最常被綁的
    binds_in_multi = sum(binds[k] for k in binds if len(people[k]) >= 2)
    tot_binds = sum(binds.values())

    print(f"[{args.label}] {args.run_root}")
    print(f"  編號總數 {n_chefs}、綁定總數 {tot_binds:,}")
    print(f"  **對到 ≥2 個真人的編號:{n_multi}({n_multi / n_chefs:.1%})**,"
          f"它們吃掉 {binds_in_multi:,} 次綁定({binds_in_multi / tot_binds:.1%})")
    print(f"  一個編號最多對到 {max((len(people[k]) for k in binds), default=0)} 個不同真人")
    print("\n  分布(一個編號對到幾個真人 → 幾個編號):")
    for n in sorted(hist):
        tag = "  ← 只對到誤偵" if n == 0 else ""
        print(f"    {n:>2} 個真人：{hist[n]:>4} 個編號{tag}")
    print("\n  前五名:")
    for k in worst:
        print(f"    {k[0]} chef {k[1]:<4} → {len(people[k])} 個真人"
              f"(綁定 {binds[k]} 次,其中誤偵 {ghost_binds[k]} 次)"
              f"  真人={sorted(people[k])}")

    out = dict(label=args.label, run_root=args.run_root, seqs=args.seqs,
               n_chefs=n_chefs, n_bindings=tot_binds,
               n_chefs_multi_person=n_multi,
               share_chefs_multi_person=round(n_multi / n_chefs, 4) if n_chefs else None,
               bindings_in_multi_person_chefs=binds_in_multi,
               share_bindings_in_multi_person_chefs=(
                   round(binds_in_multi / tot_binds, 4) if tot_binds else None),
               max_people_per_chef=max((len(people[k]) for k in binds), default=0),
               histogram={str(n): c for n, c in sorted(hist.items())},
               worst=[dict(seq=k[0], chef_id=k[1], n_people=len(people[k]),
                           n_bindings=binds[k], n_ghost_bindings=ghost_binds[k],
                           people=sorted(people[k])) for k in worst],
               per_seq=per_seq)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2),
                                  encoding="utf-8")
        print(f"\n已存 {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
