"""把「同一瞬間一個編號綁著兩條 track」的衝突,分成**誤傷**與**該擋**兩類。

## 為什麼需要這支

`same_cam_conflicts`(交付這批資料 2,056 次)只是一個總數,它不區分:

  A. **同一個人被拆成兩個框** —— 偵測器把一個人切成兩塊(遮擋、反光、半身),
     M4 給了兩個 track_id,但**它們是同一個人**。這時候綁到同一個編號是**對的**,
     F2 互斥會把第二條擋掉、開一個新編號 → **碎裂**。這是誤傷。
  B. **真的是兩個不同的人** —— 這時候綁到同一個編號是**錯的**,F2 該擋。

「F2 會誤傷」這句話能不能成立,完全取決於 A 佔多少。這支把它數出來。

順帶量第二件事:**跨鏡頭同一瞬間一個編號出現在兩台鏡頭上**,並按鏡頭對分三檔
(見 `pair_tier`)—— 因為**三檔適用的規則完全不同**,混在一起數會得到錯的結論。

## ⚠ 一個踩過的坑:`overlapping` 清單不等於「看得到彼此」

拓撲 `overlapping` 的判準是「**兩邊的框都 ≥50×120**」,也就是**適不適合做幾何比對**,
不是「看不看得到同一個人」。2026-09-18 我誤把「不在 overlapping 裡」當成
「同時出現是物理上不可能」,據此宣稱有 2,658 次不可能的綁定 —— **那是錯的**:
`camera_1` 不在任何 overlapping 對裡,但它與 `camera_3` 實際共現 **116,224 次**
(透過標記板旁的縫隙望進去)。真正從未共現的只有 5 對。
→ 「同時出現不可能」只對 `COOCCUR` 查得 0 的那幾對成立。

## 口徑與限制

- 「這條 track 是誰」用 `track_gt`(整段序列 IoU 多數決),與誤併率同一套判定,
  ⚠ 與 `diag_chef_label_check.py` 的逐幀 IoU 不同口徑,不可混用。
- ⚠ 整段多數決會低估 A:一條 track 若前半是甲後半是乙,只會被記成其中一個。
- ⚠ 分類**必須與集合迭代順序無關**。第一版踩過:同鏡頭有多條夥伴時依雜湊順序
  挑一條來比,同一份資料兩次執行得到不同分類(總數仍對)。現在改成看所有夥伴。
- ⚠ **重放不是原始計數**。原始衝突在 `identity_st.py:306` 綁定當下計數,
  `chef.track_ids` 只在 `removed` 時剪除(見 identity_st.py:91 註解)。
  這裡照同樣規則重放,**重放總數必須等於 run_meta.same_cam_conflicts**,
  對不上就代表重建錯了,下面的分類全部不可信 —— 所以先印這個對照。
- ⚠ `COOCCUR` 抄自全 10 序列的實測,**含評估集** → 拿它當規則有資訊洩漏,
  要在報告裡照實揭露(與拓撲 `overlapping` 同樣的處置)。
- 框對框 IoU **算不出來**:`tracks.csv` 被 `.gitignore:102` 排除,本機沒有副本,
  但**遠端有** —— 第一檔的 IoU 門檻要在遠端定。

用法:
    python scripts/diag_conflict_kinds.py --run-root results/m5_step3/m5_cbiou \\
        --track-gt-dir results/m5_step3/m5_cbiou/track_gt \\
        --topology configs/fix_grid/base.yaml \\
        --seqs seq_004 seq_006 --label base --out results/levels/conflict_kinds_base.json
"""
import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import yaml


# ⚠ 真值共現次數,抄自 docs/CHIRLA_鏡頭佈局實測_20260903.md 的實測表(全 10 序列)。
#   表上沒有的 5 對(1-6、1-7、2-6、3-6、4-7)= 整個資料集從未同時看到同一個人。
COOCCUR = {
    ("camera_2", "camera_3"): 148250, ("camera_1", "camera_3"): 116224,
    ("camera_1", "camera_2"): 58333, ("camera_2", "camera_5"): 14937,
    ("camera_4", "camera_5"): 10742, ("camera_6", "camera_7"): 6870,
    ("camera_3", "camera_4"): 6375, ("camera_1", "camera_5"): 5109,
    ("camera_3", "camera_5"): 3253, ("camera_2", "camera_4"): 1336,
    ("camera_5", "camera_6"): 738, ("camera_5", "camera_7"): 432,
    ("camera_1", "camera_4"): 415, ("camera_4", "camera_6"): 44,
    ("camera_3", "camera_7"): 3, ("camera_2", "camera_7"): 3,
}

TIER_NAME = {
    "same_space": "同一空間（同時出現合法，只能靠 F4 腳點一致性分辨）",
    "doorway": "門口互看（看得到但框小，不可無條件擋）",
    "never": "**從未共現**（同時出現物理上不可能 → 可無條件互斥）",
}


def load_track_gt(path):
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return {(k.split("|")[0], int(k.split("|")[1])): v for k, v in raw.items()}


def load_overlapping(path):
    topo = yaml.safe_load(Path(path).read_text(encoding="utf-8"))["camera_topology"]
    return {tuple(sorted(p)) for p in (topo.get("overlapping") or [])}


def pair_tier(pair, overlapping):
    """把鏡頭對分成三檔 —— 三種不同的規則適用範圍。"""
    if pair in overlapping:
        return "same_space"
    return "never" if COOCCUR.get(pair, 0) == 0 else "doorway"


def load_removals(path):
    """loop_i -> [(camera_id, track_id)]，只取 removed（track_ids 的剪除時機）。"""
    out = defaultdict(list)
    with open(path, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["kind"] == "removed":
                out[int(r["loop_i"])].append((r["camera_id"], int(r["track_id"])))
    return out


def classify(a, b):
    """兩條 track 的真值身份 → 類別。"""
    if a is None or b is None:
        return "ghost"          # 至少一邊沒配到真人 → 誤偵,兩種說法都不成立
    return "same_person" if a == b else "different_person"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root", required=True)
    ap.add_argument("--track-gt-dir", required=True)
    ap.add_argument("--topology", default="configs/fix_grid/base.yaml")
    ap.add_argument("--seqs", nargs="+", required=True)
    ap.add_argument("--label", default="run")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    overlapping = load_overlapping(args.topology)
    same_cam = Counter()                 # 類別 -> 次數
    cross_cam = Counter()                # (檔位, 類別) -> 次數
    pair_counts = Counter()              # (cam_a, cam_b) -> 次數
    per_seq = {}
    worst = Counter()                    # (seq, chef) -> 同鏡頭衝突次數
    recon_ok = True

    for seq in args.seqs:
        d = Path(args.run_root) / seq
        tgt = load_track_gt(Path(args.track_gt_dir) / f"{seq}.json")
        removals = load_removals(d / "track_events.csv")
        meta = json.loads((d / "run_meta.json").read_text(encoding="utf-8"))
        recorded = meta["resident_final"]["same_cam_conflicts"]

        held = defaultdict(set)          # chef -> {(cam, track)}
        n_conf = 0
        cur_loop = None

        for line in (d / "chef_events.jsonl").read_text(
                encoding="utf-8").splitlines():
            if not line.strip():
                continue
            e = json.loads(line)
            loop = e["loop_i"]
            if cur_loop is not None and loop != cur_loop:
                # 進到新的一圈之前，先套用上一圈的移除（與執行期同序）
                for lp in range(cur_loop, loop):
                    for cam, tid in removals.get(lp, []):
                        for s in held.values():
                            s.discard((cam, tid))
                cur_loop = loop
            elif cur_loop is None:
                cur_loop = loop

            chef, cam, tid = e["chef_id"], e["camera_id"], e["track_id"]
            mine = held[chef]
            gid_new = tgt.get((cam, tid))

            # ── 同鏡頭：這個編號在本台鏡頭上是不是已經綁著別條 ──
            # ⚠ mine 是 set。第一版寫成 `for c, t in mine: ... break`，於是同鏡頭有
            #   多條夥伴時會依**雜湊順序**挑到不同那條 → 同一份資料兩次執行分類不同
            #   （2026-09-18 實測 97/905/1054 vs 80/909/1067，總數都對但分類會漂）。
            #   改成看**所有**同鏡頭夥伴並排序，結果與迭代順序無關。
            #   語意：F2 一擋就是整筆綁定不成立，所以「只要有任一條夥伴其實是同一個人，
            #   擋下去就是誤傷」—— 取最寬鬆的那一類，不會低估誤傷。
            partners = sorted(t for c, t in mine if c == cam and t != tid)
            if partners:
                n_conf += 1          # 與 identity_st.py:306 的 any(...) 同語意，只計一次
                worst[(seq, chef)] += 1
                kinds = {classify(gid_new, tgt.get((cam, t))) for t in partners}
                same_cam["same_person" if "same_person" in kinds else
                         "different_person" if "different_person" in kinds
                         else "ghost"] += 1

            # ── 跨鏡頭：這個編號同時還在哪些別台鏡頭上 ──
            for c, t in mine:
                if c == cam:
                    continue
                key = tuple(sorted((cam, c)))
                pair_counts[key] += 1
                cross_cam[(pair_tier(key, overlapping),
                           classify(gid_new, tgt.get((c, t))))] += 1

            mine.add((cam, tid))

        per_seq[seq] = dict(recorded=recorded, replayed=n_conf, match=(recorded == n_conf))
        if recorded != n_conf:
            recon_ok = False

    # ── 先印重建對照，不過就不要看下面 ──
    print(f"[{args.label}] 重放對照（必須全部相同，否則下面的分類不可信）")
    for seq, v in per_seq.items():
        flag = "OK" if v["match"] else "**對不上**"
        print(f"  {seq}  run_meta={v['recorded']:>5}  重放={v['replayed']:>5}  {flag}")
    tot_rec = sum(v["recorded"] for v in per_seq.values())
    tot_rep = sum(v["replayed"] for v in per_seq.values())
    print(f"  合計   run_meta={tot_rec:,}  重放={tot_rep:,}")
    if not recon_ok:
        print("\n⚠ 重建與執行期不一致，分類結果**不可採信**。")

    names = {"same_person": "同一個人被拆成兩個框 → F2 會**誤傷**（換成碎裂）",
             "different_person": "真的是兩個不同的人 → F2 **該擋**",
             "ghost": "至少一邊是誤偵（沒配到真人）"}

    n_sc = sum(same_cam.values())
    print(f"\n── 同一台鏡頭、同一瞬間、同一個編號綁兩條 track（{n_sc:,} 次）──")
    for k in ("same_person", "different_person", "ghost"):
        v = same_cam[k]
        print(f"  {v:>6,}  {v / n_sc:>6.1%}  {names[k]}" if n_sc else f"  {v}  {names[k]}")

    n_cc = sum(cross_cam.values())
    print(f"\n── 跨鏡頭、同一瞬間、同一個編號在兩台鏡頭上（{n_cc:,} 次）──")
    for tier in ("same_space", "doorway", "never"):
        sub = {k[1]: v for k, v in cross_cam.items() if k[0] == tier}
        s = sum(sub.values())
        print(f"\n  {TIER_NAME[tier]}：{s:,} 次"
              + (f"（{s / n_cc:.1%}）" if n_cc else ""))
        for k in ("same_person", "different_person", "ghost"):
            print(f"      {sub.get(k, 0):>6,}  {names[k].split(' → ')[0]}")

    # ⚠ most_common() 平手時照插入順序，而插入順序來自集合迭代 → 檔案位元組會漂。
    #   值本身不受影響，但「同一份輸入產出同一個檔案」是驗證的前提，所以明確定序。
    pairs_sorted = sorted(pair_counts.items(), key=lambda kv: (-kv[1], kv[0]))
    worst_sorted = sorted(worst.items(), key=lambda kv: (-kv[1], kv[0]))[:5]

    print("\n── 逐鏡頭對（次數、真值共現、檔位）──")
    for pair, n in pairs_sorted:
        tier = pair_tier(pair, overlapping)
        print(f"    {pair[0]}+{pair[1]}  {n:>6,} 次   真值共現 "
              f"{COOCCUR.get(pair, 0):>7,}   {tier}")

    print("\n── 同鏡頭衝突最多的五個編號 ──")
    for (seq, chef), n in worst_sorted:
        print(f"    {seq} chef {chef:<4} → {n} 次")

    out = dict(label=args.label, run_root=args.run_root, topology=args.topology,
               reconstruction_ok=recon_ok, per_seq=per_seq,
               same_camera={k: same_cam[k] for k in
                            ("same_person", "different_person", "ghost")},
               same_camera_total=n_sc,
               cross_camera={f"{tier}|{k}": v for (tier, k), v in cross_cam.items()},
               cross_camera_total=n_cc,
               cross_camera_by_tier={
                   tier: sum(v for k, v in cross_cam.items() if k[0] == tier)
                   for tier in ("same_space", "doorway", "never")},
               pairs=[dict(pair=list(p), n=n, gt_cooccurrence=COOCCUR.get(p, 0),
                           tier=pair_tier(p, overlapping))
                      for p, n in pairs_sorted],
               worst=[dict(seq=s, chef_id=c, n_conflicts=n)
                      for (s, c), n in worst_sorted])
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        # sort_keys：讓同一份輸入永遠產出逐位相同的檔案（見上面的定序說明）。
        Path(args.out).write_text(
            json.dumps(out, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8")
        print(f"\n已存 {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
