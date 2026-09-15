"""M5 四層歸因彙總:把 `chef_events.jsonl` 的診斷欄位整理成可比較的表。

2026-09-13 的根因分解(`docs/M5_根因分解_20260913.md` §4)用的是對話裡的臨時程式,
**沒有提交腳本** —— 與 9/14 §8.8 的 72.4% 同一個問題。步驟 3 的判準要靠這些數字,
所以補成正式工具。

輸入是 `m5_track_video.py --track-gt` 跑出來的目錄(每筆決策帶 `layer` 等欄位)。輸出三件事:

  1. 四層分解   L1 誤偵 / L0 初登場 / L2 候選集 / L3 評分 / L4 門檻 / OK
  2. 交叉核對   層 × 指標結果(正確 / 誤併 / 碎裂),並**斷言逐位重現** `metrics.summarize`
                —— 對不上就代表分解與指標口徑分歧,整份結果不能用
  3. L2 的成因  正確的 chef 在決策那一刻:
                · 不在任何畫面裡            → 候選路徑 (c) 的目標(9/13:391)
                · 在不重疊的別台鏡頭上       → 拓撲 overlapping 的目標(9/13:550)
                · 就在本鏡頭上              → 物理上不該綁(9/13:71)
                · 只在重疊鏡頭上            → 路徑 (b) 本該收(9/13:0)

⚠ 「此刻在哪台鏡頭」取自 `tracks.csv` 同一個 `loop_i` 的 `chef_id`。runner 在同一迴圈裡
  先寫 tracks.csv(①心跳)、後處理 new_track 事件(②),所以它是**決策之前**的狀態 ——
  9/13 的 391 / 550 / 71 就是用這個口徑算的。
⚠ id 一律加序列命名空間(`<seq>:<id>`),與 `eval_m4m5_chirla.build_records` 修正後一致。

驗收(必須先過):
    python scripts/diag_m5_layers.py --run-root results/chirla_diag/base \\
        --seqs seq_004 seq_006 seq_007 seq_020 seq_024 seq_025 seq_026 \\
        --topology configs/fix_grid/base.yaml
    應重現 9/13:L2 1,012 / L3 238 / L4 48 / OK 1,163 / L1 1,216 / L0 59;
    正確 257 / 誤併 1,433 / 碎裂 771;L2 成因 550 / 391 / 71。
"""
import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

LAYERS = ["L1_ghost", "L0_first", "L2_not_in_candidates", "L3_not_top1",
          "L4_below_threshold", "OK"]
ATTRIB = ["L2_not_in_candidates", "L3_not_top1", "L4_below_threshold", "OK"]
OUTCOMES = ["正確", "誤併", "碎裂"]
L2_CAUSES = ["不在任何畫面", "在不重疊的別台鏡頭", "就在本鏡頭", "只在重疊鏡頭", "無上一次綁定"]


def classify_outcomes(rows, seq):
    """逐筆判定正確 / 誤併 / 碎裂。

    ⚠ 刻意與 `m5_reid.metrics.binding_outcomes` 逐條相同:owner 是 pred 的多數票歸屬;
      轉場才計入;沒綁 → 碎裂;owner 不是自己 → 誤併;綁回上次那個 → 正確;其餘 → 誤併。
      main() 會拿總數與 `metrics.summarize` 對帳,口徑一旦分歧就中止。
    """
    recs = [(f"{seq}:{r['gt_id']}", f"{seq}:{r['chef_id']}", bool(r["matched"]), r)
            for r in sorted(rows, key=lambda x: (x["video_fid"], x["camera_id"]))
            if r.get("gt_id") is not None]
    votes = defaultdict(Counter)
    for gt, pred, _m, _r in recs:
        votes[pred][gt] += 1
    owner = {p: v.most_common(1)[0][0] for p, v in votes.items()}
    seen, last, out, metric_recs = set(), {}, [], []
    for gt, pred, m, r in recs:
        is_tr = gt in seen
        metric_recs.append((gt, pred, m, is_tr))
        if is_tr:
            if not m:
                o = "碎裂"
            elif owner.get(pred) != gt:
                o = "誤併"
            elif last.get(gt) == pred:
                o = "正確"
            else:
                o = "誤併"
            out.append((r["layer"], o))
        seen.add(gt)
        last[gt] = pred
    return out, metric_recs


def l2_causes(rows, tracks_csv, topo):
    vis = defaultdict(set)                          # loop_i -> {(chef_id, camera)}
    with open(tracks_csv, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r.get("chef_id"):
                vis[int(r["loop_i"])].add((int(r["chef_id"]), r["camera_id"]))
    c = Counter()
    for r in rows:
        if r.get("layer") != "L2_not_in_candidates":
            continue
        want, cam = r.get("want_chef_id"), r["camera_id"]
        if want is None:
            c["無上一次綁定"] += 1
            continue
        on = {cm for cid, cm in vis.get(int(r["loop_i"]), ()) if cid == want}
        if not on:
            c["不在任何畫面"] += 1
        elif cam in on:
            c["就在本鏡頭"] += 1
        elif all(topo.is_overlapping(cm, cam) for cm in on):
            c["只在重疊鏡頭"] += 1
        else:
            c["在不重疊的別台鏡頭"] += 1
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root", required=True, help="內含 <seq>/chef_events.jsonl 與 tracks.csv")
    ap.add_argument("--seqs", nargs="+", required=True)
    ap.add_argument("--topology", required=True, help="判斷鏡頭是否重疊;要與該次 M5 執行用的相同")
    ap.add_argument("--label", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from m5_reid import metrics
    from m5_reid.spatiotemporal import CameraTopology
    topo = CameraTopology.from_yaml(args.topology)

    layers, cross, causes, all_metric = Counter(), Counter(), Counter(), []
    per_seq = {}
    for seq in args.seqs:
        d = Path(args.run_root) / seq
        rows = [json.loads(l) for l in
                (d / "chef_events.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
        if rows and "layer" not in rows[0]:
            raise SystemExit(f"{d} 的 chef_events 沒有 layer 欄位 —— 這次 M5 沒有給 --track-gt")
        lc = Counter(r["layer"] for r in rows)
        oc, mrecs = classify_outcomes(rows, seq)
        cc = l2_causes(rows, d / "tracks.csv", topo)
        layers += lc
        cross += Counter(oc)
        causes += cc
        all_metric += mrecs
        per_seq[seq] = dict(n_decisions=len(rows), layers=dict(lc), l2_causes=dict(cc))

    n = sum(layers.values())
    att = sum(layers[k] for k in ATTRIB)
    out_tot = Counter()
    for (_l, o), v in cross.items():
        out_tot[o] += v

    # ── 與 metrics.summarize 對帳(口徑分歧就中止)──────────────────────
    s = metrics.summarize(all_metric, expected_headcount=len({g for g, _, _, _ in all_metric}))
    n_tr = s["n_transitions"]
    # ⚠ summarize 回傳的比率**已四捨五入到小數 4 位**(0.1044),所以要用同樣的位數比。
    #   第一版用未四捨五入的值配 1e-9 容許誤差,把完全對得上的 257/2461 = 0.104429 判成 FAIL。
    checks = [("可歸因決策 = n_transitions", att == n_tr, f"{att} vs {n_tr}")]
    for name, key, o in (("正確率", "p_correct", "正確"), ("誤併率", "p_false_merge", "誤併"),
                         ("碎裂率", "p_break", "碎裂")):
        ours = out_tot[o] / n_tr
        checks.append((name, round(ours, 4) == s[key],
                       f"{out_tot[o]:,} / {n_tr:,} = {ours:.6f} → {round(ours, 4)} vs {s[key]}"))
    label = args.label or Path(args.run_root).name
    print(f"[{label}] 綁定決策 {n:,}(可歸因 {att:,})\n")
    print("  自檢:與 metrics.summarize 對帳")
    for name, ok, msg in checks:
        print(f"    [{'PASS' if ok else 'FAIL'}] {name}  {msg}")
    if not all(ok for _n, ok, _m in checks):
        print("\n  口徑分歧,不印分解。")
        return 2

    print(f"\n  {'層':<24}{'次數':>8}{'佔全部':>9}{'佔可歸因':>10}")
    for k in LAYERS:
        share = f"{layers[k] / att:>9.1%}" if k in ATTRIB else f"{'—':>10}"
        print(f"  {k:<24}{layers[k]:>8,}{layers[k] / n:>9.1%}{share}")

    print(f"\n  {'層 × 結果':<24}" + "".join(f"{o:>8}" for o in OUTCOMES) + f"{'合計':>8}")
    for k in ATTRIB:
        r = [cross[(k, o)] for o in OUTCOMES]
        print(f"  {k:<24}" + "".join(f"{v:>8,}" for v in r) + f"{sum(r):>8,}")
    print(f"  {'合計':<24}" + "".join(f"{out_tot[o]:>8,}" for o in OUTCOMES) + f"{att:>8,}")

    l2 = layers["L2_not_in_candidates"]
    print(f"\n  L2 的成因(共 {l2:,})")
    for k in L2_CAUSES:
        if causes[k] or k != "無上一次綁定":
            print(f"    {k:<18}{causes[k]:>6,}  ({causes[k] / l2 if l2 else 0:.1%})")

    summary = dict(label=label, run_root=args.run_root, seqs=args.seqs, topology=args.topology,
                   n_decisions=n, n_attributable=att, layers=dict(layers),
                   layer_share_attributable={k: layers[k] / att for k in ATTRIB},
                   cross={f"{k}|{o}": cross[(k, o)] for k in ATTRIB for o in OUTCOMES},
                   outcomes=dict(out_tot), l2_causes=dict(causes),
                   metrics=dict(idf1=s["idf1"], p_correct=s["p_correct"],
                                p_false_merge=s["p_false_merge"], p_break=s["p_break"],
                                n_transitions=n_tr),
                   per_seq=per_seq)
    out = Path(args.out or ROOT / "results" / "m5_layers" / f"{label}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已存 {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
