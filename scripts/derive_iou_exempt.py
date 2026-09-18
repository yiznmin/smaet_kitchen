"""從**推導集**估 F2 的 IoU 豁免門檻 τ(預先登記 M5_三檔互斥網格_20260918 §3)。

## 這支在回答什麼

同鏡頭互斥(F2)會誤傷「偵測器把一個人切成兩塊」的情況 —— 那兩條 track 其實是
同一個人,擋掉第二條只是把靜默的誤併換成看得見的碎裂。兩個框若高度重疊,
它們幾乎不可能是兩個並排站著的人 → 放行。門檻就是這支要估的 τ。

## ⚠ 程序在跑之前就定死了,不得事後改

1. 只用**推導集** seq_000/001/002 —— 依 docs/CHIRLA_M4M5驗證_預先登記_20260903.md
   §4.3,拓撲的每一個參數只能從推導集估,不得回頭用評估集校準。
   **本腳本硬性拒絕推導集以外的序列**(與 chirla_build_crossview.py 同樣的擋法)。
2. τ = 兩組分布的**交會點(等錯誤率點)**,四捨五入到小數兩位。
3. 沒有交會點(兩組完全分開或完全重疊)→ **τ = 0.50**,並照實揭露退回預設。

⚠ **不得**用「讓誤傷降到某個比例」之類的規則選 τ —— 那是拿答案調參數。

## 口徑

- 「同鏡頭衝突」在這裡是**逐幀**定義:同一 video_fid、同一鏡頭、同一 chef_id
  掛著兩條以上的 track_id。
  ⚠ 這與 `run_meta.same_cam_conflicts`(綁定**事件**發生當下計數)**不同口徑**,
    兩個數字不會一樣,不可混用。這裡要的是「同時存在的兩個框長什麼樣」,
    逐幀才量得到。
- 「這條 track 是誰」用 `track_gt`(整段序列 IoU 多數決),與誤併率同一套判定。
- 至少一邊配不到真人(誤偵)的配對**不列入** τ 的估計 —— 那類既不是「同一個人
  被拆開」也不是「兩個人」,拿它調門檻沒有意義。但它的數量會一併報出來。

用法:
    python scripts/derive_iou_exempt.py --run-root results/m5_step3/m5_cbiou_deriv \\
        --track-gt-dir results/m5_step3/m5_cbiou_deriv/track_gt \\
        --out results/levels/iou_exempt.json
"""
import argparse
import csv
import json
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

DERIVATION = ("seq_000", "seq_001", "seq_002")
FALLBACK_TAU = 0.50


def load_track_gt(path):
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return {(k.split("|")[0], int(k.split("|")[1])): v for k, v in raw.items()}


def iou_xyxy(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def collect(run_root, gt_dir, seqs):
    """回傳 (同一個人的 IoU 清單, 兩個人的 IoU 清單, 誤偵配對數)。"""
    same, diff, ghost = [], [], 0
    for seq in seqs:
        tgt = load_track_gt(Path(gt_dir) / f"{seq}.json")
        # (video_fid, camera_id, chef_id) -> [(track_id, bbox)]
        frames = defaultdict(list)
        with open(Path(run_root) / seq / "tracks.csv", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                if not r.get("chef_id") or r["x1"] in ("", None):
                    continue            # 沒綁到 chef，或這一幀沒有框
                box = tuple(float(r[k]) for k in ("x1", "y1", "x2", "y2"))
                frames[(r["video_fid"], r["camera_id"], r["chef_id"])].append(
                    (int(r["track_id"]), box))
        for (_fid, cam, _chef), rows in frames.items():
            if len(rows) < 2:
                continue
            for (ta, ba), (tb, bb) in combinations(sorted(rows), 2):
                ga, gb = tgt.get((cam, ta)), tgt.get((cam, tb))
                if ga is None or gb is None:
                    ghost += 1
                elif ga == gb:
                    same.append(iou_xyxy(ba, bb))
                else:
                    diff.append(iou_xyxy(ba, bb))
    return same, diff, ghost


def pick_tau(same, diff):
    """等錯誤率點。回傳 (tau, 是否退回預設, 掃描表)。"""
    if not same or not diff:
        return FALLBACK_TAU, True, []
    table = []
    for i in range(101):
        tau = i / 100.0
        # 同一個人卻沒被豁免 → 會被錯擋(換成碎裂)
        fn = sum(1 for v in same if v < tau) / len(same)
        # 兩個不同的人卻被豁免 → 該擋沒擋
        fp = sum(1 for v in diff if v >= tau) / len(diff)
        table.append((tau, fn, fp))
    # ⚠ 平手時取**區間中點**,不是最左邊那個。
    #   兩組完全分得開時,整段空隙內 FN 與 FP 都是 0 → 有一整區並列最小。
    #   min() 會回傳最左邊那個,那正好緊貼著「兩個不同的人」那一群,
    #   稍有雜訊就失效(合成測試:同組 0.9 / 異組 0.1 竟取到 τ=0.11)。
    #   取中點才是「交會點」這個詞的本意。
    lo = min(abs(fn - fp) for _t, fn, fp in table)
    tied = [t for t, fn, fp in table if abs(fn - fp) <= lo + 1e-12]
    tau = tied[len(tied) // 2]
    # 落在邊界 = 兩組分不開,門檻沒有意義 → 退回預設並揭露
    fallback = tau <= 0.0 or tau >= 1.0
    return (FALLBACK_TAU if fallback else round(tau, 2)), fallback, table


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root", required=True, help="含 <seq>/tracks.csv")
    ap.add_argument("--track-gt-dir", required=True)
    ap.add_argument("--seqs", nargs="+", default=list(DERIVATION))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    bad = [s for s in args.seqs if s not in DERIVATION]
    if bad:
        raise SystemExit(
            f"拒絕執行:{bad} 不在推導集 {list(DERIVATION)} 內。\n"
            "拓撲的每一個參數只能從推導集估(預先登記 20260903 §4.3)——\n"
            "拿評估集校準會讓後面所有數字失去意義。")

    same, diff, ghost = collect(args.run_root, args.track_gt_dir, args.seqs)
    tau, fallback, table = pick_tau(same, diff)

    print(f"推導集 {args.seqs}")
    print(f"  同鏡頭同時共存的配對:同一個人 {len(same):,}、"
          f"兩個不同的人 {len(diff):,}、至少一邊是誤偵 {ghost:,}(不列入估計)")
    if not same or not diff:
        print("  ⚠ 有一組是空的 → 無法估交會點,退回預設 τ = 0.50")
    else:
        def q(v, p):
            s = sorted(v)
            return s[min(len(s) - 1, int(p * len(s)))]
        print(f"  同一個人的 IoU  中位 {q(same, .5):.3f}  p10 {q(same, .1):.3f}")
        print(f"  兩個人的 IoU    中位 {q(diff, .5):.3f}  p90 {q(diff, .9):.3f}")
        print("\n  掃描(τ, 同一個人被錯擋, 兩個人被錯放):")
        for tau_i, fn, fp in table[::10]:
            print(f"    {tau_i:.2f}   {fn:6.1%}   {fp:6.1%}")

    print(f"\n  **τ = {tau:.2f}**" + ("(⚠ 退回預設:兩組分不開)" if fallback else
                                      "(等錯誤率點)"))

    out = dict(tau=tau, fallback_to_default=fallback, seqs=args.seqs,
               run_root=args.run_root,
               n_same_person=len(same), n_different_person=len(diff),
               n_ghost_pairs=ghost,
               histogram_same={str(round(i / 10, 1)): 0 for i in range(11)},
               histogram_diff={str(round(i / 10, 1)): 0 for i in range(11)},
               sweep=[dict(tau=t, same_person_blocked=round(fn, 4),
                           different_person_exempted=round(fp, 4))
                      for t, fn, fp in table])
    # ⚠ 不可用 `v // 0.1` 分桶 —— 浮點地板除法會把 0.3 丟進 0.2 桶、
    #   把 1.0 丟進 0.9 桶,宣告 11 個桶實際只用到 10 個,各桶數量也不均。
    #   合成測試(0.00~1.00 每 0.01 一點)量到 0.2→11、0.3→9、1.0→0。
    def _bucket(v):
        return str(round(min(max(int(v * 10), 0), 10) / 10, 1))

    for v in same:
        out["histogram_same"][_bucket(v)] += 1
    for v in diff:
        out["histogram_diff"][_bucket(v)] += 1
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(
            json.dumps(out, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8")
        print(f"\n已存 {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
