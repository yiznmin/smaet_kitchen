"""M5 實際在用什麼規則綁定身份 —— 探索性診斷(2026-09-15)。

步驟 3 顯示 M4 換成 cbiou 後,M5 的判斷次數少了、特定失敗少了,但認對的次數沒增加。
這支回答「為什麼 M4 改善了,M5 仍綁不對」,每一段都只讀既有輸出:
`m5_track_video.py --track-gt` 的 chef_events.jsonl(帶 layer 等診斷欄位)、tracks.csv、run_meta.json。

  A 結果 × 層                 誤併、碎裂各來自哪一層(口徑與 metrics.binding_outcomes 相同)
  B 誤併的源頭                 (真人 g, 身份 c) 且 g 不是 c 的多數擁有者:g 第一次被綁到 c 那個判斷
                              是哪一層、分數類型、是否同分、c 是否由誤偵開出;以後續綁定次數加權
  C 誤偵開出的身份
  D 判斷當下的候選組成          沒有候選 / 第一名是常數重疊分數 / 第一名是非常數分數
  E 成功綁定的分數類型
  F L3(正確的人在候選裡卻輸了)的組成
  G 一個人第一次出現時被綁到既有身份的比例
  H 認對的判斷中與第二名同分的比例

⚠ 「常數重疊分數」= `overlap_llr − ln(k) + app_lr.llr(0)`,k 是重疊鏡頭上的人數。
  **只在 embedder=none 時成立**(外觀 cosine 恆為 0),所以 run_meta 的 embedder 不是 none 就中止。
  常數由拓撲設定計算,不寫死。
⚠ B 的「對方此刻可見性」只是 tracks.csv 在決策那一迴圈的可見性,**不等於候選路徑** ——
  重疊路徑(identity_st._score_candidates 的 (b))看的是廚師名下的 track 清單,
  而 track 要到 removed 才從清單剪掉,跟丟中的 track 也算數。2026-09-15 的查證中,
  80 次「對方不在畫面卻拿常數分數」只有 42 次找得到跟丟中的 track,38 次未解釋,所以不據以下結論。

用法:
    python scripts/diag_m5_binding_rules.py --run-root results/m5_step3/m5_cbiou \\
        --topology configs/fix_grid/base.yaml
"""
import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

EVAL_SEQS = ["seq_004", "seq_006", "seq_007", "seq_020", "seq_024", "seq_025", "seq_026"]
# 常數重疊分數裡的 k(identity_st 的 overlap_pool)是**重疊鏡頭上掛著的身份條目數**,
# 包含誤偵開出的身份與跟丟未移除的 track,**不是真人數**。
# ⚠ 第二版曾以「CHIRLA 單台最多約 9 人」為由把上限設 20 —— 前提錯了:實測 k 到 20 以上,
#   而且當時「非常數分數最高 0.762」恰好就是 k = 21 的常數分數(3.80625 − ln 21 = 0.7617)。
# 判定改在分數空間比對(見 const_k),上限只用來限制假命中機率。
K_MAX = 60
# 分數寫檔時四捨五入到小數 4 位,真的常數分數與公式相差 ≤ 0.00005。
# 窗寬 ±0.0001、k ≤ 60 時,連續分布的分數碰巧落入窗內的機率 < 1%。
SCORE_TOL = 1e-4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root", required=True)
    ap.add_argument("--seqs", nargs="+", default=EVAL_SEQS)
    ap.add_argument("--topology", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from m5_reid.spatiotemporal import CameraTopology
    topo = CameraTopology.from_yaml(args.topology)
    const = float(topo.fusion["overlap_llr"]) + topo.app_lr.llr(0.0)
    run = Path(args.run_root)

    metas = [json.loads((run / s / "run_meta.json").read_text(encoding="utf-8")) for s in args.seqs]
    if {m.get("embedder") for m in metas} != {"none"}:
        raise SystemExit("本診斷的「常數重疊分數」只在 embedder=none 時成立")
    thr_set = {m.get("llr_threshold") for m in metas}
    if len(thr_set) != 1:
        raise SystemExit(f"各序列門檻不一致:{thr_set}")
    thr = thr_set.pop()

    def const_k(score):
        """分數符合 `const − ln k`(k 為 1~K_MAX 的整數)就回傳 k,否則 None。

        ⚠ 第一版在 k 空間比對(|k − round(k)| < 0.01、無上限):非常數分數代回去的 k 常在一百以上,
          ±0.01 的窗在那裡相當寬,約 3% 的非常數分數碰巧被判成常數;寫死常數 3.8063 與精確值
          3.80625… 的差異又讓 18 次判斷換了分類。改在分數空間比對,窗寬固定為 ±SCORE_TOL。
        ⚠ 「所有綁定都來自常數分數」**不靠這個函式證明**:轉場路徑的理論上限全部低於門檻
          (diag_m5_score_ceiling.py),重疊路徑在無地面校正、F4 關閉時的分數就是這個公式。
          這裡的分類只用於 D、F 段的描述性組成。
        """
        if score is None:
            return None
        r = round(math.exp(const - score))
        if not 1 <= r <= K_MAX:
            return None
        return r if abs(score - (const - math.log(r))) <= SCORE_TOL else None

    A, C, D, Dm, E, F, G, H = (Counter() for _ in range(8))
    B_pairs, B_vol, B_dim = Counter(), Counter(), Counter()
    nonconst_top1, kmax = [], 0

    for seq in args.seqs:
        d = run / seq
        rows = [json.loads(l) for l in (d / "chef_events.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
        if rows and "layer" not in rows[0]:
            raise SystemExit(f"{d} 沒有 layer 欄位 —— 這次 M5 沒有給 --track-gt")
        rows.sort(key=lambda r: (r["video_fid"], r["camera_id"]))
        vis = defaultdict(lambda: defaultdict(set))
        with open(d / "tracks.csv", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r["chef_id"]:
                    vis[int(r["loop_i"])][int(r["chef_id"])].add(r["camera_id"])

        recs = [r for r in rows if r.get("gt_id") is not None]
        votes = defaultdict(Counter)
        for r in recs:
            votes[r["chef_id"]][r["gt_id"]] += 1
        owner = {c: v.most_common(1)[0][0] for c, v in votes.items()}

        seen, last = set(), {}
        for r in recs:
            g, c, m = r["gt_id"], r["chef_id"], bool(r["matched"])
            if g in seen:
                o = "碎裂" if not m else ("誤併" if owner.get(c) != g else ("正確" if last.get(g) == c else "誤併"))
                A[(r["layer"], o)] += 1
            seen.add(g)
            last[g] = c

        founder = {}
        for r in rows:
            founder.setdefault(r["chef_id"], "ghost" if r.get("gt_id") is None else "real")
        ghost_founded = {c for c, f in founder.items() if f == "ghost"}
        C["誤偵開出的身份"] += len(ghost_founded)
        C["其中後來有真人綁入"] += len({r["chef_id"] for r in recs if r["chef_id"] in ghost_founded})
        C["真人綁入誤偵開出的身份(次數)"] += sum(1 for r in recs if r["chef_id"] in ghost_founded)

        first, vol = {}, Counter()
        for r in recs:
            g, c = r["gt_id"], r["chef_id"]
            if owner.get(c) == g:
                continue
            vol[(g, c)] += 1
            first.setdefault((g, c), r)
        for (g, c), r in first.items():
            B_pairs[r["layer"]] += 1
            B_vol[r["layer"]] += vol[(g, c)]
            on = vis[int(r["loop_i"])].get(c, set())
            cam = r["camera_id"]
            if not r["matched"]:
                v = "此配對的第一筆是 c 的創立"
            elif any(cm != cam and topo.is_overlapping(cm, cam) for cm in on):
                v = "c 此刻在重疊鏡頭上可見"
            elif not on:
                v = "c 此刻不在任何畫面"
            elif cam in on:
                v = "c 此刻就在本鏡頭上"
            else:
                v = "c 此刻只在不重疊的鏡頭上"
            k = const_k(r.get("score"))
            sk = ("—" if not r["matched"] else (f"常數重疊分數 k={k}" if k else "非常數分數"))
            tie = ("單一候選" if (r.get("n_cands") or 0) < 2 else
                   ("與第二名同分" if r.get("margin_1_2") == 0 else "第一名分數領先"))
            fb = "c 由誤偵開出" if c in ghost_founded else "c 由真人開出"
            for dim, val in (("對方此刻可見性", v), ("分數類型", sk), ("同分", tie), ("創立者", fb)):
                B_dim[(dim, val)] += vol[(g, c)]

        for r in rows:
            # ⚠ D 段用 runner 對**每一筆**都寫的 n_candidates / score,不用 --track-gt 的 n_cands / top1_score。
            #   誤偵決策(L1_ghost)的診斷欄位只有 gt_id 與 layer —— 第一版把讀不到的 n_cands 當 0,
            #   1,113 次誤偵決策全被算成「沒有任何候選」,組成表因此錯誤,而且與 E 段
            #   「成功綁定 2,286 次」自相矛盾(有候選的判斷最多才 2,111 次)。
            #   score:綁定時是第一名分數;沒綁時也是第一名分數,只有完全沒候選才寫 0.0。
            n = r.get("n_candidates") or 0
            t = r.get("score")
            grp = "誤偵決策" if r.get("gt_id") is None else "真人決策"
            if n == 0:
                D[(grp, "沒有任何候選")] += 1
                Dm[(grp, "沒有任何候選")] += bool(r["matched"])
            elif const_k(t) is not None:
                D[(grp, "第一名是常數重疊分數")] += 1
                Dm[(grp, "第一名是常數重疊分數")] += bool(r["matched"])
            else:
                D[(grp, "第一名是非常數分數")] += 1
                Dm[(grp, "第一名是非常數分數")] += bool(r["matched"])
                nonconst_top1.append((t, bool(r["matched"])))
            if r["matched"]:
                k = const_k(r.get("score"))
                E["常數重疊分數" if k else "非常數分數"] += 1
                if k:
                    kmax = max(kmax, k)
            if r.get("layer") == "L3_not_top1":
                gk, tk = const_k(r.get("gt_score")), const_k(r.get("top1_score"))
                if gk is None or tk is None:
                    F["有非常數分數"] += 1
                elif gk == tk:
                    F["與第一名同分(k 相同)"] += 1
                else:
                    F["正確的人所在鏡頭人較多(k 較大)" if gk > tk else "正確的人所在鏡頭人較少(k 較小)"] += 1
            if r.get("layer") == "L0_first":
                G["綁到既有身份" if r["matched"] else "開新身份"] += 1
            if r.get("layer") == "OK":
                if n >= 2:
                    H["候選 ≥ 2"] += 1
                    H["其中與第二名同分"] += r.get("margin_1_2") == 0
                else:
                    H["只有一個候選"] += 1

    print(f"[{run.name}]  門檻 {thr:.4f}   常數重疊分數 = {const:.4f} − ln k(k ≤ {math.floor(math.exp(const - thr))} 必過門檻)\n")
    print("A 結果 × 層")
    for o in ("正確", "誤併", "碎裂"):
        parts = "  ".join(f"{l}:{v}" for (l, oo), v in sorted(A.items()) if oo == o and v)
        print(f"   {o} {sum(v for (l, oo), v in A.items() if oo == o):>5,}  ←  {parts}")
    tv = sum(B_vol.values())
    print(f"\nB 誤併源頭(非多數擁有者配對 {sum(B_pairs.values())} 組,後續綁定 {tv:,} 次)")
    for k, v in B_pairs.most_common():
        print(f"   {k:<22} 配對 {v:>4}  後續綁定 {B_vol[k]:>5,}({B_vol[k] / tv:.1%})")
    for dim in ("分數類型", "同分", "創立者", "對方此刻可見性"):
        print(f"   {dim}(加權):" + ("  ⚠ 不等於候選路徑" if dim == "對方此刻可見性" else ""))
        for (dd, val), v in sorted(B_dim.items(), key=lambda x: -x[1]):
            if dd == dim:
                print(f"      {val:<28} {v:>5,}({v / tv:.1%})")
    print("\nC 誤偵開出的身份:" + "  ".join(f"{k} {v:,}" for k, v in C.items()))
    nd = sum(D.values())
    print(f"\nD 判斷當下的候選組成(共 {nd:,};括號內為其中成功綁定的次數)")
    for grp in ("真人決策", "誤偵決策"):
        parts = "  ".join(f"{k} {D[(g, k)]:,}({Dm[(g, k)]:,})" for (g, k) in sorted(D) if g == grp)
        print(f"   {grp}:{parts}")
    n_matched = sum(Dm.values())
    n_matched_const = sum(v for (g, k), v in Dm.items() if k == "第一名是常數重疊分數")
    print(f"   一致性:成功綁定 {n_matched:,} 次,其中第一名為常數分數 {n_matched_const:,} 次 → "
          f"{'全部來自常數分數' if n_matched == n_matched_const else '** 有非常數分數的綁定 **'}")
    if nonconst_top1:
        mx = max(s for s, _m in nonconst_top1)
        print(f"   第一名是非常數分數:最高 {mx:.3f}(門檻 {thr:.4f}),≥ 門檻 {sum(s >= thr for s, _m in nonconst_top1)} 次,"
              f"成功綁定 {sum(m for _s, m in nonconst_top1)} 次")
    print(f"\nE 成功綁定的分數類型:{dict(E)}  觀察到最大 k = {kmax}")
    print(f"F L3 組成:{dict(F)}")
    print(f"G 一個人第一次出現:{dict(G)}")
    print(f"H 認對的判斷:{dict(H)}")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(dict(
            run_root=str(run), seqs=args.seqs, threshold=thr, overlap_const=const,
            A={f"{l}|{o}": v for (l, o), v in A.items()}, B_pairs=dict(B_pairs), B_volume=dict(B_vol),
            B_dims={f"{d}|{v}": n for (d, v), n in B_dim.items()}, C=dict(C),
            D={f"{g}|{k}": v for (g, k), v in D.items()},
            D_matched={f"{g}|{k}": v for (g, k), v in Dm.items()},
            nonconst_top1_max=max((s for s, _m in nonconst_top1), default=None),
            nonconst_top1_matched=sum(m for _s, m in nonconst_top1), E=dict(E), kmax=kmax,
            F=dict(F), G=dict(G), H=dict(H)), ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n已存 {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
