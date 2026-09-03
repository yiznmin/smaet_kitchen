"""交叉檢核:同一份 HDF5 embedding,用官方 evaluate_reid.py 與本專案的
`reid_eval_market1501.evaluate_cmc_map` 各算一次 CMC/mAP。

預先登記 §4 要求這一步:「兩者結果應該一致(±小數點誤差);不一致就是協定理解錯了」。

實際跑起來會有系統性差異,而且**差異本身是預期的**,原因寫在這裡以免下次又查一遍:

1. **同相機排除**。本專案的 evaluate_cmc_map 走 Market-1501 協定,排除
   「同 pid + 同 cam」的 gallery 項;官方 evaluate_reid.py 不排除。
   CHIRLA 的 gallery(train_k)與 query(test_k)本來就取自不同片段,
   官方協定不做這件事。→ 交叉檢核時把 gallery 的 cam 全設成一個不會與 query
   重疊的值,讓排除條件永不成立,才是在比「同一個協定下的兩份實作」。
2. **AP 的算法**。官方用 sklearn.average_precision_score(以相似度為分數),
   本專案用 Re-ID 慣用的 sum(precision@k * rel_k) / num_rel。寫這支的時候預期
   兩者會有小數點級距離 —— **2026-09-03 實跑的結果是完全一致(四個 scenario
   的 rank1/5/10 與 mAP 差距 ≤ 0.01pp)**。原因是沒有並列分數時,
   sklearn 的 sum((R_n − R_{n−1})·P_n) 與 Re-ID 的公式在數學上同一式。
   所以這裡真正的判準是**兩者應該相等**;出現肉眼可見的差距就是協定理解錯了。
3. **無正解的 query**。本專案 `continue` 跳過;官方在 closed-set 前已把
   「gallery 裡沒有的 id」濾掉,效果接近但不完全相同。

配對方式與官方 `--per-subset` 相同:query 的 test_k 對 gallery 的 train_k,
最後對各 subset 取等權平均。

用法:
    python scripts/crosscheck_chirla_eval.py \
        --gallery embeddings/chirla_armS/resnet50_bnneck/reid_long_term_gallery_embeddings.h5 \
        --query   embeddings/chirla_armS/resnet50_bnneck/reid_long_term_query_embeddings.h5
"""
import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))


def load_h5(p):
    import h5py
    with h5py.File(p, "r") as f:
        return f["embeddings"][:], f["ids"][:], f["paths"][:].astype(str)


def subset_of(path):
    """與官方 extract_subset_from_path 同義:找 /train/train_k/ 或 /test/test_k/。"""
    parts = path.replace("\\", "/").split("/")
    for i, p in enumerate(parts):
        if p in ("train", "test") and i + 1 < len(parts) \
                and parts[i + 1].startswith(("train_", "test_")):
            return parts[i + 1]
    return "unknown"


def group(ids, paths):
    g = defaultdict(list)
    for i, p in enumerate(paths):
        g[subset_of(p)].append(i)
    return {k: np.asarray(v) for k, v in g.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gallery", required=True)
    ap.add_argument("--query", required=True)
    ap.add_argument("--topk", nargs="+", type=int, default=[1, 5, 10])
    args = ap.parse_args()

    from reid_eval_market1501 import evaluate_cmc_map

    gf, gid, gpath = load_h5(args.gallery)
    qf, qid, qpath = load_h5(args.query)
    gsub, qsub = group(gid, gpath), group(qid, qpath)

    print("=" * 78)
    print("交叉檢核:本專案 evaluate_cmc_map(協定對齊官方後)")
    print("=" * 78)
    print(f"  gallery {gf.shape}  query {qf.shape}")
    print(f"  {'subset':<10}{'查詢數':>8}{'rank1':>9}{'rank5':>9}{'rank10':>9}{'mAP':>9}")
    print("  " + "-" * 56)

    rows = []
    for qs in sorted(qsub):
        gs = qs.replace("test_", "train_")
        if gs not in gsub:
            print(f"  {qs:<10}  (gallery 沒有對應的 {gs},跳過)")
            continue
        qi, gi = qsub[qs], gsub[gs]
        # closed-set:只留「非 distractor(id>=0)且 id 出現在 gallery」的 query,
        # 與官方 evaluate_cmc_map_with_unknowns 的 truly_known_mask 同義
        keep = (qid[qi] >= 0) & np.isin(qid[qi], np.unique(gid[gi]))
        qi = qi[keep]
        if not len(qi):
            print(f"  {qs:<10}  (closed-set 後沒有 query,跳過)")
            continue
        # 相機欄位:官方協定不排除同相機,所以給 gallery 一個絕不與 query 相同的值,
        # 讓 Market-1501 的排除條件永不成立 —— 這樣比的才是實作而不是協定
        r = evaluate_cmc_map(qf[qi], qid[qi], np.zeros(len(qi), int),
                             gf[gi], gid[gi], np.ones(len(gi), int),
                             topk=tuple(args.topk))
        rows.append(r)
        print(f"  {qs:<10}{len(qi):>8}{r['rank1']*100:>8.2f}%{r['rank5']*100:>8.2f}%"
              f"{r['rank10']*100:>8.2f}%{r['mAP']*100:>8.2f}%")

    if not rows:
        print("\n沒有可比對的 subset")
        return 1
    print("  " + "-" * 56)
    avg = {k: float(np.mean([r[k] for r in rows]))
           for k in ("rank1", "rank5", "rank10", "mAP")}
    print(f"  {'等權平均':<10}{'':>8}{avg['rank1']*100:>8.2f}%{avg['rank5']*100:>8.2f}%"
          f"{avg['rank10']*100:>8.2f}%{avg['mAP']*100:>8.2f}%")
    print("\n拿這組數字與官方 evaluate_reid.py --per-subset 的輸出並列。")
    print("2026-09-03 實測:四個 scenario 兩邊完全一致(≤0.01pp)。")
    print("出現肉眼可見的差距就是協定理解錯了,先查清楚再報任何數字。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
