"""§8.91(探索性):NG3 第 3 輪誤報 27 件經第三人人工檢視後,以修正真值重算(不再調規則)。

人工檢視表:D:/yizhen/NG3_碰臉誤報人工檢視_20261105/人工檢視表.xlsx(1 有碰臉 / 2 沒碰臉 / 3 看不出來)。
- 1 → 改正例;2 → 維持反例;3 → 排除。
- 只檢視了誤報(預測為正的反例),預測為負的反例(TN)沒檢視:P 已完整修正;R 可能高估(TN 中若有漏標的碰臉會變 FN)。
報兩種:固定 §8.85 選到的 r0.5_hand_m0.0_any;以及修正真值下重跑參與者留一(F1 選格點)。
輸出 results/m6/ng/ng3_face_r3_review/{per_clip.csv, summary.csv}(含團隊標註,不進 git)。

用法:
    python scripts/eval_ng3_review.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "results/m6/ng/ng3_face_r3/per_clip.csv"
REVIEW = Path("D:/yizhen/NG3_碰臉誤報人工檢視_20261105/人工檢視表.xlsx")
OUT = ROOT / "results/m6/ng/ng3_face_r3_review"
FIXED = "r0.5_hand_m0.0_any"
RES_COL = "檢視結果(1 有碰臉 / 2 沒碰臉 / 3 看不出來)"


def prf(y, p):
    y, p = np.asarray(y, bool), np.asarray(p, bool)
    tp, fp, fn = int((y & p).sum()), int((~y & p).sum()), int((y & ~p).sum())
    R, P = tp / max(tp + fn, 1), tp / max(tp + fp, 1)
    return dict(n=len(y), TP=tp, FP=fp, FN=fn, TN=len(y) - tp - fp - fn, R=round(R, 3), P=round(P, 3),
                F1=round(2 * R * P / max(R + P, 1e-9), 3))


def lopo(d, tags):
    pairs_y, pairs_p, chosen = [], [], []
    for pid in sorted(d.pid.unique()):
        tr, te = d[d.pid != pid], d[d.pid == pid]
        best = max(tags, key=lambda t: (prf(tr.y, tr[t])["F1"], -tags.index(t)))
        chosen.append(best)
        pairs_y += te.y.tolist()
        pairs_p += te[best].tolist()
    return prf(pairs_y, pairs_p), pd.Series(chosen).value_counts().to_dict()


def main():
    d = pd.read_csv(SRC, encoding="utf-8-sig")
    d = d[~d.note.fillna("").str.contains("擦鼻子")].copy()       # §8.85 排除擦鼻子(衛生紙擦鼻子不算碰臉)
    rv = pd.read_excel(REVIEW, sheet_name="人工檢視表")
    res = dict(zip(rv["片段"], rv[RES_COL].astype(int)))
    tags = [c for c in d.columns if c.startswith("r0.") and not c.startswith("n_")]
    d["y_old"] = d.y.astype(int)
    d["review"] = d.key.map(res)
    d.loc[d.review == 1, "y"] = 1
    new = d[d.review != 3].copy()
    new["y"] = new.y.astype(int)
    OUT.mkdir(parents=True, exist_ok=True)
    new.to_csv(OUT / "per_clip.csv", index=False, encoding="utf-8-sig")
    old = d.copy()
    old["y"] = old.y_old
    rows = []
    for name, df in (("原真值(§8.85)", old), ("修正真值", new)):
        rows.append(dict(item=f"{name} 固定 {FIXED}", **prf(df.y, df[FIXED])))
        cv, ch = lopo(df, tags)
        rows.append(dict(item=f"{name} LOPO", chosen=str(ch), **cv))
    s = pd.DataFrame(rows)
    s.to_csv(OUT / "summary.csv", index=False, encoding="utf-8-sig")
    print(f"檢視結果:有碰臉 {sum(v == 1 for v in res.values())} / 沒碰臉 {sum(v == 2 for v in res.values())} / "
          f"看不出來 {sum(v == 3 for v in res.values())}")
    print(s.to_string())


if __name__ == "__main__":
    main()
