"""§8.34:只用「食材有被 M3 偵測到」的候選測 NG4a 觸發(第 3 輪凍結參數)。

預先登記:docs/M6_NG準則評估_預先登記_20261027.md §8.34
可偵測(只看 M3 偵測,不看觸發結果;時間用 EPFL 標註的清洗區間):
  D1(主):標註清洗區間內,Boutput1 / 2 有「食材框在洗手台 zone 且與水槽內的手框重疊」至少 1 步(信心 ≥ 0.4)
  D2(嚴):D1 的步數 ≥ 3(0.3 秒,等於觸發的清洗門檻)
輸出 results/m6/ng/ng4_detectable/(不進 git)

用法:
    python scripts/eval_ng4_detectable.py
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import eval_ng4_trigger as X                                              # noqa: E402
from m6_events import ng4                                                 # noqa: E402

PARAMS = dict(X.GRID3[3], objects=["食材"])                               # §8.33 全資料最佳格點
OUT = ROOT / "results/m6/ng/ng4_detectable"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    p = dict(ng4.DEFAULT, **PARAMS)
    rows = []
    for c in X.R.load():
        t0 = max(0.0, c["wash_start"] - X.R.PRE_S)
        cs = X.load(X.R.CACHE / c["id"])
        views = {v: ng4._View(cs[v], X.SINK.get(v), p) for v in p["views"] if v in cs}
        T = min(len(ng4.steps(x)[0]) for x in cs.values())
        raw = ng4.wash_series(views, T, p)[:, 0]
        a, b = int(round((c["wash_start"] - t0) / X.STEP)), int(round((c["wash_end"] - t0) / X.STEP)) + 1
        n_raw = int(raw[max(0, a):min(T, b)].sum())
        ev = ng4.detect_ng4(cs, X.SINK, X.STEP, params=PARAMS, t0=t0)
        ng = [e for e in ev if e["ng4"]]
        rows.append(dict(id=c["id"], food=c["food"], gt_id=c["gt_id"], n_raw=n_raw, d1=n_raw >= 1, d2=n_raw >= 3,
                         n_events=len(ev), pred=int(bool(ng)), pred_sure=int(any(e["status"] == "確定" for e in ng))))
        print(f"  {c['id']} 原始清洗步數 {n_raw} 事件 {len(ev)} NG4 {len(ng)}", flush=True)
    d = pd.DataFrame(rows)
    d.to_csv(OUT / "per_candidate.csv", index=False, encoding="utf-8-sig")
    summ = []
    for name, sub in (("全部", d), ("D1 可偵測", d[d.d1]), ("D1 不可偵測", d[~d.d1]), ("D2 可偵測", d[d.d2])):
        u = sub[sub.gt_id.isin([1, 2])]
        for col, m in (("pred", "確定 + 待確認(主)"), ("pred_sure", "只算確定")):
            summ.append(dict(subset=name, method=m, base=round((u.gt_id == 2).mean(), 3),
                             has_event=round((u.n_events > 0).mean(), 3),
                             **X.R.prf(list(zip(u[col].astype(bool), u.gt_id == 2)))))
    s = pd.DataFrame(summ)
    s.to_csv(OUT / "summary.csv", index=False, encoding="utf-8-sig")
    print("\n" + s.to_string())


if __name__ == "__main__":
    main()
