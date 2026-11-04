"""§8.40:NG4「清洗期間也找放下」測試(與現行正式設定比較)。

預先登記:docs/M6_NG準則評估_預先登記_20261027.md §8.40
測試集:§8.34 D1(食材可偵測、人工檢視 1 / 2,115 件);只看食材類事件。
輸出 results/m6/ng/ng4_during/(含檢視標註,不進 git)

用法:
    python scripts/eval_ng4_during.py
"""
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import eval_ng4_trigger as X                                              # noqa: E402
from m6_events.ng4 import detect_ng4                                      # noqa: E402

OUT = ROOT / "results/m6/ng/ng4_during"
DET = ROOT / "results/m6/ng/ng4_detectable/per_candidate.csv"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load((X.E.INTEG / "configs/violation_rules.yaml").read_text(encoding="utf-8"))["ng4"]
    cur = {k: v for k, v in cfg.items() if k not in ("enable", "sink_cams")}
    cur["objects"] = ["食材"]
    arms = {"現行": cur, "清洗期間也找": dict(cur, during_wash=True)}
    d1 = set(pd.read_csv(DET).query("d1").id)
    rows = []
    for c in X.R.load():
        if c["id"] not in d1 or c["gt_id"] not in (1, 2):
            continue
        t0 = max(0.0, c["wash_start"] - X.R.PRE_S)
        cs = X.load(X.R.CACHE / c["id"])
        r = dict(id=c["id"], food=c["food"], y=int(c["gt_id"] == 2))
        for name, p in arms.items():
            ev = detect_ng4(cs, X.SINK, X.STEP, params=p, t0=t0)
            r[f"{name}_pred"] = int(any(e["ng4"] for e in ev))
            r[f"{name}_n"] = len(ev)
            r[f"{name}_events"] = "、".join(f"{e['put']}秒 {e['place']}" for e in ev)
        rows.append(r)
        print(f"  {c['id']} 真值 {r['y']} 現行 {r['現行_pred']} 新 {r['清洗期間也找_pred']}", flush=True)
    d = pd.DataFrame(rows)
    d.to_csv(OUT / "per_candidate.csv", index=False, encoding="utf-8-sig")
    summ = [dict(arm=a, **X.R.prf(list(zip(d[f"{a}_pred"].astype(bool), d.y == 1)))) for a in arms]
    flips = d[d["現行_pred"] != d["清洗期間也找_pred"]]
    summ.append(dict(arm="改變的件數", n=len(flips), tp=int(((flips.清洗期間也找_pred == 1) & (flips.y == 1)).sum()),
                     fp=int(((flips.清洗期間也找_pred == 1) & (flips.y == 0)).sum())))
    s = pd.DataFrame(summ)
    s.to_csv(OUT / "summary.csv", index=False, encoding="utf-8-sig")
    print("\n" + s.to_string())


if __name__ == "__main__":
    main()
