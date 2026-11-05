"""§8.51:NG4 放下來源限制(origin_s)+ 砧板 / 容器位置記憶(sup_mem_s),依受試者留一選參數。

預先登記:docs/M6_NG準則評估_預先登記_20261027.md §8.51
測試集:§8.34 D1 115 件;只看食材類事件。基準 = 正式設定(violation_rules.yaml ng4,§8.46)。
輸出 results/m6/ng/ng4_origin/(不進 git)

用法:
    python scripts/eval_ng4_origin.py
    python scripts/eval_ng4_origin.py --start   # §8.53
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import eval_ng4_trigger as X                                              # noqa: E402
from m6_events.ng4 import detect_ng4                                      # noqa: E402

OUT = ROOT / "results/m6/ng/ng4_origin"
DET = ROOT / "results/m6/ng/ng4_detectable/per_candidate.csv"
# 格點 0 = 現行(兩者都關)
GRID = [dict(origin_s=o, sup_mem_s=m) for o in (0, 3, 5) for m in (0, 2, 5)]
if "--start" in sys.argv:                                    # §8.53
    OUT = ROOT / "results/m6/ng/ng4_start"
    GRID = [dict(start_s=s, start_mode=md, sup_mem_s=m) for s, md in [(0, "skip"), (1, "skip"), (2, "skip"),
                                                                       (1, "base"), (2, "base")] for m in (0, 5)]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load((X.E.INTEG / "configs/violation_rules.yaml").read_text(encoding="utf-8"))["ng4"]
    cur = {k: v for k, v in cfg.items() if k not in ("enable", "sink_cams")}
    cur["objects"] = ["食材"]
    d1 = set(pd.read_csv(DET).query("d1").id)
    rows = []
    for c in X.R.load():
        if c["id"] not in d1 or c["gt_id"] not in (1, 2):
            continue
        t0 = max(0.0, c["wash_start"] - X.R.PRE_S)
        cs = X.load(X.R.CACHE / c["id"])
        r = dict(id=c["id"], pid=c["id"][:6], food=c["food"], y=int(c["gt_id"] == 2))
        r["cur"] = int(any(e["ng4"] for e in detect_ng4(cs, X.SINK, X.STEP, params=cur, t0=t0)))
        for gi, g in enumerate(GRID):
            ev = detect_ng4(cs, X.SINK, X.STEP, params=dict(cur, **g), t0=t0)
            r[f"g{gi}"] = int(any(e["ng4"] for e in ev))
            r[f"g{gi}_n"] = len(ev)
        rows.append(r)
        print(f"  {c['id']} 真值 {r['y']} 現行 {r['cur']} " + " ".join(str(r[f'g{i}']) for i in range(len(GRID))),
              flush=True)
    d = pd.DataFrame(rows)
    d.to_csv(OUT / "per_candidate.csv", index=False, encoding="utf-8-sig")
    prf = lambda sub, col: X.R.prf(list(zip(sub[col].astype(bool), sub.y == 1)))
    grid = [dict(grid=gi, **g, events_per_clip=round(d[f"g{gi}_n"].mean(), 2), **prf(d, f"g{gi}"))
            for gi, g in enumerate(GRID)]
    pairs, chosen = [], []
    for pid in sorted(d.pid.unique()):
        tr = d[d.pid != pid]
        f1 = [prf(tr, f"g{gi}")["F1"] for gi in range(len(GRID))]
        best = int(np.argmax(f1))
        chosen.append(best)
        te = d[d.pid == pid]
        pairs += list(zip(te[f"g{best}"].astype(bool), te.y == 1))
    cv = X.R.prf(pairs)
    full = max(range(len(GRID)), key=lambda gi: (grid[gi]["F1"], -gi))
    summ = [dict(item="現行", **prf(d, "cur")), dict(item="LOPO", chosen=str(pd.Series(chosen).value_counts().to_dict()), **cv),
            dict(item=f"全資料最佳格點 {full}", **GRID[full], **prf(d, f"g{full}"))]
    pd.DataFrame(grid).to_csv(OUT / "grid.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(summ).to_csv(OUT / "summary.csv", index=False, encoding="utf-8-sig")
    print("\n" + pd.DataFrame(grid).to_string() + "\n\n" + pd.DataFrame(summ).to_string())
    ck = d[d.id == "YH2004_2023_07_06_08_59_27_ng002"]
    print("黃瓜:", ck[[f"g{i}" for i in range(len(GRID))]].to_dict("records"))


if __name__ == "__main__":
    main()
