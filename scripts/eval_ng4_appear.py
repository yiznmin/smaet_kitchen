"""§8.67:NG4 觸發框外觀比對(app_dist)+ 放下要靜止(put_still_iou),依受試者留一選參數。

預先登記:docs/M6_NG準則評估_預先登記_20261027.md §8.67
測試集:§8.34 D1 115 件;只看食材類事件。基準 = 正式設定(violation_rules.yaml ng4,§8.65)。
直方圖快取 results/m6/ng/ng4_appear_cache/<id>/<鏡頭>.npz;輸出 results/m6/ng/ng4_appear/(皆含 EPFL 衍生,不進 git)

用法:
    python scripts/eval_ng4_appear.py --build   # 先建直方圖快取
    python scripts/eval_ng4_appear.py
"""
import sys
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import eval_ng4_trigger as X                                              # noqa: E402
from build_ng_report import zone                                          # noqa: E402
from m3.classes import NAME_TO_CLASS                                      # noqa: E402
from m6_events.ng4 import DEFAULT, detect_ng4                             # noqa: E402
from m6_events.violation import at, steps                                 # noqa: E402

OUT = ROOT / "results/m6/ng/ng4_appear"
APP = ROOT / "results/m6/ng/ng4_appear_cache"
DET = ROOT / "results/m6/ng/ng4_detectable/per_candidate.csv"
FOOD, THR = NAME_TO_CLASS["食材"], 0.3
# 格點 0 = 現行(兩者都關)
GRID = [dict(app_dist=a, put_still_iou=s) for a in (0, 0.35, 0.5, 0.65) for s in (0, 0.5, 0.7)]


def hist(crop):
    """色相 × 飽和度直方圖(18 × 8),總和正規化為 1。"""
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    h = cv2.calcHist([hsv], [0, 1], None, [18, 8], [0, 180, 0, 256]).ravel()
    return (h / max(h.sum(), 1e-6)).astype(np.float32)


def build_one(cid):
    """單一候選:逐鏡頭讀影片,每步(每 3 幀)對食材框算直方圖。"""
    cs = X.load(X.R.CACHE / cid)
    (APP / cid).mkdir(parents=True, exist_ok=True)
    for vn, c in cs.items():
        out = APP / cid / f"{vn}.npz"
        if out.exists():
            continue
        fids, *_ = steps(c)
        idx = steps(c)
        stride = int(c.meta["cache_stride"])
        cap = cv2.VideoCapture(str(ROOT / "data/ng4_ok_win" / cid / f"{vn}.mp4"))
        st, bx, hs = [], [], []
        fi = 0
        for i in range(len(fids)):
            while fi < i * stride:
                cap.grab()
                fi += 1
            ok, fr = cap.read()
            fi += 1
            if not ok:
                break
            xy, cf, cl = at(c, idx, i)
            for x in xy[(cl == FOOD) & (cf >= THR)]:
                x1, y1, x2, y2 = np.clip(x, 0, [fr.shape[1], fr.shape[0]] * 2).astype(int)
                if x2 - x1 < 2 or y2 - y1 < 2:
                    continue
                st.append(i)
                bx.append(x)
                hs.append(hist(fr[y1:y2, x1:x2]))
        cap.release()
        np.savez_compressed(out, step=np.array(st, np.int32), box=np.array(bx, np.float32).reshape(-1, 4),
                            hist=np.array(hs, np.float32).reshape(-1, 144))
    return cid


def load_app(cid):
    out = {}
    for f in (APP / cid).glob("*.npz"):
        z = np.load(f)
        d = {}
        for s in np.unique(z["step"]):
            m = z["step"] == s
            d[int(s)] = (z["box"][m].astype(float), z["hist"][m])
        out[f.stem] = d
    return out


def cands():
    d1 = set(pd.read_csv(DET).query("d1").id)
    return [c for c in X.R.load() if c["id"] in d1 and c["gt_id"] in (1, 2)]


def main():
    if "--build" in sys.argv:
        ids = [c["id"] for c in cands()]
        with Pool(6) as pool:
            for k, cid in enumerate(pool.imap_unordered(build_one, ids), 1):
                print(f"  {k}/{len(ids)} {cid}", flush=True)
        return
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load((X.E.INTEG / "configs/violation_rules.yaml").read_text(encoding="utf-8"))["ng4"]
    cur = {k: v for k, v in cfg.items() if k not in ("enable", "sink_cams", "sink_zone")}
    cur["objects"] = ["食材"]
    sinks = {v: zone(v, cfg.get("sink_zone", "洗手台")) for v in cfg["sink_cams"]}
    rows = []
    for c in cands():
        t0 = max(0.0, c["wash_start"] - X.R.PRE_S)
        cs = X.load(X.R.CACHE / c["id"])
        apps = load_app(c["id"])
        r = dict(id=c["id"], pid=c["id"][:6], food=c["food"], y=int(c["gt_id"] == 2))
        r["cur"] = int(any(e["ng4"] for e in detect_ng4(cs, sinks, X.STEP, params=cur, t0=t0)))
        for gi, g in enumerate(GRID):
            ev = detect_ng4(cs, sinks, X.STEP, params=dict(cur, **g), t0=t0, apps=apps)
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


if __name__ == "__main__":
    main()
