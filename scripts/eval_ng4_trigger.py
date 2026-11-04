"""§8.28:NG4a 自動觸發(清洗 → 放下 → 判位置)測試。

預先登記:docs/M6_NG準則評估_預先登記_20261027.md §8.28
偵測:integrated/src/m6_events/ng4.py detect_ng4(只用規則)
- 測試 1:人工檢視 128 件候選視窗(results/det_cache/ng4_ok_win_s3/),只看食材類,真值 = 檢視結果
- 測試 2:團隊 NG4a 5 次(results/det_cache/ng4_win_s3/,§8.13 視窗),逐次召回(只報)
- 測試 3:8 個完整場次(results/det_cache/epfl_sessions_s3/),觸發率與 EPFL「Wash」標註佐證率
輸出 results/m6/ng/ng4_trigger/(含團隊 / 檢視標註,不進 git)

用法:
    python scripts/eval_ng4_trigger.py
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import eval_ng_clips as E                                                 # noqa: E402
import eval_ng4_instances as I                                            # noqa: E402
import eval_ng4_ok_review as R                                            # noqa: E402
from common.zones import get_zone                                         # noqa: E402
from m4_track.det_cache import DetCache                                   # noqa: E402
from m6_events.ng4 import DEFAULT, detect_ng4                             # noqa: E402

FULL = ROOT / "results/det_cache/epfl_sessions_s3"
ANN = ROOT / "data/epfl_ann"
OUT = ROOT / "results/m6/ng/ng4_trigger"
STEP, MATCH_S = 0.1, 2.0
SINK = {v: get_zone(E.INTEG / f"configs/zones/epfl_{v}.json", "洗手台") for v in ("Boutput1", "Boutput2")}


def load(d):
    return {v: DetCache(d / f"{v}.npz") for v in DEFAULT["views"] if (d / f"{v}.npz").exists()}


def test1():
    rows = R.load()
    per = []
    for n, c in enumerate(rows, 1):
        t0 = max(0.0, c["wash_start"] - R.PRE_S)
        ev = detect_ng4(load(R.CACHE / c["id"]), SINK, STEP, t0=t0)
        food = [e for e in ev if e["obj"] == "食材"]
        ng = [e for e in food if e["ng4"]]
        best = min(food, key=lambda e: abs(e["put"] - c["put_end"]), default=None)
        per.append(dict(id=c["id"], food=c["food"], gt_id=c["gt_id"], gt_content=c["gt_content"], auto_b=c["auto"],
                        n_food_events=len(food), pred=int(bool(ng)),
                        pred_sure=int(any(e["status"] == "確定" for e in ng)),
                        place=best["place"] if best else "未偵測到", put_err=round(best["put"] - c["put_end"], 2)
                        if best else None, n_all_ng4=sum(e["ng4"] for e in ev)))
        print(f"  [1] {n}/{len(rows)} {c['id']} 食材事件 {len(food)} NG4 {len(ng)} "
              f"{per[-1]['place']}", flush=True)
    per = pd.DataFrame(per)
    summ = []
    for gk in ("gt_id", "gt_content"):
        u = per[per[gk].isin([1, 2])]
        base = round((u[gk] == 2).mean(), 3)
        for col, name in (("pred", "確定 + 待確認(主)"), ("pred_sure", "只算確定")):
            summ.append(dict(test="1", truth=gk, method=name, base=base,
                             **R.prf(list(zip(u[col].astype(bool), u[gk] == 2)))))
    e = per.put_err.dropna().abs()
    summ.append(dict(test="1", truth="", method="放下時間誤差(秒)", n=len(e), median=round(e.median(), 2),
                     within2=round((e <= 2).mean(), 3), detected=round(per.n_food_events.gt(0).mean(), 3)))
    return per, summ


def test2():
    gt = [g for g in I.truth() if g["kind"] == "NG4a"]
    wins = I.windows(I.truth())
    dets, hit, dur = [], set(), 0.0
    for w in wins:
        ev = [e for e in detect_ng4(load(I.CACHE / w["key"]), SINK, STEP, t0=w["t0"]) if e["ng4"]]
        dur += w["dur"]
        for e in ev:
            m = [i for i, g in enumerate(gt) if g["session"] == w["session"] and I.overlap(e["start"], e["end"],
                                                                                           g["start"], g["end"])]
            hit.update(m)
            dets.append(dict(window=w["key"], **e, match=int(bool(m))))
        print(f"  [2] {w['key']} NG4 觸發 {len(ev)}", flush=True)
    summ = [dict(test="2", method="團隊 NG4a 逐次召回", n=len(gt), tp=len(hit), R=round(len(hit) / len(gt), 3),
                 n_det=len(dets), per_min=round(len(dets) / dur * 60, 2),
                 P=round(sum(d["match"] for d in dets) / max(len(dets), 1), 3))]
    return pd.DataFrame(dets), summ


def wash_ann(sn):
    pid, sess = sn.split("_", 1)
    a = pd.read_excel(ANN / pid / sess / "annotations/actions_annotations.xlsx")
    a = a[a.Verbs.astype(str) == "Wash"]
    hand = a.Nouns.astype(str) == "Hand"
    return a[~hand][["Start", "End"]].to_numpy(), a[hand][["Start", "End"]].to_numpy()


def test3():
    rows, summ, hours = [], [], 0.0
    for sn in sorted(os.listdir(FULL)):
        cs = load(FULL / sn)
        ev = detect_ng4(cs, SINK, STEP)
        obj_w, hand_w = wash_ann(sn)
        hours += cs["Boutput1"].meta["max_fid"] / 30 / 3600
        ov = lambda e, A: bool(len(A)) and bool(((A[:, 0] - MATCH_S <= e["wash_end"]) &
                                                 (A[:, 1] + MATCH_S >= e["start"])).any())
        for e in ev:
            rows.append(dict(session=sn, **e, wash_obj_ann=int(ov(e, obj_w)), wash_hand_ann=int(ov(e, hand_w))))
        print(f"  [3] {sn} 清洗→放下 {len(ev)} 次、NG4 {sum(e['ng4'] for e in ev)};"
              f"EPFL Wash 物件 {len(obj_w)} 次、洗手 {len(hand_w)} 次", flush=True)
    d = pd.DataFrame(rows)
    for name, sub in (("NG4 觸發(確定 + 待確認)", d[d.ng4]), ("NG4 確定", d[d.status == "確定"]),
                      ("全部清洗→放下(含合格)", d)):
        summ.append(dict(test="3", method=name, n=len(sub), per_hour=round(len(sub) / hours, 1),
                         backed=round(sub.wash_obj_ann.mean(), 3) if len(sub) else None,
                         hand_wash=round(sub.wash_hand_ann.mean(), 3) if len(sub) else None, hours=round(hours, 2)))
    # 這 8 場內的人工檢視候選:完整場次模式下能否觸發
    rv = [c for c in R.load() if c["session"].split("/")[1] + "_" + c["session"].split("/")[2] in os.listdir(FULL)]
    pairs = []
    for c in rv:
        sn = c["session"].split("/")[1] + "_" + c["session"].split("/")[2]
        m = d[(d.session == sn) & (d.obj == "食材") & d.ng4 & (d.start <= c["put_end"] + MATCH_S)
              & (d.end >= c["wash_start"] - MATCH_S)] if len(d) else d
        if c["gt_id"] in (1, 2):
            pairs.append((len(m) > 0, c["gt_id"] == 2))
    if pairs:
        summ.append(dict(test="3", method="場內人工檢視候選", **R.prf(pairs)))
    return d, summ


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    p1, s1 = test1()
    d2, s2 = test2()
    d3, s3 = test3()
    p1.to_csv(OUT / "test1_per_candidate.csv", index=False, encoding="utf-8-sig")
    d2.to_csv(OUT / "test2_detections.csv", index=False, encoding="utf-8-sig")
    d3.to_csv(OUT / "test3_events.csv", index=False, encoding="utf-8-sig")
    s = pd.DataFrame(s1 + s2 + s3)
    s.to_csv(OUT / "summary.csv", index=False, encoding="utf-8-sig")
    print("\n" + s.to_string())
    print(f"\n輸出 {OUT}")


if __name__ == "__main__" and "--round2" not in sys.argv:
    main()


# ── §8.30 第 2 輪:放寬清洗 / 放下門檻,依受試者留一交叉驗證選參數 ──────────────
GRID2 = [dict(wash_smooth=True, wash_min_s=1.0, wash_gap_s=1.0)] + \
        [dict(wash_smooth=False, wash_min_s=m, wash_gap_s=g) for m in (0.3, 0.5) for g in (1.0, 3.0)]
GRID2 = [dict(g, put_min_s=pm) for g in GRID2 for pm in (1.0, 0.5)]
OUT2 = ROOT / "results/m6/ng/ng4_trigger_r2"


def round2():
    OUT2.mkdir(parents=True, exist_ok=True)
    rows = R.load()
    pred = {}                                                   # (格點序號, 候選) → 0/1
    for n, c in enumerate(rows, 1):
        t0 = max(0.0, c["wash_start"] - R.PRE_S)
        cs = load(R.CACHE / c["id"])
        for gi, g in enumerate(GRID2):
            ev = detect_ng4(cs, SINK, STEP, params=dict(g, objects=["食材"]), t0=t0)
            pred[(gi, c["id"])] = int(any(e["ng4"] for e in ev))
        print(f"  [r2] {n}/{len(rows)} {c['id']}", flush=True)
    use = [c for c in rows if c["gt_id"] in (1, 2)]
    pids = sorted({c["id"][:6] for c in use})
    grid_rows = []
    for gi, g in enumerate(GRID2):
        m = R.prf([(bool(pred[(gi, c["id"])]), c["gt_id"] == 2) for c in use])
        grid_rows.append(dict(grid=gi, **g, **m))
    # 留一受試者:在其餘受試者上選 F1 最高的格點(同分取序號小者),套到留下的受試者
    cv_pairs, chosen = [], []
    for pid in pids:
        tr = [c for c in use if c["id"][:6] != pid]
        f1 = [R.prf([(bool(pred[(gi, c["id"])]), c["gt_id"] == 2) for c in tr])["F1"] for gi in range(len(GRID2))]
        best = int(np.argmax(f1))
        chosen.append(dict(pid=pid, grid=best))
        cv_pairs += [(bool(pred[(best, c["id"])]), c["gt_id"] == 2) for c in use if c["id"][:6] == pid]
    cv = R.prf(cv_pairs)
    full = max(range(len(GRID2)), key=lambda gi: (grid_rows[gi]["F1"], -gi))
    pd.DataFrame(grid_rows).to_csv(OUT2 / "grid_test1.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(chosen).to_csv(OUT2 / "cv_chosen.csv", index=False, encoding="utf-8-sig")
    print(pd.DataFrame(grid_rows).to_string())
    print(f"\nLOPO CV:{cv}\n全資料最佳格點 {full}:{GRID2[full]}")
    # 測試 3:用全資料最佳格點(所有物件)
    s3 = []
    hours, ev_all = 0.0, []
    for sn in sorted(os.listdir(FULL)):
        cs = load(FULL / sn)
        hours += cs["Boutput1"].meta["max_fid"] / 30 / 3600
        obj_w, hand_w = wash_ann(sn)
        ov = lambda e, A: bool(len(A)) and bool(((A[:, 0] - MATCH_S <= e["wash_end"]) &
                                                 (A[:, 1] + MATCH_S >= e["start"])).any())
        for e in detect_ng4(cs, SINK, STEP, params=GRID2[full]):
            ev_all.append(dict(session=sn, **e, wash_obj_ann=int(ov(e, obj_w)), wash_hand_ann=int(ov(e, hand_w))))
        print(f"  [r2-3] {sn}", flush=True)
    d = pd.DataFrame(ev_all)
    d.to_csv(OUT2 / "test3_events.csv", index=False, encoding="utf-8-sig")
    ng = d[d.ng4]
    s3 = dict(n=len(ng), per_hour=round(len(ng) / hours, 1), backed=round(ng.wash_obj_ann.mean(), 3),
              hand_wash=round(ng.wash_hand_ann.mean(), 3))
    summ = [dict(item="LOPO CV(測試 1)", **cv), dict(item=f"全資料最佳格點 {full}", **grid_rows[full]),
            dict(item="測試 3(全資料最佳格點)", **s3)]
    pd.DataFrame(summ).to_csv(OUT2 / "summary.csv", index=False, encoding="utf-8-sig")
    print(f"測試 3:{s3}\n輸出 {OUT2}")


if __name__ == "__main__" and "--round2" in sys.argv:
    round2()
