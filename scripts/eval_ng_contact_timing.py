"""§8.22:接觸垃圾桶時間門檻(min_s)網格 + 回推污染起點(backtrack)網格。

預先登記:docs/M6_NG準則評估_預先登記_20261027.md §8.22
真值:同 §8.21(團隊回覆修正;NG2 = 事件結束 + 1 秒內)
輸出 results/m6/ng/contact_timing/(含團隊標註,不進 git)

用法:
    python scripts/eval_ng_contact_timing.py
"""
import argparse
import copy
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import eval_ng_clips as E                                                 # noqa: E402
import eval_ng_details as D                                               # noqa: E402
import eval_ng_trash_events as M                                          # noqa: E402
from m6_events.ng import backtrack_contacts                               # noqa: E402
from m6_events.violation import HandHygieneFSM                            # noqa: E402

OUT = ROOT / "results/m6/ng/contact_timing"
MIN_S = [0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0]
GAP_S, MAX_S, NEAR_S = [0.5, 1.0, 2.0], [3, 5, 10, 15], 1.0
BASE_MIN_S = 1.0
EARLY = {"b4bc7660", "98426e2b", "00184e88", "cefcfc35", "3d339029", "ef2ab21d"}   # 團隊起點早 ≥ 2 秒
NO_CONTACT = {"52fbe2c2", "1fe9866a"}                                       # 團隊回覆「手沒碰到」
# §8.24:M3 認得的丟棄物(食材類),依團隊事件備註
FOOD_KW = ("廚餘", "果皮", "籽", "子", "皮", "蛋殼")
NONFOOD_KW = ("包裝", "衛生紙", "餐巾紙")


def is_food(note):
    return any(k in note for k in FOOD_KW) and not any(k in note for k in NONFOOD_KW)


def prepare():
    """每事件算一次偵測序列(min_s 設最小,之後再篩長度)與真值。"""
    cfg = yaml.safe_load((E.INTEG / "configs/violation_rules.yaml").read_text(encoding="utf-8"))
    cfg0 = copy.deepcopy(cfg)
    cfg0["trash_event"]["min_s"] = 0.1
    inst = D.load_instances()
    events = M.load_events()
    corr = D.load_corrections(events)
    out = []
    for n, e in enumerate(events, 1):
        k = e["key"][-8:]
        c = corr.get(k)
        if c and c.get("exclude"):
            continue
        r = E.rule_part(dict(key=e["key"], caches={v: M.CACHE / e["key"] / f"{v}.npz" for v in M.VIEWS}), cfg0)
        info = inst.get(k)
        gt1 = info["ng1"] if info else 0
        gt2 = info["ng2"] if info else 0
        if c:
            gt1 = c.get("ng1", gt1)
            gt2 = c["ng2"] if "ng2" in c else gt2
        team_c = [a for kind, a, b in (info["details"] if info else []) if kind == "contact"]
        out.append(dict(event=k, pid=e["pid"], note=e["note"], start=e["start"], end=e["end"], w0=e["f0"] / M.FPS,
                        trig=r["trig_base"] or [], wash=r["wash"], touch=r["touch"], gt_ng1=gt1, gt_ng2=gt2,
                        team_start=min(team_c) if team_c else None))
        print(f"  {n}/{len(events)} {k}", flush=True)
    return cfg, out


def judge(ev, cfg, min_s, bt=None):
    """一個事件在指定設定下的 NG1 / NG2 預測與系統接觸起點(同 eval_ng_details --corrected 口徑)。"""
    step, hh = cfg["step_s"], cfg["hand_hygiene"]
    need = max(1, int(round(min_s / step)))
    contacts = [(s, e) for s, e in ev["trig"] if e - s >= need]
    if bt:
        food = ev["touch"][:, cfg["touch_classes"].index("食材")]
        contacts = backtrack_contacts(contacts, food, int(round(NEAR_S / step)), max(1, int(round(bt[0] / step))),
                                      int(round(bt[1] / step)))
    tsec = lambda i: ev["w0"] + i * step
    s_ev, e_ev = int((ev["start"] - ev["w0"]) / step), int((ev["end"] - ev["w0"]) / step)
    tr = [(s, x) for s, x in contacts if s <= e_ev + 10 and x >= s_ev - 10]
    p1 = p2 = 0
    sys_start = None
    if tr:
        sys_start = tsec(tr[0][0])
        end = tr[-1][1]
        dirty = end + int(round(hh["grace_s"] / step))
        fsm = HandHygieneFSM(hh["wash_s"], step, cfg["touch_classes"])
        p1 = int(ev["wash"][dirty:end + int(M.NG1_WASH_WITHIN_S / step)].sum() < fsm.need)
        v = fsm.run(dirty, ev["wash"], ev["touch"])
        p2 = int(v.result == "違規" and tsec(v.at) <= ev["end"] + D.TOL_S)
    err = round(sys_start - ev["team_start"], 2) if sys_start is not None and ev["team_start"] is not None else None
    return dict(pred_ng1=p1, pred_ng2=p2, contact=int(bool(tr)), sys_start=sys_start, err=err)


def prf(rows, ng):
    sub = [x for x in rows if x[f"gt_{ng}"] is not None]
    tp = sum(x[f"pred_{ng}"] and x[f"gt_{ng}"] for x in sub)
    fp = sum(x[f"pred_{ng}"] and not x[f"gt_{ng}"] for x in sub)
    fn = sum(x[f"gt_{ng}"] and not x[f"pred_{ng}"] for x in sub)
    R, P = tp / max(tp + fn, 1), tp / max(tp + fp, 1)
    return dict(tp=tp, fp=fp, fn=fn, R=round(R, 3), P=round(P, 3), F1=round(2 * P * R / max(P + R, 1e-9), 3))


def err_stats(rows, prefix=""):
    x = np.array([r["err"] for r in rows if r["err"] is not None], float)
    if not len(x):
        return {f"{prefix}n": 0}
    return {f"{prefix}n": len(x), f"{prefix}med_abs": round(float(np.median(np.abs(x))), 2),
            f"{prefix}within1": round(float(np.mean(np.abs(x) <= 1)), 2),
            f"{prefix}within2": round(float(np.mean(np.abs(x) <= 2)), 2), f"{prefix}mean": round(float(x.mean()), 2)}


def rows_of(evs, cfg, min_s, bt=None):
    return [dict(event=ev["event"], pid=ev["pid"], gt_ng1=ev["gt_ng1"], gt_ng2=ev["gt_ng2"],
                 team_start=ev["team_start"], **judge(ev, cfg, min_s, bt)) for ev in evs]


def summarize_a(rows):
    team_c = [r for r in rows if r["team_start"] is not None]
    return dict(**{f"ng1_{a}": b for a, b in prf(rows, "ng1").items()},
                **{f"ng2_{a}": b for a, b in prf(rows, "ng2").items()},
                contact_R=round(sum(r["contact"] for r in team_c) / max(len(team_c), 1), 3),
                no_contact_hit=sum(r["contact"] for r in rows if r["event"] in NO_CONTACT))


def summarize_b(rows):
    return dict(**err_stats(rows), **err_stats([r for r in rows if r["event"] in EARLY], "early_"),
                **err_stats([r for r in rows if r["event"] not in EARLY], "other_"),
                **{f"ng1_{a}": prf(rows, "ng1")[a] for a in ("tp", "fp", "fn")},
                **{f"ng2_{a}": prf(rows, "ng2")[a] for a in ("tp", "fp", "fn")})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", choices=["food"], help="§8.24:只留 M3 認得的丟棄物(食材類)")
    args = ap.parse_args()
    cfg, evs = prepare()
    out = OUT
    if args.subset == "food":
        evs = [ev for ev in evs if is_food(ev["note"])]
        out = OUT.parent / "contact_timing_food"
        print(f"子集 food:{len(evs)} 個事件,有團隊時間軸 {sum(ev['team_start'] is not None for ev in evs)};"
              f"早起點 {sorted(ev['event'] for ev in evs if ev['event'] in EARLY)}")
    E.OUT = out
    out.mkdir(parents=True, exist_ok=True)
    pids = sorted({ev["pid"] for ev in evs})

    # ── A. 時間門檻 ──
    res_a = {m: rows_of(evs, cfg, m) for m in MIN_S}
    grid_a = [dict(min_s=m, **summarize_a(res_a[m])) for m in MIN_S]
    E.write_csv("grid_min_s", grid_a)
    base = summarize_a(res_a[BASE_MIN_S])
    print("A. 時間門檻(與 §8.21 對照:NG1 26/4/5、NG2 16/5/5)")
    for g in grid_a:
        print(f"  min_s {g['min_s']:>4}  NG1 {g['ng1_tp']}/{g['ng1_fp']}/{g['ng1_fn']} F1 {g['ng1_F1']:.3f}  "
              f"NG2 {g['ng2_tp']}/{g['ng2_fp']}/{g['ng2_fn']} F1 {g['ng2_F1']:.3f}  接觸R {g['contact_R']:.3f}  "
              f"手沒碰到仍判接觸 {g['no_contact_hit']}/2")
    if not args.subset and (base["ng1_tp"], base["ng1_fp"], base["ng1_fn"],
                            base["ng2_tp"], base["ng2_fp"], base["ng2_fn"]) != (26, 4, 5, 16, 5, 5):
        print("!! min_s 1.0 與 §8.21 不一致,停止")
        return

    def pick_a(sub_pids):
        s = [dict(m=m, f=prf([r for r in res_a[m] if r["pid"] in sub_pids], "ng1")["F1"]) for m in MIN_S]
        return max(s, key=lambda x: (x["f"], -abs(x["m"] - BASE_MIN_S)))["m"]

    cv_rows, cv_pick = [], []
    for p in pids:
        m = pick_a(set(pids) - {p})
        cv_pick.append(dict(held_out=p, min_s=m))
        cv_rows += [r for r in res_a[m] if r["pid"] == p]
    cv_a = dict(**{f"ng1_{a}": b for a, b in prf(cv_rows, "ng1").items()},
                **{f"ng2_{a}": b for a, b in prf(cv_rows, "ng2").items()})
    best_a = pick_a(set(pids))
    adopt_a = cv_a["ng1_F1"] >= base["ng1_F1"] and cv_a["ng2_F1"] >= base["ng2_F1"] and best_a != BASE_MIN_S
    print(f"  全資料最佳 min_s {best_a};交叉驗證 NG1 F1 {cv_a['ng1_F1']}(現行 {base['ng1_F1']})"
          f" NG2 F1 {cv_a['ng2_F1']}(現行 {base['ng2_F1']});各折選值 {[x['min_s'] for x in cv_pick]}")
    print(f"  → 採用新值:{'是' if adopt_a else '否'}")
    E.write_csv("cv_min_s", cv_pick + [dict(held_out="合併", **cv_a)])

    # ── B. 回推 ──
    settings = [None] + [(g, mx) for g in GAP_S for mx in MAX_S]
    ms_list = [BASE_MIN_S] + ([best_a] if adopt_a else [])
    grid_b, res_b = [], {}
    for ms in ms_list:
        for bt in settings:
            rows = rows_of(evs, cfg, ms, bt)
            res_b[(ms, bt)] = rows
            grid_b.append(dict(min_s=ms, gap_s=bt[0] if bt else "", max_s=bt[1] if bt else "", **summarize_b(rows)))
    E.write_csv("grid_backtrack", grid_b)
    print("B. 回推(起點誤差 = 系統 − 團隊,秒)")
    for g in grid_b:
        print(f"  min_s {g['min_s']} gap {g['gap_s']!s:>3} max {g['max_s']!s:>2}  全部 med|e| {g.get('med_abs')} "
              f"±2s {g.get('within2')}  早起點 med|e| {g.get('early_med_abs')} mean {g.get('early_mean')}  "
              f"其他 med|e| {g.get('other_med_abs')} ±2s {g.get('other_within2')}  "
              f"NG1 {g['ng1_tp']}/{g['ng1_fp']}/{g['ng1_fn']} NG2 {g['ng2_tp']}/{g['ng2_fp']}/{g['ng2_fn']}")

    def pick_b(sub_pids, ms):
        sub = lambda bt: [r for r in res_b[(ms, bt)] if r["pid"] in sub_pids]
        b0 = summarize_b(sub(None))
        ok = [bt for bt in settings[1:] if summarize_b(sub(bt)).get("other_within2", 0) >= b0.get("other_within2", 0)]
        cand = [None] + ok
        return min(cand, key=lambda bt: (summarize_b(sub(bt)).get("med_abs", 99), bt is not None))

    ms = BASE_MIN_S
    cv_rows, cv_pick = [], []
    for p in pids:
        bt = pick_b(set(pids) - {p}, ms)
        cv_pick.append(dict(held_out=p, gap_s=bt[0] if bt else "", max_s=bt[1] if bt else ""))
        cv_rows += [r for r in res_b[(ms, bt)] if r["pid"] == p]
    cv_b = summarize_b(cv_rows)
    base_b = summarize_b(res_b[(ms, None)])
    best_b = pick_b(set(pids), ms)
    print(f"  全資料最佳 {best_b};交叉驗證 med|e| {cv_b.get('med_abs')}(基準 {base_b.get('med_abs')})"
          f" 其他 ±2s {cv_b.get('other_within2')}(基準 {base_b.get('other_within2')})"
          f" 早起點 med|e| {cv_b.get('early_med_abs')}(基準 {base_b.get('early_med_abs')})")
    E.write_csv("cv_backtrack", cv_pick + [dict(held_out="合併", **cv_b)])

    # 逐事件:基準與最佳回推
    per = []
    best_rows = {r["event"]: r for r in res_b[(ms, best_b)]}
    for r in res_b[(ms, None)]:
        b = best_rows[r["event"]]
        per.append(dict(event=r["event"], pid=r["pid"], early=int(r["event"] in EARLY), team_start=r["team_start"],
                        base_start=r["sys_start"], base_err=r["err"], bt_start=b["sys_start"], bt_err=b["err"],
                        gt_ng1=r["gt_ng1"], base_ng1=r["pred_ng1"], bt_ng1=b["pred_ng1"],
                        gt_ng2=r["gt_ng2"], base_ng2=r["pred_ng2"], bt_ng2=b["pred_ng2"]))
    E.write_csv("per_event", per)


if __name__ == "__main__":
    main()
