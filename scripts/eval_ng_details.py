"""用團隊細部時間軸(ng_instances.csv)逐環節比對:接觸污染源、洗手、碰觸;並重算 NG1 / NG2。

預先登記:docs/M6_NG準則評估_預先登記_20261027.md §8.15
快取沿用 §8.8:results/det_cache/ng_trash_ev_s3/<事件>/<鏡頭>.npz
輸出 results/m6/ng/details/{details.csv, extra.csv, events.csv, summary.csv}(含團隊標註,不進 git)

用法:
    python scripts/eval_ng_details.py
"""
import argparse
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import eval_ng_clips as E                                                 # noqa: E402
import eval_ng_trash_events as M                                          # noqa: E402
from m6_events.violation import HandHygieneFSM, intervals                 # noqa: E402

INST = Path("D:/yizhen/NG_clips/ng_instances.csv")
ANSWER = ROOT / "results/m6/ng/details/team_questions_result.csv"     # §8.20 團隊回覆
OUT = ROOT / "results/m6/ng/details"
TOL_S, TOUCH_GAP_S, TOUCH_MIN_S = 1.0, 1.0, 0.3
KINDS = {"接觸污染源": "contact", "洗手": "wash", "手接觸食材、餐具或廚具": "touch"}
LINE = re.compile(r"^[＊　\s]*(\d+:\d+:\d+\.\d+)(?:–(\d+:\d+:\d+\.\d+))?\s+(.+)$")


def sec(hms):
    h, m, s = hms.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def load_instances():
    """annotation_id 前 8 碼 -> dict(details=[(kind, a, b)], scope_end, ng1, ng1_ok, ng2)。"""
    out = {}
    for r in csv.DictReader(open(INST, encoding="utf-8-sig")):
        k = r["annotation_id"][:8]
        d = out.setdefault(k, dict(details=None, scope_end=float(r["event_end_sec"]), ng1=0, ng1_ok=0, ng2=0))
        d["scope_end"] = max(d["scope_end"], float(r["end_sec"]))
        if r["ng_type"] == "NG1":
            d["ng1" if r["result"] == "NG" else "ng1_ok"] = 1
        if r["ng_type"] == "NG2" and r["result"] == "NG":
            d["ng2"] = 1
        if d["details"] is None:
            det = []
            for line in r["event_timeline"].splitlines():
                m = LINE.match(line.strip())
                if not m:
                    continue
                kind = next((v for key, v in KINDS.items() if key in m.group(3)), None)
                if kind:
                    a = sec(m.group(1))
                    det.append((kind, a, sec(m.group(2)) if m.group(2) else a))
            d["details"] = det
    return out


def load_corrections(events):
    """§8.20:團隊回覆 → {事件 8 碼: dict(ng1=0/1, ng2=0/1/None 表示排除, exclude=bool, note)}。"""
    out = {}
    for r in csv.DictReader(open(ANSWER, encoding="utf-8-sig")):
        ans = list(r.values())[-1] or ""
        a = sec(r["事件時間"].split("–")[0])
        e = next(x for x in events if x["session"] == r["場次"] and abs(x["start"] - a) < 0.2)
        k = e["key"][-8:]
        if "略過" in ans:
            out[k] = dict(exclude=True, note="略過")
        elif "系統標對" in ans or "應為接觸汙染源" in ans:
            out[k] = dict(ng1=1, ng2=None, note="團隊改 NG1 正例")
        elif "應該是NG2" in ans:
            out[k] = dict(ng2=1, note="團隊改 NG2 正例")
    return out


def touch_ivs(touch_any):
    gap, mn = int(round(TOUCH_GAP_S / 0.1)), int(round(TOUCH_MIN_S / 0.1))
    merged = []
    for s, e in intervals(touch_any):
        if merged and s - merged[-1][1] < gap:
            merged[-1][1] = e
        else:
            merged.append([s, e])
    return [(s, e) for s, e in merged if e - s >= mn]


def hit(a0, a1, b0, b1):
    return min(a1, b1) + 2 * TOL_S >= max(a0, b0)


def q(x):
    if not len(x):
        return dict(n=0)
    p = np.percentile(x, [25, 50, 75])
    return dict(n=len(x), med=round(p[1], 2), q1=round(p[0], 2), q3=round(p[2], 2),
                within1=round(float(np.mean(np.abs(x) <= 1)), 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corrected", action="store_true", help="§8.20:依團隊回覆修正真值、NG2 只算事件時間內")
    args = ap.parse_args()
    out_dir = OUT.parent / "details_corrected" if args.corrected else OUT
    cfg = yaml.safe_load((E.INTEG / "configs/violation_rules.yaml").read_text(encoding="utf-8"))
    step, hh = cfg["step_s"], cfg["hand_hygiene"]
    inst = load_instances()
    det_rows, extra_rows, ev_rows = [], [], []
    events = M.load_events()
    corr = load_corrections(events) if args.corrected else {}
    for e in events:
        k = e["key"][-8:]
        c = dict(key=e["key"], caches={v: M.CACHE / e["key"] / f"{v}.npz" for v in M.VIEWS})
        r = E.rule_part(c, cfg)
        w0 = e["f0"] / M.FPS
        tsec = lambda i: w0 + i * step
        sysiv = dict(contact=[(tsec(a), tsec(b)) for a, b in (r["trig_base"] or [])],
                     wash=[(tsec(a), tsec(b)) for a, b in intervals(r["wash"])],
                     touch=[(tsec(a), tsec(b)) for a, b in touch_ivs(r["touch"].any(1))])
        info = inst.get(k)
        scope_end = info["scope_end"] if info else e["end"]

        # A. 逐環節(只有時間軸的事件)
        if info:
            for kind, a, b in info["details"]:
                m = [s for s in sysiv[kind] if hit(a, b, *s)]
                det_rows.append(dict(event=k, kind=kind, team_start=round(a, 2), team_end=round(b, 2), hit=int(bool(m)),
                                     d_start=round(m[0][0] - a, 2) if m else ""))
            team = defaultdict(list)
            for kind, a, b in info["details"]:
                team[kind].append((a, b))
            c_end = max((b for a, b in team["contact"]), default=e["end"])
            w_start = min((a for a, b in team["wash"] if a >= c_end), default=scope_end)
            for kind in ("contact", "wash", "touch"):
                lo, hi = (c_end, w_start) if kind == "touch" else (e["start"], scope_end)
                for s0, s1 in sysiv[kind]:
                    if s1 < lo or s0 > hi:
                        continue
                    if not any(hit(s0, s1, a, b) for a, b in team[kind]):
                        extra_rows.append(dict(event=k, kind=kind, sys_start=round(s0, 2), sys_end=round(s1, 2),
                                               team_ng=e["ng"], note=e["note"]))

        # B. NG1 / NG2 重算
        s_ev, e_ev = int((e["start"] - w0) / step), int((e["end"] - w0) / step)
        tr = [(s, x) for s, x in (r["trig_base"] or []) if s <= e_ev + 10 and x >= s_ev - 10]
        p1 = p2 = 0
        at = ""
        if tr:
            end = tr[-1][1]
            dirty = end + int(round(hh["grace_s"] / step))
            fsm = HandHygieneFSM(hh["wash_s"], step, cfg["touch_classes"])
            p1 = int(r["wash"][dirty:end + int(M.NG1_WASH_WITHIN_S / step)].sum() < fsm.need)
            v = fsm.run(dirty, r["wash"], r["touch"])
            p2 = int(v.result == "違規" and tsec(v.at) <= scope_end + TOL_S)
            at = round(tsec(v.at), 2) if v.result == "違規" else ""
        row = dict(event=k, note=e["note"], team_ng=e["ng"], has_timeline=int(bool(info)),
                   gt_ng1=info["ng1"] if info else 0, gt_ng1_ok=info["ng1_ok"] if info else 0,
                   gt_ng2=info["ng2"] if info else 0, pred_ng1=p1, pred_ng2=p2, m3_train=e["m3_train"],
                   ev_end=e["end"], ng2_at=at, correction="")
        c = corr.get(k)
        if c and c.get("exclude"):
            continue
        if c:
            row["correction"] = c["note"]
            if "ng1" in c:
                row["gt_ng1"] = c["ng1"]
            if "ng2" in c:
                row["gt_ng2"] = c["ng2"]
        if args.corrected:
            # 團隊:NG2 只算事件時間內的碰觸(主分析容忍 1 秒;另報 0 秒、舊範圍)
            row["pred_ng2_old"] = p2
            row["pred_ng2_t0"] = int(at != "" and at <= e["end"])
            row["pred_ng2"] = int(at != "" and at <= e["end"] + TOL_S)
        ev_rows.append(row)

    out_dir.mkdir(parents=True, exist_ok=True)
    E.OUT = out_dir
    E.write_csv("details", det_rows)
    E.write_csv("extra", extra_rows)
    E.write_csv("events", ev_rows)
    summ = []
    print("A. 逐環節(37 個有時間軸的事件)")
    for kind in ("contact", "wash", "touch"):
        d = [x for x in det_rows if x["kind"] == kind]
        k_ = sum(x["hit"] for x in d)
        lo, hi = E.wilson(k_, len(d))
        tq = q(np.array([x["d_start"] for x in d if x["hit"]], float))
        n_extra = sum(x["kind"] == kind for x in extra_rows)
        summ.append(dict(part="A", item=kind, n=len(d), hit=k_, R=round(k_ / max(len(d), 1), 3), R_ci=f"{lo}-{hi}",
                         extra=n_extra, **{f"dstart_{a}": b for a, b in tq.items()}))
        print(f"  {kind:8s} 團隊 {len(d):3d}  抓到 {k_:3d}(R {k_ / max(len(d), 1):.2f},{lo}–{hi})  "
              f"起點差 {tq}  範圍內多出 {n_extra}")
    print("B. NG1 / NG2 重算(53 個事件)")
    subsets = [("全部", ev_rows), ("排除M3訓練場次", [x for x in ev_rows if not x["m3_train"]])]
    if args.corrected:
        subsets.append(("排除031d4d5b", [x for x in ev_rows if x["event"] != "031d4d5b"]))
    preds = [("ng1", "pred_ng1"), ("ng2", "pred_ng2")]
    if args.corrected:
        preds += [("ng2", "pred_ng2_t0"), ("ng2", "pred_ng2_old")]
    for name, sub0 in subsets:
        for ng, pk in preds:
            sub = [x for x in sub0 if x[f"gt_{ng}"] is not None]
            tp = sum(x[pk] and x[f"gt_{ng}"] for x in sub)
            fp = sum(x[pk] and not x[f"gt_{ng}"] for x in sub)
            fn = sum(x[f"gt_{ng}"] and not x[pk] for x in sub)
            R, P = tp / max(tp + fn, 1), tp / max(tp + fp, 1)
            (rl, rh), (pl, ph) = E.wilson(tp, tp + fn), E.wilson(tp, tp + fp)
            summ.append(dict(part="B", item=f"{pk}_{name}", n=len(sub), hit=tp, R=round(R, 3), R_ci=f"{rl}-{rh}",
                             P=round(P, 3), P_ci=f"{pl}-{ph}", fp=fp, fn=fn))
            print(f"  {name} {pk:13s} n {len(sub)}  TP {tp} FP {fp} FN {fn}  R {R:.2f}({rl}–{rh})  P {P:.2f}({pl}–{ph})")
        ok = [x for x in sub0 if x["gt_ng1_ok"]]
        print(f"  {name} NG1 合格反例 {len(ok)} 個,系統判 NG1:{sum(x['pred_ng1'] for x in ok)}")
    E.write_csv("summary", summ)


if __name__ == "__main__":
    main()
