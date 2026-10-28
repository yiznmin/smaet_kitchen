"""垃圾桶 NG1 / NG2 事件層評估:每個團隊垃圾桶事件從完整影片切「事件前 1 秒 ~ 事件後 30 秒」,只用規則判定。

預先登記:docs/M6_NG準則評估_預先登記_20261027.md §8.8
- 影片:data/epfl_sessions/<參與者>/<場次>/<鏡頭>.mp4(不進 git)
- 片段:data/ng_trash_ev/<事件>/<鏡頭>.mp4;快取:results/det_cache/ng_trash_ev_s3/<事件>/<鏡頭>.npz(不進 git)
- 輸出:results/m6/ng/trash_ev/per_event.csv、summary.csv

用法:
    python scripts/eval_ng_trash_events.py --prepare   # 切片 + 快取
    python scripts/eval_ng_trash_events.py             # 評估
"""
import argparse
import csv
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import eval_ng_clips as E                                                 # noqa: E402
from fetch_ng_windows import FPS, VID_ROOT, VIEWS, cache                  # noqa: E402
from m3.classes import NAME_TO_CLASS                                      # noqa: E402
from m4_track.det_cache import DetCache                                   # noqa: E402
from m6_events.violation import HandHygieneFSM, at, in_poly, steps        # noqa: E402

EV_NG = ROOT / "results/ng_clips/team_events_vs_ng.csv"
CUT = ROOT / "data/ng_trash_ev"
CACHE = ROOT / "results/det_cache/ng_trash_ev_s3"
OUT = ROOT / "results/m6/ng/trash_ev"
PRE_S, POST_S = 1.0, 30.0
NG1_WASH_WITHIN_S = 5.0                 # 團隊 NG1:最後一次接觸後 5 秒內要洗手
EV_MATCH_S = 1.0                        # 觸發與標註區間相距 ≤ 1 秒才算這個事件的觸發
TOOL_CLASSES = ("砧板", "容器")          # 「用工具倒 / 刮」:這些物件中心在垃圾桶區
TOOL_THR, TOOL_FRAC = 0.4, 0.5
IGNORE_WASH = (1.0, 0.5)                # §8.10 B:主設定 1.0


def load_events():
    rows = list(csv.DictReader(open(EV_NG, encoding="utf-8-sig")))
    hit = [(r["session_id"], float(r["annot_start_sec"]), float(r["annot_end_sec"])) for r in rows if r["ng"] != "無NG"]
    out = []
    for r in rows:
        if r["event_category"] != "垃圾桶" or r["session_id"] == "test/YH2003/2023_05_17_09_08_58":
            continue
        a, b = float(r["annot_start_sec"]), float(r["annot_end_sec"])
        if r["ng"] == "無NG" and any(s == r["session_id"] and min(b, y) > max(a, x) for s, x, y in hit):
            continue                                             # §8.4 重複標註
        pid, sess = r["session_id"].split("/")[1:]
        f0 = int(max(0.0, a - PRE_S) * FPS)
        out.append(dict(key=f"{pid}_{sess}_{r['annotation_id'][:8]}", pid=pid, session=r["session_id"],
                        start=a, end=b, f0=f0, dur=b + POST_S - f0 / FPS, ng=r["ng"], note=r["note"],
                        m3_train=int(any(x in r["session_id"] for x in ("YH2040/2023_09_12_09_50_41",)))))
    return out


def prepare(events):
    for e in events:
        src_dir = VID_ROOT / e["session"].split("/", 1)[1]
        for v in VIEWS:
            src, out = src_dir / f"{v}.mp4", CUT / e["key"] / f"{v}.mp4"
            if out.exists() or (CACHE / e["key"] / f"{v}.npz").exists() or not src.exists():
                continue
            out.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{(e['f0'] - 0.5) / FPS:.4f}", "-i", str(src),
                            "-t", f"{e['dur']:.3f}", "-an", "-c:v", "libx264", "-preset", "veryfast",
                            "-crf", "18", "-r", "30", str(out)], check=True)
        cache([CUT / e["key"] / f"{v}.mp4" for v in VIEWS], list(VIEWS), CACHE / e["key"])


def tool_in_zone(c, T, poly):
    ids = [NAME_TO_CLASS[n] for n in TOOL_CLASSES]
    idx, out = steps(c), np.zeros(T, bool)
    for i in range(T):
        xy, cf, cl = at(c, idx, i)
        b = xy[np.isin(cl, ids) & (cf >= TOOL_THR)]
        if len(b):
            out[i] = in_poly(np.c_[(b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2], poly).any()
    return out


def judge(e, cfg):
    step, hh = cfg["step_s"], cfg["hand_hygiene"]
    caches = {v: CACHE / e["key"] / f"{v}.npz" for v in VIEWS}
    if not all(p.exists() for p in caches.values()):
        return None
    c = dict(key=e["key"], caches=caches)
    r = E.rule_part(c, cfg)
    T, wash, touch = r["T"], r["wash"], r["touch"]
    w0 = e["f0"] / FPS
    s_ev, e_ev = int((e["start"] - w0) / step), int((e["end"] - w0) / step)
    m = int(EV_MATCH_S / step)
    tool = tool_in_zone(DetCache(caches["Aoutput1"]), T, E.get_zone(E.INTEG / "configs/zones/epfl_Aoutput1.json",
                                                                       cfg["trash_event"]["zone"]))
    fsm = HandHygieneFSM(hh["wash_s"], step, cfg["touch_classes"])
    grace = int(round(hh["grace_s"] / step))
    row = dict(key=e["key"], pid=e["pid"], ng=e["ng"], note=e["note"], m3_train=e["m3_train"],
               gt_ng1=int("NG1" in e["ng"]), gt_ng2=int("NG2" in e["ng"]), T=T)
    for name, trig in (("base", r["trig_base"]), ("mv", r["trig_mv"])):
        for variant in ("all", "notool"):
            tr = [(s, t) for s, t in (trig or []) if s <= e_ev + m and t >= s_ev - m]
            if variant == "notool":
                tr = [(s, t) for s, t in tr if tool[s:t].mean() < TOOL_FRAC]
            k = f"{name}_{variant}"
            if not tr:
                row.update({f"{k}_contam": 0, f"{k}_ng1": 0, f"{k}_ng2": 0})
                if k == "base_all":
                    row.update({f"base_iw{iw}_ng2": 0 for iw in IGNORE_WASH})
                continue
            end = tr[-1][1]                                      # 事件內最後一次接觸
            dirty = end + grace
            washed5 = wash[dirty:end + int(NG1_WASH_WITHIN_S / step)].sum() >= fsm.need
            v = fsm.run(dirty, wash, touch)
            row.update({f"{k}_contam": 1, f"{k}_ng1": int(not washed5), f"{k}_ng2": int(v.result == "違規")})
            if k == "base_all":                                  # §8.10 B:洗手中碰觸不計
                for iw in IGNORE_WASH:
                    f2 = HandHygieneFSM(hh["wash_s"], step, cfg["touch_classes"], ignore_wash_s=iw)
                    row[f"base_iw{iw}_ng2"] = int(f2.run(dirty, wash, touch).result == "違規")
    row["tool_frac"] = round(float(tool[max(0, s_ev - m):e_ev + m].mean()), 2)
    return row


def prf(rows, pred, gt):
    tp = sum(r[pred] and r[gt] for r in rows)
    fp = sum(r[pred] and not r[gt] for r in rows)
    fn = sum(r[gt] and not r[pred] for r in rows)
    R, P = tp / max(tp + fn, 1), tp / max(tp + fp, 1)
    (rl, rh), (pl, ph) = E.wilson(tp, tp + fn), E.wilson(tp, tp + fp)
    return dict(tp=tp, fp=fp, fn=fn, tn=len(rows) - tp - fp - fn, R=round(R, 3), R_ci=f"{rl}-{rh}",
                P=round(P, 3), P_ci=f"{pl}-{ph}", F1=round(2 * P * R / max(P + R, 1e-9), 3))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepare", action="store_true")
    args = ap.parse_args()
    events = load_events()
    print(f"垃圾桶事件 {len(events)}:{dict(Counter(e['ng'] for e in events))}", flush=True)
    if args.prepare:
        prepare(events)
        print("全部完成", flush=True)
        return
    cfg = yaml.safe_load((E.INTEG / "configs/violation_rules.yaml").read_text(encoding="utf-8"))
    rows = [x for x in (judge(e, cfg) for e in events) if x is not None]
    print(f"有快取 {len(rows)} / {len(events)}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    E.OUT = OUT
    E.write_csv("per_event", rows)
    summ = []
    for subset, data in (("全部", rows), ("排除M3訓練場次", [r for r in rows if not r["m3_train"]])):
        for name in ("base", "mv"):
            for variant in ("all", "notool"):
                k = f"{name}_{variant}"
                for gt, pred in (("NG1-team", f"{k}_ng1"), ("NG2(=NG1-order)", f"{k}_ng2"), ("污染(任一 NG)", f"{k}_contam")):
                    g = {"NG1-team": "gt_ng1", "NG2(=NG1-order)": "gt_ng2"}.get(gt)
                    data2 = [dict(r, _gt=r[g] if g else int(r["gt_ng1"] or r["gt_ng2"])) for r in data]
                    m = prf(data2, pred, "_gt")
                    summ.append(dict(subset=subset, trigger=name, variant=variant, gt=gt, n=len(data2), **m))
                    if subset == "全部":
                        print(f"  {k:12s} {gt:16s} R {m['R']:.2f} P {m['P']:.2f} F1 {m['F1']:.2f} "
                              f"({m['tp']}/{m['fp']}/{m['fn']})", flush=True)
        for iw in IGNORE_WASH:
            m = prf([dict(r, _gt=r["gt_ng2"]) for r in data], f"base_iw{iw}_ng2", "_gt")
            summ.append(dict(subset=subset, trigger="base", variant=f"ignore_wash_{iw}", gt="NG2(=NG1-order)", n=len(data), **m))
            if subset == "全部":
                print(f"  base iw{iw}     NG2              R {m['R']:.2f} P {m['P']:.2f} F1 {m['F1']:.2f} "
                      f"({m['tp']}/{m['fp']}/{m['fn']})", flush=True)
    E.write_csv("summary", summ)


if __name__ == "__main__":
    main()
