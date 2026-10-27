"""M6 違規先後順序(整場):丟垃圾 → 洗手? → 碰食材 / 器具。

預先登記:docs/M6_違規先後順序_預先登記_20261027.md §2
真值由 EPFL 逐秒動作標註自動推出;系統全部用 integrated/ 模組與凍結設定。

用法:
    python scripts/eval_violation_sequence.py --session yh2040
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
INTEG = ROOT / "integrated"
sys.path.insert(0, str(INTEG / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.append(str(ROOT / "src"))                       # 只為 m6_assoc.gt

from common.zones import get_zone                                         # noqa: E402
from m4_track.det_cache import DetCache                                   # noqa: E402
from m6_events.smoothing import hysteresis                                # noqa: E402
from m6_events.violation import (HandHygieneFSM, intervals, side_series, steps, touch_series,  # noqa: E402
                                 trash_intervals, zone_any_hand_series, zone_body_series)
from m6_assoc.gt import CONTACT_VERBS, NOUN_TO_CLASS, load_actions       # noqa: E402
import eval_multiview_triggers as MV                                      # noqa: E402

SESSIONS = {"yh2040": dict(cache=ROOT / "results/det_cache/epfl_allcls_armA_s3",
                           xlsx=ROOT / "data/epfl_ann/YH2040/2023_09_12_09_50_41/annotations/actions_annotations.xlsx")}
OUT = ROOT / "results/m6/sequence"
HORIZON_S, TOL, DRY_S, MERGE_GAP = 120.0, 2.0, 15.0, 10
MV_CFG = dict(arm="kofn_strict", k=3, thr_hand=0.2, T=0.5)


# ── 真值 ─────────────────────────────────────────────────────────

def gt_sequences(acts, touch_classes):
    """每個 Put Trash → 合格 / 違規 / 未解除(登記 §2.1)。"""
    acts = sorted(acts, key=lambda a: a["start"])
    trig = [a for a in acts if a["verb"] == "Put" and a["noun"] == "Trash"]
    out = []
    for k, t in enumerate(trig):
        lim = min(t["end"] + HORIZON_S, trig[k + 1]["start"] if k + 1 < len(trig) else 1e9)
        res, at, what = "未解除", None, ""
        for a in acts:
            if a["start"] < t["end"] or a["start"] >= lim:
                continue
            if a["verb"] == "Wash" and a["noun"] == "Hand":
                res, at, what = "合格", a["start"], "Wash Hand"
                break
            if a["noun"] in ("Hand",) or a["verb"] not in CONTACT_VERBS:
                continue
            if NOUN_TO_CLASS.get(a["noun"]) in touch_classes:
                res, at, what = "違規", a["start"], f"{a['verb']} {a['noun']}"
                break
        out.append(dict(start=t["start"], end=t["end"], gt=res, gt_at=at, gt_what=what))
    return out


# ── 系統 ─────────────────────────────────────────────────────────

def wash_episode_ends(wash, need):
    iv, merged = intervals(wash), []
    for s, e in iv:
        if merged and s - merged[-1][1] < MERGE_GAP:
            merged[-1][1] = e
        else:
            merged.append([s, e])
    return [e for s, e in merged if wash[s:e].sum() >= need]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", choices=list(SESSIONS), required=True)
    args = ap.parse_args()
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    S = SESSIONS[args.session]
    cfg = yaml.safe_load((INTEG / "configs/violation_rules.yaml").read_text(encoding="utf-8"))
    step, cams, hh, te = cfg["step_s"], cfg["cameras"], cfg["hand_hygiene"], cfg["trash_event"]
    views = list(dict.fromkeys(["output0"] + MV.TRASH_VIEWS + cams["close"]["A"] + cams["close"]["B"]
                               + hh["close_cams"]))
    cs = {v: DetCache(S["cache"] / f"{v}.npz") for v in views}
    T = len(steps(cs["output0"])[0])
    zone = lambda v, n: get_zone(INTEG / f"configs/zones/epfl_{v}.json", n)

    gt = gt_sequences(load_actions(S["xlsx"]), cfg["touch_classes"])
    print(f"{args.session}:{T * step:.0f} 秒;丟垃圾 {len(gt)}(真值 {[g['gt'] for g in gt]})", flush=True)

    # 零件
    side = side_series(cs[cams["overview"]], T, **cams["side_x"])
    touch = touch_series(cs, T, side, cams["close"], cfg["touch_classes"], cfg["association"])
    wash = np.any([zone_any_hand_series(cs[v], T, zone(v, hh["sink_zone"]), hh["thr_hand"])
                   for v in hh["close_cams"]], 0)
    print(f"零件完成 {time.time() - t0:.0f} 秒;碰觸成立 {touch.any(1).sum() * step:.0f} 秒、洗手 {wash.sum() * step:.0f} 秒",
          flush=True)
    need = max(1, int(round(hh["wash_s"] / step)))
    ep_ends = wash_episode_ends(wash, need)

    trig = {"base": trash_intervals(cs["Aoutput1"], cs["output0"], T, zone("Aoutput1", te["zone"]),
                                    zone("output0", te["zone"]), te, step)}
    tz = {v: zone(v, te["zone"]) for v in MV.TRASH_VIEWS}
    d = dict(T=T, hand={(v, MV_CFG["thr_hand"]): zone_any_hand_series(cs[v], T, tz[v], MV_CFG["thr_hand"])
                        for v in MV.TRASH_VIEWS}, body=None)
    sm = hysteresis(MV.raw_of(d, MV_CFG), *MV.EV_SMOOTH)
    trig["mv"] = [(s, e) for s, e in intervals(sm) if e - s >= max(1, int(round(MV_CFG["T"] / step)))]

    fsm = HandHygieneFSM(hh["wash_s"], step, cfg["touch_classes"])
    grace = int(round(hh["grace_s"] / step))
    rows, summ = [], []
    for tname, ivs in trig.items():
        for dry in (0, 1):
            kept = [(s, e) for s, e in ivs
                    if not (dry and any(0 <= s - x <= DRY_S / step for x in ep_ends))]
            res = []
            for k, (s, e) in enumerate(kept):
                lim = min(e + int(HORIZON_S / step), kept[k + 1][0] if k + 1 < len(kept) else T, T)
                v = fsm.run(e + grace, wash[:lim], touch[:lim])
                g = [i for i, x in enumerate(gt) if s * step <= x["end"] + TOL and e * step >= x["start"] - TOL]
                res.append(dict(trigger=tname, dry_filter=dry, start_s=round(s * step, 1), end_s=round(e * step, 1),
                                pred=v.result, at_s=None if v.at is None else round(v.at * step, 1),
                                touched="、".join(v.touched), gt_idx=g[0] if g else "",
                                gt=gt[g[0]]["gt"] if g else "", gt_at=gt[g[0]]["gt_at"] if g else "",
                                gt_what=gt[g[0]]["gt_what"] if g else ""))
            matched = {r["gt_idx"] for r in res if r["gt_idx"] != ""}
            hit = [r for r in res if r["gt_idx"] != ""]
            correct = sum(r["pred"] == r["gt"] for r in hit)
            gt_v = [i for i, x in enumerate(gt) if x["gt"] == "違規"]
            caught = sum(any(r["gt_idx"] == i and r["pred"] == "違規" for r in res) for i in gt_v)
            false_v = sum(r["gt_idx"] == "" and r["pred"] == "違規" for r in res)
            dt = [abs(r["at_s"] - r["gt_at"]) for r in hit if r["pred"] == "違規" and r["gt"] == "違規"]
            hours = T * step / 3600
            summ.append(dict(trigger=tname, dry_filter=dry, n_trig=len(res), gt_matched=len(matched), n_gt=len(gt),
                             correct=correct, violations_caught=caught, n_gt_violation=len(gt_v),
                             false_violation=false_v, false_violation_per_h=round(false_v / hours, 1),
                             unresolved=sum(r["pred"] == "未解除" for r in res),
                             at_err_median_s=round(float(np.median(dt)), 1) if dt else ""))
            rows += res
            print(f"  {tname:4s} 擦手排除 {'開' if dry else '關'}:觸發 {len(res)}、對到真值 {len(matched)}/{len(gt)}、"
                  f"違規抓到 {caught}/{len(gt_v)}、假違規 {false_v}(每小時 {false_v / hours:.1f})、"
                  f"未解除 {summ[-1]['unresolved']}", flush=True)

    for name, data in (("triggers", rows), ("summary", summ),
                       ("gt", [dict(i=i, **g) for i, g in enumerate(gt)])):
        with open(OUT / f"{args.session}_{name}.csv", "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in data for k in r)))
            w.writeheader()
            w.writerows(data)
    print(f"輸出 {OUT},{time.time() - t0:.0f} 秒")


if __name__ == "__main__":
    main()
