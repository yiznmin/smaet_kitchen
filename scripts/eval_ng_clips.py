"""NG 準則評估:團隊 NG 片段(只有正例)上的 Recall 與跨類誤觸發。

預先登記:docs/M6_NG準則評估_預先登記_20261027.md §8.1
全部用凍結值,不調參數。VLM 分數存 results/m6/ng/vlm/<片段>.json,重跑會沿用。

用法:
    python scripts/eval_ng_clips.py
    python scripts/eval_ng_clips.py --no-vlm        # 只跑偵測規則部分
"""
import argparse
import csv
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
INTEG = ROOT / "integrated"
sys.path.insert(0, str(INTEG / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from common.zones import get_zone                                         # noqa: E402
from m3.classes import NAME_TO_CLASS                                      # noqa: E402
from m4_track.det_cache import DetCache                                   # noqa: E402
from m6_events.association import ioa                                     # noqa: E402
from m6_events.smoothing import hysteresis                                # noqa: E402
from m6_events.violation import (HAND, HandHygieneFSM, at, intervals, side_series, steps,  # noqa: E402
                                 touch_series, trash_intervals, zone_any_hand_series)
import eval_multiview_triggers as MV                                      # noqa: E402

NG_DIR = Path("D:/yizhen/NG_clips")
CACHE = ROOT / "results/det_cache/ng_clips_s3"
OUT = ROOT / "results/m6/ng"
VLM_OUT = OUT / "vlm"
MV_CFG = dict(arm="kofn_strict", k=3, thr_hand=0.2, T=0.5)     # 多鏡頭凍結(§3)
VLM_VIEWS = ["Aoutput1", "output0"]
PROMPTS = {
    "eat": ("Watch this kitchen video. Does the person put food or anything into their own mouth "
            "(eating or tasting)? Answer only yes or no."),                      # 事件測試凍結
    "contam": ("Watch this kitchen video. Does the person touch a phone, wipe their nose with a tissue, "
               "pick something up from the floor, or wipe the floor? Answer only yes or no."),
    "ng4": ("Watch this kitchen video. Does the person put washed food or a utensil directly on the bare "
            "countertop (not in a bowl, pot or on a cutting board)? Answer only yes or no."),
}
THR = dict(eat=0.3, contam=0.5, ng4=0.5)
NG4_OBJ_THR, NG4_WASH_S, NG4_PLACE_S, NG4_SUPPORT_IOA = 0.4, 1.0, 1.0, 0.3
TOUCH_GAP_S, TOUCH_MIN_S = 1.0, 0.3


def load_clips():
    clips = {}
    for r in csv.DictReader(open(NG_DIR / "ng_clips_index.csv", encoding="utf-8-sig")):
        pid, sess = r["session_id"].split("/")[1:]
        no = re.search(r"_(ng\d+)_", r["clip_file"]).group(1)
        k = f"{pid}_{sess}_{no}"
        c = clips.setdefault(k, dict(key=k, ng=r["ng_type"], cat=r["category"], note=r["note"],
                                     dur=float(r["clip_duration_sec"]), files={}))
        if (CACHE / k / f"{r['view']}.npz").exists():
            c["files"][r["view"]] = NG_DIR / r["clip_file"]
    return {k: c for k, c in clips.items() if c["files"]}


def wilson(k, n, z=1.96):
    if n == 0:
        return ("", "")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return round(c - h, 2), round(c + h, 2)


# ── VLM ─────────────────────────────────────────────────────────

def run_vlm(clips):
    todo = [c for c in clips.values() if not (VLM_OUT / f"{c['key']}.json").exists()]
    if not todo:
        return
    import torch
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    from m7_event_vlm import EVERY_S, MODEL, read_frames
    proc = AutoProcessor.from_pretrained(MODEL)
    model = Qwen3VLForConditionalGeneration.from_pretrained(MODEL, dtype=torch.bfloat16).to("cuda").eval()
    tok = proc.tokenizer
    yes = [tok.encode(w, add_special_tokens=False)[0] for w in ("yes", "Yes")]
    no = [tok.encode(w, add_special_tokens=False)[0] for w in ("no", "No")]
    texts = {k: proc.apply_chat_template([{"role": "user", "content": [{"type": "video"}, {"type": "text", "text": p}]}],
                                         tokenize=False, add_generation_prompt=True) for k, p in PROMPTS.items()}
    VLM_OUT.mkdir(parents=True, exist_ok=True)
    for n, c in enumerate(todo, 1):
        res = {}
        for v in VLM_VIEWS:
            if v not in c["files"]:
                continue
            fr = read_frames(c["files"][v])
            meta = dict(fps=1 / EVERY_S, total_num_frames=len(fr), frames_indices=list(range(len(fr))))
            for k, text in texts.items():
                inp = proc(text=[text], videos=[fr], video_metadata=[meta], do_sample_frames=False,
                           cap_pixels_per_frame=True, return_tensors="pt").to("cuda")
                with torch.inference_mode():
                    logits = model(**inp).logits[0, -1].float()
                p = torch.softmax(logits[yes + no], 0)
                res[f"{k}_{v}"] = round(float(p[:len(yes)].sum()), 4)
        (VLM_OUT / f"{c['key']}.json").write_text(json.dumps(res), encoding="utf-8")
        print(f"  VLM {n}/{len(todo)} {c['key']} {res}", flush=True)


def vlm_flag(c, kind):
    p = VLM_OUT / f"{c['key']}.json"
    if not p.exists():
        return None, {}
    s = json.loads(p.read_text(encoding="utf-8"))
    sc = {v: s[f"{kind}_{v}"] for v in VLM_VIEWS if f"{kind}_{v}" in s}
    return (max(sc.values()) >= THR[kind]) if sc else None, sc


# ── 偵測規則 ─────────────────────────────────────────────────────

def ng4_rule(cs, T, sink, step):
    """清洗(手 + 物件同在洗手台)→ 物件放在洗手台外、不在手中、不在砧板 / 容器上 ≥ NG4_PLACE_S。"""
    objs = [NAME_TO_CLASS[n] for n in ("食材", "刀具", "砧板", "鍋鏟", "鍋子", "容器")]
    sup_ids = [NAME_TO_CLASS[n] for n in ("砧板", "容器")]
    bcams = [v for v in ("Boutput1", "Boutput2") if v in cs and v in sink]
    if not bcams:
        return None, ""
    from m6_events.violation import in_poly
    wash = np.zeros((T, len(objs)), bool)
    for v in bcams:
        idx = steps(cs[v])
        for i in range(T):
            xy, cf, cl = at(cs[v], idx, i)
            h = xy[(cl == HAND) & (cf >= 0.2)]
            if not len(h) or not in_poly(np.c_[(h[:, 0] + h[:, 2]) / 2, (h[:, 1] + h[:, 3]) / 2], sink[v]).any():
                continue
            for j, oid in enumerate(objs):
                b = xy[(cl == oid) & (cf >= NG4_OBJ_THR)]
                if len(b) and in_poly(np.c_[(b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2], sink[v]).any():
                    wash[i, j] = True
    need_w, need_p = int(round(NG4_WASH_S / step)), int(round(NG4_PLACE_S / step))
    washed = [j for j in range(len(objs)) if wash[:, j].sum() >= need_w]
    if not washed:
        return False, "沒偵測到清洗"
    pcams = [v for v in ("Aoutput1", "Aoutput2", "Aoutput3", "Boutput1", "Boutput2") if v in cs]
    for j in washed:
        end = int(np.flatnonzero(wash[:, j]).max()) + 1
        placed = np.zeros(T, bool)
        for v in pcams:
            idx = steps(cs[v])
            for i in range(end, T):
                xy, cf, cl = at(cs[v], idx, i)
                b = xy[(cl == objs[j]) & (cf >= NG4_OBJ_THR)]
                if not len(b):
                    continue
                if v in sink:
                    b = b[~in_poly(np.c_[(b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2], sink[v])]
                h = xy[(cl == HAND) & (cf >= 0.2)]
                if len(b) and len(h):
                    b = b[ioa(b, h).max(1) == 0]
                sup = xy[np.isin(cl, [s for s in sup_ids if s != objs[j]]) & (cf >= NG4_OBJ_THR)]
                if len(b) and len(sup):
                    b = b[ioa(b, sup).max(1) < NG4_SUPPORT_IOA]
                placed[i] |= len(b) > 0
        if any(e - s >= need_p for s, e in intervals(placed)):
            return True, f"清洗並放桌面:{[k for k, x in NAME_TO_CLASS.items() if x == objs[j]][0]}"
    return False, "有清洗、沒偵測到放桌面"


def rule_part(c, cfg):
    """偵測規則部分:污染觸發(垃圾桶)、洗手、碰觸、NG4 規則。"""
    step, cams, hh, te = cfg["step_s"], cfg["cameras"], cfg["hand_hygiene"], cfg["trash_event"]
    cs = {v: DetCache(CACHE / c["key"] / f"{v}.npz") for v in c["files"]}
    T = min(len(steps(x)[0]) for x in cs.values())
    zone = lambda v, n: get_zone(INTEG / f"configs/zones/epfl_{v}.json", n)
    r = dict(T=T)
    # 垃圾桶觸發
    if all(v in cs for v in MV.TRASH_VIEWS):
        d = dict(T=T, hand={(v, MV_CFG["thr_hand"]): zone_any_hand_series(cs[v], T, zone(v, "垃圾桶區"),
                                                                           MV_CFG["thr_hand"]) for v in MV.TRASH_VIEWS},
                 body=None)
        sm = hysteresis(MV.raw_of(d, MV_CFG), *MV.EV_SMOOTH)
        r["trig_mv"] = [(s, e) for s, e in intervals(sm) if e - s >= max(1, int(round(MV_CFG["T"] / step)))]
    else:
        r["trig_mv"] = None
    r["trig_base"] = (trash_intervals(cs["Aoutput1"], cs["output0"], T, zone("Aoutput1", te["zone"]),
                                      zone("output0", te["zone"]), te, step)
                      if {"Aoutput1", "output0"} <= set(cs) else None)
    # 洗手、碰觸
    r["wash"] = (np.any([zone_any_hand_series(cs[v], T, zone(v, hh["sink_zone"]), hh["thr_hand"])
                         for v in hh["close_cams"]], 0) if all(v in cs for v in hh["close_cams"]) else None)
    need_close = cams["close"]["A"] + cams["close"]["B"] + [cams["overview"]]
    if all(v in cs for v in need_close):
        side = side_series(cs[cams["overview"]], T, **cams["side_x"])
        r["touch"] = touch_series(cs, T, side, cams["close"], cfg["touch_classes"], cfg["association"])
    else:
        r["touch"] = None
    sink = {v: zone(v, hh["sink_zone"]) for v in ("Boutput1", "Boutput2")}
    r["ng4"], r["ng4_why"] = ng4_rule(cs, T, sink, step)
    return r


def touch_onsets(touch_any, start, stop, step):
    """碰觸起點:間隔 < TOUCH_GAP_S 合併、長度 ≥ TOUCH_MIN_S。"""
    gap, mn = int(round(TOUCH_GAP_S / step)), int(round(TOUCH_MIN_S / step))
    merged = []
    for s, e in intervals(touch_any[start:stop]):
        if merged and s - merged[-1][1] < gap:
            merged[-1][1] = e
        else:
            merged.append([s, e])
    return [s + start for s, e in merged if e - s >= mn]


def judge(c, r, cfg, contam_vlm):
    """依 §8.1 對每類 NG 給判定。"""
    step, hh = cfg["step_s"], cfg["hand_hygiene"]
    fsm = HandHygieneFSM(hh["wash_s"], step, cfg["touch_classes"])
    grace = int(round(hh["grace_s"] / step))
    T, wash, touch = r["T"], r["wash"], r["touch"]
    out = {}
    for tname in ("mv", "base", "vlm"):
        if tname == "vlm":
            trig = [(0, 0)] if contam_vlm else ([] if contam_vlm is not None else None)
        else:
            trig = r[f"trig_{tname}"]
        if trig is None or wash is None:
            out[tname] = dict(contam=None, ng1=None, ng2=None, n_touch="")
            continue
        if not trig:
            out[tname] = dict(contam=False, ng1=False, ng2=False, n_touch=0)
            continue
        e = trig[0][1]                                  # 第一次污染接觸(5 秒內合併 → 片段內視為一次)
        dirty = e + grace if tname != "vlm" else 0
        washed = wash[dirty:].sum() >= fsm.need
        ng2, n_touch = None, ""
        if touch is not None:
            v = fsm.run(dirty, wash, touch)
            ng2 = v.result == "違規"
            stop = v.at if v.result == "合格" else T
            n_touch = len(touch_onsets(touch.any(1), dirty, stop, step))
        out[tname] = dict(contam=True, ng1=not washed, ng2=ng2, n_touch=n_touch)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-vlm", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load((INTEG / "configs/violation_rules.yaml").read_text(encoding="utf-8"))
    clips = load_clips()
    print(f"片段 {len(clips)}:" + "、".join(f"{k} {v}" for k, v in sorted(
        defaultdict(int, {n: sum(c['ng'] == n for c in clips.values()) for n in ('NG1', 'NG2', 'NG3', 'NG4')}).items())),
          flush=True)
    if not args.no_vlm:
        run_vlm(clips)

    rows = []
    for c in clips.values():
        r = rule_part(c, cfg)
        eat, eat_sc = vlm_flag(c, "eat")
        con, con_sc = vlm_flag(c, "contam")
        n4v, n4_sc = vlm_flag(c, "ng4")
        j = judge(c, r, cfg, con)
        row = dict(key=c["key"], ng=c["ng"], cat=c["cat"], note=c["note"], dur_s=c["dur"],
                   views=len(c["files"]), steps=r["T"],
                   trig_mv=r["trig_mv"] and [(round(s * 0.1, 1), round(e * 0.1, 1)) for s, e in r["trig_mv"]],
                   trig_base=r["trig_base"] and [(round(s * 0.1, 1), round(e * 0.1, 1)) for s, e in r["trig_base"]],
                   wash_s=None if r["wash"] is None else round(r["wash"].sum() * 0.1, 1),
                   touch_s=None if r["touch"] is None else round(r["touch"].any(1).sum() * 0.1, 1),
                   eat=eat, eat_sc=eat_sc, contam_vlm=con, contam_sc=con_sc,
                   ng4_rule=r["ng4"], ng4_why=r["ng4_why"], ng4_vlm=n4v, ng4_sc=n4_sc)
        for t, d in j.items():
            for k, x in d.items():
                row[f"{t}_{k}"] = x
        rows.append(row)
        print(f"  {c['key']} {c['ng']} {c['cat']}:mv {j['mv']} | eat {eat} | ng4 {r['ng4']}", flush=True)

    # 每類 Recall(正例 = 該類片段)與跨類誤觸發(其他片段)
    detectors = {
        "NG1_垃圾桶_mv": ("NG1", "垃圾桶", lambda x: x["mv_ng1"]),
        "NG1_垃圾桶_base": ("NG1", "垃圾桶", lambda x: x["base_ng1"]),
        "NG1_垃圾桶_污染偵測mv": ("NG1", "垃圾桶", lambda x: x["mv_contam"]),
        "NG1_沒洗手_vlm": ("NG1", "沒洗手", lambda x: x["vlm_ng1"]),
        "NG2_垃圾桶_mv": ("NG2", "垃圾桶", lambda x: x["mv_ng2"]),
        "NG2_垃圾桶_base": ("NG2", "垃圾桶", lambda x: x["base_ng2"]),
        "NG2_沒洗手_vlm": ("NG2", "沒洗手", lambda x: x["vlm_ng2"]),
        "NG3_vlm": ("NG3", None, lambda x: x["eat"]),
        "NG4_rule": ("NG4", None, lambda x: x["ng4_rule"]),
        "NG4_vlm": ("NG4", None, lambda x: x["ng4_vlm"]),
    }
    summ, cross = [], []
    for name, (ng, cat, f) in detectors.items():
        pos = [x for x in rows if x["ng"] == ng and (cat is None or x["cat"] == cat)]
        val = [f(x) for x in pos if f(x) is not None]
        k, n = sum(map(bool, val)), len(val)
        lo, hi = wilson(k, n)
        neg = [x for x in rows if x["ng"] != ng]
        nv = [f(x) for x in neg if f(x) is not None]
        summ.append(dict(detector=name, ng=ng, cat=cat or "全部", n_clips=len(pos), n_judgeable=n, hit=k,
                         recall=round(k / n, 3) if n else "", ci_lo=lo, ci_hi=hi,
                         cross_fire=sum(map(bool, nv)), cross_n=len(nv),
                         cross_rate=round(sum(map(bool, nv)) / len(nv), 3) if nv else ""))
        for x in neg:
            if f(x):
                cross.append(dict(detector=name, key=x["key"], ng=x["ng"], cat=x["cat"]))
        print(f"  {name:22s} R {k}/{n}  跨類誤觸發 {sum(map(bool, nv))}/{len(nv)}", flush=True)

    for nm, data in (("clips_per_clip", rows), ("clips_summary", summ), ("clips_cross", cross)):
        if not data:
            continue
        with open(OUT / f"{nm}.csv", "w", newline="", encoding="utf-8-sig") as fh:
            w = csv.DictWriter(fh, fieldnames=list(dict.fromkeys(k for r in data for k in r)))
            w.writeheader()
            w.writerows(data)
    print(f"輸出 {OUT},{time.time() - t0:.0f} 秒")


if __name__ == "__main__":
    main()
