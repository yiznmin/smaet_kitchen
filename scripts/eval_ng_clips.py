"""NG 準則評估:NG 片段(正例)+ 無 NG 事件片段(反例)。

預先登記:docs/M6_NG準則評估_預先登記_20261027.md
  §8.1:只用 test 場次的團隊片段,凍結值,報 Recall 與跨類誤觸發 → clips_*.csv
  §8.2(--cv):參與者留一交叉驗證,報 R / P / F1 → cv_*.csv
VLM 分數存 results/m6/ng/vlm/<片段>.json,重跑會沿用。

用法:
    python scripts/eval_ng_clips.py --cv
    python scripts/eval_ng_clips.py --cv --no-vlm   # 只跑偵測規則部分(冒煙)
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
from m6_events.violation import (HAND, PERSON, HandHygieneFSM, at, intervals, side_series, steps,  # noqa: E402
                                 touch_series, trash_intervals, zone_any_hand_series)
import eval_multiview_triggers as MV                                      # noqa: E402

NG_DIR = Path("D:/yizhen/NG_clips")
CUT = ROOT / "data/ng_clips_cut"                              # 自己切的片段(§8.2)
EV_DIR = Path("D:/yizhen/event_clips")                        # 團隊事件片段(反例來源)
EV_CACHE = ROOT / "results/det_cache/event_clips_s3"
EV_NG = ROOT / "results/ng_clips/team_events_vs_ng.csv"
CACHE = ROOT / "results/det_cache/ng_clips_s3"
M3_TRAIN = ("YH2040_2023_09_12_09_50_41",)                    # §8.2:M3 round2 s1 場次,另報排除後的數字
OUT = ROOT / "results/m6/ng"
VLM_OUT = OUT / "vlm"
EXCLUDE = ("YH2003_2023_05_17_09_08_58",)                    # 登記 §1:疑為 M3 訓練場次,只列不統計
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
# §8.2 可調參數(第一個 = §3 預設值)
GRID_VLM = [0.3, 0.5, 0.7]
GRID_NG4 = [(1.0, 1.0), (0.5, 1.0), (2.0, 1.0), (0.5, 2.0), (1.0, 2.0), (2.0, 2.0)]
# §8.6 NG3 規則:手在人框頭部 + 手邊有食材;(thr_hand, head_frac, thr_food, T 秒),第一組 = 預設
GRID_NG3 = [(h, f, fd, t) for h in (0.3, 0.2) for f in (0.25, 0.35) for fd in (0.3, 0.2) for t in (0.3, 0.5)]
NG3_PAD = 0.5                                             # 手框四邊各擴大手框寬高的比例,用來判斷「手邊」
NEG_CATS = dict(NG1=("垃圾桶", "沒洗手"), NG2=("垃圾桶", "沒洗手"), NG3=("吃東西",), NG4=("食材衛生",))
TOUCH_GAP_S, TOUCH_MIN_S = 1.0, 0.3


def load_clips():
    """正例 = NG 片段(團隊影片或自己切的);反例 = 沒有 NG 的團隊事件片段。"""
    clips = {}
    for r in csv.DictReader(open(NG_DIR / "ng_clips_index.csv", encoding="utf-8-sig")):
        pid, sess = r["session_id"].split("/")[1:]
        no = re.search(r"_(ng\d+)_", r["clip_file"]).group(1)
        k = f"{pid}_{sess}_{no}"
        c = clips.setdefault(k, dict(key=k, pid=pid, session=r["session_id"], ng=r["ng_type"], cat=r["category"],
                                     note=r["note"], dur=float(r["clip_duration_sec"]), files={}, caches={},
                                     team=set()))
        cp, cut, team = CACHE / k / f"{r['view']}.npz", CUT / k / f"{r['view']}.mp4", NG_DIR / r["clip_file"]
        if cp.exists():
            c["caches"][r["view"]] = cp
            c["files"][r["view"]] = cut if cut.exists() else team
            if not cut.exists():
                c["team"].add(r["view"])
    ev = list(csv.DictReader(open(EV_NG, encoding="utf-8-sig")))
    # §8.4:與同場次 NG 事件時間重疊的無NG 事件 = 同一件事被標兩次,不當反例
    hit = [(r["session_id"], float(r["annot_start_sec"]), float(r["annot_end_sec"])) for r in ev if r["ng"] != "無NG"]
    no_ng = {r["annotation_id"] for r in ev if r["ng"] == "無NG" and not any(
        s == r["session_id"] and min(b, float(r["annot_end_sec"])) > max(a, float(r["annot_start_sec"]))
        for s, a, b in hit)}
    for r in csv.DictReader(open(EV_DIR / "clips_index.csv", encoding="utf-8-sig")):
        if r["annotation_id"] not in no_ng:
            continue
        pid, sess = r["session_id"].split("/")[1:]
        k = f"ev_{pid}_{sess}_{r['annotation_id'][:8]}"
        c = clips.setdefault(k, dict(key=k, pid=pid, session=r["session_id"], ng="無NG", cat=r["event_category"],
                                     note=r["note"], dur=float(r["clip_duration_sec"]), files={}, caches={},
                                     team=set()))
        cp = EV_CACHE / f"{Path(r['clip_file']).stem}.npz"
        if cp.exists():
            c["caches"][r["view"]] = cp
            c["files"][r["view"]] = EV_DIR / r["clip_file"]
    return {k: c for k, c in clips.items() if c["caches"]}


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


def vlm_scores(c, kind):
    p = VLM_OUT / f"{c['key']}.json"
    if not p.exists():
        return {}
    s = json.loads(p.read_text(encoding="utf-8"))
    return {v: s[f"{kind}_{v}"] for v in VLM_VIEWS if f"{kind}_{v}" in s}


def vlm_flag(c, kind, thr=None):
    sc = vlm_scores(c, kind)
    return (max(sc.values()) >= (THR[kind] if thr is None else thr)) if sc else None, sc


# ── 偵測規則 ─────────────────────────────────────────────────────

def ng4_rule(cs, T, sink, step):
    """清洗(手 + 物件同在洗手台)→ 物件放在洗手台外、不在手中、不在砧板 / 容器上 ≥ NG4_PLACE_S。"""
    objs = [NAME_TO_CLASS[n] for n in ("食材", "刀具", "砧板", "鍋鏟", "鍋子", "容器")]
    sup_ids = [NAME_TO_CLASS[n] for n in ("砧板", "容器")]
    bcams = [v for v in ("Boutput1", "Boutput2") if v in cs and v in sink]
    if not bcams:
        return {wp: None for wp in GRID_NG4}, "缺 B 鏡頭"
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
    washed = [j for j in range(len(objs)) if wash[:, j].any()]
    grid = {wp: False for wp in GRID_NG4}
    if not washed:
        return grid, "沒偵測到清洗"
    why = "有清洗、沒偵測到放桌面"
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
        run = max((e - s for s, e in intervals(placed)), default=0)
        for w, p in GRID_NG4:
            if wash[:, j].sum() >= int(round(w / step)) and run >= int(round(p / step)):
                grid[(w, p)] = True
                why = f"清洗並放桌面:{[k for k, x in NAME_TO_CLASS.items() if x == objs[j]][0]}"
    return grid, why


def ng3_rule(cs, T):
    """NG3:任一鏡頭連續 ≥ T 秒「手中心在人框頭部區 且 食材框與擴大後的手框重疊」。回傳 {grid: bool}。"""
    FOOD = NAME_TO_CLASS["食材"]
    base = {}
    for h, f, fd in {g[:3] for g in GRID_NG3}:
        hit = np.zeros(T, bool)
        for c in cs.values():
            idx = steps(c)
            for i in range(T):
                xy, cf, cl = at(c, idx, i)
                P, H, F = xy[(cl == PERSON) & (cf >= 0.3)], xy[(cl == HAND) & (cf >= h)], xy[(cl == FOOD) & (cf >= fd)]
                if not (len(P) and len(H) and len(F)):
                    continue
                cx, cy = (H[:, 0] + H[:, 2]) / 2, (H[:, 1] + H[:, 3]) / 2
                top = P[:, 1] + f * (P[:, 3] - P[:, 1])
                head = ((cx[:, None] >= P[None, :, 0]) & (cx[:, None] <= P[None, :, 2])
                        & (cy[:, None] >= P[None, :, 1]) & (cy[:, None] <= top[None])).any(1)
                if not head.any():
                    continue
                Hh = H[head]
                w, hh = Hh[:, 2] - Hh[:, 0], Hh[:, 3] - Hh[:, 1]
                Hx = np.c_[Hh[:, 0] - NG3_PAD * w, Hh[:, 1] - NG3_PAD * hh, Hh[:, 2] + NG3_PAD * w, Hh[:, 3] + NG3_PAD * hh]
                hit[i] |= bool((ioa(F, Hx) > 0).any())
        base[(h, f, fd)] = max((e - s for s, e in intervals(hit)), default=0)
    return {g: base[g[:3]] >= max(1, int(round(g[3] / 0.1))) for g in GRID_NG3}


def rule_part(c, cfg):
    """偵測規則部分:污染觸發(垃圾桶)、洗手、碰觸、NG4 規則。"""
    step, cams, hh, te = cfg["step_s"], cfg["cameras"], cfg["hand_hygiene"], cfg["trash_event"]
    cs = {v: DetCache(p) for v, p in c["caches"].items()}
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
    r["ng3"] = ng3_rule(cs, T)
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


def write_csv(name, data):
    if not data:
        return
    with open(OUT / f"{name}.csv", "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(dict.fromkeys(k for r in data for k in r)))
        w.writeheader()
        w.writerows(data)


def clip_row(c, cfg):
    """一個片段的所有判定(含每組可調參數)。"""
    r = rule_part(c, cfg)
    fr = lambda iv: iv and [(round(a * 0.1, 1), round(b * 0.1, 1)) for a, b in iv]
    row = dict(key=c["key"], pid=c["pid"], session=c["session"], split=c["session"].split("/")[0],
               excluded=int(c["key"].startswith(EXCLUDE)), m3_train=int(any(x in c["key"] for x in M3_TRAIN)),
               ng=c["ng"], cat=c["cat"], note=c["note"], dur_s=c["dur"], views=len(c["caches"]),
               team_views=len(c["team"]), steps=r["T"], trig_mv=fr(r["trig_mv"]), trig_base=fr(r["trig_base"]),
               wash_s=None if r["wash"] is None else round(r["wash"].sum() * 0.1, 1),
               touch_s=None if r["touch"] is None else round(r["touch"].any(1).sum() * 0.1, 1),
               ng4_why=r["ng4_why"])
    for t, d in judge(c, r, cfg, None).items():
        if t != "vlm":
            row.update({f"{t}_{k}": x for k, x in d.items()})
    for thr in GRID_VLM:
        con, _ = vlm_flag(c, "contam", thr)
        d = judge(c, r, cfg, con)["vlm"]
        row.update({f"vlm{thr}_{k}": x for k, x in d.items()})
        row[f"eat{thr}"] = vlm_flag(c, "eat", thr)[0]
        row[f"ng4vlm{thr}"] = vlm_flag(c, "ng4", thr)[0]
    for g, x in r["ng3"].items():
        row["ng3rule" + "_".join(map(str, g))] = x
    for (w, pp), x in r["ng4"].items():
        row[f"ng4rule{w}_{pp}"] = x
    for kind in PROMPTS:
        row[f"{kind}_sc"] = vlm_scores(c, kind)
    return row


# 偵測器:(NG, 類別(None = 全部), [(參數標籤, 欄位)]);第一組 = §3 / §8.1 預設值
DETECTORS = {
    "NG1_垃圾桶_mv": ("NG1", "垃圾桶", [("frozen", "mv_ng1")]),
    "NG1_垃圾桶_base": ("NG1", "垃圾桶", [("frozen", "base_ng1")]),
    "NG1_垃圾桶_污染偵測mv": ("NG1", "垃圾桶", [("frozen", "mv_contam")]),
    "NG1_沒洗手_vlm": ("NG1", "沒洗手", [(f"thr{t}", f"vlm{t}_ng1") for t in [0.5, 0.3, 0.7]]),
    "NG2_垃圾桶_mv": ("NG2", "垃圾桶", [("frozen", "mv_ng2")]),
    "NG2_垃圾桶_base": ("NG2", "垃圾桶", [("frozen", "base_ng2")]),
    "NG2_沒洗手_vlm": ("NG2", "沒洗手", [(f"thr{t}", f"vlm{t}_ng2") for t in [0.5, 0.3, 0.7]]),
    "NG3_vlm": ("NG3", None, [(f"thr{t}", f"eat{t}") for t in GRID_VLM]),
    "NG3_rule": ("NG3", None, [("h{}_f{}_fd{}_T{}".format(*g), "ng3rule" + "_".join(map(str, g))) for g in GRID_NG3]),
    "NG4_rule": ("NG4", None, [(f"w{w}_p{p}", f"ng4rule{w}_{p}") for w, p in GRID_NG4]),
    "NG4_vlm": ("NG4", None, [(f"thr{t}", f"ng4vlm{t}") for t in [0.5, 0.3, 0.7]]),
}


def labelled(rows, ng, cat, strict_cross):
    """依 §8.2 給正 / 反例;strict_cross=False 時 NG1、NG2 片段不互當反例(片段本身同時符合兩者)。"""
    out = []
    for x in rows:
        if x["ng"] == ng:
            if cat is None or x["cat"] == cat:
                out.append((x, True))
        elif x["ng"] == "無NG":
            if x["cat"] in NEG_CATS[ng]:
                out.append((x, False))
        elif strict_cross or {x["ng"], ng} != {"NG1", "NG2"}:
            out.append((x, False))
    return out


def prf(pairs, col):
    v = [(bool(x[col]), y) for x, y in pairs if x[col] is not None]
    tp = sum(p and y for p, y in v)
    fp = sum(p and not y for p, y in v)
    fn = sum(y and not p for p, y in v)
    R, P = tp / max(tp + fn, 1), tp / max(tp + fp, 1)
    return dict(tp=tp, fp=fp, fn=fn, tn=len(v) - tp - fp - fn, n=len(v),
                R=round(R, 3), P=round(P, 3), F1=round(2 * P * R / max(P + R, 1e-9), 3))


def cross_validate(rows, ng, cat, grid, strict_cross):
    """參與者留一:每折用其他人選 F1 最高的參數(平手取預設 = 第一組),套到留出的人。"""
    pairs = labelled(rows, ng, cat, strict_cross)
    pids = sorted({x["pid"] for x, _ in pairs})
    pred, folds = [], []
    for pid in pids:
        tr = [(x, y) for x, y in pairs if x["pid"] != pid]
        te = [(x, y) for x, y in pairs if x["pid"] == pid]
        best = grid[0]
        for g in grid[1:]:
            if prf(tr, g[1])["F1"] > prf(tr, best[1])["F1"]:
                best = g
        pred += [(dict(x, _cv=x[best[1]]), y) for x, y in te]
        folds.append(dict(pid=pid, chosen=best[0], n_pos=sum(y for _, y in te), **prf(te, best[1])))
    return pairs, pred, folds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-vlm", action="store_true")
    ap.add_argument("--cv", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load((INTEG / "configs/violation_rules.yaml").read_text(encoding="utf-8"))
    clips = load_clips()
    cnt = defaultdict(int)
    for c in clips.values():
        cnt[c["ng"]] += 1
    print(f"片段 {len(clips)}:{dict(cnt)}", flush=True)
    if not args.no_vlm:
        run_vlm(clips)

    rows = []
    for n, c in enumerate(clips.values(), 1):
        rows.append(clip_row(c, cfg))
        if n % 20 == 0:
            print(f"  規則 {n}/{len(clips)}  {time.time() - t0:.0f} 秒", flush=True)
    write_csv("cv_per_clip" if args.cv else "clips_per_clip", rows)
    stat = [x for x in rows if not x["excluded"]]

    # §8.1:test 場次團隊片段(只有正例),預設值,Recall + 跨類誤觸發
    t81 = [x for x in stat if x["split"] == "test" and x["ng"] != "無NG" and x["team_views"] > 0]
    summ81 = []
    for name, (ng, cat, grid) in DETECTORS.items():
        col = grid[0][1]
        pos = [x[col] for x in t81 if x["ng"] == ng and (cat is None or x["cat"] == cat) and x[col] is not None]
        neg = [x[col] for x in t81 if x["ng"] != ng and x[col] is not None]
        k = sum(map(bool, pos))
        lo, hi = wilson(k, len(pos))
        summ81.append(dict(detector=name, setting=grid[0][0], n=len(pos), hit=k,
                           recall=round(k / len(pos), 3) if pos else "", ci_lo=lo, ci_hi=hi,
                           cross_fire=sum(map(bool, neg)), cross_n=len(neg)))
        print(f"  §8.1 {name:20s} R {k}/{len(pos)}  跨類 {sum(map(bool, neg))}/{len(neg)}", flush=True)
    write_csv("clips_summary", summ81)
    if not args.cv:
        print(f"輸出 {OUT},{time.time() - t0:.0f} 秒")
        return

    # §8.2:參與者留一
    summ, folds_all = [], []
    for subset, data in (("全部", stat), ("排除M3訓練場次", [x for x in stat if not x["m3_train"]])):
        for strict in (True, False):
            for name, (ng, cat, grid) in DETECTORS.items():
                pairs, pred, folds = cross_validate(data, ng, cat, grid, strict)
                cv, fz = prf(pred, "_cv"), prf(pairs, grid[0][1])
                rl, rh = wilson(cv["tp"], cv["tp"] + cv["fn"])
                pl, ph = wilson(cv["tp"], cv["tp"] + cv["fp"])
                f1s = [f["F1"] for f in folds if f["n_pos"]]
                summ.append(dict(subset=subset, neg_def="登記版" if strict else "NG1NG2不互為反例",
                                 detector=name, **{f"cv_{k}": v for k, v in cv.items()},
                                 cv_R_ci=f"{rl}-{rh}", cv_P_ci=f"{pl}-{ph}",
                                 fold_F1_median=round(float(np.median(f1s)), 3) if f1s else "",
                                 fold_F1_min=min(f1s) if f1s else "", fold_F1_max=max(f1s) if f1s else "",
                                 chosen="、".join(sorted({f["chosen"] for f in folds})),
                                 **{f"frozen_{k}": fz[k] for k in ("R", "P", "F1", "tp", "fp", "fn")}))
                folds_all += [dict(subset=subset, strict=int(strict), detector=name, **f) for f in folds]
                if subset == "全部":
                    tag = "登記" if strict else "寬鬆"
                    print(f"  {tag} {name:20s} CV R {cv['R']:.2f} P {cv['P']:.2f} F1 {cv['F1']:.2f} "
                          f"({cv['tp']}/{cv['fp']}/{cv['fn']}) | 預設 F1 {fz['F1']:.2f}", flush=True)
    write_csv("cv_summary", summ)
    write_csv("cv_folds", folds_all)
    print(f"輸出 {OUT},{time.time() - t0:.0f} 秒")


if __name__ == "__main__":
    main()
