"""事件測試:抹布、垃圾桶、吃東西(片段層級,其他類別片段當反例)。

預先登記:docs/M7_事件測試_預先登記_20261020.md
先跑 M3 偵測快取(results/det_cache/event_clips_s3/<片段檔名>.npz);SAM 3 比較組另跑 scripts/m7_event_sam3.py。

組別:
  cloth      :M6 V 規則 + 角色分工融合(HY3 中抹布走 V),參數凍結
  cloth_sam3 :cloth + SAM 3 "sponge" / "towel" 框(每 1 秒一幀)併入抹布框
  cloth_sam3f:cloth_sam3 加過濾(§9.0 補登):與 M3 容器框(conf ≥ 0.4)IoU ≥ 0.5 的 SAM 3 框丟掉,
               剩下的每鏡頭每幀只留信心最高的 1 個(一手只會拿一個海綿,另一手是正在洗的碗盤)
  trash      :指派給人的手框中心落在固定 zone「垃圾桶區」(output0 / Aoutput1)
  trash_sam3 :zone 換成每支片段第一幀 SAM 3 "trash can" 框(外擴 10 像素;找不到退回 zone)
  eat        :指派給人的手框中心落在該人框頭部區域
事件:互動成立的步做遲滯平滑 (3, 2, k_off 5),連續 ≥ T 秒 → 片段被判有該類。

用法:
    python scripts/eval_events.py --split dev
    python scripts/eval_events.py --split test --frozen results/m7/events/dev/selected.json
"""
import argparse
import csv
import itertools
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import eval_m6_round1 as R1                                               # noqa: E402
from common.zones import get_zone                                         # noqa: E402
from m3.classes import NAME_TO_CLASS                                      # noqa: E402
from m4_track.det_cache import DetCache                                   # noqa: E402
from m6_assoc.association import _above, frame_features_r1, ioa          # noqa: E402
from m6_assoc.gt import SCORED                                            # noqa: E402
from m6_assoc.roles import auto_side, role_fuse                           # noqa: E402
from m6_assoc.smoothing import hysteresis                                 # noqa: E402

CLIPS = Path("D:/yizhen/event_clips")
CACHE = ROOT / "results/det_cache/event_clips_s3"
SAM3_DIR = ROOT / "results/m7/events/sam3"
CLASSES = ["抹布", "垃圾桶", "吃東西"]
EXCLUDE_SESSIONS = {"test/YH2003/2023_05_17_09_08_58"}        # 疑為 M3 訓練場次 s2(登記 §2)
CLOSE = {"A": ["Aoutput1", "Aoutput2", "Aoutput3"], "B": ["Boutput1", "Boutput2"]}
VIEWS = ["output0", "Aoutput1", "Aoutput2", "Aoutput3", "Boutput1", "Boutput2"]
AUTO_X = dict(x_a=480.9, x_b=729.3)
CLOTH_CFG = dict(thr_obj=0.4, thr_hand=0.3, d=0.1, iou_move=0.9, win=(5, 3), k_off=10, role_k=2)
EV_SMOOTH = (3, 2, 5)
GRID = {
    "cloth": dict(T=[0, 0.5, 1, 2]),
    "cloth_sam3": dict(T=[0, 0.5, 1, 2]),
    "cloth_sam3f": dict(T=[0, 0.5, 1, 2]),
    "trash": dict(thr_hand=[0.2, 0.3], cams=["output0", "Aoutput1", "any"], T=[0, 0.3, 0.5, 1]),
    "trash_sam3": dict(thr_hand=[0.2, 0.3], cams=["output0", "Aoutput1", "any"], T=[0, 0.3, 0.5, 1]),
    "eat": dict(thr_hand=[0.2, 0.3], head_frac=[0.15, 0.25], T=[0.3, 0.5, 1, 2]),
}
ARM_CLASS = {"cloth": "抹布", "cloth_sam3": "抹布", "cloth_sam3f": "抹布", "trash": "垃圾桶", "trash_sam3": "垃圾桶", "eat": "吃東西"}
PERSON, HAND, CLOTH = NAME_TO_CLASS["人"], NAME_TO_CLASS["手"], NAME_TO_CLASS["抹布"]
CONTAINER = NAME_TO_CLASS["容器"]


def iou(a, b):
    """a [N,4]、b [M,4] → [N,M]。"""
    x1, y1 = np.maximum(a[:, None, 0], b[None, :, 0]), np.maximum(a[:, None, 1], b[None, :, 1])
    x2, y2 = np.minimum(a[:, None, 2], b[None, :, 2]), np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = lambda z: (z[:, 2] - z[:, 0]) * (z[:, 3] - z[:, 1])
    return inter / np.maximum(area(a)[:, None] + area(b)[None, :] - inter, 1e-9)


def sam3_cloth_boxes(rows, xy, cf, cl, filt):
    """SAM 3 抹布框(列 = [x1,y1,x2,y2,信心,提示])→ 要併入的框 [K,4]。"""
    if not rows:
        return np.zeros((0, 4))
    r = np.asarray(rows, float)
    if filt:
        cont = xy[(cl == CONTAINER) & (cf >= 0.4)].astype(float)
        if len(cont):
            r = r[iou(r[:, :4], cont).max(1) < 0.5]
        r = r[np.argsort(-r[:, 4])[:1]]
    return r[:, :4]
P_FLOOR = 0.60


def load_events(split):
    rows = list(csv.DictReader(open(CLIPS / "clips_index.csv", encoding="utf-8-sig")))
    ev = {}
    for r in rows:
        if r["session_id"] in EXCLUDE_SESSIONS or r["view"] not in VIEWS:
            continue
        if (r["session_id"].startswith("train/")) != (split == "dev"):
            continue
        e = ev.setdefault(r["annotation_id"], dict(id=r["annotation_id"], cat=r["event_category"],
                                                    session=r["session_id"], note=r["note"],
                                                    a0=float(r["annot_start_sec"]) - float(r["clip_start_sec"]),
                                                    a1=float(r["annot_end_sec"]) - float(r["clip_start_sec"]),
                                                    clips={}))
        e["clips"][r["view"]] = Path(r["clip_file"]).stem
    return list(ev.values())


def steps(c):
    """片段的時間步(0.1 秒)與每步偵測切片。"""
    stride = int(c.meta["cache_stride"])
    fids = np.arange(0, int(c.meta["max_fid"]) + 1, stride)
    return fids, np.searchsorted(c._fid, fids, "left"), np.searchsorted(c._fid, fids, "right")


def owned_hands(xy, cf, cl, thr_hand):
    """回傳 (手框, 該手所屬人框)。"""
    persons = xy[(cl == PERSON) & _above(cf, 0.3)]
    hands = xy[(cl == HAND) & _above(cf, thr_hand)]
    if not len(hands) or not len(persons):
        return np.zeros((0, 4)), np.zeros((0, 4))
    m = ioa(hands, persons)
    best = m.argmax(1)
    ok = m[np.arange(len(hands)), best] >= 0.5
    return hands[ok], persons[best[ok]]


def in_poly(pts, poly):
    p = np.asarray(poly, np.float32).reshape(-1, 1, 2)
    return np.array([cv2.pointPolygonTest(p, (float(x), float(y)), False) >= 0 for x, y in pts], bool)


# ── 互動(逐步布林)────────────────────────────────────────────────

def cloth_raw(e, T_len, sam3=None, filt=False):
    """抹布:各近拍鏡頭 V 規則 → 角色分工融合 → M6 平滑。sam3:{stem: {step: [[x1,y1,x2,y2],...]}}"""
    cfg = CLOTH_CFG
    ci = SCORED.index("抹布")
    cls_ids = [NAME_TO_CLASS[c] for c in SCORED]
    raws = {}
    for v in CLOSE["A"] + CLOSE["B"]:
        c = DetCache(CACHE / f"{e['clips'][v]}.npz")
        _, lo, hi = steps(c)
        sb = (sam3 or {}).get(e["clips"][v], {})
        rows = []
        for i in range(T_len):
            def cut(j):
                xy, cf, cl = c._xyxy[lo[j]:hi[j]], c._conf[lo[j]:hi[j]], c._cls[lo[j]:hi[j]]
                extra = sam3_cloth_boxes(sb.get(str(j - j % 10)), xy, cf, cl, filt)  # SAM 3 每 1 秒一幀,之間沿用
                if len(extra):
                    xy = np.vstack([xy, np.asarray(extra, xy.dtype)])
                    cf = np.concatenate([cf, np.full(len(extra), 0.99, cf.dtype)])
                    cl = np.concatenate([cl, np.full(len(extra), CLOTH, cl.dtype)])
                return xy, cf, cl
            j = min(i, len(lo) - 1)
            xy, cf, cl = cut(j)
            pxy, pcf, pcl = cut(max(0, j - R1.LAG)) if j >= R1.LAG else (xy[:0], cf[:0], cl[:0])
            r = frame_features_r1(xy, cf, cl, pxy, pcf, pcl, cls_ids, thr_obj=cfg["thr_obj"],
                                  thr_hand=cfg["thr_hand"], iou_moves=R1.IOU_MOVES, **R1.FIXED)
            rows.append(r)
        f = {k: np.stack([r[k] for r in rows]) for k in rows[0]}
        raws[v] = R1.raw("V", f, cfg)
    # 定位:output0 信心最高人框的腳點
    c = DetCache(CACHE / f"{e['clips']['output0']}.npz")
    _, lo, hi = steps(c)
    fx = np.full(T_len, np.nan)
    for i in range(T_len):
        j = min(i, len(lo) - 1)
        cl, cf, xy = c._cls[lo[j]:hi[j]], c._conf[lo[j]:hi[j]], c._xyxy[lo[j]:hi[j]]
        k = (cl == PERSON) & _above(cf, 0.3)
        if k.any():
            b = xy[np.flatnonzero(k)[cf[k].argmax()]]
            fx[i] = (b[0] + b[2]) / 2
    side = auto_side(fx, **AUTO_X)
    fused = role_fuse(raws, side, CLOSE, cfg["role_k"])[:, ci]
    return hysteresis(fused, cfg["win"][0], cfg["win"][1], cfg["k_off"])


def zone_raw(e, T_len, view, thr_hand, poly_of):
    c = DetCache(CACHE / f"{e['clips'][view]}.npz")
    _, lo, hi = steps(c)
    poly = poly_of(e, view)
    out = np.zeros(T_len, bool)
    for i in range(T_len):
        j = min(i, len(lo) - 1)
        hands, _ = owned_hands(c._xyxy[lo[j]:hi[j]].astype(float), c._conf[lo[j]:hi[j]], c._cls[lo[j]:hi[j]], thr_hand)
        if len(hands):
            ctr = np.stack([(hands[:, 0] + hands[:, 2]) / 2, (hands[:, 1] + hands[:, 3]) / 2], 1)
            out[i] = in_poly(ctr, poly).any()
    return out


def eat_raw(e, T_len, thr_hand, head_frac):
    out = np.zeros(T_len, bool)
    for v in VIEWS:
        c = DetCache(CACHE / f"{e['clips'][v]}.npz")
        _, lo, hi = steps(c)
        for i in range(T_len):
            j = min(i, len(lo) - 1)
            hands, owner = owned_hands(c._xyxy[lo[j]:hi[j]].astype(float), c._conf[lo[j]:hi[j]],
                                       c._cls[lo[j]:hi[j]], thr_hand)
            if not len(hands):
                continue
            cx, cy = (hands[:, 0] + hands[:, 2]) / 2, (hands[:, 1] + hands[:, 3]) / 2
            top = owner[:, 1] + head_frac * (owner[:, 3] - owner[:, 1])
            out[i] |= bool(((cx >= owner[:, 0]) & (cx <= owner[:, 2]) & (cy <= top)).any())
    return out


def fires(raw, T, step=0.1):
    """平滑後連續成立 ≥ T 秒?回傳 (是否, 第一次發出的秒數)。"""
    sm = hysteresis(raw, *EV_SMOOTH)
    run = 0
    need = max(1, int(round(T / step)))
    for i, x in enumerate(sm):
        run = run + 1 if x else 0
        if run >= need:
            return True, round((i - need + 1) * step, 1)
    return False, None


# ── 評估 ─────────────────────────────────────────────────────────

def score(flags, cats, cls):
    pos = np.array([c == cls for c in cats], bool)
    f = np.array(flags, bool)
    tp, fp = int((f & pos).sum()), int((f & ~pos).sum())
    fn, tn = int((~f & pos).sum()), int((~f & ~pos).sum())
    R = tp / max(tp + fn, 1)
    P = tp / max(tp + fp, 1)
    return dict(R=round(R, 4), P=round(P, 4), F1=round(2 * P * R / max(P + R, 1e-9), 4),
                FAR=round(fp / max(fp + tn, 1), 4), tp=tp, fp=fp, fn=fn, tn=tn)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["dev", "test"], required=True)
    ap.add_argument("--arms", nargs="+", default=list(GRID))
    ap.add_argument("--frozen", default=None, help="測試集:用開發集選定的設定,不再選")
    ap.add_argument("--out-dir", default=str(ROOT / "results/m7/events"))
    args = ap.parse_args()
    out = Path(args.out_dir) / args.split
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    events = [e for e in load_events(args.split)
              if all((CACHE / f"{e['clips'][v]}.npz").exists() for v in VIEWS)]
    print(f"{args.split}:{len(events)} 個事件(快取齊全)")
    cats = [e["cat"] for e in events]
    zones = {v: get_zone(ROOT / f"configs/zones/epfl_{v}.json", "垃圾桶區") for v in ("output0", "Aoutput1")}
    sam3 = {}
    if any(a.endswith("_sam3") for a in args.arms):
        for e in events:
            p = SAM3_DIR / f"{e['id']}.json"
            sam3[e["id"]] = json.loads(p.read_text(encoding="utf-8")) if p.exists() else None
        miss = sum(v is None for v in sam3.values())
        if miss:
            raise SystemExit(f"SAM 3 結果缺 {miss} 個事件,先跑 scripts/m7_event_sam3.py")

    def poly_fixed(e, v):
        return zones[v]

    def poly_sam3(e, v):
        b = (sam3[e["id"]] or {}).get("trash", {}).get(v)
        if not b:
            return zones[v]
        x1, y1, x2, y2 = b[0] - 10, b[1] - 10, b[2] + 10, b[3] + 10
        return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]

    T_lens = {e["id"]: len(steps(DetCache(CACHE / f"{e['clips']['output0']}.npz"))[0]) for e in events}
    frozen = json.loads(Path(args.frozen).read_text(encoding="utf-8")) if args.frozen else None
    selected, per_clip, grid_rows = {}, defaultdict(dict), []
    for arm in args.arms:
        cls = ARM_CLASS[arm]
        keys = list(GRID[arm])
        cfgs = [dict(zip(keys, v)) for v in itertools.product(*GRID[arm].values())]
        if frozen:
            cfgs = [c for key in ("best_f1", "recall_first") if frozen[arm].get(key)
                    for c in [frozen[arm][key]["cfg"]]]
        raw_cache = {}
        rows = []
        for cfg in cfgs:
            flags, first = [], []
            for e in events:
                T_len = T_lens[e["id"]]
                rk = (e["id"],) + tuple((k, cfg[k]) for k in keys if k != "T")
                if rk not in raw_cache:
                    if arm.startswith("cloth"):
                        sb = (sam3.get(e["id"]) or {}).get("cloth") if arm != "cloth" else None
                        raw_cache[rk] = cloth_raw(e, T_len, sb, filt=arm == "cloth_sam3f")
                    elif arm.startswith("trash"):
                        pf = poly_sam3 if arm == "trash_sam3" else poly_fixed
                        vs = ["output0", "Aoutput1"] if cfg["cams"] == "any" else [cfg["cams"]]
                        raw_cache[rk] = np.any([zone_raw(e, T_len, v, cfg["thr_hand"], pf) for v in vs], 0)
                    else:
                        raw_cache[rk] = eat_raw(e, T_len, cfg["thr_hand"], cfg["head_frac"])
                f, t1 = fires(raw_cache[rk], cfg["T"])
                flags.append(f)
                first.append(None if t1 is None else round(t1 - max(e["a0"], 0), 1))
            s = score(flags, cats, cls)
            rows.append(dict(cfg=cfg, score=s, flags=flags, first=first))
            grid_rows.append([arm, json.dumps(cfg, ensure_ascii=False)] + [s[k] for k in ("R", "P", "F1", "FAR", "tp", "fp", "fn", "tn")])
        best = max(rows, key=lambda r: (r["score"]["F1"], r["score"]["R"]))
        ok = [r for r in rows if r["score"]["P"] >= P_FLOOR]
        rf = max(ok, key=lambda r: (r["score"]["R"], r["score"]["F1"])) if ok else None
        if frozen:
            best = next(r for r in rows if r["cfg"] == frozen[arm]["best_f1"]["cfg"])
            rf = (next(r for r in rows if r["cfg"] == frozen[arm]["recall_first"]["cfg"])
                  if frozen[arm].get("recall_first") else None)
        selected[arm] = {k: (dict(cfg=v["cfg"], score=v["score"]) if v else None)
                         for k, v in (("best_f1", best), ("recall_first", rf))}
        for e, f, t1 in zip(events, best["flags"], best["first"]):
            per_clip[e["id"]].update({"cat": e["cat"], "session": e["session"], "note": e["note"],
                                      f"{arm}": int(f), f"{arm}_first_s": t1})
        s = best["score"]
        print(f"[{arm}] {cls}:R {s['R']:.3f} P {s['P']:.3f} F1 {s['F1']:.3f} FAR {s['FAR']:.3f} "
              f"(tp {s['tp']} fp {s['fp']} fn {s['fn']})  {json.dumps(best['cfg'], ensure_ascii=False)}  "
              f"{time.time() - t0:.0f} 秒", flush=True)

    (out / "selected.json").write_text(json.dumps(selected, ensure_ascii=False, indent=2), encoding="utf-8")
    with open(out / "grid.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["arm", "cfg", "R", "P", "F1", "FAR", "tp", "fp", "fn", "tn"])
        w.writerows(grid_rows)
    keys = sorted({k for v in per_clip.values() for k in v})
    with open(out / "per_clip.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["id"] + keys)
        w.writeheader()
        for k, v in per_clip.items():
            w.writerow(dict(id=k, **v))
    # 混淆表:真實類別 × 被判類別(主組)
    main_arms = [a for a in ("cloth", "trash", "eat") if a in args.arms]
    allcats = sorted(set(cats))
    with open(out / "confusion.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["真實類別", "事件數"] + [f"判為{ARM_CLASS[a]}" for a in main_arms] + ["都沒判"])
        for c in allcats:
            ids = [k for k, v in per_clip.items() if v["cat"] == c]
            w.writerow([c, len(ids)] + [sum(per_clip[k].get(a, 0) for k in ids) for a in main_arms] +
                       [sum(all(not per_clip[k].get(a, 0) for a in main_arms) for k in ids)])
    print(f"輸出 {out},{time.time() - t0:.0f} 秒")


if __name__ == "__main__":
    main()
