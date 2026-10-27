"""M6 多鏡頭:整場誤報測試(YH2040 / 2023_09_12_09_50_41,780 秒)。

預先登記:docs/M6_多鏡頭觸發_預先登記_20261027.md §8.1
需要先有:results/m6/multiview/frozen.json(垃圾桶,開發集)、results/m6/violation/r3_summary.json(洗手 L1 最佳)。
不據此再調設定。

用法:
    python scripts/eval_multiview_fullsession.py
"""
import csv
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
INTEG = ROOT / "integrated"
sys.path.insert(0, str(INTEG / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.append(str(ROOT / "src"))                      # 只為 m6_assoc.gt;其餘一律用 integrated/

from common.zones import get_zone                                         # noqa: E402
from m4_track.det_cache import DetCache                                   # noqa: E402
from m6_events.smoothing import hysteresis                                # noqa: E402
from m6_events.violation import (HAND, at, dilate, intervals, steps, zone_any_hand_series,  # noqa: E402
                                 zone_body_series, zone_hand_series)
from m6_assoc.gt import load_actions                                      # noqa: E402
import eval_multiview_triggers as MV                                      # noqa: E402

CACHE = ROOT / "results/det_cache/epfl_allcls_armA_s3"
VIDEO = ROOT / "data/epfl"
XLSX = ROOT / "data/epfl_ann/YH2040/2023_09_12_09_50_41/annotations/actions_annotations.xlsx"
OUT = ROOT / "results/m6/multiview"
FP_DIR = OUT / "fullsession_fp"
STEP, SYNC, TOL = 0.1, 5, 2.0                 # 0.1 秒一步;±0.5 秒同步;真值 ±2 秒
B4 = ["Boutput0", "Boutput1", "Boutput2", "Boutput3"]
CLOSE2 = ["Boutput1", "Boutput2"]
MERGE_GAP = 10                                 # 洗手片段:間隔 < 1 秒合併
THR_SINK = 0.2                                 # 同 violation_rules.yaml hand_hygiene.thr_hand
BODY_X = 0.2


def zone(v, name):
    return get_zone(INTEG / f"configs/zones/epfl_{v}.json", name)


def gt_windows(acts):
    trash = [(a["start"], a["end"]) for a in acts if a["verb"] == "Put" and a["noun"] == "Trash"]
    wash = [(a["start"], a["end"]) for a in acts if a["verb"] == "Wash" and a["noun"] == "Hand"]
    water, opened = [], None                                              # Open Water … Close Water
    for a in acts:
        if a["noun"] == "Water" and a["verb"] == "Open":
            opened = a["start"]
        elif a["noun"] == "Water" and a["verb"] == "Close" and opened is not None:
            water.append((opened, a["end"]))
            opened = None
    return trash, wash, water


def overlaps(seg, wins, tol=TOL):
    s, e = seg
    return [i for i, (a, b) in enumerate(wins) if s <= b + tol and e >= a - tol]


def episodes(mask, need):
    """洗手序列 → 片段:間隔 < MERGE_GAP 合併,片段內成立步數累計 ≥ need。"""
    iv = intervals(mask)
    merged = []
    for s, e in iv:
        if merged and s - merged[-1][1] < MERGE_GAP:
            merged[-1][1] = e
        else:
            merged.append([s, e])
    return [(s, e) for s, e in merged if mask[s:e].sum() >= need]


def grab(v, t):
    cap = cv2.VideoCapture(str(VIDEO / f"{v}.mp4"))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * 30))
    ok, f = cap.read()
    cap.release()
    return f if ok else np.zeros((720, 1280, 3), np.uint8)


def snapshot(path, t, views, zname, caches, title):
    cells = []
    for v in views:
        f = grab(v, t)
        z = zone(v, zname)
        if z:
            cv2.polylines(f, [np.int32(z)], True, (0, 255, 255), 3)
        c = caches[v]
        xy, cf, cl = at(c, steps(c), int(round(t / STEP)))
        for b in xy[(cl == HAND) & (cf >= THR_SINK)]:
            cv2.rectangle(f, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 0, 255), 3)
        cv2.putText(f, f"{v}  t={t:.1f}s", (10, 40), 0, 1.2, (0, 255, 255), 3)
        cells.append(cv2.resize(f, (640, 360)))
    while len(cells) % 2:
        cells.append(np.zeros_like(cells[0]))
    g = np.vstack([np.hstack(cells[i:i + 2]) for i in range(0, len(cells), 2)])
    cv2.putText(g, title, (10, g.shape[0] - 15), 0, 1.0, (255, 255, 255), 2)
    cv2.imencode(".jpg", g)[1].tofile(str(path))


def write_csv(path, rows):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def main():
    t0 = time.time()
    FP_DIR.mkdir(parents=True, exist_ok=True)
    trash_gt, wash_gt, water = gt_windows(load_actions(XLSX))
    hours = 780 / 3600
    caches = {v: DetCache(CACHE / f"{v}.npz") for v in dict.fromkeys(MV.TRASH_VIEWS + B4)}
    T = len(steps(caches["output0"])[0])
    print(f"整場 {T * STEP:.0f} 秒;丟垃圾 {len(trash_gt)}、洗手 {len(wash_gt)}、開水時段 {len(water)}", flush=True)

    # ── 垃圾桶 ──
    fz = json.loads((OUT / "frozen.json").read_text(encoding="utf-8"))["trash"]
    mvc = fz["multiview"]
    tz = {v: zone(v, "垃圾桶區") for v in MV.TRASH_VIEWS}
    d = dict(T=T, hand={}, body=zone_body_series(caches["output0"], T, tz["output0"], BODY_X))
    for v in MV.TRASH_VIEWS:
        d["hand"][(v, mvc["thr_hand"])] = zone_any_hand_series(caches[v], T, tz[v], mvc["thr_hand"])
    b = fz["base"]
    d["base"] = (dilate(zone_hand_series(caches["Aoutput1"], T, tz["Aoutput1"], b["thr_hand"]), SYNC)
                 & dilate(zone_body_series(caches["output0"], T, tz["output0"], b["x"]), SYNC))
    variants = [("base", dict(arm="base", T=b["T"]))]
    variants += [(f"{arm}_k{k}", dict(arm=arm, k=k, thr_hand=mvc["thr_hand"], T=mvc["T"]))
                 for arm in ("kofn", "kofn_body", "kofn_strict", "kofn_strict_body") for k in (1, 2, 3)]
    rows, fps = [], []
    for name, cfg in variants:
        sm = hysteresis(MV.raw_of(d, cfg), *MV.EV_SMOOTH)
        need = max(1, int(round(cfg["T"] / STEP)))
        fires = [(s * STEP, e * STEP) for s, e in intervals(sm) if e - s >= need]
        hit = {i for f in fires for i in overlaps(f, trash_gt)}
        fp = [f for f in fires if not overlaps(f, trash_gt)]
        frozen_flag = int(cfg.get("arm") == mvc["arm"] and cfg.get("k") == mvc["k"])
        rows.append(dict(kind="trash", name=name, frozen=frozen_flag, hit=len(hit), n_gt=len(trash_gt),
                         fires=len(fires), fp=len(fp), fp_per_h=round(len(fp) / hours, 1)))
        print(f"  垃圾桶 {name:12s}{' (凍結)' if frozen_flag else '':6s} 抓到 {len(hit)}/{len(trash_gt)}  "
              f"觸發 {len(fires)}  誤報 {len(fp)}(每小時 {len(fp) / hours:.1f})", flush=True)
        for s, e in fp:
            fps.append(dict(kind="trash", name=name, start=round(s, 1), end=round(e, 1)))
    # 誤報截圖(去重:同一時段只截一次)
    seen = set()
    for r in fps:
        key = round((r["start"] + r["end"]) / 2)
        if key in seen:
            continue
        seen.add(key)
        r["img"] = f"trash_{key:04d}.jpg"
        snapshot(FP_DIR / r["img"], (r["start"] + r["end"]) / 2, MV.TRASH_VIEWS, "垃圾桶區", caches,
                 f"trash FP {r['start']}-{r['end']}s")

    # ── 洗手 ──
    r3 = json.loads((ROOT / "results/m6/violation/r3_summary.json").read_text(encoding="utf-8"))["l1_best"]
    need = max(1, int(round(r3["wash_s"] / STEP)))
    sink = {v: zone(v, "洗手台") for v in B4 + ["output0"]}
    raw = {v: zone_any_hand_series(caches[v], T, sink[v], THR_SINK) for v in B4}
    body_k = dilate(zone_body_series(caches["output0"], T, sink["output0"], BODY_X), SYNC)
    dl = {v: dilate(x, SYNC) for v, x in raw.items()}
    wash = dict(close_hand=np.any([raw[v] for v in CLOSE2], 0))
    any_raw = np.any(list(raw.values()), 0)
    for k in (1, 2):
        wash[f"close4_{k}"] = np.sum(list(dl.values()), 0) >= k
        wash[f"close4s_{k}"] = wash[f"close4_{k}"] & any_raw                 # §8.3 strict
    for m in list(wash):
        wash[f"{m}_body"] = wash[m] & body_k
    wseen = set()
    for name, mask in wash.items():
        eps = [(s * STEP, e * STEP) for s, e in episodes(mask, need)]
        hit = {i for ep in eps for i in overlaps(ep, wash_gt)}
        fp = [ep for ep in eps if not overlaps(ep, wash_gt)]
        fp_water = [ep for ep in fp if overlaps(ep, water, 0)]
        rows.append(dict(kind="wash", name=name, frozen=int(name == r3["wash_mode"]), hit=len(hit),
                         n_gt=len(wash_gt), fires=len(eps), fp=len(fp), fp_water=len(fp_water),
                         fp_per_h=round(len(fp) / hours, 1)))
        print(f"  洗手 {name:18s} 抓到 {len(hit)}/{len(wash_gt)}  片段 {len(eps)}  假洗手 {len(fp)}"
              f"(水槽其他用途 {len(fp_water)};每小時 {len(fp) / hours:.1f})", flush=True)
        for s, e in fp:
            r = dict(kind="wash", name=name, start=round(s, 1), end=round(e, 1),
                     in_water=int(bool(overlaps((s, e), water, 0))))
            key = round((s + e) / 2)
            if key not in wseen:
                wseen.add(key)
                r["img"] = f"wash_{key:04d}.jpg"
                snapshot(FP_DIR / r["img"], (s + e) / 2, B4, "洗手台", caches, f"wash FP {s:.1f}-{e:.1f}s")
            fps.append(r)

    write_csv(OUT / "fullsession_summary.csv", rows)
    write_csv(OUT / "fullsession_fp.csv", fps)
    print(f"輸出 {OUT},{time.time() - t0:.0f} 秒")


if __name__ == "__main__":
    main()
