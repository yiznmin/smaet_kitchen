"""M6 多鏡頭觸發:垃圾桶事件(事件片段分類)。

預先登記:docs/M6_多鏡頭觸發_預先登記_20261027.md §3
只用 integrated/ 的模組(m6_events.violation)與 zone 設定。

  --split dev :跑網格,選設定 → results/m6/multiview/frozen.json
  --split test:讀 frozen.json,只跑一次 + 斷鏡頭測試

用法:
    python scripts/eval_multiview_triggers.py --split dev
    python scripts/eval_multiview_triggers.py --split test
"""
import argparse
import csv
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
INTEG = ROOT / "integrated"
sys.path.insert(0, str(INTEG / "src"))

from common.zones import get_zone                                         # noqa: E402
from m4_track.det_cache import DetCache                                   # noqa: E402
from m6_events.smoothing import hysteresis                                # noqa: E402
from m6_events.violation import dilate, steps, zone_any_hand_series, zone_body_series, zone_hand_series  # noqa: E402

CLIPS = Path("D:/yizhen/event_clips")
CACHE = ROOT / "results/det_cache/event_clips_s3"
OUT = ROOT / "results/m6/multiview"
EXCLUDE = {"test/YH2003/2023_05_17_09_08_58"}          # 疑為 M3 訓練場次(同事件測試)
POS = "垃圾桶"
TRASH_VIEWS = ["output0", "Aoutput0", "Aoutput1", "Aoutput2"]
STEP, SYNC = 0.1, 5                                      # 0.1 秒一步;±0.5 秒
EV_SMOOTH = (3, 2, 5)
BASE = dict(thr_hand=0.2, x=0.2, T=1.0)                  # 事件測試第 3 輪 trash_body 凍結值
GRID = dict(thr_hand=[0.2, 0.3], T=[0.5, 1.0, 2.0])
KS = [1, 2, 3]
BODY_X = 0.2


def load_events(split):
    ev = {}
    for r in csv.DictReader(open(CLIPS / "clips_index.csv", encoding="utf-8-sig")):
        if r["session_id"] in EXCLUDE or r["session_id"].startswith("train/") != (split == "dev"):
            continue
        e = ev.setdefault(r["annotation_id"], dict(id=r["annotation_id"], cat=r["event_category"],
                                                    session=r["session_id"], clips={}))
        e["clips"][r["view"]] = Path(r["clip_file"]).stem
    return list(ev.values())


def fires(raw, T):
    """平滑後連續成立 ≥ T 秒?(同事件測試)"""
    sm = hysteresis(raw, *EV_SMOOTH)
    need, run = max(1, int(round(T / STEP))), 0
    for x in sm:
        run = run + 1 if x else 0
        if run >= need:
            return True
    return False


def score(flags, cats):
    pos = np.array([c == POS for c in cats], bool)
    f = np.array(flags, bool)
    tp, fp, fn, tn = int((f & pos).sum()), int((f & ~pos).sum()), int((~f & pos).sum()), int((~f & ~pos).sum())
    R, P = tp / max(tp + fn, 1), tp / max(tp + fp, 1)
    return dict(R=round(R, 3), P=round(P, 3), F1=round(2 * P * R / max(P + R, 1e-9), 3),
                FAR=round(fp / max(fp + tn, 1), 3), tp=tp, fp=fp, fn=fn, tn=tn)


def series(events, zones):
    """每個事件:各鏡頭「手在垃圾桶區」(兩種 thr)、output0 body、base 用的歸屬手。"""
    out = {}
    for e in events:
        cs = {v: DetCache(CACHE / f"{e['clips'][v]}.npz") for v in TRASH_VIEWS}
        T = len(steps(cs["output0"])[0])
        d = dict(T=T, hand={}, body=zone_body_series(cs["output0"], T, zones["output0"], BODY_X))
        for thr in GRID["thr_hand"]:
            for v in TRASH_VIEWS:
                d["hand"][(v, thr)] = zone_any_hand_series(cs[v], T, zones[v], thr)
        d["base"] = (dilate(zone_hand_series(cs["Aoutput1"], T, zones["Aoutput1"], BASE["thr_hand"]), SYNC)
                     & dilate(zone_body_series(cs["output0"], T, zones["output0"], BASE["x"]), SYNC))
        out[e["id"]] = d
    return out


def raw_of(d, cfg, drop=None):
    """依設定組出原始觸發序列;drop = 模擬失效的鏡頭(該台全不成立)。"""
    z = np.zeros(d["T"], bool)
    arm = cfg["arm"]
    if arm == "base":
        if drop in ("Aoutput1", "output0"):
            return z
        return d["base"]
    if arm.startswith("single_"):
        v = arm[len("single_"):]
        return z if drop == v else d["hand"][(v, cfg["thr_hand"])]
    hs = [dilate(d["hand"][(v, cfg["thr_hand"])], SYNC) if v != drop else z for v in TRASH_VIEWS]
    raw = np.sum(hs, 0) >= cfg["k"]
    if arm == "kofn_body":
        raw &= z if drop == "output0" else dilate(d["body"], SYNC)
    return raw


def evaluate(events, S, cfg, drop=None):
    flags = [fires(raw_of(S[e["id"]], cfg, drop), cfg["T"]) for e in events]
    return score(flags, [e["cat"] for e in events]), flags


def configs():
    for thr, T in itertools.product(GRID["thr_hand"], GRID["T"]):
        for v in TRASH_VIEWS:
            yield dict(arm=f"single_{v}", thr_hand=thr, T=T)
        for arm, k in itertools.product(["kofn", "kofn_body"], KS):
            yield dict(arm=arm, k=k, thr_hand=thr, T=T)


def pick(rows, pred):
    c = [r for r in rows if pred(r)]
    return max(c, key=lambda r: (r["F1"], r["R"], -r.get("k", 0)))


def write_csv(path, rows):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["dev", "test"], required=True)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    zp = {v: INTEG / f"configs/zones/epfl_{v}.json" for v in TRASH_VIEWS}
    zones = {v: get_zone(zp[v], "垃圾桶區") for v in TRASH_VIEWS}
    events = load_events(args.split)
    print(f"{args.split}:{len(events)} 個事件,垃圾桶 {sum(e['cat'] == POS for e in events)}", flush=True)
    S = series(events, zones)
    print(f"序列完成 {time.time() - t0:.0f} 秒", flush=True)
    base_cfg = dict(arm="base", **BASE)
    fz_p = OUT / "frozen.json"

    if args.split == "dev":
        rows = []
        for cfg in [base_cfg] + list(configs()):
            s, _ = evaluate(events, S, cfg)
            rows.append(dict(**cfg, **s))
        write_csv(OUT / "dev_trash_grid.csv", rows)
        frozen = dict(multiview=pick(rows, lambda r: r["arm"].startswith("kofn")),
                      **{f"single_{v}": pick(rows, lambda r, v=v: r["arm"] == f"single_{v}") for v in TRASH_VIEWS},
                      base=rows[0])
        fz_p.write_text(json.dumps(dict(trash=frozen), ensure_ascii=False, indent=2), encoding="utf-8")
        for k, r in frozen.items():
            print(f"  {k:16s} {r['arm']:10s} k={r.get('k', '-')} thr={r['thr_hand']} T={r['T']}  "
                  f"R {r['R']:.3f} P {r['P']:.3f} F1 {r['F1']:.3f} FAR {r['FAR']:.3f}", flush=True)
        return

    frozen = json.loads(fz_p.read_text(encoding="utf-8"))["trash"]
    keep = ("arm", "k", "thr_hand", "T", "x")
    rows, per_clip = [], []
    for name, r in frozen.items():
        cfg = {k: r[k] for k in keep if k in r}
        s, flags = evaluate(events, S, cfg)
        rows.append(dict(name=name, drop="", **cfg, **s, dev_F1=r["F1"]))
        per_clip += [dict(name=name, id=e["id"], cat=e["cat"], fired=int(f)) for e, f in zip(events, flags)]
        print(f"  {name:16s} R {s['R']:.3f} P {s['P']:.3f} F1 {s['F1']:.3f} FAR {s['FAR']:.3f} "
              f"({s['tp']}/{s['fp']}/{s['fn']})", flush=True)
        if name in ("multiview", "base"):
            for v in TRASH_VIEWS:
                sd, _ = evaluate(events, S, cfg, drop=v)
                rows.append(dict(name=name, drop=v, **cfg, **sd, dev_F1=r["F1"]))
                print(f"      斷 {v:9s} F1 {sd['F1']:.3f}  R {sd['R']:.3f}  FAR {sd['FAR']:.3f}", flush=True)
    write_csv(OUT / "test_trash.csv", rows)
    write_csv(OUT / "test_trash_per_clip.csv", per_clip)
    print(f"輸出 {OUT},{time.time() - t0:.0f} 秒")


if __name__ == "__main__":
    main()
