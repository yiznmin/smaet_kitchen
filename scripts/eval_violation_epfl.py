"""M6 違規判定(單人)第 1 輪:EPFL「沒洗手」24 個事件片段。

預先登記:docs/M6_違規判定_單人_預先登記_20261021.md
使用 integrated/ 的模組(integrated/src/m6_events/violation.py)與設定(integrated/configs/violation_rules.yaml)。
真值由標註備註判讀:備註含「有洗手」或「"有"洗手」= 合格,其餘 = 違規。

L1:觸發時間 = 標註起點,髒手從起點 + D 秒開始;網格 wash_mode × wash_s × D
L2:只限垃圾類觸發(8 個),觸發由垃圾桶事件自動偵測;用 L1 的 wash_s 2、body、設定檔其餘值
第 2 輪(§8.2 補登,--round 2):洗手方式加 close_hand、close_hand_body(Boutput1/2 上方鏡頭看水槽),L2 用 close_hand_body
第 3 輪(--round 3,docs/M6_多鏡頭觸發_預先登記_20261027.md §4):洗手加 Boutput0–3 四台 k-of-4(± body);
  L2 觸發改用多鏡頭垃圾桶觸發(results/m6/multiview/frozen.json)、洗手用 L1 最佳組;另做斷鏡頭測試

用法:
    python scripts/eval_violation_epfl.py
    python scripts/eval_violation_epfl.py --round 2
    python scripts/eval_violation_epfl.py --round 3
"""
import argparse
import csv
import itertools
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
INTEG = ROOT / "integrated"
sys.path.insert(0, str(INTEG / "src"))                  # 只用 integrated/ 的模組

from common.zones import get_zone                                         # noqa: E402
from m4_track.det_cache import DetCache                                   # noqa: E402
from m6_events.violation import (HandHygieneFSM, dilate, side_series, steps,  # noqa: E402
                                 touch_series, trash_intervals, zone_any_hand_series, zone_body_series,
                                 zone_hand_series)

CLIPS = Path("D:/yizhen/event_clips")
CACHE = ROOT / "results/det_cache/event_clips_s3"
OUT = ROOT / "results/m6/violation"
TRASH_WORDS = ("丟", "垃圾", "廚餘", "果皮")
GRID = dict(wash_mode=["hand", "body"], wash_s=[1.0, 2.0, 4.0], D=[1.0, 2.0])
AMBIGUOUS = "洗鍋具"
B4 = ["Boutput0", "Boutput1", "Boutput2", "Boutput3"]          # 第 3 輪:上方鏡頭看洗手台


def mv_trash_intervals(e, T):
    """第 3 輪 L2:多鏡頭垃圾桶觸發(凍結設定),平滑後連續 ≥ T 秒的區間(步)。"""
    import eval_multiview_triggers as MV
    from m6_events.smoothing import hysteresis
    from m6_events.violation import intervals
    cfg = json.loads((MV.OUT / "frozen.json").read_text(encoding="utf-8"))["trash"]["multiview"]
    zones = {v: get_zone(INTEG / f"configs/zones/epfl_{v}.json", "垃圾桶區") for v in MV.TRASH_VIEWS}
    d = MV.series([e], zones)[e["id"]]
    sm = hysteresis(MV.raw_of(d, cfg), *MV.EV_SMOOTH)
    need = max(1, int(round(cfg["T"] / MV.STEP)))
    return [(a, b) for a, b in intervals(sm[:T]) if b - a >= need]


def load_events():
    ev = {}
    for r in csv.DictReader(open(CLIPS / "clips_index.csv", encoding="utf-8-sig")):
        if r["event_category"] != "沒洗手":
            continue
        e = ev.setdefault(r["annotation_id"], dict(
            id=r["annotation_id"], session=r["session_id"], note=r["note"],
            a0=float(r["annot_start_sec"]) - float(r["clip_start_sec"]),
            a1=float(r["annot_end_sec"]) - float(r["clip_start_sec"]), clips={}))
        e["clips"][r["view"]] = Path(r["clip_file"]).stem
    out = list(ev.values())
    for e in out:
        e["gt"] = "合格" if ("有洗手" in e["note"] or '"有"洗手' in e["note"]) else "違規"
        e["trash_trigger"] = any(w in e["note"] for w in TRASH_WORDS)
        e["ambiguous"] = AMBIGUOUS in e["note"]
    return out


def score(rows, key="pred"):
    """違規 R、合格辨識率、正確率;「未解除」歸為違規。"""
    pv = lambda r: "違規" if r[key] in ("違規", "未解除") else r[key]
    v = [r for r in rows if r["gt"] == "違規"]
    ok = [r for r in rows if r["gt"] == "合格"]
    vr = sum(pv(r) == "違規" for r in v) / max(len(v), 1)
    okr = sum(pv(r) == "合格" for r in ok) / max(len(ok), 1)
    acc = sum(pv(r) == r["gt"] for r in rows) / max(len(rows), 1)
    return dict(n=len(rows), violation_R=round(vr, 3), compliant_rate=round(okr, 3), acc=round(acc, 3),
                unresolved=sum(r[key] == "未解除" for r in rows))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", type=int, default=1, choices=[1, 2, 3])
    args = ap.parse_args()
    pre = "" if args.round == 1 else f"r{args.round}_"
    if args.round == 3:
        sys.path.insert(0, str(ROOT / "scripts"))
    if args.round == 2:
        GRID["wash_mode"] = ["hand", "body", "close_hand", "close_hand_body"]
    if args.round == 3:
        GRID.update(wash_mode=["close_hand"] + [f"close4_{k}{b}" for b in ("", "_body") for k in (1, 2)],
                    wash_s=[1.0, 2.0], D=[1.0])
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load((INTEG / "configs/violation_rules.yaml").read_text(encoding="utf-8"))
    step, cams, hh = cfg["step_s"], cfg["cameras"], cfg["hand_hygiene"]
    close = cams["close"]
    zp = {v: INTEG / p for v, p in cfg["zones"].items()}
    sink = get_zone(zp[hh["wash_cam"]], hh["sink_zone"])
    tz = {v: get_zone(zp[v], cfg["trash_event"]["zone"]) for v in ("output0", "Aoutput1")}
    events = load_events()
    print(f"{len(events)} 個事件:合格 {sum(e['gt'] == '合格' for e in events)}、違規 {sum(e['gt'] == '違規' for e in events)};"
          f"垃圾類觸發 {sum(e['trash_trigger'] for e in events)}", flush=True)

    # 每個事件的零件序列(與網格無關,只算一次)
    parts, b4_parts = {}, {}
    for e in events:
        cs = {v: DetCache(CACHE / f"{e['clips'][v]}.npz")
              for v in dict.fromkeys(["output0"] + close["A"] + close["B"] + hh["close_cams"])}
        T = len(steps(cs["output0"])[0])
        side = side_series(cs[cams["overview"]], T, **cams["side_x"])
        touch = touch_series(cs, T, side, close, cfg["touch_classes"], cfg["association"])
        wash = dict(hand=zone_hand_series(cs[hh["wash_cam"]], T, sink, hh["thr_hand"]),
                    body=zone_body_series(cs[hh["wash_cam"]], T, sink, hh["body_ioa"]))
        if args.round == 3:
            k = int(round(hh["sync_s"] / step))
            b4_raw = {v: zone_any_hand_series(DetCache(CACHE / f"{e['clips'][v]}.npz"), T,
                                              get_zone(INTEG / f"configs/zones/epfl_{v}.json", hh["sink_zone"]),
                                              hh["thr_hand"]) for v in B4}
            b4 = {v: dilate(x, k) for v, x in b4_raw.items()}
            wash["close_hand"] = np.any([b4_raw[v] for v in hh["close_cams"]], 0)       # 同第 2 輪
            body_k = dilate(wash["body"], k)
            for kk in (1, 2):
                wash[f"close4_{kk}"] = np.sum(list(b4.values()), 0) >= kk
                wash[f"close4_{kk}_body"] = wash[f"close4_{kk}"] & body_k
            b4_parts[e["id"]] = (b4, body_k, b4_raw)
        if args.round == 2:
            k = int(round(hh["sync_s"] / step))
            wash["close_hand"] = np.any([zone_any_hand_series(cs[v], T, get_zone(zp[v], hh["sink_zone"]),
                                                              hh["thr_hand"]) for v in hh["close_cams"]], 0)
            wash["close_hand_body"] = dilate(wash["close_hand"], k) & dilate(wash["body"], k)
        if args.round == 3:
            trash = mv_trash_intervals(e, T) if e["trash_trigger"] else []
        else:
            trash = trash_intervals(cs["Aoutput1"], cs["output0"], T, tz["Aoutput1"], tz["output0"],
                                    cfg["trash_event"], step) if e["trash_trigger"] else []
        parts[e["id"]] = dict(T=T, touch=touch, wash=wash, trash=trash, side=Counter(side))
    print(f"零件序列完成 {time.time() - t0:.0f} 秒", flush=True)

    # L1 網格
    grid_rows, per_clip = [], []
    for wm, ws, D in itertools.product(*GRID.values()):
        fsm = HandHygieneFSM(ws, step, cfg["touch_classes"])
        rows = []
        for e in events:
            p = parts[e["id"]]
            v = fsm.run(int(round((max(e["a0"], 0) + D) / step)), p["wash"][wm], p["touch"])
            rows.append(dict(id=e["id"], session=e["session"], note=e["note"], gt=e["gt"], pred=v.result,
                             at_s=None if v.at is None else round(v.at * step, 1),
                             touched="、".join(v.touched), wash_s=round(v.wash_steps * step, 1),
                             ambiguous=int(e["ambiguous"])))
        s_all = score(rows)
        s_amb = score([r for r in rows if not r["ambiguous"]])
        grid_rows.append(dict(wash_mode=wm, wash_s=ws, D=D, **s_all,
                              **{f"excl_amb_{k}": v for k, v in s_amb.items() if k != "n"}))
        for r in rows:
            per_clip.append(dict(wash_mode=wm, wash_s_cfg=ws, D=D, **r))
        print(f"  L1 {wm:4s} wash {ws} D {D}:違規 R {s_all['violation_R']:.2f}  合格辨識 {s_all['compliant_rate']:.2f}"
              f"  正確 {s_all['acc']:.2f}  未解除 {s_all['unresolved']}", flush=True)

    # 零件診斷(與網格無關)
    diag = []
    for e in events:
        p = parts[e["id"]]
        d0 = int(round((max(e["a0"], 0) + 1) / step))
        diag.append(dict(id=e["id"], gt=e["gt"], note=e["note"],
                         **{f"wash_{m}_s": round(p["wash"][m][d0:].sum() * step, 1) for m in p["wash"]},
                         touch_any_s=round(p["touch"][d0:].any(1).sum() * step, 1),
                         side=dict(p["side"])))

    # 第 3 輪:L1 最佳組(正確率最高,同分取違規 R 高)+ 斷鏡頭
    best, drop_rows = None, []
    if args.round == 3:
        best = max(grid_rows, key=lambda r: (r["acc"], r["violation_R"]))
        print(f"  L1 最佳:{best['wash_mode']} wash {best['wash_s']} D {best['D']} 正確 {best['acc']:.2f}", flush=True)
        fsm_b = HandHygieneFSM(best["wash_s"], step, cfg["touch_classes"])
        wm = best["wash_mode"]
        for dv in B4:
            rows = []
            for e in events:
                p = parts[e["id"]]
                if wm.startswith("close4_"):
                    b4, body_k, _ = b4_parts[e["id"]]
                    kk = int(wm.split("_")[1])
                    w = np.sum([x for v, x in b4.items() if v != dv], 0) >= kk
                    if wm.endswith("_body"):
                        w &= body_k
                elif dv in hh["close_cams"]:
                    w = np.any([b4_parts[e["id"]][2][v] for v in hh["close_cams"] if v != dv], 0)
                else:
                    w = p["wash"][wm]
                vv = fsm_b.run(int(round((max(e["a0"], 0) + best["D"]) / step)), w, p["touch"])
                rows.append(dict(gt=e["gt"], pred=vv.result))
            s = score(rows)
            drop_rows.append(dict(drop=dv, wash_mode=wm, **s))
            print(f"    斷 {dv}:違規 R {s['violation_R']:.2f}  合格辨識 {s['compliant_rate']:.2f}  正確 {s['acc']:.2f}",
                  flush=True)

    # L2:垃圾類觸發,觸發自動偵測(body、wash 2 秒、設定檔其餘值)
    fsm = HandHygieneFSM(hh["wash_s"], step, cfg["touch_classes"])
    l2 = []
    for e in [e for e in events if e["trash_trigger"]]:
        p = parts[e["id"]]
        if not p["trash"]:
            l2.append(dict(id=e["id"], note=e["note"], gt=e["gt"], pred="無觸發", trigger_s=None, at_s=None))
            continue
        s, en = p["trash"][0]
        l2_mode = {2: "close_hand_body", 3: best["wash_mode"]}.get(args.round, hh["wash_mode"])
        if args.round == 3:
            fsm = HandHygieneFSM(best["wash_s"], step, cfg["touch_classes"])
        v = fsm.run(en + int(round(hh["grace_s"] / step)), p["wash"][l2_mode], p["touch"])
        l2.append(dict(id=e["id"], note=e["note"], gt=e["gt"], pred=v.result, trigger_s=round(s * step, 1),
                       at_s=None if v.at is None else round(v.at * step, 1)))
    l2_ok = sum(("違規" if r["pred"] in ("違規", "未解除") else r["pred"]) == r["gt"] for r in l2)
    print(f"  L2 垃圾類 {len(l2)} 個:判對 {l2_ok}", flush=True)

    outs = [("l1_grid.csv", grid_rows), ("l1_per_clip.csv", per_clip), ("diag.csv", diag), ("l2_per_clip.csv", l2)]
    if drop_rows:
        outs.append(("l1_dropout.csv", drop_rows))
    for name, rows in outs:
        with open(OUT / (pre + name), "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    (OUT / f"{pre}summary.json").write_text(json.dumps(dict(grid=grid_rows, l2_correct=l2_ok, l2_n=len(l2),
                                                        l1_best=best, dropout=drop_rows),
                                                 ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"輸出 {OUT},{time.time() - t0:.0f} 秒")


if __name__ == "__main__":
    main()
