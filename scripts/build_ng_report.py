"""NG1–NG4 結果報告用:逐筆清單 + 片段影片(依 正確 / 誤報 / 漏報 分資料夾)。

只讀既有評估結果與快取,不改任何判定:
- NG1 / NG2:results/m6/ng/details_corrected/events.csv(登記 §8.20–8.21,主分析),系統時間由同一份快取重算(邏輯同 eval_ng_details.py)
- NG3:results/m6/ng/cv_per_clip.csv 依 eval_ng_clips.cross_validate 重建每個片段的 CV 預測(登記 §8.6–8.7,規則版)
- NG4:results/m6/ng/ng4/truth.csv、detections.csv(登記 §8.13–8.14,預設 w 1、p 1)

輸出 results/m6/ng/report/NG{1..4}/{正確_有違規, 正確_無違規, 誤報, 漏報}/*.mp4 與 NG{1..4}/清單.csv
(含團隊標註與 EPFL 影像,不進 git)。

用法:
    python scripts/build_ng_report.py            # 清單 + 片段
    python scripts/build_ng_report.py --no-video # 只出清單
"""
import argparse
import csv
import glob
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import eval_ng_clips as E                                                 # noqa: E402
import eval_ng_details as D                                               # noqa: E402
import eval_ng_trash_events as M                                          # noqa: E402
from m6_events.violation import HandHygieneFSM, intervals                 # noqa: E402

OUT = ROOT / "results/m6/ng/report"
FPS = 30
CAT = {"TP": "正確_有違規", "TN": "正確_無違規", "FP": "誤報", "FN": "漏報"}
NG_DIR = Path("D:/yizhen/NG_clips")
EV_DIR = Path("D:/yizhen/event_clips")
FONT = "C:/Windows/Fonts/msjh.ttc"


def hms(t):
    return f"{int(t // 3600):02d}:{int(t % 3600 // 60):02d}:{t % 60:04.1f}"


def cat_of(gt, pred):
    return "TP" if gt and pred else "FN" if gt else "FP" if pred else "TN"


# ── NG1 / NG2 ────────────────────────────────────────────────────

def ng12():
    cfg = yaml.safe_load((E.INTEG / "configs/violation_rules.yaml").read_text(encoding="utf-8"))
    step, hh = cfg["step_s"], cfg["hand_hygiene"]
    res = {r["event"]: r for r in csv.DictReader(open(ROOT / "results/m6/ng/details_corrected/events.csv",
                                                       encoding="utf-8-sig"))}
    events = M.load_events()
    corr = D.load_corrections(events)
    rows = {"NG1": [], "NG2": []}
    for e in events:
        k = e["key"][-8:]
        c = corr.get(k, {})
        if c.get("exclude"):
            for ng in rows:
                rows[ng].append(dict(event=k, session=e["session"], note=e["note"], team_ng=e["ng"],
                                     category="排除", correction="團隊建議略過"))
            continue
        x = res[k]
        r = E.rule_part(dict(key=e["key"], caches={v: M.CACHE / e["key"] / f"{v}.npz" for v in M.VIEWS}), cfg)
        w0 = e["f0"] / FPS
        tsec = lambda i: w0 + i * step
        s_ev, e_ev = int((e["start"] - w0) / step), int((e["end"] - w0) / step)
        tr = [(s, q) for s, q in (r["trig_base"] or []) if s <= e_ev + 10 and q >= s_ev - 10]
        base = dict(event=k, key=e["key"], session=e["session"], note=e["note"], team_ng=e["ng"],
                    correction=x["correction"], ev_start=e["start"], ev_end=e["end"], w0=w0,
                    dur=e["dur"], sys_contact="", sys_wash_first="", sys_wash5_s="", sys_ng2_at=x["ng2_at"], sys_touched="")
        if tr:
            a, b = tr[0][0], tr[-1][1]
            base["sys_contact"] = f"{hms(tsec(a))}–{hms(tsec(b))}"
            base["_cend"] = tsec(b)
            dirty = b + int(round(hh["grace_s"] / step))
            ws = np.flatnonzero(r["wash"][dirty:])                 # 髒手後第一個洗手步(含之前就開始、延續過來的)
            base["sys_wash_first"] = hms(tsec(dirty + ws[0])) if len(ws) else "無"
            # NG1 判準本身:接觸結束 + grace 到接觸結束 + 5 秒內的洗手累計秒數(< wash_s 即判 NG1)
            base["sys_wash5_s"] = round(r["wash"][dirty:b + int(M.NG1_WASH_WITHIN_S / step)].sum() * step, 1)
            v = HandHygieneFSM(hh["wash_s"], step, cfg["touch_classes"]).run(dirty, r["wash"], r["touch"])
            if v.result == "違規":
                base["sys_touched"] = "、".join(v.touched)
        for ng in ("NG1", "NG2"):
            gt = x[f"gt_{ng.lower()}"]
            if gt == "":                                  # 團隊未回答 NG2 → 該類排除
                rows[ng].append(dict(base, category="排除", gt="未知", pred=x[f"pred_{ng.lower()}"]))
                continue
            gt, pred = int(gt), int(x[f"pred_{ng.lower()}"])
            row = dict(base, gt=gt, pred=pred, category=CAT[cat_of(gt, pred)])
            if ng == "NG1":
                row["team_ng1_ok"] = x["gt_ng1_ok"]
            rows[ng].append(row)
    return rows


# ── NG3 ──────────────────────────────────────────────────────────

def ng3():
    raw = list(csv.DictReader(open(ROOT / "results/m6/ng/cv_per_clip.csv", encoding="utf-8-sig")))
    conv = lambda v: None if v == "" else v in ("True", "1")
    rows = []
    for r in raw:
        if r["excluded"] in ("1", "True"):
            continue
        d = dict(r)
        for col in r:
            if col.startswith("ng3rule"):
                d[col] = conv(r[col])
        d["m3_train"] = r["m3_train"] in ("1", "True")
        rows.append(d)
    _, grid = E.DETECTORS["NG3_rule"][1:]
    pairs = E.labelled(rows, "NG3", None, True)
    pids = sorted({x["pid"] for x, _ in pairs})
    out = []
    for pid in pids:
        tr = [(x, y) for x, y in pairs if x["pid"] != pid]
        te = [(x, y) for x, y in pairs if x["pid"] == pid]
        best = grid[0]
        for g in grid[1:]:
            if E.prf(tr, g[1])["F1"] > E.prf(tr, best[1])["F1"]:
                best = g
        for x, y in te:
            p = x[best[1]]
            if p is None:
                continue
            out.append(dict(key=x["key"], session=x["session"], pid=x["pid"], team=x["ng"], cat_event=x["cat"],
                            note=x["note"], dur=float(x["dur_s"]), param=best[0], gt=int(y), pred=int(bool(p)),
                            category=CAT[cat_of(y, p)]))
    return out


def ng3_source(key, view):
    """NG3 片段的影片與其在原影片的起點(秒)。"""
    if key.startswith("ev_"):
        a8 = key.split("_")[-1]
        for r in csv.DictReader(open(EV_DIR / "clips_index.csv", encoding="utf-8-sig")):
            if r["annotation_id"].startswith(a8) and r["view"] == view:
                return EV_DIR / r["clip_file"], float(r["clip_start_sec"])
        return None, 0
    s = {f'{x["session_id"].split("/")[1]}_{x["session_id"].split("/")[2]}_{x["no"]}': x
         for x in csv.DictReader(open(ROOT / "results/ng_clips/ng_clip_summary.csv", encoding="utf-8-sig"))}[key]
    t0 = float(s["clip_start_sec"])
    cut = ROOT / "data/ng_clips_cut" / key / f"{view}.mp4"
    if cut.exists():
        return cut, t0
    for r in csv.DictReader(open(NG_DIR / "ng_clips_index.csv", encoding="utf-8-sig")):
        if (r["session_id"] == s["session_id"] and r["view"] == view and r["ng_type"] == s["ng_type"]
                and abs(float(r["clip_start_sec"]) - t0) < 0.01):
            return NG_DIR / "NG" / r["clip_file"], t0
    return None, t0


# ── NG4 ──────────────────────────────────────────────────────────

def ng4():
    tag = "w1.0_p1.0"
    truth = list(csv.DictReader(open(ROOT / "results/m6/ng/ng4/truth.csv", encoding="utf-8-sig")))
    dets = [d for d in csv.DictReader(open(ROOT / "results/m6/ng/ng4/detections.csv", encoding="utf-8-sig"))
            if d["grid"] == tag]
    import eval_ng4_instances as N4
    wins = N4.windows([dict(session=t["session"], start=float(t["start"]), end=float(t["end"])) for t in truth])
    out = []
    for t in truth:
        w = next(w for w in wins if w["session"] == t["session"] and w["t0"] <= float(t["start"]) <= w["t0"] + w["dur"])
        hit = int(t[f"hit_{tag}"])
        m = [d for d in dets if d["window"] == w["key"] and N4.overlap(float(d["start"]), float(d["end"]),
                                                                        float(t["start"]), float(t["end"]))]
        out.append(dict(kind_row="團隊", session=t["session"], window=w["key"], t0=w["t0"], wdur=w["dur"],
                        ng4_kind=t["kind"], start=float(t["start"]), end=float(t["end"]),
                        desc=f'{t["reason"]}:{t["note"]}', gt=1, pred=hit, category=CAT["TP" if hit else "FN"],
                        matched="、".join(f'{d["kind"]}/{d["obj"]} {d["start"]}–{d["end"]}' for d in m)))
    for d in dets:
        if d["match_any"] == "1":
            continue
        w = next(w for w in wins if w["key"] == d["window"])
        out.append(dict(kind_row="系統", session=w["session"], window=w["key"], t0=w["t0"], wdur=w["dur"],
                        ng4_kind=d["kind"], start=float(d["start"]), end=float(d["end"]), desc=f'物件 {d["obj"]}',
                        gt=0, pred=1, category=CAT["FP"], matched=""))
    return out


# ── 影片 ─────────────────────────────────────────────────────────

def render(job, tmp):
    """job:dict(out, inputs=[(檔, 片段內起點秒)], dur, offset, label)。三個鏡頭並排 + 字幕(含原影片時間)。"""
    out = Path(job["out"])                                  # 同名檔直接覆寫(ffmpeg -y)
    out.parent.mkdir(parents=True, exist_ok=True)
    lab = Path(tmp) / f"{abs(hash(str(out)))}.txt"
    lab.write_text(job["label"] + f"\n影片時間 %{{pts:hms:{job['offset']:.2f}}}", encoding="utf-8")
    cmd = ["ffmpeg", "-v", "error", "-y"]
    for f, ss in job["inputs"]:
        cmd += ["-ss", f"{max(0.0, ss):.2f}", "-t", f"{job['dur']:.2f}", "-i", str(f)]
    n = len(job["inputs"])
    fc = "".join(f"[{i}]scale=640:360,setsar=1[v{i}];" for i in range(n))
    fc += "".join(f"[v{i}]" for i in range(n)) + f"hstack={n}[s];" if n > 1 else "[v0]copy[s];"
    fc += f"[s]drawtext=fontfile=msjh.ttc:textfile={lab.name}:fontsize=22:fontcolor=yellow:borderw=2:x=10:y=10:line_spacing=6"
    cmd += ["-filter_complex", fc, "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "26", str(out)]
    p = subprocess.run(cmd, cwd=tmp, capture_output=True, text=True)
    if p.returncode:
        print("失敗", out.name, p.stderr[-300:], flush=True)


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys and not k.startswith("_")]
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--only", nargs="*", help="只剪這些 NG 的片段(例:NG3)")
    args = ap.parse_args()
    jobs = []
    r12 = ng12()
    for ng, rows in r12.items():
        for n, r in enumerate(rows, 1):
            if r["category"] == "排除":
                r["video"] = ""
                continue
            pid = r["session"].split("/")[1]
            name = f"{n:02d}_{pid}_{r['event']}_{r['note'] or '無說明'}.mp4"
            r["video"] = f"{ng}/{r['category']}/{name}"
            if ng == "NG1":
                stop = max(r["ev_end"], r.get("_cend", 0)) + 8
            else:
                stop = max(r["ev_end"], float(r["sys_ng2_at"] or 0)) + 4
            dur = min(stop - r["w0"], r["dur"])
            label = (f"{ng} {r['category']} | {pid} {r['note']} | 團隊 {r['team_ng']}"
                     f"{'(依團隊回覆修正)' if r['correction'] else ''} → 真值 {'有' if r['gt'] else '無'}{ng},"
                     f"系統 {'有' if r['pred'] else '無'}\n事件 {hms(r['ev_start'])}–{hms(r['ev_end'])} | "
                     f"系統碰垃圾桶 {r['sys_contact'] or '無'} | 系統髒手後第一次洗手 {r['sys_wash_first'] or '無'}"
                     + (f"(5 秒內洗手 {r['sys_wash5_s']} 秒)" if r["sys_wash5_s"] != "" else "")
                     + (f" | 系統髒手碰觸 {hms(float(r['sys_ng2_at']))} {r['sys_touched']}" if r["sys_ng2_at"] else ""))
            src = M.CUT / r["key"]
            jobs.append(dict(out=OUT / r["video"], dur=dur, offset=r["w0"], label=label,
                             inputs=[(src / f"{v}.mp4", 0) for v in ("output0", "Aoutput1", "Boutput1")]))
        write_csv(OUT / ng / "清單.csv", rows)

    r3 = ng3()
    for n, r in enumerate(r3, 1):
        name = f"{n:03d}_{r['key']}.mp4"
        r["video"] = f"NG3/{r['category']}/{name}"
        ins, off = [], 0
        for v in ("output0", "Aoutput1", "Aoutput2"):
            f, t0 = ng3_source(r["key"], v)
            if f is not None and f.exists():
                ins.append((f, 0))
                off = t0
        r["video_views"] = len(ins)
        if not ins:
            r["video"] = ""
            continue
        label = (f"NG3 {r['category']} | {r['pid']} {r['cat_event']} {r['note']} | 團隊 {r['team']} → "
                 f"真值 {'有' if r['gt'] else '無'}NG3,系統(CV 參數 {r['param']}){'有' if r['pred'] else '無'}")
        jobs.append(dict(out=OUT / r["video"], dur=r["dur"] + 0.5, offset=off, label=label, inputs=ins))
    write_csv(OUT / "NG3" / "清單.csv", r3)

    r4 = ng4()
    for n, r in enumerate(r4, 1):
        pid = r["session"].split("/")[1]
        name = f"{n:02d}_{r['kind_row']}_{pid}_{r['ng4_kind']}_{int(r['start'])}s.mp4"
        r["video"] = f"NG4/{r['category']}/{name}"
        pad = 3 if r["kind_row"] == "團隊" else 2
        a = max(r["t0"], r["start"] - pad)
        b = min(r["t0"] + r["wdur"], r["end"] + pad)
        label = (f"NG4 {r['category']} | {pid} {r['kind_row']}{r['ng4_kind']} {r['desc']}\n"
                 f"範圍 {hms(r['start'])}–{hms(r['end'])}" + (f" | 系統對到:{r['matched']}" if r["matched"] else ""))
        src = ROOT / "data/ng4_win" / r["window"]
        jobs.append(dict(out=OUT / r["video"], dur=b - a, offset=a, label=label,
                         inputs=[(src / f"{v}.mp4", a - r["t0"]) for v in ("Aoutput1", "Aoutput2", "Boutput1")]))
    write_csv(OUT / "NG4" / "清單.csv", r4)

    for ng, rows in list(r12.items()) + [("NG3", r3), ("NG4", r4)]:
        cnt = {}
        for r in rows:
            cnt[r["category"]] = cnt.get(r["category"], 0) + 1
        print(ng, cnt, flush=True)
    if args.no_video:
        return
    if args.only:
        jobs = [j for j in jobs if Path(j["out"]).relative_to(OUT).parts[0] in args.only]
    tmp = tempfile.mkdtemp()
    shutil.copy(FONT, Path(tmp) / "msjh.ttc")
    with ThreadPoolExecutor(4) as ex:
        list(ex.map(lambda j: render(j, tmp), jobs))
    shutil.rmtree(tmp, ignore_errors=True)
    print("片段", len(jobs), "完成", flush=True)


if __name__ == "__main__":
    main()
