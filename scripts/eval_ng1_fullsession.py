"""NG1 整場誤報測試:用 integrated/ 的 NG 規則跑完整場次,系統接觸段與團隊事件、EPFL Trash 動作比對。

預先登記:docs/M6_NG準則評估_預先登記_20261027.md §8.17
- 快取:results/det_cache/epfl_sessions_s3/<參與者>_<場次>/<鏡頭>.npz(不進 git)
- 輸出:results/m6/ng/fullsession/episodes.csv、summary.csv、fp_review/*.jpg(含團隊標註與 EPFL 影像,不進 git)

用法:
    python scripts/eval_ng1_fullsession.py            # 跑系統 + 比對 + 疑似誤報抽圖
    python scripts/eval_ng1_fullsession.py --no-img   # 不抽圖
"""
import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
INTEG = ROOT / "integrated"
sys.path.insert(0, str(INTEG / "src"))
from m4_track.det_cache import DetCache                                   # noqa: E402
from m6_events.ng import _episodes, detect_ng, series                     # noqa: E402

SESSIONS = ["YH2002/2023_12_11_16_11_15", "YH2004/2023_07_12_14_02_36", "YH2011/2023_07_10_15_03_05",
            "YH2018/2023_05_31_09_07_02", "YH2025/2023_08_04_09_01_53", "YH2029/2023_06_12_15_05_02",
            "YH2040/2023_08_25_09_12_12", "YH2043/2023_10_09_09_06_48"]
CACHE = ROOT / "results/det_cache/epfl_sessions_s3"
VID = ROOT / "data/epfl_sessions"
ANN = ROOT / "data/epfl_ann"
TEAM = ROOT / "results/ng_clips/team_events_vs_ng.csv"
OUT = ROOT / "results/m6/ng/fullsession"
PAD_S, MERGE_REF_S = 2.0, 5.0
FONT = "C:/Windows/Fonts/msjh.ttc"


def overlap(a0, a1, b0, b1):
    return min(a1, b1) >= max(a0, b0)


def epfl_trash(s):
    d = pd.read_excel(ANN / s / "annotations/actions_annotations.xlsx")
    d = d[d["Nouns"].astype(str).str.contains("Trash")]
    return [(float(a), float(b), f"{v} {n}") for a, b, v, n in zip(d["Start"], d["End"], d["Verbs"], d["Nouns"])]


def team_trash(s):
    return [(float(r["annot_start_sec"]), float(r["annot_end_sec"]), r["note"])
            for r in csv.DictReader(open(TEAM, encoding="utf-8-sig"))
            if r["session_id"].endswith(s) and r["event_category"] == "垃圾桶"]


def merged(ivs, gap):
    out = []
    for a, b in sorted(ivs):
        if out and a - out[-1][1] < gap:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


def frame(video, t, w=640, h=360):
    p = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{t:.2f}", "-i", str(video), "-frames:v", "1",
                        "-vf", f"scale={w}:{h}", "-f", "image2pipe", "-vcodec", "png", "-"],
                       capture_output=True, check=True).stdout
    from io import BytesIO
    return Image.open(BytesIO(p)).convert("RGB")


def review_sheets(fps_rows):
    """疑似誤報:每段取接觸中點 output0 + Aoutput1 各一幀,每張圖 6 段。"""
    d = OUT / "fp_review"
    d.mkdir(parents=True, exist_ok=True)
    font = ImageFont.truetype(FONT, 22)
    for n in range(0, len(fps_rows), 6):
        rows = fps_rows[n:n + 6]
        sheet = Image.new("RGB", (1280, 360 * len(rows)))
        for i, r in enumerate(rows):
            t = (r["start"] + r["end"]) / 2
            src = VID / r["session"]
            for j, v in enumerate(("output0", "Aoutput1")):
                sheet.paste(frame(src / f"{v}.mp4", t), (640 * j, 360 * i))
            ImageDraw.Draw(sheet).text((8, 360 * i + 8), f"#{r['idx']} {r['session']} {r['start']:.1f}-{r['end']:.1f}s",
                                       fill=(255, 255, 0), font=font, stroke_width=2, stroke_fill=(0, 0, 0))
        sheet.save(d / f"fp_{n // 6 + 1:02d}.jpg", quality=85)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-img", action="store_true")
    args = ap.parse_args()
    cfg = yaml.safe_load((INTEG / "configs/violation_rules.yaml").read_text(encoding="utf-8"))
    step = cfg["step_s"]
    OUT.mkdir(parents=True, exist_ok=True)
    eps_rows, summ = [], []
    for s in SESSIONS:
        key = s.replace("/", "_")
        caches = {p.stem: DetCache(p) for p in sorted((CACHE / key).glob("*.npz"))}
        ser = series(caches, cfg, INTEG)
        ng = detect_ng(ser, cfg, 0.0)
        ng1_starts = {e["start"] for e in ng if e["type"] == "NG1"}
        eps = _episodes(ser["contacts"], int(round(cfg["ng"]["merge_s"] / step)))
        ep_trash, tm = epfl_trash(s), team_trash(s)
        hours = ser["T"] * step / 3600
        cnt = dict(T=0, E=0, F=0)
        for a, b, parts in eps:
            a_s, b_s = round(a * step, 2), round(b * step, 2)
            t_hit = [x for x in tm if overlap(a_s, b_s, x[0] - PAD_S, x[1] + PAD_S)]
            e_hit = [x for x in ep_trash if overlap(a_s, b_s, x[0] - PAD_S, x[1] + PAD_S)]
            cls = "T" if t_hit else "E" if e_hit else "F"
            cnt[cls] += 1
            eps_rows.append(dict(idx=len(eps_rows), session=s, start=a_s, end=b_s, n_contacts=len(parts),
                                 ng1=int(a_s in ng1_starts), cls=cls, team="、".join(x[2] for x in t_hit),
                                 epfl="、".join(sorted({x[2] for x in e_hit}))))
        put = merged([(a, b) for a, b, lab in ep_trash if lab.split()[0] in ("Put", "Throw")], MERGE_REF_S)
        put_hit = sum(any(overlap(r["start"], r["end"], a - PAD_S, b + PAD_S) for r in eps_rows if r["session"] == s)
                      for a, b in put)
        n = len(eps)
        summ.append(dict(session=s, hours=round(hours, 3), episodes=n, ng1=sum(e["type"] == "NG1" for e in ng),
                         ep_per_h=round(n / hours, 1), ng1_per_h=round(sum(e["type"] == "NG1" for e in ng) / hours, 1),
                         **cnt, P=round((cnt["T"] + cnt["E"]) / max(n, 1), 3), F_per_h=round(cnt["F"] / hours, 2),
                         put_clusters=len(put), put_covered=put_hit, team_events=len(tm)))
        print(json.dumps(summ[-1], ensure_ascii=False), flush=True)
    tot = {k: sum(x[k] for x in summ) for k in ("hours", "episodes", "ng1", "T", "E", "F", "put_clusters",
                                                 "put_covered", "team_events")}
    tot.update(session="全部", ep_per_h=round(tot["episodes"] / tot["hours"], 1),
               ng1_per_h=round(tot["ng1"] / tot["hours"], 1), P=round((tot["T"] + tot["E"]) / tot["episodes"], 3),
               F_per_h=round(tot["F"] / tot["hours"], 2), hours=round(tot["hours"], 3))
    summ.append(tot)
    print(json.dumps(tot, ensure_ascii=False))
    for name, rows in (("episodes", eps_rows), ("summary", summ)):
        with open(OUT / f"{name}.csv", "w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, list(rows[-1] if name == "summary" else rows[0]))
            w.writeheader()
            w.writerows(rows)
    if not args.no_img:
        review_sheets([r for r in eps_rows if r["cls"] == "F"])


if __name__ == "__main__":
    main()
