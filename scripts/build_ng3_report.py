"""NG3 手碰臉結果報告(§8.92 正式偵測 detect_ng3;真值 = §8.91 人工檢視修正後):逐筆清單 + 詳細資訊片段影片。

- 片段:results/m6/ng/ng3_face_r3_review/per_clip.csv 的 186 個(擦鼻子 7 件、看不出來 1 件列為排除,不剪片段)。
- 預測:正式偵測 integrated/src/m6_events/ng3.py(與評估逐片段一致,§8.92)。
- 影片:output0、Aoutput1 並排;手框、臉中心與判定範圍(0.5 × 肩寬)、手到臉距離、碰臉狀態、違規紀錄、時間軸。
輸出 results/m6/ng/report/NG3/{正確_有違規, 正確_無違規, 誤報, 漏報}/*.mp4 與 NG3/清單.csv(含 EPFL 影像,不進 git)。
舊版(§8.7 手在頭部 + 食材規則)報告移到 results/m6/ng/report/NG3_舊規則_§8.7/。

用法:
    python scripts/build_ng3_report.py            # 清單 + 片段
    python scripts/build_ng3_report.py --no-video # 只出清單
    python scripts/build_ng3_report.py --first 3  # 試跑前 3 段
"""
import argparse
import os
import shutil
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import eval_ng3_face as F                                                 # noqa: E402
from build_ng_report import CAT, write_csv                                # noqa: E402
from m4_track.det_cache import DetCache                                   # noqa: E402
from m6_events.ng3 import detect_ng3, faces, hand_face_dist               # noqa: E402
from m6_events.violation import HAND, at, steps                           # noqa: E402

OUT = ROOT / "results/m6/ng/report/NG3"
OLD = ROOT / "results/m6/ng/report/NG3_舊規則_§8.7"
LINKS = ROOT / "results/m6/ng/report/_links_ng3"
PER = ROOT / "results/m6/ng/ng3_face_r3_review/per_clip.csv"
SRC = ROOT / "results/m6/ng/ng3_face_r3/per_clip.csv"
REVIEW = Path("D:/yizhen/NG3_碰臉誤報人工檢視_20261105/人工檢視表.xlsx")
FPS = 30
USER_EX = {}                                                              # 使用者回查排除:{片段: 說明}


def link(c, v):
    """OpenCV 開不了中文路徑:建 ASCII 檔名的硬連結(同一磁碟,不複製內容)。"""
    p = Path(c["files"][v])
    if not p.exists() and (p.parent / "NG" / p.name).exists():
        p = p.parent / "NG" / p.name
    if str(p).isascii():
        return p
    LINKS.mkdir(parents=True, exist_ok=True)
    q = LINKS / f"{c['key']}_{v}.mp4"
    if not q.exists():
        os.link(p, q)
    return q


def status(c, y, n3, step):
    """逐步狀態:各鏡頭手在臉範圍、臉中心與半徑、手到臉距離(肩寬倍數)、碰臉段落。"""
    vs = [v for v in n3["views"] if v in c["caches"] and F.pose_path(c, v).exists()]
    dcs = {v: DetCache(c["caches"][v]) for v in vs}
    pzs = {v: dict(np.load(F.pose_path(c, v))) for v in vs}
    tr = {}
    ev = detect_ng3(dcs, pzs, step, n3, trace=tr)
    if not vs:
        return None, ev
    T = tr["T"]
    face, ratio = {}, {}
    for v in vs:
        idx = steps(dcs[v])
        fd, rt = {}, np.full(T, np.nan)
        for i in np.unique(pzs[v]["step"]):
            i = int(i)
            if i >= T:
                continue
            fs = faces(pzs[v], i, n3["kp_thr"])
            if not fs:
                continue
            xy, cf, cl = at(dcs[v], idx, i)
            H = xy[(cl == HAND) & (cf >= n3["thr_hand"])]
            d = [(hand_face_dist(H, fc) / sw).min() for fc, sw in fs if len(H)]
            for j in range(i, min(T, i + n3["pose_every"])):
                fd[j] = [(tuple(fc), n3["r"] * sw) for fc, sw in fs]
                rt[j] = min(d) if d else np.nan
        face[v], ratio[v] = fd, rt
    st = dict(T=T, views=vs, raws=tr["raws"], raw=tr["raw"], segs=[tuple(s) for s in tr["segs"]], faces=face,
              ratio=ratio, r=n3["r"], y=y, mn=max(1, int(round(n3["min_s"] / step))))
    return st, ev


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--first", type=int, default=0)
    args = ap.parse_args()
    cfg = yaml.safe_load((ROOT / "integrated/configs/violation_rules.yaml").read_text(encoding="utf-8"))
    n3, step = {k: v for k, v in cfg["ng3"].items() if k != "enable"}, cfg["step_s"]
    per = pd.read_csv(PER, encoding="utf-8-sig").set_index("key")
    allc = pd.read_csv(SRC, encoding="utf-8-sig")
    rv = pd.read_excel(REVIEW, sheet_name="人工檢視表")
    rv = dict(zip(rv["片段"], rv.iloc[:, 8].astype(int)))
    C = F.E.load_clips()
    u = pd.read_csv(ROOT / "results/m6/ng/details_corrected/ng3_user_corrections_20261109.csv", encoding="utf-8-sig",
                    dtype=str).fillna("")
    USER_EX.update({k: n for k, e, n in zip(u.key, u.exclude, u.note) if e})
    if OUT.exists() and not OLD.exists() and not (OUT / "誤報分類.csv").exists():
        shutil.move(str(OUT), str(OLD))                          # 第一次執行:舊規則報告移走保留
    rows, jobs, n = [], [], 0
    for _, a in allc.iterrows():
        key = a.key
        c = C[key]
        base = dict(key=key, pid=c["pid"], team_ng=c["ng"], team_cat=c["cat"], note=c["note"] or "",
                    dur_s=round(c["dur"], 1), review={1: "有碰臉", 2: "沒碰臉", 3: "看不出來"}.get(rv.get(key), ""))
        if key not in per.index:
            why = ("擦鼻子(衛生紙擦鼻子不算碰臉)" if "擦鼻子" in str(c["note"]) else
                   USER_EX[key] if key in USER_EX else "人工檢視看不出來")
            rows.append(dict(base, category=f"排除({why})", gt="", pred="", sys_events="", video=""))
            continue
        y = int(per.loc[key, "y"])
        st, ev = status(c, y, n3, step)
        pred = int(bool(ev))
        cat = CAT[("T" if pred == y else "F") + ("P" if pred else "N")]
        n += 1
        name = f"{n:03d}_{key}.mp4"
        r = dict(base, gt=y, pred=pred, category=cat,
                 gt_source=("使用者回查改正例" if isinstance(per.loc[key, "user"], str) and per.loc[key, "user"]
                            else "人工檢視改正例" if rv.get(key) == 1 else "團隊標註"),
                 sys_events="、".join(f"{e['start']:.1f}–{e['end']:.1f} 秒({e['views']})" for e in ev) or "無",
                 video=f"NG3/{cat}/{name}")
        rows.append(r)
        if st is None:
            r["video"] = ""
            continue
        team = f"團隊 {c['ng']} {c['cat']}" + (f" {c['note']}" if c["note"] else "")
        label = (f"NG3 {cat} | {c['pid']} | {team} | 真值 {'有' if y else '無'}碰臉"
                 f"{'(修正)' if r['gt_source'] != '團隊標註' else ''} / 系統 {'有' if pred else '無'} NG3")
        jobs.append(dict(out=OUT / cat / name, n_frames=int(c["dur"] * FPS), start_frame=0, offset=0.0, label=label,
                         kind="ng3d", status=st, thr={"手": n3["thr_hand"]},
                         views=[dict(name=v, video=link(c, v), cache=c["caches"][v], zones={}) for v in st["views"]]))
    OUT.mkdir(parents=True, exist_ok=True)
    write_csv(OUT / "清單.csv", rows)
    cnt = pd.Series([r["category"] for r in rows]).value_counts().to_dict()
    tp, fp, fn = cnt.get(CAT["TP"], 0), cnt.get(CAT["FP"], 0), cnt.get(CAT["FN"], 0)
    print(f"NG3 {cnt}  R {tp / max(tp + fn, 1):.3f}  P {tp / max(tp + fp, 1):.3f}", flush=True)
    if args.no_video:
        return
    if args.first:
        jobs = jobs[:args.first]
    for j in jobs:
        Path(j["out"]).parent.mkdir(parents=True, exist_ok=True)
    import ng_overlay
    with Pool(args.workers) as pool:
        for k, _ in enumerate(pool.imap_unordered(ng_overlay.render, jobs), 1):
            if k % 20 == 0:
                print(f"  片段 {k}/{len(jobs)}", flush=True)
    print("片段", len(jobs), "完成", flush=True)


if __name__ == "__main__":
    main()
