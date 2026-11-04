"""NG4 結果報告(正式設定;§8.39 洗完才找 → §8.46 起清洗期間也找):逐筆清單 + 片段影片,依 正確_有違規 / 正確_無違規 / 誤報 / 漏報 分資料夾。

只讀既有片段與快取、用 integrated/configs/violation_rules.yaml 的 ng4 設定重算,不改任何判定。
- 範圍:人工檢視表(已修改).xlsx 的候選中,食材可偵測 D1(§8.34)且檢視結果為 1(反例)/ 2(NG4)者;
  其餘(都不是、食材不可偵測)列在清單中標「排除」,不剪片段。
- 真值:檢視結果 2 = 有 NG4;預測:視窗內有食材類 NG4(確定 + 待確認)。
輸出 results/m6/ng/report/NG4/{正確_有違規, 正確_無違規, 誤報, 漏報}/*.mp4 與 NG4/清單.csv(含 EPFL 影像,不進 git)。

用法:
    python scripts/build_ng4_report.py            # 清單 + 片段
    python scripts/build_ng4_report.py --no-video # 只出清單
"""
import argparse
import sys
from multiprocessing import Pool
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import eval_ng4_trigger as X                                              # noqa: E402
from build_ng_report import CAT, hms, write_csv, zone                     # noqa: E402
from m6_events.ng4 import detect_ng4                                      # noqa: E402

OUT = ROOT / "results/m6/ng/report/NG4"
DET = ROOT / "results/m6/ng/ng4_detectable/per_candidate.csv"
VIDEO_VIEWS = ("Aoutput1", "Aoutput2", "Boutput1")
GT_NAME = {1: "反例(放砧板 / 容器)", 2: "NG4(放桌面)", 3: "都不是"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    cfg = yaml.safe_load((X.E.INTEG / "configs/violation_rules.yaml").read_text(encoding="utf-8"))["ng4"]
    params = {k: v for k, v in cfg.items() if k not in ("enable", "sink_cams")}
    det = pd.read_csv(DET).set_index("id")
    rows, jobs, n = [], [], 0
    for c in X.R.load():
        t0 = max(0.0, c["wash_start"] - X.R.PRE_S)
        cs = X.load(X.R.CACHE / c["id"])
        ev = detect_ng4(cs, X.SINK, X.STEP, params=params, t0=t0)
        food = [e for e in ev if e["obj"] == "食材"]
        pred = int(any(e["ng4"] for e in food))
        gt = c["gt_id"]
        d1 = bool(det.loc[c["id"], "d1"])
        if gt not in (1, 2):
            cat = "排除(都不是)"
        elif not d1:
            cat = "排除(食材不可偵測)"
        else:
            cat = CAT[("T" if pred == (gt == 2) else "F") + ("P" if pred else "N")]
        sys_txt = "、".join(f"{e['obj']} {hms(e['start'])}–{hms(e['end'])} {e['place']}"
                           + (f"({e['status']})" if e["ng4"] else "(合格)") for e in ev) or "無清洗 → 放下事件"
        r = dict(id=c["id"], session=c["session"], food=c["food"], wash=f"{hms(c['wash_start'])}–{hms(c['wash_end'])}",
                 put_end=hms(c["put_end"]), review=GT_NAME.get(gt, gt), gt=int(gt == 2), pred=pred, category=cat,
                 sys_food_events=len(food), sys_events=sys_txt, video="")
        if not cat.startswith("排除"):
            n += 1
            pid = c["id"][:6]
            name = f"{n:03d}_{c['id']}_{c['food']}.mp4"
            r["video"] = f"NG4/{cat}/{name}"
            label = (f"NG4 {cat} | {pid} {c['food']} | 人工檢視 {GT_NAME[gt]} → 系統{'有' if pred else '無'} NG4\n"
                     f"標註清洗 {r['wash']} 放下 {r['put_end']} | 系統:{sys_txt}")
            src = ROOT / "data/ng4_ok_win" / c["id"]
            dets = [(e["start"], e["end"], f"{e['obj']}/{e['place']}" + (f"/{e['status']}" if e["ng4"] else "/合格"))
                    for e in ev]
            n_frames = int((c["put_end"] + X.R.POST_S - t0) * 30)
            jobs.append(dict(out=OUT / cat / name, n_frames=n_frames, start_frame=0, offset=t0, label=label,
                             kind="ng4", status=dict(dets=dets),
                             views=[dict(name=v, video=src / f"{v}.mp4", cache=X.R.CACHE / c["id"] / f"{v}.npz",
                                         zones={"洗手台": zone(v, "洗手台")} if v == "Boutput1" else {})
                                    for v in VIDEO_VIEWS]))
        rows.append(r)
        print(f"  {c['id']} {cat}", flush=True)
    write_csv(OUT / "清單.csv", rows)
    cnt = pd.Series([r["category"] for r in rows]).value_counts().to_dict()
    tp, fp, fn = cnt.get(CAT["TP"], 0), cnt.get(CAT["FP"], 0), cnt.get(CAT["FN"], 0)
    print(f"NG4 {cnt}  R {tp / max(tp + fn, 1):.3f}  P {tp / max(tp + fp, 1):.3f}", flush=True)
    if args.no_video:
        return
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
