"""§8.26:NG4a 用人工檢視的候選片段測試(現有規則 A + 候選初判 B)。

預先登記:docs/M6_NG準則評估_預先登記_20261027.md §8.26
真值:D:/yizhen/NG4_OK_candidates_20261102/人工檢視表(已修改).xlsx(1 反例 / 2 NG4 / 3 都不是)
片段 data/ng4_ok_win/<編號>/<鏡頭>.mp4、快取 results/det_cache/ng4_ok_win_s3/<編號>/<鏡頭>.npz、
輸出 results/m6/ng/ng4_ok_review/(皆不進 git)

用法:
    python scripts/eval_ng4_ok_review.py
"""
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import eval_ng_clips as E                                                 # noqa: E402
from fetch_ng_windows import cache                                        # noqa: E402
from m4_track.det_cache import DetCache                                   # noqa: E402
from m6_events.violation import steps                                     # noqa: E402
from common.zones import get_zone                                         # noqa: E402

DELIV = Path("D:/yizhen/NG4_OK_candidates_20261102")
ORIG, REVIEWED = DELIV / "人工檢視表.xlsx", DELIV / "人工檢視表(已修改).xlsx"
VID = Path("D:/yizhen/Public_release_videos")
WIN = ROOT / "data/ng4_ok_win"
CACHE = ROOT / "results/det_cache/ng4_ok_win_s3"
OUT = ROOT / "results/m6/ng/ng4_ok_review"
VIEWS = ["Aoutput1", "Aoutput2", "Aoutput3", "Boutput1", "Boutput2"]
PRE_S, POST_S, FPS, STEP = 1.0, 5.0, 30, 0.1
R_COL = "檢視結果(反例/NG4/都不是)"
MAIN_WP = (1.0, 1.0)


def sec(hms):
    h, m, s = hms.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def load():
    """回傳候選清單(依原始檢視表的編號)與兩種真值解讀。"""
    a = pd.read_excel(ORIG).fillna("")
    b = pd.read_excel(REVIEWED).fillna("")
    key = lambda r: (r["場次"], r["食材"], r["清洗"], r["放下"])
    by_content = {key(r): r["編號"] for _, r in a.iterrows()}
    gt_id = {r["編號"]: r[R_COL] for _, r in b.iterrows()}                   # 主分析:以編號為準
    gt_content = {by_content[key(r)]: r[R_COL] for _, r in b.iterrows()}   # 敏感度:以內容為準
    rows = []
    for _, r in a.iterrows():
        w0, w1 = [sec(x) for x in r["清洗"].split("–")]
        rows.append(dict(id=r["編號"], session=r["場次"], food=r["食材"], wash_start=w0, wash_end=w1,
                         put_end=sec(r["放下"]), auto=r["系統初判"], example=r["範例已有"] != "",
                         gt_id=int(gt_id[r["編號"]]), gt_content=int(gt_content[r["編號"]])))
    return rows


def prepare(c):
    """切片段 + 建快取(已存在跳過)。"""
    t0 = max(0.0, c["wash_start"] - PRE_S)
    dur = c["put_end"] + POST_S - t0
    d = WIN / c["id"]
    for v in VIEWS:
        out = d / f"{v}.mp4"
        src = VID / c["session"] / "videos" / f"{v}.mp4"
        if out.exists() or not src.exists():
            continue
        d.mkdir(parents=True, exist_ok=True)
        subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{t0:.3f}", "-i", str(src), "-t", f"{dur:.3f}", "-an",
                        "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-r", str(FPS), str(out)], check=True)
    cache([d / f"{v}.mp4" for v in VIEWS], VIEWS, CACHE / c["id"])


def rule_a(c):
    cs = {v: DetCache(CACHE / c["id"] / f"{v}.npz") for v in VIEWS if (CACHE / c["id"] / f"{v}.npz").exists()}
    T = min(len(steps(x)[0]) for x in cs.values())
    sink = {v: get_zone(E.INTEG / f"configs/zones/epfl_{v}.json", "洗手台") for v in ("Boutput1", "Boutput2")}
    grid, why = E.ng4_rule(cs, T, {k: v for k, v in sink.items() if v}, STEP)
    return grid, why


def prf(pairs):
    tp = sum(p and g for p, g in pairs)
    fp = sum(p and not g for p, g in pairs)
    fn = sum(g and not p for p, g in pairs)
    tn = sum(not p and not g for p, g in pairs)
    R, P = tp / max(tp + fn, 1), tp / max(tp + fp, 1)
    (rl, rh), (pl, ph) = E.wilson(tp, tp + fn), E.wilson(tp, tp + fp)
    return dict(n=len(pairs), tp=tp, fp=fp, fn=fn, tn=tn, R=round(R, 3), R_ci=f"{rl}-{rh}", P=round(P, 3),
                P_ci=f"{pl}-{ph}", F1=round(2 * P * R / max(P + R, 1e-9), 3))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = load()
    for n, c in enumerate(rows, 1):
        prepare(c)
        c["grid"], c["why"] = rule_a(c)
        c["pred_a"] = int(bool(c["grid"][MAIN_WP]))
        print(f"  {n}/{len(rows)} {c['id']} A={c['pred_a']} {c['why']}", flush=True)

    summ = []
    for gk in ("gt_id", "gt_content"):
        use = [c for c in rows if c[gk] in (1, 2)]
        base = sum(c[gk] == 2 for c in use) / len(use)
        print(f"\n== 真值 {gk}:{len(use)} 件,NG4 {sum(c[gk] == 2 for c in use)},基準比例 {base:.3f}")
        for wp in E.GRID_NG4:
            m = prf([(bool(c["grid"][wp]), c[gk] == 2) for c in use])
            summ.append(dict(truth=gk, method=f"A ng4_rule w{wp[0]} p{wp[1]}", base=round(base, 3), **m))
            print(f"  A w{wp[0]} p{wp[1]}: {m}")
        ub = [c for c in use if not c["example"]]
        dec = [c for c in ub if c["auto"] in ("桌面", "砧板或容器")]
        m = prf([(c["auto"] == "桌面", c[gk] == 2) for c in dec])
        cov = round(len(dec) / len(ub), 3)
        summ.append(dict(truth=gk, method="B 初判(有判斷者)", base=round(base, 3), coverage=cov, **m))
        print(f"  B 有判斷 {len(dec)}/{len(ub)}(覆蓋率 {cov}):{m}")
        for name, as_ng in (("無法判斷當 NG4", True), ("無法判斷當非 NG4", False)):
            m = prf([((c["auto"] == "桌面") or (as_ng and c["auto"] == "無法判斷"), c[gk] == 2) for c in ub])
            summ.append(dict(truth=gk, method=f"B 初判({name})", base=round(base, 3), **m))
            print(f"  B {name}:{m}")
    pd.DataFrame(summ).to_csv(OUT / "summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([{k: v for k, v in c.items() if k != "grid"} | {f"A_w{w}_p{p}": int(bool(c["grid"][(w, p)]))
                                                                   for w, p in E.GRID_NG4} for c in rows]
                 ).to_csv(OUT / "per_candidate.csv", index=False, encoding="utf-8-sig")
    print(f"\n輸出 {OUT}")


if __name__ == "__main__":
    main()
