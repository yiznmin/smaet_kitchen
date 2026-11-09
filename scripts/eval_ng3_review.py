"""§8.91(探索性):NG3 第 3 輪誤報 27 件經第三人人工檢視後,以修正真值重算(不再調規則)。

人工檢視表:D:/yizhen/NG3_碰臉誤報人工檢視_20261105/人工檢視表.xlsx(1 有碰臉 / 2 沒碰臉 / 3 看不出來)。
- 1 → 改正例;2 → 維持反例;3 → 排除。
- 只檢視了誤報(預測為正的反例),預測為負的反例(TN)沒檢視:P 已完整修正;R 可能高估(TN 中若有漏標的碰臉會變 FN)。
報兩種:固定 §8.85 選到的 r0.5_hand_m0.0_any;以及修正真值下重跑參與者留一(F1 選格點)。
輸出 results/m6/ng/ng3_face_r3_review/{per_clip.csv, summary.csv}(含團隊標註,不進 git)。

用法:
    python scripts/eval_ng3_review.py
    python scripts/eval_ng3_review.py --integrated   # 核對 integrated/src/m6_events/ng3.py 與評估逐片段一致
    python scripts/eval_ng3_review.py --r4           # §8.94 門檻 / 補空隙 / 多鏡頭融合 12 組
    python scripts/eval_ng3_review.py --r5           # §8.96 另一台看清楚兩隻手才否決
    python scripts/eval_ng3_review.py --r6           # §8.98 放寬距離門檻 / 最短持續(F2 選格點)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "results/m6/ng/ng3_face_r3/per_clip.csv"
REVIEW = Path("D:/yizhen/NG3_碰臉誤報人工檢視_20261105/人工檢視表.xlsx")
USER = ROOT / "results/m6/ng/details_corrected/ng3_user_corrections_20261109.csv"   # §8.93 使用者回查(優先於第三人檢視)
OUT = ROOT / "results/m6/ng/ng3_face_r3_review"
FIXED = "r0.5_hand_m0.0_any"
RES_COL = "檢視結果(1 有碰臉 / 2 沒碰臉 / 3 看不出來)"


def prf(y, p):
    y, p = np.asarray(y, bool), np.asarray(p, bool)
    tp, fp, fn = int((y & p).sum()), int((~y & p).sum()), int((y & ~p).sum())
    R, P = tp / max(tp + fn, 1), tp / max(tp + fp, 1)
    return dict(n=len(y), TP=tp, FP=fp, FN=fn, TN=len(y) - tp - fp - fn, R=round(R, 3), P=round(P, 3),
                F1=round(2 * R * P / max(R + P, 1e-9), 3), F2=round(5 * R * P / max(4 * P + R, 1e-9), 3))


def lopo(d, tags, metric="F1"):
    pairs_y, pairs_p, chosen = [], [], []
    for pid in sorted(d.pid.unique()):
        tr, te = d[d.pid != pid], d[d.pid == pid]
        best = max(tags, key=lambda t: (prf(tr.y, tr[t])[metric], -tags.index(t)))
        chosen.append(best)
        pairs_y += te.y.tolist()
        pairs_p += te[best].tolist()
    return prf(pairs_y, pairs_p), pd.Series(chosen).value_counts().to_dict()


def check_integrated(d):
    """用正式偵測 detect_ng3 重算每個片段,和評估欄位 FIXED 逐片段比對;回傳不一致的片段。"""
    sys.path.insert(0, str(ROOT / "scripts"))
    import yaml
    import eval_ng3_face as F
    from m4_track.det_cache import DetCache
    from m6_events.ng3 import detect_ng3
    cfg = yaml.safe_load((ROOT / "integrated/configs/violation_rules.yaml").read_text(encoding="utf-8"))
    n3 = {k: v for k, v in cfg["ng3"].items() if k != "enable"}
    C = F.E.load_clips()
    col = d[FIXED]
    if n3.get("gap_fill_s", 0) or n3.get("fusion", "any") != "any":         # §8.97 起:比對 r5 對應設定
        r6 = pd.read_csv(ROOT / "results/m6/ng/ng3_r6/per_clip.csv", encoding="utf-8-sig").set_index("key")   # §8.99
        col = r6.loc[d.key, tag4(dict(r=n3["r"], min_s=n3["min_s"], gap_fill_s=float(n3["gap_fill_s"]),
                                      fusion=n3["fusion"]))].values
    bad, n_ev = [], 0
    for key, want in zip(d.key, col):
        c = C[key]
        vs = [v for v in n3["views"] if v in c["caches"] and F.pose_path(c, v).exists()]
        ev = detect_ng3({v: DetCache(c["caches"][v]) for v in vs}, {v: dict(np.load(F.pose_path(c, v))) for v in vs},
                        cfg["step_s"], n3)
        n_ev += len(ev)
        if int(bool(ev)) != int(want):
            bad.append((key, int(want), len(ev)))
    print(f"正式偵測核對:{len(d)} 片段,不一致 {len(bad)};事件 {n_ev} 筆", bad[:10])
    return bad


GRID4 = [dict(min_s=m, gap_fill_s=g, fusion=f) for f in ("any", "veto", "agree") for m in (0.3, 0.2) for g in (0.0, 0.2)]
GRID6 = [dict(r=r, min_s=m, gap_fill_s=0.2, fusion="any") for m in (0.3, 0.2) for r in (0.5, 0.6, 0.75, 1.0)]
GRID5 = [dict(min_s=0.3, gap_fill_s=g, fusion=f) for f in ("any", "veto2", "agree2") for g in (0.0, 0.2)]


def tag4(g):
    return (f"r{g['r']}_" if "r" in g else "") + f"m{g['min_s']}_g{g['gap_fill_s']}_{g['fusion']}"


def run_r4(d, grid=None, out_name="ng3_r4", metric="F1"):
    """§8.94 / §8.96:每個片段算一次各鏡頭距離比值與手框數,再套各組設定;參與者留一選格點。"""
    grid = grid or GRID4
    sys.path.insert(0, str(ROOT / "scripts"))
    import yaml
    import eval_ng3_face as F
    from m4_track.det_cache import DetCache
    from m6_events.ng3 import DEFAULT, face_ratio_nh, fill_gaps, fuse, touch_segments
    from m6_events.violation import steps
    cfg = yaml.safe_load((ROOT / "integrated/configs/violation_rules.yaml").read_text(encoding="utf-8"))
    base = dict(DEFAULT, **{k: v for k, v in cfg["ng3"].items() if k != "enable"})
    step = cfg["step_s"]
    C = F.E.load_clips()
    tags = [tag4(g) for g in grid]
    rows = []
    for _, r in d.iterrows():
        c = C[r.key]
        vs = [v for v in base["views"] if v in c["caches"] and F.pose_path(c, v).exists()]
        row = dict(key=r["key"], pid=r["pid"], y=int(r["y"]), ng=r["ng"], cat=r["cat"])
        if vs:
            dcs = {v: DetCache(c["caches"][v]) for v in vs}
            T = min(len(steps(x)[0]) for x in dcs.values())
            rn = {v: face_ratio_nh(dcs[v], dict(np.load(F.pose_path(c, v))), T, base) for v in vs}
            ratios, nhs = {v: rn[v][0] for v in vs}, {v: rn[v][1] for v in vs}
        for g, t in zip(grid, tags):
            if not vs:
                row[t] = 0
                continue
            pp = dict(base, **g)
            raw = fill_gaps(fuse(ratios, pp, nhs), int(round(pp["gap_fill_s"] / step)))
            row[t] = int(bool(touch_segments(raw, step, pp["min_s"], pp["merge_s"])))
        rows.append(row)
    x = pd.DataFrame(rows)
    out = ROOT / "results/m6/ng" / out_name
    out.mkdir(parents=True, exist_ok=True)
    x.to_csv(out / "per_clip.csv", index=False, encoding="utf-8-sig")
    if out_name != "ng3_r6":
        assert (x[tags[0]] == d[FIXED].values).all(), "第一組(現行)應與評估一致"
    grid = pd.DataFrame([dict(setting=t, **prf(x.y, x[t])) for t in tags])
    cv, ch = lopo(x, tags, metric)
    folds_y, folds_p = x.y, None
    summ = pd.DataFrame([dict(item="現行 " + tags[0], **prf(x.y, x[tags[0]])), dict(item="LOPO", chosen=str(ch), **cv)])
    grid.to_csv(out / "grid.csv", index=False, encoding="utf-8-sig")
    summ.to_csv(out / "summary.csv", index=False, encoding="utf-8-sig")
    print(grid.to_string())
    print(summ.to_string())
    k = x[x.key == "YH2002_2023_12_11_16_11_15_ng005"]
    print("ng005:", {t: int(k[t].iloc[0]) for t in tags} if len(k) else "不在資料中")
    # 變動片段(相對現行),供回查
    for t in tags[1:]:
        ch_ = x[x[t] != x[tags[0]]]
        if len(ch_):
            print(t, "新增抓到", ch_[(ch_[t] == 1)].apply(lambda r: f"{r.key}({'真有' if r.y else '真無'})", axis=1).tolist()[:8],
                  "少抓", ch_[(ch_[t] == 0)].apply(lambda r: f"{r.key}({'真有' if r.y else '真無'})", axis=1).tolist()[:8])


def main():
    d = pd.read_csv(SRC, encoding="utf-8-sig")
    d = d[~d.note.fillna("").str.contains("擦鼻子")].copy()       # §8.85 排除擦鼻子(衛生紙擦鼻子不算碰臉)
    rv = pd.read_excel(REVIEW, sheet_name="人工檢視表")
    res = dict(zip(rv["片段"], rv[RES_COL].astype(int)))
    tags = [c for c in d.columns if c.startswith("r0.") and not c.startswith("n_")]
    d["y_old"] = d.y.astype(int)
    d["review"] = d.key.map(res)
    d.loc[d.review == 1, "y"] = 1
    d["user"] = ""
    if USER.exists():
        for _, u in pd.read_csv(USER, encoding="utf-8-sig", dtype=str).fillna("").iterrows():
            m = d.key == u.key
            if u.exclude:
                d.loc[m, "review"] = 3                     # 視同排除
            if u.gt_ng3:
                d.loc[m, "y"] = int(u.gt_ng3)
            d.loc[m, "user"] = u.note
    new = d[d.review != 3].copy()
    new["y"] = new.y.astype(int)
    OUT.mkdir(parents=True, exist_ok=True)
    new.to_csv(OUT / "per_clip.csv", index=False, encoding="utf-8-sig")
    old = d.copy()
    old["y"] = old.y_old
    rows = []
    for name, df in (("原真值(§8.85)", old), ("修正真值", new)):
        rows.append(dict(item=f"{name} 固定 {FIXED}", **prf(df.y, df[FIXED])))
        cv, ch = lopo(df, tags)
        rows.append(dict(item=f"{name} LOPO", chosen=str(ch), **cv))
    s = pd.DataFrame(rows)
    s.to_csv(OUT / "summary.csv", index=False, encoding="utf-8-sig")
    print(f"檢視結果:有碰臉 {sum(v == 1 for v in res.values())} / 沒碰臉 {sum(v == 2 for v in res.values())} / "
          f"看不出來 {sum(v == 3 for v in res.values())}")
    print(s.to_string())
    if "--integrated" in sys.argv:
        check_integrated(new)
    if "--r4" in sys.argv:
        run_r4(new)
    if "--r5" in sys.argv:
        run_r4(new, GRID5, "ng3_r5")
    if "--r6" in sys.argv:
        run_r4(new, GRID6, "ng3_r6", "F2")


if __name__ == "__main__":
    main()
