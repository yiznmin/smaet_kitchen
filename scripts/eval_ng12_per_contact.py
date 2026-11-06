"""§8.86:NG1 / NG2 事件層評估改用正式偵測的逐次接觸結果(m6_events.ng.detect_ng 邏輯)。

舊評估(eval_ng_details.py,§8.20–8.21)只看事件範圍內「最後一次」接觸垃圾桶,
前面幾次接觸後沒洗手(NG1)或髒手碰觸(NG2)都沒算到,和正式偵測、報告影片不一致。
新判定:事件範圍(±1 秒)內每一次接觸各自判 NG1 / NG2(同 build_ng_report.ng12_status,已和 detect_ng 核對)。
  - NG1:任一次接觸後 5 秒內洗手累計 < wash_s。
  - NG2:任一次接觸後髒手碰觸,且碰觸開始 ≤ 事件結束 + 1 秒(同 §8.21 主分析容忍)。
真值沿用 results/m6/ng/details_corrected/events.csv(已含團隊回覆修正)。
輸出 results/m6/ng/ng12_per_contact/{per_event.csv, summary.csv}(含團隊標註,不進 git)。

用法:
    python scripts/eval_ng12_per_contact.py
"""
import csv
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import build_ng_report as B                                               # noqa: E402
import eval_ng_clips as E                                                 # noqa: E402
import eval_ng_trash_events as M                                          # noqa: E402

OUT = ROOT / "results/m6/ng/ng12_per_contact"
TOL_S = 1.0


def prf(y, p):
    tp = sum(a and b for a, b in zip(y, p))
    fp = sum(b and not a for a, b in zip(y, p))
    fn = sum(a and not b for a, b in zip(y, p))
    tn = len(y) - tp - fp - fn
    R, P = tp / max(tp + fn, 1), tp / max(tp + fp, 1)
    return dict(n=len(y), TP=tp, FP=fp, FN=fn, TN=tn, R=round(R, 3), P=round(P, 3),
                F1=round(2 * R * P / max(R + P, 1e-9), 3), R_ci="-".join(map(str, E.wilson(tp, tp + fn))),
                P_ci="-".join(map(str, E.wilson(tp, tp + fp))))


def main():
    cfg = yaml.safe_load((E.INTEG / "configs/violation_rules.yaml").read_text(encoding="utf-8"))
    step = cfg["step_s"]
    res = {r["event"]: r for r in csv.DictReader(open(ROOT / "results/m6/ng/details_corrected/events.csv",
                                                       encoding="utf-8-sig"))}
    rows = []
    for e in M.load_events():
        k = e["key"].split("_")[-1]
        if k not in res:
            continue                                                  # 團隊建議略過
        x = res[k]
        r = E.rule_part(dict(key=e["key"], caches={v: M.CACHE / e["key"] / f"{v}.npz" for v in M.VIEWS}), cfg)
        st = B.ng12_status(r, cfg, r["T"])
        w0 = e["f0"] / B.FPS
        s_ev, e_ev = int((e["start"] - w0) / step), int((e["end"] - w0) / step)
        n1, n2, eps = B.window_preds(st, s_ev, e_ev, w0, e["end"], step, TOL_S)
        rows.append(dict(event=k, key=e["key"], note=e["note"], team_ng=e["ng"], n_contacts=len(eps),
                         gt_ng1=x["gt_ng1"], old_ng1=int(x["pred_ng1"]), new_ng1=int(bool(n1)),
                         gt_ng2=x["gt_ng2"], old_ng2=int(x["pred_ng2"]), new_ng2=int(bool(n2)),
                         ng1_at="、".join(f"{w0 + ev['at'] * step:.1f}" for ev in n1),
                         ng2_at="、".join(f"{w0 + ev['at'] * step:.1f}" for ev in n2)))
        print(f"  {k} 接觸 {len(eps)} NG1 {x['gt_ng1']}/{x['pred_ng1']}→{int(bool(n1))} "
              f"NG2 {x['gt_ng2']}/{x['pred_ng2']}→{int(bool(n2))}", flush=True)
    d = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    d.to_csv(OUT / "per_event.csv", index=False, encoding="utf-8-sig")
    summ = []
    for ng in ("ng1", "ng2"):
        s = d[d[f"gt_{ng}"].astype(str) != ""]
        y = s[f"gt_{ng}"].astype(int).astype(bool).tolist()
        for ver in ("old", "new"):
            summ.append(dict(ng=ng.upper(), version=ver, **prf(y, s[f"{ver}_{ng}"].astype(bool).tolist())))
        ch = s[s[f"old_{ng}"] != s[f"new_{ng}"]]
        summ[-1]["changed"] = "、".join(f"{a}({'真值有' if int(g) else '真值無'})" for a, g in zip(ch.event, ch[f"gt_{ng}"]))
    pd.DataFrame(summ).to_csv(OUT / "summary.csv", index=False, encoding="utf-8-sig")
    print(pd.DataFrame(summ).to_string())


if __name__ == "__main__":
    main()
