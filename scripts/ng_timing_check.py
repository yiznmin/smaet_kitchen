"""垃圾桶事件:由團隊 NG 片段反推接觸 / 碰觸時間,與系統偵測的時間點比對(診斷)。

預先登記:docs/M6_NG準則評估_預先登記_20261027.md §8.10 A
- 團隊接觸污染源 = [NG1 片段起點, NG1 片段終點 − 5 秒];NG2 片段終點 = 髒手後第一次碰觸結束
- 系統:base 觸發區間(§8.8 視窗)、FSM 判違規的碰觸步
輸出 results/m6/ng/trash_ev/timing.csv(含團隊標註,不進 git)

用法:
    python scripts/ng_timing_check.py
"""
import csv
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import eval_ng_clips as E                                                 # noqa: E402
import eval_ng_trash_events as M                                          # noqa: E402
from m6_events.violation import HandHygieneFSM                            # noqa: E402

NG1_TAIL_S = 5.0


def team_times():
    """annotation_id -> {ng1: (起, 訖), ng2_end: 秒}(取 Aoutput1 那列,各視角時間相同)。"""
    out = {}
    for r in csv.DictReader(open(M.E.NG_DIR / "ng_clips_index.csv", encoding="utf-8-sig")):
        if r["category"] != "垃圾桶" or r["view"] != "Aoutput1":
            continue
        d = out.setdefault(r["annotation_id"], {})
        a, b = float(r["clip_start_sec"]), float(r["clip_end_sec"])
        if r["ng_type"] == "NG1":
            d["ng1"] = (a, b - NG1_TAIL_S)
        elif r["ng_type"] == "NG2":
            d["ng2_start"], d["ng2_end"] = a, b
    return out


def main():
    cfg = yaml.safe_load((E.INTEG / "configs/violation_rules.yaml").read_text(encoding="utf-8"))
    step, hh = cfg["step_s"], cfg["hand_hygiene"]
    team = team_times()
    ann = {r["annotation_id"][:8]: r["annotation_id"]
           for r in csv.DictReader(open(M.EV_NG, encoding="utf-8-sig"))}
    rows = []
    for e in M.load_events():
        t = team.get(ann[e["key"][-8:]])
        if not t or "ng1" not in t:
            continue
        c = dict(key=e["key"], caches={v: M.CACHE / e["key"] / f"{v}.npz" for v in M.VIEWS})
        r = E.rule_part(c, cfg)
        w0 = e["f0"] / M.FPS
        s_ev, e_ev = int((e["start"] - w0) / step), int((e["end"] - w0) / step)
        tr = [(s, x) for s, x in (r["trig_base"] or []) if s <= e_ev + 10 and x >= s_ev - 10]
        row = dict(key=e["key"], ng=e["ng"], note=e["note"],
                   team_contact_start=round(t["ng1"][0] - w0, 1), team_contact_end=round(t["ng1"][1] - w0, 1),
                   team_touch_end=round(t["ng2_end"] - w0, 1) if "ng2_end" in t else "",
                   ng2_start_eq_ng1=(abs(t["ng2_start"] - t["ng1"][0]) < 0.05) if "ng2_start" in t else "")
        if tr:
            row.update(sys_trig_start=round(tr[0][0] * step, 1), sys_trig_end=round(tr[-1][1] * step, 1))
            row["d_start"] = round(row["sys_trig_start"] - row["team_contact_start"], 1)
            row["d_end"] = round(row["sys_trig_end"] - row["team_contact_end"], 1)
            v = HandHygieneFSM(hh["wash_s"], step, cfg["touch_classes"]).run(
                tr[-1][1] + int(round(hh["grace_s"] / step)), r["wash"], r["touch"])
            if v.result == "違規":
                row.update(sys_touch_at=round(v.at * step, 1), sys_touch_cls="、".join(v.touched))
                if row["team_touch_end"] != "":
                    row["d_touch"] = round(row["sys_touch_at"] - row["team_touch_end"], 1)
            row["sys_verdict"] = v.result
        rows.append(row)
    M.OUT.mkdir(parents=True, exist_ok=True)
    E.OUT = M.OUT
    E.write_csv("timing", rows)

    def desc(name):
        x = np.array([r[name] for r in rows if r.get(name, "") != ""], float)
        if not len(x):
            return f"{name}: 無"
        q = np.percentile(x, [25, 50, 75])
        return (f"{name}: n {len(x)}  中位數 {q[1]:+.1f}  四分位 {q[0]:+.1f} ~ {q[2]:+.1f}  "
                f"|差|≤1 秒 {np.mean(np.abs(x) <= 1):.2f}")
    print(f"有團隊 NG1 片段的事件 {len(rows)},系統有觸發 {sum('d_start' in r for r in rows)}")
    print("NG2 起點 = NG1 起點:", sum(r["ng2_start_eq_ng1"] is True for r in rows), "/",
          sum(r["ng2_start_eq_ng1"] != "" for r in rows))
    for n in ("d_start", "d_end", "d_touch"):
        print(desc(n))
    print("NG1-only 事件系統判違規的碰觸:")
    for r in rows:
        if r["ng"] == "NG1" and r.get("sys_verdict") == "違規":
            print(f"  {r['key'][-8:]} {r['note']}  接觸結束 團隊 {r['team_contact_end']} / 系統 {r['sys_trig_end']}  "
                  f"碰觸 {r['sys_touch_at']} 秒 {r['sys_touch_cls']}")


if __name__ == "__main__":
    main()
