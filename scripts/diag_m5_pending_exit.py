"""量化「舊 track 還在 M4 跟丟緩衝期 → 正確身份不在候選名單」這個漏洞(只讀,不改系統)。

## 背景(2026-09-17,L2_02 查到)

M5 的候選名單只有兩個來源(`identity_st._score_candidates`):
  (a) gone:舊 track 已被 M4 **移除**(跟丟緩衝 5 秒到期)
  (b) active 且正被**另一台重疊鏡頭**看著
同一台鏡頭上 M4 斷軌、馬上開新 track 時,舊 track 還在緩衝期(pending exit),
它的 chef 既不是 gone、也不在別台重疊鏡頭 → 正確答案**結構上不可能**進候選名單。
現有的同鏡頭重關聯證據(SameCameraTransit + PositionLR)只接在 (a),5 秒後才用得到。

## 這支做什麼

逐序列重放 M4 事件(`track_events.csv`:new / lost / reacquired / removed),
在每一次綁定決策當下重建「同一台鏡頭上正在緩衝期的 track」,然後:

  1. **漏洞規模**:正確身份(= 該真人上一次拿到的 chef)有一條緩衝期 track 在本鏡頭上的次數
  2. **能救回幾次**:用系統現有的證據函式試算正確身份若被放進候選名單的分數
     (同鏡頭時間 + 外觀常數 + 位置),是否 ≥ 門檻且 ≥ 當次實際最高分
  3. **會多錯幾次**:其他 chef 的緩衝期 track 若放進候選,會不會搶走決策
     (原本正確或原本開新身份的決策被搶走 = 新增誤併)

⚠ 這是**事前估計**,不是實驗結果:
  - 只把新候選加進「當次實際候選」比較,不模擬改判之後的連鎖(後續決策的候選名單會不同)
  - 同一迴圈內先處理 lost / reacquired / removed 再處理 new_track(修法需要的順序)
  - 離開時間 = 跟丟前最後一個框的時間
  - 分數公式照抄 `_score_candidates` 的 gone 路徑(direction 關閉、transit_place 對同鏡頭為 None)

用法:
    python scripts/diag_m5_pending_exit.py --run-root results/m5_step3/m5_cbiou \\
        --seqs seq_004 seq_006 seq_007 seq_020 seq_024 seq_025 seq_026 \\
        --topology configs/fix_grid/base.yaml --out results/m5_pending_exit/estimate_m5_cbiou.json
"""
import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from m5_reid.identity import cosine                     # noqa: E402
from m5_reid.embedder import l2norm                     # noqa: E402
from m5_reid.spatiotemporal import CameraTopology       # noqa: E402


def pending_score(topo, cam, t_exit, exit_box, t_new, new_box, app_cos):
    """照抄 identity_st._score_candidates 的 gone 路徑(同鏡頭)。"""
    ok, llr_t = topo.transit_llr(cam, t_exit, cam, t_new)
    if not ok:
        return None
    s = llr_t + topo.app_lr.llr(app_cos)
    if topo.pos_lr is not None:
        s += topo.pos_lr.llr(exit_box, new_box, t_new - t_exit)
    s += topo.direction_llr(cam, cam, None, None)
    tp = topo.transit_place(cam, cam)
    if tp is not None:
        s += tp.llr(exit_box, new_box)
    return s


def analyse_seq(run_dir, topo, fps, thr):
    run_dir = Path(run_dir)
    meta = json.loads((run_dir / "run_meta.json").read_text(encoding="utf-8"))
    embedder = meta.get("embedder")
    if embedder != "none":
        raise SystemExit(f"{run_dir}:embedder={embedder},本估計假設外觀向量全零")
    app_cos = cosine(l2norm(np.zeros(64)), l2norm(np.zeros(64)))

    # 每條 track 逐幀的框(找跟丟前最後一個框)
    boxes = defaultdict(dict)
    with open(run_dir / "tracks.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["x1"] != "":
                boxes[(r["camera_id"], int(r["track_id"]))][int(r["video_fid"])] = tuple(
                    float(r[k]) for k in ("x1", "y1", "x2", "y2"))

    decisions = [json.loads(l) for l in
                 (run_dir / "chef_events.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    chef_of = {(d["camera_id"], d["track_id"]): d["chef_id"] for d in decisions}
    dec_at = {(d["camera_id"], d["track_id"], d["video_fid"]): d for d in decisions}

    events = []
    with open(run_dir / "track_events.csv", encoding="utf-8") as f:
        for i, r in enumerate(csv.DictReader(f)):
            # ⚠ 同一迴圈內 M4 先送 new_track 才送 lost_track(tracker.py 的事件順序)。
            #   修法必須先處理 lost 才看得到「剛斷掉的舊 track」→ 估計照修法的順序重放。
            order = 1 if r["kind"] == "new_track" else 0
            events.append((int(r["video_fid"]), order, i, r["camera_id"], r["kind"],
                           int(r["track_id"])))
    events.sort()

    pending = {}                    # (cam, tid) -> (t_exit, exit_box)
    rows = []
    for fid, _o, _i, cam, kind, tid in events:
        key = (cam, tid)
        if kind == "lost_track":
            # ⚠ 離開時間用「最後一個框」的時間,不是發現跟丟的那一刻 ——
            #   發現跟丟與新 track 出現常在同一迴圈,Δt = 0 會被 transit_llr 直接拒絕。
            prev = [f for f in boxes.get(key, {}) if f < fid]
            last = max(prev) if prev else None
            box = boxes[key][last] if prev else None
            pending[key] = ((last if last is not None else fid) / fps, box)
        elif kind in ("reacquired", "removed"):
            pending.pop(key, None)
        elif kind == "new_track":
            d = dec_at.get((cam, tid, fid))
            if d is None:
                continue
            new_box = tuple(d["bbox"]) if d.get("bbox") else None
            t_new = fid / fps
            # 本鏡頭上正在緩衝期、且已綁定 chef 的 track
            cands = []
            for (pc, ptid), (t_exit, ebox) in pending.items():
                if pc != cam or (pc, ptid) == key:
                    continue
                cid = chef_of.get((pc, ptid))
                if cid is None:
                    continue
                s = pending_score(topo, cam, t_exit, ebox, t_new, new_box, app_cos)
                if s is not None:
                    cands.append((s, cid, round(t_new - t_exit, 3)))
            actual_top = d.get("top1_score")
            actual_top = float("-inf") if actual_top is None else actual_top
            want = d.get("want_chef_id")
            layer = d.get("layer")
            # 正確身份是否有緩衝期 track 在本鏡頭
            want_c = max((c for c in cands if c[1] == want), default=None)
            best_new = max(cands, default=None)
            # 若把緩衝期候選加進去,決策會變成誰(不含連鎖)
            new_choice = None
            if best_new and best_new[0] >= thr and best_new[0] > actual_top:
                new_choice = best_new[1]
            rows.append(dict(fid=fid, cam=cam, tid=tid, layer=layer, gt_id=d.get("gt_id"),
                             want=want, chef=d["chef_id"], matched=d["matched"],
                             actual_top=None if actual_top == float("-inf") else actual_top,
                             n_pending=len(cands),
                             want_pending=want_c is not None,
                             want_score=None if want_c is None else round(want_c[0], 3),
                             want_dt=None if want_c is None else want_c[2],
                             new_choice=new_choice))
    return rows


def summarise(rows, thr):
    s = Counter()
    dts, scores = [], []
    for r in rows:
        s["decisions"] += 1
        if r["layer"] not in ("L2_not_in_candidates", "L3_not_top1", "L4_below_threshold", "OK"):
            # L0_first / L1_ghost:沒有「正確答案」,但仍可能被新候選搶走 → 算風險
            if r["new_choice"] is not None:
                s["risk_ghost_or_first_now_bound"] += 1
            continue
        s["attributable"] += 1
        if r["want_pending"]:
            s["want_pending"] += 1
            s[f"want_pending|{r['layer']}"] += 1
            dts.append(r["want_dt"])
            scores.append(r["want_score"])
            if r["want_score"] >= thr:
                s["want_pending_score_ge_thr"] += 1
        if r["new_choice"] is None:
            continue
        if r["new_choice"] == r["want"]:
            if r["layer"] == "OK":
                s["ok_unchanged"] += 1
            else:
                s["rescued"] += 1                        # 原本錯 → 綁回正確身份
                s[f"rescued|{r['layer']}"] += 1
        else:
            if r["layer"] == "OK":
                s["harm_ok_to_wrong"] += 1               # 原本對 → 被別人的緩衝期身份搶走
            elif not r["matched"]:
                s["harm_break_to_merge"] += 1            # 原本開新身份 → 改綁錯人
            else:
                s["wrong_to_wrong"] += 1
    out = dict(counts=dict(s))
    if dts:
        out["want_dt_s"] = dict(p50=float(np.median(dts)), p90=float(np.percentile(dts, 90)),
                                max=float(max(dts)))
        out["want_score"] = dict(p10=float(np.percentile(scores, 10)),
                                 p50=float(np.median(scores)), min=float(min(scores)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root", required=True)
    ap.add_argument("--seqs", nargs="+", required=True)
    ap.add_argument("--topology", required=True)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    topo = CameraTopology.from_yaml(args.topology)
    thr = topo.llr_threshold
    all_rows, per_seq = [], {}
    for seq in args.seqs:
        rows = analyse_seq(Path(args.run_root) / seq, topo, args.fps, thr)
        per_seq[seq] = summarise(rows, thr)
        all_rows += [dict(seq=seq, **r) for r in rows]
    total = summarise(all_rows, thr)
    out = dict(run_root=args.run_root, seqs=args.seqs, topology=args.topology,
               threshold=thr, total=total, per_seq=per_seq)
    p = Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    with open(p.with_suffix(".rows.csv"), "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0]))
        w.writeheader()
        w.writerows(all_rows)
    print(json.dumps(total, ensure_ascii=False, indent=2))
    print(f"門檻 {thr:.4f};已存 {p}")


if __name__ == "__main__":
    raise SystemExit(main())
