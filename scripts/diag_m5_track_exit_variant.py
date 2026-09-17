"""事前估計:候選改成「每台鏡頭上最近結束的 track」+「沒有外觀時不扣分」(只讀,不改系統)。

## 背景(2026-09-17,使用者提議「時間 + 指定鏡頭」)

L3 的兩段影片(seq_000 9 號 c6+c7、seq_004 7 號 c6→c7)同一人沒綁成同一 ID。查到三個原因:
  1. 轉場候選以**身份**為單位:身份在**所有鏡頭**都消失才算離場。被多人共用的 chef
     (例:seq_004 的 chef 48)永遠不會離場 → 就算它在 camera_6 剛結束也進不了候選。
  2. `embedder none` 讓 cosine 恆為 0,`app_lr.llr(0) = −1.1937`,每個候選都被扣分;
     c6→c7 轉場最高 2.678 − 1.194 = 1.484 < 門檻 1.609(見 docs/M5_綁定規則實測_20260915.md)。
  3. c6、c7 在拓撲只是轉場連結,不是重疊。

## 三個變體(都與實際執行比較)

  E1 只改外觀:所有候選分數 +1.1937(沒有外觀 = 不加不扣)
  E2 只改候選:對每條**已結束**的 track(跟丟中或已移除,100 秒內)以它的 chef 當候選,
     分數 = 該鏡頭對的轉場證據(同鏡頭:SameCameraTransit + PositionLR;
     跨鏡頭:連結 / 未知路徑 + transit_place)+ 外觀常數(−1.1937)
  E3 兩者都改

每次綁定決策比較「實際結果」與「變體結果」:對(綁回該真人上一次的 chef)/ 開新 / 綁錯。

## ⚠ 限制(這是估計,不是實驗)

  - 實際候選只存了最高分那一個(top1),其他候選的分數無法重算 → 變體的「最佳」
    = max(實際 top1(+調整), 新候選)。若實際第二名在調整後會超過,這裡看不到。
  - 不模擬連鎖:改判之後後續決策的候選與 want 都會變,這裡照實際執行的歷史算。
  - 同迴圈內先處理 lost / reacquired / removed 再處理 new_track;離開時間 = 最後一個框。
  - F2 同鏡頭互斥、margin 照實際設定(皆關閉)。

用法:
    python scripts/diag_m5_track_exit_variant.py --run-root results/m5_step3/m5_cbiou \\
        --seqs seq_004 seq_006 seq_007 seq_020 seq_024 seq_025 seq_026 \\
        --topology configs/fix_grid/base.yaml --out results/m5_pending_exit/variant_m5_cbiou.json
"""
import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from m5_reid.spatiotemporal import CameraTopology       # noqa: E402

HORIZON_S = 100.0          # = ttl 600 迴圈 × stride 5 / 30 fps(recently_disappeared_ttl)
VARIANTS = ("E1", "E2", "E3")


def exit_score(topo, cam_from, t_exit, box_exit, cam_to, t_new, box_new, app_llr):
    ok, llr_t = topo.transit_llr(cam_from, t_exit, cam_to, t_new)
    if not ok:
        return None
    s = llr_t + app_llr
    if cam_from == cam_to and topo.pos_lr is not None:
        s += topo.pos_lr.llr(box_exit, box_new, t_new - t_exit)
    s += topo.direction_llr(cam_from, cam_to, None, None)
    tp = topo.transit_place(cam_from, cam_to)
    if tp is not None:
        s += tp.llr(box_exit, box_new)
    return s


def outcome(choice, want):
    if choice is None:
        return "開新"
    return "對" if choice == want else "錯"


def analyse_seq(run_dir, topo, fps, thr):
    run_dir = Path(run_dir)
    meta = json.loads((run_dir / "run_meta.json").read_text(encoding="utf-8"))
    if meta.get("embedder") != "none":
        raise SystemExit(f"{run_dir}:本估計假設 embedder none")
    app0 = topo.app_lr.llr(0.0)                 # −1.1937

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
            order = 1 if r["kind"] == "new_track" else 0
            events.append((int(r["video_fid"]), order, i, r["camera_id"], r["kind"],
                           int(r["track_id"])))
    events.sort()

    ended = {}                      # (cam, tid) -> (t_exit, box)
    rows = []
    for fid, _o, _i, cam, kind, tid in events:
        key = (cam, tid)
        t_now = fid / fps
        if kind == "lost_track":
            prev = [x for x in boxes.get(key, {}) if x < fid]
            last = max(prev) if prev else None
            ended[key] = ((last if last is not None else fid) / fps,
                          boxes[key][last] if prev else None)
        elif kind == "reacquired":
            ended.pop(key, None)
        elif kind == "removed":
            pass                    # 已在 lost 時記錄;移除後仍保留到 HORIZON_S
        elif kind == "new_track":
            for k in [k for k, (te, _b) in ended.items() if t_now - te > HORIZON_S]:
                del ended[k]
            d = dec_at.get((cam, tid, fid))
            if d is None:
                continue
            box_new = tuple(d["bbox"]) if d.get("bbox") else None
            per_chef = {}
            for (pc, ptid), (te, pb) in ended.items():
                if (pc, ptid) == key:
                    continue
                cid = chef_of.get((pc, ptid))
                if cid is None:
                    continue
                s = exit_score(topo, pc, te, pb, cam, t_now, box_new, app0)
                if s is not None and s > per_chef.get(cid, float("-inf")):
                    per_chef[cid] = s
            top1_id, top1 = d.get("top1_chef_id"), d.get("top1_score")
            want = d.get("want_chef_id")
            actual = d["chef_id"] if d["matched"] else None
            row = dict(fid=fid, cam=cam, tid=tid, layer=d.get("layer"), want=want,
                       actual=actual, actual_outcome=outcome(actual, want))
            for v in VARIANTS:
                adj = -app0 if v in ("E1", "E3") else 0.0
                pool = {}
                if top1_id is not None and top1 is not None:
                    pool[top1_id] = top1 + adj
                if v in ("E2", "E3"):
                    for cid, s in per_chef.items():
                        pool[cid] = max(pool.get(cid, float("-inf")), s + adj)
                best = max(pool.items(), key=lambda x: x[1], default=(None, float("-inf")))
                choice = best[0] if best[1] >= thr else None
                # 實際有綁、而變體沒有更好的選擇時,維持實際(top1 就是實際綁的那個)
                row[f"{v}_choice"] = choice
                row[f"{v}_outcome"] = outcome(choice, want)
                row[f"{v}_want_score"] = (None if want is None or want not in pool
                                          else round(pool[want], 3))
            rows.append(row)
    return rows


def summarise(rows):
    out = {}
    attributable = [r for r in rows if r["layer"] in
                    ("OK", "L2_not_in_candidates", "L3_not_top1", "L4_below_threshold")]
    first = [r for r in rows if r["layer"] == "L0_first"]
    ghost = [r for r in rows if r["layer"] == "L1_ghost"]
    base = Counter(r["actual_outcome"] for r in attributable)
    out["actual"] = dict(base)
    for v in VARIANTS:
        c = Counter(r[f"{v}_outcome"] for r in attributable)
        trans = Counter(f"{r['actual_outcome']}→{r[f'{v}_outcome']}" for r in attributable
                        if r["actual_outcome"] != r[f"{v}_outcome"])
        cost = lambda cc: 5 * cc.get("錯", 0) + cc.get("開新", 0)
        out[v] = dict(
            outcomes=dict(c), transitions=dict(trans),
            delta=dict(對=c.get("對", 0) - base.get("對", 0),
                       開新=c.get("開新", 0) - base.get("開新", 0),
                       錯=c.get("錯", 0) - base.get("錯", 0)),
            weighted_cost=dict(actual=cost(base), variant=cost(c)),
            first_appearance_bound=dict(actual=sum(r["actual"] is not None for r in first),
                                        variant=sum(r[f"{v}_choice"] is not None for r in first)),
            ghost_bound=dict(actual=sum(r["actual"] is not None for r in ghost),
                             variant=sum(r[f"{v}_choice"] is not None for r in ghost)))
    out["n"] = dict(decisions=len(rows), attributable=len(attributable),
                    first=len(first), ghost=len(ghost))
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
    all_rows = []
    for seq in args.seqs:
        all_rows += [dict(seq=seq, **r) for r in analyse_seq(Path(args.run_root) / seq,
                                                              topo, args.fps, thr)]
    total = summarise(all_rows)
    p = Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(dict(run_root=args.run_root, seqs=args.seqs, threshold=thr,
                                 horizon_s=HORIZON_S, total=total),
                            ensure_ascii=False, indent=2), encoding="utf-8")
    with open(p.with_suffix(".rows.csv"), "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0]))
        w.writeheader()
        w.writerows(all_rows)
    print(json.dumps(total, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
