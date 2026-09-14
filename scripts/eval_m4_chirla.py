"""M4 專用評估:從偵測快取直接驅動 tracker —— 不解碼影片、不跑 M5。

## 為什麼

2026-09-13 的根因分解指出六個 M5 修法全部落在只佔 11.7% 的層 3/4;規矩改為由下而上。
2026-09-14 在本機把層 0(同一真人、同一鏡頭的相鄰兩條 track)拆成:

    A 前一條還在追,又多開一條        13.0%
    B 前一條還在 lost 緩衝內卻另開新 ID  31.0%   ← M4 關聯失敗
    C 緩衝過期後才出現               56.0%   ← 多半是人真的走出畫面

並查到兩個可能造成 B 的結構問題:BYTE 第二輪低分關聯從未作用(偵測先被 0.3 過濾)、
tracker 以 stride 5 被餵資料。要受控檢驗,必須對**同一批偵測**換門檻與 stride 重跑追蹤,
而且每一格要夠快 —— 所以這支完全不碰 RF-DETR 與 M5。

## 量什麼(逐鏡頭 + 合計)

- **真值框召回**,兩種分母並列:
  · `track 幀`:沿用 `eval_m4m5_chirla.match_tracks` 的定義(只數有 track 的幀),
    用來重現既有的 88.0%
  · `全部取樣幀`:所有被取樣的幀上的 GT 都進分母 —— 較誠實,會比前者低
- **誤偵率**:沒有對到任何真值的 track / 全部 track(層 1)
- **A / B / C** 三類次數與間隔(定義同上;以事件時間排序,同一幀內依 tracker 發事件的順序)
- **每位真人每台鏡頭的 track 數**
- **身份混雜的 track**:對到 ≥2 個真值、且次多者 ≥3 幀 —— 一條 track 漂到別人身上的代理量

## ⚠ 緩衝秒數固定

supervision 的 `max_time_lost = int(frame_rate/30 × lost_track_buffer)` 是**更新次數**。
換 stride 時若不換算,stride 1 的 30 次只有 1 秒,與 stride 5 的 5 秒不可比。
所以本腳本以 `--lost-buffer-seconds` 為準,自動換算並斷言換算結果。

## 重用,不重寫

真值讀取與配對直接 import `eval_m4m5_chirla.load_gt` / `match_tracks`。

用法:
    # CHIRLA(遠端)
    python scripts/eval_m4_chirla.py --cache-dir results/det_cache/coco_nano \\
        --root "D:/.../CHIRLA" --seqs seq_004 seq_006 --thr 0.3 --stride 5 --label base

    # 本機 EPFL:沒有真值,只輸出 track 與事件供逐位比對
    python scripts/eval_m4_chirla.py --cache-dir results/det_cache/epfl_smoke \\
        --cameras cam1 cam2 --thr 0.3 --stride 5 --max-loops 25 --dump-dir <dir>
"""
import argparse
import csv
import json
import statistics as st
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from m4_track import KitchenTracker                                  # noqa: E402
from m4_track.det_cache import DetCache, weights_id                  # noqa: E402

# tracker 在同一次 update 裡發事件的順序(src/m4_track/tracker.py)
_ORDER = {"new_track": 0, "reacquired": 1, "lost_track": 2, "removed": 3}
_STATE = {"new_track": "active", "reacquired": "active",
          "lost_track": "lost", "removed": "removed"}


def tracker_config(tcfg, stride, fps, seconds):
    """回傳換算好緩衝次數的 tracker 設定。"""
    frame_rate = float(tcfg.get("frame_rate", 30))
    target = round(seconds * fps / stride)              # 要的更新次數
    lost = round(target * 30.0 / frame_rate)
    got = int(frame_rate / 30.0 * lost)
    if got != target:
        raise ValueError(f"緩衝換算失準:要 {target} 次,實際 {got} 次")
    return dict(tcfg, lost_track_buffer=lost), dict(
        updates=target, seconds=target * stride / fps, lost_track_buffer=lost)


def run_camera(cache, cam, *, thr, stride, tcfg, max_loops):
    meta = cache.meta
    fps = float(meta["video_meta"]["fps"])
    tr = KitchenTracker.from_config(tcfg, camera_id=cam)
    tracks, events = [], []
    for loop_i, fid in enumerate(range(0, int(meta["max_fid"]) + 1, stride)):
        if max_loops >= 0 and loop_i >= max_loops:
            break
        out = tr.update(cache.get(fid, thr), loop_i, timestamp=fid / fps)
        for t in out.tracks:
            tracks.append(dict(loop=loop_i, fid=fid, cam=cam, tid=t.track_id,
                               bbox=None if t.bbox is None else tuple(t.bbox),
                               conf=t.confidence, hits=t.hits, start=t.start_frame))
        for e in out.events:
            events.append(dict(loop=loop_i, fid=fid, t=e.t_sec, cam=cam,
                               kind=e.kind, tid=e.track_id))
    return tracks, events, fps


def classify_pairs(events, track_gt):
    """同一真人、同一鏡頭的相鄰兩條 track → A / B / C 與間隔秒數。"""
    per = defaultdict(list)
    for e in events:
        per[(e["cam"], e["tid"])].append((e["t"], _ORDER[e["kind"]], e["kind"]))
    by = defaultdict(list)
    for (cam, tid), es in per.items():
        es.sort()
        if not any(k == "new_track" for _, _, k in es):
            continue
        gid = track_gt.get((cam, tid))
        if gid is None:
            continue
        by[(cam, gid)].append((next(t for t, _, k in es if k == "new_track"), es))
    out = defaultdict(list)                              # 類別 -> [(cam, 間隔秒)]
    for (cam, _gid), lst in by.items():
        lst.sort(key=lambda x: x[0])
        for (_pt, pes), (nt, _nes) in zip(lst, lst[1:]):
            state, lost_t = None, None
            for t, _o, k in pes:
                if t > nt:
                    break
                state = _STATE[k]
                if k == "lost_track":
                    lost_t = t
            if state == "active":
                out["A"].append((cam, 0.0))
            elif state == "lost":
                out["B"].append((cam, nt - lost_t))
            elif state == "removed":
                out["C"].append((cam, nt - (lost_t if lost_t is not None else nt)))
    return out


def dump(dirpath, tracks, events):
    """格式對齊 m5_track_video.py 的 tracks.csv / track_events.csv(chef_id 留空)。"""
    dirpath.mkdir(parents=True, exist_ok=True)
    with (dirpath / "tracks.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["loop_i", "video_fid", "camera_id", "track_id",
                    "x1", "y1", "x2", "y2", "conf", "hits", "start_frame"])
        for t in tracks:
            b = t["bbox"] or (None,) * 4
            w.writerow([t["loop"], t["fid"], t["cam"], t["tid"],
                        *[None if v is None else round(float(v), 1) for v in b],
                        "" if t["conf"] is None else round(t["conf"], 3),
                        t["hits"], t["start"]])
    with (dirpath / "track_events.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["loop_i", "video_fid", "t_sec", "camera_id", "kind", "track_id"])
        for e in events:
            w.writerow([e["loop"], e["fid"], round(e["t"], 3), e["cam"], e["kind"], e["tid"]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--root", default=None, help="CHIRLA 根目錄;不給就不算真值指標")
    ap.add_argument("--seqs", nargs="+", default=None,
                    help="給了就讀 <cache-dir>/<seq>/<camera>.npz;不給就讀 <cache-dir>/<camera>.npz")
    ap.add_argument("--cameras", nargs="+", default=None)
    ap.add_argument("--thr", type=float, default=0.3)
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--lost-buffer-seconds", type=float, default=5.0)
    ap.add_argument("--tracker", default=str(ROOT / "configs" / "tracker.yaml"))
    ap.add_argument("--variant", default="nano")
    ap.add_argument("--weights", default=None)
    ap.add_argument("--person-cls", type=int, default=None)
    ap.add_argument("--max-loops", type=int, default=-1)
    ap.add_argument("--dump-dir", default=None)
    ap.add_argument("--label", default="run")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    person_cls = (args.person_cls if args.person_cls is not None
                  else (0 if args.weights else 1))
    expect = dict(variant=args.variant, weights_id=weights_id(args.weights),
                  person_cls=person_cls)
    with open(args.tracker, encoding="utf-8") as f:
        base_tcfg = (yaml.safe_load(f) or {}).get("tracker", {})

    if args.seqs:
        from run_m4m5_chirla_grid import CAMERAS
        units = [(s, args.cameras or CAMERAS, Path(args.cache_dir) / s) for s in args.seqs]
    else:
        d = Path(args.cache_dir)
        cams = args.cameras or sorted(p.stem for p in d.glob("*.npz"))
        units = [("_", cams, d)]

    gt_mod = None
    if args.root:
        if not args.seqs:
            raise SystemExit("--root 需要搭配 --seqs")
        import eval_m4m5_chirla as gt_mod

    print(f"[{args.label}] thr={args.thr}  stride={args.stride}  "
          f"緩衝 {args.lost_buffer_seconds}s  單元 {len(units)}")
    per_cam = defaultdict(lambda: dict(n_tracks=0, n_ghost=0, gtm=0, gtt_trk=0,
                                       gtt_all=0, mixed=0, pairs=Counter()))
    per_gt_cam = Counter()
    all_gaps = defaultdict(list)
    buf_info, t0 = None, time.time()

    for seq, cams, cdir in units:
        tracks_all, events_all, sampled = [], [], {}
        for cam in cams:
            cache = DetCache(cdir / f"{cam}.npz", expect=expect)
            fps = float(cache.meta["video_meta"]["fps"])
            tcfg, buf_info = tracker_config(base_tcfg, args.stride, fps,
                                            args.lost_buffer_seconds)
            tr, ev, _ = run_camera(cache, cam, thr=args.thr, stride=args.stride,
                                   tcfg=tcfg, max_loops=args.max_loops)
            tracks_all += tr
            events_all += ev
            sampled[cam] = sorted({t["fid"] for t in tr} |
                                  set(range(0, int(cache.meta["max_fid"]) + 1, args.stride)))
            if args.max_loops >= 0:
                sampled[cam] = sampled[cam][:args.max_loops]
        if args.dump_dir:
            dump(Path(args.dump_dir) / seq if args.seqs else Path(args.dump_dir),
                 tracks_all, events_all)
        if gt_mod is None:
            print(f"  {seq}: {len(tracks_all):,} track-幀、{len(events_all):,} 事件(無真值)")
            continue

        gt = gt_mod.load_gt(args.root, seq)
        with_box = [t for t in tracks_all if t["bbox"] is not None]
        track_gt, _stats, gtm, gtt, votes = gt_mod.match_tracks(gt, with_box, return_votes=True)
        for (cam, tid), gid in track_gt.items():
            d = per_cam[cam]
            d["n_tracks"] += 1
            d["n_ghost"] += gid is None
            if gid is not None:
                per_gt_cam[(seq, cam, gid)] += 1
                top = votes[(cam, tid)].most_common(2)
                if len(top) >= 2 and top[1][1] >= 3:
                    d["mixed"] += 1
        for cam in cams:
            per_cam[cam]["gtm"] += gtm[cam]
            per_cam[cam]["gtt_trk"] += gtt[cam]
            per_cam[cam]["gtt_all"] += sum(len(gt.get(cam, {}).get(fid + 1, []))
                                           for fid in sampled[cam])
        for k, lst in classify_pairs(events_all, track_gt).items():
            for cam, gap in lst:
                per_cam[cam]["pairs"][k] += 1
                all_gaps[k].append(gap)

    if gt_mod is None:
        print(f"完成({time.time() - t0:.0f} 秒)。緩衝:{buf_info}")
        return 0

    def pct(a, b):
        return a / b if b else float("nan")

    print(f"\n  {'鏡頭':<10}{'track':>7}{'誤偵率':>8}{'召回(track幀)':>14}{'召回(全部幀)':>14}"
          f"{'A':>6}{'B':>6}{'C':>6}{'混雜':>6}")
    tot = dict(n_tracks=0, n_ghost=0, gtm=0, gtt_trk=0, gtt_all=0, mixed=0, pairs=Counter())
    for cam in sorted(per_cam):
        d = per_cam[cam]
        print(f"  {cam:<10}{d['n_tracks']:>7}{pct(d['n_ghost'], d['n_tracks']):>8.1%}"
              f"{pct(d['gtm'], d['gtt_trk']):>14.1%}{pct(d['gtm'], d['gtt_all']):>14.1%}"
              f"{d['pairs']['A']:>6}{d['pairs']['B']:>6}{d['pairs']['C']:>6}{d['mixed']:>6}")
        for k in ("n_tracks", "n_ghost", "gtm", "gtt_trk", "gtt_all", "mixed"):
            tot[k] += d[k]
        tot["pairs"].update(d["pairs"])
    tpg = list(per_gt_cam.values())
    summary = dict(
        label=args.label, thr=args.thr, stride=args.stride, buffer=buf_info,
        seqs=args.seqs, n_tracks=tot["n_tracks"],
        ghost_rate=pct(tot["n_ghost"], tot["n_tracks"]),
        recall_trackframes=pct(tot["gtm"], tot["gtt_trk"]),
        recall_allframes=pct(tot["gtm"], tot["gtt_all"]),
        pairs=dict(tot["pairs"]), mixed_tracks=tot["mixed"],
        tracks_per_gt_per_cam=dict(median=st.median(tpg) if tpg else None,
                                   mean=round(st.mean(tpg), 3) if tpg else None),
        gap_median_s={k: round(st.median(v), 3) for k, v in all_gaps.items() if v},
        per_camera={c: dict(per_cam[c], pairs=dict(per_cam[c]["pairs"])) for c in per_cam},
        seconds=round(time.time() - t0, 1))
    print(f"  {'合計':<10}{tot['n_tracks']:>7}{summary['ghost_rate']:>8.1%}"
          f"{summary['recall_trackframes']:>14.1%}{summary['recall_allframes']:>14.1%}"
          f"{tot['pairs']['A']:>6}{tot['pairs']['B']:>6}{tot['pairs']['C']:>6}{tot['mixed']:>6}")
    print(f"\n  每位真人每台鏡頭 track 數:{summary['tracks_per_gt_per_cam']}")
    print(f"  間隔中位(秒):{summary['gap_median_s']}")
    if summary["recall_trackframes"] < 0.2:
        print("  ⚠ 召回極低 —— 先懷疑幀號對齊(GT 1-based、fid 0-based),不要先懷疑模型")

    out = Path(args.out or ROOT / "results" / "m4_layer0" / f"{args.label}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已存 {out}({summary['seconds']} 秒)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
