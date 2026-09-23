"""逐片段獨立推論的**內部檢視**影片:系統框 + 標註對照 + 逐幀對錯。

預先登記 `docs/難度分級_逐片段獨立推論_預先登記_20260923.md` §3、§7-8。

## ⚠ 這不是交付影片

交付版(`render_level_video.py`,`--gt-mode none`)刻意只畫系統的判斷,理由寫在它的檔頭:
畫面上的對錯是**逐幀**的,而誤併率是**逐次綁定**的,兩者分母不同,並排給出資方看會被讀成錯誤率。
本支是內部檢查用,所以**每一幀的字幕都寫死「內部檢查用,不可交付」**,
且輸出固定在 `results/clips/videos/`。

## 重放,不重跑

框與編號讀自該片段執行的 `tracks.csv`,對錯讀自 `metrics/<clip>/s<stride>/per_frame.csv` ——
**與指標同一份資料**,影片和數字不可能分歧(V8 驗幀數與列數)。

用法:
    python scripts/render_clip_video.py --manifest results/clips/clip_manifest.json \
        --run-root results/clips/runs --metrics-root results/clips/metrics \
        --out-dir results/clips/videos --strides 5 1
"""
import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from common.draw_tracks import chef_color                            # noqa: E402
from render_level_video import Encoder, caption, even, probe         # noqa: E402

CAPTION = "內部檢查用,不可交付 · 逐片段獨立推論 · CHIRLA CC-BY-4.0"


def load_rows(metrics_dir):
    """per_frame.csv → {(cam, fid): 列}。指標與影片共用這一份。"""
    out = {}
    with open(Path(metrics_dir) / "per_frame.csv", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            out[(r["camera_id"], int(r["video_fid"]))] = r
    return out


def load_tracks(run_dir):
    per = defaultdict(list)
    chefs = {}
    with open(Path(run_dir) / "tracks.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            cam, fid, tid = r["camera_id"], int(r["video_fid"]), int(r["track_id"])
            box = (None if r["x1"] == ""
                   else tuple(float(r[k]) for k in ("x1", "y1", "x2", "y2")))
            per[(cam, fid)].append((tid, box))
            if r.get("chef_id"):
                chefs[(cam, tid)] = int(r["chef_id"])
    return per, chefs


def put(img, text, org, color, scale=0.6, thick=2):
    import cv2
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 3,
                cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def panel(frame, cam, fid, tracks, chefs, row, target, width):
    """一台鏡頭的畫面:系統框(依 chef 上色)+ 標註框(綠)+ 這一幀的判定。"""
    import cv2
    img = frame.copy()
    for tid, box in tracks:
        if box is None:
            continue
        x1, y1, x2, y2 = map(int, box)
        ch = chefs.get((cam, tid))
        cv2.rectangle(img, (x1, y1), (x2, y2), chef_color(ch), 3)
        put(img, f"chef {ch}" if ch else f"track {tid} 未綁定",
            (x1, max(22, y1 - 8)), chef_color(ch), 0.7)
    gt_box = row and row["gt_box"]
    if gt_box:
        x1, y1, x2, y2 = (int(float(v)) for v in gt_box.split())
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 220, 0), 2)
        put(img, f"GT person {target}", (x1, min(img.shape[0] - 8, y2 + 24)), (0, 220, 0), 0.6)
    if row:
        if not int(row["gt_present"]):
            st, col = "target absent", (200, 200, 200)
        elif not int(row["matched"]):
            st, col = "MISS (no box)", (0, 0, 255)
        elif row["in_B"] == "1" and row["is_continuity"] == "1":
            st, col = f"OK  IoU {float(row['iou']):.2f}", (0, 255, 0)
        elif row["in_B"] == "1":
            st, col = f"WRONG ID (chef {row['chef_id']})  IoU {float(row['iou']):.2f}", (0, 0, 255)
        else:
            st, col = f"unbound  IoU {float(row['iou']):.2f}", (0, 165, 255)
        put(img, f"{cam}  fid {fid}  {st}", (10, 28), col, 0.7)
    h, w = img.shape[:2]
    return cv2.resize(img, (width, int(h * width / w)))


def render(clip, stride, run_root, metrics_root, out_dir, args):
    import cv2
    run_dir = Path(run_root) / clip["clip_id"] / f"s{stride}"
    met_dir = Path(metrics_root) / clip["clip_id"] / f"s{stride}"
    tracks, chefs = load_tracks(run_dir)
    rows = load_rows(met_dir)
    meta = json.loads((run_dir / "run_meta.json").read_text(encoding="utf-8"))
    fps_src = float(meta["videos"][clip["cameras"][0]]["fps"])
    fps_out = fps_src / stride
    caps = {}
    for c in clip["cameras"]:
        cap = cv2.VideoCapture(clip["videos"][c])
        if not cap.isOpened():
            raise SystemExit(f"開不了影片 {clip['videos'][c]}")
        caps[c] = [cap, 0]
    fids = list(range(clip["start_fid"], clip["end_fid"] + 1, stride))
    enc, size, strip, n_ok, n_bad = None, None, None, 0, 0
    for fid in fids:
        panels = []
        for c in clip["cameras"]:
            cap, nxt = caps[c]
            while nxt < fid:                     # grab 快轉,不用 POS_FRAMES
                if not cap.grab():
                    break
                nxt += 1
            ok, frame = cap.read()
            caps[c][1] = nxt + 1
            if not ok:
                frame = np.full((int(meta["videos"][c]["height"]),
                                 int(meta["videos"][c]["width"]), 3), 20, np.uint8)
            r = rows.get((c, fid))
            if r and int(r["gt_present"]) and r["in_B"] == "1":
                n_ok += int(r["is_continuity"] == "1")
                n_bad += int(r["is_continuity"] != "1")
            panels.append(panel(frame, c, fid, tracks.get((c, fid), []), chefs, r,
                                clip["gt_id"], args.width))
        canvas = even(np.hstack(panels))
        if strip is None:
            txt = (f"{clip['rule']} {clip['clip_id']} | {clip['seq']} "
                   f"{'+'.join(clip['cameras'])} | 目標 person {clip['gt_id']} | "
                   f"stride={stride} → {fps_out:g} fps | {CAPTION}")
            strip = caption(txt, canvas.shape[1], args.font)
        canvas = even(np.vstack([canvas, strip]))
        if enc is None:
            size = (canvas.shape[1], canvas.shape[0])
            enc = Encoder(out_dir / f"{clip['clip_id']}_s{stride}.mp4", fps_out, size,
                          args.crf, args.encoder)
        enc.write(canvas)
    for cap, _ in caps.values():
        cap.release()
    enc.close()
    path = out_dir / f"{clip['clip_id']}_s{stride}.mp4"
    codec, _ = probe(path)
    return dict(clip_id=clip["clip_id"], stride=stride, frames=len(fids),
                expected_frames=(clip["end_fid"] - clip["start_fid"]) // stride + 1,
                n_correct_cells=n_ok, n_wrong_cells=n_bad,
                fps=fps_out, size=list(size), codec=codec,
                bytes=path.stat().st_size, path=str(path))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="results/clips/clip_manifest.json")
    ap.add_argument("--run-root", default="results/clips/runs")
    ap.add_argument("--metrics-root", default="results/clips/metrics")
    ap.add_argument("--out-dir", default="results/clips/videos")
    ap.add_argument("--strides", nargs="+", type=int, default=[5])
    ap.add_argument("--clips", nargs="+", default=None)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--crf", type=int, default=23)
    ap.add_argument("--encoder", choices=("ffmpeg", "cv2"), default="ffmpeg")
    ap.add_argument("--font", default="C:/Windows/Fonts/msjh.ttc")
    ap.add_argument("--out", default="results/clips/agg/render_summary.json")
    args = ap.parse_args()

    man = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    clips = man["clips"]
    if args.clips:
        clips = [c for c in clips if c["clip_id"] in set(args.clips)]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, bad = [], []
    for i, c in enumerate(clips, 1):
        for s in args.strides:
            if not (Path(args.metrics_root) / c["clip_id"] / f"s{s}" / "per_frame.csv").exists():
                bad.append(f"{c['clip_id']} s{s}:沒有指標,先跑 eval_clips.py")
                continue
            r = render(c, s, args.run_root, args.metrics_root, out_dir, args)
            rows.append(r)
            flag = "" if r["frames"] == r["expected_frames"] else "  ← 幀數不符"
            print(f"[{i:>3}/{len(clips)}] {r['clip_id']:<34} s{s} "
                  f"{r['frames']:>4} 幀  {r['bytes']/1e6:.2f} MB  {r['codec']}{flag}")
            if r["frames"] != r["expected_frames"] or r["codec"] != "h264":
                bad.append(f"{r['clip_id']} s{s}:幀數 {r['frames']}/{r['expected_frames']} "
                           f"codec {r['codec']}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(dict(rows=rows, failures=bad),
                                         ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n{len(rows)} 支影片;V8 失敗 {len(bad)} 項;已存 {args.out}")
    for b in bad[:10]:
        print("  -", b)
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
