"""目視檢查 chef 標籤到底標在誰身上 —— 系統畫面與標註答案並排。

2026-09-17 使用者要看「標錯的真實影片狀況」。交付影片只畫系統的判斷,
看不出 chef 21 其實分給了好幾個人。這支產生兩種檢查輸出(僅供內部,不交付):

  1. 對照影片:左 = 系統框 + chef 標籤;右 = 標註框 + 真實人物編號;
     每個系統框再註明「這一幀」實際對到誰(IoU ≥ 0.5),或是誤偵。
  2. 整理圖:某個 chef 被綁到的每一位真人各裁一張(綁定當下那一幀)。

⚠ 「實際是誰」一律用**當下那一幀**與標註的 IoU 判定,不用 match_tracks 的整段多數決 ——
  混人的 track 整段多數決只會給一個人,正好把要看的東西藏起來。
⚠ 影片跳轉不可靠(見 render_level_video.py):一律 grab 順序快轉。

用法:
    python scripts/diag_chef_label_check.py --root "D:/yizhen/CHIRLA/CHIRLA_data/CHIRLA" \\
        --run-dir results/m5_step3/m5_cbiou/seq_026 --seq seq_026 \\
        --window camera_5 5835 6055 --chef 21 --out-dir results/levels/check/L1_03
"""
import argparse
import csv
import json
import shutil
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from common.draw_tracks import chef_color                # noqa: E402
from eval_m4m5_chirla import load_gt                     # noqa: E402

OLD_ROOT = "D:/新增資料夾/CHIRLA/CHIRLA_data/CHIRLA"


def iou(a, b):
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def who(box, gt_dets, thr=0.5):
    best, bi = None, 0.0
    for gid, gb in gt_dets:
        v = iou(box, gb)
        if v > bi:
            best, bi = gid, v
    return (best, bi) if bi >= thr else (None, bi)


def video_path(meta, cam, root):
    p = meta["camera_video_map"][cam]
    return root + p[len(OLD_ROOT):] if p.startswith(OLD_ROOT) else p


def read_frames(path, fids):
    """依序 grab 到每個 fid,回傳 {fid: frame}。"""
    import cv2
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise SystemExit(f"開不了影片 {path}")
    out, nxt = {}, 0
    for fid in sorted(set(fids)):
        while nxt < fid:
            if not cap.grab():
                break
            nxt += 1
        ok, fr = cap.read()
        nxt += 1
        if ok:
            out[fid] = fr
    cap.release()
    return out


def load_run(run_dir):
    tracks = defaultdict(list)                     # (cam, fid) -> [(tid, box)]
    chefs = {}
    with open(Path(run_dir) / "tracks.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["x1"] == "":
                continue
            key = (r["camera_id"], int(r["video_fid"]))
            tid = int(r["track_id"])
            tracks[key].append((tid, tuple(float(r[k]) for k in ("x1", "y1", "x2", "y2"))))
            if r.get("chef_id"):
                chefs[(r["camera_id"], tid)] = int(r["chef_id"])
    events = [json.loads(l) for l in
              (Path(run_dir) / "chef_events.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    meta = json.loads((Path(run_dir) / "run_meta.json").read_text(encoding="utf-8"))
    return tracks, chefs, events, meta


def put(img, text, org, color, scale=0.6, thick=2):
    import cv2
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def compare_video(args, tracks, chefs, meta, gt, out_dir):
    import cv2
    cam, lo, hi = args.window[0], int(args.window[1]), int(args.window[2])
    stride = int(meta["stride"])
    lo_fid = lo - ((lo) % stride)
    fids = list(range(lo_fid, hi + 1, stride))
    frames = read_frames(video_path(meta, cam, args.root), fids)
    tmp = Path(tempfile.gettempdir()) / "diag_chef_label_check.mp4"
    vw, rows = None, []
    for fid in fids:
        fr = frames.get(fid)
        if fr is None:
            continue
        left, right = fr.copy(), fr.copy()
        g = gt.get(cam, {}).get(fid + 1, [])
        for gid, b in g:
            x1, y1, x2, y2 = map(int, b)
            cv2.rectangle(right, (x1, y1), (x2, y2), (0, 200, 0), 3)
            put(right, f"person {gid}", (x1, max(20, y1 - 8)), (0, 255, 0), 0.8)
        for tid, b in tracks.get((cam, fid), []):
            x1, y1, x2, y2 = map(int, b)
            ch = chefs.get((cam, tid))
            col = chef_color(ch)
            cv2.rectangle(left, (x1, y1), (x2, y2), col, 3)
            gid, v = who(b, g)
            truth = f"= person {gid}" if gid is not None else "= no GT match"
            put(left, f"chef {ch}", (x1, max(20, y1 - 40)), col, 1.0)
            put(left, truth, (x1, max(44, y1 - 6)), (0, 255, 0) if gid is not None else (0, 0, 255), 0.9)
            rows.append(dict(fid=fid, track_id=tid, chef_id=ch, gt_person=gid, iou=round(v, 3)))
        put(left, f"SYSTEM  {cam}  t={fid / 30:.2f}s", (10, 30), (255, 255, 255), 0.8)
        put(right, "ANNOTATION (ground truth)", (10, 30), (255, 255, 255), 0.8)
        canvas = np.hstack([left, np.full((left.shape[0], 8, 3), 255, np.uint8), right])
        canvas = cv2.resize(canvas, (int(canvas.shape[1] * 0.75), int(canvas.shape[0] * 0.75)))
        if canvas.shape[1] % 2:
            canvas = canvas[:, :-1]
        if canvas.shape[0] % 2:
            canvas = canvas[:-1]
        if vw is None:
            vw = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), 30 / stride,
                                 (canvas.shape[1], canvas.shape[0]))
        vw.write(canvas)
    vw.release()
    final = out_dir / f"compare_{cam}_{lo}-{hi}.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(tmp), "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", "-crf", "23", str(tmp.with_suffix(".h264.mp4"))], check=True)
    shutil.move(str(tmp.with_suffix(".h264.mp4")), str(final))
    tmp.unlink(missing_ok=True)
    with open(out_dir / f"compare_{cam}_{lo}-{hi}.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"對照影片 {final}({len(fids)} 幀)")
    return rows


def chef_sheet(args, tracks, chefs, events, meta, gt, out_dir):
    """某個 chef 在綁定當下落在哪些真人身上:每人一張 + 誤偵一張。"""
    import cv2
    picks, ghost = {}, None
    summary = defaultdict(int)
    for e in sorted(events, key=lambda x: (x["video_fid"], x["camera_id"])):
        if e["chef_id"] != args.chef:
            continue
        cam, fid = e["camera_id"], e["video_fid"]
        box = next((b for t, b in tracks.get((cam, fid), []) if t == e["track_id"]), None)
        if box is None and e.get("bbox"):
            box = tuple(e["bbox"])
        if box is None:
            continue
        gid, _ = who(box, gt.get(cam, {}).get(fid + 1, []))
        summary[gid] += 1
        if gid is None:
            ghost = ghost or (cam, fid, e["track_id"], box, None)
        elif gid not in picks:
            picks[gid] = (cam, fid, e["track_id"], box, gid)
    items = [picks[k] for k in sorted(picks)] + ([ghost] if ghost else [])
    by_cam = defaultdict(list)
    for it in items:
        by_cam[it[0]].append(it[1])
    frames = {}
    for cam, fl in by_cam.items():
        for fid, fr in read_frames(video_path(meta, cam, args.root), fl).items():
            frames[(cam, fid)] = fr
    tiles = []
    for cam, fid, tid, box, gid in items:
        fr = frames.get((cam, fid))
        if fr is None:
            continue
        img = fr.copy()
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255) if gid is None else (0, 220, 255), 4)
        h, w = img.shape[:2]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        half = max(180, int(max(x2 - x1, y2 - y1) * 0.8))
        crop = img[max(0, cy - half):min(h, cy + half), max(0, cx - half):min(w, cx + half)]
        crop = cv2.resize(crop, (360, 360))
        bar = np.zeros((70, 360, 3), np.uint8)
        put(bar, f"chef {args.chef} -> " + (f"person {gid}" if gid is not None else "no GT match"),
            (8, 28), (0, 255, 0) if gid is not None else (0, 0, 255), 0.7)
        put(bar, f"{cam}  t={fid / 30:.1f}s  track {tid}", (8, 58), (255, 255, 255), 0.55, 1)
        tiles.append(np.vstack([bar, crop]))
    cols = 5
    while len(tiles) % cols:
        tiles.append(np.full_like(tiles[0], 30))
    grid = np.vstack([np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)])
    head = np.zeros((50, grid.shape[1], 3), np.uint8)
    n_people = len([k for k in summary if k is not None])
    put(head, f"{args.seq}: chef {args.chef} bindings at the moment of binding -> "
              f"{n_people} different people + {summary.get(None, 0)} detections with no GT match",
        (10, 34), (255, 255, 255), 0.8)
    out = out_dir / f"chef{args.chef}_{args.seq}_sheet.png"
    ok, buf = cv2.imencode(".png", np.vstack([head, grid]))
    out.write_bytes(buf.tobytes())
    print(f"整理圖 {out};綁定當下對到:{dict(summary)}")
    return dict(summary)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--seq", required=True)
    ap.add_argument("--window", nargs=3, metavar=("鏡頭", "起始fid", "結束fid"))
    ap.add_argument("--chef", type=int, default=None)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tracks, chefs, events, meta = load_run(args.run_dir)
    gt = load_gt(args.root, args.seq)
    report = {}
    if args.window:
        rows = compare_video(args, tracks, chefs, meta, gt, out_dir)
        agg = defaultdict(lambda: defaultdict(int))
        for r in rows:
            agg[(r["track_id"], r["chef_id"])][r["gt_person"]] += 1
        report["window"] = {f"track {t} / chef {c}": {str(k): v for k, v in d.items()}
                            for (t, c), d in agg.items()}
        print("窗內每條 track 逐幀對到:", json.dumps(report["window"], ensure_ascii=False))
    if args.chef is not None:
        report["chef_sheet"] = {str(k): v for k, v in chef_sheet(args, tracks, chefs, events, meta, gt, out_dir).items()}
    (out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
