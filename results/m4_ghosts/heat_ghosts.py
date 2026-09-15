import sys, glob
from collections import defaultdict
import numpy as np
import cv2
sys.path.insert(0, "scripts"); sys.path.insert(0, "src")
from diag_m4_ghosts import load_rows, per_frame_assign, features, CATS
from eval_m4m5_chirla import load_gt, match_tracks
from common.video_io import iter_frames
R = "D:/新增資料夾/CHIRLA/CHIRLA_data/CHIRLA"
SP = sys.argv[1]
SEQS = ["seq_004", "seq_006", "seq_007", "seq_020", "seq_024", "seq_025", "seq_026"]
boxes = defaultdict(list)   # cam -> [(bbox, cat)] 每條誤偵 track 的每一幀框
ntr = defaultdict(lambda: defaultdict(int))
for seq in SEQS:
    gt = load_gt(R, seq)
    rows = load_rows(f"results/m5_step3/m5_cbiou/m4dump/{seq}/tracks.csv")
    tg, *_ = match_tracks(gt, rows)
    info, _ = per_frame_assign(gt, rows)
    by = defaultdict(list)
    for r in rows:
        by[(r["cam"], r["tid"])].append(r)
    for k, rs in by.items():
        if k[0] not in ("camera_5", "camera_6") or tg[k] is not None:
            continue
        f = features(sorted(info[k]), [r["bbox"] for r in rs], [r["conf"] for r in rs])
        ntr[k[0]][f["cat"]] += 1
        if f["cat"] == CATS[0]:
            boxes[k[0]] += [r["bbox"] for r in rs]
for cam in ("camera_5", "camera_6"):
    vid = sorted(glob.glob(f"{R}/videos/seq_004/{cam}_*.avi"))[0]
    img = next(fr for f, _t, fr in iter_frames(vid) if f == 0)
    h, w = img.shape[:2]
    heat = np.zeros((h, w), np.float32)
    for b in boxes[cam]:
        x1, y1, x2, y2 = [int(round(v)) for v in b]
        heat[max(0, y1):min(h, y2), max(0, x1):min(w, x2)] += 1
    xs = np.array([(b[0] + b[2]) / 2 for b in boxes[cam]])
    hm = cv2.applyColorMap((255 * heat / heat.max()).astype(np.uint8), cv2.COLORMAP_JET)
    out = cv2.addWeighted(img, 0.5, hm, 0.5, 0)
    p = f"{SP}/heat_{cam}.png"
    cv2.imwrite(p, out)
    print(cam, "① tracks", ntr[cam][CATS[0]], "box-frames", len(boxes[cam]),
          "center-x quartiles", np.percentile(xs, [10, 25, 50, 75, 90]).round(0).tolist(), "->", p)
