"""② 貼著人但框不準:隨機抽 8 條(固定種子 0,不挑選),各取「與人框 IoU ≥ 0.1」幀的中間那一幀。
紅 = track 的框、綠 = IoU 最大的那個人框、黃 = 同幀其他人的標註框。
用法:.venv/Scripts/python.exe results/m4_5cam/viz_boxes.py <輸出 png>
"""
import glob, random, sys
from collections import defaultdict
import numpy as np
import cv2
sys.path.insert(0, "scripts"); sys.path.insert(0, "src")
from diag_m4_ghosts import CATS, NEAR_IOU, features, load_rows, per_frame_assign
from eval_m4m5_chirla import load_gt, match_tracks
from common.video_io import iter_frames
R = "D:/新增資料夾/CHIRLA/CHIRLA_data/CHIRLA"
SEQS = ["seq_004", "seq_006", "seq_007", "seq_020", "seq_024", "seq_025", "seq_026"]
cands, gts = [], {}
for seq in SEQS:
    gt = load_gt(R, seq); gts[seq] = gt
    rows = load_rows(f"results/m4_5cam/cbiou_dump/{seq}/tracks.csv")
    tg, *_ = match_tracks(gt, rows)
    info, _ = per_frame_assign(gt, rows)
    by = defaultdict(list)
    for r in rows:
        by[(r["cam"], r["tid"])].append(r)
    for k, rs in by.items():
        if tg[k] is not None:
            continue
        fr = sorted(info[k])
        if features(fr, [r["bbox"] for r in rs], [r["conf"] for r in rs])["cat"] != CATS[1]:
            continue
        near = [x for x in fr if x[1] >= NEAR_IOU]
        fid, bi, bg, _ = near[len(near) // 2]
        cands.append((seq, k[0], k[1], fid, bg, {r["fid"]: r for r in rs}[fid]))
print("② 條數", len(cands))
random.seed(0)
pick = random.sample(cands, 8)
need = defaultdict(set)
for p in pick:
    need[(p[0], p[1])].add(p[3])
frames = {}
for (seq, cam), fids in need.items():
    vid = sorted(glob.glob(f"{R}/videos/{seq}/{cam}_*.avi"))[0]
    last = max(fids)
    for f, _t, img in iter_frames(vid):
        if f in fids:
            frames[(seq, cam, f)] = img.copy()
        if f >= last:
            break
tiles = []
for seq, cam, tid, fid, bg, row in pick:
    img = frames[(seq, cam, fid)]
    T = row["bbox"]
    for g, b in gts[seq][cam][fid + 1]:
        col = (0, 255, 0) if g == bg else (0, 255, 255)
        cv2.rectangle(img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), col, 2)
    cv2.rectangle(img, (int(T[0]), int(T[1])), (int(T[2]), int(T[3])), (0, 0, 255), 2)
    G = dict(gts[seq][cam][fid + 1])[bg]
    x1, y1 = min(T[0], G[0]), min(T[1], G[1])
    x2, y2 = max(T[2], G[2]), max(T[3], G[3])
    pw, ph = (x2 - x1) * 0.5 + 20, (y2 - y1) * 0.25 + 20
    h, w = img.shape[:2]
    c = img[int(max(0, y1 - ph)):int(min(h, y2 + ph)), int(max(0, x1 - pw)):int(min(w, x2 + pw))]
    s = 360 / c.shape[0]
    c = cv2.resize(c, (max(1, int(c.shape[1] * s)), 360))
    tile = np.zeros((400, 300, 3), np.uint8)
    cw = min(300, c.shape[1]); off = (300 - cw) // 2
    tile[40:400, off:off + cw] = c[:, (c.shape[1] - cw) // 2:(c.shape[1] - cw) // 2 + cw]
    cv2.putText(tile, f"{seq[-3:]} {cam[-1]} t{tid} f{fid}", (5, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    tiles.append(tile)
    print(seq, cam, tid, fid, "frame", img.shape)
out = np.vstack([np.hstack(tiles[:4]), np.hstack(tiles[4:])])
cv2.imwrite(sys.argv[1], out)
print("->", sys.argv[1])
