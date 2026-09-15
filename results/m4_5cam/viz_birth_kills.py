"""被幾何規則誤殺的「真的新人」長什麼樣:從出生標記為 N 新人、包含比例 ≥ 0.9、
且同一幀沒有同一人的舊 track 的出生中,隨機抽 8 個(固定種子 0,不挑選)。
紅 = 新 track 的框、藍 = 包含它最多的較早 track 框、綠 = 標註框(被新 track 配到的人加粗)。
用法:.venv/Scripts/python.exe results/m4_5cam/viz_birth_kills.py <輸出 png>
"""
import csv, glob, json, random, sys
from collections import defaultdict
import cv2
import numpy as np
sys.path.insert(0, "scripts"); sys.path.insert(0, "src")
from eval_m4m5_chirla import load_gt
from common.video_io import iter_frames
R = "D:/新增資料夾/CHIRLA/CHIRLA_data/CHIRLA"
d = json.load(open("results/m4_5cam/birth_geometry_cbiou.json", encoding="utf-8"))
pool = [b for b in d["births"] if b["label"] == "N 新人" and b["cont"] >= 0.9 and not b["n_same_person_old"]]
print("候選", len(pool))
random.seed(0)
pick = random.sample(pool, min(8, len(pool)))

def inter(a, b):
    w = min(a[2], b[2]) - max(a[0], b[0]); h = min(a[3], b[3]) - max(a[1], b[1])
    return w * h if w > 0 and h > 0 else 0.0

tiles = []
rows_cache, gt_cache = {}, {}
need = defaultdict(set)
for p in pick:
    need[(p["seq"], p["cam"])].add(p["fid"])
frames = {}
for (seq, cam), fids in need.items():
    vid = sorted(glob.glob(f"{R}/videos/{seq}/{cam}_*.avi"))[0]
    last = max(fids)
    for f, _t, img in iter_frames(vid):
        if f in fids:
            frames[(seq, cam, f)] = img.copy()
        if f >= last:
            break
for p in pick:
    seq, cam, tid, fid = p["seq"], p["cam"], p["tid"], p["fid"]
    if seq not in rows_cache:
        rows_cache[seq] = list(csv.DictReader(open(f"results/m4_5cam/cbiou_dump/{seq}/tracks.csv", encoding="utf-8")))
        gt_cache[seq] = load_gt(R, seq)
    rs = [r for r in rows_cache[seq] if r["camera_id"] == cam]
    first = {}
    for r in rs:
        first[int(r["track_id"])] = min(first.get(int(r["track_id"]), 10**9), int(r["video_fid"]))
    box = lambda r: tuple(float(r[k]) for k in ("x1", "y1", "x2", "y2"))
    here = [r for r in rs if int(r["video_fid"]) == fid]
    N = box(next(r for r in here if int(r["track_id"]) == tid))
    aN = (N[2] - N[0]) * (N[3] - N[1])
    E = max((box(r) for r in here if int(r["track_id"]) != tid and first[int(r["track_id"])] < fid),
            key=lambda b: inter(N, b) / aN)
    img = frames[(seq, cam, fid)]
    for g, G in gt_cache[seq][cam].get(fid + 1, []):
        cv2.rectangle(img, (int(G[0]), int(G[1])), (int(G[2]), int(G[3])), (0, 255, 0), 1)
    cv2.rectangle(img, (int(E[0]), int(E[1])), (int(E[2]), int(E[3])), (255, 128, 0), 3)
    cv2.rectangle(img, (int(N[0]), int(N[1])), (int(N[2]), int(N[3])), (0, 0, 255), 2)
    x1, y1 = min(N[0], E[0]), min(N[1], E[1]); x2, y2 = max(N[2], E[2]), max(N[3], E[3])
    h, w = img.shape[:2]
    c = img[int(max(0, y1 - 30)):int(min(h, y2 + 30)), int(max(0, x1 - 40)):int(min(w, x2 + 40))]
    s = 360 / c.shape[0]
    c = cv2.resize(c, (max(1, int(c.shape[1] * s)), 360))
    tile = np.zeros((400, 320, 3), np.uint8)
    cw = min(320, c.shape[1]); off = (320 - cw) // 2; x0 = (c.shape[1] - cw) // 2
    tile[40:400, off:off + cw] = c[:, x0:x0 + cw]
    cv2.putText(tile, f"{seq[-3:]} {cam[-1]} t{tid} f{fid} c{p['cont']:.2f}", (5, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    tiles.append(tile)
    print(seq, cam, tid, fid, "包含", round(p["cont"], 2), "IoU", round(p["iou"], 2))
while len(tiles) < 8:
    tiles.append(np.zeros((400, 320, 3), np.uint8))
cv2.imwrite(sys.argv[1], np.vstack([np.hstack(tiles[:4]), np.hstack(tiles[4:])]))
print("->", sys.argv[1])
