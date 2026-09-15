"""標註慣例查證:被前面的人擋住時,CHIRLA 的標註框有沒有畫到被擋住的部位?

隨機抽 10 個(固定種子 0,不挑選)「被另一個人的標註框蓋住 ≥ 30%,而且那個人在前面
(框底比被擋者的框底更低 ≥ 5% 框高,也就是離鏡頭較近)」的標註框。
綠 = 被擋住的人的標註框、黃 = 擋住他的人的標註框。只看影像,不做數值推論。
為了控制讀影片的時間,每 30 幀取一幀作為候選。

用法:.venv/Scripts/python.exe results/m4_5cam/viz_occluded_gt.py <輸出 png>
"""
import glob
import random
import sys
from collections import defaultdict

import cv2
import numpy as np

sys.path.insert(0, "scripts")
sys.path.insert(0, "src")
from eval_m4m5_chirla import load_gt          # noqa: E402
from common.video_io import iter_frames       # noqa: E402

R = "D:/新增資料夾/CHIRLA/CHIRLA_data/CHIRLA"
SEQS = ["seq_004", "seq_006", "seq_007", "seq_020", "seq_024", "seq_025", "seq_026"]
CAMS = ["camera_1", "camera_2", "camera_3", "camera_4", "camera_7"]


def area(b):
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def inter(a, b):
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    return w * h if w > 0 and h > 0 else 0.0


cands = []
for seq in SEQS:
    gt = load_gt(R, seq)
    for cam in CAMS:
        for fr1, boxes in gt.get(cam, {}).items():
            if (fr1 - 1) % 30:
                continue
            for g, G in boxes:
                gh = G[3] - G[1]
                for o, O in boxes:
                    if o == g or area(G) <= 0:
                        continue
                    if inter(G, O) / area(G) >= 0.3 and O[3] > G[3] + 0.05 * gh:
                        cands.append((seq, cam, fr1 - 1, g, G, o, O))
                        break
print("候選", len(cands))
random.seed(0)
pick = random.sample(cands, 10)
need = defaultdict(set)
for p in pick:
    need[(p[0], p[1])].add(p[2])
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
for seq, cam, fid, g, G, o, O in pick:
    img = frames[(seq, cam, fid)].copy()
    cv2.rectangle(img, (int(O[0]), int(O[1])), (int(O[2]), int(O[3])), (0, 255, 255), 2)
    cv2.rectangle(img, (int(G[0]), int(G[1])), (int(G[2]), int(G[3])), (0, 255, 0), 3)
    x1, y1 = min(G[0], O[0]), min(G[1], O[1])
    x2, y2 = max(G[2], O[2]), max(G[3], O[3])
    h, w = img.shape[:2]
    c = img[int(max(0, y1 - 20)):int(min(h, y2 + 20)), int(max(0, x1 - 20)):int(min(w, x2 + 20))]
    s = 360 / c.shape[0]
    c = cv2.resize(c, (max(1, int(c.shape[1] * s)), 360))
    tile = np.zeros((400, 320, 3), np.uint8)
    cw = min(320, c.shape[1])
    off = (320 - cw) // 2
    x0 = (c.shape[1] - cw) // 2
    tile[40:400, off:off + cw] = c[:, x0:x0 + cw]
    cv2.putText(tile, f"{seq[-3:]} {cam[-1]} f{fid} id{g}", (5, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    tiles.append(tile)
    print(seq, cam, fid, "被擋", g, "擋住的人", o, "蓋住比例", round(inter(G, O) / area(G), 2))
cv2.imwrite(sys.argv[1], np.vstack([np.hstack(tiles[:5]), np.hstack(tiles[5:])]))
print("->", sys.argv[1])
