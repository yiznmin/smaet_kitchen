"""外觀在換人時刻為什麼選錯:從「a 還有標註」的換人事件中,armS 選錯(cos(模板,b) ≥ cos(模板,a))的,
隨機抽 8 個(固定種子 0,不挑選)。每列:模板(track 在 f0 配到 a 的框)、a 在 f1 的標註框、b 在 f1 的標註框。
樣本選取規則與 scripts/diag_m4_appearance.py 的 S 組相同。
用法:.venv/Scripts/python.exe results/m4_5cam/viz_appearance_fail.py <輸出 png>
"""
import glob, json, random, sys
from collections import defaultdict
import cv2
import numpy as np
sys.path.insert(0, "scripts"); sys.path.insert(0, "src")
from diag_m4_ghosts import load_rows
from eval_m4m5_chirla import load_gt
from m5_reid.chirla_embedder import ChirlaEmbedder
R = "D:/新增資料夾/CHIRLA/CHIRLA_data/CHIRLA"
sw = json.load(open("results/m4_5cam/switch_cbiou.json", encoding="utf-8"))["switches"]
items, gts, boxes = [], {}, {}
for seq in sorted({s["seq"] for s in sw}):
    gts[seq] = load_gt(R, seq)
    for r in load_rows(f"results/m4_5cam/cbiou_dump/{seq}/tracks.csv"):
        boxes[(seq, r["cam"], r["tid"], r["fid"])] = r["bbox"]
for s in sw:
    if not s["a_visible"]:
        continue
    g = dict(gts[s["seq"]][s["cam"]].get(s["f1"] + 1, []))
    if s["b"] not in g:
        continue
    items.append((s, boxes[(s["seq"], s["cam"], s["tid"], s["f0"])], g[s["a"]], g[s["b"]]))
print("S 樣本", len(items))
need = defaultdict(set)
for s, *_ in items:
    need[(s["seq"], s["cam"])] |= {s["f0"], s["f1"]}
frames = {}
for (seq, cam), fids in need.items():
    cap = cv2.VideoCapture(sorted(glob.glob(f"{R}/videos/{seq}/{cam}_*.avi"))[0])
    f, last = 0, max(fids)
    while f <= last and cap.grab():
        if f in fids:
            frames[(seq, cam, f)] = cap.retrieve()[1]
        f += 1
    cap.release()
def crop(seq, cam, f, b):
    img = frames[(seq, cam, f)]; h, w = img.shape[:2]
    return img[max(0, int(b[1])):min(h, int(round(b[3]))), max(0, int(b[0])):min(w, int(round(b[2])))].copy()
trip = [(s, crop(s["seq"], s["cam"], s["f0"], t), crop(s["seq"], s["cam"], s["f1"], a), crop(s["seq"], s["cam"], s["f1"], b))
        for s, t, a, b in items]
emb = ChirlaEmbedder("model_result/reid/armS_reid_multi_camera/best.pth", quiet=True)
F = emb.extract_batch([c for _s, *cs in trip for c in cs]).reshape(len(trip), 3, -1)
wrong = [i for i in range(len(trip)) if float(F[i, 0] @ F[i, 2]) >= float(F[i, 0] @ F[i, 1])]
print("armS 選錯", len(wrong), "/", len(trip))
random.seed(0)
pick = random.sample(wrong, 8)
def tile(img, label):
    s = 240 / img.shape[0]
    im = cv2.resize(img, (max(1, min(160, int(img.shape[1] * s))), 240))
    t = np.zeros((270, 170, 3), np.uint8)
    t[30:270, (170 - im.shape[1]) // 2:(170 - im.shape[1]) // 2 + im.shape[1]] = im
    cv2.putText(t, label, (4, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    return t
rows = []
for i in pick:
    s, ct, ca, cb = trip[i]
    sa, sb = float(F[i, 0] @ F[i, 1]), float(F[i, 0] @ F[i, 2])
    rows.append(np.hstack([tile(ct, f"{s['seq'][-3:]} {s['cam'][-1]} t{s['tid']} T"), tile(ca, f"a {sa:.2f}"), tile(cb, f"b {sb:.2f}")]))
    print(s["seq"], s["cam"], s["tid"], s["f0"], s["f1"], s["cat"], "cos a", round(sa, 3), "cos b", round(sb, 3))
cv2.imwrite(sys.argv[1], np.vstack([np.hstack(rows[:4]), np.hstack(rows[4:])]))
print("->", sys.argv[1])
