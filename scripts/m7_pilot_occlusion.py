"""M7 階段 A 試跑:§6.3 的 9 段遮擋誤報,遮罩版「在動」判定能不能把它們判回靜止。

預先登記:docs/M7_模型評估_預先登記_20261020.md §3
每段逐步(0.1 秒)檢查該類、在手邊(間距 ≤ d × 人高)的物件:
  框版 :與 0.5 秒前任一同類框的最大 IoU(= V)
  遮罩版:同一個物件框在 t 與 t−0.5 秒各分割一次,扣掉兩幀的手遮罩後算 IoU
回報每段「被判為在動」的步數比例(框 vs 遮罩,m = 0.7 / 0.8 / 0.9),並畫中間幀。

用法:
    python scripts/m7_pilot_occlusion.py --model sam2.1_b+
"""
import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from common.video_io import imwrite_unicode                               # noqa: E402
from m3.classes import NAME_TO_CLASS                                      # noqa: E402
from m4_track.det_cache import DetCache                                   # noqa: E402
from m6_assoc.association import _above, gap, ioa, motion_iou             # noqa: E402
from m7_mask.contact import mask_motion_iou                               # noqa: E402
from m7_mask.segment import BoxSegmenter                                  # noqa: E402

# §6.3 歸為「靜止物件被判為在動(遮擋 / 框抖動)」的 9 段:(類別, 起, 訖, 鏡頭)
CASES = [("刀具", 215.9, 225.1, "Aoutput3"), ("刀具", 348.0, 353.3, "Aoutput1"),
         ("砧板", 351.0, 353.4, "Aoutput1"), ("砧板", 194.9, 196.9, "Aoutput1"),
         ("砧板", 65.8, 67.6, "Aoutput1"), ("鍋鏟", 444.6, 447.1, "Aoutput2"),
         ("鍋鏟", 338.8, 340.5, "Aoutput2"), ("鍋子", 348.0, 351.8, "Aoutput1"),
         ("鍋子", 635.0, 638.5, "Aoutput1")]
CFG = dict(thr_person=0.3, thr_hand=0.2, thr_obj=0.4, hand_in_person=0.5, d=0.1, iou_move=0.9)
STRIDE, LAG = 3, 5
MOVES = (0.7, 0.8, 0.9)          # 登記 §3 的 m 網格


class Frames:
    """依序讀影片並快取最近幾幀。"""
    def __init__(self, path):
        self.cap = cv2.VideoCapture(str(path))
        self.cache = {}

    def get(self, fid):
        if fid not in self.cache:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
            ok, f = self.cap.read()
            self.cache[fid] = f
        return self.cache[fid]


def dets(c, fid):
    lo, hi = np.searchsorted(c._fid, fid, "left"), np.searchsorted(c._fid, fid, "right")
    return c._xyxy[lo:hi].astype(np.float64), c._conf[lo:hi], c._cls[lo:hi]


def owned_hands(xy, cf, cl):
    """指派給某人框的手框,與該人的框高。"""
    persons = xy[(cl == NAME_TO_CLASS["人"]) & _above(cf, CFG["thr_person"])]
    hands = xy[(cl == NAME_TO_CLASS["手"]) & _above(cf, CFG["thr_hand"])]
    if not len(hands) or not len(persons):
        return np.zeros((0, 4)), np.zeros(0)
    m = ioa(hands, persons)
    best = m.argmax(1)
    ok = m[np.arange(len(hands)), best] >= CFG["hand_in_person"]
    return hands[ok], (persons[best, 3] - persons[best, 1])[ok]


def step_check(seg, emb_cache, frames, c, k, obj):
    """回傳這一步在手邊的物件:[(box, 框版 IoU, 遮罩版 IoU, 遮罩們)]。"""
    fid, pfid = k * STRIDE, (k - LAG) * STRIDE
    xy, cf, cl = dets(c, fid)
    pxy, pcf, pcl = dets(c, pfid)
    hands, heights = owned_hands(xy, cf, cl)
    phands, _ = owned_hands(pxy, pcf, pcl)
    sel = (cl == obj) & _above(cf, CFG["thr_obj"])
    objs = xy[sel]
    if not len(objs) or not len(hands):
        return []
    near = (gap(objs, hands) / heights[None, :]).min(1) <= CFG["d"]
    objs = objs[near]
    if not len(objs):
        return []
    psel = (pcl == obj) & _above(pcf, CFG["thr_obj"])
    box_iou = motion_iou(objs, np.full(len(objs), obj), pxy[psel], pcl[psel])
    for f in (fid, pfid):
        if f not in emb_cache:
            emb_cache[f] = seg.embed(frames.get(f)[:, :, ::-1].copy())
    m_t = seg.masks(emb_cache[fid], np.vstack([objs, hands]))
    m_p = seg.masks(emb_cache[pfid], np.vstack([objs, phands]) if len(phands) else objs)
    o_t, h_t = m_t[:len(objs)], m_t[len(objs):]
    o_p, h_p = m_p[:len(objs)], m_p[len(objs):]
    hand = h_t.any(0) | (h_p.any(0) if len(h_p) else False)
    out = []
    for i in range(len(objs)):
        out.append((objs[i], box_iou[i], mask_motion_iou(o_t[i], o_p[i], hand), o_t[i], o_p[i], hand))
    return out


def render(frame_t, frame_p, rec, title):
    box, biou, miou, o_t, o_p, hand = rec
    cells = []
    for f, o, tag in ((frame_t, o_t, "t"), (frame_p, o_p, "t-0.5s")):
        v = f.copy()
        v[o] = (0.5 * v[o] + 0.5 * np.array([0, 220, 0])).astype(np.uint8)
        v[hand] = (0.5 * v[hand] + 0.5 * np.array([0, 220, 255])).astype(np.uint8)
        cv2.rectangle(v, tuple(map(int, box[:2])), tuple(map(int, box[2:])), (0, 0, 255), 2)
        cv2.putText(v, tag, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
        cells.append(cv2.resize(v, (640, 360)))
    img = np.hstack(cells)
    cv2.putText(img, f"{title}  box IoU {biou:.2f} -> mask IoU (no hand) {miou:.2f}",
                (10, 350), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="sam2.1_b+")
    ap.add_argument("--cache-dir", default=str(ROOT / "results/det_cache/epfl_allcls_armA_s3"))
    ap.add_argument("--out-dir", default=str(ROOT / "results/m7/pilot_occlusion"))
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    seg = BoxSegmenter(args.model)

    rows, panels = [], []
    for n, (cls, s, e, cam) in enumerate(CASES):
        c = DetCache(Path(args.cache_dir) / f"{cam}.npz")
        frames = Frames(ROOT / "data/epfl" / f"{cam}.mp4")
        emb, obj = {}, NAME_TO_CLASS[cls]
        steps = range(int(round(s * 10)), int(round(e * 10)) + 1)
        n_any, mid = 0, None
        n_box, n_mask = np.zeros(len(MOVES)), np.zeros(len(MOVES))
        for k in steps:
            recs = step_check(seg, emb, frames, c, k, obj)
            if not recs:
                continue
            n_any += 1
            n_box += [any(r[1] < m for r in recs) for m in MOVES]
            n_mask += [any(r[2] < m for r in recs) for m in MOVES]
            if mid is None or abs(k - (s + e) * 5) < abs(mid[0] - (s + e) * 5):
                mid = (k, min(recs, key=lambda r: r[1]))
        rows.append(dict(cls=cls, start=s, end=e, cam=cam, steps_near=n_any,
                         **{f"box_moving@{m}": round(b / max(n_any, 1), 3) for m, b in zip(MOVES, n_box)},
                         **{f"mask_moving@{m}": round(b / max(n_any, 1), 3) for m, b in zip(MOVES, n_mask)}))
        print(rows[-1])
        if mid:
            k, rec = mid
            panels.append(render(frames.get(k * STRIDE), frames.get((k - LAG) * STRIDE), rec,
                                 f"#{n + 1} {s:.1f}s {cam}"))
    imwrite_unicode(out / f"{args.model}.jpg", np.vstack(panels))
    with open(out / f"{args.model}.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
