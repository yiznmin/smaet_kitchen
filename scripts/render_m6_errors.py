"""把 M6 關聯的錯誤畫成圖,供人工檢視(只產本機檔案;EPFL 影像不進 git、不對外)。

系統:角色分工(自動定位)+ V(只算在動的物件),設定取 results/m6/dev_yh2040_r1/selected.json 的 V。
錯誤:
  誤報 FP = 系統說有接觸、真值沒有;漏報 FN = 真值有接觸、系統沒抓到
每類取最長的 n 段誤報、n 段漏報,畫中間那一幀:
  · FP 用「有成立」的那台近拍;FN 用該側近拍第一台(人在中間時用 Aoutput2)
  · 框:人(灰)、手(黃)、該類物件(綠 = 在動、紅 = 靜止,附 motion IoU)
  · 標頭:錯誤類型、類別、區間、該時刻真值裡的 EPFL 動作

輸出:results/m6/errors/<類別>.jpg、results/m6/errors/errors.csv

用法:
    python scripts/render_m6_errors.py
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import eval_m6_epfl as E                                                  # noqa: E402
import eval_m6_roles as R                                                 # noqa: E402
import eval_m6_round1 as R1                                               # noqa: E402
from common.video_io import imwrite_unicode                               # noqa: E402
from m3.classes import NAME_TO_CLASS                                      # noqa: E402
from m4_track.det_cache import DetCache                                   # noqa: E402
from m6_assoc.association import _above, motion_iou                       # noqa: E402
from m6_assoc.gt import SCORED, build_gt, load_actions                    # noqa: E402
from m6_assoc.metrics import intervals_from_mask, mask_from_intervals     # noqa: E402
from m6_assoc.roles import auto_side, load_roles, role_fuse               # noqa: E402

CELL_W, CELL_H, HEAD = 640, 360, 70
FONT = "C:/Windows/Fonts/msjh.ttc"


def put_text(img, lines, color=(255, 255, 255)):
    """在圖上方黑底寫中文(cv2 不支援中文,改用 PIL)。"""
    pil = Image.fromarray(img[:, :, ::-1])
    d = ImageDraw.Draw(pil)
    font = ImageFont.truetype(FONT, 18)
    d.rectangle([0, 0, img.shape[1], HEAD], fill=(0, 0, 0))
    for k, s in enumerate(lines):
        d.text((6, 4 + 22 * k), s, font=font, fill=color[::-1] if isinstance(color, tuple) else color)
    return np.array(pil)[:, :, ::-1].copy()


def read_frame(cam, fid):
    cap = cv2.VideoCapture(str(ROOT / "data" / "epfl" / f"{cam}.mp4"))
    cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
    ok, f = cap.read()
    cap.release()
    return f


def draw_cell(cam, step_i, cls_name, cfg, kind, iv, actions_now):
    c = DetCache(Path(ROOT / "results/det_cache/epfl_allcls_armA_s3") / f"{cam}.npz")
    fid = step_i * 3
    f = read_frame(cam, fid)
    lo, hi = np.searchsorted(c._fid, fid, "left"), np.searchsorted(c._fid, fid, "right")
    pf = max(0, fid - R1.LAG * 3)
    plo, phi = np.searchsorted(c._fid, pf, "left"), np.searchsorted(c._fid, pf, "right")
    xy, cf, cl = c._xyxy[lo:hi], c._conf[lo:hi], c._cls[lo:hi]
    person, hand, obj = NAME_TO_CLASS["人"], NAME_TO_CLASS["手"], NAME_TO_CLASS[cls_name]
    for k in np.flatnonzero((cl == person) & _above(cf, R1.FIXED["thr_person"])):
        cv2.rectangle(f, tuple(map(int, xy[k, :2])), tuple(map(int, xy[k, 2:])), (180, 180, 180), 2)
    for k in np.flatnonzero((cl == hand) & _above(cf, cfg["thr_hand"])):
        cv2.rectangle(f, tuple(map(int, xy[k, :2])), tuple(map(int, xy[k, 2:])), (0, 255, 255), 3)
    sel = np.flatnonzero((cl == obj) & _above(cf, cfg["thr_obj"]))
    psel = (c._cls[plo:phi] == obj) & _above(c._conf[plo:phi], cfg["thr_obj"])
    m = motion_iou(xy[sel].astype(float), cl[sel], c._xyxy[plo:phi][psel].astype(float), c._cls[plo:phi][psel])
    for k, mi in zip(sel, m):
        col = (0, 220, 0) if mi < cfg["iou_move"] else (0, 0, 255)
        cv2.rectangle(f, tuple(map(int, xy[k, :2])), tuple(map(int, xy[k, 2:])), col, 3)
        cv2.putText(f, f"{mi:.2f}", (int(xy[k, 0]), int(xy[k, 3]) + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, 2)
    f = cv2.resize(f, (CELL_W, CELL_H))
    f = np.vstack([np.zeros((HEAD, CELL_W, 3), np.uint8), f])
    title = f"{'誤報' if kind == 'FP' else '漏報'} {cls_name}  {iv[0]:.1f}–{iv[1]:.1f} 秒  {cam}"
    acts = "真值動作:" + ("、".join(actions_now) if actions_now else "(無)")
    return put_text(f, [title, acts[:60]], (0, 255, 255) if kind == "FP" else (255, 160, 0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default=str(ROOT / "results/det_cache/epfl_allcls_armA_s3"))
    ap.add_argument("--ann", default=str(ROOT / "data/epfl_ann/YH2040/2023_09_12_09_50_41/annotations/actions_annotations.xlsx"))
    ap.add_argument("--selected", default=str(ROOT / "results/m6/dev_yh2040_r1/selected.json"))
    ap.add_argument("--roles-selected", default=str(ROOT / "results/m6/dev_yh2040_roles/selected_roles.json"))
    ap.add_argument("--out-dir", default=str(ROOT / "results/m6/errors"))
    ap.add_argument("--n", type=int, default=3)
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    cfg = json.loads(Path(args.selected).read_text(encoding="utf-8"))["V"]["role_auto"]["cfg"]
    print("設定", cfg)
    roles = json.loads((ROOT / "configs/camera_roles.epfl.json").read_text(encoding="utf-8"))
    close = roles["close"]
    cams = close["A"] + close["B"]
    acts = load_actions(args.ann)
    gt, _ = build_gt(acts)
    t, step, F = R1.load_r1(args.cache_dir, cams)
    auto_x = json.loads(Path(args.roles_selected).read_text(encoding="utf-8"))["auto_x"]
    side = auto_side(R.foot_x(args.cache_dir, roles["locator"], t), **auto_x)
    raws = {c: R1.raw("V", F[R1.fkey(cfg)][c], cfg) for c in cams}
    pred = E.smooth_all(role_fuse(raws, side, close, cfg["role_k"]), cfg)

    rows = []
    for i, cls in enumerate(SCORED):
        g = mask_from_intervals(t, gt[cls]["union"])
        cells = []
        for kind, mask in (("FP", pred[:, i] & ~g), ("FN", g & ~pred[:, i])):
            ivs = sorted(intervals_from_mask(t, mask, step), key=lambda v: v[0] - v[1])[:args.n]
            for s, e in ivs:
                k = int(round((s + e) / 2 / step))
                sd = side[k]
                group = close[sd] if sd in ("A", "B") else cams
                if kind == "FP":
                    fired = [c for c in group if raws[c][k, i]]
                    cam = fired[0] if fired else group[0]
                else:
                    cam = group[0] if sd in ("A", "B") else "Aoutput2"
                now = [f"{a['verb']} {a['noun']}" for a in acts
                       if a["start"] <= t[k] < a["end"] and a["verb"] not in ("Wait", "Squat", "Read")]
                cells.append(draw_cell(cam, k, cls, cfg, kind, (s, e), now))
                rows.append(dict(cls=cls, kind=kind, start=round(s, 1), end=round(e, 1),
                                 dur=round(e - s, 1), cam=cam, side=sd, gt_actions="; ".join(now)))
            while len(cells) % args.n:
                cells.append(np.zeros((CELL_H + HEAD, CELL_W, 3), np.uint8))
        grid = np.vstack([np.hstack(cells[r:r + args.n]) for r in range(0, len(cells), args.n)])
        imwrite_unicode(out / f"{cls}.jpg", grid)
        print(f"  {cls}: {out / (cls + '.jpg')}")
    with open(out / "errors.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
