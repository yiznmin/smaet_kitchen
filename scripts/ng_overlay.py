"""NG 報告片段:在影片上畫 M3 偵測框、zone 與系統逐步狀態(供 build_ng_report.py 使用)。

每 0.1 秒(快取一步 = 3 幀)更新一次框,和系統判斷的頻率相同;只畫規則實際用到的門檻以上的框。
畫面:上方字幕列(中文,PIL 繪製並快取)+ 下方 N 個鏡頭並排(每台 640×360)。
"""
import subprocess
from functools import lru_cache

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from m3.classes import NAME_TO_CLASS
from m4_track.det_cache import DetCache
from m6_events.violation import at, in_poly, steps

FONT = "C:/Windows/Fonts/msjh.ttc"
W, H, S = 640, 360, 0.5                    # 每台輸出大小、原影片 1280×720 縮放比
HEAD_H = 132                               # 上方字幕列高度
FPS, STRIDE = 30, 3
# 類別顏色(BGR)
COLOR = {"人": (200, 200, 200), "手": (0, 255, 255), "食材": (0, 200, 0), "刀具": (255, 0, 255),
         "砧板": (255, 160, 0), "鍋鏟": (0, 128, 255), "鍋子": (255, 255, 0), "容器": (180, 105, 255)}
THR = {"人": 0.3, "手": 0.2, "食材": 0.4, "刀具": 0.4, "砧板": 0.4, "鍋鏟": 0.4, "鍋子": 0.4, "容器": 0.4}
ZONE_COLOR = {"垃圾桶區": (0, 0, 255), "洗手台": (255, 128, 0)}
LEGEND = "框:人(灰) 手(黃) 食材(綠) 刀具(紫) 砧板(藍) 鍋鏟(橘) 鍋子(青) 容器(粉);zone:垃圾桶區(紅) 洗手台(藍)"


@lru_cache(maxsize=4096)
def text_img(text, size=20, color=(255, 255, 0), width=1920):
    """中文字串 → BGR 圖(黑底),快取重複使用。"""
    font = ImageFont.truetype(FONT, size)
    im = Image.new("RGB", (width, size + 8), (0, 0, 0))
    ImageDraw.Draw(im).text((8, 2), text, fill=color, font=font)
    return np.asarray(im)[:, :, ::-1].copy()


def paste(canvas, img, y):
    h = min(img.shape[0], canvas.shape[0] - y)
    w = min(img.shape[1], canvas.shape[1])
    canvas[y:y + h, :w] = img[:h, :w]


def draw_boxes(frame, cache, idx, step, thr):
    """frame 已縮放成 640×360;畫該步通過門檻的框。回傳該步偵測(原座標)。"""
    xy, cf, cl = at(cache, idx, step)
    for name, t in thr.items():
        k = (cl == NAME_TO_CLASS[name]) & (cf > t)
        for b, c in zip(xy[k], cf[k]):
            x1, y1, x2, y2 = (b * S).astype(int)
            cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR[name], 2 if name != "人" else 1)
            cv2.putText(frame, f"{c:.2f}", (x1 + 2, max(10, y1 - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR[name], 1)
    return xy, cf, cl


def draw_zone(frame, poly, name):
    p = (np.asarray(poly, float) * S).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(frame, [p], True, ZONE_COLOR[name], 2)


def ng3_hit(xy, cf, cl, h, f, fd, pad=0.5):
    """NG3 規則單步(同 eval_ng_clips.ng3_rule):手在人框頭部區 且 食材與擴大手框重疊。回傳 (成立, 頭部區框)。"""
    P = xy[(cl == NAME_TO_CLASS["人"]) & (cf >= 0.3)]
    Hn = xy[(cl == NAME_TO_CLASS["手"]) & (cf >= h)]
    F = xy[(cl == NAME_TO_CLASS["食材"]) & (cf >= fd)]
    heads = [(p[0], p[1], p[2], p[1] + f * (p[3] - p[1])) for p in P]
    if not (len(P) and len(Hn) and len(F)):
        return False, heads
    cx, cy = (Hn[:, 0] + Hn[:, 2]) / 2, (Hn[:, 1] + Hn[:, 3]) / 2
    top = P[:, 1] + f * (P[:, 3] - P[:, 1])
    head = ((cx[:, None] >= P[None, :, 0]) & (cx[:, None] <= P[None, :, 2])
            & (cy[:, None] >= P[None, :, 1]) & (cy[:, None] <= top[None])).any(1)
    if not head.any():
        return False, heads
    Hh = Hn[head]
    w, hh = Hh[:, 2] - Hh[:, 0], Hh[:, 3] - Hh[:, 1]
    Hx = np.c_[Hh[:, 0] - pad * w, Hh[:, 1] - pad * hh, Hh[:, 2] + pad * w, Hh[:, 3] + pad * hh]
    x1 = np.maximum(F[:, None, 0], Hx[None, :, 0])
    y1 = np.maximum(F[:, None, 1], Hx[None, :, 1])
    x2 = np.minimum(F[:, None, 2], Hx[None, :, 2])
    y2 = np.minimum(F[:, None, 3], Hx[None, :, 3])
    return bool(((x2 > x1) & (y2 > y1)).any()), heads


def render(job):
    """job:
    out, label(字幕前兩行), offset(片段第 0 幀的原影片秒數), start_frame, n_frames,
    views=[dict(name, video, cache, zones={名稱: 多邊形})], kind(ng12 / ng3 / ng4), status(依 kind)
    """
    caps = [cv2.VideoCapture(str(v["video"])) for v in job["views"]]
    for c in caps:
        for _ in range(job["start_frame"]):
            c.grab()
    caches = [DetCache(v["cache"]) for v in job["views"]]
    idxs = [steps(c) for c in caches]
    n = len(caps)
    width = W * n
    out_h = H + HEAD_H
    ff = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{out_h}",
                           "-r", str(FPS), "-i", "-", "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "26",
                           "-pix_fmt", "yuv420p", str(job["out"])], stdin=subprocess.PIPE)
    lines = job["label"].split("\n")
    head_static = np.zeros((HEAD_H, width, 3), np.uint8)
    for k, line in enumerate(lines[:2]):
        paste(head_static, text_img(line, 20, (255, 255, 0), width), 26 * k)
    paste(head_static, text_img(LEGEND, 16, (220, 220, 220), width), 26 * 2)
    st = job.get("status", {})
    last = [None] * n
    for fi in range(job["start_frame"], job["start_frame"] + job["n_frames"]):
        frames = []
        for c in caps:
            ok, fr = c.read()
            frames.append(cv2.resize(fr, (W, H)) if ok else np.zeros((H, W, 3), np.uint8))
        if not any(f.any() for f in frames):
            break
        step = fi // STRIDE
        msgs = []
        for j, (fr, v, cache, idx) in enumerate(zip(frames, job["views"], caches, idxs)):
            for zn, poly in v["zones"].items():
                draw_zone(fr, poly, zn)
            thr = job.get("thr", THR)
            xy, cf, cl = draw_boxes(fr, cache, idx, step, thr)
            if job["kind"] == "ng3":
                hit, heads = ng3_hit(xy, cf, cl, *job["ng3_param"])
                for (x1, y1, x2, y2) in heads:
                    cv2.rectangle(fr, (int(x1 * S), int(y1 * S)), (int(x2 * S), int(y2 * S)), (0, 0, 255), 1)
                if hit:
                    cv2.rectangle(fr, (0, 0), (W - 1, H - 1), (0, 255, 0), 6)
                    msgs.append(f"{v['name']} 手在頭部區且手邊有食材")
            cv2.putText(fr, v["name"], (6, H - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        # 狀態列
        if job["kind"] == "ng12":
            i = step
            seq = lambda a: bool(a[i]) if 0 <= i < len(a) else False
            parts = []
            parts.append("碰垃圾桶:是" if seq(st["contact"]) else "碰垃圾桶:否")
            parts.append("洗手中:是" if seq(st["wash"]) else "洗手中:否")
            tc = st["touch"][i] if 0 <= i < len(st["touch"]) else []
            parts.append("碰觸:" + ("、".join(tc) if tc else "否"))
            parts.append("髒手:是" if seq(st["dirty"]) else "髒手:否")
            if st.get("ng2_step") is not None and abs(i - st["ng2_step"]) <= 3:
                parts.append("← 系統判 NG2 的碰觸")
            msgs = ["  |  ".join(parts)]
        elif job["kind"] == "ng4":
            t = job["offset"] + fi / FPS
            act = [d for d in st["dets"] if d[0] <= t <= d[1]]
            msgs = ["系統 NG4 偵測中:" + ("、".join(sorted({d[2] for d in act})) if act else "無")]
        elif not msgs:
            msgs = ["NG3 規則:本步無鏡頭成立"]
        head = head_static.copy()
        paste(head, text_img(" / ".join(msgs)[:150], 20, (0, 255, 0), width), 26 * 3)
        t = job["offset"] + fi / FPS
        cv2.putText(head, f"video time {int(t // 3600):02d}:{int(t % 3600 // 60):02d}:{t % 60:05.2f}  step {step}",
                    (width - 420, HEAD_H - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        canvas = np.vstack([head, np.hstack(frames)])
        ff.stdin.write(canvas.tobytes())
    ff.stdin.close()
    ff.wait()
    for c in caps:
        c.release()
    return job["out"]
