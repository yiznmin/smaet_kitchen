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
from m6_events.violation import at, in_poly, intervals, steps

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


def draw_boxes(frame, cache, idx, step, thr, show_conf=True, touched=()):
    """frame 已縮放成 640×360;畫該步通過門檻的框。回傳該步偵測(原座標)。

    touched:系統這一步判定碰觸的類別;這些類別中和手框(往外擴 50%)重疊的框加粗並標「碰觸」。
    """
    xy, cf, cl = at(cache, idx, step)
    hk = (cl == NAME_TO_CLASS["手"]) & (cf > thr.get("手", 0.2))
    hs = xy[hk]
    if len(hs):
        w, h = hs[:, 2] - hs[:, 0], hs[:, 3] - hs[:, 1]
        hs = np.c_[hs[:, 0] - 0.5 * w, hs[:, 1] - 0.5 * h, hs[:, 2] + 0.5 * w, hs[:, 3] + 0.5 * h]
    for name, t in thr.items():
        k = (cl == NAME_TO_CLASS[name]) & (cf > t)
        for b, c in zip(xy[k], cf[k]):
            x1, y1, x2, y2 = (b * S).astype(int)
            hit = name in touched and len(hs) and bool(((np.minimum(b[2], hs[:, 2]) > np.maximum(b[0], hs[:, 0]))
                                                        & (np.minimum(b[3], hs[:, 3]) > np.maximum(b[1], hs[:, 1]))).any())
            cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR[name], 4 if hit else (2 if name != "人" else 1))
            if hit:
                tag = text_img(f"碰觸 {name}", 14, COLOR[name][::-1], 90)
                yy = max(0, y1 - tag.shape[0])
                xx = max(0, min(x1, frame.shape[1] - tag.shape[1]))
                frame[yy:yy + tag.shape[0], xx:xx + tag.shape[1]] = tag
            elif show_conf:
                cv2.putText(frame, f"{c:.2f}", (x1 + 2, max(10, y1 - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR[name], 1)
    return xy, cf, cl


# ── NG1 / NG2 畫面(時間軸、橫幅)──────────────────────────────────
HEAD12 = 26 * 5 + 6                       # 5 行資訊
TL_ROWS = [("團隊標註", (150, 150, 150)), ("接觸垃圾桶", (0, 0, 255)), ("髒手", (0, 140, 255)),
           ("洗手", (255, 128, 0)), ("碰觸", (0, 200, 0)), ("違規觸發", (0, 0, 255))]
TL_LW, TL_RH = 120, 18
LEGEND12 = ("框:人(灰) 手(黃) 食材(綠) 刀具(紫) 砧板(藍) 鍋鏟(橘) 鍋子(青) 容器(粉),粗框 = 系統判定碰觸 | "
            "畫面外框:橘 = 髒手、藍 = 洗手中、紅 = 違規觸發")


def hms(t):
    return f"{int(t // 3600):02d}:{int(t % 3600 // 60):02d}:{t % 60:04.1f}"


def tl_x(i, n, width):
    return TL_LW + int(min(max(i, 0), n) / max(n, 1) * (width - TL_LW - 10))


def timeline_img(st, n, width):
    """整段片段的時間軸(靜態部分):每列一種狀態,違規觸發畫紅三角形 + NG 類別。"""
    h = TL_RH * len(TL_ROWS) + 8 + 24
    im = np.zeros((h, width, 3), np.uint8)
    T = len(st["dirty"])
    rows = [np.zeros(T, bool), st["contact"], st["dirty"], st["wash"], np.array([bool(x) for x in st["touch"]])]
    a, b = st["team"]
    rows[0][max(a, 0):max(b, 0)] = True
    for r, ((name, col), m) in enumerate(zip(TL_ROWS, rows + [None])):
        y = 4 + r * TL_RH
        paste_at(im, text_img(name, 14, (220, 220, 220), TL_LW - 4), y - 3, 0)
        cv2.rectangle(im, (TL_LW, y + 2), (width - 10, y + TL_RH - 2), (40, 40, 40), -1)
        if m is None:
            for e in st["events"]:
                if e["at"] > n:                              # 片段結束後才觸發 → 不畫
                    continue
                x = tl_x(e["at"], n, width)
                cv2.fillPoly(im, [np.array([[x, y + 1], [x - 6, y + TL_RH - 1], [x + 6, y + TL_RH - 1]])], col)
                tx = x + 8 if x < width - 60 else x - 48
                cv2.putText(im, e["type"], (tx, y + TL_RH - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            continue
        for s, e in intervals(m[:n + 1]):
            cv2.rectangle(im, (tl_x(s, n, width), y + 2), (max(tl_x(e, n, width), tl_x(s, n, width) + 1), y + TL_RH - 2), col, -1)
    paste_at(im, text_img(LEGEND12, 14, (200, 200, 200), width), h - 22, 0)
    return im


def paste_at(canvas, img, y, x):
    h = min(img.shape[0], canvas.shape[0] - y)
    w = min(img.shape[1], canvas.shape[1] - x)
    canvas[y:y + h, x:x + w] = img[:h, :w]


def banner(text, size=40):
    """違規橫幅:紅字、黑底,寬度依字數。"""
    font = ImageFont.truetype(FONT, size)
    w = int(font.getlength(text)) + 24
    im = Image.new("RGB", (w, size + 16), (0, 0, 0))
    ImageDraw.Draw(im).text((12, 4), text, fill=(255, 40, 40), font=font)
    return np.asarray(im)[:, :, ::-1].copy()


def ng12_lines(st, i, t_of):
    """回傳 [(文字, RGB 顏色)] 4 行:接觸污染源、髒手計時、碰觸物件、違規紀錄。"""
    T = len(st["dirty"])
    i = min(max(i, 0), T - 1)
    ep = st["ep"][i]
    if st["contact"][i]:
        a = st["eps"][ep][0] if ep >= 0 else i
        l2 = (f"接觸污染源:接觸垃圾桶中(第 {ep + 1} 次,{hms(t_of(a))} 起)", (255, 80, 80))
    elif ep >= 0:
        a, b = st["eps"][ep]
        l2 = (f"接觸污染源:第 {ep + 1} 次接觸垃圾桶 {hms(t_of(a))}–{hms(t_of(b))}", (255, 255, 255))
    else:
        l2 = ("接觸污染源:尚未接觸", (180, 180, 180))
    l2 = (l2[0] + f"      目前 {hms(t_of(i))}", l2[1])
    if st["dirty"][i]:
        p = [f"髒手 {st['since'][i]:.1f} 秒(從接觸開始)"]
        b, lim = st["eps"][ep][1], st.get("ng1_s", 5.0)
        if i >= b:                                       # NG1 計時從接觸結束起算
            d = (i - b) * STRIDE / FPS
            p.append(f"接觸結束後 {d:.1f} / {lim:.0f} 秒" + ("(已超過)" if d > lim else ""))
        if st["grace"][i]:
            p.append("緩衝期間:碰觸不計 NG2")
        acc = st["wash_acc"][i]
        p.append(f"洗手累計 {0 if np.isnan(acc) else acc:.1f} / {st['need_s']:.1f} 秒")
        left = st["ng1_left"][i]
        if not np.isnan(left):
            p.append(f"NG1 洗手期限剩 {left:.1f} 秒")
        elif not st["grace"][i]:
            p.append("NG1 洗手期限已過")
        l3 = ("手部狀態:" + " | ".join(p), (255, 160, 0))
    elif st["cleared"][i]:
        l3 = (f"手部狀態:已洗手 {st['need_s']:.1f} 秒,解除髒手", (80, 160, 255))
    else:
        l3 = ("手部狀態:乾淨", (120, 220, 120))
    tc = st["touch"][i]
    if not tc:
        l4 = ("碰觸物件:無", (180, 180, 180))
    else:
        why = ("緩衝期間,不計 NG2" if st["grace"][i] else "髒手碰觸 → NG2") if st["dirty"][i] else "手乾淨,不算違規"
        l4 = (f"碰觸物件:{'、'.join(tc)}({why})", (120, 255, 120))
    done = [e for e in st["events"] if e["at"] <= i]
    if done:
        cnt, parts = {}, []
        for e in done:
            cnt[e["type"]] = cnt.get(e["type"], 0) + 1
            parts.append(f"{e['type']} #{cnt[e['type']]} {hms(t_of(e['at']))} {e['text']}")
        l5 = ("違規紀錄:" + "  /  ".join(parts[-4:]) + (f"(共 {len(done)} 筆)" if len(done) > 4 else ""), (255, 60, 60))
    else:
        l5 = ("違規紀錄:無", (180, 180, 180))
    return [l2, l3, l4, l5]


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
    ng12 = job["kind"] == "ng12"
    st = job.get("status", {})
    n_steps = job["n_frames"] // STRIDE
    tl = timeline_img(st, n_steps, width) if ng12 else None
    head_h = HEAD12 if ng12 else HEAD_H
    out_h = H + head_h + (tl.shape[0] if ng12 else 0)
    t_of = lambda i: job["offset"] + i * STRIDE / FPS
    ff = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{out_h}",
                           "-r", str(FPS), "-i", "-", "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "26",
                           "-pix_fmt", "yuv420p", str(job["out"])], stdin=subprocess.PIPE)
    lines = job["label"].split("\n")
    head_static = np.zeros((head_h, width, 3), np.uint8)
    if ng12:
        paste(head_static, text_img(lines[0], 20, (255, 255, 0), width), 0)
    else:
        for k, line in enumerate(lines[:2]):
            paste(head_static, text_img(line, 20, (255, 255, 0), width), 26 * k)
        paste(head_static, text_img(LEGEND, 16, (220, 220, 220), width), 26 * 2)
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
            tc = (st["touch"][min(step, len(st["touch"]) - 1)] if st["touch"] else []) if ng12 else ()
            xy, cf, cl = draw_boxes(fr, cache, idx, step, thr, show_conf=not ng12, touched=tuple(tc))
            if job["kind"] == "ng3":
                hit, heads = ng3_hit(xy, cf, cl, *job["ng3_param"])
                for (x1, y1, x2, y2) in heads:
                    cv2.rectangle(fr, (int(x1 * S), int(y1 * S)), (int(x2 * S), int(y2 * S)), (0, 0, 255), 1)
                if hit:
                    cv2.rectangle(fr, (0, 0), (W - 1, H - 1), (0, 255, 0), 6)
                    msgs.append(f"{v['name']} 手在頭部區且手邊有食材")
            cv2.putText(fr, v["name"], (6, H - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        # 狀態列
        if ng12:
            i = min(step, len(st["dirty"]) - 1)
            head = head_static.copy()
            for k, (txt, col) in enumerate(ng12_lines(st, i, t_of), 1):
                paste(head, text_img(txt, 20, col, width), 26 * k)
            body = np.hstack(frames)
            alert = [e for e in st["events"] if e["at"] <= i < e["at"] + int(2.0 * FPS / STRIDE)]
            if alert:
                cv2.rectangle(body, (0, 0), (width - 1, H - 1), (0, 0, 255), 12)
                bn = banner("  ".join(f"{e['type']} 違規:{e['text']}" for e in alert))
                paste_at(body, bn, H // 2 - bn.shape[0] // 2, max(0, (width - bn.shape[1]) // 2))
            elif st["wash"][i]:
                cv2.rectangle(body, (0, 0), (width - 1, H - 1), (255, 128, 0), 6)
            elif st["dirty"][i]:
                cv2.rectangle(body, (0, 0), (width - 1, H - 1), (0, 140, 255), 6)
            bar = tl.copy()
            x = tl_x(step, n_steps, width)
            cv2.line(bar, (x, 0), (x, TL_RH * len(TL_ROWS) + 6), (255, 255, 255), 2)
            ff.stdin.write(np.vstack([head, body, bar]).tobytes())
            continue
        if job["kind"] == "ng4":
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
