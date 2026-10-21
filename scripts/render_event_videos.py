"""事件判斷結果影片:低成本流程(不用大模型)逐步畫出偵測、規則狀態與事件發出。

低成本組(設定 = 開發集選定、results/m7/events/frozen.json):
  垃圾桶 trash_body:Aoutput1 手在垃圾桶 zone,且 output0 人框蓋住垃圾桶 zone(±0.5 秒),連續 ≥ 1 秒
  抹布   cloth     :M6 V 規則 + 角色分工融合 + 平滑(HY3 凍結參數),T 0
  吃東西 eat_pose  :ViTPose 手腕與頭點距離 ≤ 1.2 × 肩寬,output0 / Aoutput1,連續 ≥ 0.5 秒

畫面:output0 | Aoutput1 | 近拍鏡頭(M3 抹布框最多的那台,不含 Aoutput1);上方為事件資訊與當下狀態,下方為時間軸
  綠 = 人、紅 = 手、紫 = 抹布、青 = 垃圾桶 zone(成立時變黃)、黃點 = 頭點、紅點 = 手腕
時間軸:標註區間(白)、各類「規則成立」(淡色)與「事件發出」(實色)、目前時間(白線)

輸出 results/m7/events/videos/<split>/(本機;EPFL 影像衍生物不進 git)與 index.csv
需 ffmpeg(轉 H.264,瀏覽器 / 播放器可直接播)。

用法:
    python scripts/render_event_videos.py --split test
"""
import argparse
import csv
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import eval_events as EV                                                  # noqa: E402
from common.zones import get_zone                                         # noqa: E402
from m4_track.det_cache import DetCache                                   # noqa: E402

FONT = "C:/Windows/Fonts/msjh.ttc"
PW, PH = 640, 360
HEAD, TL = 96, 120
STEP = 0.1
ARMS = [("trash", "垃圾桶", (255, 255, 0)), ("cloth", "抹布", (255, 0, 255)), ("eat", "吃東西", (0, 165, 255))]
CFG = dict(trash=dict(thr_hand=0.2, x=0.2, T=1.0), cloth=dict(T=0.0), eat=dict(r=1.2, T=0.5))


def fire_mask(raw, T):
    """平滑後連續成立 ≥ T 秒的整段標為「事件發出」。"""
    sm = EV.hysteresis(raw, *EV.EV_SMOOTH)
    need = max(1, int(round(T / STEP)))
    out = np.zeros(len(sm), bool)
    i = 0
    while i < len(sm):
        if sm[i]:
            j = i
            while j < len(sm) and sm[j]:
                j += 1
            if j - i >= need:
                out[i + need - 1:j] = True              # 從滿足 T 的那一刻起算發出
            i = j
        else:
            i += 1
    return sm, out


def pose_at(z, s):
    """取 ≤ s 的最近一次姿態(0.2 秒一幀)。"""
    if z is None or not len(z["step"]):
        return []
    k = z["step"][z["step"] <= s]
    if not len(k):
        return []
    return z["kp"][z["step"] == k.max()]


def draw_panel(img, c, lo, hi, j, poly=None, poly_on=False, kps=()):
    xy, cf, cl = c._xyxy[lo[j]:hi[j]], c._conf[lo[j]:hi[j]], c._cls[lo[j]:hi[j]]
    if poly is not None:
        p = np.int32(poly)
        if poly_on:
            ov = img.copy()
            cv2.fillPoly(ov, [p], (0, 255, 255))
            img[:] = cv2.addWeighted(ov, 0.35, img, 0.65, 0)
        cv2.polylines(img, [p], True, (0, 255, 255) if poly_on else (255, 255, 0), 3)
    for b, s, k in zip(xy, cf, cl):
        col = {EV.PERSON: ((0, 255, 0), 0.3), EV.HAND: ((0, 0, 255), 0.2), EV.CLOTH: ((255, 0, 255), 0.4)}.get(int(k))
        if col and s >= col[1]:
            cv2.rectangle(img, tuple(map(int, b[:2])), tuple(map(int, b[2:])), col[0], 3)
    for kp in kps:
        for i in range(5):
            if kp[i, 2] >= EV.KP_THR:
                cv2.circle(img, (int(kp[i, 0]), int(kp[i, 1])), 7, (0, 255, 255), -1)
        for i in (9, 10):
            if kp[i, 2] >= EV.KP_THR:
                cv2.circle(img, (int(kp[i, 0]), int(kp[i, 1])), 9, (0, 0, 255), -1)
    return cv2.resize(img, (PW, PH))


def text_block(w, h, lines):
    pil = Image.new("RGB", (w, h), (0, 0, 0))
    d = ImageDraw.Draw(pil)
    f = ImageFont.truetype(FONT, 22)
    for k, (s, col) in enumerate(lines):
        x = 8
        for part, pc in ([(s, col)] if isinstance(s, str) else s):
            d.text((x, 6 + 30 * k), part, font=f, fill=pc[::-1])
            x += d.textlength(part, font=f) + 6
    return np.array(pil)[:, :, ::-1].copy()


def timeline(w, T_len, e, states, cur):
    img = np.zeros((TL, w, 3), np.uint8)
    lab_w, x0 = 110, 120
    sx = lambda s: int(x0 + (w - x0 - 10) * s / max(T_len - 1, 1))
    rows = [("標註", None)] + [(n, k) for k, n, _ in ARMS]
    pil = Image.fromarray(img[:, :, ::-1])
    d = ImageDraw.Draw(pil)
    f = ImageFont.truetype(FONT, 18)
    for r, (n, _) in enumerate(rows):
        d.text((8, 4 + 28 * r), n, font=f, fill=(255, 255, 255))
    img = np.array(pil)[:, :, ::-1].copy()
    y = lambda r: 8 + 28 * r
    a0, a1 = int(max(e["a0"], 0) / STEP), int(e["a1"] / STEP)
    cv2.rectangle(img, (sx(a0), y(0)), (sx(min(a1, T_len - 1)), y(0) + 18), (255, 255, 255), -1)
    for r, (k, _, col) in enumerate(ARMS, 1):
        sm, fired = states[k]
        dim = tuple(int(v * 0.35) for v in col)
        for i in np.flatnonzero(sm):
            cv2.rectangle(img, (sx(i), y(r)), (sx(i + 1), y(r) + 18), dim, -1)
        for i in np.flatnonzero(fired):
            cv2.rectangle(img, (sx(i), y(r)), (sx(i + 1), y(r) + 18), col, -1)
    cv2.line(img, (sx(cur), 0), (sx(cur), TL), (255, 255, 255), 2)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["dev", "test"], required=True)
    ap.add_argument("--out-dir", default=str(ROOT / "results/m7/events/videos"))
    ap.add_argument("--limit", type=int, default=0, help="只畫前 N 個事件(冒煙用)")
    args = ap.parse_args()
    out = Path(args.out_dir) / args.split
    out.mkdir(parents=True, exist_ok=True)
    zones = {v: get_zone(ROOT / f"configs/zones/epfl_{v}.json", "垃圾桶區") for v in ("output0", "Aoutput1")}
    rows = []
    events = EV.load_events(args.split)
    for n, e in enumerate(events[:args.limit] if args.limit else events):
        c0 = DetCache(EV.CACHE / f"{e['clips']['output0']}.npz")
        T_len = len(EV.steps(c0)[0])
        a1 = EV.zone_raw(e, T_len, "Aoutput1", CFG["trash"]["thr_hand"], lambda e_, v: zones[v])
        body = EV.body_raw(e, T_len, zones["output0"], CFG["trash"]["x"])
        raws = dict(trash=EV.dilate(a1, 5) & EV.dilate(body, 5),
                    cloth=EV.cloth_raw(e, T_len),
                    eat=EV.pose_raw(e, T_len, CFG["eat"]["r"], ["output0", "Aoutput1"]))
        states = {k: fire_mask(raws[k], CFG[k]["T"]) for k, _, _ in ARMS}
        verdict = {k: bool(states[k][1].any()) for k, _, _ in ARMS}
        truth = {k: e["cat"] == name for k, name, _ in ARMS}
        tag = "、".join(f"{name}{'TP' if verdict[k] and truth[k] else 'FN' if truth[k] else 'FP'}"
                       for k, name, _ in ARMS if verdict[k] or truth[k]) or "都沒判"
        # 近拍鏡頭:M3 抹布框(conf ≥ 0.4)出現最多的那台
        best, best_n = "Aoutput2", -1
        for v in [x for x in EV.CLOSE["A"] + EV.CLOSE["B"] if x != "Aoutput1"]:      # 第 2 格已是 Aoutput1
            c = DetCache(EV.CACHE / f"{e['clips'][v]}.npz")
            k = int(((c._cls == EV.CLOTH) & (c._conf >= 0.4)).sum())
            if k > best_n:
                best, best_n = v, k
        views = ["output0", "Aoutput1", best]
        caches = {v: DetCache(EV.CACHE / f"{e['clips'][v]}.npz") for v in views}
        idx = {v: EV.steps(caches[v]) for v in views}
        poses = {v: (np.load(EV.POSE / f"{e['clips'][v]}.npz") if (EV.POSE / f"{e['clips'][v]}.npz").exists() else None)
                 for v in ("output0", "Aoutput1")}
        caps = {v: cv2.VideoCapture(str(EV.CLIPS / f"{e['clips'][v]}.mp4")) for v in views}
        fname = f"{n:03d}_{e['cat']}_{tag}_{e['session'].split('/')[-1]}.mp4"
        W, H = PW * 3, HEAD + PH + TL
        ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
                               "-s", f"{W}x{H}", "-r", "10", "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                               "-crf", "23", str(out / fname)], stdin=subprocess.PIPE)
        frames = {v: None for v in views}
        fid = 0
        for s in range(T_len):
            while fid <= s * 3:                              # 讀到第 s 步對應的幀(stride 3)
                for v in views:
                    ok, f = caps[v].read()
                    if ok:
                        frames[v] = f
                fid += 1
            panels = []
            for v in views:
                fr = frames[v] if frames[v] is not None else np.zeros((720, 1280, 3), np.uint8)
                _, lo, hi = idx[v]
                j = min(s, len(lo) - 1)
                on = {"output0": bool(body[s]), "Aoutput1": bool(a1[s])}.get(v, False)
                p = draw_panel(fr.copy(), caches[v], lo, hi, j, zones.get(v), on, pose_at(poses.get(v), s))
                cv2.putText(p, v, (8, PH - 12), 0, 0.9, (255, 255, 255), 2)
                panels.append(p)
            st = [(f"{name}:" + ("事件發出" if states[k][1][s] else "規則成立" if states[k][0][s] else "–"),
                   col if states[k][0][s] else (160, 160, 160)) for k, name, col in ARMS]
            head = text_block(W, HEAD, [
                (f"#{n} 標註:{e['cat']}  {e['note'] or ''}   場次 {e['session']}", (255, 255, 255)),
                (f"t = {s * STEP:5.1f} s   標註區間 {max(e['a0'], 0):.1f}–{e['a1']:.1f} s   整支判定:{tag}", (255, 255, 255)),
                (st, None)])
            frame = np.vstack([head, np.hstack(panels), timeline(W, T_len, e, states, s)])
            ff.stdin.write(frame.tobytes())
        ff.stdin.close()
        ff.wait()
        for cp in caps.values():
            cp.release()
        rows.append(dict(file=fname, id=e["id"], cat=e["cat"], note=e["note"], session=e["session"],
                         close_cam=best, **{f"{k}_fired": int(verdict[k]) for k, _, _ in ARMS}, verdict=tag))
        print(f"  {fname}", flush=True)
    with open(out / "index.csv", "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"輸出 {out}({len(rows)} 支)")


if __name__ == "__main__":
    main()
