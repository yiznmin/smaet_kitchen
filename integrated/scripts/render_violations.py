"""把 run_violations.py 判出的每件違規剪成一支對照影片,供人工檢查與驗收。

每支影片:6 台鏡頭並排(3 x 2),疊上 M3 偵測框與區域;上方是違規類別、原因、時間;
下方時間軸標出違規區間與目前播放位置;播放到違規區間內時畫面外框變紅。

用法(先跑完 run_violations.py,--out 用同一個資料夾):
  python scripts/render_violations.py --video-dir <影片資料夾> --out outputs/ng
輸出:<out>/report/<編號>_<NG 類別>.mp4
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from common.video_io import iter_frames, video_meta   # noqa: E402
from common.zones import load_zones                   # noqa: E402
from m3.classes import NAME_TO_CLASS                  # noqa: E402
from m4_track.det_cache import DetCache               # noqa: E402

CAMS = ["output0", "Aoutput1", "Aoutput2", "Aoutput3", "Boutput1", "Boutput2"]
TILE_W, TILE_H = 640, 360
HEADER_H, BAR_H = 96, 44
# 畫框的類別、門檻與顏色(BGR);改顏色時要一併改 render() 標題列的圖例文字
DRAW = {"人": (0.3, (200, 200, 200)), "手": (0.2, (0, 220, 255)), "食材": (0.4, (60, 200, 60)),
        "刀具": (0.4, (255, 120, 0)), "砧板": (0.4, (180, 90, 200)), "鍋鏟": (0.4, (255, 200, 0)),
        "鍋子": (0.4, (0, 120, 255)), "容器": (0.4, (200, 255, 120))}
ZONE_COLOR = (0, 0, 255)


def text_img(lines, width, height, font_path, size=26):
    """多行中文文字 → BGR 圖(cv2 無法畫中文,改用 PIL)。lines:[(文字, RGB)]。"""
    font = ImageFont.truetype(font_path, size)
    im = Image.new("RGB", (width, height), (25, 25, 25))
    d = ImageDraw.Draw(im)
    for i, (t, rgb) in enumerate(lines):
        d.text((12, 6 + i * (size + 8)), t, fill=rgb, font=font)
    return cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)


def describe(e):
    detail = []
    if e.get("touched"):
        detail.append("碰觸:" + "、".join(e["touched"]))
    if e.get("obj"):
        detail.append(f"物件:{e['obj']}、位置:{e.get('place', '')}、{e.get('status', '')}")
    if e.get("views"):
        detail.append(f"成立鏡頭:{e['views']}")
    return f"{e['type']}  {e['reason']}" + ("  |  " + "  ".join(detail) if detail else "")


def draw_tile(frame, cam, fid, cache, zones):
    vis = frame.copy()
    for z in zones:
        cv2.polylines(vis, [np.array(z["points"], np.int32)], True, ZONE_COLOR, 2)
    if cache is not None:
        stride, last = int(cache.meta["cache_stride"]), int(cache.meta["max_fid"])
        f = min(fid - fid % stride, last - last % stride)
        det = cache.get(f, 0.2)
        for xy, cf, cl in zip(det.xyxy, det.confidence, det.class_id):
            for name, (thr, color) in DRAW.items():
                if cl == NAME_TO_CLASS[name] and cf > thr:
                    x1, y1, x2, y2 = map(int, xy)
                    cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
    vis = cv2.resize(vis, (TILE_W, TILE_H))
    cv2.putText(vis, cam, (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return vis


def time_bar(t, t_s, t_e, e_s, e_e, width):
    bar = np.full((BAR_H, width, 3), 25, np.uint8)
    x = lambda v: int(12 + (v - t_s) / max(t_e - t_s, 1e-6) * (width - 24))
    cv2.rectangle(bar, (12, 14), (width - 12, 30), (80, 80, 80), -1)
    cv2.rectangle(bar, (x(e_s), 14), (x(e_e), 30), (0, 0, 230), -1)
    cv2.line(bar, (x(t), 6), (x(t), 38), (255, 255, 255), 3)
    cv2.putText(bar, f"{t:.1f}s", (min(x(t) + 6, width - 90), 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return bar


def render(k, e, args, cfg, t0, caches, zones_of, font):
    metas = {c: video_meta(Path(args.video_dir) / f"{c}.mp4") for c in CAMS}
    fps = metas[CAMS[0]]["fps"]
    e_s, e_e = e["start"] - t0, e["end"] - t0
    t_s = max(0.0, e_s - args.pad)
    t_e = min(min(m["nb_frames"] for m in metas.values()) / fps, e_e + args.pad)
    f_s, f_e = int(t_s * fps), int(t_e * fps)
    W = TILE_W * 3
    header = text_img([(f"#{k}  {describe(e)}", (255, 90, 90)),
                       (f"違規區間 {e_s:.1f}–{e_e:.1f} 秒(影片時間)    框:灰=人 黃=手 綠=食材 藍=刀具 紫=砧板 "
                        f"淺藍=鍋鏟 橘=鍋子 淺綠=容器    紅線=區域", (230, 230, 230))], W, HEADER_H, font)
    out = Path(args.out) / "report" / f"{k:03d}_{e['type']}.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)
    vw = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, HEADER_H + TILE_H * 2 + BAR_H))
    gens = [iter_frames(str(Path(args.video_dir) / f"{c}.mp4"), start=f_s, end=f_e) for c in CAMS]
    for items in zip(*gens):
        fid = items[0][0]
        tiles = [draw_tile(fr, c, fid, caches.get(c), zones_of.get(c, [])) for c, (_f, _t, fr) in zip(CAMS, items)]
        grid = np.vstack([np.hstack(tiles[:3]), np.hstack(tiles[3:])])
        t = fid / fps
        if e_s <= t <= e_e:
            cv2.rectangle(grid, (0, 0), (W - 1, TILE_H * 2 - 1), (0, 0, 255), 8)
        vw.write(np.vstack([header, grid, time_bar(t, t_s, t_e, e_s, e_e, W)]))
    vw.release()
    return out


def main():
    ap = argparse.ArgumentParser(description="把每件違規剪成對照影片")
    ap.add_argument("--video-dir", required=True, help="與 run_violations.py 相同的影片資料夾")
    ap.add_argument("--out", required=True, help="run_violations.py 的輸出資料夾")
    ap.add_argument("--pad", type=float, default=2.0, help="違規區間前後各多剪幾秒")
    ap.add_argument("--font", default="C:/Windows/Fonts/msjh.ttc", help="中文字型檔(非 Windows 請改成可用的字型)")
    args = ap.parse_args()

    out = Path(args.out)
    events = [json.loads(l) for l in (out / "ng_events.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    summary = json.loads((out / "ng_summary.json").read_text(encoding="utf-8"))
    cfg = yaml.safe_load((ROOT / summary.get("config", "configs/violation_rules.yaml")).read_text(encoding="utf-8"))
    cache_dir = Path(summary["det_cache"])
    caches = {c: DetCache(cache_dir / f"{c}.npz") for c in CAMS if (cache_dir / f"{c}.npz").exists()}
    zones_of = {c: load_zones(ROOT / p)["zones"] for c, p in cfg["zones"].items()}
    if not events:
        print("沒有違規事件")
        return
    for k, e in enumerate(events, start=1):
        print(f"{k}/{len(events)} {e['type']} → {render(k, e, args, cfg, summary.get('t0', 0.0), caches, zones_of, args.font)}",
              flush=True)


if __name__ == "__main__":
    main()
