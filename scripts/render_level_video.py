r"""把一個難度窗渲染成結果影片:每一幀都有偵測框與 re-ID 標籤。

## ⚠ 重放,不重跑

輸入是 `tracks.csv` —— 它已經逐幀存了每個框與它的 `chef_id`
(`loop_i, video_fid, t_sec, camera_id, track_id, x1..y2, conf, hits, start_frame, chef_id`)。
所以這支**不載入 RF-DETR、不跑追蹤、不跑 M5**,只解碼影片並把已經發生過的結果畫上去。
這保證影片與指標來自**同一次執行**;若在這裡重跑管線,兩者可能分歧而沒有人會發現。

繪圖走 `src/common/draw_tracks.py`,與線上 runner 同一份實作
(同一位 chef 在每台鏡頭同色 —— L3 的影片要看得懂就靠這個)。

## ⚠ 三個踩過的坑

1. **fps 由 stride 決定**,不是 30。`m4_track_video.py` 寫死 30 fps 而它接受 `--stride`,
   於是 stride 2 的影片以兩倍速播放。這裡一律 `fps_src / stride`(stride 5 → 6 fps),
   **不補幀、不插值** —— 假的流暢會讓人誤判追蹤品質。
2. **中文路徑**:`cv2.VideoWriter` 與 ffmpeg 子行程在 `D:\產學合作\...` 都可能失敗。
   一律寫 ASCII 暫存再 `shutil.move`(`make_mouse_video.py` 的成例)。
3. **不要用 CAP_PROP_POS_FRAMES 跳轉**:這些影片跳轉不可靠,幀號一偏整支影片的標註就錯位。
   窗前用 `grab()` 快轉、窗內才 `retrieve()` —— 精確且便宜。

## 交付版只顯示系統知道的東西

`--gt-mode none`(預設)只畫系統的判斷。`label` 會畫真值與 ✓✗,**僅供內部**:
畫面上的 ✓✗ 是**逐幀**的,而誤併率是**逐次綁定**的,兩者分母不同,
並排放給出資方看會被讀成錯誤率。

用法:
    python scripts/render_level_video.py --manifest results/levels/level_manifest.json \\
        --run-root results/m5_step3/m5_cbiou --all --rendered-only \\
        --out-dir results/levels/videos
"""
import argparse
import csv
import json
import shutil
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from common.draw_tracks import DrawTrack, draw_panel, stitch    # noqa: E402


def load_tracks(run_dir, cams, lo, hi):
    """回傳 {video_fid: {cam: [DrawTrack]}} 與 {(cam, track_id): chef_id}。

    ⚠ 2026-09-23 修正:鍵改用 `video_fid`,不用 `loop_i`。舊版假設
      `video_fid == loop_i × stride`,這對全長執行成立,但對**片段執行**
      (`--start-fid` 非 0)是 `start_fid + loop_i × stride` → 每個框會畫到
      錯誤的幀上,而且不會報任何錯。全長執行下兩者等價,行為不變(V8b)。
    """
    per_loop = defaultdict(lambda: defaultdict(list))
    chefs, fids = {}, defaultdict(set)
    with open(Path(run_dir) / "tracks.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            cam, fid = r["camera_id"], int(r["video_fid"])
            if cam not in cams or not (lo <= fid <= hi):
                continue
            loop = fid
            fids[cam].add(fid)
            box = None
            if r["x1"] != "":
                box = (float(r["x1"]), float(r["y1"]), float(r["x2"]), float(r["y2"]))
            tid = int(r["track_id"])
            per_loop[loop][cam].append(DrawTrack(tid, box))
            if r.get("chef_id"):
                chefs[(cam, tid)] = int(r["chef_id"])
    return per_loop, chefs, fids


# 字型缺字會畫成豆腐方塊(微軟正黑體沒有 ⚠)。畫面上出現 □ 看起來像程式壞了,
# 而且交付影片不該有那種東西 —— 已知會缺的符號先換成 ASCII。
_CAPTION_SUBST = {"⚠": "[!]", "✓": "[v]", "✗": "[x]", "·": "-", "—": "-", "→": "->"}


def caption(text, width, font_path, height=34):
    """中文字幕條。⚠ cv2.putText 畫不了中文,所以用 PIL 畫一次再逐幀貼上。

    兩個實際踩到的坑(2026-09-16 看 EPFL 排練影片時發現):
      1. **缺字變豆腐**:見上面的替換表。
      2. **文字太長被右邊切掉**:640 寬的單鏡頭面板放不下完整說明,而被切掉的
         往往正是授權與「不可交付」那半句 —— 那是最不能掉的。所以自動縮字級到放得下。

    找不到字型就**明確失敗**,不要默默退回 ASCII —— 交付影片少了出處說明,
    被轉寄出去之後沒有人知道它是什麼。
    """
    from PIL import Image, ImageDraw, ImageFont

    p = Path(font_path)
    if not p.exists():
        raise SystemExit(f"找不到中文字型 {p} —— 請用 --font 指定(例:C:/Windows/Fonts/msjh.ttc)")
    for a, b in _CAPTION_SUBST.items():
        text = text.replace(a, b)
    img = Image.new("RGB", (width, height), (18, 18, 18))
    d = ImageDraw.Draw(img)
    size, font = 18, None
    while size >= 10:
        font = ImageFont.truetype(str(p), size)
        if d.textlength(text, font=font) <= width - 20:
            break
        size -= 1
    d.text((10, max(0, (height - size) // 2 - 1)), text, font=font, fill=(235, 235, 235))
    return np.array(img)[:, :, ::-1].copy()          # RGB → BGR


def even(img):
    """libx264 + yuv420p 不吃奇數寬高 —— 補一列/一行黑,而不是讓 ffmpeg 在最後才炸。"""
    h, w = img.shape[:2]
    if h % 2 or w % 2:
        img = np.pad(img, ((0, h % 2), (0, w % 2), (0, 0)), constant_values=20)
    return img


class Encoder:
    """ffmpeg 優先(H.264,簡報與瀏覽器都能播);沒有就退回 cv2 的 mp4v。"""

    def __init__(self, path, fps, size, crf, kind):
        self.final = Path(path)
        self.tmp = Path(tempfile.gettempdir()) / f"lvlvid_{self.final.stem}.mp4"
        self.kind, self.proc, self.vw = kind, None, None
        w, h = size
        if kind == "ffmpeg":
            self.proc = subprocess.Popen(
                ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
                 "-s", f"{w}x{h}", "-r", f"{fps}", "-i", "-", "-c:v", "libx264",
                 # ⚠ yuv420p 不能省:從 bgr24 進來 x264 會挑 yuv444p,
                 #   而 PowerPoint 與 Windows 內建播放器開不起來 —— 會在會議上才發現。
                 "-pix_fmt", "yuv420p", "-crf", str(crf), "-movflags", "+faststart",
                 str(self.tmp)], stdin=subprocess.PIPE)
        else:
            import cv2
            self.vw = cv2.VideoWriter(str(self.tmp), cv2.VideoWriter_fourcc(*"mp4v"),
                                      fps, (w, h))
            if not self.vw.isOpened():
                raise SystemExit("cv2.VideoWriter 開不起來")

    def write(self, frame):
        if self.proc:
            self.proc.stdin.write(frame.tobytes())
        else:
            self.vw.write(frame)

    def close(self):
        if self.proc:
            self.proc.stdin.close()
            if self.proc.wait() != 0:
                raise SystemExit("ffmpeg 編碼失敗")
        else:
            self.vw.release()
        self.final.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(self.tmp), str(self.final))   # ⚠ ASCII 暫存 → 中文路徑


def probe(path):
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=codec_name,nb_frames", "-of", "json", str(path)],
            capture_output=True, text=True)
        s = json.loads(out.stdout)["streams"][0]
        return s.get("codec_name"), s.get("nb_frames")
    except Exception:
        return None, None


def render(win, run_root, out_dir, args):
    import cv2

    seq, cams = win["seq"], win["cameras"]
    run_dir = Path(run_root) / seq if (Path(run_root) / seq).exists() else Path(run_root)
    meta = json.loads((run_dir / "run_meta.json").read_text(encoding="utf-8"))
    stride = int(meta["stride"])
    vmap = meta["camera_video_map"]
    missing = [c for c in cams if c not in vmap]
    if missing:
        raise SystemExit(f"{win['window_id']}:run_meta 沒有這些鏡頭 {missing}")

    lo, hi = win["start_fid"], win["end_fid"]
    per_loop, chefs, fids = load_tracks(run_dir, set(cams), lo, hi)
    if not per_loop:
        raise SystemExit(f"{win['window_id']}:窗內沒有任何 track —— run-root 與序列對不上?")
    # 自檢:tracks.csv 的實際取樣間距必須等於 run_meta 宣稱的 stride
    for cam, s in fids.items():
        gaps = {b - a for a, b in zip(sorted(s), sorted(s)[1:])}
        if gaps and min(gaps) != stride:
            raise SystemExit(f"{cam} 的 fid 間距 {sorted(gaps)[:3]} 與 stride={stride} 不符")

    fps_src = float(meta["videos"][cams[0]]["fps"])
    fps_out = fps_src / stride if args.fps_mode == "realtime" else fps_src
    caps = {}
    for c in cams:
        src = vmap[c]
        # 資料集搬家後 run_meta 仍是舊路徑(2026-09-17 CHIRLA 搬到 D:/yizhen/CHIRLA)。
        # 只換前綴、不改 run_meta —— 已提交的執行結果保持原樣
        for old, new in args.video_root_remap:
            if src.startswith(old):
                src = new + src[len(old):]
                break
        p = Path(src)
        p = p if p.is_absolute() else ROOT / p
        cap = cv2.VideoCapture(str(p))
        if not cap.isOpened():
            raise SystemExit(f"開不了影片 {p}")
        caps[c] = [cap, 0]                       # [cap, 下一個要讀的 fid]

    # ⚠ 2026-09-17 修正:迭代窗內**每個取樣幀**,不是只迭代 tracks.csv 有列的迴圈。
    #   人離開畫面時選定鏡頭上沒有 track → 舊版整段跳過,影片從離開前直接跳到回來後,
    #   正好剪掉 L2 的消失期間與 L3T 的轉場(V3 幀數檢查抓到 8/17 段)。
    #   已驗證 10 個序列的 tracks.csv 全部 video_fid == loop_i × stride。
    # ⚠ 2026-09-23:直接迭代 video_fid(見 load_tracks 的註解),片段執行才不會錯位。
    loops = list(range(lo, hi + 1, stride))
    if lo % stride:
        raise SystemExit(f"{win['window_id']}:start_fid={lo} 不在 stride={stride} 的格點上")
    enc, n_box, n_chef, size = None, 0, 0, None
    strip = None
    for loop in loops:
        panels = []
        for c in cams:
            cap, nxt = caps[c]
            fid = loop
            while nxt < fid:                     # ⚠ grab 快轉,不用 POS_FRAMES 跳轉
                if not cap.grab():
                    break
                nxt += 1
            ok, frame = cap.read()
            caps[c][1] = nxt + 1
            if not ok:
                frame = np.full((int(meta["videos"][c]["height"]),
                                 int(meta["videos"][c]["width"]), 3), 20, np.uint8)
                cv2.putText(frame, f"{c} no signal", (30, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (60, 60, 200), 3)
            trks = per_loop.get(loop, {}).get(c, [])
            n_box += sum(t.bbox is not None for t in trks)
            n_chef += sum((c, t.track_id) in chefs for t in trks)
            panels.append(draw_panel(frame, trks, c, chefs, fid / fps_src, width=args.width))
        canvas = even(stitch(panels))
        if strip is None:
            # ⚠ 授權與資料來源由 --label 帶進來,**不可寫死** ——
            #   寫死的話用別的資料集渲染時,畫面上會掛著錯誤的授權宣告,
            #   而影片一旦被轉寄出去,看的人只會相信畫面上那一行。
            txt = (f"{win['level']} {win['window_id']}  |  {seq} {'+'.join(cams)}  |  "
                   f"{args.label}  |  "
                   f"stride={stride} → {fps_out:g} fps"
                   + ("(實時)" if args.fps_mode == "realtime" else f"({stride}倍速)"))
            strip = caption(txt, canvas.shape[1], args.font)
        canvas = even(np.vstack([canvas, strip]))
        if enc is None:
            size = (canvas.shape[1], canvas.shape[0])
            enc = Encoder(out_dir / f"{win['window_id']}.mp4", fps_out, size,
                          args.crf, args.encoder)
        enc.write(canvas)
    for cap, _ in caps.values():
        cap.release()
    enc.close()

    path = out_dir / f"{win['window_id']}.mp4"
    n = len(loops)
    if n_box == 0:
        raise SystemExit(f"{win['window_id']}:一個框都沒畫到")
    if path.stat().st_size < 10_000:
        raise SystemExit(f"{win['window_id']}:輸出只有 {path.stat().st_size} bytes,編碼失敗")
    codec, _ = probe(path)
    if args.encoder == "ffmpeg" and codec and codec != "h264":
        raise SystemExit(f"{win['window_id']}:codec 是 {codec},不是 h264")
    warn = "" if n_chef else "  ⚠ 每個框都沒有 chef_id"
    print(f"  {win['window_id']:<46}{n:>5} 幀  {fps_out:g} fps  "
          f"{size[0]}x{size[1]}  {path.stat().st_size / 1e6:.2f} MB{warn}")
    return dict(window_id=win["window_id"], frames=n, fps=fps_out,
                size=list(size), boxes=n_box, with_chef=n_chef,
                bytes=path.stat().st_size, codec=codec, path=str(path))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--run-root", required=True, help="含 <seq>/tracks.csv 的目錄")
    ap.add_argument("--out-dir", default=str(ROOT / "results" / "levels" / "videos"))
    ap.add_argument("--window-id", nargs="+", default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--rendered-only", action="store_true",
                    help="只渲染 manifest 標記 rendered 的窗(排序鍵決定,不看結果好壞)")
    ap.add_argument("--level", nargs="+", default=None)
    ap.add_argument("--width", type=int, default=640, help="每台鏡頭的面板寬度")
    ap.add_argument("--crf", type=int, default=23)
    ap.add_argument("--fps-mode", choices=("realtime", "per-sample"), default="realtime")
    ap.add_argument("--gt-mode", choices=("none",), default="none",
                    help="交付版只顯示系統知道的東西(box / label 待內部版再加)")
    ap.add_argument("--encoder", choices=("ffmpeg", "cv2"), default="ffmpeg")
    ap.add_argument("--font", default="C:/Windows/Fonts/msjh.ttc")
    ap.add_argument("--label", default="m5_cbiou", help="字幕條上的執行標籤")
    ap.add_argument("--out", default=None, help="渲染摘要 json")
    ap.add_argument("--video-root-remap", nargs="+", default=[], metavar="舊前綴=新前綴",
                    help="run_meta 影片路徑的前綴替換;不給時行為不變")
    args = ap.parse_args()
    args.video_root_remap = [tuple(x.split("=", 1)) for x in args.video_root_remap]

    man = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    wins = man["windows"]
    if args.rendered_only:
        wins = [w for w in wins if w["rendered"]]
    if args.level:
        wins = [w for w in wins if w["level"] in args.level]
    if args.window_id:
        wins = [w for w in wins if w["window_id"] in args.window_id]
    if not args.all and not args.window_id:
        raise SystemExit("要 --all 或 --window-id")
    if not wins:
        raise SystemExit("沒有符合條件的窗")

    out_dir = Path(args.out_dir)
    print(f"渲染 {len(wins)} 段(編碼器 {args.encoder})")
    rows = [render(w, args.run_root, out_dir, args) for w in wins]
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(rows, ensure_ascii=False, indent=2),
                                  encoding="utf-8")
    print(f"共 {sum(r['bytes'] for r in rows) / 1e6:.1f} MB → {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
