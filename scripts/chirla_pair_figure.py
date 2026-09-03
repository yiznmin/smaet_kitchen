"""畫「相機兩兩對照」圖:同一幀、同一個人,同時出現在兩台畫面裡。

配合 `scripts/chirla_overlap_stats.py` 使用 —— 那支給數字,這支給畫面。
2026-09-03 的教訓是**數字會騙人**:cam1+cam3 共現 11.6 萬次看起來像同一個房間,
把畫面調出來才知道那是隔著一道門互看,cam1 拍到的只是門口的一小塊。
所以每一對都要有一張圖佐證,不能只看共現次數。

⚠ 找不到共現樣本時要標明「在這個序列裡」——某兩台在 seq_020 沒共現,
  不代表整個資料集都沒有(cam5+cam6 全資料集有 738 次,seq_020 是 0 次)。

用法:
    python scripts/chirla_pair_figure.py --root <CHIRLA根> \
        --pairs camera_2+camera_3 camera_1+camera_3 camera_5+camera_6 camera_4+camera_7
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

_FONTS = [r"C:\Windows\Fonts\msjh.ttc", r"C:\Windows\Fonts\msyh.ttc",
          "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc"]


def _font(size):
    from PIL import ImageFont
    for p in _FONTS:
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def phys(n):
    return "_".join(n.split("_")[:2])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--seq", default=None, help="用哪個序列(預設挑標註最多的)")
    ap.add_argument("--pairs", nargs="+", required=True, help="camera_2+camera_3 ...")
    ap.add_argument("--stats", default="results/m5_reid/chirla_overlap.json",
                    help="chirla_overlap_stats.py 的輸出,用來標全資料集的共現次數")
    ap.add_argument("--out", default="results/m5_reid/chirla_samples/CHIRLA_相機兩兩對照.png")
    ap.add_argument("--tile-width", type=int, default=520)
    args = ap.parse_args()

    import cv2
    from PIL import Image, ImageDraw

    root = Path(args.root)
    stats = {}
    if Path(args.stats).exists():
        stats = json.loads(Path(args.stats).read_text(encoding="utf-8")).get("pairs", {})

    seq = args.seq or max((p.name for p in (root / "annotations").iterdir() if p.is_dir()),
                          key=lambda s: sum(f.stat().st_size
                                            for f in (root / "annotations" / s).glob("*.json")))
    ann, vid = {}, {}
    for f in sorted((root / "annotations" / seq).glob("*.json")):
        cam = phys(f.stem)
        ann[cam] = json.loads(f.read_text(encoding="utf-8"))
        vid[cam] = str(next((root / "videos" / seq).glob(cam + "_*.avi")))
    seen = defaultdict(dict)
    for cam, d in ann.items():
        for fr, dets in d.items():
            for o in dets:
                seen[(int(fr), int(o["id"]))][cam] = o["BboxP"]

    def grab(cam, fr):
        cap = cv2.VideoCapture(vid[cam])
        cap.set(cv2.CAP_PROP_POS_FRAMES, fr)
        ok, img = cap.read()
        cap.release()
        return img if ok else None

    # 每一對挑「兩邊框都最大」的那一刻,才看得出是不是同一個空間
    picked = []
    for spec in args.pairs:
        a, b = spec.split("+")
        best = None
        for (fr, pid), v in seen.items():
            if a in v and b in v:
                sa = (v[a][2] - v[a][0]) * (v[a][3] - v[a][1])
                sb = (v[b][2] - v[b][0]) * (v[b][3] - v[b][1])
                score = min(sa, sb)
                if best is None or score > best[0]:
                    best = (score, fr, pid, v[a], v[b])
        picked.append(((a, b), best))

    TW = args.tile_width
    th = int(TW * 720 / 1080)
    left, top = 66, 62
    cv = Image.new("RGB", (left + 2 * (TW + 8) + 8, top + len(picked) * (th + 34) + 8),
                   (250, 250, 250))
    d = ImageDraw.Draw(cv)
    d.text((10, 6), f"CHIRLA 相機兩兩對照 —— 同一幀、同一個人(黃框)· {seq}",
           fill=(15, 15, 15), font=_font(18))
    d.text((10, 32), "判斷哪幾台其實在拍同一個空間。兩邊的框都大 = 同一空間;"
                     "一邊只有細長條 = 隔著門口互看",
           fill=(105, 105, 105), font=_font(12))
    for r, ((a, b), best) in enumerate(picked):
        y = top + r * (th + 34)
        st = stats.get(f"{a}+{b}") or stats.get(f"{b}+{a}") or {}
        tag = (f"全資料集共現 {st['n']:,} 次 · 兩邊都清楚 {st['pct_both_good']}%"
               f" → {st['verdict']}") if st else ""
        d.text((6, y + th // 2), f"{a[-1]}+{b[-1]}", fill=(70, 70, 70), font=_font(14))
        if best is None:
            d.rectangle([left, y + 26, left + 2 * (TW + 8), y + 26 + th], outline=(222, 222, 222))
            d.text((left + 14, y + 26 + th // 2 - 20),
                   f"這兩台在 {seq} 裡從未同時看到同一個人", fill=(180, 60, 60), font=_font(16))
            d.text((left + 14, y + 26 + th // 2 + 4), tag, fill=(120, 120, 120), font=_font(13))
            continue
        _s, fr, pid, ba, bb = best
        d.text((left, y + 6), tag, fill=(90, 90, 90), font=_font(12))
        for c, (cam, box) in enumerate(((a, ba), (b, bb))):
            img = grab(cam, fr)
            cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 255, 255), 4)
            x = left + c * (TW + 8)
            cv.paste(Image.fromarray(cv2.cvtColor(cv2.resize(img, (TW, th)),
                                                  cv2.COLOR_BGR2RGB)), (x, y + 26))
            d.text((x, y + 26 + th + 2),
                   f"{cam} · 第 {fr} 幀 · id {pid} · 框寬 {box[2]-box[0]}px",
                   fill=(40, 40, 40), font=_font(12))
            d.rectangle([x, y + 26, x + TW, y + 26 + th], outline=(150, 150, 150))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv.save(out)
    print(f"→ {out}  {cv.size}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
