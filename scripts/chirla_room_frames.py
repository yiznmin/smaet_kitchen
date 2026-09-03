"""從 CHIRLA 的 70 支影片各抽一張代表幀,並拼成「10 序列 × 7 相機」總覽。

為什麼需要這個(2026-09-03 踩到的坑):

專案文件從 9/1 起一路寫著「CHIRLA 是 7 台**非重疊**室內鏡頭」,而
`handoff/給遠端Claude.md` 據此推論「一切都走轉場路徑,所以地面校正與軌跡的
架構缺口會直接顯現」。**這個推論是錯的**,而錯誤只要把畫面調出來看一眼就會發現:
cam1/cam2/cam3 是相鄰空間、共用門口,人站在交界會同時進入多台畫面。
逐幀標註實測有 40% 的 (幀, 身份) 組合被 2 台以上同時看到。

所以這支腳本存在的理由是:**論文的一句規格描述不能取代看畫面**。
量化證據在 `scripts/chirla_overlap_stats.py`,這裡負責可視化。

代表幀的挑法:用 `annotations/` 找「該支影片裡標註人數最多」的那一幀 ——
挑固定時間點常常抽到空房間,看不出那台相機負責哪個空間。

輸出:
    <out>/frames/seq_XXX_cameraN.jpg     70 張全解析度單張
    <out>/CHIRLA_房間總覽.png             10 × 7 總覽(列=序列,欄=相機)

用法:
    python scripts/chirla_room_frames.py --root "D:/新增資料夾/CHIRLA/CHIRLA_data/CHIRLA"
"""
import argparse
import json
import sys
from pathlib import Path

# 中文標籤用的字型。找不到就退回 PIL 內建點陣字(英數可讀,中文會變方框)
_FONT_CANDIDATES = [
    r"C:\Windows\Fonts\msjh.ttc",        # 微軟正黑體
    r"C:\Windows\Fonts\msyh.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
]


def _font(size):
    from PIL import ImageFont
    for p in _FONT_CANDIDATES:
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def busiest_frame(ann_path):
    """回傳該支影片標註人數最多的幀號。沒有標註就回 None。"""
    d = json.loads(Path(ann_path).read_text(encoding="utf-8"))
    if not d:
        return None
    return int(max(d.items(), key=lambda kv: len(kv[1]))[0])


def grab(video, frame_idx):
    import cv2
    cap = cv2.VideoCapture(str(video))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, min(frame_idx if frame_idx is not None else n // 3,
                                         max(n - 1, 0)))
    ok, img = cap.read()
    cap.release()
    return img if ok else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="CHIRLA 根目錄(底下有 videos/ annotations/)")
    ap.add_argument("--out", default="results/m5_reid/chirla_samples")
    ap.add_argument("--tile-width", type=int, default=300)
    args = ap.parse_args()

    import cv2
    from PIL import Image, ImageDraw

    root = Path(args.root)
    vroot, aroot = root / "videos", root / "annotations"
    out = Path(args.out)
    (out / "frames").mkdir(parents=True, exist_ok=True)

    seqs = sorted(p.name for p in vroot.iterdir() if p.is_dir())
    cams = [f"camera_{i}" for i in range(1, 8)]
    print(f"序列 {len(seqs)} 個 × 相機 {len(cams)} 台")

    grid = {}
    for seq in seqs:
        for v in sorted((vroot / seq).glob("*.avi")):
            cam = "_".join(v.stem.split("_")[:2])
            a = aroot / seq / f"{v.stem}.json"
            fi = busiest_frame(a) if a.exists() else None
            img = grab(v, fi)
            if img is None:
                print(f"  讀不到 {seq}/{v.name}")
                continue
            # ⚠ 用 imencode+tofile 而非 imwrite —— 輸出路徑含中文時 imwrite 會靜默失敗
            dst = out / "frames" / f"{seq}_{cam}.jpg"
            ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 88])
            if ok:
                buf.tofile(str(dst))
            grid[(seq, cam)] = img
            print(f"  {seq}/{cam}  第 {fi} 幀 → {dst.name}")

    # ── 總覽:列 = 序列,欄 = 相機 ──
    TW = args.tile_width
    th = int(TW * 720 / 1080)
    left, top, pad = 78, 60, 4
    W = left + len(cams) * (TW + pad) + pad
    H = top + len(seqs) * (th + pad) + pad
    cv = Image.new("RGB", (W, H), (250, 250, 250))
    d = ImageDraw.Draw(cv)
    d.text((10, 8), "CHIRLA 房間總覽 —— 10 個序列 × 7 台相機(各取該片人最多的一幀)",
           fill=(15, 15, 15), font=_font(18))
    d.text((10, 32),
           "七台拍的是七個不同空間,不是同一個景;但 cam1/2/3 相鄰共用門口,"
           "實測 40% 的(幀,身份)被 2 台以上同時看到",
           fill=(105, 105, 105), font=_font(12))
    for c, cam in enumerate(cams):
        d.text((left + pad + c * (TW + pad), 46), cam, fill=(50, 50, 50), font=_font(13))
    for r, seq in enumerate(seqs):
        y = top + pad + r * (th + pad)
        d.text((8, y + th // 2 - 8), seq, fill=(70, 70, 70), font=_font(13))
        for c, cam in enumerate(cams):
            x = left + pad + c * (TW + pad)
            img = grid.get((seq, cam))
            if img is None:
                d.rectangle([x, y, x + TW, y + th], outline=(220, 220, 220))
                continue
            cv.paste(Image.fromarray(cv2.cvtColor(cv2.resize(img, (TW, th)),
                                                  cv2.COLOR_BGR2RGB)), (x, y))
            d.rectangle([x, y, x + TW, y + th], outline=(160, 160, 160))
    dst = out / "CHIRLA_房間總覽.png"
    cv.save(dst)
    print(f"\n總覽 → {dst}  {cv.size}")
    print(f"單張 → {out/'frames'}  共 {len(grid)} 張")
    return 0


if __name__ == "__main__":
    sys.exit(main())
