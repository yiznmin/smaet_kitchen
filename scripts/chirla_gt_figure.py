"""畫「同一個人在同一時刻、7 台相機的 ground-truth 框」。

**這張圖要回答的問題**:為什麼 CHIRLA 能量誤併率而 EPFL 不能?

EPFL 全片只有一個人 → 「把兩個人併成一個」結構上不可能發生,分母是零。
七輪實驗的誤併數字因此全部來自模擬。CHIRLA 的標註把**同一個 id 跨相機連好了**,
所以「系統說 cam2 這個人就是剛從 cam1 走掉的那位」這件事第一次有答案可以對。

圖上每一格是同一時刻的一台相機:
  · 黃框 = 這次要追蹤的那個身份
  · 灰框 = 同時在場的**其他**身份 —— 它們就是「可以被併錯的對象」,
    EPFL 沒有這些框,所以誤併量不到

用法:
    python scripts/chirla_gt_figure.py --root <CHIRLA根> --seq seq_020 --identity 1
"""
import argparse
import json
import sys
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
    ap.add_argument("--seq", default="seq_020")
    ap.add_argument("--identity", type=int, default=None,
                    help="要標成黃框的身份;不給則挑在最多台相機同時出現的那個")
    ap.add_argument("--out", default="results/m5_reid/chirla_samples/CHIRLA_跨相機GT.png")
    ap.add_argument("--tile-width", type=int, default=430)
    args = ap.parse_args()

    import cv2
    from PIL import Image, ImageDraw

    root = Path(args.root)
    ann, vid = {}, {}
    for f in sorted((root / "annotations" / args.seq).glob("*.json")):
        cam = phys(f.stem)
        ann[cam] = json.loads(f.read_text(encoding="utf-8"))
        vid[cam] = str(next((root / "videos" / args.seq).glob(cam + "_*.avi")))
    cams = sorted(ann)

    # 挑一個「目標身份同時出現在最多台相機」的幀 —— 那才看得出跨相機 GT 的價值
    best = None
    frames = set()
    for d in ann.values():
        frames |= set(d)
    for fr in sorted(frames, key=int):
        present = {c: [o for o in ann[c].get(fr, [])] for c in cams}
        ids_here = {abs(int(o["id"])) for v in present.values() for o in v}
        cand = [args.identity] if args.identity else sorted(ids_here)
        for pid in cand:
            n = sum(1 for c in cams
                    if any(abs(int(o["id"])) == pid for o in present[c]))
            if best is None or n > best[0]:
                best = (n, int(fr), pid, present)
    n_cam, fr, pid, present = best
    n_others = len({abs(int(o["id"])) for v in present.values() for o in v}) - 1
    print(f"  {args.seq} 第 {fr} 幀:id {pid} 同時出現在 {n_cam} 台相機,"
          f"另有 {n_others} 個其他身份在場")

    TW = args.tile_width
    th = int(TW * 720 / 1080)
    cols = 4
    rows = (len(cams) + cols - 1) // cols
    cv = Image.new("RGB", (cols * (TW + 8) + 8, 64 + rows * (th + 26) + 8), (250, 250, 250))
    d = ImageDraw.Draw(cv)
    d.text((10, 6), f"CHIRLA 的跨相機 ground truth —— {args.seq} 第 {fr} 幀(同一時刻)",
           fill=(15, 15, 15), font=_font(18))
    d.text((10, 30), f"黃框 = id {pid}(同一個人,7 台相機共用同一個 id)　"
                     f"灰框 = 同時在場的其他 {n_others} 個身份 = 可以被併錯的對象",
           fill=(105, 105, 105), font=_font(13))
    d.text((10, 48), "EPFL 全片單人 → 沒有灰框 → 誤併結構上量不到。這就是 CHIRLA 解掉的事。",
           fill=(150, 60, 60), font=_font(13))

    for i, cam in enumerate(cams):
        r, c = divmod(i, cols)
        x, y = 8 + c * (TW + 8), 64 + r * (th + 26)
        cap = cv2.VideoCapture(vid[cam])
        cap.set(cv2.CAP_PROP_POS_FRAMES, fr - 1)      # ⚠ 標註 1-based,影片 0-based
        ok, img = cap.read()
        cap.release()
        if not ok:
            continue
        hit = False
        for o in present[cam]:
            x1, y1, x2, y2 = map(int, o["BboxP"])
            if abs(int(o["id"])) == pid:
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 255), 5)
                cv2.putText(img, f"id {pid}", (x1, max(24, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
                hit = True
            else:
                cv2.rectangle(img, (x1, y1), (x2, y2), (170, 170, 170), 2)
        cv.paste(Image.fromarray(cv2.cvtColor(cv2.resize(img, (TW, th)),
                                              cv2.COLOR_BGR2RGB)), (x, y + 18))
        d.text((x, y + 2), f"{cam}" + ("" if hit else "   (這台沒拍到 id %d)" % pid),
               fill=(40, 40, 40) if hit else (150, 90, 90), font=_font(13))
        d.rectangle([x, y + 18, x + TW, y + 18 + th], outline=(150, 150, 150))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv.save(out)
    print(f"  → {out}  {cv.size}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
