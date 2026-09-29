"""把同一片段在多個設定下的檢視影片橫向並排,每格加標題。

只做重放與拼接,不重跑推論、不改任何指標 —— 輸入就是 `render_clip_video.py` 的產物。

⚠ 與那支同樣是**內部檢視用**:每格畫面自己已經有「不可交付」的字幕條。

用法:
    python scripts/stack_clip_compare.py --clips L2-xxx L2-yyy \
        --cells "w=0 外觀關閉=results/l2_app_probe/videos/clip_app_w0.0" \
                "armS0 w=0.2=results/l2_app_probe/videos/clip_app_armS0_w0.2" \
        --out-dir results/l2_app_probe/videos/compare
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from render_level_video import Encoder, caption          # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", nargs="+", required=True)
    ap.add_argument("--cells", nargs="+", required=True,
                    help='每格一項,格式 "標題=影片目錄"(目錄下找 <clip>_s<stride>.mp4)。'
                         '⚠ 用**最後一個** = 當分隔符 —— 標題本身常含 = (如 "w=0")')
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--crf", type=int, default=23)
    ap.add_argument("--font", default="C:/Windows/Fonts/msjh.ttc")
    ap.add_argument("--encoder", choices=("ffmpeg", "cv2"), default="ffmpeg")
    args = ap.parse_args()

    cells = [c.rsplit("=", 1) for c in args.cells]   # 標題可含 =,所以從右切
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    n_ok = 0
    for cid in args.clips:
        paths = [Path(d) / f"{cid}_s{args.stride}.mp4" for _, d in cells]
        missing = [p for p in paths if not p.exists()]
        if missing:
            print(f"  - {cid}:缺 {len(missing)} 支({missing[0].name}),跳過")
            continue
        caps = [cv2.VideoCapture(str(p)) for p in paths]
        fps = caps[0].get(cv2.CAP_PROP_FPS) or 6.0
        # 標題條只畫一次,逐幀貼上(與 render_level_video.caption 同一條路)
        bars, enc, size = None, None, None
        n = 0
        while True:
            frames = []
            for cap in caps:
                ok, img = cap.read()
                if not ok:
                    frames = None
                    break
                frames.append(img)
            if frames is None:
                break
            h = min(f.shape[0] for f in frames)
            frames = [f if f.shape[0] == h else
                      cv2.resize(f, (int(f.shape[1] * h / f.shape[0]), h)) for f in frames]
            if bars is None:
                bars = [caption(t, f.shape[1], args.font, height=30)
                        for (t, _), f in zip(cells, frames)]
            panels = [np.vstack([b, f]) for b, f in zip(bars, frames)]
            row = np.hstack(panels)
            if enc is None:
                # H.264 要偶數尺寸
                size = (row.shape[1] - row.shape[1] % 2, row.shape[0] - row.shape[0] % 2)
                enc = Encoder(out_dir / f"{cid}_{len(cells)}up.mp4", fps, size,
                              args.crf, args.encoder)
            enc.write(np.ascontiguousarray(row[:size[1], :size[0]]))
            n += 1
        for cap in caps:
            cap.release()
        if enc is not None:
            enc.close()
            n_ok += 1
            print(f"  {cid}:{n} 幀 × {len(cells)} 格 → {out_dir / f'{cid}_{len(cells)}up.mp4'}")
    print(f"\n完成 {n_ok} / {len(args.clips)} 支並排影片")


if __name__ == "__main__":
    main()
