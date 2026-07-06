"""
在 EPFL 真畫面上實跑 M3(RF-DETR),把偵測到的人/物件框出來。
用途:直觀看「M3 在你的鏡頭視角上偵測得到什麼」(人、瓶子…),
     以及小物件(刀)在俯視遠距下抓不抓得到。

注意:用 COCO 預訓練的 RF-DETR(未微調),所以小餐具會偏弱(見 M3 決策文件)。

用法:
  python scripts/m3_on_frame.py                       # 預設幾個幀 + nano
  python scripts/m3_on_frame.py --variant medium --frames 300 400 500
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from common.video_io import iter_frames, imwrite_unicode   # noqa: E402

VIDEO = ROOT / "data" / "epfl" / "Boutput0.mp4"
OUT = ROOT / "results" / "m3_acc" / "epfl_vis"


def load_model(variant):
    if variant == "nano":
        from rfdetr import RFDETRNano as M
    elif variant == "small":
        from rfdetr import RFDETRSmall as M
    else:
        from rfdetr import RFDETRMedium as M
    return M()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="nano", choices=["nano", "small", "medium"])
    ap.add_argument("--frames", type=int, nargs="*", default=[300, 400, 500, 600])
    ap.add_argument("--thr", type=float, default=0.3)
    args = ap.parse_args()

    want = set(args.frames)
    frames = {}
    for fid, ts, fr in iter_frames(VIDEO):
        if fid in want:
            frames[fid] = fr
        if len(frames) == len(want):
            break

    print(f"載入 RF-DETR-{args.variant} …(首次會下載 COCO 權重)")
    model = load_model(args.variant)
    OUT.mkdir(parents=True, exist_ok=True)

    saved = []
    for fid in args.frames:
        if fid not in frames:
            continue
        bgr = frames[fid]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        det = model.predict(Image.fromarray(rgb), threshold=args.thr)
        names = det.data.get("class_name") if det.data else None

        vis = bgr.copy()
        counts = {}
        for i in range(len(det)):
            x1, y1, x2, y2 = [int(v) for v in det.xyxy[i]]
            nm = str(names[i]) if names is not None else str(int(det.class_id[i]))
            cf = float(det.confidence[i]) if det.confidence is not None else 0
            counts[nm] = counts.get(nm, 0) + 1
            color = (0, 220, 0) if nm == "person" else (0, 165, 255)
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            cv2.putText(vis, f"{nm} {cf:.2f}", (x1, max(14, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        summ = ", ".join(f"{k}x{v}" for k, v in sorted(counts.items()))
        cv2.putText(vis, f"RF-DETR-{args.variant} thr={args.thr}: {summ}", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        fp = OUT / f"m3_{args.variant}_f{fid}.png"
        if imwrite_unicode(fp, vis):
            saved.append(fp.relative_to(ROOT).as_posix())
        print(f"  f{fid}: {len(det)} 偵測 → {summ or '(無)'}")
    print(f"完成,存 {len(saved)} 張:{saved}")


if __name__ == "__main__":
    main()
