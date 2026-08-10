"""在畫面上實跑 M3(RF-DETR)並畫框。支援兩種模型與每類信心門檻。

模式:
  - 未給 --weights:用 COCO 預訓練 RF-DETR(未微調,英文類別;小餐具偏弱)。
  - 給 --weights:載入我們微調的 11 類模型(中文類別;畫框用 ASCII 代號,console 印中文)。
  - 給 --thresholds:先低門檻 predict 再「每類各自把關」過濾(configs/m3_thresholds.json)。

輸入來源:
  - 預設:從 EPFL 影片抽指定 --frames。
  - --images a.jpg b.jpg …:直接對這些圖跑(例如 test 集)。

用法:
  python scripts/m3_on_frame.py                                    # COCO 模型 + 影片幀
  python scripts/m3_on_frame.py --weights out_final/checkpoint_best_regular.pth \\
      --images data/m3_finetune_r2/test/*.jpg --thresholds configs/m3_thresholds.json
"""
import argparse
import glob
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from common.video_io import iter_frames, imwrite_unicode   # noqa: E402
from m3.classes import NAMES, ROMAN, FOOD_SAFETY           # noqa: E402

VIDEO = ROOT / "data" / "epfl" / "Boutput0.mp4"
OUT = ROOT / "results" / "m3_acc" / "epfl_vis"


def load_model(variant, weights):
    from rfdetr import RFDETRNano, RFDETRSmall, RFDETRMedium
    ctor = {"nano": RFDETRNano, "small": RFDETRSmall, "medium": RFDETRMedium}[variant]
    if weights:
        return ctor(pretrain_weights=weights, num_classes=len(NAMES))
    return ctor()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="nano", choices=["nano", "small", "medium"])
    ap.add_argument("--weights", help="微調權重 .pth(不給=用 COCO 預訓練)")
    ap.add_argument("--thresholds", help="每類門檻 json(configs/m3_thresholds.json)")
    ap.add_argument("--frames", type=int, nargs="*", default=[300, 400, 500, 600])
    ap.add_argument("--images", nargs="*", help="改對這些圖片跑(可用萬用字元)")
    ap.add_argument("--thr", type=float, default=0.3, help="全域/後備門檻")
    args = ap.parse_args()

    per_class = None
    if args.thresholds:
        per_class = json.loads(Path(args.thresholds).read_text(encoding="utf-8"))
        print(f"套用每類門檻:{args.thresholds}")

    # 收集要跑的影像 {key: bgr}
    items = {}
    if args.images:
        paths = []
        for pat in args.images:
            paths.extend(glob.glob(pat) or [pat])
        for p in paths:
            bgr = cv2.imdecode(np.fromfile(p, dtype=np.uint8), cv2.IMREAD_COLOR)
            if bgr is not None:
                items[Path(p).stem] = bgr
    else:
        want = set(args.frames)
        for fid, ts, fr in iter_frames(VIDEO):
            if fid in want:
                items[f"f{fid}"] = fr
            if len(items) == len(want):
                break

    print(f"載入 RF-DETR-{args.variant}{'(微調)' if args.weights else '(COCO)'} …")
    model = load_model(args.variant, args.weights)
    finetuned = bool(args.weights)
    base_thr = 0.1 if per_class else args.thr
    OUT.mkdir(parents=True, exist_ok=True)

    saved = []
    for key, bgr in items.items():
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        det = model.predict(Image.fromarray(rgb), threshold=base_thr)
        coco_names = det.data.get("class_name") if (not finetuned and det.data) else None

        vis = bgr.copy()
        counts = {}
        for i in range(len(det)):
            cid = int(det.class_id[i])
            cf = float(det.confidence[i]) if det.confidence is not None else 0.0
            zh = NAMES.get(cid, str(cid)) if finetuned else (str(coco_names[i]) if coco_names else str(cid))
            # 每類門檻過濾(僅微調模型 + 有門檻表時)
            if per_class is not None and finetuned:
                if cf < per_class.get(zh, args.thr):
                    continue
            label = ROMAN.get(cid, str(cid)) if finetuned else zh
            counts[zh] = counts.get(zh, 0) + 1
            if zh == "person" or zh == "人":
                color = (0, 220, 0)
            elif finetuned and zh in FOOD_SAFETY:
                color = (0, 0, 255)          # 食安類 = 紅
            else:
                color = (0, 165, 255)
            x1, y1, x2, y2 = [int(v) for v in det.xyxy[i]]
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            cv2.putText(vis, f"{label} {cf:.2f}", (x1, max(14, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        summ = ", ".join(f"{k}x{v}" for k, v in sorted(counts.items()))
        tag = "FT" if finetuned else "COCO"
        thr_tag = "per-class" if per_class else f"thr={args.thr}"
        cv2.putText(vis, f"RF-DETR-{args.variant} {tag} {thr_tag}", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        fp = OUT / f"m3_{args.variant}_{'ft' if finetuned else 'coco'}_{key}.png"
        if imwrite_unicode(fp, vis):
            saved.append(fp.relative_to(ROOT).as_posix())
        print(f"  {key}: {sum(counts.values())} 偵測 → {summ or '(無)'}")
    print(f"完成,存 {len(saved)} 張 → {OUT.relative_to(ROOT).as_posix()}/")


if __name__ == "__main__":
    main()
