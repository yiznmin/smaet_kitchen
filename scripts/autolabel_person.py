"""
用 RF-DETR 自動標註 person(它抓人很準),建立 COCO 骨架。
knife 之後手動補(RF-DETR 抓不到小刀)。

輸出:data/m3_finetune/_draft.json(COCO 格式,含 person 標註;knife 之後 merge)
用法:python scripts/autolabel_person.py
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

CATEGORIES = [{"id": 1, "name": "person"}, {"id": 2, "name": "knife"}]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img_dir", default=str(ROOT / "data" / "m3_finetune" / "images"))
    ap.add_argument("--out", default=str(ROOT / "data" / "m3_finetune" / "_draft.json"))
    args = ap.parse_args()
    IMG_DIR = Path(args.img_dir)
    OUT = Path(args.out).resolve()

    from rfdetr import RFDETRNano
    model = RFDETRNano()

    imgs = sorted(IMG_DIR.glob("*.jpg"))
    print(f"對 {len(imgs)} 張自動標 person …")

    coco = {"images": [], "annotations": [], "categories": CATEGORIES}
    ann_id = 1
    for img_id, p in enumerate(imgs, 1):
        bgr = cv2.imdecode(
            __import__("numpy").fromfile(str(p), dtype="uint8"), cv2.IMREAD_COLOR)
        h, w = bgr.shape[:2]
        coco["images"].append({"id": img_id, "file_name": p.name, "width": w, "height": h})
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        det = model.predict(Image.fromarray(rgb), threshold=0.5)
        names = det.data.get("class_name") if det.data else None
        n_person = 0
        for i in range(len(det)):
            nm = str(names[i]) if names is not None else ""
            if nm != "person":
                continue
            x1, y1, x2, y2 = [float(v) for v in det.xyxy[i]]
            coco["annotations"].append({
                "id": ann_id, "image_id": img_id, "category_id": 1,
                "bbox": [round(x1, 1), round(y1, 1), round(x2 - x1, 1), round(y2 - y1, 1)],
                "area": round((x2 - x1) * (y2 - y1), 1), "iscrowd": 0})
            ann_id += 1
            n_person += 1
        print(f"  {p.name}: person×{n_person}")

    OUT.write_text(json.dumps(coco, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n完成:{OUT.relative_to(ROOT).as_posix()}")
    print(f"  images={len(coco['images'])}, person 標註={len(coco['annotations'])}")
    print("  下一步:手動補 knife 標註後 merge,再切 train/val")


if __name__ == "__main__":
    main()
