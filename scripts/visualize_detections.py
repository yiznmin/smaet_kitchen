"""
把「正確答案(綠)」與「模型偵測(紅)」畫在資料集照片上,直觀看 IoU / 漏抓 / 亂報。
用 rfdetr_nano 在幾張含刀的圖上推論,輸出到 results/m3_acc/vis/。
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from common.video_io import imwrite_unicode   # noqa: E402

DATA = ROOT / "data" / "roboflow_kitchen"
OUT = ROOT / "results" / "m3_acc" / "vis"
SPLIT = "train"          # train 圖較多,容易挑到含刀的
N = 4


def imread_unicode(path):
    return cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)


def main():
    j = json.load(open(DATA / SPLIT / "_annotations.coco.json", encoding="utf-8"))
    cat = {c["id"]: c["name"] for c in j["categories"]}
    knife_ids = {cid for cid, n in cat.items() if n.lower() == "knife"}
    from collections import defaultdict
    by_img = defaultdict(list)
    for a in j["annotations"]:
        by_img[a["image_id"]].append(a)
    # 挑含刀的圖
    imgs = [im for im in j["images"]
            if any(a["category_id"] in knife_ids for a in by_img.get(im["id"], []))][:N]

    from rfdetr import RFDETRNano
    model = RFDETRNano()
    OUT.mkdir(parents=True, exist_ok=True)

    saved = []
    for im in imgs:
        path = DATA / SPLIT / im["file_name"]
        img = imread_unicode(path)
        if img is None:
            continue
        # 綠:正確答案(GT)
        for a in by_img.get(im["id"], []):
            x, y, w, h = [int(v) for v in a["bbox"]]
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 200, 0), 2)
            cv2.putText(img, cat[a["category_id"]], (x, max(12, y - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1)
        # 紅:模型偵測(pred)
        det = model.predict(Image.open(path).convert("RGB"), threshold=0.3)
        names = det.data.get("class_name") if det.data else None
        for i in range(len(det)):
            x1, y1, x2, y2 = [int(v) for v in det.xyxy[i]]
            nm = str(names[i]) if names is not None else str(det.class_id[i])
            cf = float(det.confidence[i]) if det.confidence is not None else 0
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 220), 2)
            cv2.putText(img, f"{nm} {cf:.2f}", (x1, y2 + 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 220), 1)
        fp = OUT / f"vis_{Path(im['file_name']).stem}.png"
        if imwrite_unicode(fp, img):
            saved.append(fp.relative_to(ROOT).as_posix())
            print("saved", saved[-1])
    print("done", len(saved))


if __name__ == "__main__":
    main()
