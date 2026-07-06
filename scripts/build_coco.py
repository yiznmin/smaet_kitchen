"""
把 person(自動標)+ knife(固定框,靜態前景刀)合併成 COCO,切 train/valid。

knife 為固定框:攝影機與刀都靜止,前景那把刀在每幀位置幾乎不變,故用一個框套全部。
(bootstrap 用途;出貨級需人工逐張複核)

輸出:data/m3_finetune/{train,valid}/ 各含 images + _annotations.coco.json
用法:python scripts/build_coco.py
"""
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT / "data" / "m3_finetune"
DRAFT = BASE / "_draft.json"
IMG_DIR = BASE / "images"

KNIFE_BBOX = [425, 630, 155, 82]        # x, y, w, h(固定前景刀)
VALID_EVERY = 5                         # 每 5 張挑 1 張進 valid(其餘 train)
CATEGORIES = [{"id": 1, "name": "person"}, {"id": 2, "name": "knife"}]


def write_split(name, images, anns):
    d = BASE / name
    d.mkdir(parents=True, exist_ok=True)
    for im in images:
        shutil.copy(IMG_DIR / im["file_name"], d / im["file_name"])
    coco = {"images": images, "annotations": anns, "categories": CATEGORIES}
    (d / "_annotations.coco.json").write_text(
        json.dumps(coco, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"  {name}: {len(images)} 圖, {len(anns)} 標註")


def main():
    draft = json.loads(DRAFT.read_text(encoding="utf-8"))
    # 依 image 收 person 標註
    by_img = {}
    for a in draft["annotations"]:
        by_img.setdefault(a["image_id"], []).append(a)

    ann_id = 1
    train_imgs, valid_imgs, train_ann, valid_ann = [], [], [], []
    for idx, im in enumerate(draft["images"]):
        to_valid = (idx % VALID_EVERY == 0)
        target_imgs = valid_imgs if to_valid else train_imgs
        target_ann = valid_ann if to_valid else train_ann
        target_imgs.append(im)
        # person(承接自動標)
        for a in by_img.get(im["id"], []):
            target_ann.append({"id": ann_id, "image_id": im["id"], "category_id": 1,
                               "bbox": a["bbox"], "area": a["area"], "iscrowd": 0})
            ann_id += 1
        # knife(固定框)
        x, y, w, h = KNIFE_BBOX
        target_ann.append({"id": ann_id, "image_id": im["id"], "category_id": 2,
                           "bbox": [x, y, w, h], "area": w * h, "iscrowd": 0})
        ann_id += 1

    print("切分結果:")
    write_split("train", train_imgs, train_ann)
    write_split("valid", valid_imgs, valid_ann)
    kn_tr = sum(1 for a in train_ann if a["category_id"] == 2)
    kn_va = sum(1 for a in valid_ann if a["category_id"] == 2)
    print(f"knife 標註:train {kn_tr} + valid {kn_va} = {kn_tr + kn_va}")
    print(f"資料夾:{(BASE/'train').relative_to(ROOT).as_posix()}/、{(BASE/'valid').relative_to(ROOT).as_posix()}/")


if __name__ == "__main__":
    main()
