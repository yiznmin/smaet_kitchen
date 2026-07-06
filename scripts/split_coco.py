"""
把標好的 _labeled.json(person + knife)切成 train/valid/test,COCO 格式供 rfdetr 微調。
knife 幀盡量平均分到各 split(讓 valid 也有刀可評估)。

輸出:data/m3_finetune/{train,valid,test}/ 各含 images + _annotations.coco.json
用法:python scripts/split_coco.py
"""
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT / "data" / "m3_finetune"
LABELED = BASE / "_labeled.json"
IMG_DIR = BASE / "images"
CATEGORIES = [{"id": 1, "name": "person"}, {"id": 2, "name": "knife"}]


def write_split(name, images, all_ann):
    d = BASE / name
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    ids = {im["id"] for im in images}
    anns = [a for a in all_ann if a["image_id"] in ids]
    for im in images:
        shutil.copy(IMG_DIR / im["file_name"], d / im["file_name"])
    coco = {"images": images, "annotations": anns, "categories": CATEGORIES}
    (d / "_annotations.coco.json").write_text(
        json.dumps(coco, ensure_ascii=False, indent=1), encoding="utf-8")
    k = sum(1 for a in anns if a["category_id"] == 2)
    print(f"  {name}: {len(images)} 圖, person={len(anns)-k}, knife={k}")


def main():
    d = json.loads(LABELED.read_text(encoding="utf-8"))
    by = {}
    for a in d["annotations"]:
        by.setdefault(a["image_id"], []).append(a)
    # 依「有沒有刀」分兩群,各自每 5 張挑 1 進 valid(其餘 train);test 另挑
    knife_imgs = [im for im in d["images"] if any(a["category_id"] == 2 for a in by.get(im["id"], []))]
    plain_imgs = [im for im in d["images"] if im not in knife_imgs]

    train, valid, test = [], [], []
    for group in (knife_imgs, plain_imgs):
        for i, im in enumerate(group):
            if i % 6 == 0:
                valid.append(im)
            elif i % 6 == 3:
                test.append(im)
            else:
                train.append(im)

    print(f"總 {len(d['images'])} 張 → train/valid/test")
    write_split("train", train, d["annotations"])
    write_split("valid", valid, d["annotations"])
    write_split("test", test, d["annotations"])
    print("完成。dataset_dir = data/m3_finetune(含 train/valid/test)")


if __name__ == "__main__":
    main()
