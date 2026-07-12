"""
把標好的 COCO 切成 train/valid/test,供 rfdetr 微調。類別直接沿用輸入檔。

用法:
  python scripts/split_coco.py                              # 預設 _labeled.json + images
  python scripts/split_coco.py --labeled data/m3_finetune/_mv_labeled.json \\
        --img_dir data/m3_finetune/mv_images --out_base data/m3_finetune_mv
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def write_split(out_base, img_dir, name, images, all_ann, categories):
    d = out_base / name
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    ids = {im["id"] for im in images}
    anns = [a for a in all_ann if a["image_id"] in ids]
    for im in images:
        shutil.copy(img_dir / im["file_name"], d / im["file_name"])
    coco = {"images": images, "annotations": anns, "categories": categories}
    (d / "_annotations.coco.json").write_text(
        json.dumps(coco, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"  {name}: {len(images)} 圖, {len(anns)} 框")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labeled", default=str(ROOT / "data" / "m3_finetune" / "_labeled.json"))
    ap.add_argument("--img_dir", default=str(ROOT / "data" / "m3_finetune" / "images"))
    ap.add_argument("--out_base", default=str(ROOT / "data" / "m3_finetune"))
    args = ap.parse_args()

    d = json.loads(Path(args.labeled).read_text(encoding="utf-8"))
    cats = d["categories"]
    img_dir = Path(args.img_dir)
    out_base = Path(args.out_base).resolve()

    # 均勻切:每 6 張,1 進 valid、1 進 test、其餘 train
    train, valid, test = [], [], []
    for i, im in enumerate(d["images"]):
        (valid if i % 6 == 0 else test if i % 6 == 3 else train).append(im)

    print(f"總 {len(d['images'])} 張,{len(cats)} 類 → train/valid/test")
    write_split(out_base, img_dir, "train", train, d["annotations"], cats)
    write_split(out_base, img_dir, "valid", valid, d["annotations"], cats)
    write_split(out_base, img_dir, "test", test, d["annotations"], cats)
    print(f"dataset_dir = {out_base.relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    main()
