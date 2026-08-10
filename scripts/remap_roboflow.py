"""把 Roboflow 廚房資料集(14 類家用物件)對映成我們的 11 類,產出 stage-1 訓練素材。

只保留與我們重疊的類別:
  knife → 刀具;bowl/cup/plate/wineglass/bottel(bottle) → 容器
其餘(blender/countertop*/fork/spoon/microwave/refrigerator/sink)丟棄。
輸出 categories = 完整 11 類(未出現的類 0 框),讓 stage2 載入時偵測頭不重置。

只留「至少有一個保留框」的圖(= 專門補刀具/容器範例)。

Roboflow 授權 = CC BY 4.0(可商用,需標來源);但之後與 EPFL(CC-NC)混用的模型仍為驗證版。

輸出:
  data/m3_finetune/_roboflow11_labeled.json      (合併 COCO)
  data/m3_finetune/roboflow11_images/*.jpg        (對應圖片)
接著用現有 split_coco.py 切分即可:
  python scripts/split_coco.py --labeled data/m3_finetune/_roboflow11_labeled.json \\
      --img_dir data/m3_finetune/roboflow11_images --out_base data/m3_finetune_rf
"""
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from m3.classes import COCO_CATEGORIES, NAME_TO_CAT   # noqa: E402

SRC = ROOT / "data" / "roboflow_kitchen"
OUT_JSON = ROOT / "data" / "m3_finetune" / "_roboflow11_labeled.json"
OUT_IMG = ROOT / "data" / "m3_finetune" / "roboflow11_images"

# Roboflow 類名 → 我們的類名(未列出的丟棄)。bottel 是 Roboflow 的拼錯。
ROBOFLOW_MAP = {
    "knife": "刀具",
    "bowl": "容器", "cup": "容器", "plate": "容器",
    "wineglass": "容器", "bottel": "容器", "bottle": "容器",
}


def main():
    images, annotations = [], []
    iid, aid = 1, 1
    kept_by_class = {}

    if OUT_IMG.exists():
        shutil.rmtree(OUT_IMG)
    OUT_IMG.mkdir(parents=True)

    for split in ["train", "valid", "test"]:
        jp = SRC / split / "_annotations.coco.json"
        if not jp.exists():
            continue
        d = json.loads(jp.read_text(encoding="utf-8"))
        rcat = {c["id"]: c["name"] for c in d["categories"]}      # roboflow catid -> 名
        anns_by_img = {}
        for a in d["annotations"]:
            our_name = ROBOFLOW_MAP.get(rcat.get(a["category_id"]))
            if our_name is None:
                continue
            anns_by_img.setdefault(a["image_id"], []).append((our_name, a["bbox"]))

        for im in d["images"]:
            keep = anns_by_img.get(im["id"])
            if not keep:                       # 只留有保留框的圖
                continue
            src_img = SRC / split / im["file_name"]
            if not src_img.exists():
                continue
            shutil.copy(src_img, OUT_IMG / im["file_name"])
            images.append({"id": iid, "file_name": im["file_name"],
                           "width": im["width"], "height": im["height"]})
            for our_name, bbox in keep:
                cat = NAME_TO_CAT[our_name]
                x, y, w, h = bbox
                annotations.append({"id": aid, "image_id": iid, "category_id": cat,
                                    "bbox": [x, y, w, h], "area": w * h, "iscrowd": 0})
                aid += 1
                kept_by_class[our_name] = kept_by_class.get(our_name, 0) + 1
            iid += 1

    out = {"images": images, "annotations": annotations, "categories": COCO_CATEGORIES}
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"輸出:{OUT_JSON.relative_to(ROOT).as_posix()}")
    print(f"  {len(images)} 圖,{len(annotations)} 框(categories = 完整 11 類)")
    for n, c in sorted(kept_by_class.items(), key=lambda kv: -kv[1]):
        print(f"    {n}: {c}")
    print(f"圖片 → {OUT_IMG.relative_to(ROOT).as_posix()}/")
    print("\n下一步:")
    print("  python scripts/split_coco.py --labeled data/m3_finetune/_roboflow11_labeled.json \\")
    print("      --img_dir data/m3_finetune/roboflow11_images --out_base data/m3_finetune_rf")


if __name__ == "__main__":
    main()
