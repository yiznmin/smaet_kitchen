"""把標註頁匯出的 coco.json(可多份)合併,切成 train / valid / test,建立 M3 訓練資料集。

輸出結構(train_m3_rfdetr.py 的 --dataset 直接指向它):
  <out>/train/_annotations.coco.json + 圖片
  <out>/valid/_annotations.coco.json + 圖片
  <out>/test/_annotations.coco.json  + 圖片

切分:依檔名排序後每 6 張 1 張進 valid、1 張進 test、其餘進 train(約 67% / 17% / 17%)。
同一張圖出現在多份匯出檔時,以後面的為準。沒有任何框的圖仍保留(當作「沒有物件」的範例)。

用法:
  python scripts/build_dataset.py --labels coco_1.json coco_2.json --img-dir datasets/kitchen/images --out datasets/kitchen/m3
"""
import argparse
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from m3.classes import COCO_CATEGORIES   # noqa: E402


def merge(files):
    """回傳 {file_name: (image, [annotation])},以 file_name 去重。"""
    by_name = {}
    for f in files:
        d = json.loads(Path(f).read_text(encoding="utf-8"))
        if [c["name"] for c in d["categories"]] != [c["name"] for c in COCO_CATEGORIES]:
            raise SystemExit(f"{f} 的類別與 M3 的 11 類不同,請用 make_labeler.py 產生的標註頁")
        anns = {}
        for a in d["annotations"]:
            anns.setdefault(a["image_id"], []).append(a)
        for im in d["images"]:
            by_name[im["file_name"]] = (im, anns.get(im["id"], []))
    return by_name


def write_split(out, img_dir, name, items):
    d = out / name
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    images, anns = [], []
    for i, (im, al) in enumerate(items, start=1):
        shutil.copy(img_dir / im["file_name"], d / im["file_name"])
        images.append(dict(im, id=i))
        for a in al:
            anns.append(dict(a, id=len(anns) + 1, image_id=i))
    coco = {"images": images, "annotations": anns, "categories": COCO_CATEGORIES}
    (d / "_annotations.coco.json").write_text(json.dumps(coco, ensure_ascii=False, indent=1), encoding="utf-8")
    return anns


def main():
    ap = argparse.ArgumentParser(description="合併標註、切分 M3 訓練資料集")
    ap.add_argument("--labels", nargs="+", required=True, help="標註頁匯出的 coco.json(可多份)")
    ap.add_argument("--img-dir", required=True, help="圖片資料夾(extract_frames.py 的輸出)")
    ap.add_argument("--out", required=True, help="資料集輸出資料夾")
    args = ap.parse_args()

    img_dir, out = Path(args.img_dir), Path(args.out)
    by_name = merge(args.labels)
    missing = [n for n in by_name if not (img_dir / n).exists()]
    if missing:
        raise SystemExit(f"{len(missing)} 張圖在 {img_dir} 找不到,例如 {missing[:3]}")
    items = [by_name[n] for n in sorted(by_name)]
    splits = {"train": [], "valid": [], "test": []}
    for i, it in enumerate(items):
        splits["valid" if i % 6 == 0 else "test" if i % 6 == 3 else "train"].append(it)

    names = {c["id"]: c["name"] for c in COCO_CATEGORIES}
    print(f"共 {len(items)} 張")
    for name, its in splits.items():
        anns = write_split(out, img_dir, name, its)
        cnt = Counter(names[a["category_id"]] for a in anns)
        print(f"  {name}:{len(its)} 張、{len(anns)} 框  {dict(cnt)}")
    zero = [n for n in names.values() if n not in
            Counter(names[a["category_id"]] for it in items for a in it[1])]
    if zero:
        print(f"注意:這些類別沒有任何標註,訓練後偵測不到:{zero}")
    print(f"資料集:{out}(訓練時 --dataset 指向這裡)")


if __name__ == "__main__":
    main()
