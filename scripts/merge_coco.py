"""
把幫手分批標好的多個 coco.json 合併成一個,供 split_coco.py 切分訓練。

分批標註器(make_bbox_labeler_html.py --batch)會產生 coco_1.json、coco_2.json…
每個檔各自從 image_id=1 起編號,直接堆疊會撞號 → 本腳本重新編號後合併。
以 file_name 為準去重(同一張圖若出現在兩批,後者覆蓋)。

用法:
  python scripts/merge_coco.py --out data/m3_finetune/_mv_labeled.json coco_1.json coco_2.json
  python scripts/merge_coco.py --out merged.json labels/*.json      # 萬用字元
"""
import argparse
import glob
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+", help="要合併的 coco.json(可多個或用 *)")
    ap.add_argument("--out", required=True, help="輸出合併後的 coco.json")
    args = ap.parse_args()

    # 展開萬用字元
    files = []
    for pat in args.inputs:
        hit = glob.glob(pat)
        files.extend(hit if hit else [pat])
    files = [f for f in files if Path(f).exists()]
    if not files:
        raise SystemExit("找不到任何輸入檔")

    categories = None
    by_name = {}          # file_name -> {"image": {...}, "anns": [...]}
    for f in files:
        d = json.loads(Path(f).read_text(encoding="utf-8"))
        if categories is None:
            categories = d["categories"]
        # 類別一致性檢查
        names = [c["name"] for c in d["categories"]]
        if names != [c["name"] for c in categories]:
            raise SystemExit(f"類別不一致:{f} 的類別與第一個檔不同,請確認都用同一組 --classes")
        anns_by_img = {}
        for a in d["annotations"]:
            anns_by_img.setdefault(a["image_id"], []).append(a)
        for im in d["images"]:
            by_name[im["file_name"]] = {          # 同名後者覆蓋
                "w": im.get("width") or im.get("w"),
                "h": im.get("height") or im.get("h"),
                "anns": anns_by_img.get(im["id"], []),
            }

    # 重新編號輸出
    images, annotations = [], []
    iid, aid = 1, 1
    for name in sorted(by_name):
        rec = by_name[name]
        images.append({"id": iid, "file_name": name, "width": rec["w"], "height": rec["h"]})
        for a in rec["anns"]:
            x, y, w, h = a["bbox"]
            annotations.append({"id": aid, "image_id": iid, "category_id": a["category_id"],
                                "bbox": [x, y, w, h], "area": w * h, "iscrowd": 0})
            aid += 1
        iid += 1

    out = {"images": images, "annotations": annotations, "categories": categories}
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")

    # 統計
    id2name = {c["id"]: c["name"] for c in categories}
    cnt = {}
    for a in annotations:
        n = id2name.get(a["category_id"], a["category_id"])
        cnt[n] = cnt.get(n, 0) + 1
    print(f"合併 {len(files)} 檔 → {args.out}")
    print(f"  {len(images)} 張圖,{len(annotations)} 個框")
    for n, c in sorted(cnt.items(), key=lambda kv: -kv[1]):
        print(f"    {n}: {c}")


if __name__ == "__main__":
    main()
