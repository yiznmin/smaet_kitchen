"""
建立 Arm C(混合訓練)資料集:EPFL train + Roboflow train 合併成單一 train,
valid / test 維持純 EPFL,確保與 Arm A / Arm B 的評估完全可比。

為什麼 valid/test 不混:A、B 兩臂都在純 EPFL 的 valid/test 上評估,
混進 Roboflow 會讓分數失去對照意義。

用法:
  python scripts/build_mix_dataset.py                       # 原始混合(與 Arm B 用同一份 rf 資料)
  python scripts/build_mix_dataset.py --clean-rf            # 先剔除漏標/錯標嚴重的圖再混
  python scripts/build_mix_dataset.py --out data/xxx --clean-rf
"""
import argparse
import json
import shutil
from collections import Counter
from pathlib import Path

ANN = "_annotations.coco.json"

# 已人工抽查確認漏標嚴重:整個廚房場景只標角落一個物件,
# 畫面中央的砧板/食材/容器全無標註(見 docs/資料集_Roboflow廚房_說明)
SCENE_PREFIXES = ("kitchenwoodcounter", "kitchenstonecounter", "ussink")
# 單框佔全圖面積過大 → 擺拍特寫,或多物件被合併成一個大框(如 knife32 一框包 4 把刀+砧板)
BLOB_AREA_RATIO = 0.40


def load(split_dir: Path):
    d = json.loads((split_dir / ANN).read_text(encoding="utf-8"))
    by_img = {}
    for im in d["images"]:
        by_img[im["id"]] = {"image": im, "anns": []}
    for a in d["annotations"]:
        if a["image_id"] in by_img:
            by_img[a["image_id"]]["anns"].append(a)
    return d["categories"], list(by_img.values())


def filter_rf(records, verbose=True):
    """剔除標註品質有問題的 Roboflow 圖。回傳 (保留, 剔除原因統計)"""
    kept, dropped = [], Counter()
    for r in records:
        fn = r["image"]["file_name"]
        if fn.startswith(SCENE_PREFIXES):
            dropped["場景照漏標"] += 1
            continue
        w, h = r["image"]["width"], r["image"]["height"]
        area = w * h
        if any((a["bbox"][2] * a["bbox"][3]) / area > BLOB_AREA_RATIO for a in r["anns"]):
            dropped["大框(疑似多物件合併/特寫)"] += 1
            continue
        kept.append(r)
    if verbose:
        for k, v in dropped.items():
            print(f"    剔除 {k}: {v} 張")
    return kept


def write_split(out_dir: Path, categories, sources, link=False):
    """sources = [(src_split_dir, records), ...] → 合併寫出到 out_dir"""
    out_dir.mkdir(parents=True, exist_ok=True)
    images, annotations = [], []
    iid = aid = 1
    seen = set()
    for src_dir, records in sources:
        for r in records:
            fn = r["image"]["file_name"]
            if fn in seen:
                print(f"    ⚠ 檔名重複,跳過:{fn}")
                continue
            seen.add(fn)
            im = dict(r["image"], id=iid)
            images.append(im)
            for a in r["anns"]:
                annotations.append(dict(a, id=aid, image_id=iid))
                aid += 1
            iid += 1
            dst = out_dir / fn
            if not dst.exists():
                src = src_dir / fn
                if link:
                    try:
                        dst.hardlink_to(src)
                        continue
                    except OSError:
                        pass
                shutil.copy2(src, dst)
    (out_dir / ANN).write_text(
        json.dumps({"images": images, "annotations": annotations, "categories": categories},
                   ensure_ascii=False),
        encoding="utf-8")
    return images, annotations, categories


def summarize(tag, images, annotations, categories):
    id2n = {c["id"]: c["name"] for c in categories}
    cnt = Counter(id2n[a["category_id"]] for a in annotations)
    avg = len(annotations) / max(1, len(images))
    print(f"  {tag}: {len(images)} 圖 / {len(annotations)} 框 (平均 {avg:.2f} 框/圖)")
    if cnt:
        print("    " + "  ".join(f"{k}={v}" for k, v in cnt.most_common()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epfl", default="data/m3_finetune_r2")
    ap.add_argument("--rf", default="data/m3_finetune_rf")
    ap.add_argument("--out", default=None, help="預設 data/m3_finetune_mix[_clean]")
    ap.add_argument("--clean-rf", action="store_true", help="剔除漏標/錯標嚴重的 Roboflow 圖")
    ap.add_argument("--link", action="store_true", help="用 hardlink 取代複製(省空間)")
    args = ap.parse_args()

    epfl, rf = Path(args.epfl), Path(args.rf)
    out = Path(args.out or f"data/m3_finetune_mix{'_clean' if args.clean_rf else ''}")

    cats_e, tr_e = load(epfl / "train")
    cats_r, tr_r = load(rf / "train")
    if [c["name"] for c in cats_e] != [c["name"] for c in cats_r]:
        raise SystemExit("兩份資料的類別清單不一致,無法合併")

    print(f"來源:EPFL train {len(tr_e)} 圖 / Roboflow train {len(tr_r)} 圖")
    if args.clean_rf:
        print("  清洗 Roboflow:")
        tr_r = filter_rf(tr_r)
        print(f"    保留 {len(tr_r)} 張")

    print(f"\n輸出 → {out}")
    summarize("train", *write_split(out / "train", cats_e,
                                    [(epfl / "train", tr_e), (rf / "train", tr_r)], args.link))

    # valid / test 維持純 EPFL,與 Arm A / B 完全可比
    for split in ("valid", "test"):
        cats, recs = load(epfl / split)
        summarize(f"{split}(純 EPFL)", *write_split(out / split, cats, [(epfl / split, recs)], args.link))

    print(f"\n完成。訓練時 dataset_dir 指向:{out}")


if __name__ == "__main__":
    main()
