"""
M3 準確度初步評估:在 Roboflow kitchen-object-detection(CC BY 4.0)上量 8 個模型的 mAP。

- 只評「與 COCO 重疊的 10 類」:knife/fork/spoon/sink/bowl/cup/bottle/microwave/refrigerator/wineglass。
  (countertop/blender/plate/kitchen-object 非 COCO → 跳過)
- 模型用原生框架推論以正確解碼:transformers(D-FINE/RT-DETR)、rfdetr(RF-DETR)。
- 用 supervision 算 mAP@50、mAP@50:95、per-class AP、以及小/中/大物件分項。
- 模型為 COCO 預訓練(未在廚房微調)→ 數字代表「現成模型」表現。初步參考,非最終依據。

輸出:results/m3_acc/m3_accuracy.csv + m3_accuracy.md
用法:python scripts/eval_accuracy.py
"""
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import supervision as sv
import torch
from PIL import Image
from supervision.metrics import MeanAveragePrecision

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "roboflow_kitchen"
OUT = ROOT / "results" / "m3_acc"
SPLITS = ["train", "test"]          # 模型未在此訓練,全部當評估資料
THRESHOLD = 0.05                    # 低門檻以利 AP 積分(取較完整的 PR 曲線)

COMMON = ["knife", "fork", "spoon", "sink", "bowl", "cup",
          "bottle", "microwave", "refrigerator", "wineglass"]
COMMON_IDX = {n: i for i, n in enumerate(COMMON)}
DS_FIX = {"bottel": "bottle"}       # 資料集拼字修正


def norm(name):
    return str(name).lower().replace(" ", "").replace("_", "")


def to_common(name):
    name = DS_FIX.get(str(name).lower(), name)
    return COMMON_IDX.get(norm(name))   # None 表示不在重疊類別


# ---------- Ground truth ----------
def load_items():
    """回傳 [(image_path, gt_Detections(in COMMON space)), ...](含無相關標註的圖)。"""
    items = []
    for split in SPLITS:
        j = json.load(open(DATA / split / "_annotations.coco.json", encoding="utf-8"))
        cat = {c["id"]: c["name"] for c in j["categories"]}
        by_img = defaultdict(list)
        for a in j["annotations"]:
            by_img[a["image_id"]].append(a)
        for im in j["images"]:
            boxes, cls = [], []
            for a in by_img.get(im["id"], []):
                ci = to_common(cat[a["category_id"]])
                if ci is not None:
                    x, y, w, h = a["bbox"]
                    boxes.append([x, y, x + w, y + h]); cls.append(ci)
            det = (sv.Detections(xyxy=np.array(boxes, float), class_id=np.array(cls, int))
                   if boxes else sv.Detections.empty())
            items.append((DATA / split / im["file_name"], det))
    return items


def to_det(boxes, cls, conf):
    if not boxes:
        return sv.Detections.empty()
    return sv.Detections(xyxy=np.array(boxes, float).reshape(-1, 4),
                         class_id=np.array(cls, int),
                         confidence=np.array(conf, float))


# ---------- 各模型推論 ----------
def predict_transformers(hf_id, items):
    from transformers import AutoImageProcessor, AutoModelForObjectDetection
    proc = AutoImageProcessor.from_pretrained(hf_id)
    model = AutoModelForObjectDetection.from_pretrained(hf_id).eval()
    id2label = model.config.id2label
    preds = []
    for path, _ in items:
        img = Image.open(path).convert("RGB")
        inp = proc(images=img, return_tensors="pt")
        with torch.no_grad():
            out = model(**inp)
        res = proc.post_process_object_detection(
            out, target_sizes=[(img.height, img.width)], threshold=THRESHOLD)[0]
        boxes, cls, conf = [], [], []
        for s, l, b in zip(res["scores"], res["labels"], res["boxes"]):
            ci = COMMON_IDX.get(norm(id2label[int(l)]))
            if ci is not None:
                boxes.append(b.tolist()); cls.append(ci); conf.append(float(s))
        preds.append(to_det(boxes, cls, conf))
    return preds


def predict_rfdetr(ctor, items):
    model = ctor()
    preds = []
    for path, _ in items:
        img = Image.open(path).convert("RGB")
        det = model.predict(img, threshold=THRESHOLD)
        names = det.data.get("class_name") if det.data else None
        boxes, cls, conf = [], [], []
        for i in range(len(det)):
            nm = names[i] if names is not None else det.class_id[i]
            ci = COMMON_IDX.get(norm(nm))
            if ci is not None:
                boxes.append(det.xyxy[i].tolist()); cls.append(ci)
                conf.append(float(det.confidence[i]) if det.confidence is not None else 1.0)
        preds.append(to_det(boxes, cls, conf))
    return preds


def model_specs():
    from rfdetr import RFDETRMedium, RFDETRNano, RFDETRSmall
    return [
        ("dfine_n", "tf", "ustc-community/dfine-nano-coco"),
        ("dfine_s", "tf", "ustc-community/dfine-small-coco"),
        ("dfine_m", "tf", "ustc-community/dfine-medium-coco"),
        ("rtdetrv2_r18", "tf", "PekingU/rtdetr_v2_r18vd"),
        ("rtdetrv2_r34", "tf", "PekingU/rtdetr_v2_r34vd"),
        ("rfdetr_nano", "rf", RFDETRNano),
        ("rfdetr_small", "rf", RFDETRSmall),
        ("rfdetr_medium", "rf", RFDETRMedium),
    ]


def main():
    items = load_items()
    targets = [d for _, d in items]
    n_gt = sum(len(d) for d in targets)
    print(f"評估圖片 {len(items)} 張,重疊類別標註 {n_gt} 個(COMMON {len(COMMON)} 類)")
    OUT.mkdir(parents=True, exist_ok=True)

    rows = []
    for name, kind, ref in model_specs():
        print(f"\n=== {name} 推論中 …")
        try:
            preds = (predict_transformers(ref, items) if kind == "tf"
                     else predict_rfdetr(ref, items))
            r = MeanAveragePrecision().update(preds, targets).compute()
            # per-class AP(對 IoU 取平均)
            ap_pc = {}
            if r.ap_per_class is not None and len(r.matched_classes):
                per = np.array(r.ap_per_class).mean(axis=1)
                for cid, ap in zip(r.matched_classes, per):
                    ap_pc[COMMON[int(cid)]] = round(float(ap), 4)
            small = getattr(getattr(r, "small_objects", None), "map50_95", None)
            row = dict(model=name, mAP50=round(float(r.map50), 4),
                       mAP50_95=round(float(r.map50_95), 4),
                       mAP_small=round(float(small), 4) if small is not None else "",
                       knife_AP=ap_pc.get("knife", ""))
            for c in COMMON:
                row[f"AP_{c}"] = ap_pc.get(c, "")
            rows.append(row)
            print(f"  mAP50={row['mAP50']} mAP50:95={row['mAP50_95']} "
                  f"knife_AP={row['knife_AP']} 小物件mAP={row['mAP_small']}")
        except Exception as e:
            print(f"  ✗ {name} 失敗: {repr(e)[:200]}")
            rows.append(dict(model=name, mAP50="ERROR", mAP50_95="", mAP_small="",
                             knife_AP="", **{f"AP_{c}": "" for c in COMMON}))

    # CSV
    csv_path = OUT / "m3_accuracy.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    # 報告
    md = [
        "# M3 物件偵測準確度(初步)",
        "",
        f"- 資料集:Roboflow kitchen-object-detection v1(CC BY 4.0),{len(items)} 張、重疊標註 {n_gt} 個。",
        "- 只評與 COCO 重疊的 10 類;模型為 **COCO 預訓練、未在廚房微調**。",
        "- ⚠ 初步參考:資料小、單一域、缺 person/cutting_board/生肉;非最終依據。",
        "",
        "| model | mAP@50 | mAP@50:95 | knife AP | 小物件 mAP |",
        "|---|---|---|---|---|",
    ]
    for r in rows:
        md.append(f"| {r['model']} | {r['mAP50']} | {r['mAP50_95']} | {r['knife_AP']} | {r['mAP_small']} |")
    md += ["", "完整 per-class AP 見 `m3_accuracy.csv`。",
           "", "> 資料來源:Roboflow kitchen-object-detection (CC BY 4.0)。"]
    (OUT / "m3_accuracy.md").write_text("\n".join(md), encoding="utf-8")
    print(f"\n報告:{(OUT / 'm3_accuracy.md').relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    main()
