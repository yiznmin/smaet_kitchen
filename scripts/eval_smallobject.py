"""
小物件模擬實驗:把資料集物件「人工縮小」(縮放+灰底填充),模擬俯視遠距小物件,
量 AP 隨尺寸的衰減,並可選測「提高輸入解析度能否救回」。

做法(關鍵):不能只縮圖(模型會 resize 回固定輸入,佔比不變);
正解是把整張圖縮 s 倍貼到原尺寸畫布(灰底填充),使物件佔畫面比例變小,GT 框同步 ×s+偏移。

輸出:results/m3_acc/smallobject.csv + 視覺化 results/m3_acc/vis_small/
用法:python scripts/eval_smallobject.py
"""
import csv
import importlib.util
import sys
from pathlib import Path

import numpy as np
import supervision as sv
import torch
from PIL import Image
from supervision.metrics import MeanAveragePrecision

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from common.video_io import imwrite_unicode   # noqa: E402

# 重用 eval_accuracy 的載入/類別邏輯
spec = importlib.util.spec_from_file_location("ea", ROOT / "scripts" / "eval_accuracy.py")
ea = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ea)

OUT = ROOT / "results" / "m3_acc"
VIS = OUT / "vis_small"
SCALES = [1.0, 0.5, 0.25]
PAD = (114, 114, 114)
THRESHOLD = 0.05
VIS_THRESHOLD = 0.3


def shrink_pad(pil, s):
    """把圖縮 s 倍貼到原尺寸畫布(灰底)。回傳 (canvas, ox, oy)。"""
    W, H = pil.size
    nw, nh = max(1, int(W * s)), max(1, int(H * s))
    canvas = Image.new("RGB", (W, H), PAD)
    ox, oy = (W - nw) // 2, (H - nh) // 2
    canvas.paste(pil.resize((nw, nh)), (ox, oy))
    return canvas, ox, oy


def transform_gt(det, s, ox, oy):
    if len(det) == 0:
        return det
    xyxy = det.xyxy * s + np.array([ox, oy, ox, oy])
    return sv.Detections(xyxy=xyxy, class_id=det.class_id)


# ---- 各模型:回傳一個「吃 PIL → 出 COMMON-space Detections(畫布座標)」的函式 ----
def make_tf_predictor(hf_id):
    from transformers import AutoImageProcessor, AutoModelForObjectDetection
    proc = AutoImageProcessor.from_pretrained(hf_id)
    model = AutoModelForObjectDetection.from_pretrained(hf_id).eval()
    id2label = model.config.id2label

    def predict(pil):
        inp = proc(images=pil, return_tensors="pt")
        with torch.no_grad():
            out = model(**inp)
        res = proc.post_process_object_detection(
            out, target_sizes=[(pil.height, pil.width)], threshold=THRESHOLD)[0]
        b, c, cf = [], [], []
        for s_, l, bx in zip(res["scores"], res["labels"], res["boxes"]):
            ci = ea.COMMON_IDX.get(ea.norm(id2label[int(l)]))
            if ci is not None:
                b.append(bx.tolist()); c.append(ci); cf.append(float(s_))
        return ea.to_det(b, c, cf)
    return predict


def make_rf_predictor(ctor):
    model = ctor()

    def predict(pil):
        det = model.predict(pil, threshold=THRESHOLD)
        names = det.data.get("class_name") if det.data else None
        b, c, cf = [], [], []
        for i in range(len(det)):
            nm = names[i] if names is not None else det.class_id[i]
            ci = ea.COMMON_IDX.get(ea.norm(nm))
            if ci is not None:
                b.append(det.xyxy[i].tolist()); c.append(ci)
                cf.append(float(det.confidence[i]) if det.confidence is not None else 1.0)
        return ea.to_det(b, c, cf)
    return predict


def knife_ap(r):
    if r.ap_per_class is None or not len(r.matched_classes):
        return ""
    per = np.array(r.ap_per_class).mean(axis=1)
    ki = ea.COMMON_IDX["knife"]
    for cid, ap in zip(r.matched_classes, per):
        if int(cid) == ki:
            return round(float(ap), 4)
    return 0.0


def main():
    items = ea.load_items()
    OUT.mkdir(parents=True, exist_ok=True)
    VIS.mkdir(parents=True, exist_ok=True)

    specs = [("rfdetr_nano", "rf", None), ("dfine_n", "tf", "ustc-community/dfine-nano-coco")]
    from rfdetr import RFDETRNano
    ctors = {"rfdetr_nano": RFDETRNano}

    # 預先載入縮圖(避免重複開檔)
    base = [(Image.open(p).convert("RGB"), d) for p, d in items]
    # 找含刀的圖供視覺化
    knife_idx = next((i for i, (_, d) in enumerate(items)
                      if len(d) and ea.COMMON_IDX["knife"] in d.class_id), 0)

    rows = []
    for name, kind, ref in specs:
        print(f"\n=== {name} ===")
        predict = (make_rf_predictor(ctors[name]) if kind == "rf"
                   else make_tf_predictor(ref))
        for s in SCALES:
            preds, targets = [], []
            for img, gt in base:
                canvas, ox, oy = shrink_pad(img, s)
                preds.append(predict(canvas))
                targets.append(transform_gt(gt, s, ox, oy))
            r = MeanAveragePrecision().update(preds, targets).compute()
            small = getattr(getattr(r, "small_objects", None), "map50_95", None)
            row = dict(model=name, scale=s,
                       mAP50=round(float(r.map50), 4),
                       mAP50_95=round(float(r.map50_95), 4),
                       knife_AP=knife_ap(r),
                       small_mAP=round(float(small), 4) if small is not None else "")
            rows.append(row)
            print(f"  scale={s}: mAP50={row['mAP50']} knife_AP={row['knife_AP']} 小物件mAP={row['small_mAP']}")

            # 視覺化:含刀那張,每個 scale 存一張
            img, gt = base[knife_idx]
            canvas, ox, oy = shrink_pad(img, s)
            import cv2
            vis = cv2.cvtColor(np.array(canvas), cv2.COLOR_RGB2BGR)
            for bx in transform_gt(gt, s, ox, oy).xyxy:
                x1, y1, x2, y2 = [int(v) for v in bx]
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 200, 0), 2)
            pred = predict(canvas)
            for i in range(len(pred)):
                x1, y1, x2, y2 = [int(v) for v in pred.xyxy[i]]
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 220), 2)
            imwrite_unicode(VIS / f"{name}_s{s}.png", vis)

    csv_path = OUT / "smallobject.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\n數據:{csv_path.relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    main()
