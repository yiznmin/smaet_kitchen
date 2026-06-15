"""
解析度回升實驗:對「縮小後(小物件)」的圖,提高模型輸入解析度,看 knife AP 是否回升。
回答:「事後查詢可用高解析度 → 能否救回小物件偵測?」

- 固定縮放 s = 0.25(刀已崩盤的尺寸);也跑 s=1.0 當對照。
- rfdetr_nano:用 predict(shape=(R,R)) 改輸入解析度(R 為 56 的倍數)。
- dfine_n:用 transformers image processor 的 size 改輸入解析度。
輸出:results/m3_acc/resolution_recovery.csv
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

spec = importlib.util.spec_from_file_location("ea", ROOT / "scripts" / "eval_accuracy.py")
ea = importlib.util.module_from_spec(spec); spec.loader.exec_module(ea)

OUT = ROOT / "results" / "m3_acc"
PAD = (114, 114, 114)
THRESHOLD = 0.05
RES_RF = [384, 640, 960, 1280]     # rfdetr:predict(shape) 要求 32 的倍數
RES_TF = [640, 960, 1280]          # dfine:transformers processor size


def shrink_pad(pil, s):
    W, H = pil.size
    nw, nh = max(1, int(W * s)), max(1, int(H * s))
    canvas = Image.new("RGB", (W, H), PAD)
    ox, oy = (W - nw) // 2, (H - nh) // 2
    canvas.paste(pil.resize((nw, nh)), (ox, oy))
    return canvas, ox, oy


def transform_gt(det, s, ox, oy):
    if len(det) == 0:
        return det
    return sv.Detections(xyxy=det.xyxy * s + np.array([ox, oy, ox, oy]), class_id=det.class_id)


def knife_ap(r):
    if r.ap_per_class is None or not len(r.matched_classes):
        return 0.0
    per = np.array(r.ap_per_class).mean(axis=1)
    ki = ea.COMMON_IDX["knife"]
    for cid, ap in zip(r.matched_classes, per):
        if int(cid) == ki:
            return round(float(ap), 4)
    return 0.0


def eval_set(predict_fn, base, s):
    preds, targets = [], []
    for img, gt in base:
        canvas, ox, oy = shrink_pad(img, s)
        preds.append(predict_fn(canvas))
        targets.append(transform_gt(gt, s, ox, oy))
    r = MeanAveragePrecision().update(preds, targets).compute()
    return round(float(r.map50), 4), knife_ap(r)


def rf_predict_factory(model, R):
    def f(pil):
        det = model.predict(pil, threshold=THRESHOLD, shape=(R, R))
        names = det.data.get("class_name") if det.data else None
        b, c, cf = [], [], []
        for i in range(len(det)):
            nm = names[i] if names is not None else det.class_id[i]
            ci = ea.COMMON_IDX.get(ea.norm(nm))
            if ci is not None:
                b.append(det.xyxy[i].tolist()); c.append(ci)
                cf.append(float(det.confidence[i]) if det.confidence is not None else 1.0)
        return ea.to_det(b, c, cf)
    return f


def tf_predict_factory(proc, model, id2label, R):
    def f(pil):
        inp = proc(images=pil, return_tensors="pt", size={"height": R, "width": R})
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
    return f


def main():
    items = ea.load_items()
    base = [(Image.open(p).convert("RGB"), d) for p, d in items]
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []

    # rfdetr_nano(top 候選)
    try:
        from rfdetr import RFDETRNano
        model = RFDETRNano()
        for s in [1.0, 0.25]:
            for R in RES_RF:
                m50, kap = eval_set(rf_predict_factory(model, R), base, s)
                rows.append(dict(model="rfdetr_nano", scale=s, input_res=R, mAP50=m50, knife_AP=kap))
                print(f"  rfdetr_nano s={s} R={R}: mAP50={m50} knife_AP={kap}")
    except Exception as e:
        print("rfdetr 失敗:", repr(e)[:200])

    # dfine_n(transformers,改 processor size)
    try:
        from transformers import AutoImageProcessor, AutoModelForObjectDetection
        hf = "ustc-community/dfine-nano-coco"
        proc = AutoImageProcessor.from_pretrained(hf)
        model = AutoModelForObjectDetection.from_pretrained(hf).eval()
        id2label = model.config.id2label
        for s in [1.0, 0.25]:
            for R in RES_TF:
                m50, kap = eval_set(tf_predict_factory(proc, model, id2label, R), base, s)
                rows.append(dict(model="dfine_n", scale=s, input_res=R, mAP50=m50, knife_AP=kap))
                print(f"  dfine_n s={s} R={R}: mAP50={m50} knife_AP={kap}")
    except Exception as e:
        print("dfine 失敗:", repr(e)[:200])

    csv_path = OUT / "resolution_recovery.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\n數據:{csv_path.relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    main()
