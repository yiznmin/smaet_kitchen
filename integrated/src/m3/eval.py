"""M3 每類偵測評估工具(把 notebook eval_model 的邏輯抽成可 import 的函式)。

用法概念:
  gt    = load_gt(test_dir)                       # {file_name: [(class_id, [x1,y1,x2,y2])]}
  preds = gather_predictions(model, test_dir)     # 低門檻 predict 一次,拿全部候選
  tp,fp,fn,n = counts_for_class(gt, preds, cls, thr)   # 單類、單門檻
  prf(tp,fp,fn) -> (precision, recall, f1)

class_id 一律用模型的 0..10(GT 的 COCO cat_id 1..11 在 load_gt 內以 CAT2CLS 轉好)。
"""
import json
import os

from m3.classes import CAT2CLS, NAMES


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def load_gt(test_dir):
    """讀 COCO GT → {file_name: [(class_id, [x1,y1,x2,y2])]}(含 0 框的圖也列入,以計 FP)。"""
    j = json.loads((open(os.path.join(test_dir, "_annotations.coco.json"), encoding="utf-8")).read())
    id2fn = {im["id"]: im["file_name"] for im in j["images"]}
    by = {im["file_name"]: [] for im in j["images"]}
    for a in j["annotations"]:
        cls = CAT2CLS.get(a["category_id"])
        if cls is None:
            continue
        x, y, w, h = a["bbox"]
        by[id2fn[a["image_id"]]].append((cls, [x, y, x + w, y + h]))
    return by


def gather_predictions(model, test_dir, base_thr=0.05):
    """低門檻 predict 一次,回 {file_name: [(class_id, conf, [x1,y1,x2,y2])]}。之後掃門檻不必重跑模型。"""
    from PIL import Image
    gt = load_gt(test_dir)
    preds = {}
    for fn in gt:
        det = model.predict(Image.open(os.path.join(test_dir, fn)).convert("RGB"), threshold=base_thr)
        rows = []
        n = len(det)
        confs = getattr(det, "confidence", None)
        for i in range(n):
            s = float(confs[i]) if confs is not None else 1.0
            rows.append((int(det.class_id[i]), s, [float(v) for v in det.xyxy[i]]))
        preds[fn] = rows
    return preds


def counts_for_class(gt, preds, cls, thr, iou_thr=0.5):
    """單一類別、單一門檻的 TP/FP/FN/n_gt(信心排序貪婪配對,同類才配)。"""
    tp = fp = n_gt = 0
    for fn, objs in gt.items():
        gboxes = [b for (c, b) in objs if c == cls]
        n_gt += len(gboxes)
        matched = [False] * len(gboxes)
        rows = sorted([(s, b) for (c, s, b) in preds.get(fn, []) if c == cls and s >= thr],
                      key=lambda r: -r[0])
        for s, b in rows:
            best, bv = -1, iou_thr
            for j, gb in enumerate(gboxes):
                if matched[j]:
                    continue
                v = iou(b, gb)
                if v >= bv:
                    bv, best = v, j
            if best >= 0:
                matched[best] = True
                tp += 1
            else:
                fp += 1
    return tp, fp, n_gt - tp, n_gt


def _thr_for(thresholds, cls, default=0.3):
    if isinstance(thresholds, dict):
        if cls in thresholds:
            return thresholds[cls]
        return thresholds.get(NAMES.get(cls), default)
    return thresholds


def per_class_counts(gt, preds, thresholds, iou_thr=0.5):
    """所有類別的 counts。thresholds 可為單一 float,或 {class_id: thr} / {名: thr}。"""
    out = {}
    for cls in NAMES:
        thr = _thr_for(thresholds, cls)
        out[cls] = counts_for_class(gt, preds, cls, thr, iou_thr)
    return out


def prf(tp, fp, fn):
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f
