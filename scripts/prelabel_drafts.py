"""模型預標草稿(B,內部測試):M3 / SAM 3 / 合併 → COCO 草稿,與人工框比對算修正量。

預先登記:docs/預標草稿_預先登記_20261027.md
資料:D:/yizhen/m3_fintune_data/m3_finetune_r2/{valid,test}
  valid:選 SAM 3 逐類門檻與合併範圍 → results/prelabel/frozen.json
  test :讀 frozen.json,只跑一次

用法:
    python scripts/prelabel_drafts.py --split valid
    python scripts/prelabel_drafts.py --split test
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "integrated/src"))
sys.path.insert(0, str(ROOT / "src"))

from m3.classes import CLASSES, NAME_TO_CLASS                 # noqa: E402
import m3.eval as E                                              # noqa: E402

DATA = Path("D:/yizhen/m3_fintune_data/m3_finetune_r2")
OUT = ROOT / "results/prelabel"
M3_W = ROOT / "integrated/weights/m3_nano_r2_armA/checkpoint_best_total.pth"
SAM3_W = ROOT / "integrated/weights/sam3"
ALL = CLASSES + ["手機", "衛生紙"]                                  # 草稿類別:11 類 + 新類
PROMPTS = {"人": ["person"], "刀具": ["knife"], "砧板": ["cutting board"], "食材": ["food", "vegetable"],
           "鍋鏟": ["spatula"], "鍋子": ["pot", "pan"], "手": ["hand"], "容器": ["bowl", "plate", "container"],
           "抹布": ["sponge", "towel"], "夾子": ["tongs"], "手套": ["glove"], "手機": ["phone"], "衛生紙": ["tissue"]}
SAM_RAW_THR = 0.2
SAM_GRID = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
SCOPES = {"cloth_hand": ["抹布", "手"], "all": CLASSES}
MATCH_IOU, TIGHT_IOU = 0.5, 0.75


def iou_mat(a, b):
    a, b = np.asarray(a, float).reshape(-1, 4), np.asarray(b, float).reshape(-1, 4)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    ar = lambda z: (z[:, 2] - z[:, 0]) * (z[:, 3] - z[:, 1])
    return inter / np.maximum(ar(a)[:, None] + ar(b)[None, :] - inter, 1e-9)


def nms(rows, thr=0.5):
    """rows:[(cls, s, box)];同類 NMS。"""
    keep = []
    for r in sorted(rows, key=lambda r: -r[1]):
        if all(k[0] != r[0] or iou_mat(k[2], r[2])[0, 0] < thr for k in keep):
            keep.append(r)
    return keep


# ── 跑模型(原始框存檔,之後選門檻不必重跑)─────────────────────────

def run_models(split):
    from PIL import Image
    import torch
    from transformers import Sam3Model, Sam3Processor
    from m3.model import load_rfdetr
    gt = E.load_gt(str(DATA / split))
    m3 = load_rfdetr("nano", M3_W, "auto")
    sm = Sam3Model.from_pretrained(SAM3_W).to("cuda").eval()
    sp = Sam3Processor.from_pretrained(SAM3_W)
    raw = {}
    for k, fn in enumerate(sorted(gt)):
        img = Image.open(DATA / split / fn).convert("RGB")
        det = m3.predict(img, threshold=0.05)
        m3_rows = [(int(det.class_id[i]), float(det.confidence[i]), [float(v) for v in det.xyxy[i]])
                   for i in range(len(det))]
        sam_rows = []
        for name, texts in PROMPTS.items():
            c = ALL.index(name)
            rows = []
            for t in texts:
                inp = sp(images=img, text=t, return_tensors="pt").to("cuda")
                with torch.inference_mode():
                    out = sm(**inp)
                r = sp.post_process_instance_segmentation(out, threshold=SAM_RAW_THR, mask_threshold=0.5,
                                                          target_sizes=inp["original_sizes"].tolist())[0]
                rows += [(c, float(s), [float(v) for v in b.tolist()]) for b, s in zip(r["boxes"], r["scores"])]
            sam_rows += nms(rows)
        raw[fn] = dict(m3=m3_rows, sam3=sam_rows)
        if k % 10 == 0:
            print(f"  {k + 1}/{len(gt)} {fn}:M3 {len(m3_rows)}、SAM 3 {len(sam_rows)}", flush=True)
    return raw


# ── 草稿組合 ─────────────────────────────────────────────────────

def m3_thresholds():
    import yaml
    cfg = yaml.safe_load((ROOT / "integrated/pipeline_config.yaml").read_text(encoding="utf-8"))
    return {NAME_TO_CLASS[n]: float(t) for n, t in cfg["m3"]["class_thresholds"].items()}


def draft_A(r, m3_thr):
    return [x for x in r["m3"] if x[1] >= m3_thr[x[0]]]


def draft_S(r, sam_thr):
    return [x for x in r["sam3"] if x[1] >= sam_thr.get(ALL[x[0]], 0.5)]


def cloth_filter(sam_rows, m3_rows):
    """SAM 3 抹布框若與 M3 容器框(信心 ≥ 0.4)IoU ≥ 0.5 → 丟掉。"""
    cont = [b for c, s, b in m3_rows if c == NAME_TO_CLASS["容器"] and s >= 0.4]
    cloth = NAME_TO_CLASS["抹布"]
    return [x for x in sam_rows if x[0] != cloth or not cont or iou_mat(x[2], cont).max() < 0.5]


def draft_C(r, m3_thr, sam_thr, scope):
    a = draft_A(r, m3_thr)
    add_cls = {ALL.index(n) for n in SCOPES[scope]} | {ALL.index("手機"), ALL.index("衛生紙")}
    out = list(a)
    for x in cloth_filter(draft_S(r, sam_thr), r["m3"]):
        if x[0] not in add_cls:
            continue
        same = [b for c, s, b in a if c == x[0]]
        if not same or iou_mat(x[2], same).max() < MATCH_IOU:
            out.append(x)
    return out


# ── 計分 ─────────────────────────────────────────────────────────

def match(gt, drafts, classes):
    """逐類 tp / fp / fn / loose(配對但 IoU < 0.75)。"""
    res = {c: dict(tp=0, fp=0, fn=0, loose=0, n_gt=0) for c in classes}
    for fn, objs in gt.items():
        for c in classes:
            g = [b for cc, b in objs if cc == c]
            p = sorted([x for x in drafts.get(fn, []) if x[0] == c], key=lambda x: -x[1])
            res[c]["n_gt"] += len(g)
            used = np.zeros(len(g), bool)
            for _, _, b in p:
                if len(g):
                    io = iou_mat(b, g)[0]
                    io[used] = -1
                    j = int(io.argmax())
                    if io[j] >= MATCH_IOU:
                        used[j] = True
                        res[c]["tp"] += 1
                        res[c]["loose"] += int(io[j] < TIGHT_IOU)
                        continue
                res[c]["fp"] += 1
            res[c]["fn"] += int((~used).sum())
    return res


def prf(d):
    p = d["tp"] / max(d["tp"] + d["fp"], 1)
    r = d["tp"] / max(d["n_gt"], 1)
    return p, r, 2 * p * r / max(p + r, 1e-9)


def summarize(res):
    tot = {k: sum(d[k] for d in res.values()) for k in ("tp", "fp", "fn", "loose", "n_gt")}
    p, r, f = prf(tot)
    return dict(n_gt=tot["n_gt"], tp=tot["tp"], fp=tot["fp"], fn=tot["fn"], loose=tot["loose"],
                P=round(p, 3), R=round(r, 3), F1=round(f, 3),
                corr_rate=round((tot["fp"] + tot["fn"]) / max(tot["n_gt"], 1), 3))


def scored_classes(gt):
    return sorted({c for objs in gt.values() for c, _ in objs})


def pick_sam_thr(gt, raw):
    """每類在 valid 上取 F1 最高的門檻;沒有人工框的類別用 0.5。"""
    out = {}
    for name in ALL:
        c = ALL.index(name)
        best = (0.5, -1)
        for t in SAM_GRID:
            d = match(gt, {fn: [x for x in r["sam3"] if x[0] == c and x[1] >= t] for fn, r in raw.items()}, [c])[c]
            if d["n_gt"] == 0:
                break
            f = prf(d)[2]
            if f > best[1]:
                best = (t, f)
        out[name] = best[0]
    return out


def to_coco(split, drafts):
    src = json.loads((DATA / split / "_annotations.coco.json").read_text(encoding="utf-8"))
    fid = {im["file_name"]: im["id"] for im in src["images"]}
    anns, k = [], 1
    for fn, rows in drafts.items():
        for c, s, (x1, y1, x2, y2) in rows:
            anns.append(dict(id=k, image_id=fid[fn], category_id=c + 1, bbox=[round(x1, 1), round(y1, 1),
                             round(x2 - x1, 1), round(y2 - y1, 1)], area=round((x2 - x1) * (y2 - y1), 1),
                             iscrowd=0, score=round(s, 3)))
            k += 1
    return dict(images=src["images"], annotations=anns,
                categories=[dict(id=i + 1, name=n) for i, n in enumerate(ALL)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["valid", "test"], required=True)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    raw_p = OUT / f"{args.split}_raw.json"
    if raw_p.exists():
        raw = json.loads(raw_p.read_text(encoding="utf-8"))
    else:
        raw = run_models(args.split)
        raw_p.write_text(json.dumps(raw), encoding="utf-8")
    print(f"模型完成 {time.time() - t0:.0f} 秒", flush=True)

    gt = E.load_gt(str(DATA / args.split))
    classes = scored_classes(gt)
    m3_thr = m3_thresholds()
    frozen_p = OUT / "frozen.json"
    if args.split == "valid":
        sam_thr = pick_sam_thr(gt, raw)
        scope_score = {}
        for sc in SCOPES:
            d = {fn: draft_C(r, m3_thr, sam_thr, sc) for fn, r in raw.items()}
            scope_score[sc] = summarize(match(gt, d, classes))["corr_rate"]
        scope = min(scope_score, key=scope_score.get)
        frozen = dict(sam_thr=sam_thr, scope=scope, scope_corr_rate_valid=scope_score)
        frozen_p.write_text(json.dumps(frozen, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"SAM 3 門檻 {sam_thr}\n合併範圍 {scope}(valid 修正率 {scope_score})")
    else:
        frozen = json.loads(frozen_p.read_text(encoding="utf-8"))
        sam_thr, scope = frozen["sam_thr"], frozen["scope"]

    groups = {"A": lambda r: draft_A(r, m3_thr), "S": lambda r: cloth_filter(draft_S(r, sam_thr), r["m3"]),
              "C": lambda r: draft_C(r, m3_thr, sam_thr, scope)}
    rows, summ = [], {}
    for g, fn_ in groups.items():
        drafts = {fn: fn_(r) for fn, r in raw.items()}
        res = match(gt, drafts, classes)
        summ[g] = summarize(res)
        for c in classes:
            p, r, f = prf(res[c])
            rows.append(dict(group=g, cls=ALL[c], **res[c], P=round(p, 3), R=round(r, 3), F1=round(f, 3)))
        rows.append(dict(group=g, cls="全部", **{k: summ[g][k] for k in ("tp", "fp", "fn", "loose", "n_gt")},
                         P=summ[g]["P"], R=summ[g]["R"], F1=summ[g]["F1"]))
        new = {n: sum(x[0] == ALL.index(n) for v in drafts.values() for x in v) for n in ("手機", "衛生紙")}
        summ[g]["new_class_boxes"] = new
        (OUT / f"{args.split}_draft_{g}.json").write_text(json.dumps(to_coco(args.split, drafts), ensure_ascii=False),
                                                         encoding="utf-8")
        print(f"  {g}:P {summ[g]['P']:.3f}  R {summ[g]['R']:.3f}  F1 {summ[g]['F1']:.3f}  "
              f"補 {summ[g]['fn']}  刪 {summ[g]['fp']}  修正率 {summ[g]['corr_rate']:.3f}  "
              f"要拉邊 {summ[g]['loose']}  新類框 {new}", flush=True)
    with open(OUT / f"{args.split}_metrics.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (OUT / f"{args.split}_summary.json").write_text(json.dumps(dict(summary=summ, frozen=frozen), ensure_ascii=False,
                                                               indent=2), encoding="utf-8")
    print(f"輸出 {OUT},{time.time() - t0:.0f} 秒")


if __name__ == "__main__":
    main()
