"""用 EPFL 同步多視角驗證 M5 Re-ID(廚房域)。

EPFL 每個場次單人、9 台鏡頭同步 → **同一時刻不同視角 = 同一人(天生 ground truth)**;
不同場次(不同參與者)= 不同身份。用它量:
  1. 跨視角一致性:同一人、同一時刻、不同視角的特徵有多像(視角不變性)。
  2. 身份判別 Re-ID:Rank-1 / mAP over 場次(排除同 場次+同 視角)。
  3. M5 綁定:餵 IdentityManager 掃門檻,量綁對/誤併(重用 market1501 metric)。

輸入:同步幀 <tag>_<view>_f<fid>.jpg(tag=場次=身份, view=相機, fid=時刻)。
  預設用現成的 data/m3_finetune/round2_images(已有 s1/s2 兩身份);
  更多身份用 scripts/prep_m5_epfl.py 產生。

需要 rfdetr(person 偵測)+ torch(dinov2/osnet)。⚠ EPFL/OSNet 研究限定,僅驗證不出貨。

用法:
  python scripts/reid_eval_epfl.py --embedder dinov2
  python scripts/reid_eval_epfl.py --frames data/m5_reid_epfl/frames --embedder dinov2
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import reid_eval_market1501 as rm       # noqa: E402  複用 evaluate_cmc_map/gallery_reps/binding_sweep
from m5_reid.embedder import l2norm     # noqa: E402

_FN = re.compile(r"(s\d+)_([A-Za-z0-9]+)_f(\d+)")   # s1_Aoutput0_f01170.jpg


def parse_frames(frames_dir):
    """回傳 [(path, identity, view, fid)]。identity=場次 tag、view=鏡頭、fid=時刻。"""
    out = []
    for p in sorted(Path(frames_dir).glob("*.jpg")):
        m = _FN.match(p.name)
        if m:
            out.append((str(p), m.group(1), m.group(2), int(m.group(3))))
    return out


def detect_and_crop(records, weights=None, variant="nano", thr=0.25, save_dir=None):
    """對每幀偵測 person、取最高信心框裁切。回傳對齊的 crops 與 meta。

    weights 給微調權重(11 類,人=class 0)→ 在 EPFL 俯視視角偵測穩定(推薦);
    不給則用 COCO 預訓,人以 class_name=='person' 判定(俯視偏弱)。
    """
    import cv2
    from PIL import Image
    from rfdetr import RFDETRNano, RFDETRSmall, RFDETRMedium
    ctor = {"nano": RFDETRNano, "small": RFDETRSmall, "medium": RFDETRMedium}[variant]
    if weights:
        model = ctor(pretrain_weights=weights, num_classes=11)
        use_name = False                       # 微調:人 = class_id 0
    else:
        model = ctor()
        use_name = True                        # COCO:以 class_name 'person' 判定
    crops, meta = [], []
    saved = 0
    for path, ident, view, fid in records:
        bgr = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        det = model.predict(Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)), threshold=thr)
        names = det.data.get("class_name") if (use_name and det.data) else None
        best_i, best_c = -1, -1.0
        for i in range(len(det)):
            is_person = (str(names[i]) == "person") if use_name else (int(det.class_id[i]) == 0)
            if not is_person:
                continue
            c = float(det.confidence[i]) if det.confidence is not None else 0.0
            if c > best_c:
                best_c, best_i = c, i
        if best_i < 0:
            continue
        x1, y1, x2, y2 = [int(v) for v in det.xyxy[best_i]]
        x1, y1 = max(0, x1), max(0, y1)
        crop = bgr[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        crops.append(crop)
        meta.append((ident, view, fid))
        if save_dir and saved < 12:
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(Path(save_dir) / f"{ident}_{view}_f{fid}.jpg"), crop)
            saved += 1
    return crops, meta


def cross_view_consistency(feats, meta):
    """同一人、同一時刻、不同視角的平均 cosine(within);對照不同人平均(across)。"""
    from collections import defaultdict
    groups = defaultdict(list)
    for i, (ident, view, fid) in enumerate(meta):
        groups[(ident, fid)].append(i)
    within = []
    for idxs in groups.values():
        if len(idxs) < 2:
            continue
        f = feats[idxs]
        sim = f @ f.T
        iu = np.triu_indices(len(idxs), k=1)
        within.append(float(sim[iu].mean()))
    idents = np.array([m[0] for m in meta])
    sims = feats @ feats.T
    n = len(feats)
    iu = np.triu_indices(n, k=1)
    same = idents[iu[0]] == idents[iu[1]]
    return {"within_id_crossview_mean": round(float(np.mean(within)), 4) if within else None,
            "same_id_pair_mean": round(float(sims[iu][same].mean()), 4),
            "diff_id_pair_mean": round(float(sims[iu][~same].mean()), 4),
            "n_groups_multiview": len(within)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", default=str(ROOT / "data" / "m3_finetune" / "round2_images"))
    ap.add_argument("--embedder", required=True, choices=["dinov2", "osnet"])
    ap.add_argument("--model-name", default="")
    ap.add_argument("--variant", default="nano", help="RF-DETR person 偵測用")
    ap.add_argument("--det-weights", default=str(ROOT / "model_result" / "nano" / "checkpoint_best_regular.pth"),
                    help="偵測 person 用的微調權重(人=class0);設 '' 用 COCO")
    ap.add_argument("--device", default=None)
    ap.add_argument("--save-crops", default=str(ROOT / "results" / "m5_reid" / "crops"))
    ap.add_argument("--out", default=str(ROOT / "reid_epfl_results.json"))
    args = ap.parse_args()

    records = parse_frames(args.frames)
    idents = sorted(set(r[1] for r in records))
    print(f"幀 {len(records)} 張、身份(場次){len(idents)}:{idents}")
    if len(idents) < 2:
        print("⚠ 少於 2 個身份,判別指標意義有限(仍可看跨視角一致性)。")

    dw = args.det_weights if args.det_weights and Path(args.det_weights).exists() else None
    print(f"RF-DETR 偵測 person + 裁切 …({'微調權重' if dw else 'COCO'})")
    crops, meta = detect_and_crop(records, weights=dw, variant=args.variant, save_dir=args.save_crops)
    print(f"  裁出 {len(crops)} 個 person crop(存樣本於 {args.save_crops})")
    if len(crops) < 2:
        print("⚠ 有效 person crop < 2,無法評估。請確認偵測權重/門檻或換視角。")
        return

    emb = rm.build_embedder(args.embedder, args.model_name, args.device)
    print(f"embedder = {args.embedder}({args.model_name or 'default'}), dim={emb.dim};抽特徵 …")
    feats = emb.extract_batch(crops)
    feats = np.stack([l2norm(f) for f in feats])

    ident_ids = np.array([{v: k for k, v in enumerate(idents)}[m[0]] for m in meta])
    view_ids = np.array([hash(m[1]) % 100000 for m in meta])

    cv = cross_view_consistency(feats, meta)
    cmc = rm.evaluate_cmc_map(feats, ident_ids, view_ids, feats, ident_ids, view_ids)
    reps, rep_ids = rm.gallery_reps(feats, ident_ids)
    sweep = rm.binding_sweep(feats, ident_ids, reps, rep_ids,
                             [round(0.05 * i, 2) for i in range(6, 19)])   # 0.30..0.90
    best = max(sweep, key=lambda r: r["accuracy"] - r["false_merge"])

    print("\n=== 跨視角一致性(同人不同視角特徵像不像)===")
    print(cv)
    print("\n=== 身份判別(Rank-1/mAP over 場次)===")
    print(cmc)
    print("\n=== chef_id 綁定(掃門檻)===")
    print(f"{'門檻':>6}{'綁定正確':>10}{'誤併':>8}{'漏綁':>8}")
    for r in sweep:
        print(f"{r['thr']:>6}{r['accuracy']:>10}{r['false_merge']:>8}{r['reject']:>8}")
    print(f"最佳門檻 ~ {best['thr']}(正確 {best['accuracy']}、誤併 {best['false_merge']})")

    out = {"embedder": args.embedder, "model_name": args.model_name or "default",
           "num_identities": len(idents), "num_crops": len(crops),
           "cross_view": cv, "cmc_map": cmc, "binding_sweep": sweep, "best_binding": best}
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已輸出 {args.out}")


if __name__ == "__main__":
    main()
