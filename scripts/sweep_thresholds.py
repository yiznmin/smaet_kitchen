"""在 valid 集掃描每個類別的最佳信心門檻,輸出 configs/m3_thresholds.json。

策略(依 docs/M3_訓練SOP_迭代式_20260806.md §4、§6):
- 食安關鍵類(刀具/食材/砧板/人/手):在 precision ≥ 下限(預設 0.6)下,選能達到「最大 recall」的最低門檻。
- 一般類:選最大化 F1 的門檻。
- 該類在 valid 沒有樣本(如夾子/手套)→ 用預設門檻,不調。

作法:對 valid 低門檻(0.05)predict 一次拿全部候選,再在記憶體掃門檻(不重跑模型)。
需要環境有 rfdetr + 權重(Colab 或本機 inference env)。

用法:
  python scripts/sweep_thresholds.py --weights out_final/checkpoint_best_regular.pth \\
      --variant nano --valid data/m3_finetune_r2/valid
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from m3.classes import NAMES, FOOD_SAFETY   # noqa: E402
import m3.eval as E                          # noqa: E402

CANDIDATES = [round(0.05 * i, 2) for i in range(1, 19)]   # 0.05 .. 0.90


def pick_threshold(name, sweep, floor, default):
    """sweep = [(thr, p, r, f1)]。回傳選定門檻。"""
    valid_rows = [row for row in sweep if row[3] is not None]
    if not valid_rows:
        return default
    if name in FOOD_SAFETY:
        ok = [row for row in valid_rows if row[1] >= floor]      # precision ≥ floor
        if ok:
            return max(ok, key=lambda r: (r[2], r[1]))[0]        # 最大 recall,tie-break precision
        return max(valid_rows, key=lambda r: r[3])[0]            # 退而求其次:最大 F1
    return max(valid_rows, key=lambda r: r[3])[0]                # 一般類:最大 F1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="微調權重 .pth")
    ap.add_argument("--variant", default="nano", choices=["nano", "small", "medium"])
    ap.add_argument("--valid", default=str(ROOT / "data" / "m3_finetune_r2" / "valid"))
    ap.add_argument("--out", default=str(ROOT / "configs" / "m3_thresholds.json"))
    ap.add_argument("--precision-floor", type=float, default=0.6)
    ap.add_argument("--default", type=float, default=0.3, help="無樣本類別的預設門檻")
    ap.add_argument("--iou", type=float, default=0.5)
    args = ap.parse_args()

    from rfdetr import RFDETRNano, RFDETRSmall, RFDETRMedium
    ctor = {"nano": RFDETRNano, "small": RFDETRSmall, "medium": RFDETRMedium}[args.variant]
    print(f"載入 {args.variant} 權重:{args.weights}")
    model = ctor(pretrain_weights=args.weights, num_classes=len(NAMES))

    print(f"對 valid 低門檻預測一次:{args.valid}")
    gt = E.load_gt(args.valid)
    preds = E.gather_predictions(model, args.valid, base_thr=0.05)

    result, summary = {}, []
    for cls, name in NAMES.items():
        n_gt = sum(1 for objs in gt.values() for (c, _) in objs if c == cls)
        if n_gt == 0:
            result[name] = args.default
            summary.append((name, n_gt, args.default, None, None, None, "無樣本→預設"))
            continue
        sweep = []
        for thr in CANDIDATES:
            tp, fp, fn, _ = E.counts_for_class(gt, preds, cls, thr, args.iou)
            p, r, f = E.prf(tp, fp, fn)
            sweep.append((thr, p, r, f))
        chosen = pick_threshold(name, sweep, args.precision_floor, args.default)
        row = next(s for s in sweep if s[0] == chosen)
        result[name] = chosen
        policy = "食安:P≥%.2f 下最大R" % args.precision_floor if name in FOOD_SAFETY else "最大F1"
        summary.append((name, n_gt, chosen, row[1], row[2], row[3], policy))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n{'類別':<5}{'n':>4}{'門檻':>7}{'P':>7}{'R':>7}{'F1':>7}  策略")
    print("-" * 52)
    for name, n, thr, p, r, f, pol in summary:
        ps = "  -  " if p is None else f"{p:.3f}"
        rs = "  -  " if r is None else f"{r:.3f}"
        fs = "  -  " if f is None else f"{f:.3f}"
        print(f"{name:<5}{n:>4}{thr:>7}{ps:>7}{rs:>7}{fs:>7}  {pol}")
    print(f"\n已輸出:{args.out}")


if __name__ == "__main__":
    main()
