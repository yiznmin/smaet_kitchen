"""在本機微調 M3(RF-DETR)—— notebooks/m3_experiment_staged_vs_epfl.ipynb 的 Arm A 本機版。

為什麼需要這支:文件上 F1 0.81 的 Arm A / B / C 權重只存在 Colab,沒有帶回來;
repo 的 model_result/nano 是 2026-07-23 一鍵比較那次的早期小資料權重(round2 test F1 約 0.6)。
見 integrated/交接說明.md 4.3 節。

超參數預設值與 notebook 的 Arm A 逐項相同(60 epochs / 704 / batch 4 / grad_accum 4 / lr 1e-4)。

需要訓練用的相依(pytorch_lightning、pycocotools 等),專案的 .venv 只有推論版:
    pip install "rfdetr[train,loggers]==1.9.4"
為了不動到推論環境,建議裝在獨立的虛擬環境。

資料集目錄須為 rfdetr 的 COCO 格式:<dataset>/{train,valid,test}/_annotations.coco.json。

用法:
    python scripts/train_m3_rfdetr.py --dataset ../m3_fintune_data/m3_finetune_r2 \\
        --out model_result/nano_r2_armA

注意:訓練資料含 EPFL(CC BY-NC),產出的權重只能用於驗證,不可出貨。
"""
import argparse
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--variant", default="nano", choices=["nano", "small", "medium"])
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--grad-accum-steps", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--resolution", type=int, default=704, help="須為 32 的倍數")
    # Windows 的 DataLoader 用 spawn 開子行程,worker 太多時啟動慢且偶爾卡住
    ap.add_argument("--num-workers", type=int, default=2)
    args = ap.parse_args()

    from rfdetr import RFDETRMedium, RFDETRNano, RFDETRSmall
    ctor = {"nano": RFDETRNano, "small": RFDETRSmall, "medium": RFDETRMedium}[args.variant]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    model = ctor()      # COCO 預訓起始
    model.train(dataset_dir=str(Path(args.dataset).resolve()), epochs=args.epochs,
                batch_size=args.batch_size, grad_accum_steps=args.grad_accum_steps,
                lr=args.lr, resolution=args.resolution, output_dir=str(out.resolve()),
                num_workers=args.num_workers)

    # 這顆權重是怎麼來的 —— 現有兩顆就是因為沒留這份紀錄,事後只能從步數反推
    (out / "train_args.json").write_text(json.dumps(
        dict(vars(args), wall_minutes=round((time.time() - t0) / 60, 1),
             start_weights="COCO 預訓(rfdetr 內建)",
             license_note="訓練資料含 EPFL(CC BY-NC),不可出貨"),
        ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"完成,耗時 {(time.time() - t0) / 60:.1f} 分;輸出 {out}")


if __name__ == "__main__":
    main()
