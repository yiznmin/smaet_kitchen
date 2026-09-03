"""把 CHIRLA 官方 metadata CSV 轉成 Windows 磁碟上實際可解析的路徑。

為什麼需要這一步(踩過的坑):

1. **冒號**。官方 CSV 的 image_path 內含相機目錄名 `camera_3_2023-06-02-11:14:26`,
   但 NTFS 不允許檔名含 `:`,HuggingFace 下載時會靜默換成 `_`。
   結果是 16,141 列 CSV **一列都對不到磁碟**,而 chirla_prep.py 的抽樣檢查
   會退回 rglob 全樹搜尋 —— 看起來像「找得到」,實際慢到不可用。
2. **前綴**。官方 CSV 的路徑相對於 `benchmark/`(開頭是 `reid/...`),
   但 chirla_prep.py 是拿 `root / image_path` 解析,root 要是資料集根目錄
   (底下有 annotations/ benchmark/ videos/)。所以要補上 `benchmark/` 前綴。

轉完的 CSV 寫到 <data-root>/benchmark/metadata/,原始 CSV 不動。
只改 image_path 一欄,id/camera/sequence/subset 原樣保留 ——
切分本身完全沿用官方,這裡不做任何重切。

用法:
    python scripts/chirla_fix_paths_windows.py \
        --meta "D:/新增資料夾/CHIRLA/benchmark/metadata" \
        --data-root "D:/新增資料夾/CHIRLA/CHIRLA_data/CHIRLA"
"""
import argparse
import csv
import sys
from pathlib import Path


def fix(meta_dir, data_root, prefix="benchmark"):
    meta_dir, data_root = Path(meta_dir), Path(data_root)
    out_dir = data_root / prefix / "metadata"
    out_dir.mkdir(parents=True, exist_ok=True)

    srcs = sorted(meta_dir.glob("reid_*.csv"))
    if not srcs:
        print(f"[FAIL] {meta_dir} 下找不到 reid_*.csv")
        return 1

    print(f"{'CSV':<44}{'列數':>8}{'命中':>8}{'缺檔':>8}")
    print("-" * 68)
    total = missing_total = 0
    for src in srcs:
        rows, miss = [], 0
        with open(src, encoding="utf-8-sig", newline="") as f:
            rdr = csv.DictReader(f)
            fields = rdr.fieldnames
            for r in rdr:
                rel = f"{prefix}/{r['image_path'].replace(':', '_')}"
                if not (data_root / rel).exists():
                    miss += 1
                r["image_path"] = rel
                rows.append(r)
        with open(out_dir / src.name, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
        total += len(rows)
        missing_total += miss
        print(f"{src.name:<44}{len(rows):>8,}{len(rows)-miss:>8,}{miss:>8,}")

    print("-" * 68)
    print(f"{'合計':<44}{total:>8,}{total-missing_total:>8,}{missing_total:>8,}")
    print(f"\n寫出到 {out_dir}")
    if missing_total:
        # 缺檔不是路徑格式問題就是下載不完整,兩種都必須先解決再訓練
        print(f"[FAIL] 有 {missing_total:,} 列在磁碟上找不到對應影像 —— 先確認下載完整性")
        return 1
    print("[OK] 每一列都對得到磁碟上的影像")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", required=True, help="官方 benchmark/metadata 目錄")
    ap.add_argument("--data-root", required=True,
                    help="資料集根目錄(底下有 annotations/ benchmark/ videos/)")
    args = ap.parse_args()
    return fix(args.meta, args.data_root)


if __name__ == "__main__":
    sys.exit(main())
