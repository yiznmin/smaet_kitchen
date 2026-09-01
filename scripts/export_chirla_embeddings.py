"""把模型抽出的特徵匯出成 CHIRLA 官方 benchmark 要求的 HDF5 格式。

**為什麼需要這支**:CHIRLA 的 benchmark 是「你產生 embedding,我算 CMC/mAP」。
不照他們的格式匯出,就無法跑官方的 `evaluate_reid.py`,也就**無法跟論文的數字對照**
(論文最佳基線 ResNet101 是 CMC@1 18.81% / mAP 23.24%,那是我們的參照點)。

官方契約(benchmark/reid/README.md):
    embeddings/{method_name}/{model_name}/{scenario}_{split}_embeddings.h5
      · embeddings : float32 [N, D]
      · ids        : int32   [N]      ⚠ 負數 = distractor
      · paths      : UTF-8 字串 [N]   階層路徑,他們靠它解析 subset

⚠ **ids 必須跨 gallery/query 一致**。同一個人在 gallery 與 query 若拿到不同的整數,
  評估會全錯而且**看起來只是分數很低**,不會報錯。所以 id 對應表由本腳本統一產生
  並存成 JSON,兩邊共用同一份。這是這個格式最容易出錯的地方。

用法:
    # 先確認 id 對應表(所有 split 一起建,保證一致)
    python scripts/export_chirla_embeddings.py --index chirla_index.json \\
        --scenario multi_camera --ckpt model_result/reid/armS/best.pth \\
        --method chirla_armS --model resnet50_bnneck --out embeddings/

    # 之後跑官方評估
    python evaluate_reid.py --gallery embeddings/.../multi_camera_gallery_embeddings.h5 \\
        --query embeddings/.../multi_camera_query_embeddings.h5 --topk 1 5 10 --per-subset
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))


def build_id_map(index_path, scenario):
    """把 identity 字串映成 int32,**所有 split 共用同一份**。

    ⚠ 這是整個格式最容易出錯的地方:若 gallery 與 query 各自建表,
      同一個人會拿到不同整數 → 評估結果全錯,而且症狀只是「分數很低」,
      不會有任何錯誤訊息。所以一次把所有 split 的身份收齊再編號。
    """
    d = json.loads(Path(index_path).read_text(encoding="utf-8"))
    subs = d["index"][scenario]
    idents = sorted({r[1] for rows in subs.values() for r in rows})
    return {s: i for i, s in enumerate(idents)}, d["root"]


def extract(rows, root, embedder, batch=64, quiet=False):
    """讀圖 → 抽特徵。回傳 (feats [N,D] float32, kept_rows)。

    ⚠ 用 cv2.imdecode(np.fromfile(...)) 而非 imread —— 路徑含非 ASCII 時
      imread 在 Windows 上會靜默回 None(本專案既有腳本踩過)。
    """
    import cv2
    feats, kept, buf, buf_rows, bad = [], [], [], [], 0
    for r in rows:
        p = Path(r[0])
        f = p if p.is_absolute() else Path(root) / p
        img = cv2.imdecode(np.fromfile(str(f), dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            bad += 1
            continue
        buf.append(img)
        buf_rows.append(r)
        if len(buf) >= batch:
            feats.append(embedder.extract_batch(buf))
            kept += buf_rows
            buf, buf_rows = [], []
    if buf:
        feats.append(embedder.extract_batch(buf))
        kept += buf_rows
    if bad and not quiet:
        print(f"    ⚠ {bad} 張讀不到,已跳過")
    if not feats:
        return np.zeros((0, embedder.dim), dtype=np.float32), []
    return np.concatenate(feats, 0).astype(np.float32), kept


def write_h5(path, feats, ids, paths):
    import h5py
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as h:
        h.create_dataset("embeddings", data=feats.astype(np.float32))
        h.create_dataset("ids", data=np.asarray(ids, dtype=np.int32))
        # 官方要求 UTF-8 字串。h5py 要用 special_dtype 才存得了變長字串。
        h.create_dataset("paths", data=np.array(paths, dtype=object),
                         dtype=h5py.special_dtype(vlen=str))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True, help="chirla_prep.py --index 的輸出")
    ap.add_argument("--scenario", required=True)
    ap.add_argument("--ckpt", required=True, help="train_reid_chirla.py 的 best.pth")
    ap.add_argument("--method", default="chirla_armS", help="官方路徑的 {method_name}")
    ap.add_argument("--model", default="resnet50_bnneck", help="官方路徑的 {model_name}")
    ap.add_argument("--splits", nargs="+", default=["gallery", "query"],
                    help="要匯出哪些。⚠ 最終數字只能報 gallery vs query;"
                         "train/val 只能用於開發")
    ap.add_argument("--out", default="embeddings")
    ap.add_argument("--device", default=None)
    ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args()

    from m5_reid.chirla_embedder import ChirlaEmbedder

    id_map, root = build_id_map(args.index, args.scenario)
    idx = json.loads(Path(args.index).read_text(encoding="utf-8"))["index"][args.scenario]

    print("=" * 70)
    print(f"匯出 CHIRLA embedding · scenario={args.scenario}")
    print("=" * 70)
    print(f"  身份對應表:{len(id_map)} 個身份(所有 split 共用同一份)")

    emb = ChirlaEmbedder(args.ckpt, device=args.device)
    out_dir = Path(args.out) / args.method / args.model
    manifest = {"scenario": args.scenario, "ckpt": str(args.ckpt),
                "provenance": emb.provenance, "arm": emb.arm, "dim": emb.dim,
                "id_map": id_map, "files": {}}

    for split in args.splits:
        rows = idx.get(split, [])
        if not rows:
            print(f"  ⚠ 沒有 {split} 切分,跳過")
            continue
        print(f"\n  {split}:{len(rows):,} 張 → 抽特徵…")
        feats, kept = extract(rows, root, emb, batch=args.batch)
        ids = [id_map[r[1]] for r in kept]
        paths = [r[0] for r in kept]

        # 自檢:契約要求 L2-normalized,evaluate_reid 靠內積當 cosine
        n = np.linalg.norm(feats, axis=1) if len(feats) else np.array([1.0])
        if abs(n.min() - 1.0) > 1e-3 or abs(n.max() - 1.0) > 1e-3:
            raise SystemExit(f"❌ {split} 的特徵沒有 L2-normalized "
                             f"(範數 {n.min():.4f}~{n.max():.4f})—— "
                             "embedder 契約被破壞,所有相似度都會是錯的")

        f = out_dir / f"{args.scenario}_{split}_embeddings.h5"
        write_h5(f, feats, ids, paths)
        manifest["files"][split] = str(f)
        print(f"    {feats.shape} · {len(set(ids))} 個身份 · L2 範數 {n.min():.5f}")
        print(f"    → {f}")

    (out_dir / f"{args.scenario}_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n  權重來源:{emb.provenance}")
    if emb.arm == "R":
        print("  ⚠ arm R 的起始權重是研究限定的外部 Re-ID 權重,**這組數字不可用於出貨主張**")
    print(f"  manifest → {out_dir}/{args.scenario}_manifest.json")
    print("\n下一步(在 CHIRLA repo 裡):")
    print(f"  python evaluate_reid.py \\")
    print(f"      --gallery {out_dir}/{args.scenario}_gallery_embeddings.h5 \\")
    print(f"      --query   {out_dir}/{args.scenario}_query_embeddings.h5 \\")
    print(f"      --topk 1 5 10 --per-subset")
    return 0


if __name__ == "__main__":
    sys.exit(main())
