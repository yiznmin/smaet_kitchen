"""自建 CHIRLA 跨鏡頭 Re-ID 訓練集,依 `docs/CHIRLA_自建訓練集_預先登記_20260904.md` §2。

**為什麼要自建**:官方 `_train` 只有 23~65 張、單一實體相機,batch-hard triplet
看不到任何一組跨鏡頭正樣本對(9/1 §8 查明)。而 `annotations/` 有 963,554 個逐幀框、
`videos/` 有 8.95 GB 原始影像 —— 材料一直都在,只是官方切分沒給。

⚠⚠ **規劃時的假設被實測推翻,這裡記下來以免下次又踩**:
   原本打算「只從 seq_000/001/002 裁切」當乾淨來源,但官方 `_gallery`/`_query`
   **用到全部 10 個序列**,沒有任何序列是乾淨的。改用**時間隔離**。

§2.2 先寫死的三條規則(不得在看到結果後調整):
  1. 與**任何**官方 CSV(四種切分都算)裡同 (序列, 實體相機, 身份) 的樣本,
     幀距 **> `--isolate` 秒**(預設 10 秒 = 300 幀 @30fps)
  2. 框寬 ≥ 50px **且** 框高 ≥ 120px
  3. `id > 0`(distractor 取絕對值後仍為負的排除)

⚠ 只做時間隔離**不能**消除身份層級的重疊 —— CHIRLA 只有 21 個人,
  官方 gallery/query 用到全部 21 個。這是 closed-set 協定,
  **不得宣稱證明了跨身份泛化**(§2.3,報告要照抄)。

用法:
    python scripts/chirla_build_reid_trainset.py --root <CHIRLA根> --isolate 10
"""
import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent

MIN_W, MIN_H = 50, 120        # §2.2 規則 2,沿用 chirla_overlap_stats.py 的門檻
FPS = 30.0
SCENARIO = "chirla_selfbuilt"


def phys(name):
    return "_".join(name.split("_")[:2])


def official_frames(root):
    """{(seq, cam, id): np.array(已排序的幀號)} —— 官方 CSV 用到的每一張。

    ⚠ 四種切分都要收,不是只收 gallery/query。`_val` 是開發期用的,
      混進訓練集一樣會讓模型選擇失去意義。
    """
    out = defaultdict(set)
    meta = Path(root) / "benchmark" / "metadata"
    for p in sorted(meta.glob("reid_*.csv")):
        with open(p, encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                fr = int(Path(r["image_path"]).stem.split("_")[-1])
                out[(r["sequence"], phys(r["camera"]), abs(int(r["id"])))].add(fr)
    return {k: np.array(sorted(v)) for k, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--isolate", type=float, default=10.0,
                    help="與官方樣本的最小幀距(秒)。§2.2 定死 10;30 是敏感度檢查")
    ap.add_argument("--out", default=None,
                    help="crop 輸出目錄(預設 results/chirla_selfbuilt/iso<N>s)")
    ap.add_argument("--index-out", default=None)
    ap.add_argument("--max-per-id-cam", type=int, default=400,
                    help="每個 (身份,相機) 最多取幾張,避免少數人主導")
    ap.add_argument("--dry-run", action="store_true", help="只統計,不裁切")
    args = ap.parse_args()

    import cv2

    root = Path(args.root)
    iso_frames = int(args.isolate * FPS)
    out_dir = Path(args.out or f"results/chirla_selfbuilt/iso{int(args.isolate)}s")
    idx_out = Path(args.index_out or f"chirla_selfbuilt_iso{int(args.isolate)}s.json")

    print("=" * 78)
    print(f"自建 CHIRLA 跨鏡頭 Re-ID 訓練集 · 隔離 ±{args.isolate:.0f}s({iso_frames} 幀)")
    print("=" * 78)

    off = official_frames(root)
    print(f"  官方 CSV 涵蓋 {sum(len(v) for v in off.values()):,} 個 (序列,相機,身份,幀)")

    # ── 掃描標註,套用三條規則 ──
    reasons = Counter()
    keep = defaultdict(list)          # (seq, cam, ident) -> [(frame, bbox)]
    for seq_dir in sorted(p for p in (root / "annotations").iterdir() if p.is_dir()):
        seq = seq_dir.name
        for f in sorted(seq_dir.glob("*.json")):
            cam = phys(f.stem)
            for fr, dets in json.loads(f.read_text(encoding="utf-8")).items():
                fr = int(fr)
                for o in dets:
                    ident = int(o["id"])
                    reasons["總框數"] += 1
                    if ident <= 0:
                        reasons["排除:distractor(負號 id)"] += 1
                        continue
                    x1, y1, x2, y2 = map(int, o["BboxP"])
                    if (x2 - x1) < MIN_W or (y2 - y1) < MIN_H:
                        reasons[f"排除:小於 {MIN_W}x{MIN_H}px"] += 1
                        continue
                    bad = off.get((seq, cam, ident))
                    if bad is not None and np.min(np.abs(bad - fr)) <= iso_frames:
                        reasons[f"排除:距官方樣本 <= {args.isolate:.0f}s"] += 1
                        continue
                    keep[(seq, cam, ident)].append((fr, (x1, y1, x2, y2)))
                    reasons["保留"] += 1

    print(f"\n  {'項目':<34}{'數量':>12}{'占比':>9}")
    tot = reasons["總框數"]
    for k, v in reasons.most_common():
        if k == "總框數":
            continue
        print(f"  {k:<34}{v:>12,}{v / tot * 100:>8.1f}%")
    print(f"  {'總框數':<34}{tot:>12,}")

    # 每個 (身份,相機) 取樣上限 —— 均勻抽,不是取前 N 張(前 N 張會全擠在片頭)
    rng = np.random.default_rng(0)
    sampled = {}
    for k, v in keep.items():
        v.sort()
        if len(v) > args.max_per_id_cam:
            pick = np.linspace(0, len(v) - 1, args.max_per_id_cam).astype(int)
            v = [v[i] for i in pick]
        sampled[k] = v
    n_after = sum(len(v) for v in sampled.values())
    ids = sorted({k[2] for k in sampled})
    cams = sorted({k[1] for k in sampled})
    print(f"\n  每個 (身份,相機) 上限 {args.max_per_id_cam} 張後:{n_after:,} 張")
    print(f"  身份 {len(ids)} 個 {ids}")
    print(f"  相機 {len(cams)} 台")

    # 跨鏡頭覆蓋 —— 這才是自建的意義所在
    per_id_cams = defaultdict(set)
    for (_s, cam, i) in sampled:
        per_id_cams[i].add(cam)
    multi = [i for i, c in per_id_cams.items() if len(c) >= 2]
    print(f"  出現在 ≥2 台相機的身份:{len(multi)} / {len(ids)}"
          f"   ← 官方 _train 是 0 / 8(全部來自 camera_3)")

    if args.dry_run:
        print("\n  --dry-run,不裁切。")
        return 0

    # ── 裁切 ──
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, n_written, n_badread = [], 0, 0
    by_seq_cam = defaultdict(list)
    for (seq, cam, ident), v in sampled.items():
        by_seq_cam[(seq, cam)].append((ident, v))

    for (seq, cam), items in sorted(by_seq_cam.items()):
        vids = sorted((root / "videos" / seq).glob(cam + "_*.avi"))
        if not vids:
            print(f"    缺影片:{seq}/{cam}")
            continue
        # 一支影片開一次,依幀號循序讀 —— 每張都 seek 會慢一個數量級
        want = defaultdict(list)      # frame -> [(ident, bbox)]
        for ident, v in items:
            for fr, bb in v:
                want[fr].append((ident, bb))
        cap = cv2.VideoCapture(str(vids[0]))
        fid = 0
        while True:
            ok, img = cap.read()
            if not ok:
                break
            fid += 1                                  # ⚠ 標註是 1-based
            if fid not in want:
                continue
            for ident, (x1, y1, x2, y2) in want[fid]:
                h, w = img.shape[:2]
                crop = img[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
                if crop.size == 0:
                    n_badread += 1
                    continue
                d = out_dir / seq / cam / str(ident)
                d.mkdir(parents=True, exist_ok=True)
                p = d / f"frame_{fid}.png"
                # ⚠ imencode+tofile 而非 imwrite —— 路徑含中文時 imwrite 靜默失敗
                ok2, buf = cv2.imencode(".png", crop)
                if not ok2:
                    n_badread += 1
                    continue
                buf.tofile(str(p))
                rows.append([str(p), str(ident), cam, fid])
                n_written += 1
        cap.release()
        print(f"    {seq}/{cam}: 累計 {n_written:,} 張", flush=True)

    print(f"\n  寫出 {n_written:,} 張,失敗 {n_badread}")

    # ── 索引(格式與 chirla_prep.py --index 相同,路徑用絕對路徑)──
    idx_out.write_text(json.dumps(
        {"root": str(out_dir.resolve()),
         "index": {SCENARIO: {"train": rows, "val": []}}},
        ensure_ascii=False), encoding="utf-8")
    print(f"  → {idx_out}")

    # ── 自檢:與官方樣本的隔離確實成立 ──
    viol = 0
    for p, ident, cam, fid in rows:
        seq = Path(p).parts[-4]
        bad = off.get((seq, cam, int(ident)))
        if bad is not None and np.min(np.abs(bad - fid)) <= iso_frames:
            viol += 1
    print(f"\n  自檢:違反 ±{args.isolate:.0f}s 隔離的樣本 {viol} 個"
          f"{'  [OK]' if viol == 0 else '  [FAIL]'}")
    tr_ids = {r[1] for r in rows}
    tr_cams = {r[2] for r in rows}
    print(f"  自檢:訓練集 {len(tr_ids)} 個身份 / {len(tr_cams)} 台相機"
          f"(官方 _train 是 8 個身份 / 1 台)")
    return 1 if viol else 0


if __name__ == "__main__":
    sys.exit(main())
