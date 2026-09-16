"""⚠⚠ 這不是真值。這是本機排練用的假標註,**任何由它算出的指標都不是效能評估**。

## 為什麼存在

2026-09-16 使用者要求:三個難度層級的交付先在本機用 EPFL 走一遍流程再上遠端。
但 **EPFL 沒有逐幀標註** —— `data/epfl/` 只有 9 支影片,沒有人工標框。
挑窗程式吃的是 CHIRLA 格式的逐幀標註,所以要排練就得先造一份。

這支從**已經跑完的九鏡頭全長結果**(`results/m5_full9/tracks.csv`)造出 CHIRLA 格式的
假標註,身份一律填 1。

  · 身份 = 1 這件事**是真的** —— EPFL 全片只有一個人,這是資料集的事實
  · 框的位置**來自追蹤器自己** —— 所以「人在不在畫面上」是系統說了算

→ **循環論證**:拿系統的輸出當真值,系統當然跟自己一致。
  所以排練只能證明「工具會跑、影片長這樣」,**不能證明追蹤準不準**。
  要證明追蹤準不準,需要**人工標註的移動軌跡**,而那只有 CHIRLA 有。

## 但有兩件事不用標註也能驗(這兩個是真的)

  1. **每一幀應該恰好有 1 條 track** —— 0 條 = 漏偵、≥2 條 = 幽靈軌跡。
     判準來自「全片只有一個人」這個事實,不是來自系統。
  2. **九台鏡頭應該綁成同一個身份** —— 跨鏡頭認人對不對,這個看得出來。

這兩項本支會直接算給你看。

用法:
    python scripts/epfl_rehearsal_gt.py --run-dir results/m5_full9 --out-root <排練目錄>
"""
import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default=str(ROOT / "results" / "m5_full9"))
    ap.add_argument("--out-root", required=True, help="假標註要寫到哪(⚠ 不要放進 repo)")
    ap.add_argument("--seq", default="epfl9", help="假序列名")
    args = ap.parse_args()

    run = Path(args.run_dir)
    meta = json.loads((run / "run_meta.json").read_text(encoding="utf-8"))
    stride = int(meta["stride"])

    per_cam = defaultdict(dict)          # cam -> {frame(1-based): [det]}
    box_per_loop = defaultdict(Counter)  # cam -> {該迴圈的框數: 次數}
    chef_ids = defaultdict(set)
    loops = defaultdict(set)
    with open(run / "tracks.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            cam, loop = r["camera_id"], int(r["loop_i"])
            loops[cam].add(loop)
            if r["x1"] == "":                 # 卡爾曼預測框,沒有實際偵測
                continue
            fid = int(r["video_fid"])
            box = [float(r["x1"]), float(r["y1"]), float(r["x2"]), float(r["y2"])]
            # ⚠ GT 幀號是 1-based,video_fid 是 0-based
            per_cam[cam].setdefault(str(fid + 1), []).append({"id": 1, "BboxP": box})
            if r.get("chef_id"):
                chef_ids[cam].add(r["chef_id"])

    for cam, frames in per_cam.items():
        c = Counter()
        for dets in frames.values():
            c[len(dets)] += 1
        n_loops = len(loops[cam])
        c[0] = n_loops - sum(c.values())      # 該迴圈完全沒有框 = 漏偵
        box_per_loop[cam] = c

    aroot = Path(args.out_root) / "annotations" / args.seq
    aroot.mkdir(parents=True, exist_ok=True)
    for cam, frames in per_cam.items():
        # ⚠ 檔名只能是鏡頭名,不可加序列後綴。`eval_m4m5_chirla.phys()` 取**前兩段底線**
        #   當實體鏡頭名(CHIRLA 是 `camera_1_時間戳` → `camera_1`)。
        #   若寫成 `cam1_epfl9.json`,phys 會回 `cam1_epfl9`,而 run_meta 裡只有 `cam1`
        #   → 挑出來的窗帶著不存在的鏡頭名,渲染器直接報錯。
        (aroot / f"{cam}.json").write_text(json.dumps(frames), encoding="utf-8")

    print("=" * 74)
    print("⚠ 這是**排練用假標註**,不是真值。由它算出的任何指標都不是效能評估。")
    print("=" * 74)
    print(f"\n寫出 {len(per_cam)} 台鏡頭的假標註 → {aroot}")
    print(f"stride={stride},假序列名 {args.seq}\n")

    print("【真的能驗的第 1 項】每個取樣迴圈應該恰好 1 條 track(EPFL 全片單人)")
    print(f"  {'鏡頭':<8}{'迴圈':>7}{'0 條(漏偵)':>14}{'1 條':>9}{'≥2 條(幽靈)':>14}")
    tot = Counter()
    for cam in sorted(box_per_loop):
        c = box_per_loop[cam]
        ge2 = sum(v for k, v in c.items() if k >= 2)
        n = len(loops[cam])
        tot.update({"n": n, "zero": c[0], "one": c[1], "ge2": ge2})
        print(f"  {cam:<8}{n:>7}{c[0]:>8}({c[0] / n:>5.1%}){c[1]:>9}"
              f"{ge2:>8}({ge2 / n:>5.1%})")
    print(f"  {'合計':<8}{tot['n']:>7}{tot['zero']:>8}({tot['zero'] / tot['n']:>5.1%})"
          f"{tot['one']:>9}{tot['ge2']:>8}({tot['ge2'] / tot['n']:>5.1%})")

    print("\n【真的能驗的第 2 項】九台鏡頭應該綁成同一個身份")
    allc = set().union(*chef_ids.values()) if chef_ids else set()
    for cam in sorted(chef_ids):
        print(f"  {cam:<8}chef_id {sorted(chef_ids[cam])}")
    ok = len(allc) == 1
    print(f"  → 全部鏡頭的身份集合 = {sorted(allc)} … {'✅ 一致' if ok else '❌ 不一致'}")

    print("\n【量不到的】IDF1、ID 切換、框得準不準、track 跟的是人還是倒影")
    print("  這些需要**人工標註的移動軌跡**,EPFL 沒有 → 只能在 CHIRLA 上量。")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
