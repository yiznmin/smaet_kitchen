"""量 CHIRLA 七台相機到底有沒有重疊視野,以及重疊到什麼程度。

**為什麼要量這個**:專案文件從 2026-09-01 起一路寫著「CHIRLA 是 7 台**非重疊**
室內鏡頭」(照抄論文的規格描述),`handoff/給遠端Claude.md` 更據此推論
「CHIRLA 一切都走轉場路徑 → 地面校正與軌跡的架構缺口會直接顯現」。
2026-09-03 把畫面調出來看,發現 **cam2 與 cam3 根本在拍同一個房間**。
那條推論因此至少對一部分鏡頭對不成立。

**判準**:如果兩台真的重疊,同一個人會在**同一幀**同時出現在兩台的標註裡。
資料同步性沒問題 —— 同一序列的七支影片起始時間戳相同、幀數差 10 幀內(0.3 秒)。

⚠ 但「同時出現」還不夠,要再看**框的大小**:
  · 兩邊的框都大 → 兩台真的在拍同一個空間
  · 一邊大一邊是細長條 → 只是隔著門口/走道遠遠看到,不是同一個房間
  只看共現次數會把「共用一道門」誤判成「同一個房間」。cam1+cam3 共現
  11 萬次,但小框寬中位只有 58px(cam2+cam3 是 133px)—— 兩者性質完全不同。

輸出 JSON 與表格。用法:
    python scripts/chirla_overlap_stats.py --root "D:/新增資料夾/CHIRLA/CHIRLA_data/CHIRLA"
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

# 「兩邊都拍得清楚」的門檻。取 Re-ID 慣用輸入 128x256 的一半,
# 低於這個尺寸的 crop 放大到訓練解析度後細節已經糊掉。
GOOD_W, GOOD_H = 50, 120


def phys(name):
    return "_".join(name.split("_")[:2])


def collect(root):
    """回傳 per_seq 統計與 pair 統計。"""
    aroot = Path(root) / "annotations"
    if not aroot.exists():
        raise SystemExit(f"找不到 {aroot} —— 這支需要逐幀標註")

    per_seq, pair = {}, defaultdict(list)
    n_cams_hist = defaultdict(int)
    for seq in sorted(p for p in aroot.iterdir() if p.is_dir()):
        seen = defaultdict(dict)            # (frame, id) -> cam -> bbox
        for f in sorted(seq.glob("*.json")):
            cam = phys(f.stem)
            for fr, dets in json.loads(f.read_text(encoding="utf-8")).items():
                for o in dets:
                    seen[(int(fr), int(o["id"]))][cam] = o["BboxP"]
        one = multi = 0
        for _k, v in seen.items():
            n_cams_hist[len(v)] += 1
            if len(v) < 2:
                one += 1
                continue
            multi += 1
            cams = sorted(v)
            for i in range(len(cams)):
                for j in range(i + 1, len(cams)):
                    a, b = v[cams[i]], v[cams[j]]
                    wa, ha = a[2] - a[0], a[3] - a[1]
                    wb, hb = b[2] - b[0], b[3] - b[1]
                    sm, lg = ((wa, ha), (wb, hb)) if wa * ha < wb * hb else ((wb, hb), (wa, ha))
                    pair[(cams[i], cams[j])].append((*sm, *lg))
        per_seq[seq.name] = dict(total=one + multi, single=one, multi=multi)
    return per_seq, pair, dict(n_cams_hist)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", default="results/m5_reid/chirla_overlap.json")
    args = ap.parse_args()

    per_seq, pair, hist = collect(args.root)
    tot = sum(v["total"] for v in per_seq.values())
    mul = sum(v["multi"] for v in per_seq.values())

    print("=" * 78)
    print("CHIRLA 鏡頭重疊實測 —— 同一幀、同一身份,被幾台相機同時看到")
    print("=" * 78)
    print(f"\n{'序列':<10}{'(幀,身份) 組合':>16}{'只被1台':>12}{'被>=2台':>12}{'重疊率':>10}")
    print("-" * 62)
    for s, v in per_seq.items():
        print(f"{s:<10}{v['total']:>16,}{v['single']:>12,}{v['multi']:>12,}"
              f"{v['multi']/v['total']*100:>9.1f}%")
    print("-" * 62)
    print(f"{'合計':<10}{tot:>16,}{tot-mul:>12,}{mul:>12,}{mul/tot*100:>9.1f}%")
    print(f"\n同時被 N 台看到的分布:{dict(sorted(hist.items()))}")

    print(f"\n{'相機對':<22}{'共現次數':>10}{'小框寬':>8}{'小框高':>8}"
          f"{'兩邊都>=' + f'{GOOD_W}x{GOOD_H}':>18}{'  判定'}")
    print("-" * 82)
    rows, out_pairs = [], {}
    for k, v in pair.items():
        a = np.array(v)
        n = len(a)
        good = int(((a[:, 0] >= GOOD_W) & (a[:, 1] >= GOOD_H)).sum())
        pct = good / n * 100
        # 判定規則:共現要夠多**且**兩邊都拍得清楚的比例要高,才算同一個空間。
        # 只有其中一項的話,多半是隔著門口互看。
        verdict = ("同一空間" if pct >= 60 and n >= 500 else
                   "門口/走道互看" if n >= 500 else
                   "偶發(樣本太少,不下判斷)")
        rows.append((n, k, np.median(a[:, 0]), np.median(a[:, 1]), good, pct, verdict))
        out_pairs[f"{k[0]}+{k[1]}"] = dict(
            n=n, median_small_w=float(np.median(a[:, 0])),
            median_small_h=float(np.median(a[:, 1])),
            n_both_good=good, pct_both_good=round(pct, 1), verdict=verdict)
    for n, k, mw, mh, good, pct, verdict in sorted(rows, reverse=True):
        print(f"{k[0] + ' + ' + k[1]:<22}{n:>10,}{mw:>8.0f}{mh:>8.0f}"
              f"{good:>11,} ({pct:>4.1f}%)  {verdict}")

    # 從來沒有同時看到過的相機對 —— 這些才是真正的非重疊
    cams = sorted({c for k in pair for c in k})
    never = [(a, b) for i, a in enumerate(cams) for b in cams[i + 1:]
             if (a, b) not in pair]
    print(f"\n從未同時看到同一個人的相機對({len(never)} 對,這些才是真正非重疊):")
    print("  " + ("、".join(f"{a[-1]}+{b[-1]}" for a, b in never) if never else "(無)"))

    good_pairs = sum(v["n_both_good"] for v in out_pairs.values())
    print(f"\n可用於跨鏡頭訓練的同人配對(兩邊都 >= {GOOD_W}x{GOOD_H}px):{good_pairs:,} 組")
    print(f"對照:官方 multi_camera 的 _train 切分提供 0 組(40 張全部來自 camera_3)")

    o = Path(args.out)
    o.parent.mkdir(parents=True, exist_ok=True)
    o.write_text(json.dumps(dict(
        per_sequence=per_seq, n_cameras_hist=hist, pairs=out_pairs,
        never_together=[f"{a}+{b}" for a, b in never],
        total_instances=tot, multi_camera_instances=mul,
        usable_cross_camera_pairs=good_pairs,
        threshold=dict(w=GOOD_W, h=GOOD_H)), ensure_ascii=False, indent=2),
        encoding="utf-8")
    print(f"\n→ {o}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
