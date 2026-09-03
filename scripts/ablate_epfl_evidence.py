"""EPFL 真實影片上的證據消融 —— 拆開「拓撲」與「外觀」各自貢獻多少。

## 為什麼需要這個

`docs/M4_M5_實片全長評估_20260901.md` 的主結果是「780 秒、三台鏡頭、17 次重新綁定,
全部綁回同一個 chef_id」。但那次用的 `camera_topology.epfl_demo.yaml` 是**我們建的**
—— 把 cam1/cam2/cam3 三對都宣告為重疊。

於是「綁對」有多少來自系統的能力、多少來自**我們給的資訊**,分不出來。
這支腳本用負對照把它拆開。

## 五組條件(只改兩個因素)

| 組 | 拓撲 | unknown_path | 外觀 | 問的問題 |
|---|---|---|---|---|
| A | 三對重疊 | — | none | 主實驗(基準) |
| B | **無** | off | none | **負對照**:證明 A 的結果來自拓撲 |
| C | 無 | **on** | none | 允許未建模路徑,但不給外觀 |
| D | 無 | on | **dinov2** | **純外觀能不能認人**(C 與 D 的差 = 外觀的貢獻) |
| E | 三對重疊 | — | dinov2 | 有拓撲時外觀還有沒有加分(A 與 E 的差) |

⚠ B 的 `transit_llr` 直接回絕(−1e9),候選**進不了評分階段**,所以外觀連發言
  機會都沒有 —— 這就是為什麼需要 C:先讓候選進得了門,外觀才有得比。

⚠ 開跑前已知:`unknown_path` 在 Δt=5s 給 **+1.92 nats**,而門檻是 1.61 ——
  **它自己就過門了**。而 DINOv2 的最大可能加分只有 +1.06、典型同人只有 +0.03。
  所以預期 C≈D。若真是如此,那就是在真實影片上再一次確認「外觀幾乎沒有發言權」。

用法:
    python scripts/ablate_epfl_evidence.py --stride 10 --max-frames 120
    python scripts/ablate_epfl_evidence.py --report        # 只讀既有結果重印表
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "m5_ablate"

VIDEOS = ["data/epfl/Boutput0.mp4", "data/epfl/output0.mp4", "data/epfl/Aoutput0.mp4"]
CAMS = ["cam1", "cam2", "cam3"]

CONDITIONS = [
    ("A", "epfl_demo",          "none",   "三對重疊 / 無外觀", "主實驗(基準)"),
    ("B", "epfl_blind",         "none",   "無拓撲 / 無外觀",   "負對照"),
    ("C", "epfl_blind_unknown", "none",   "無拓撲+未建模路徑 / 無外觀", "讓候選進得了門"),
    ("D", "epfl_blind_unknown", "dinov2", "無拓撲+未建模路徑 / DINOv2", "純外觀能不能認人"),
    ("E", "epfl_demo",          "dinov2", "三對重疊 / DINOv2", "有拓撲時外觀有沒有加分"),
]


def run(tag, topo, embedder, stride, max_frames):
    d = OUT / tag
    d.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(ROOT / "scripts" / "m5_track_video.py"),
           "--videos", *VIDEOS, "--cameras", *CAMS,
           "--topology", str(ROOT / "configs" / f"camera_topology.{topo}.yaml"),
           "--embedder", embedder,
           "--stride", str(stride), "--max-frames", str(max_frames),
           "--out", str(d / "chef_events.jsonl")]
    print(f"\n{'='*70}\n{tag}  topo={topo}  embedder={embedder}\n{'='*70}")
    r = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    if r.returncode != 0:
        print(f"  ❌ 失敗(exit {r.returncode})")
        print("  " + "\n  ".join((r.stderr or "").splitlines()[-6:]))
        return False
    for line in (r.stdout or "").splitlines():
        if any(k in line for k in ("最終 chef_id", "綁定決策", "候選數分布", "偵測到")):
            print("  " + line.strip())
    return True


def summarize():
    print("\n" + "=" * 78)
    print("消融結果 —— EPFL 單人、三台鏡頭")
    print("=" * 78)
    print(f"  {'組':<4}{'條件':<30}{'chef_id':>9}{'綁定':>7}{'沿用':>7}{'平均分數':>10}")
    print("  " + "-" * 68)
    rows = {}
    for tag, topo, emb, label, _why in CONDITIONS:
        p = OUT / tag / "chef_events.jsonl"
        if not p.exists():
            print(f"  {tag:<4}{label:<30}{'(未跑)':>9}")
            continue
        ev = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
        if not ev:
            continue
        ids = {e["chef_id"] for e in ev}
        matched = [e for e in ev if e["matched"]]
        avg = sum(e["score"] for e in matched) / len(matched) if matched else 0.0
        rows[tag] = dict(n_ids=len(ids), n_bind=len(ev), n_matched=len(matched),
                         avg_score=avg, topo=topo, embedder=emb, label=label)
        print(f"  {tag:<4}{label:<30}{len(ids):>9}{len(ev):>7}"
              f"{len(matched):>7}{avg:>10.2f}")

    print("\n  判讀:")
    if "A" in rows and "B" in rows:
        print(f"    A vs B(拓撲的貢獻):{rows['A']['n_ids']} 個 → {rows['B']['n_ids']} 個 身份")
        print("      → 拿掉拓撲資訊,系統完全綁不起來。**能力來自知道鏡頭怎麼擺。**")
    if "C" in rows and "D" in rows:
        d = rows["D"]["avg_score"] - rows["C"]["avg_score"]
        print(f"    C vs D(外觀的貢獻,無拓撲時):身份 {rows['C']['n_ids']} → "
              f"{rows['D']['n_ids']},平均分數 {d:+.2f} nats")
    if "A" in rows and "E" in rows:
        d = rows["E"]["avg_score"] - rows["A"]["avg_score"]
        print(f"    A vs E(外觀的貢獻,有拓撲時):身份 {rows['A']['n_ids']} → "
              f"{rows['E']['n_ids']},平均分數 {d:+.2f} nats")

    (OUT / "ablation.json").write_text(
        json.dumps({"conditions": [dict(zip(("tag", "topo", "embedder", "label", "why"), c))
                                   for c in CONDITIONS], "results": rows},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  已存 {OUT/'ablation.json'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stride", type=int, default=10)
    ap.add_argument("--max-frames", type=int, default=120)
    ap.add_argument("--report", action="store_true", help="只讀既有結果重印表")
    ap.add_argument("--only", nargs="+", help="只跑指定的組別,例如 --only C D")
    args = ap.parse_args()

    if not args.report:
        for tag, topo, emb, _label, _why in CONDITIONS:
            if args.only and tag not in args.only:
                continue
            run(tag, topo, emb, args.stride, args.max_frames)
    summarize()
    return 0


if __name__ == "__main__":
    sys.exit(main())
