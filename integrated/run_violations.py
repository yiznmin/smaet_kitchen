"""M6 團隊 NG 準則判定(單人):讀各鏡頭 M3 全類別偵測快取 → NG1 / NG2 事件。

    M3 偵測快取(每台一個 <鏡頭>.npz,stride 3 = 0.1 秒一步,全類別)→ 垃圾桶接觸 / 洗手 / 碰觸 → NG 事件

規則與參數:configs/violation_rules.yaml(ng、trash_event、hand_hygiene、association 各節);
zone:configs/zones/epfl_<鏡頭>.json。需要的鏡頭:output0、Aoutput1~3、Boutput1~2。
說明見 src/m6_events/ng.py。

輸出(--out 目錄):
  ng_events.jsonl   每個 NG 一筆:type(NG1 / NG2)、start、end(秒)、reason、n_contacts 或 touched
  ng_summary.json   各類次數、輸入、設定(下游自檢用)

用法:
  python run_violations.py --det-cache <含 output0.npz、Aoutput1.npz… 的資料夾> --out outputs/ng
  python run_violations.py --det-cache <資料夾> --t0 1085.9 --out outputs/ng   # 快取從影片 1085.9 秒開始
"""
import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from m4_track.det_cache import DetCache   # noqa: E402
from m6_events.ng import detect_ng, series  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--det-cache", required=True, help="各鏡頭 <鏡頭>.npz 所在資料夾")
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", default="configs/violation_rules.yaml")
    ap.add_argument("--t0", type=float, default=0.0, help="快取第 0 步對應的影片秒數")
    args = ap.parse_args()
    t_start = time.time()
    cfg = yaml.safe_load((ROOT / args.config).read_text(encoding="utf-8"))
    src = Path(args.det_cache)
    caches = {p.stem: DetCache(p) for p in sorted(src.glob("*.npz"))}
    s = series(caches, cfg, ROOT)
    events = detect_ng(s, cfg, args.t0)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "ng_events.jsonl", "w", encoding="utf-8") as fh:
        for e in events:
            fh.write(json.dumps(e, ensure_ascii=False) + "\n")
    summary = dict(counts=dict(Counter(e["type"] for e in events)), det_cache=str(src), cameras=sorted(caches),
                   t0=args.t0, steps=s["T"], step_s=cfg["step_s"], config=args.config, ng_config=cfg["ng"],
                   seconds=round(time.time() - t_start, 1))
    (out / "ng_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"NG 事件 {len(events)}:{summary['counts']},輸出 {out}")


if __name__ == "__main__":
    main()
