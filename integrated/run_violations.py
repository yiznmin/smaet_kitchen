"""M6 團隊 NG 準則判定(單人):讀各鏡頭 M3 全類別偵測快取 → NG1 / NG2 / NG3 / NG4 事件。

    M3 偵測快取(每台一個 <鏡頭>.npz,stride 3 = 0.1 秒一步,全類別)→ 垃圾桶接觸 / 洗手 / 碰觸 → NG 事件
    NG3(手碰臉)另需 output0、Aoutput1 的人體姿態(臉中心、肩寬、手腕來自 ViTPose 關鍵點,M3 偵測沒有臉):
    給 --video-dir 就從影片現算(需 transformers 與 CUDA),結果存在 <out>/pose_cache 供重跑直接讀;
    --pose-cache 選用,可指定已算好的姿態檔資料夾。兩者都沒給就跳過 NG3,ng_summary.json 的 ng3 欄註明。

規則與參數:configs/violation_rules.yaml(ng、trash_event、hand_hygiene、association 各節);
zone:configs/zones/epfl_<鏡頭>.json。需要的鏡頭:output0、Aoutput1~3、Boutput1~2。
說明見 src/m6_events/ng.py。

輸出(--out 目錄):
  ng_events.jsonl   每個 NG 一筆:type(NG1 / NG2 / NG3 / NG4)、start、end(秒)、reason、n_contacts / touched / obj、place、
                    status(NG4:確定 / 待確認)、views(NG3:成立的鏡頭)
  ng_summary.json   各類次數、輸入、設定(下游自檢用)

用法:
  python run_violations.py --det-cache <含 output0.npz、Aoutput1.npz… 的資料夾> --out outputs/ng
  python run_violations.py --det-cache <資料夾> --t0 1085.9 --out outputs/ng   # 快取從影片 1085.9 秒開始
  python run_violations.py --det-cache <資料夾> --video-dir <含 output0.mp4、Aoutput1.mp4> --out outputs/ng   # 含 NG3
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
from m6_events.ng3 import detect_ng3  # noqa: E402


def load_poses(caches, n3, args):
    """NG3 姿態:給 --video-dir 就用 ViTPose 從影片現算(存到 --pose-cache,沒給則存 <out>/pose_cache,重跑可直接讀);
    只給 --pose-cache 則讀現成的姿態檔。兩者都沒給回傳 {}(跳過 NG3)。"""
    import numpy as np
    if not args.pose_cache and not args.video_dir:
        return {}
    pdir = Path(args.pose_cache) if args.pose_cache else Path(args.out) / "pose_cache"
    est, out = None, {}
    for v in n3["views"]:
        f = pdir / f"{v}.npz"
        if not f.exists() and args.video_dir and v in caches and (Path(args.video_dir) / f"{v}.mp4").exists():
            if est is None:
                from m6_events.pose import PoseEstimator
                est = PoseEstimator()
            print(f"ViTPose 估 {v} 姿態…", flush=True)
            est.run(Path(args.video_dir) / f"{v}.mp4", caches[v], f, every=n3["pose_every"])
        if f.exists():
            out[v] = dict(np.load(f))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--det-cache", required=True, help="各鏡頭 <鏡頭>.npz 所在資料夾")
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", default="configs/violation_rules.yaml")
    ap.add_argument("--t0", type=float, default=0.0, help="快取第 0 步對應的影片秒數")
    ap.add_argument("--video-dir", help="NG3 用:影片資料夾(<鏡頭>.mp4),從影片現算姿態")
    ap.add_argument("--pose-cache", help="選用:姿態檔資料夾(<鏡頭>.npz);預設 <out>/pose_cache")
    args = ap.parse_args()
    t_start = time.time()
    cfg = yaml.safe_load((ROOT / args.config).read_text(encoding="utf-8"))
    src = Path(args.det_cache)
    caches = {p.stem: DetCache(p) for p in sorted(src.glob("*.npz"))}
    s = series(caches, cfg, ROOT)
    events = detect_ng(s, cfg, args.t0)
    ng3_note = "設定關閉"
    n3 = cfg.get("ng3", {})
    if n3.get("enable"):
        poses = load_poses(caches, n3, args)
        if poses:
            params = {k: v for k, v in n3.items() if k != "enable"}
            events = sorted(events + detect_ng3(caches, poses, cfg["step_s"], params, args.t0),
                            key=lambda x: (x["start"], x["type"]))
            ng3_note = f"姿態鏡頭 {sorted(poses)}"
        else:
            ng3_note = "跳過:沒有姿態(給 --video-dir 從影片現算)"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "ng_events.jsonl", "w", encoding="utf-8") as fh:
        for e in events:
            fh.write(json.dumps(e, ensure_ascii=False) + "\n")
    summary = dict(counts=dict(Counter(e["type"] for e in events)), det_cache=str(src), cameras=sorted(caches),
                   t0=args.t0, steps=s["T"], step_s=cfg["step_s"], config=args.config, ng_config=cfg["ng"], ng3=ng3_note,
                   seconds=round(time.time() - t_start, 1))
    (out / "ng_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"NG 事件 {len(events)}:{summary['counts']},輸出 {out}")


if __name__ == "__main__":
    main()
