"""M6 違規判定(單人):各鏡頭影片 → M3 全類別偵測 → NG1 / NG2 / NG3 / NG4 事件。

    M3 偵測(每台每 3 幀一次 = 0.1 秒一步,全類別,存成 <鏡頭>.npz 偵測快取)→ 垃圾桶接觸 / 洗手 / 碰觸 → NG 事件
    給 --video-dir 時,缺少的偵測快取會用微調權重從影片現算,存在 --det-cache(預設 <out>/det_cache),重跑直接讀。
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
  python run_violations.py --video-dir <含 output0.mp4、Aoutput1.mp4… 的資料夾> --out outputs/ng   # 從影片跑完整流程
  python run_violations.py --det-cache <含 output0.npz… 的資料夾> --out outputs/ng                # 只用現成快取(不含 NG3)
  python run_violations.py --video-dir <資料夾> --t0 1085.9 --out outputs/ng                      # 影片第 0 幀 = 原始錄影 1085.9 秒
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


CAMERAS = ("output0", "Aoutput1", "Aoutput2", "Aoutput3", "Boutput1", "Boutput2")
DET_WEIGHTS = ROOT / "weights/m3_nano_r2_armA/checkpoint_best_total.pth"
DET_STRIDE = 3        # 30 fps 每 3 幀一步 = 0.1 秒(對應 violation_rules.yaml 的 step_s)
DET_THR = 0.05        # 快取低門檻;各規則再用自己的門檻過濾


def build_det_caches(video_dir, cache_dir):
    """缺少的 <鏡頭>.npz 用 M3 微調權重從 <video_dir>/<鏡頭>.mp4 現算(全類別、stride 3、門檻 0.05)。"""
    import numpy as np
    from PIL import Image
    from common.video_io import iter_frames, video_meta
    from m3.model import load_rfdetr, model_resolution
    from m4_track.det_cache import save, weights_id
    model = None
    for cam in CAMERAS:
        out_path, video = Path(cache_dir) / f"{cam}.npz", Path(video_dir) / f"{cam}.mp4"
        if out_path.exists() or not video.exists():
            continue
        if model is None:
            model = load_rfdetr("nano", str(DET_WEIGHTS), "auto")
        print(f"M3 偵測 {cam}…", flush=True)
        t0 = time.time()
        fids, xy, cf, cl = [], [], [], []
        n, last = 0, -1
        for fid, _t, frame in iter_frames(str(video), stride=DET_STRIDE):
            det = model.predict(Image.fromarray(frame[:, :, ::-1]), threshold=DET_THR)
            if len(det):
                fids.append(np.full(len(det), fid, dtype=np.int64))
                xy.append(det.xyxy)
                cf.append(det.confidence)
                cl.append(det.class_id)
            n, last = n + 1, fid
        save(out_path,
             np.concatenate(fids) if fids else np.zeros(0, dtype=np.int64),
             np.concatenate(xy) if xy else np.zeros((0, 4), dtype=np.float32),
             np.concatenate(cf) if cf else np.zeros(0, dtype=np.float32),
             np.concatenate(cl) if cl else np.zeros(0, dtype=np.int64),
             dict(variant="nano", weights_id=weights_id(str(DET_WEIGHTS)), person_cls=None, classes="all",
                  cache_thr=DET_THR, cache_stride=DET_STRIDE, complete=True, n_scanned=n, max_fid=last,
                  resolution=model_resolution(model), video=str(video), video_meta=video_meta(str(video)),
                  seconds=round(time.time() - t0, 1)))
        print(f"  {cam}:{n} 幀,{time.time() - t0:.0f} 秒", flush=True)


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
    ap.add_argument("--det-cache", help="偵測快取資料夾(<鏡頭>.npz);預設 <out>/det_cache")
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", default="configs/violation_rules.yaml")
    ap.add_argument("--t0", type=float, default=0.0, help="快取第 0 步對應的影片秒數")
    ap.add_argument("--video-dir", help="影片資料夾(<鏡頭>.mp4):缺少的偵測快取與 NG3 姿態從影片現算")
    ap.add_argument("--pose-cache", help="選用:姿態檔資料夾(<鏡頭>.npz);預設 <out>/pose_cache")
    args = ap.parse_args()
    t_start = time.time()
    cfg = yaml.safe_load((ROOT / args.config).read_text(encoding="utf-8"))
    if not args.det_cache and not args.video_dir:
        ap.error("至少要給 --video-dir 或 --det-cache")
    src = Path(args.det_cache) if args.det_cache else Path(args.out) / "det_cache"
    if args.video_dir:
        build_det_caches(args.video_dir, src)
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
