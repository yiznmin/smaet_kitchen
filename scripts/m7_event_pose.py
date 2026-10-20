"""事件測試第 2 輪:對 M3 人框跑 ViTPose,存關鍵點(吃東西用)。

預先登記:docs/M7_事件測試_預先登記_20261020.md §9.2
每 0.2 秒一幀(偵測快取的偶數步),人框 conf ≥ 0.3。
輸出 results/m7/events/pose/<片段檔名>.npz(本機):
  step [N]、box [N,4]、kp [N,17,3](x, y, 分數;COCO 17 點順序)

用法:
    python scripts/m7_event_pose.py --split dev
"""
import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from transformers import AutoProcessor, VitPoseForPoseEstimation

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from eval_events import CACHE, CLIPS, PERSON, VIEWS, load_events, steps     # noqa: E402
from m4_track.det_cache import DetCache                                    # noqa: E402

MODEL = "usyd-community/vitpose-base-simple"
OUT = ROOT / "results/m7/events/pose"
EVERY = 2                   # 步;0.1 秒一步 → 0.2 秒
THR_PERSON = 0.3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["dev", "test"], required=True)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    proc = AutoProcessor.from_pretrained(MODEL)
    model = VitPoseForPoseEstimation.from_pretrained(MODEL).to("cuda").eval()
    stems = sorted({e["clips"][v] for e in load_events(args.split) for v in VIEWS})
    todo = [s for s in stems if not (OUT / f"{s}.npz").exists()]
    print(f"{args.split}:{len(stems)} 支片段,待跑 {len(todo)}", flush=True)
    for n, stem in enumerate(todo, 1):
        c = DetCache(CACHE / f"{stem}.npz")
        fids, lo, hi = steps(c)
        want = {int(fids[i]): i for i in range(0, len(fids), EVERY)}
        cap = cv2.VideoCapture(str(CLIPS / f"{stem}.mp4"))
        rec_s, rec_b, rec_k, fid = [], [], [], 0
        while want:
            ok, f = cap.read()
            if not ok:
                break
            i = want.pop(fid, None)
            fid += 1
            if i is None:
                continue
            cl, cf, xy = c._cls[lo[i]:hi[i]], c._conf[lo[i]:hi[i]], c._xyxy[lo[i]:hi[i]].astype(float)
            pb = xy[(cl == PERSON) & (cf >= THR_PERSON)]
            if not len(pb):
                continue
            xywh = np.c_[pb[:, :2], pb[:, 2:] - pb[:, :2]]
            inp = proc(f[:, :, ::-1].copy(), boxes=[xywh], return_tensors="pt").to("cuda")
            with torch.inference_mode():
                out = model(**inp)
            res = proc.post_process_pose_estimation(out, boxes=[xywh])[0]
            for b, r in zip(pb, res):
                rec_s.append(i)
                rec_b.append(b)
                rec_k.append(np.c_[r["keypoints"].cpu().numpy(), r["scores"].cpu().numpy()])
        cap.release()
        np.savez_compressed(OUT / f"{stem}.npz", step=np.array(rec_s, np.int32),
                            box=np.array(rec_b, np.float32).reshape(-1, 4),
                            kp=np.array(rec_k, np.float32).reshape(-1, 17, 3))
        if n % 50 == 0 or n == len(todo):
            print(f"  {n}/{len(todo)}  {time.time() - t0:.0f} 秒", flush=True)


if __name__ == "__main__":
    main()
