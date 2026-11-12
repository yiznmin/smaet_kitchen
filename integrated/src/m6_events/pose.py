"""NG3 用的人體姿態:ViTPose 對 M3 人框估 17 個 COCO 關鍵點(臉中心、肩寬、手腕;做法同 scripts/eval_ng3_face.py run_pose)。

輸出 npz:step(int32[N],偵測快取的步)、box(float32[N, 4],人框 xyxy)、kp(float32[N, 17, 3],x、y、分數)。
每 every 步(預設 2 = 0.2 秒)估一幀。需要 transformers(>=4.48,含 VitPoseForPoseEstimation)與 CUDA。
"""
from pathlib import Path

import numpy as np

from m6_events.violation import PERSON, steps

MODEL = "usyd-community/vitpose-base-simple"


class PoseEstimator:
    def __init__(self, model=MODEL, device="cuda"):
        import torch
        from transformers import AutoProcessor, VitPoseForPoseEstimation
        self.torch, self.device = torch, device
        self.proc = AutoProcessor.from_pretrained(model)
        self.model = VitPoseForPoseEstimation.from_pretrained(model).to(device).eval()

    def run(self, video, dc, out_path, every=2, thr_person=0.3):
        """video:影片路徑(OpenCV 開得了的 ASCII 路徑);dc:同一支影片的 DetCache。"""
        import cv2
        fids, lo, hi = steps(dc)
        want = {int(fids[i]): i for i in range(0, len(fids), every)}
        cap = cv2.VideoCapture(str(video))
        if not cap.isOpened():
            raise RuntimeError(f"開不了影片:{video}")
        rec_s, rec_b, rec_k, fid = [], [], [], 0
        while want:
            ok, f = cap.read()
            if not ok:
                break
            i = want.pop(fid, None)
            fid += 1
            if i is None:
                continue
            cl, cf, xy = dc._cls[lo[i]:hi[i]], dc._conf[lo[i]:hi[i]], dc._xyxy[lo[i]:hi[i]].astype(float)
            pb = xy[(cl == PERSON) & (cf >= thr_person)]
            if not len(pb):
                continue
            xywh = np.c_[pb[:, :2], pb[:, 2:] - pb[:, :2]]
            inp = self.proc(f[:, :, ::-1].copy(), boxes=[xywh], return_tensors="pt").to(self.device)
            with self.torch.inference_mode():
                out = self.model(**inp)
            res = self.proc.post_process_pose_estimation(out, boxes=[xywh])[0]
            for b, r in zip(pb, res):
                rec_s.append(i)
                rec_b.append(b)
                rec_k.append(np.c_[r["keypoints"].cpu().numpy(), r["scores"].cpu().numpy()])
        cap.release()
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out_path, step=np.array(rec_s, np.int32), box=np.array(rec_b, np.float32).reshape(-1, 4),
                            kp=np.array(rec_k, np.float32).reshape(-1, 17, 3))
        return out_path
