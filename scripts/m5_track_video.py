"""M5 端到端:多支影片 → M3 偵測 → M4 追蹤 → M5 跨鏡頭 chef_id。

**這是 M4 與 M5 第一次真的串起來。** 在此之前兩者只有各自的合成驗證,
全 repo 沒有任何檔案同時 import `m4_track` 與 `m5_reid`
(見 docs/M5_v3_重設計與可行性驗證_20260824.md §1)。

模擬連續兩輪卡在自己的 bug 裡,所以改用真實影片當獨立的現實檢查點 ——
第五輪那個「M5 收不到每幀位置」的缺口就是被真實情境逼出來的。

輸出 `chef_events.jsonl`,每筆是一次綁定決策,含**可稽核的欄位**:
  camera_id / track_id / chef_id / t_sec / bbox / matched / score / n_candidates
`score` 與 `n_candidates` 是事後查「為什麼綁成這樣」的依據,M6/M9 與驗收都要。

用法:
  # 單鏡頭(先確認管線通)
  python scripts/m5_track_video.py --videos data/epfl/Boutput0.mp4 --max-frames 60

  # 多鏡頭(EPFL 9 視角是同步的,以幀號對齊)
  python scripts/m5_track_video.py \\
      --videos data/epfl/Boutput0.mp4 data/epfl/Aoutput0.mp4 \\
      --cameras cam1 cam2 --stride 5 --max-frames 300

⚠ EPFL 是 CC-BY-NC,僅供驗證。⚠ 無 homography 校正時 world_xy=None,
  重疊路徑退回常數證據(見 docs/M5_模擬預先登記_地面校正_20260825.md R8)。
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from common.video_io import iter_frames                       # noqa: E402
from m3.classes import NAMES                                  # noqa: E402
from m4_track import KitchenTracker                           # noqa: E402
from m5_reid.identity_st import SpatioTemporalIdentityManager  # noqa: E402
from m5_reid.spatiotemporal import CameraTopology             # noqa: E402

# ⚠ 「人」的 class_id 依權重而異,這是端到端才會踩到的整合陷阱:
#     COCO 預訓 RF-DETR  → person = 1(COCO category_id)
#     我們微調的 11 類    → 人 = 0(src/m3/classes.py 的 CAT2CLS 是 category_id − 1)
#   接錯的話一個人都偵測不到,而且**不會報錯**,只會安靜地輸出 0 筆綁定。
PERSON_CLS_COCO = 1
PERSON_CLS_FINETUNED = 0


def load_model(variant, weights):
    from rfdetr import RFDETRMedium, RFDETRNano, RFDETRSmall
    ctor = {"nano": RFDETRNano, "small": RFDETRSmall, "medium": RFDETRMedium}[variant]
    return ctor(pretrain_weights=weights, num_classes=len(NAMES)) if weights else ctor()


def detect_person(model, frame_bgr, thr, person_cls):
    """回傳 supervision Detections,只留「人」。"""
    import supervision as sv
    pil = Image.fromarray(frame_bgr[:, :, ::-1])
    det = model.predict(pil, threshold=thr)
    if len(det) == 0:
        return sv.Detections.empty()
    keep = np.array([int(c) == person_cls for c in det.class_id])
    if not keep.any():
        return sv.Detections.empty()
    return sv.Detections(xyxy=det.xyxy[keep],
                         confidence=det.confidence[keep],
                         class_id=det.class_id[keep])


def build_embedder(name):
    """外觀 embedder。none = 不用外觀(實測顯示它對本架構幾乎沒影響)。"""
    if name == "none":
        class _Zero:
            dim = 64

            def extract(self, crop):
                return np.zeros(self.dim, dtype=np.float32)
        return _Zero()
    if name == "color":
        from m5_reid.embedder import ColorHistogramEmbedder
        return ColorHistogramEmbedder()
    if name == "dinov2":
        from m5_reid.dino_embedder import DinoV2Embedder
        return DinoV2Embedder()
    raise ValueError(f"未知的 embedder: {name}")


def crop_of(frame_bgr, bbox):
    h, w = frame_bgr.shape[:2]
    x1, y1, x2, y2 = (int(max(0, bbox[0])), int(max(0, bbox[1])),
                      int(min(w, bbox[2])), int(min(h, bbox[3])))
    if x2 <= x1 or y2 <= y1:
        return None
    return frame_bgr[y1:y2, x1:x2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", nargs="+", required=True)
    ap.add_argument("--cameras", nargs="+", default=None,
                    help="與 --videos 一一對應的 camera_id;預設用檔名")
    ap.add_argument("--topology", default=str(ROOT / "configs" / "camera_topology.yaml"))
    ap.add_argument("--tracker", default=str(ROOT / "configs" / "tracker.yaml"))
    ap.add_argument("--weights", default=None, help="M3 微調權重;不給則用 COCO 預訓")
    ap.add_argument("--variant", default="nano")
    ap.add_argument("--thr", type=float, default=0.3)
    ap.add_argument("--person-cls", type=int, default=None,
                    help="「人」的 class_id;預設依有無 --weights 自動判斷")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--max-frames", type=int, default=120)
    ap.add_argument("--embedder", default="none", choices=["none", "color", "dinov2"])
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--out", default=str(ROOT / "results" / "m5_track" / "chef_events.jsonl"))
    args = ap.parse_args()

    cams = args.cameras or [Path(v).stem for v in args.videos]
    if len(cams) != len(args.videos):
        raise SystemExit("--cameras 數量必須與 --videos 相同")

    topo = CameraTopology.from_yaml(args.topology)
    unknown = [c for c in cams if c not in topo.all_cameras()]
    if unknown:
        print(f"⚠ 這些 camera_id 不在拓撲裡:{unknown}")
        print("  → 它們之間沒有連結也不重疊,跨鏡頭一律開新 chef_id。")
        print(f"  拓撲裡有的是:{sorted(topo.all_cameras())}")

    if topo.homographies:
        print("地面校正:")
        for h in topo.homographies.values():
            print("  " + h.describe())
        print(f"  → GroundPlaneLR 用實測殘差 σ={topo.calib_sigma_m:.3f}m"
              f"(第六輪 R8:σ 需 ≤0.2m,≥0.8m 則此證據無用)")
    else:
        print("⚠ 沒有任何鏡頭做地面校正 → 重疊路徑退回常數證據 "
              "(overlap_llr,已知過度自信,見第六輪 R8)")

    with open(args.tracker, encoding="utf-8") as f:
        tcfg = (yaml.safe_load(f) or {}).get("tracker", {})

    person_cls = (args.person_cls if args.person_cls is not None
                  else (PERSON_CLS_FINETUNED if args.weights else PERSON_CLS_COCO))
    print(f"載入 M3({args.variant}{'/微調' if args.weights else '/COCO 預訓'}),"
          f"人的 class_id = {person_cls}…")
    model = load_model(args.variant, args.weights)
    emb = build_embedder(args.embedder)

    trackers = {c: KitchenTracker.from_config(tcfg, camera_id=c) for c in cams}
    m5 = SpatioTemporalIdentityManager(topo, embedder=emb, fps=args.fps / max(args.stride, 1))

    streams = {c: iter_frames(v, stride=args.stride) for c, v in zip(cams, args.videos)}
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows, n_frames, n_det, t0 = [], 0, 0, time.time()
    with out_path.open("w", encoding="utf-8") as fout:
        while n_frames < args.max_frames:
            frames = {}
            for c, it in streams.items():
                nxt = next(it, None)
                if nxt is not None:
                    frames[c] = nxt
            if not frames:
                break
            # 用影片自己的時間戳,不用幀號換算 —— 各鏡頭 fps 可能不同,
            # 而 M5 的轉場時間窗完全建立在時間軸上,算錯會全面失準。
            t_sec = min(item[1] for item in frames.values())

            per_cam = {}
            for c, (fid, t_cam, frame) in frames.items():
                det = detect_person(model, frame, args.thr, person_cls)
                n_det += len(det)
                out = trackers[c].update(det, n_frames, timestamp=t_cam)
                per_cam[c] = (frame, out)

            # ① 心跳:先把「誰此刻在畫面上、在哪裡」全部餵進去,再做綁定決策。
            #    M4 每幀都有 tracks,但 M5 只吃事件 —— 這個介面缺口是第五、六輪
            #    在模擬裡踩出來的,真實管線同樣需要,而且順序必須在事件之前。
            for c, (_frame, out) in per_cam.items():
                for tr in out.tracks:
                    m5.on_track_update(tr.track_id, camera_id=c, frame_id=n_frames,
                                       t_sec=t_sec, bbox=tr.bbox,
                                       world_xy=topo.world_xy(c, tr.bbox))

            # ② 事件 → M5
            for c, (frame, out) in per_cam.items():
                for ev in out.events:
                    if ev.kind == "new_track":
                        cr = crop_of(frame, ev.bbox) if ev.bbox is not None else None
                        r = m5.on_new_track(ev.track_id, camera_id=c, frame_id=ev.frame_id,
                                            t_sec=ev.t_sec, bbox=ev.bbox, crop=cr,
                                            world_xy=topo.world_xy(c, ev.bbox))
                        row = dict(t_sec=round(ev.t_sec, 3), camera_id=c,
                                   track_id=ev.track_id, chef_id=r.chef_id,
                                   matched=bool(r.matched), score=r.similarity,
                                   n_candidates=m5.last_candidates,
                                   bbox=[round(float(v), 1) for v in (ev.bbox or [])])
                        rows.append(row)
                        fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                    elif ev.kind == "reacquired":
                        m5.on_track_reacquired(ev.track_id, camera_id=c,
                                               frame_id=ev.frame_id, t_sec=ev.t_sec,
                                               bbox=ev.bbox)
                    elif ev.kind == "lost_track":
                        m5.on_track_lost(ev.track_id, camera_id=c, frame_id=ev.frame_id,
                                         t_sec=ev.t_sec, bbox=ev.bbox)
                    elif ev.kind == "removed":
                        m5.on_track_removed(ev.track_id, camera_id=c, frame_id=ev.frame_id,
                                            t_sec=ev.t_sec, bbox=ev.bbox)
            n_frames += 1

    dt = time.time() - t0
    print()
    print("=" * 66)
    print(f"處理 {n_frames} 幀 × {len(cams)} 鏡頭,耗時 {dt:.1f}s "
          f"({dt/max(n_frames,1)*1000:.0f} ms/幀)")
    print(f"偵測到「人」{n_det} 次")
    if n_det == 0:
        print("  ⚠ 一個人都沒偵測到 —— 先確認 --person-cls 對不對(COCO=1、微調=0)")
    print(f"綁定決策 {len(rows)} 次,其中沿用既有 chef {sum(r['matched'] for r in rows)} 次")
    print(f"最終 chef_id 數:{m5.stats()['total_chefs']}")
    print(f"候選數分布(0=開新人、1=唯一候選):{m5.candidate_histogram()}")
    print(f"常駐狀態:{m5.resident_stats()}")
    print(f"輸出:{out_path}")
    if rows:
        print()
        print(f"  {'t(s)':>7} {'鏡頭':<12}{'track':>6}{'chef':>6}{'綁定':>6}{'分數':>8}{'候選':>5}")
        for r in rows[:15]:
            print(f"  {r['t_sec']:>7.2f} {r['camera_id']:<12}{r['track_id']:>6}"
                  f"{r['chef_id']:>6}{'是' if r['matched'] else '新':>6}"
                  f"{r['score']:>8.2f}{r['n_candidates']:>5}")
        if len(rows) > 15:
            print(f"  …共 {len(rows)} 筆")


if __name__ == "__main__":
    main()
