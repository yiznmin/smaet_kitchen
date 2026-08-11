"""M4 detect->track demo:RF-DETR 偵測 + KitchenTracker(ByteTrack)追蹤,畫 track_id。

預設用 COCO 預訓 RF-DETR 只追「人」(免微調權重),對應 M4->M5(廚師身份)->M6(停留超時)。
也可 --weights 載入微調 11 類模型追廚具。

輸出:標註幀 + tracks.json / tracks.csv / events.json。

用法:
  python scripts/m4_track_video.py --max-frames 300
  python scripts/m4_track_video.py --video data/epfl/Boutput0.mp4 --stride 2 --max-frames 500
  python scripts/m4_track_video.py --weights "checkpoint_best_regular (4).pth" --max-frames 200
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from common.video_io import iter_frames, imwrite_unicode   # noqa: E402
from m3.classes import NAMES, ROMAN                         # noqa: E402
from m4_track import KitchenTracker                         # noqa: E402

VIDEO = ROOT / "data" / "epfl" / "Boutput0.mp4"


def load_model(variant, weights):
    from rfdetr import RFDETRNano, RFDETRSmall, RFDETRMedium
    ctor = {"nano": RFDETRNano, "small": RFDETRSmall, "medium": RFDETRMedium}[variant]
    return ctor(pretrain_weights=weights, num_classes=len(NAMES)) if weights else ctor()


def color_for(tid):
    return (int((tid * 53) % 256), int((tid * 91) % 256), int((tid * 37) % 256))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default=str(VIDEO))
    ap.add_argument("--variant", default="nano", choices=["nano", "small", "medium"])
    ap.add_argument("--weights", help="微調權重(不給=COCO base,只追 person)")
    ap.add_argument("--config", default=str(ROOT / "configs" / "tracker.yaml"))
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--max-frames", type=int, default=300)
    ap.add_argument("--thr", type=float, default=0.1, help="偵測門檻(低→餵 ByteTrack 高/低分兩層)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--save-video", action="store_true", help="另存 mp4(預設只存幀,CJK 路徑較穩)")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))["tracker"]
    finetuned = bool(args.weights)
    person_only = cfg.get("person_only", True)
    stem = Path(args.video).stem
    out_dir = Path(args.out) if args.out else ROOT / "results" / "m4_track" / stem
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"載入 RF-DETR-{args.variant}{'(微調)' if finetuned else '(COCO)'} + ByteTrack …")
    model = load_model(args.variant, args.weights)
    tracker = KitchenTracker.from_config(cfg, class_names=NAMES)

    frame_rows, event_rows, log = [], [], []
    upd_ms, n_frames, vw = 0.0, 0, None
    for fid, ts, bgr in iter_frames(Path(args.video), args.stride):
        if n_frames >= args.max_frames:
            break
        n_frames += 1
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        det = model.predict(Image.fromarray(rgb), threshold=args.thr)

        # person_only 過濾
        if person_only and len(det):
            if finetuned:
                mask = det.class_id == 0
            else:
                names = det.data.get("class_name") if det.data else None
                mask = np.array([str(n) == "person" for n in names]) if names is not None \
                    else np.ones(len(det), dtype=bool)
            det = det[mask]

        t0 = time.perf_counter()
        out = tracker.update(det, fid, ts)
        upd_ms += (time.perf_counter() - t0) * 1000

        vis = bgr.copy()
        for tr in out.tracks:
            if tr.bbox is None:
                continue
            x1, y1, x2, y2 = [int(v) for v in tr.bbox]
            c = color_for(tr.track_id)
            # 標籤:微調模型用我們的 11 類 ROMAN;COCO base(person_only)一律標 person
            if finetuned:
                name = ROMAN.get(tr.class_id, "obj") if tr.class_id is not None else "obj"
            else:
                name = "person"
            cv2.rectangle(vis, (x1, y1), (x2, y2), c, 2)
            cv2.putText(vis, f"{name}#{tr.track_id}", (x1, max(14, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, c, 2)
        cv2.putText(vis, f"M4 ByteTrack  f{fid}  active={len(out.tracks)}", (10, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2)

        if args.save_video:
            if vw is None:
                h, w = vis.shape[:2]
                vw = cv2.VideoWriter(str(out_dir / "track.mp4"),
                                     cv2.VideoWriter_fourcc(*"mp4v"), 30, (w, h))
            vw.write(vis)
        else:
            imwrite_unicode(out_dir / f"f{fid:05d}.png", vis)

        frame_rows.append({"frame_id": fid, "ts": ts,
                           "tracks": [{"track_id": tr.track_id, "bbox": tr.bbox,
                                       "class_id": tr.class_id, "conf": tr.confidence,
                                       "status": tr.status.value} for tr in out.tracks],
                           "events": [{"kind": e.kind, "track_id": e.track_id,
                                       "class_id": e.class_id} for e in out.events]})
        for tr in out.tracks:
            log.append([fid, tr.track_id,
                        *([round(v, 1) for v in tr.bbox] if tr.bbox else ["", "", "", ""]),
                        tr.class_id if tr.class_id is not None else "",
                        round(tr.confidence, 3) if tr.confidence is not None else "",
                        tr.status.value])
        for e in out.events:
            event_rows.append({"frame_id": fid, "kind": e.kind, "track_id": e.track_id,
                               "class_id": e.class_id})

    if vw is not None:
        vw.release()

    (out_dir / "tracks.json").write_text(
        json.dumps(frame_rows, ensure_ascii=False, indent=1), encoding="utf-8")
    (out_dir / "events.json").write_text(
        json.dumps(event_rows, ensure_ascii=False, indent=1), encoding="utf-8")
    with open(out_dir / "tracks.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["frame_id", "track_id", "x1", "y1", "x2", "y2", "class_id", "conf", "status"])
        w.writerows(log)

    distinct = {r[1] for r in log}
    n_new = sum(1 for e in event_rows if e["kind"] == "new_track")
    n_lost = sum(1 for e in event_rows if e["kind"] == "lost_track")
    n_rem = sum(1 for e in event_rows if e["kind"] == "removed")
    print(f"\n處理 {n_frames} 幀 | 不同 track_id: {len(distinct)} 個")
    print(f"事件:new={n_new} lost={n_lost} removed={n_rem}")
    print(f"追蹤器平均 {upd_ms / max(1, n_frames):.3f} ms/幀(目標 <5ms;不含 RF-DETR;首幀含暖機)")
    try:
        disp = out_dir.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        disp = str(out_dir)
    print(f"輸出 → {disp}/(標註幀 + tracks.json/csv + events.json)")


if __name__ == "__main__":
    main()
