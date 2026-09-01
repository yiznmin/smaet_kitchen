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
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import psutil
import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from common.video_io import iter_frames, video_meta            # noqa: E402
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
        # ⚠ 類別名是 DINOv2Embedder(大寫 IN),不是 DinoV2Embedder。
        #   寫錯會在 --embedder dinov2 時直接 ImportError,而預設是 none 所以一直沒被觸發。
        from m5_reid.dino_embedder import DINOv2Embedder
        return DINOv2Embedder()
    raise ValueError(f"未知的 embedder: {name}")


def crop_of(frame_bgr, bbox):
    h, w = frame_bgr.shape[:2]
    x1, y1, x2, y2 = (int(max(0, bbox[0])), int(max(0, bbox[1])),
                      int(min(w, bbox[2])), int(min(h, bbox[3])))
    if x2 <= x1 or y2 <= y1:
        return None
    return frame_bgr[y1:y2, x1:x2]


# ── 視覺化 ────────────────────────────────────────────────────────────
# M4 的 m4_track_video.py 會存標註幀,但 M5 原本只吐 JSONL —— 而 M5 才是
# 要給業主與教授看的東西。這裡補上:同一位 chef_id 在**所有鏡頭裡同一個顏色**,
# 這樣「跨鏡頭認得是同一個人」這件事才看得出來,不然一堆數字沒人看得懂。
# 顏色由 chef_id 決定(不是 track_id),因為 track_id 每台鏡頭各自編號。
_PALETTE = [(80, 200, 120), (80, 160, 255), (200, 120, 255), (60, 220, 240),
            (255, 170, 80), (140, 220, 90), (255, 120, 170), (110, 190, 255)]


def chef_color(chef_id):
    return _PALETTE[(int(chef_id) - 1) % len(_PALETTE)] if chef_id else (140, 140, 140)


def draw_panel(frame_bgr, tracks, cam, m5, t_sec, width=640):
    """畫一台鏡頭:每個 active track 標上它的 chef_id(不是 track_id)。"""
    import cv2
    img = frame_bgr.copy()
    # ⚠ 標題列必須**先**畫。原本畫在最後,會把貼齊上緣的人的標籤整個蓋掉
    #   —— cam3 那格看起來像沒認出人,實際上只是標籤被蓋住了。
    cv2.rectangle(img, (0, 0), (img.shape[1], 42), (28, 28, 28), -1)
    cv2.putText(img, f"{cam}   t={t_sec:.2f}s", (12, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, (240, 240, 240), 2)
    for tr in tracks:
        if tr.bbox is None:
            continue
        cid = m5.track_to_chef.get((cam, tr.track_id))
        col = chef_color(cid)
        x1, y1, x2, y2 = (int(v) for v in tr.bbox)
        cv2.rectangle(img, (x1, y1), (x2, y2), col, 3)
        label = f"chef {cid}" if cid else f"track {tr.track_id} unbound"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        # 標籤預設畫在框上方,但人貼齊畫面上緣時會被切掉(頂部還有鏡頭名的橫條)
        # → 放不下就翻到框內側。第一版沒做這件事,cam3 的標籤整個看不到。
        top = y1 - th - 10
        ty1 = top if top >= 46 else max(y1, 46)
        ty2 = y1 if top >= 46 else ty1 + th + 10
        lx = min(x1, img.shape[1] - tw - 12)          # 靠右邊界時往左收
        cv2.rectangle(img, (lx, ty1), (lx + tw + 10, ty2), col, -1)
        cv2.putText(img, label, (lx + 5, ty2 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 20, 20), 2)
    h = int(img.shape[0] * width / img.shape[1])
    return cv2.resize(img, (width, h))


def stitch(panels):
    """多鏡頭橫向拼接(高度不同就補黑),讓同一時刻的各鏡頭並排。"""
    import cv2
    if not panels:
        return None
    hmax = max(p.shape[0] for p in panels)
    padded = []
    for p in panels:
        if p.shape[0] < hmax:
            p = cv2.copyMakeBorder(p, 0, hmax - p.shape[0], 0, 0,
                                   cv2.BORDER_CONSTANT, value=(20, 20, 20))
        padded.append(p)
    return np.hstack(padded)


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
    ap.add_argument("--max-frames", type=int, default=120,
                    help="要處理幾個**迴圈**(不是影片幀數;一個迴圈 = stride 幀)。"
                         "-1 = 跑到影片結束。⚠ 預設 120 對 780 秒的影片只有 0.3%%,"
                         "而舊版摘要不會提示 —— 量出來的東西完全代表不了整支影片。")
    ap.add_argument("--embedder", default="none", choices=["none", "color", "dinov2"])
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--ttl", type=int, default=600,
                    help="recently_disappeared_ttl。⚠ 單位是**迴圈計數**不是影片幀 —— "
                         "identity.py:66 比較的是 driver 傳進去的 frame_id,而這裡傳的是"
                         "迴圈計數。實際秒數 = ttl × stride / fps(預設 600 @stride5 = 100 秒)")
    ap.add_argument("--resident-every", type=int, default=50,
                    help="每幾個迴圈取樣一次常駐狀態與 RSS(記憶體有界性的證據)")
    ap.add_argument("--out", default=str(ROOT / "results" / "m5_track" / "chef_events.jsonl"))
    ap.add_argument("--save-frames", default=None,
                    help="存跨鏡頭標註幀到這個目錄(同一 chef_id 在各鏡頭同色)")
    ap.add_argument("--save-frames-every", type=int, default=0,
                    help="每 N 迴圈存一張;0 = 只在開新 chef_id 時存。⚠ 全長 stride=5 "
                         "是 4680 張拼接 PNG,不設這個會塞爆磁碟")
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
    m5 = SpatioTemporalIdentityManager(topo, embedder=emb,
                                       fps=args.fps / max(args.stride, 1),
                                       recently_disappeared_ttl=args.ttl)

    # ── 開跑前先把規模講清楚 ────────────────────────────────────────────
    # 舊版只會在結尾說「處理 120 幀」,不會提到影片還有 99.7% 沒跑。
    # 「靜默截斷」跟「靜默假通過」是同一種病:結果看起來正常,但代表不了宣稱的東西。
    metas = {c: video_meta(v) for c, v in zip(cams, args.videos)}
    n_loops_full = min(-(-m["nb_frames"] // args.stride) for m in metas.values())
    budget = n_loops_full if args.max_frames < 0 else min(args.max_frames, n_loops_full)
    print("影片:")
    for c, m in metas.items():
        print(f"  {c:<8}{Path(dict(zip(cams, args.videos))[c]).name:<16}"
              f"{m['nb_frames']} 幀 / {m['nb_frames']/m['fps']:.0f} 秒 / {m['fps']:.1f} fps")
    print(f"stride={args.stride} → 跑完整支需 {n_loops_full} 迴圈;"
          f"本次會跑 {budget} 迴圈({budget/n_loops_full*100:.1f}%)")
    if budget < n_loops_full:
        print("  ⚠ 這是**截斷**跑,所有輸出只代表這一段,不可當成全長結果。")
    print(f"TTL={args.ttl} 迴圈 = {args.ttl * args.stride / args.fps:.0f} 秒影片時間")

    streams = {c: iter_frames(v, stride=args.stride) for c, v in zip(cams, args.videos)}
    out_path = Path(args.out)
    out_dir = out_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    frames_dir = None
    if args.save_frames:
        frames_dir = Path(args.save_frames)
        frames_dir.mkdir(parents=True, exist_ok=True)

    # ── 三個串流輸出 ────────────────────────────────────────────────────
    # ⚠ 全部邊跑邊寫、定期 flush,**不進記憶體 list** —— 這台機器記憶體吃緊,
    #   而且中途若被 OOM 砍掉,已寫出的部分仍然可以分析。
    f_tracks = (out_dir / "tracks.csv").open("w", newline="", encoding="utf-8")
    f_tev = (out_dir / "track_events.csv").open("w", newline="", encoding="utf-8")
    f_res = (out_dir / "resident.csv").open("w", newline="", encoding="utf-8")
    w_tracks, w_tev, w_res = csv.writer(f_tracks), csv.writer(f_tev), csv.writer(f_res)
    w_tracks.writerow(["loop_i", "video_fid", "t_sec", "camera_id", "track_id",
                       "x1", "y1", "x2", "y2", "conf", "hits", "start_frame", "chef_id"])
    w_tev.writerow(["loop_i", "video_fid", "t_sec", "camera_id", "kind", "track_id"])
    w_res.writerow(["loop_i", "t_sec", "rss_mb", "total_chefs", "active", "gone",
                    "track_to_chef", "_exit", "_cam", "_pending_exit", "_world", "_vel"]
                   + [f"m4_{k}_{c}" for k in ("seen", "history", "prev_removed")
                      for c in cams])
    proc = psutil.Process()

    per_cam_loops = {c: 0 for c in cams}
    loop_ms, last_fid, t_last = [], {c: -1 for c in cams}, 0.0
    rows, n_frames, n_det, t0 = [], 0, 0, time.time()
    n_det_cam = {c: 0 for c in cams}
    with out_path.open("w", encoding="utf-8") as fout:
        while args.max_frames < 0 or n_frames < args.max_frames:
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

            t_loop = time.time()
            per_cam = {}
            for c, (fid, t_cam, frame) in frames.items():
                det = detect_person(model, frame, args.thr, person_cls)
                n_det += len(det)
                n_det_cam[c] += len(det)
                out = trackers[c].update(det, n_frames, timestamp=t_cam)
                per_cam[c] = (frame, out)
                per_cam_loops[c] += 1
                last_fid[c] = fid
            t_last = max(t_last, t_sec)

            # ⚠ tick() 目前只在 on_new_track 裡被呼叫(identity.py:79)。單人長跑時
            #    可能好幾分鐘沒有新 track,期間 gone 池完全不會被回收 ——
            #    於是規格那條「活躍清單記憶體不無限成長」根本沒有機會成立。
            #    心跳既然已經是這條管線的既有模式,過期回收也該走心跳。
            #    ⚠ 這是 driver 端的補救,**library 沒改**:任何不呼叫 tick 的部署仍會漏,
            #      這件事要寫進報告的已知缺口。
            m5.tick(n_frames)

            # ① 心跳:先把「誰此刻在畫面上、在哪裡」全部餵進去,再做綁定決策。
            #    M4 每幀都有 tracks,但 M5 只吃事件 —— 這個介面缺口是第五、六輪
            #    在模擬裡踩出來的,真實管線同樣需要,而且順序必須在事件之前。
            for c, (_frame, out) in per_cam.items():
                for tr in out.tracks:
                    m5.on_track_update(tr.track_id, camera_id=c, frame_id=n_frames,
                                       t_sec=t_sec, bbox=tr.bbox,
                                       world_xy=topo.world_xy(c, tr.bbox))
                    # 逐幀軌跡:M4 在真實影片上的量化目前完全空白,這是唯一的資料來源。
                    # conf 為空 = 這一幀沒配對上、用 Kalman 預測框 → 「空轉」的證據。
                    b = tr.bbox or (None,) * 4
                    w_tracks.writerow([n_frames, last_fid[c], round(t_sec, 3), c,
                                       tr.track_id, *[None if v is None else round(float(v), 1)
                                                      for v in b],
                                       "" if tr.confidence is None else round(tr.confidence, 3),
                                       tr.hits, tr.start_frame,
                                       m5.track_to_chef.get((c, tr.track_id), "")])

            # ② 事件 → M5
            new_chef_this_loop = False
            for c, (frame, out) in per_cam.items():
                for ev in out.events:
                    # 四種事件全記。舊版只有 new_track 進 jsonl,於是「遮擋救回率」
                    # (reacquired / lost)這類 M4 的關鍵行為完全沒有紀錄可查。
                    w_tev.writerow([n_frames, last_fid[c], round(ev.t_sec, 3), c,
                                    ev.kind, ev.track_id])
                    if ev.kind == "new_track":
                        cr = crop_of(frame, ev.bbox) if ev.bbox is not None else None
                        r = m5.on_new_track(ev.track_id, camera_id=c, frame_id=ev.frame_id,
                                            t_sec=ev.t_sec, bbox=ev.bbox, crop=cr,
                                            world_xy=topo.world_xy(c, ev.bbox))
                        row = dict(loop_i=n_frames, video_fid=last_fid[c],
                                   t_sec=round(ev.t_sec, 3), camera_id=c,
                                   track_id=ev.track_id, chef_id=r.chef_id,
                                   matched=bool(r.matched), score=r.similarity,
                                   n_candidates=m5.last_candidates,
                                   bbox=[round(float(v), 1) for v in (ev.bbox or [])])
                        rows.append(row)
                        new_chef_this_loop |= not r.matched
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

            # ③ 存標註幀。⚠ 必須在事件處理**之後** —— 綁定決策就發生在這一幀,
            #    先畫的話新出現的人還沒拿到 chef_id,看起來像系統沒認出來。
            #    全長跑會產生數千張拼接 PNG,所以改成取樣;但「開新 chef_id」那幾幀
            #    一定要存 —— 那是最需要肉眼稽核的,碎裂到底是真的還是誤判就看它。
            every = args.save_frames_every
            want = new_chef_this_loop or (every > 0 and n_frames % every == 0)
            if frames_dir is not None and want:
                import cv2
                panels = [draw_panel(per_cam[c][0], per_cam[c][1].tracks, c, m5, t_sec)
                          for c in cams if c in per_cam]
                canvas = stitch(panels)
                if canvas is not None:
                    tag = "NEW" if new_chef_this_loop else "reg"
                    cv2.imwrite(str(frames_dir / f"f{n_frames:05d}_{tag}.png"), canvas)

            # ④ 常駐狀態取樣 —— 規格「活躍清單記憶體不無限成長」的唯一證據來源。
            #    M5 側有 TTL 應該有界;M4 側的 _seen_ids 是單調累積的集合,
            #    結構上是 O(累計 track 數) —— 要一起量才知道形狀,不能只看 RSS。
            if n_frames % args.resident_every == 0:
                rs = m5.resident_stats()
                w_res.writerow(
                    [n_frames, round(t_sec, 2), round(proc.memory_info().rss / 1e6, 1),
                     m5.stats()["total_chefs"], rs["active"], rs["gone"],
                     rs["track_to_chef"], rs["_exit"], rs["_cam"], rs["_pending_exit"],
                     rs["_world"], rs["_vel"]]
                    + [len(trackers[c]._seen_ids) for c in cams]
                    + [len(trackers[c]._history) for c in cams]
                    + [len(trackers[c]._prev_removed) for c in cams])
                for fh in (f_tracks, f_tev, f_res):
                    fh.flush()

            loop_ms.append((time.time() - t_loop) * 1000)
            n_frames += 1

    for fh in (f_tracks, f_tev, f_res):
        fh.close()

    dt = time.time() - t0
    q = sorted(loop_ms) or [0.0]
    ms = {"p50": round(q[len(q) // 2], 1), "p95": round(q[int(len(q) * .95) - 1], 1),
          "max": round(q[-1], 1)}
    coverage = round(n_frames / n_loops_full, 4) if n_loops_full else 0.0
    truncated = coverage < 0.999

    print()
    print("=" * 66)
    print(f"處理 {n_frames} 迴圈 × {len(cams)} 鏡頭,耗時 {dt/60:.1f} 分 "
          f"(p50 {ms['p50']:.0f} / p95 {ms['p95']:.0f} / max {ms['max']:.0f} ms/迴圈)")
    print(f"影片覆蓋:{coverage*100:.1f}%(處理到 t={t_last:.1f}s / "
          f"{max(m['nb_frames']/m['fps'] for m in metas.values()):.0f}s)")
    if truncated:
        print("  ⚠ **未跑完整支影片** —— 以下數字只代表這一段,不可當成全長結果")
    # ⚠ 主迴圈的 `if nxt is not None` 讓某支影片先耗盡時其他台會繼續跑,
    #   而 t_sec 取 min() 於是靜默倒退。EPFL 九支等長所以這裡應該永遠相等。
    if len(set(per_cam_loops.values())) > 1:
        print(f"  ⚠ 各鏡頭處理迴圈數不一致:{per_cam_loops} —— 跨鏡頭時間對齊已不可信")
    print(f"偵測到「人」{n_det} 次,各鏡頭:{n_det_cam}")
    if n_det == 0:
        print("  ⚠ 一個人都沒偵測到 —— 先確認 --person-cls 對不對(COCO=1、微調=0)")
    for c, n in n_det_cam.items():
        if n == 0:
            print(f"  ⚠ {c} 全程 0 次偵測 —— 該鏡頭等於沒接上")
    print(f"綁定決策 {len(rows)} 次,其中沿用既有 chef {sum(r['matched'] for r in rows)} 次")
    print(f"最終 chef_id 數:{m5.stats()['total_chefs']}")
    print(f"候選數分布(0=開新人、1=唯一候選):{m5.candidate_histogram()}")
    print(f"常駐狀態:{m5.resident_stats()}")

    # run_meta.json —— 下游自檢的唯一事實來源。把「這次到底跑了什麼」寫死,
    # 才能讓 eval 腳本斷言「這不是一次截斷跑 / config 沒讀錯 / 證據沒有假生效」。
    meta = {
        "camera_video_map": {c: str(v) for c, v in zip(cams, args.videos)},
        "videos": {c: metas[c] for c in cams},
        "topology_path": str(args.topology),
        "n_links": len(topo.links), "n_overlapping_pairs": len(topo.overlapping),
        "n_homographies": len(topo.homographies),
        "world_xy_available": bool(topo.homographies),
        "ground_plane_effective": topo.ground_lr is not None,
        "velocity_effective": topo.vel_lr is not None,
        "weights": args.weights, "person_cls": person_cls, "embedder": args.embedder,
        "stride": args.stride, "max_frames": args.max_frames,
        "ttl_loops": args.ttl, "ttl_seconds": args.ttl * args.stride / args.fps,
        "lost_track_buffer_loops": tcfg.get("lost_track_buffer"),
        "lost_track_buffer_seconds": (tcfg.get("lost_track_buffer", 30)
                                      * args.stride / args.fps),
        "n_loops": n_frames, "n_loops_full": n_loops_full,
        "coverage": coverage, "truncated": truncated,
        "per_cam_loops": per_cam_loops, "n_det": n_det, "n_det_per_cam": n_det_cam,
        "ms_per_loop": ms, "wall_seconds": round(dt, 1),
        "total_chefs": m5.stats()["total_chefs"],
        "candidate_histogram": m5.candidate_histogram(),
        "resident_final": m5.resident_stats(),
        "llr_threshold": getattr(topo, "llr_threshold", None),
    }
    (out_dir / "run_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"輸出:{out_path}")
    print(f"      {out_dir}/(tracks.csv, track_events.csv, resident.csv, run_meta.json)")
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
