"""智慧廚房 M1~M5 整合管線:單一進入點、單一設定檔。

    M1 讀影片 → M2 動態偵測 → M3 物件偵測 → M4 單鏡頭追蹤 → M5 跨鏡頭 chef_id

這個資料夾可以單獨交付:各模組的實作在同層的 `src/`(完整 repo 的 `src/` 的快照,
來源版本見 MANIFEST.json),這支程式只做串接。所有定案參數集中在
`pipeline_config.yaml`,用 `--profile` 切換設定組合。
設定檔與命令列裡的相對路徑(影片除外)一律相對於本資料夾。

與 `scripts/m5_track_video.py` 的關係:
  · M3 → M4 → M5 的主迴圈順序(tick → 心跳 → resolve_frame → 事件 → 存圖 → 常駐取樣)
    逐步照抄。M2 閘門關閉(預設)且參數相同時,`tracks.csv` / `track_events.csv` /
    `chef_events.jsonl` 與它逐位相同,所以既有的評估腳本(eval_clips.py、
    eval_m4m5_chirla.py)可以直接吃這裡的輸出。
  · 這裡多了 M2(分支 A 與分支 B 兩條線)與 M3 的 11 類逐類門檻輸出。
  · 這裡**沒有**四層歸因診斷(--track-gt)與上限實驗(--oracle-cue)。
    那兩個是評估工具,會讓真值進入流程,要用請回到 scripts/m5_track_video.py。

輸出(全部在 --out 目錄):
  chef_events.jsonl   每次綁定決策一筆(camera_id / track_id / chef_id / score / n_candidates)
  tracks.csv          逐幀逐 track 的框與 chef_id
  track_events.csv    M4 事件(new_track / reacquired / lost_track / removed)
  resident.csv        常駐狀態與記憶體取樣
  m2_motion.csv       每台每迴圈的 has_motion、motion_ratio、M3 是否實際執行
  m2_targets.jsonl    分支 B 的候選(動態線 / 靜態線啟用時才有)
  detections.csv      M3 的 11 類物件(有微調權重且非快取模式時才有)
  run_meta.json       這次到底跑了什麼(下游自檢的唯一事實來源)

用法:
  # CHIRLA seq_025 片段,讀資料夾自帶的偵測快取(不需要 GPU)
  python run_pipeline.py \\
      --videos <camera_1.avi> <camera_2.avi> <camera_3.avi> \\
      --cameras camera_1 camera_2 camera_3 \\
      --det-cache demo/det_cache/seq_025 \\
      --start-fid 1780 --end-fid 1890 --out outputs/demo

  # 廚房影片,微調 11 類權重,即時偵測
  python run_pipeline.py --videos <a.mp4> <b.mp4> --cameras cam1 cam2 \\
      --weights weights/m3_nano_r2_armA/checkpoint_best_total.pth \\
      --topology <該場地的拓撲檔> --out outputs/kitchen
"""
import argparse
import copy
import csv
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import psutil
import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parent        # 本資料夾;src/、weights/ 都在它底下
sys.path.insert(0, str(ROOT / "src"))
from common.draw_tracks import draw_panel, stitch                # noqa: E402
from common.video_io import imwrite_unicode, iter_frames, video_meta   # noqa: E402
from common.zones import resolve_floor                            # noqa: E402
from m2_motion.branch_b import SmallTargetDetector                # noqa: E402
from m2_motion.classifier import TargetClassifier                 # noqa: E402
from m2_motion.detector import MotionDetector                     # noqa: E402
from m2_motion.static_line import StaticTargetDetector            # noqa: E402
from m3.classes import NAMES, ROMAN                               # noqa: E402
from m3.model import load_rfdetr, model_resolution                # noqa: E402
from m4_track import KitchenTracker                               # noqa: E402
from m4_track.det_cache import DetCache, weights_id               # noqa: E402
from m5_reid.identity_st import SpatioTemporalIdentityManager     # noqa: E402
from m5_reid.spatiotemporal import CameraTopology                 # noqa: E402

DEFAULT_CONFIG = Path(__file__).resolve().parent / "pipeline_config.yaml"

# 「人」的 class_id 依權重而異。接錯的話一個人都偵測不到,而且不會報錯,
# 只會安靜地輸出 0 筆綁定(見 scripts/m5_track_video.py 檔頭)。
PERSON_CLS_COCO = 1
PERSON_CLS_FINETUNED = 0


# ── 設定檔 ────────────────────────────────────────────────────────────
def deep_merge(base, over):
    """dict 逐層合併,其餘型別(含 list)整段取代。"""
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(path, profile=None):
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    profiles = raw.get("profiles") or {}
    name = profile or raw.get("profile") or "delivery"
    if name not in profiles:
        raise SystemExit(f"設定檔沒有 profile '{name}'(有的是:{sorted(profiles)})")
    cfg = deep_merge({k: v for k, v in raw.items() if k != "profiles"}, profiles[name])
    cfg["profile"] = name
    return cfg


def rooted(path):
    """設定檔與命令列裡的相對路徑一律相對於本資料夾(與從哪裡執行無關)。"""
    if path is None:
        return None
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


# ── 資料夾自帶的預訓練權重 ────────────────────────────────────────────
PRETRAINED = ROOT / "weights" / "pretrained"


def use_bundled_torch_hub():
    """DINOv2 由 torch.hub 載入。資料夾有自帶快取就用它,離線也能跑;沒有就照常自動下載。"""
    hub = PRETRAINED / "torch_hub"
    if hub.is_dir():
        import torch
        torch.hub.set_dir(str(hub))


def use_bundled_coco_weights(variant):
    """rfdetr 的 COCO 預訓權重固定從使用者家目錄的 .roboflow/models 讀,沒有才下載。
    資料夾有自帶的話先放一份過去,省掉一次下載(目前只帶了 nano)。"""
    src = PRETRAINED / f"rf-detr-{variant}.pth"
    dst = Path.home() / ".roboflow" / "models" / src.name
    if src.is_file() and not dst.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


# ── M2 動態偵測 ───────────────────────────────────────────────────────
class M2Stage:
    """每台鏡頭各一組偵測器(背景模型是逐鏡頭的狀態,不能共用)。"""

    def __init__(self, cfg, cams):
        a, d, s = cfg["branch_a"], cfg["dynamic_line"], cfg["static_line"]
        self.a_on = bool(a.get("enabled", True))
        self.d_on = bool(d.get("enabled", False))
        self.s_on = bool(s.get("enabled", False))
        self.branch_a = ({c: MotionDetector(a["diff_threshold"], a["min_motion_ratio"],
                                            a["blur_ksize"], a["use_reference_frame"])
                          for c in cams} if self.a_on else {})
        self.dynamic = ({c: SmallTargetDetector(**{k: v for k, v in d.items() if k != "enabled"})
                         for c in cams} if self.d_on else {})
        self.static = {}
        if self.s_on:
            skip = ("enabled", "floor_zone_file", "floor_zone_name", "floor_zone",
                    "floor_zone_by_camera")
            kw = {k: v for k, v in s.items() if k not in skip}
            by_cam = s.get("floor_zone_by_camera") or {}
            shared = resolve_floor(s)
            self.static = {c: StaticTargetDetector(**kw, floor_zone=by_cam.get(c, shared))
                           for c in cams}
        clf = cfg.get("classifier") or {}
        self.classifier = TargetClassifier(
            onnx_path=rooted(clf.get("onnx")), input_size=clf.get("input_size", 224),
            target_index=clf.get("target_index", 1), threshold=clf.get("threshold", 0.5))

    def _confirm(self, crops, source):
        out = []
        for c in crops:
            is_t, conf = self.classifier.predict(c["crop"])
            # 沒有分類器權重時 is_t 是 None → 保留並標「待確認」(與 M2Pipeline 相同)
            if is_t is None or is_t:
                out.append(dict(source=source, bbox=c["bbox"], blob_bbox=c["blob_bbox"],
                                area=c["area"], is_target=is_t, confidence=round(conf, 4)))
        return out

    def process(self, cam, frame_bgr):
        # 分支 A 關閉時視為永遠有動態 —— 否則閘門會把所有幀都擋掉
        r = dict(has_motion=True, motion_ratio=None, targets=[])
        if self.a_on:
            a = self.branch_a[cam].process(frame_bgr)
            r.update(has_motion=bool(a["has_motion"]), motion_ratio=a["motion_ratio"])
        if self.d_on:
            r["targets"] += self._confirm(self.dynamic[cam].process(frame_bgr)["crops"], "dynamic")
        if self.s_on:
            r["targets"] += self._confirm(self.static[cam].process(frame_bgr)["crops"], "static")
        return r


# ── M3 物件偵測 ───────────────────────────────────────────────────────
def clone_det(det):
    """M2 閘門沿用上一次偵測時用。追蹤器後端可能就地改寫傳進去的物件,所以給複本。"""
    import supervision as sv
    if len(det) == 0:
        return sv.Detections.empty()
    return sv.Detections(xyxy=det.xyxy.copy(), confidence=det.confidence.copy(),
                         class_id=det.class_id.copy())


class M3Stage:
    def __init__(self, cfg, cams, det_cache=None):
        self.variant = cfg.get("variant", "nano")
        self.weights = rooted(cfg.get("weights"))
        self.finetuned = self.weights is not None
        self.thr = float(cfg.get("thr", 0.10))
        self.person_cls = (cfg["person_cls"] if cfg.get("person_cls") is not None
                           else (PERSON_CLS_FINETUNED if self.finetuned else PERSON_CLS_COCO))
        self.class_thr = (cfg.get("class_thresholds") or {}) if self.finetuned else {}
        # 逐類門檻可能比 thr 低(例如刀具 0.15 但 thr 0.2)→ 先用最低的門檻 predict 再各自把關
        self.base_thr = min([self.thr, *self.class_thr.values()])
        det_cache = rooted(det_cache)
        self.det_cache = None if det_cache is None else str(det_cache)
        self.caches = self.model = None
        self.resolution = None       # 實際推論解析度;快取模式下沒有模型,維持 None
        if det_cache:
            exp = dict(variant=self.variant, weights_id=weights_id(self.weights),
                       person_cls=self.person_cls)
            self.caches = {c: DetCache(Path(det_cache) / f"{c}.npz", expect=exp) for c in cams}
            print(f"M3:讀偵測快取 {det_cache}(不載入 RF-DETR),"
                  f"人的 class_id = {self.person_cls},門檻 {self.thr}")
        else:
            if not self.finetuned:
                use_bundled_coco_weights(self.variant)
            # auto:微調權重 → 訓練時的解析度;COCO 預訓 → 變體預設(見 src/m3/model.py)
            self.model = load_rfdetr(self.variant, self.weights, cfg.get("resolution", "auto"))
            self.resolution = model_resolution(self.model)
            print(f"M3:載入 RF-DETR-{self.variant}"
                  f"({'微調 11 類' if self.finetuned else 'COCO 預訓'}),"
                  f"推論解析度 {self.resolution},"
                  f"人的 class_id = {self.person_cls},門檻 {self.thr}")

    @property
    def emits_objects(self):
        """只有微調權重 + 即時偵測時才有 11 類物件可輸出(快取只存了人)。"""
        return self.finetuned and self.caches is None

    def detect(self, cam, fid, frame_bgr):
        """回傳 (persons, objects)。persons 給 M4;objects 是 [(class_id, conf, xyxy)]。"""
        import supervision as sv
        if self.caches is not None:
            return self.caches[cam].get(fid, self.thr), []
        det = self.model.predict(Image.fromarray(frame_bgr[:, :, ::-1]), threshold=self.base_thr)
        if len(det) == 0:
            return sv.Detections.empty(), []
        keep = np.array([int(c) == self.person_cls for c in det.class_id])
        if self.base_thr < self.thr:
            # 與 DetCache.get 相同:在 float32 下比較,才與 torch 在 predict 裡的判定一致
            keep &= det.confidence > np.float32(self.thr).astype(det.confidence.dtype)
        persons = (sv.Detections(xyxy=det.xyxy[keep], confidence=det.confidence[keep],
                                 class_id=det.class_id[keep])
                   if keep.any() else sv.Detections.empty())
        objects = []
        if self.finetuned:
            for i in range(len(det)):
                cid, conf = int(det.class_id[i]), float(det.confidence[i])
                if conf >= self.class_thr.get(NAMES.get(cid, ""), self.thr):
                    objects.append((cid, conf, tuple(float(v) for v in det.xyxy[i])))
        return persons, objects


# ── M5 外觀 embedder ──────────────────────────────────────────────────
def build_embedder(name):
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
        from m5_reid.dino_embedder import DINOv2Embedder
        return DINOv2Embedder()
    if name.startswith("chirla:"):
        from m5_reid.chirla_embedder import ChirlaEmbedder
        return ChirlaEmbedder(name.split(":", 1)[1])
    raise ValueError(f"未知的 embedder: {name}")


def crop_of(frame_bgr, bbox):
    h, w = frame_bgr.shape[:2]
    x1, y1, x2, y2 = (int(max(0, bbox[0])), int(max(0, bbox[1])),
                      int(min(w, bbox[2])), int(min(h, bbox[3])))
    if x2 <= x1 or y2 <= y1:
        return None
    return frame_bgr[y1:y2, x1:x2]


def load_topology(cfg):
    """回傳 (CameraTopology, 來源說明)。m5.topology_file 優先於設定檔內的 camera_topology。"""
    tf = cfg["m5"].get("topology_file")
    if tf:
        return CameraTopology.from_yaml(rooted(tf)), str(tf)
    inner = cfg.get("camera_topology") or {}
    # 與 CameraTopology.from_yaml 同一道守衛:兩者皆空時所有跨鏡頭關聯都會被拒,而且不報錯
    if not inner.get("links") and not inner.get("overlapping"):
        raise SystemExit("設定檔的 camera_topology 既沒有 links 也沒有 overlapping")
    return CameraTopology.from_config(inner), f"(設定檔內嵌,profile={cfg['profile']})"


# ── 主程式 ────────────────────────────────────────────────────────────
def parse_args():
    ap = argparse.ArgumentParser(description="智慧廚房 M1~M5 整合管線")
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--profile", default=None, help="覆蓋設定檔的 profile(delivery / single_person)")
    ap.add_argument("--videos", nargs="+", default=None)
    ap.add_argument("--cameras", nargs="+", default=None,
                    help="與 --videos 一一對應的 camera_id;預設用檔名")
    ap.add_argument("--out", default=None, help="輸出目錄(覆蓋 output.dir)")
    # 以下皆為覆蓋設定檔的常用項;不給就用設定檔的值
    ap.add_argument("--stride", type=int, default=None)
    ap.add_argument("--max-loops", type=int, default=None)
    ap.add_argument("--start-fid", type=int, default=None)
    ap.add_argument("--end-fid", type=int, default=None)
    ap.add_argument("--weights", default=None, help="M3 微調權重")
    ap.add_argument("--thr", type=float, default=None)
    ap.add_argument("--resolution", type=int, default=None,
                    help="M3 推論解析度(不給 = 設定檔的值,預設 auto)")
    ap.add_argument("--det-cache", default=None)
    ap.add_argument("--topology", default=None, help="外部拓撲檔(覆蓋 m5.topology_file)")
    ap.add_argument("--embedder", default=None)
    ap.add_argument("--gate-m3", action="store_true", help="啟用 M2 閘門(未經 M4/M5 評估驗證)")
    ap.add_argument("--save-frames", action="store_true")
    ap.add_argument("--save-frames-every", type=int, default=None)
    return ap.parse_args()


def apply_cli(cfg, args):
    for key, val in (("stride", args.stride), ("max_loops", args.max_loops),
                     ("start_fid", args.start_fid), ("end_fid", args.end_fid)):
        if val is not None:
            cfg["m1"][key] = val
    for key, val in (("weights", args.weights), ("thr", args.thr), ("det_cache", args.det_cache),
                     ("resolution", args.resolution)):
        if val is not None:
            cfg["m3"][key] = val
    if args.topology is not None:
        cfg["m5"]["topology_file"] = args.topology
    if args.embedder is not None:
        cfg["m5"]["embedder"] = args.embedder
    if args.gate_m3:
        cfg["m2"]["branch_a"]["gate_m3"] = True
    if args.out is not None:
        cfg["output"]["dir"] = args.out
    if args.save_frames:
        cfg["output"]["save_frames"] = True
    if args.save_frames_every is not None:
        cfg["output"]["save_frames_every"] = args.save_frames_every
    if args.videos:
        cams = args.cameras or [Path(v).stem for v in args.videos]
        if len(cams) != len(args.videos):
            raise SystemExit("--cameras 數量必須與 --videos 相同")
        cfg["m1"]["sources"] = dict(zip(cams, args.videos))
    return cfg


def main():
    args = parse_args()
    cfg = apply_cli(load_config(args.config, args.profile), args)
    m1, m2c, m5c, oc = cfg["m1"], cfg["m2"], cfg["m5"], cfg["output"]
    sources = m1.get("sources") or {}
    if not sources:
        raise SystemExit("沒有影片來源:請給 --videos,或在設定檔填 m1.sources")
    cams = list(sources)
    stride, fps = int(m1["stride"]), float(m1["fps"])
    start_fid, end_fid, max_loops = int(m1["start_fid"]), int(m1["end_fid"]), int(m1["max_loops"])
    ttl = int(m5c["ttl_loops"])
    gate = bool(m2c["branch_a"].get("gate_m3", False)) and bool(m2c["branch_a"].get("enabled", True))
    max_hold = int(m2c["branch_a"].get("gate_max_hold_loops", 30))
    print(f"設定檔 {args.config}(profile = {cfg['profile']})")

    # ── M5 拓撲 ──
    topo, topo_src = load_topology(cfg)
    if (topo.fusion.get("cue") or {}).get("enabled", False):
        raise SystemExit("拓撲開了 fusion.cue(上限實驗)—— 整合管線不支援,"
                         "請改用 scripts/m5_track_video.py --oracle-cue")
    unknown = [c for c in cams if c not in topo.all_cameras()]
    if unknown:
        print(f"注意:這些 camera_id 不在拓撲裡:{unknown}")
        print("  → 它們之間沒有連結也不重疊,跨鏡頭一律開新 chef_id。")
        print(f"  拓撲裡有的是:{sorted(topo.all_cameras())}")
    if not topo.homographies:
        print("注意:沒有任何鏡頭做地面校正 → 重疊路徑退回常數證據(overlap_llr,已知過度自信)")

    # ── 各階段 ──
    use_bundled_torch_hub()
    m2 = M2Stage(m2c, cams)
    m3 = M3Stage(cfg["m3"], cams, det_cache=cfg["m3"].get("det_cache"))
    tcfg = cfg["m4"]
    trackers = {c: KitchenTracker.from_config(tcfg, camera_id=c) for c in cams}
    emb = build_embedder(m5c["embedder"])
    m5 = SpatioTemporalIdentityManager(topo, embedder=emb, fps=fps / max(stride, 1),
                                       recently_disappeared_ttl=ttl)
    print(f"M2:分支 A {'開' if m2.a_on else '關'}"
          f"(閘門 {'開,無動態時沿用上次偵測' if gate else '關,只記錄'})、"
          f"動態線 {'開' if m2.d_on else '關'}、靜態線 {'開' if m2.s_on else '關'}、"
          f"分類器 {'有權重' if m2.classifier.available else '無權重(候選標待確認)'}")
    print(f"M4:backend = {tcfg.get('backend', 'bytetrack')};"
          f"M5:embedder = {m5c['embedder']}、拓撲 {topo_src}")

    # ── M1:開跑前先把規模講清楚(靜默截斷與靜默假通過是同一種病)──
    metas = {c: video_meta(v) for c, v in sources.items()}
    if start_fid % stride:
        raise SystemExit(f"start_fid {start_fid} 不在 stride={stride} 的格點上")
    if 0 <= end_fid < start_fid:
        raise SystemExit(f"end_fid {end_fid} < start_fid {start_fid}")

    def n_sample_loops(nb_frames):
        last = nb_frames - 1 if end_fid < 0 else min(end_fid, nb_frames - 1)
        return 0 if last < start_fid else (last - start_fid) // stride + 1

    n_loops_full = min(n_sample_loops(m["nb_frames"]) for m in metas.values())
    budget = n_loops_full if max_loops < 0 else min(max_loops, n_loops_full)
    print("影片:")
    for c, m in metas.items():
        print(f"  {c:<10}{Path(sources[c]).name:<40}"
              f"{m['nb_frames']} 幀 / {m['nb_frames']/m['fps']:.0f} 秒 / {m['fps']:.1f} fps")
    print(f"stride={stride} → 跑完需 {n_loops_full} 迴圈;本次會跑 {budget} 迴圈"
          f"({budget / max(n_loops_full, 1) * 100:.1f}%)")
    if budget < n_loops_full:
        print("  注意:這是截斷跑,所有輸出只代表這一段,不可當成全長結果。")
    print(f"TTL = {ttl} 迴圈 = {ttl * stride / fps:.0f} 秒影片時間")

    streams = {c: iter_frames(v, stride=stride, start=start_fid,
                              end=(None if end_fid < 0 else end_fid))
               for c, v in sources.items()}
    out_dir = rooted(oc["dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = None
    if oc.get("save_frames"):
        frames_dir = out_dir / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)
    every = int(oc.get("save_frames_every", 0))
    resident_every = int(oc.get("resident_every", 50))

    # ── 輸出檔:全部邊跑邊寫、定期 flush,中途被砍掉已寫出的部分仍可分析 ──
    def open_csv(name, header):
        fh = (out_dir / name).open("w", newline="", encoding="utf-8")
        w = csv.writer(fh)
        w.writerow(header)
        return fh, w

    f_tracks, w_tracks = open_csv("tracks.csv", [
        "loop_i", "video_fid", "t_sec", "camera_id", "track_id",
        "x1", "y1", "x2", "y2", "conf", "hits", "start_frame", "chef_id"])
    f_tev, w_tev = open_csv("track_events.csv", [
        "loop_i", "video_fid", "t_sec", "camera_id", "kind", "track_id"])
    f_res, w_res = open_csv("resident.csv", [
        "loop_i", "t_sec", "rss_mb", "total_chefs", "active", "gone",
        "track_to_chef", "_exit", "_cam", "_pending_exit", "_world", "_vel"]
        + [f"m4_{k}_{c}" for k in ("seen", "history", "prev_removed") for c in cams])
    f_m2, w_m2 = open_csv("m2_motion.csv", [
        "loop_i", "video_fid", "t_sec", "camera_id", "has_motion", "motion_ratio",
        "m3_ran", "n_targets"])
    handles = [f_tracks, f_tev, f_res, f_m2]
    f_tgt = f_obj = w_obj = None
    if m2.d_on or m2.s_on:
        f_tgt = (out_dir / "m2_targets.jsonl").open("w", encoding="utf-8")
        handles.append(f_tgt)
    if m3.emits_objects:
        f_obj, w_obj = open_csv("detections.csv", [
            "loop_i", "video_fid", "t_sec", "camera_id", "class_id", "class_name",
            "class_ascii", "conf", "x1", "y1", "x2", "y2"])
        handles.append(f_obj)
    proc = psutil.Process()

    per_cam_loops = {c: 0 for c in cams}
    last_fid = {c: -1 for c in cams}
    n_det_cam = {c: 0 for c in cams}
    n_motion = {c: 0 for c in cams}
    n_m3_run = {c: 0 for c in cams}
    last_det, hold = {}, {c: 0 for c in cams}
    stage_s = dict(m2=0.0, m3=0.0, m4=0.0, m5=0.0)
    loop_ms, rows = [], []
    n_frames, n_det, n_targets, n_objects, t_last, t0 = 0, 0, 0, 0, 0.0, time.time()
    clock = time.perf_counter

    with (out_dir / "chef_events.jsonl").open("w", encoding="utf-8") as fout:
        while max_loops < 0 or n_frames < max_loops:
            # ── M1:各鏡頭各取一張取樣幀 ──
            frames = {}
            for c, it in streams.items():
                nxt = next(it, None)
                if nxt is not None:
                    frames[c] = nxt
            if not frames:
                break
            # 用影片自己的時間戳,不用幀號換算 —— M5 的轉場時間窗完全建立在時間軸上
            t_sec = min(item[1] for item in frames.values())

            t_loop = time.time()
            per_cam = {}
            for c, (fid, t_cam, frame) in frames.items():
                # ── M2 ──
                tic = clock()
                m2r = m2.process(c, frame)
                stage_s["m2"] += clock() - tic
                n_motion[c] += int(m2r["has_motion"])

                # ── M3(M2 閘門開啟時,無動態就沿用該鏡頭上一次的偵測)──
                tic = clock()
                run_m3 = (not gate or m2r["has_motion"] or c not in last_det
                          or hold[c] >= max_hold)
                if run_m3:
                    det, objects = m3.detect(c, fid, frame)
                    hold[c] = 0
                    n_m3_run[c] += 1
                    if gate:
                        last_det[c] = (clone_det(det), objects)
                else:
                    det, objects = clone_det(last_det[c][0]), last_det[c][1]
                    hold[c] += 1
                stage_s["m3"] += clock() - tic
                n_det += len(det)
                n_det_cam[c] += len(det)

                w_m2.writerow([n_frames, fid, round(t_sec, 3), c, int(m2r["has_motion"]),
                               "" if m2r["motion_ratio"] is None else m2r["motion_ratio"],
                               int(run_m3), len(m2r["targets"])])
                for tg in m2r["targets"]:
                    n_targets += 1
                    f_tgt.write(json.dumps(dict(loop_i=n_frames, video_fid=fid,
                                                t_sec=round(t_sec, 3), camera_id=c, **tg),
                                           ensure_ascii=False) + "\n")
                if w_obj is not None:
                    for cid, conf, box in objects:
                        n_objects += 1
                        w_obj.writerow([n_frames, fid, round(t_sec, 3), c, cid,
                                        NAMES.get(cid, ""), ROMAN.get(cid, ""), round(conf, 3),
                                        *[round(v, 1) for v in box]])

                # ── M4 ──
                # McByte 系列(含外觀關聯)需要每幀畫面,依 tracker.py 的約定傳 RGB;
                # 其他後端不需要,不做多餘的轉換複製。
                tic = clock()
                out = trackers[c].update(
                    det, n_frames, timestamp=t_cam,
                    frame=(frame[:, :, ::-1] if trackers[c].needs_frame else None))
                stage_s["m4"] += clock() - tic
                per_cam[c] = (frame, out)
                per_cam_loops[c] += 1
                last_fid[c] = fid
            t_last = max(t_last, t_sec)

            # ── M5 ──
            tic = clock()
            # 過期回收走心跳:tick() 在 library 裡只由 on_new_track 呼叫,單人長跑時
            # 可能好幾分鐘沒有新 track,gone 池就不會被回收。這是 driver 端的補救。
            m5.tick(n_frames)

            # ① 心跳:先把「誰此刻在畫面上、在哪裡」全部餵進去,再做綁定決策。順序不可調換。
            for c, (_frame, out) in per_cam.items():
                for tr in out.tracks:
                    # 累積投票(revote)開啟時才抽 crop —— 那是每幀每條 track 的成本
                    cr = (crop_of(_frame, tr.bbox)
                          if getattr(topo, "revote", None) is not None
                          and tr.bbox is not None else None)
                    m5.on_track_update(tr.track_id, camera_id=c, frame_id=n_frames,
                                       t_sec=t_sec, bbox=tr.bbox, crop=cr,
                                       world_xy=topo.world_xy(c, tr.bbox), cue_token=None)
                    b = tr.bbox or (None,) * 4
                    w_tracks.writerow([n_frames, last_fid[c], round(t_sec, 3), c,
                                       tr.track_id, *[None if v is None else round(float(v), 1)
                                                      for v in b],
                                       "" if tr.confidence is None else round(tr.confidence, 3),
                                       tr.hits, tr.start_frame,
                                       m5.track_to_chef.get((c, tr.track_id), "")])
                # 沒開全域指派時是 no-op;但不呼叫的話票會卡在緩衝裡投不出去
                m5.resolve_frame(c)

            # ② 事件 → M5
            new_chef_this_loop = False
            for c, (frame, out) in per_cam.items():
                for ev in out.events:
                    w_tev.writerow([n_frames, last_fid[c], round(ev.t_sec, 3), c,
                                    ev.kind, ev.track_id])
                    if ev.kind == "new_track":
                        cr = crop_of(frame, ev.bbox) if ev.bbox is not None else None
                        r = m5.on_new_track(ev.track_id, camera_id=c, frame_id=ev.frame_id,
                                            t_sec=ev.t_sec, bbox=ev.bbox, crop=cr,
                                            world_xy=topo.world_xy(c, ev.bbox), cue_token=None)
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
                        m5.on_track_reacquired(ev.track_id, camera_id=c, frame_id=ev.frame_id,
                                               t_sec=ev.t_sec, bbox=ev.bbox)
                    elif ev.kind == "lost_track":
                        m5.on_track_lost(ev.track_id, camera_id=c, frame_id=ev.frame_id,
                                         t_sec=ev.t_sec, bbox=ev.bbox)
                    elif ev.kind == "removed":
                        m5.on_track_removed(ev.track_id, camera_id=c, frame_id=ev.frame_id,
                                            t_sec=ev.t_sec, bbox=ev.bbox)
            stage_s["m5"] += clock() - tic

            # ③ 存標註幀。必須在事件處理之後 —— 綁定決策就發生在這一幀,
            #    先畫的話新出現的人還沒拿到 chef_id。開新 chef_id 的那幾幀一定存。
            want = new_chef_this_loop or (every > 0 and n_frames % every == 0)
            if frames_dir is not None and want:
                canvas = stitch([draw_panel(per_cam[c][0], per_cam[c][1].tracks, c, m5, t_sec)
                                 for c in cams if c in per_cam])
                if canvas is not None:
                    tag = "NEW" if new_chef_this_loop else "reg"
                    imwrite_unicode(frames_dir / f"f{n_frames:05d}_{tag}.png", canvas)

            # ④ 常駐狀態取樣(「活躍清單記憶體不無限成長」的證據來源)
            if n_frames % resident_every == 0:
                rs = m5.resident_stats()
                w_res.writerow(
                    [n_frames, round(t_sec, 2), round(proc.memory_info().rss / 1e6, 1),
                     m5.stats()["total_chefs"], rs["active"], rs["gone"],
                     rs["track_to_chef"], rs["_exit"], rs["_cam"], rs["_pending_exit"],
                     rs["_world"], rs["_vel"]]
                    + [len(trackers[c]._seen_ids) for c in cams]
                    + [len(trackers[c]._history) for c in cams]
                    + [len(trackers[c]._prev_removed) for c in cams])
                for fh in handles:
                    fh.flush()

            loop_ms.append((time.time() - t_loop) * 1000)
            n_frames += 1

    for fh in handles:
        fh.close()

    # ── 摘要 ──
    dt = time.time() - t0
    q = sorted(loop_ms) or [0.0]
    ms = {"p50": round(q[len(q) // 2], 1), "p95": round(q[int(len(q) * .95) - 1], 1),
          "max": round(q[-1], 1)}
    coverage = round(n_frames / n_loops_full, 4) if n_loops_full else 0.0
    truncated = coverage < 0.999
    n_cam_loops = sum(per_cam_loops.values())
    motion_ratio = sum(n_motion.values()) / n_cam_loops if n_cam_loops else None
    m3_run_ratio = sum(n_m3_run.values()) / n_cam_loops if n_cam_loops else None

    print()
    print("=" * 66)
    print(f"處理 {n_frames} 迴圈 × {len(cams)} 鏡頭,耗時 {dt/60:.1f} 分"
          f"(p50 {ms['p50']:.0f} / p95 {ms['p95']:.0f} / max {ms['max']:.0f} ms/迴圈)")
    print("各階段累計秒數:" + "、".join(f"{k.upper()} {v:.1f}" for k, v in stage_s.items()))
    print(f"影片覆蓋:{coverage*100:.1f}%(處理到 t={t_last:.1f}s)")
    if truncated:
        print("  注意:未跑完整段 —— 以下數字只代表這一段")
    if len(set(per_cam_loops.values())) > 1:
        print(f"  注意:各鏡頭處理迴圈數不一致:{per_cam_loops} —— 跨鏡頭時間對齊已不可信")
    if m2.a_on and motion_ratio is not None:
        print(f"M2:有動態的比例 {motion_ratio*100:.1f}%;"
              f"M3 實際執行比例 {m3_run_ratio*100:.1f}%"
              + ("" if gate else "(閘門關閉,M3 每幀都跑;開啟後理論上可省 "
                 f"{(1 - motion_ratio)*100:.1f}%)"))
    if m2.d_on or m2.s_on:
        print(f"M2 分支 B:候選 {n_targets} 筆(m2_targets.jsonl)")
    print(f"M3:偵測到「人」{n_det} 次,各鏡頭:{n_det_cam}")
    if n_det == 0:
        print("  注意:一個人都沒偵測到 —— 先確認 person_cls 對不對(COCO = 1、微調 = 0)")
    for c, n in n_det_cam.items():
        if n == 0:
            print(f"  注意:{c} 全程 0 次偵測 —— 該鏡頭等於沒接上")
    if m3.emits_objects:
        print(f"M3:11 類物件 {n_objects} 筆(detections.csv)")
    print(f"M5:綁定決策 {len(rows)} 次,其中沿用既有 chef {sum(r['matched'] for r in rows)} 次;"
          f"最終 chef_id 數 {m5.stats()['total_chefs']}")
    print(f"    候選數分布(0 = 開新人、1 = 唯一候選):{m5.candidate_histogram()}")

    # run_meta.json:欄位名沿用 scripts/m5_track_video.py,既有評估腳本的自檢才讀得到
    meta = {
        "pipeline": "run_pipeline.py(整合資料夾)",
        "config_path": str(args.config), "profile": cfg["profile"],
        "camera_video_map": {c: str(v) for c, v in sources.items()},
        "videos": metas,
        "topology_path": topo_src,
        "n_links": len(topo.links), "n_overlapping_pairs": len(topo.overlapping),
        "n_homographies": len(topo.homographies),
        "world_xy_available": bool(topo.homographies),
        "ground_plane_effective": topo.ground_lr is not None,
        "velocity_effective": topo.vel_lr is not None,
        "weights": None if m3.weights is None else str(m3.weights),
        "variant": m3.variant, "person_cls": m3.person_cls, "thr": m3.thr,
        "m3_base_thr": m3.base_thr, "m3_class_thresholds": m3.class_thr,
        "m3_resolution": m3.resolution,
        "embedder": m5c["embedder"],
        "det_cache": m3.det_cache,
        "det_cache_meta": ({c: m3.caches[c].meta for c in cams}
                           if m3.caches is not None else None),
        "tracker_path": f"{args.config}#m4", "tracker_cfg": tcfg,
        "stride": stride, "max_frames": max_loops,
        "start_fid": start_fid, "end_fid": end_fid, "clip_id": None,
        "n_sample_loops_per_cam": {c: n_sample_loops(m["nb_frames"]) for c, m in metas.items()},
        "ttl_loops": ttl, "ttl_seconds": ttl * stride / fps,
        "lost_track_buffer_loops": tcfg.get("lost_track_buffer"),
        "lost_track_buffer_seconds": tcfg.get("lost_track_buffer", 30) * stride / fps,
        "n_loops": n_frames, "n_loops_full": n_loops_full,
        "coverage": coverage, "truncated": truncated,
        "per_cam_loops": per_cam_loops, "n_det": n_det, "n_det_per_cam": n_det_cam,
        "ms_per_loop": ms, "stage_seconds": {k: round(v, 2) for k, v in stage_s.items()},
        "wall_seconds": round(dt, 1),
        "total_chefs": m5.stats()["total_chefs"],
        "candidate_histogram": m5.candidate_histogram(),
        "resident_final": m5.resident_stats(),
        "llr_threshold": getattr(topo, "llr_threshold", None),
        "oracle_cue": None,
        "m2": {
            "branch_a": m2.a_on, "dynamic_line": m2.d_on, "static_line": m2.s_on,
            "classifier_ready": m2.classifier.available,
            "gate_m3": gate, "gate_max_hold_loops": max_hold,
            "n_motion_per_cam": n_motion, "n_m3_run_per_cam": n_m3_run,
            "motion_ratio": None if motion_ratio is None else round(motion_ratio, 4),
            "m3_run_ratio": None if m3_run_ratio is None else round(m3_run_ratio, 4),
            "n_targets": n_targets,
        },
        "n_objects": n_objects,
        "config": cfg,
    }
    (out_dir / "run_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"輸出目錄:{out_dir}")
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
