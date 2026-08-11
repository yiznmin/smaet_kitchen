"""M4 多目標追蹤:KitchenTracker(ByteTrack 包裝)。

把 M3 每幀無 ID 的偵測(supervision Detections)串成有 track_id 的軌跡,
維持跨幀一致、撐短暫遮擋,並輸出 new_track / lost_track / removed 事件。

- backend 目前用 supervision 的 ByteTrack(MIT,純運動 Kalman+IoU+匈牙利;不含 Re-ID)。
- ⚠ sv.ByteTrack 在 supervision 0.30 會移除;本 wrapper 是唯一觸點,將來換 `trackers`
  套件只需改 _build_backend 與 update 內少數幾行。requirements 釘 supervision<0.30。
- 事件/狀態從 ByteTrack 內部 tracked_tracks/lost_tracks/removed_tracks 推導
  (update_with_detections 只回傳有配對上的偵測,不含事件)。
"""
from collections import deque
from dataclasses import dataclass, field
from enum import Enum


class TrackStatus(str, Enum):
    ACTIVE = "active"
    LOST = "lost"
    REMOVED = "removed"


@dataclass
class Track:
    track_id: int
    bbox: tuple                    # (x1, y1, x2, y2) 本幀位置(未配對時為 Kalman 預測)
    class_id: int | None           # 最近一次配對到的類別(預測幀沿用上次已知)
    confidence: float | None       # 本幀配對到的信心;預測幀為 None
    status: TrackStatus
    frame_id: int                  # 本狀態對應的影格編號
    start_frame: int               # 軌跡首次建立的影格
    hits: int                      # 已配對到的幀數(tracklet_len)
    history: list = field(default_factory=list)   # 近期 bbox 歷史


@dataclass
class TrackEvent:
    kind: str                      # "new_track"(→M5) | "lost_track"(→中央) | "removed"
    track_id: int
    frame_id: int
    class_id: int | None = None
    bbox: tuple | None = None


@dataclass
class TrackerOutput:
    frame_id: int
    tracks: list                   # 本幀所有 active 軌跡(list[Track])
    events: list                   # 本幀觸發的事件(list[TrackEvent])


class BaseTracker:
    """可切換追蹤器介面。未來 BoT-SORT / OC-SORT 實作同一份 update 契約即可。"""

    def update(self, detections, frame_id, timestamp=None) -> TrackerOutput:
        raise NotImplementedError

    def reset(self) -> None:
        raise NotImplementedError


# 建構參數只餵這些給 sv.ByteTrack(其餘是 wrapper 自用)
_BYTETRACK_KEYS = ("track_activation_threshold", "lost_track_buffer",
                   "minimum_matching_threshold", "frame_rate", "minimum_consecutive_frames")


class KitchenTracker(BaseTracker):
    def __init__(self, backend="bytetrack",
                 track_activation_threshold=0.25, lost_track_buffer=30,
                 minimum_matching_threshold=0.8, frame_rate=30,
                 minimum_consecutive_frames=1, history_len=30, class_names=None):
        self.backend = backend
        self.history_len = history_len
        self.class_names = class_names or {}
        self._bt_kwargs = dict(track_activation_threshold=track_activation_threshold,
                               lost_track_buffer=lost_track_buffer,
                               minimum_matching_threshold=minimum_matching_threshold,
                               frame_rate=frame_rate,
                               minimum_consecutive_frames=minimum_consecutive_frames)
        self._build_backend()
        self._reset_state()

    @classmethod
    def from_config(cls, cfg, class_names=None):
        """cfg = yaml 的 tracker 區段 dict。"""
        kw = {k: cfg[k] for k in _BYTETRACK_KEYS if k in cfg}
        return cls(backend=cfg.get("backend", "bytetrack"),
                   history_len=cfg.get("history_len", 30),
                   class_names=class_names, **kw)

    def _build_backend(self):
        if self.backend != "bytetrack":
            raise NotImplementedError(f"backend '{self.backend}' 尚未實作(目前只有 bytetrack)")
        import supervision as sv
        self._bt = sv.ByteTrack(**self._bt_kwargs)

    def _reset_state(self):
        self._seen_ids = set()          # 曾經 active 過的 id(判 new_track)
        self._prev_lost = set()         # 上幀在 lost 的 id(判 lost_track 轉換)
        self._prev_removed = set()      # 已發過 removed 的 id(removed_tracks 會累積)
        self._history = {}              # id -> deque(bbox)
        self._last_class = {}           # id -> 最近已知 class_id

    def reset(self):
        self._bt.reset()
        self._reset_state()

    def update(self, detections, frame_id, timestamp=None) -> TrackerOutput:
        # ByteTrack 需要 confidence;空幀也要呼叫,好讓 lost/removed 計時前進
        if len(detections) and getattr(detections, "confidence", None) is None:
            raise ValueError("detections.confidence 不可為 None(ByteTrack 需要分數)")

        matched = self._bt.update_with_detections(detections)
        matched_map = {}
        for i in range(len(matched)):
            tid = int(matched.tracker_id[i])
            cid = int(matched.class_id[i]) if matched.class_id is not None else None
            conf = float(matched.confidence[i]) if matched.confidence is not None else None
            matched_map[tid] = (tuple(float(v) for v in matched.xyxy[i]), cid, conf)

        strack_by_id = {t.external_track_id: t for t in self._bt.tracked_tracks}
        active_ids = {t.external_track_id for t in self._bt.tracked_tracks if t.is_activated}
        lost_ids = {t.external_track_id for t in self._bt.lost_tracks}
        removed_ids = {t.external_track_id for t in self._bt.removed_tracks}

        events = []
        # new_track:首次進入 active
        for tid in active_ids - self._seen_ids:
            self._seen_ids.add(tid)
            box = matched_map[tid][0] if tid in matched_map else (
                tuple(float(v) for v in strack_by_id[tid].tlbr) if tid in strack_by_id else None)
            cid = matched_map[tid][1] if tid in matched_map else None
            events.append(TrackEvent("new_track", tid, frame_id, cid, box))
        # lost_track:本幀新進 lost
        for tid in lost_ids - self._prev_lost:
            events.append(TrackEvent("lost_track", tid, frame_id, self._last_class.get(tid)))
        # removed:本幀新進 removed(對累積清單做差集)
        for tid in removed_ids - self._prev_removed:
            events.append(TrackEvent("removed", tid, frame_id, self._last_class.get(tid)))
            self._history.pop(tid, None)
            self._last_class.pop(tid, None)
        self._prev_lost = lost_ids
        self._prev_removed = removed_ids

        # 組本幀 active 軌跡
        tracks = []
        for tid in sorted(active_ids):
            if tid in matched_map:
                box, cid, conf = matched_map[tid]
                if cid is not None:
                    self._last_class[tid] = cid
            else:  # active 但本幀未配對 → 用 Kalman 預測框,class 沿用上次已知
                st = strack_by_id.get(tid)
                box = tuple(float(v) for v in st.tlbr) if st is not None else None
                cid, conf = self._last_class.get(tid), None
            hist = self._history.setdefault(tid, deque(maxlen=self.history_len))
            if box is not None:
                hist.append(box)
            st = strack_by_id.get(tid)
            tracks.append(Track(
                track_id=tid, bbox=box, class_id=cid, confidence=conf,
                status=TrackStatus.ACTIVE, frame_id=frame_id,
                start_frame=st.start_frame if st is not None else frame_id,
                hits=st.tracklet_len if st is not None else 0,
                history=list(hist)))
        return TrackerOutput(frame_id=frame_id, tracks=tracks, events=events)
