"""M4 多目標追蹤:KitchenTracker(可切換 backend 的包裝)。

把 M3 每幀無 ID 的偵測(supervision Detections)串成有 track_id 的軌跡,
維持跨幀一致、撐短暫遮擋,並輸出 new_track / reacquired / lost_track / removed 事件。

backend(全部是純運動 Kalman+IoU+匈牙利,不含 Re-ID):
- `bytetrack`(預設):supervision 的 ByteTrack(MIT)。⚠ supervision 0.30 移除;requirements 釘 <0.30。
- `rf_botsort` / `rf_cbiou` / `rf_ocsort`:Roboflow `trackers` 套件(Apache 2.0,乾淨重寫)。
  2026-09-15 層 0 第二輪為了換掉「跟丟期間的配對方法」而加,見
  docs/M4_層0_關聯方法_預先登記_20260915.md。

兩個套件的 update 都只回傳本幀配到的偵測、不含事件,所以事件一律從 backend 內部的
track 狀態推導(supervision:tracked_tracks/lost_tracks/removed_tracks;trackers:tracks)。
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
    hits: int                      # 已配對到的幀數(bytetrack:tracklet_len;rf_*:active 次數)
    history: list = field(default_factory=list)   # 近期 bbox 歷史


@dataclass
class TrackEvent:
    """M4 → M5/中央 的事件。

    kind 語意(M5 的離場判定依賴這組語意,勿隨意更動):
      new_track   首次進入 active            → M5 抽特徵、綁 chef_id
      reacquired  從 lost 回到 active        → M5 取消預備出口(短暫遮擋,人沒走)
      lost_track  進入 ByteTrack lost        → M5 只記「預備出口」,不標 gone
      removed     lost buffer 到期、真的離開 → M5 標 gone,出口時間戳用 lost 當時的
    """
    kind: str
    track_id: int
    frame_id: int
    class_id: int | None = None
    bbox: tuple | None = None
    camera_id: str | None = None   # 哪一台相機(M5 跨鏡頭必需;track_id 各鏡頭不唯一)
    t_sec: float | None = None     # 事件時間(秒)。M5 的轉場時間窗以此計算


@dataclass
class TrackerOutput:
    frame_id: int
    tracks: list                   # 本幀所有 active 軌跡(list[Track])
    events: list                   # 本幀觸發的事件(list[TrackEvent])


class BaseTracker:
    """可切換追蹤器介面。未來 BoT-SORT / OC-SORT 實作同一份 update 契約即可。"""

    def update(self, detections, frame_id, timestamp=None, frame=None) -> TrackerOutput:
        raise NotImplementedError

    def reset(self) -> None:
        raise NotImplementedError


# 建構參數只餵這些給 sv.ByteTrack(其餘是 wrapper 自用)
_BYTETRACK_KEYS = ("track_activation_threshold", "lost_track_buffer",
                   "minimum_matching_threshold", "frame_rate", "minimum_consecutive_frames")

RF_BACKENDS = ("rf_botsort", "rf_cbiou", "rf_ocsort", "rf_mcbyte", "rf_mcbyte_nomask",
               "rf_mcbyte_dcm", "rf_mcbyte_nomask_dcm")
# McByte 需要每幀的 RGB 影像(遮罩傳遞);其他 backend 不收影像
MCBYTE_BACKENDS = ("rf_mcbyte", "rf_mcbyte_nomask", "rf_mcbyte_dcm", "rf_mcbyte_nomask_dcm")
# 2026-09-15 修法第 2 輪:*_dcm = McByte + SparseTrack 偽深度分層配對(src/m4_track/sparse_dcm.py);
# 層數由 backend_params 的 depth_levels_high / depth_levels_low 給,不給則都是 1(= 不分層)
# McByte 遮罩權重的固定位置(model_result/ 已被 .gitignore 排除)
# ⚠ Cutie cutie-base-mega 的訓練資料含 MOSE(CC BY-NC-SA 4.0,非商用)→ 只能用於驗證,不可出貨
MCBYTE_SAM_CKPT = "model_result/mcbyte/sam_vit_b_01ec64.pth"
MCBYTE_CUTIE_CKPT = "model_result/mcbyte/cutie-base-mega.pth"


def rf_kwargs(backend, common, params=None):
    """把共用鍵(supervision ByteTrack 的語意)對映成 trackers 的建構參數。

    目標是**同一份設定在兩個套件裡代表同一套配對規則**,讓網格裡每一格只差一件事:
      bytetrack  → rf_botsort   只差卡爾曼狀態表示(supervision XYAH / trackers XCYCWH 尺度相關雜訊)
      rf_botsort → rf_cbiou     只差框放大(buffer_ratio;設 0 與 rf_botsort 逐位相同,verify_m4 S6 驗)

    supervision 0.28 的規則(2026-09-15 讀 byte_tracker/core.py 確認):
      高分:score > track_activation_threshold;低分:0.1 < score < 同一門檻
      第一輪:IoU × 分數 > 1 − minimum_matching_threshold,對象 tracked + lost
      第二輪:純 IoU > 0.5,只對 tracked
      未確認 track:IoU × 分數 > 0.3,沒配到就刪
      開新 track:score ≥ track_activation_threshold + 0.1,第二次配到才啟動(tracker 的第一幀例外)
    ⚠ 邊界比較(> 與 ≥)兩邊不同,不追求逐位相同 —— 所以網格需要 rf_botsort 當橋樑格。

    OC-SORT 沒有「開新 track 門檻」與「未確認 track」的概念(低於 high_conf_det_threshold
    的偵測整批丟掉、其餘全部可開新 track),只對映高分門檻,其餘用套件預設。
    `params`(configs/tracker.yaml 的 backend_params.<backend>)最後覆蓋。
    """
    act = float(common["track_activation_threshold"])
    kw = dict(lost_track_buffer=int(common["lost_track_buffer"]),
              frame_rate=float(common["frame_rate"]),
              high_conf_det_threshold=act)
    if backend in ("rf_botsort", "rf_cbiou") + MCBYTE_BACKENDS:
        kw.update(track_activation_threshold=act + 0.1,
                  minimum_iou_threshold_first_assoc=1.0 - float(common["minimum_matching_threshold"]),
                  minimum_iou_threshold_second_assoc=0.5,
                  minimum_iou_threshold_unconfirmed_assoc=0.3,
                  # supervision「第二次配到才啟動」= trackers 的 2 次成功更新;
                  # supervision 自己的 minimum_consecutive_frames 是啟動之後再延遲發 id
                  minimum_consecutive_frames=int(common["minimum_consecutive_frames"]) + 1,
                  instant_first_frame_activation=True)
        if backend == "rf_botsort":
            kw["enable_cmc"] = False       # 固定鏡頭,且評估不解碼影片、沒有 frame 可給
        if backend in MCBYTE_BACKENDS:
            # 2026-09-15 修法第 1 輪:McByte = ByteTrack 式兩階段 + 傳遞的分割遮罩當關聯線索。
            # 固定鏡頭 → 關相機運動補償;rf_mcbyte_nomask 是只差遮罩的橋樑格。
            kw["enable_cmc"] = False
            kw["enable_mask_manager"] = backend in ("rf_mcbyte", "rf_mcbyte_dcm")
    elif backend != "rf_ocsort":
        raise ValueError(f"不是 trackers 的 backend:{backend}")
    kw.update(params or {})
    return kw


class KitchenTracker(BaseTracker):
    def __init__(self, backend="bytetrack",
                 track_activation_threshold=0.25, lost_track_buffer=30,
                 minimum_matching_threshold=0.8, frame_rate=30,
                 minimum_consecutive_frames=1, history_len=30, class_names=None,
                 camera_id=None, backend_params=None):
        self.backend = backend
        self.history_len = history_len
        self.class_names = class_names or {}
        self.camera_id = camera_id      # 每台相機一個 tracker 實例;會蓋進所有事件
        self.frame_rate = frame_rate    # 未給 timestamp 時用來換算 t_sec
        self._bt_kwargs = dict(track_activation_threshold=track_activation_threshold,
                               lost_track_buffer=lost_track_buffer,
                               minimum_matching_threshold=minimum_matching_threshold,
                               frame_rate=frame_rate,
                               minimum_consecutive_frames=minimum_consecutive_frames)
        self.backend_params = dict(backend_params or {})
        self._build_backend()
        self._reset_state()

    @classmethod
    def from_config(cls, cfg, class_names=None, camera_id=None):
        """cfg = yaml 的 tracker 區段 dict。"""
        kw = {k: cfg[k] for k in _BYTETRACK_KEYS if k in cfg}
        backend = cfg.get("backend", "bytetrack")
        return cls(backend=backend,
                   history_len=cfg.get("history_len", 30),
                   class_names=class_names, camera_id=camera_id,
                   backend_params=(cfg.get("backend_params") or {}).get(backend), **kw)

    def _build_backend(self):
        self._bt = self._rf = None
        if self.backend == "bytetrack":
            import supervision as sv
            self._bt = sv.ByteTrack(**self._bt_kwargs)
            return
        if self.backend not in RF_BACKENDS:
            raise NotImplementedError(f"backend '{self.backend}' 尚未實作"
                                      f"(可用:bytetrack、{'、'.join(RF_BACKENDS)})")
        from trackers import BoTSORTTracker, CBIoUTracker, McByteTracker, OCSORTTracker
        from m4_track.sparse_dcm import DCMMcByteTracker
        cls = dict(rf_botsort=BoTSORTTracker, rf_cbiou=CBIoUTracker, rf_ocsort=OCSORTTracker,
                   rf_mcbyte=McByteTracker, rf_mcbyte_nomask=McByteTracker,
                   rf_mcbyte_dcm=DCMMcByteTracker, rf_mcbyte_nomask_dcm=DCMMcByteTracker)[self.backend]
        kw = rf_kwargs(self.backend, self._bt_kwargs, self.backend_params)
        if kw.get("enable_mask_manager"):
            from trackers.core.mcbyte.tracker import McByteMaskConfig
            kw["mask_config"] = McByteMaskConfig(sam_checkpoint_path=MCBYTE_SAM_CKPT,
                                                 cutie_weights_path=MCBYTE_CUTIE_CKPT)
        self._rf = cls(**kw)

    def _reset_state(self):
        self._seen_ids = set()          # 曾經 active 過的 id(判 new_track)
        self._prev_lost = set()         # 上幀在 lost 的 id(判 lost_track 轉換)
        self._prev_removed = set()      # 已發過 removed 的 id(removed_tracks 會累積)
        self._history = {}              # id -> deque(bbox)
        self._last_class = {}           # id -> 最近已知 class_id
        self._prev_alive = set()        # rf_*:上幀還在 tracks 裡、已發 id 的 track
        self._hits = {}                 # rf_*:id -> active 次數
        self._start = {}                # rf_*:id -> new_track 的影格

    def reset(self):
        (self._bt if self._bt is not None else self._rf).reset()
        self._reset_state()

    def update(self, detections, frame_id, timestamp=None, frame=None) -> TrackerOutput:
        # frame:RGB 影像,只有 McByte 後端會用(遮罩);其他後端忽略
        # ByteTrack 需要 confidence;空幀也要呼叫,好讓 lost/removed 計時前進
        if len(detections) and getattr(detections, "confidence", None) is None:
            raise ValueError("detections.confidence 不可為 None(ByteTrack 需要分數)")

        t_sec = float(timestamp) if timestamp is not None else frame_id / float(self.frame_rate)
        if self._rf is not None:
            return self._update_rf(detections, frame_id, t_sec, frame)

        matched = self._bt.update_with_detections(detections)
        matched_map = {}
        for i in range(len(matched)):
            tid = int(matched.tracker_id[i])
            cid = int(matched.class_id[i]) if matched.class_id is not None else None
            conf = float(matched.confidence[i]) if matched.confidence is not None else None
            matched_map[tid] = (tuple(float(v) for v in matched.xyxy[i]), cid, conf)

        strack_by_id = {t.external_track_id: t for t in self._bt.tracked_tracks}
        active_ids = {t.external_track_id for t in self._bt.tracked_tracks if t.is_activated}
        # ⚠ 從未啟動的 track(只出現一幀、沒被確認)在 supervision 裡的 external id
        #   一律是 NO_ID(= -1)。而 `removed_tracks` 是**逐幀覆寫**的(core.py 的
        #   `self.removed_tracks = removed_stracks`),於是 -1 會反覆進出集合,
        #   下面的差集每次都把它當成「新被移除」發一個 removed 事件。
        #   2026-09-14 實測:CHIRLA 七序列 7,779 次 removed 裡 **4,192 次是 -1**(54%),
        #   EPFL 九台也有 129 次。那些不是「救不回的遺失」,是從沒成為 track 的偵測。
        #   M5 本來就查不到 (cam, -1) 所以決策不受影響,但層 0 的統計被灌水。
        #   用 tracker 實例上的 NO_ID property 而不寫死 -1。
        no_id = self._bt.external_id_counter.NO_ID
        lost_ids = {t.external_track_id for t in self._bt.lost_tracks
                    if t.external_track_id != no_id}
        removed_ids = {t.external_track_id for t in self._bt.removed_tracks
                       if t.external_track_id != no_id}

        def _last_box(tid):
            """最後已知位置。track 進 lost 後就不在 tracked_tracks 裡,只能從歷史取。

            M5 的同鏡頭重關聯需要「離場位置」才能分辨斷軌前後是不是同一個人
            —— 只有時間的話,同一台鏡頭裡有多人時會綁錯。
            """
            h = self._history.get(tid)
            return h[-1] if h else None

        def ev(kind, tid, box=None):
            return TrackEvent(kind, tid, frame_id, self._last_class.get(tid),
                              box if box is not None else _last_box(tid),
                              camera_id=self.camera_id, t_sec=t_sec)

        events = []
        # new_track:首次進入 active
        for tid in active_ids - self._seen_ids:
            self._seen_ids.add(tid)
            box = matched_map[tid][0] if tid in matched_map else (
                tuple(float(v) for v in strack_by_id[tid].tlbr) if tid in strack_by_id else None)
            cid = matched_map[tid][1] if tid in matched_map else None
            events.append(TrackEvent("new_track", tid, frame_id, cid, box,
                                     camera_id=self.camera_id, t_sec=t_sec))
        # reacquired:上幀還在 lost、本幀回到 active(短暫遮擋後找回,人並沒有離開)
        # 沒有這個事件的話,M5 會在 lost_track 時把該 chef 標成 gone 後永遠卡住
        # ——因為 new_track 只認 active_ids - _seen_ids,而 _seen_ids 是累積的。
        for tid in sorted(active_ids & self._prev_lost):
            box = matched_map[tid][0] if tid in matched_map else (
                tuple(float(v) for v in strack_by_id[tid].tlbr) if tid in strack_by_id else None)
            events.append(ev("reacquired", tid, box))
        # lost_track:本幀新進 lost
        for tid in lost_ids - self._prev_lost:
            events.append(ev("lost_track", tid))
        # removed:本幀新進 removed(對累積清單做差集)
        for tid in removed_ids - self._prev_removed:
            events.append(ev("removed", tid))
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

    def _update_rf(self, detections, frame_id, t_sec, frame=None) -> TrackerOutput:
        """trackers 套件的 backend。

        ⚠ 不傳 timestamp(固定步長模式)。傳了的話 trackers 會改以「秒」算緩衝,
          而且卡爾曼一步走 經過秒數 × frame_rate 幀 —— supervision 是每次 update 走一步,
          跟 bytetrack 比就多了一個變因。緩衝秒數由呼叫端換算成更新次數
          (`eval_m4_chirla.tracker_config`),與 bytetrack 相同。

        狀態只看**已發 id** 的 track(tracker_id == -1 的未確認 track 不發任何事件,
        對應 bytetrack 路徑略過 NO_ID):
          active   本次 update 有配到偵測(time_since_update == 0)
          lost     還在 tracks 裡、本次沒配到
          removed  上次還在、這次被 trackers 刪掉(緩衝到期)

        ⚠ 與 bytetrack 路徑的差異:沒有「active 但本幀未配對 → 輸出預測框」這一支。
          supervision 其實也一樣 —— 沒配到的 tracked track 當幀就轉 lost(core.py),
          那一支只在 update_with_detections 事後以 IoU 0.5 找不回框時才會走到。
        """
        if self.backend in MCBYTE_BACKENDS:
            if frame is None:
                raise ValueError(f"{self.backend} 需要每幀的 RGB 影像(frame=)")
            out = self._rf.update(detections, frame=frame)
        else:
            out = self._rf.update(detections)
        rows, unassigned = {}, []
        for i in range(len(out)):
            row = (tuple(float(v) for v in out.xyxy[i]),
                   int(out.class_id[i]) if out.class_id is not None else None,
                   float(out.confidence[i]) if out.confidence is not None else None)
            tid = int(out.tracker_id[i])
            if tid == -1:
                unassigned.append(row)
            else:
                rows[tid] = row

        alive = {t.tracker_id: t for t in self._rf.tracks if t.tracker_id != -1}
        active_ids = {tid for tid, t in alive.items() if t.time_since_update == 0}
        lost_ids = set(alive) - active_ids
        removed_ids = self._prev_alive - set(alive)

        def matched(tid):
            """本幀配到的 (框, 類別, 分數)。

            ⚠ OC-SORT 的 track 跟丟後連續配對次數歸零,要再連續配到
              minimum_consecutive_frames 次,輸出才會再給 id(之前輸出 -1),
              但內部 tracker_id 不變、也確實配到了偵測 —— 對 M5 而言人沒換,
              所以照樣算 active,框從 last_observation 對回輸出列。
            """
            if tid in rows:
                return rows[tid]
            obs = getattr(alive[tid], "last_observation", None)
            if obs is not None:
                key = tuple(float(v) for v in obs)
                for r in unassigned:
                    if r[0] == key:
                        return r
            box = tuple(float(v) for v in alive[tid].get_state_bbox())
            return box, self._last_class.get(tid), None

        def ev(kind, tid, box=None):
            h = self._history.get(tid)
            return TrackEvent(kind, tid, frame_id, self._last_class.get(tid),
                              box if box is not None else (h[-1] if h else None),
                              camera_id=self.camera_id, t_sec=t_sec)

        events = []
        for tid in sorted(active_ids - self._seen_ids):
            self._seen_ids.add(tid)
            self._start[tid] = frame_id
            box, cid, _ = matched(tid)
            events.append(TrackEvent("new_track", tid, frame_id, cid, box,
                                     camera_id=self.camera_id, t_sec=t_sec))
        for tid in sorted(active_ids & self._prev_lost):
            events.append(ev("reacquired", tid, matched(tid)[0]))
        for tid in sorted(lost_ids - self._prev_lost):
            events.append(ev("lost_track", tid))
        for tid in sorted(removed_ids):
            events.append(ev("removed", tid))
            for d in (self._history, self._last_class, self._hits, self._start):
                d.pop(tid, None)
        self._prev_lost = lost_ids
        self._prev_alive = set(alive)

        tracks = []
        for tid in sorted(active_ids):
            box, cid, conf = matched(tid)
            if cid is not None:
                self._last_class[tid] = cid
            hist = self._history.setdefault(tid, deque(maxlen=self.history_len))
            hist.append(box)
            self._hits[tid] = self._hits.get(tid, 0) + 1
            tracks.append(Track(
                track_id=tid, bbox=box, class_id=cid, confidence=conf,
                status=TrackStatus.ACTIVE, frame_id=frame_id,
                start_frame=self._start[tid], hits=self._hits[tid],
                history=list(hist)))
        return TrackerOutput(frame_id=frame_id, tracks=tracks, events=events)
