"""SparseTrack 的偽深度分層配對(DCM)接到 McByte 上 —— M4 修法第 2 輪(2026-09-15)。

## 來源與授權

- 分層規則照抄 SparseTrack 官方實作 `tracker/sparse_tracker.py` 的 `get_deep_range`、`get_sub_mask`、`DCM`
  (https://github.com/hustvl/SparseTrack,MIT License,Copyright (c) 2023 Hust Vision Lab)
- `update` 主體複製自 `trackers` 2.6.0 的 `McByteTracker.update`
  (Roboflow,Apache License 2.0),**只改第 1、2 階段的配對呼叫**,其餘逐行不動

## 做法(與官方一致的部分)

- 偽深度 = 2000 − 框底 y(數值越小 = 框底越低 = 越靠近鏡頭)
- 偵測與 track **各自**在自己的最小~最大偽深度之間均分 `levels` 層;由近到遠逐層配對
- 某一層沒配到的偵測與 track,**接到下一層的清單後面**一起配
- 一邊的層數多於另一邊時,多出來的層不參與配對,直接列為未配對
- 官方設定:高分偵測 1 層(所有設定檔);低分偵測 MOT17 3 層、MOT20 8 層、DanceTrack 12 層

## 刻意與官方不同的部分(為了只測「分層」這一件事)

- 每一層的配對仍用 McByte 自己的相似度、分數融合、門檻與遮罩條件
  (`_get_mask_conditioned_associated_indices`),不換成 SparseTrack 的 `1 − IoU` 距離與它的門檻
- 因此 `levels = 1` 時必須與 `McByteTracker` 逐位相同 —— 由預先登記的硬性驗收檢查
"""
from __future__ import annotations

import warnings
from typing import cast

import numpy as np
import supervision as sv

from trackers.core.mcbyte.tracker import (
    _MINIMUM_DETECTION_CONFIDENCE,
    McByteTracker,
    _fuse_score,
    _get_alive_tracklets,
)
from trackers.utils.cmc import CMC
from trackers.utils.detections import default_confidences


def pseudo_depth(boxes_xyxy):
    """SparseTrack `deep_vec[2]`:2000 − 框底 y。"""
    b = np.asarray(boxes_xyxy, dtype=float).reshape(-1, 4)
    return 2000.0 - b[:, 3]


def get_deep_range(col, step):
    """逐行對應 SparseTrack `get_deep_range`(輸入改成偽深度陣列)。"""
    col = np.asarray(col, dtype=float)
    max_len, mix_len = max(col), min(col)
    if max_len != mix_len:
        deep_range = np.arange(mix_len, max_len, (max_len - mix_len + 1) / step)
        if deep_range[-1] < max_len:
            deep_range = np.concatenate([deep_range, np.array([max_len],)])
            deep_range[0] = np.floor(deep_range[0])
            deep_range[-1] = np.ceil(deep_range[-1])
    else:
        deep_range = [mix_len, ]
    return get_sub_mask(deep_range, col)


def get_sub_mask(deep_range, col):
    """逐行對應 SparseTrack `get_sub_mask`。"""
    mix_len = deep_range[0]
    max_len = deep_range[-1]
    if max_len == mix_len:
        lc = mix_len
    mask = []
    for d in deep_range:
        if d > deep_range[0] and d < deep_range[-1]:
            mask.append((col >= lc) & (col < d))
            lc = d
        elif d == deep_range[-1]:
            mask.append((col >= lc) & (col <= d))
            lc = d
        else:
            lc = d
    return mask


class DCMMcByteTracker(McByteTracker):
    """McByte + SparseTrack 偽深度分層配對。

    depth_levels_high:第 1 階段(高分偵測 vs confirmed + lost)的層數,官方 1
    depth_levels_low: 第 2 階段(低分偵測 vs 剩下的 tracked)的層數,官方 3 / 8 / 12
    """

    def __init__(self, *args, depth_levels_high: int = 1, depth_levels_low: int = 1, **kwargs):
        super().__init__(*args, **kwargs)
        if depth_levels_high < 1 or depth_levels_low < 1:
            raise ValueError("depth_levels 必須 ≥ 1")
        self.depth_levels_high = int(depth_levels_high)
        self.depth_levels_low = int(depth_levels_low)

    def _dcm(self, tracklets, det_boxes, predicted_state_boxes, levels, similarity_fn, min_similarity_thresh):
        """對應 SparseTrack `DCM`;每一層內部改用 McByte 的配對。

        回傳與 `_get_mask_conditioned_associated_indices` 相同格式:
        (原始 (track 索引, 偵測索引) 配對、排序過的未配對 track 索引、排序過的未配對偵測索引)。
        """
        n_t, n_d = len(tracklets), len(det_boxes)
        if n_d > 0:
            det_mask = get_deep_range(pseudo_depth(det_boxes), levels)
        else:
            det_mask = []
        if n_t != 0:
            trk_boxes = np.array([predicted_state_boxes[id(t)] for t in tracklets])
            track_mask = get_deep_range(pseudo_depth(trk_boxes), levels)
        else:
            track_mask = []

        matched = []
        u_detection, u_tracks, res_det, res_track = [], [], [], []
        if len(track_mask) != 0:
            if len(track_mask) < len(det_mask):
                for i in range(len(det_mask) - len(track_mask)):
                    res_det += [int(j) for j in np.argwhere(det_mask[len(track_mask) + i]).ravel()]
            elif len(track_mask) > len(det_mask):
                for i in range(len(track_mask) - len(det_mask)):
                    res_track += [int(j) for j in np.argwhere(track_mask[len(det_mask) + i]).ravel()]

            for dm, tm in zip(det_mask, track_mask):
                det_ = [int(j) for j in np.argwhere(dm).ravel()] + u_detection
                track_ = [int(j) for j in np.argwhere(tm).ravel()] + u_tracks
                sub_tracklets = [tracklets[i] for i in track_]
                sub_boxes = det_boxes[det_] if len(det_) else np.empty((0, 4))
                similarity, raw_iou = similarity_fn(sub_tracklets, det_)
                m, ut, ud = self._get_mask_conditioned_associated_indices(
                    similarity_matrix=similarity,
                    raw_iou_similarity=raw_iou,
                    tracklets=sub_tracklets,
                    detection_boxes=sub_boxes,
                    min_similarity_thresh=min_similarity_thresh,
                )
                matched += [(track_[r], det_[c]) for r, c in m]
                u_tracks = [track_[r] for r in ut]
                u_detection = [det_[c] for c in ud]

            u_tracks = u_tracks + res_track
            u_detection = u_detection + res_det
        else:
            u_detection = list(range(n_d))
        return matched, sorted(u_tracks), sorted(u_detection)

    def update(
        self,
        detections: sv.Detections,
        frame: np.ndarray | None = None,
        timestamp: float | None = None,
    ) -> sv.Detections:
        # ── 以下複製自 trackers 2.6.0 McByteTracker.update;只有標「DCM」的兩處不同 ──
        timing = self._predict_timing(timestamp)
        if timing.skip_update:
            return self._detections_for_skipped_update(detections)

        self.frame_id += 1
        current_frame = frame
        terminated_tracklet_ids: list[int] = []

        if timing.skip_predict:
            pass
        elif self.mask_manager is not None and current_frame is not None:
            if timing.uses_elapsed_time and not self._warned_mask_manager_dynamic_rate:
                warnings.warn(
                    "enable_mask_manager=True with timestamp-based (dynamic-rate) "
                    "updates: the mask pipeline advances one step per update() call "
                    "regardless of elapsed time, while Kalman prediction and "
                    "lost-track pruning scale by timestamp. Mask propagation can "
                    "drift out of sync with track state across timestamp gaps.",
                    UserWarning,
                    stacklevel=2,
                )
                self._warned_mask_manager_dynamic_rate = True
            self._last_mask_output = self._run_mask_manager(self.mask_manager, current_frame)
        else:
            self._last_mask_output = None

        if len(self.tracks) == 0 and len(detections) == 0:
            result = sv.Detections.empty()
            result.tracker_id = np.array([], dtype=int)
            self._store_previous_mask_inputs(
                frame=current_frame,
                detections=result,
                removed_tracklet_ids=terminated_tracklet_ids,
            )
            return result

        out_det_indices: list[int] = []
        out_tracker_ids: list[int] = []

        self._predict_tracklets(self.tracks, timing)

        _budget = self._lost_track_time_budget(timing, self.maximum_time_without_update)
        self._prune_lost_tracks(timing)

        detection_boxes = detections.xyxy
        confidences = default_confidences(detections)

        high_mask = confidences >= self.high_conf_det_threshold
        low_mask = (confidences > _MINIMUM_DETECTION_CONFIDENCE) & (~high_mask)

        high_indices = np.where(high_mask)[0]
        low_indices = np.where(low_mask)[0]

        high_boxes = detection_boxes[high_indices]
        low_boxes = detection_boxes[low_indices]
        high_scores = confidences[high_indices]

        confirmed_tracks = []
        unconfirmed_tracks = []
        lost_tracks = []
        for track in self.tracks:
            if track.time_since_update > 1:
                lost_tracks.append(track)
            elif track.tracker_id != -1 or track.number_of_successful_updates >= self.minimum_consecutive_frames:
                confirmed_tracks.append(track)
            else:
                unconfirmed_tracks.append(track)

        if self.enable_cmc and self.cmc is not None and current_frame is not None:
            mask_boxes = high_boxes if len(high_boxes) > 0 else None
            H = self.cmc.estimate(current_frame, mask_boxes)
            CMC.apply_batch(H, self.tracks)

        predicted_state_boxes = {id(track): track.get_state_bbox() for track in self.tracks}

        # Step 1 ── DCM:高分偵測 vs confirmed + lost,分 depth_levels_high 層
        strack_pool = confirmed_tracks + lost_tracks

        def sim_high(sub_tracklets, det_idx):
            raw = self._get_iou_matrix(sub_tracklets, high_boxes[det_idx] if len(det_idx) else np.empty((0, 4)),
                                       predicted_state_boxes)
            fused = _fuse_score(self.iou.normalize_for_fusion(raw.copy()), high_scores[det_idx])
            return fused, raw

        matched, unmatched_pool, unmatched_high = self._dcm(
            strack_pool, high_boxes, predicted_state_boxes, self.depth_levels_high,
            sim_high, self.minimum_iou_threshold_first_assoc)

        for row, col in matched:
            track = strack_pool[row]
            track.update(high_boxes[col])
            if track.number_of_successful_updates >= self.minimum_consecutive_frames and track.tracker_id == -1:
                track.tracker_id = self._allocate_tracker_id()
            out_det_indices.append(int(high_indices[col]))
            out_tracker_ids.append(track.tracker_id)

        # Step 2 ── DCM:低分偵測 vs 剩下的 tracked,分 depth_levels_low 層(不融合分數)
        remaining_tracked = [strack_pool[i] for i in unmatched_pool if strack_pool[i].time_since_update == 1]

        def sim_low(sub_tracklets, det_idx):
            raw = self._get_iou_matrix(sub_tracklets, low_boxes[det_idx] if len(det_idx) else np.empty((0, 4)),
                                       predicted_state_boxes)
            return raw, raw

        matched, _, unmatched_low = self._dcm(
            remaining_tracked, low_boxes, predicted_state_boxes, self.depth_levels_low,
            sim_low, self.minimum_iou_threshold_second_assoc)

        for row, col in matched:
            track = remaining_tracked[row]
            track.update(low_boxes[col])
            if track.number_of_successful_updates >= self.minimum_consecutive_frames and track.tracker_id == -1:
                track.tracker_id = self._allocate_tracker_id()
            out_det_indices.append(int(low_indices[col]))
            out_tracker_ids.append(track.tracker_id)

        for det_local_idx in sorted(unmatched_low):
            out_det_indices.append(int(low_indices[det_local_idx]))
            out_tracker_ids.append(-1)

        # Step 3 以下與 McByteTracker.update 相同
        unmatched_high_list = sorted(unmatched_high)
        unmatched_uc_indices: list[int] = list(range(len(unconfirmed_tracks)))

        if len(unconfirmed_tracks) > 0 and len(unmatched_high_list) > 0:
            uh_boxes = high_boxes[unmatched_high_list]
            uh_scores = high_scores[unmatched_high_list]

            raw_iou_similarity = self._get_iou_matrix(
                unconfirmed_tracks,
                uh_boxes,
                predicted_state_boxes,
            )
            association_similarity = _fuse_score(
                self.iou.normalize_for_fusion(raw_iou_similarity.copy()),
                uh_scores,
            )

            matched_uc, unmatched_uc_indices, remaining_uh = self._get_mask_conditioned_associated_indices(
                similarity_matrix=association_similarity,
                raw_iou_similarity=raw_iou_similarity,
                tracklets=unconfirmed_tracks,
                detection_boxes=uh_boxes,
                min_similarity_thresh=self.minimum_iou_threshold_unconfirmed_assoc,
            )

            for row, col in matched_uc:
                track = unconfirmed_tracks[row]
                orig_high_idx = unmatched_high_list[col]
                track.update(high_boxes[orig_high_idx])
                if track.number_of_successful_updates >= self.minimum_consecutive_frames and track.tracker_id == -1:
                    track.tracker_id = self._allocate_tracker_id()
                out_det_indices.append(int(high_indices[orig_high_idx]))
                out_tracker_ids.append(track.tracker_id)

            unmatched_high = [unmatched_high_list[i] for i in remaining_uh]

        if len(unmatched_uc_indices) > 0:
            remove_ids = {id(unconfirmed_tracks[i]) for i in unmatched_uc_indices}
            self.tracks = [t for t in self.tracks if id(t) not in remove_ids]

        self._spawn_new_tracks(
            detection_boxes,
            confidences,
            unmatched_high,
            high_indices,
            out_det_indices,
            out_tracker_ids,
            is_first_frame=(self.frame_id == 1),
        )

        tracklet_ids_before_pruning = {int(track.tracker_id) for track in self.tracks if track.tracker_id >= 0}
        self.tracks = _get_alive_tracklets(
            tracklets=self.tracks,
            maximum_frames_without_update=self.maximum_frames_without_update,
            minimum_consecutive_frames=self.minimum_consecutive_frames,
            maximum_time_without_update=_budget,
        )
        tracklet_ids_after_pruning = {int(track.tracker_id) for track in self.tracks if track.tracker_id >= 0}
        terminated_tracklet_ids = sorted(tracklet_ids_before_pruning - tracklet_ids_after_pruning)

        if not out_det_indices:
            result = sv.Detections.empty()
            result.tracker_id = np.array([], dtype=int)
            self._store_previous_mask_inputs(
                frame=current_frame,
                detections=result,
                removed_tracklet_ids=terminated_tracklet_ids,
            )
            return result

        idx = np.array(out_det_indices)
        result = cast(sv.Detections, detections[idx])
        result.tracker_id = np.array(out_tracker_ids, dtype=int)
        self._store_previous_mask_inputs(
            frame=current_frame,
            detections=result,
            removed_tracklet_ids=terminated_tracklet_ids,
        )
        return result
