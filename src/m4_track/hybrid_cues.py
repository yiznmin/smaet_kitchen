"""Hybrid-SORT 的弱線索接到 McByte(不開遮罩)上 —— M4 修法第 3 輪(2026-09-15)。

## 來源與授權

照抄 Hybrid-SORT 官方實作(https://github.com/ymzis69/HybridSORT,MIT License,Copyright (c) 2021 Yifu Zhang):
- `hmiou`:`trackers/hybird_sort_tracker/association.py`(高度調整 IoU = IoU × 兩框垂直方向的重疊比例)
- 信心分數卡爾曼:`trackers/byte_tracker/kalman_filter_score.py` 的 `KalmanFilter_score`
- TCM 成本:`trackers/byte_tracker/matching.py` 的 `add_score_kalman`、`add_score_kalman_byte_step`
- 分數狀態更新時機:`trackers/byte_tracker/byte_tracker_score.py` 的 `STrack`(官方 README 的「ByteTrack + TCM」)

## 與官方一致的部分

- 第 1 階段:成本 += |clip(track 卡爾曼預測分數, track_thresh, 1) − 偵測分數| × 權重
- 第 2 階段(低分):成本 += |clip(score − (pre_score − score), 0.1, track_thresh) − 偵測分數| × 權重
- 分數卡爾曼:狀態 [score, vscore];初始標準差 [1e-2, 1e-5];預測雜訊 [1e-2, 1e-5];量測雜訊 1e-1
- **官方的時序照抄**:配對到時,卡爾曼先用**舊的** score 更新,再把 score 換成新偵測分數
  (`update`:pre_score = 舊 score;`re_activate`(lost 找回):pre_score = 新 score)
- 只對 strack_pool(confirmed + lost)預測分數;unconfirmed 不預測

## 刻意與官方不同的部分

- 基底是 McByte(不開遮罩)的配對:TCM 從**相似度**扣掉(McByte 是最大化相似度),門檻判斷在扣掉之後 ——
  與官方在距離上加 TCM 後才比門檻等價
- `track_thresh` 對映為 McByte 的 `high_conf_det_threshold`
- 權重 0 且不用 HMIoU 時必須與 `rf_mcbyte_nomask` 逐位相同 —— 由預先登記的硬性驗收檢查
"""
from __future__ import annotations

import numpy as np
import scipy.linalg

from m4_track.sparse_dcm import DCMMcByteTracker
from trackers.utils.iou import BaseIoU


class HMIoU(BaseIoU):
    """Hybrid-SORT `hmiou`(association.py)逐行移植。"""

    def _compute(self, boxes_1: np.ndarray, boxes_2: np.ndarray) -> np.ndarray:
        bboxes1 = np.asarray(boxes_1, dtype=np.float64)
        bboxes2 = np.asarray(boxes_2, dtype=np.float64)
        bboxes2 = np.expand_dims(bboxes2, 0)
        bboxes1 = np.expand_dims(bboxes1, 1)

        yy11 = np.maximum(bboxes1[..., 1], bboxes2[..., 1])
        yy12 = np.minimum(bboxes1[..., 3], bboxes2[..., 3])

        yy21 = np.minimum(bboxes1[..., 1], bboxes2[..., 1])
        yy22 = np.maximum(bboxes1[..., 3], bboxes2[..., 3])
        o = (yy12 - yy11) / (yy22 - yy21)

        xx1 = np.maximum(bboxes1[..., 0], bboxes2[..., 0])
        yy1 = np.maximum(bboxes1[..., 1], bboxes2[..., 1])
        xx2 = np.minimum(bboxes1[..., 2], bboxes2[..., 2])
        yy2 = np.minimum(bboxes1[..., 3], bboxes2[..., 3])
        w = np.maximum(0., xx2 - xx1)
        h = np.maximum(0., yy2 - yy1)
        wh = w * h
        o *= wh / ((bboxes1[..., 2] - bboxes1[..., 0]) * (bboxes1[..., 3] - bboxes1[..., 1])
                   + (bboxes2[..., 2] - bboxes2[..., 0]) * (bboxes2[..., 3] - bboxes2[..., 1]) - wh)
        return o


class ScoreKalman:
    """Hybrid-SORT `KalmanFilter_score`(單一 track 版本)逐行移植。"""

    def __init__(self):
        ndim, dt = 1, 1.
        self._motion_mat = np.eye(2 * ndim, 2 * ndim)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt
        self._update_mat = np.eye(ndim, 2 * ndim)

    def initiate(self, measurement):
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]
        std = [1e-2, 1e-5]
        covariance = np.diag(np.square(std))
        return mean, covariance

    def predict(self, mean, covariance):
        std_pos = [1e-2]
        std_vel = [1e-5]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))
        mean = np.dot(mean, self._motion_mat.T)
        covariance = np.linalg.multi_dot((self._motion_mat, covariance, self._motion_mat.T)) + motion_cov
        return mean, covariance

    def project(self, mean, covariance):
        std = [1e-1]
        innovation_cov = np.diag(np.square(std))
        mean = np.dot(self._update_mat, mean)
        covariance = np.linalg.multi_dot((self._update_mat, covariance, self._update_mat.T))
        return mean, covariance + innovation_cov

    def update(self, mean, covariance, measurement):
        projected_mean, projected_cov = self.project(mean, covariance)
        chol_factor, lower = scipy.linalg.cho_factor(projected_cov, lower=True, check_finite=False)
        kalman_gain = scipy.linalg.cho_solve(
            (chol_factor, lower), np.dot(covariance, self._update_mat.T).T, check_finite=False).T
        innovation = measurement - projected_mean
        new_mean = mean + np.dot(innovation, kalman_gain.T)
        new_covariance = covariance - np.linalg.multi_dot((kalman_gain, projected_cov, kalman_gain.T))
        return new_mean, new_covariance


class HybridMcByteTracker(DCMMcByteTracker):
    """McByte(不開遮罩)+ Hybrid-SORT 的 TCM(信心分數)與可選的 HMIoU。

    tcm_first_weight:第 1 階段 TCM 權重(官方 MOT17/MOT20 設定 1.0;README 的 ByteTrack 範例 0.6)
    tcm_byte_weight: 第 2 階段 TCM 權重(官方 1.0)
    use_hmiou:       以 HMIoU 取代 IoU(官方 MOT17/MOT20 設定 `asso = Height_Modulated_IoU`)
    分層層數固定為 1(不分層),沿用 DCMMcByteTracker 的 update。
    """

    _kf_score = ScoreKalman()

    def __init__(self, *args, tcm_first_weight: float = 0.0, tcm_byte_weight: float = 0.0,
                 use_hmiou: bool = False, **kwargs):
        if use_hmiou:
            if kwargs.get("iou") is not None:
                raise ValueError("use_hmiou 與自訂 iou 不可同時給")
            kwargs["iou"] = HMIoU()
        kwargs.setdefault("depth_levels_high", 1)
        kwargs.setdefault("depth_levels_low", 1)
        super().__init__(*args, **kwargs)
        self.tcm_first_weight = float(tcm_first_weight)
        self.tcm_byte_weight = float(tcm_byte_weight)
        self.use_hmiou = bool(use_hmiou)

    # ── 分數狀態存在 tracklet 物件上(不用 id() 當鍵,避免物件回收後 id 重複) ──
    @staticmethod
    def _init_score(track, score):
        track._hs_mean, track._hs_cov = HybridMcByteTracker._kf_score.initiate(np.asarray(score, dtype=float))
        track._hs_score = float(score)
        track._hs_pre_score = float(score)

    def _before_association(self, strack_pool):
        for t in strack_pool:
            if getattr(t, "_hs_mean", None) is not None:
                t._hs_mean, t._hs_cov = self._kf_score.predict(t._hs_mean, t._hs_cov)

    def _sim_high(self, sub_tracklets, det_idx, high_boxes, high_scores, predicted_state_boxes):
        fused, raw = super()._sim_high(sub_tracklets, det_idx, high_boxes, high_scores, predicted_state_boxes)
        if self.tcm_first_weight != 0 and fused.size:
            thr = self.high_conf_det_threshold
            ks = np.array([np.clip(t._hs_mean[0] if getattr(t, "_hs_mean", None) is not None else t._hs_score,
                                   thr, 1.0) for t in sub_tracklets])
            det = np.asarray(high_scores[det_idx], dtype=float)
            fused = fused - np.abs(ks[:, None] - det[None, :]) * self.tcm_first_weight
        return fused, raw

    def _sim_low(self, sub_tracklets, det_idx, low_boxes, low_scores, predicted_state_boxes):
        sim, raw = super()._sim_low(sub_tracklets, det_idx, low_boxes, low_scores, predicted_state_boxes)
        if self.tcm_byte_weight != 0 and sim.size:
            thr = self.high_conf_det_threshold
            ext = np.array([np.clip(t._hs_score - (t._hs_pre_score - t._hs_score), 0.1, thr) for t in sub_tracklets])
            det = np.asarray(low_scores[det_idx], dtype=float)
            sim = sim - np.abs(ext[:, None] - det[None, :]) * self.tcm_byte_weight
        return sim, raw

    def _on_matched(self, track, score, was_lost):
        if getattr(track, "_hs_mean", None) is None:
            self._init_score(track, score)
            return
        # 官方時序:卡爾曼先用「舊」score 更新,再換成新分數
        track._hs_mean, track._hs_cov = self._kf_score.update(
            track._hs_mean, track._hs_cov, np.asarray(track._hs_score, dtype=float))
        if was_lost:                              # 官方 re_activate
            track._hs_score = float(score)
            track._hs_pre_score = track._hs_score
        else:                                     # 官方 update
            track._hs_pre_score = track._hs_score
            track._hs_score = float(score)

    def _spawn_new_tracks(self, detection_boxes, confidences, unmatched_high_local, high_indices,
                          out_det_indices, out_tracker_ids, is_first_frame=False):
        n0 = len(self.tracks)
        super()._spawn_new_tracks(detection_boxes, confidences, unmatched_high_local, high_indices,
                                  out_det_indices, out_tracker_ids, is_first_frame=is_first_frame)
        spawned_scores = [float(confidences[int(high_indices[i])]) for i in unmatched_high_local
                          if float(confidences[int(high_indices[i])]) >= self.track_activation_threshold]
        new_tracks = self.tracks[n0:]
        if len(new_tracks) != len(spawned_scores):
            raise RuntimeError(f"新 track 數 {len(new_tracks)} 與分數數 {len(spawned_scores)} 不一致")
        for t, s in zip(new_tracks, spawned_scores):
            self._init_score(t, s)
