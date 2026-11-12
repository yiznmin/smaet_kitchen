"""M4 外觀關聯:把外觀特徵接進追蹤的關聯成本(single_person 設定使用)。

兩人交錯時框本來就重疊,只看位置分不開,因此加入外觀證據:

    相似度 = (1 − w) × 框重疊 + w × 外觀相似度
    外觀相似度 = 0.5 × (1 + cosine)        ← 映射到 [0, 1],與 IoU 同尺度

設計:
1. 只用在第一輪(高信心)關聯;低信心偵測的裁圖品質差,外觀不可信。
2. track 特徵用 EMA 更新,動量 0.9。
3. w = 0 時完全不呼叫 embedder,結果與不用外觀相同。

特徵是對「這一幀的高信心偵測」裁圖計算;track 存的是自己歷史特徵的 EMA。
"""
import numpy as np

from m4_track.sparse_dcm import DCMMcByteTracker, _fuse_score


def build_embedder(name):
    """把設定檔裡的字串轉成 embedder 物件。

    不含 `none`:外觀權重 0 時不建 embedder。dinov2 為 Apache-2.0,可商用。
    """
    if name == "dinov2":
        from m5_reid.dino_embedder import DINOv2Embedder
        return DINOv2Embedder()
    raise ValueError(f"未知的外觀模型:{name}(可用:dinov2)")


def _to_unit(v):
    """L2 正規化。零向量原樣回傳(cosine 會是 0 → 外觀相似度 0.5,中性)。"""
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


class AppearanceMcByteTracker(DCMMcByteTracker):
    """C-BIoU / McByte 的關聯成本加上外觀項。

    參數:
      embedder        有 `.extract(crop_bgr) -> np.ndarray` 的物件;w > 0 時必須給
      appearance_w    w ∈ [0, 1];0 = 完全關閉(逐位等同基準)
      ema_momentum    track 特徵的 EMA 動量,預設 0.9
    """

    def __init__(self, *args, embedder=None, appearance_w=0.0, ema_momentum=0.9, **kwargs):
        super().__init__(*args, **kwargs)
        # 設定檔只能寫字串 → 這裡轉成物件;w=0 時不建(不碰 GPU、不改行為)
        if isinstance(embedder, str):
            embedder = build_embedder(embedder) if float(appearance_w) > 0 else None
        self.embedder = embedder
        self.appearance_w = float(appearance_w)
        self.ema_momentum = float(ema_momentum)
        if self.appearance_w > 0 and embedder is None:
            raise ValueError("appearance_w > 0 但沒有給 embedder")
        self._feat = {}              # id(track) -> 特徵向量(EMA)
        self._frame = None           # 本幀畫面(BGR),由 update 設定
        self._det_feat = None        # 本幀高信心偵測的特徵,shape (n_high, d)
        self._n_extract = 0          # 診斷:總共抽了幾次特徵

    # ── 取得本幀畫面(McByte 的遮罩路徑已經在用 frame,這裡只是記下來) ──
    def update(self, detections, frame=None, timestamp=None):
        self._frame = frame
        self._det_feat = None
        return super().update(detections, frame=frame, timestamp=timestamp)

    def _crop_feature(self, box):
        """裁圖 → 特徵。框超出畫面時夾住;裁不出東西時回 None(視為中性)。

        ⚠ 色彩通道:`frame` 依 `tracker.py::update` 的約定是 RGB,但 embedder 的 `extract`
          預期收到 BGR(`dino_embedder.py` 內部再轉 RGB)。這裡先轉回 BGR 再交給 embedder,
          否則紅藍通道會對調。
        """
        h, w = self._frame.shape[:2]
        x1, y1, x2, y2 = (int(max(0, box[0])), int(max(0, box[1])),
                          int(min(w, box[2])), int(min(h, box[3])))
        if x2 <= x1 or y2 <= y1:
            return None
        self._n_extract += 1
        crop_bgr = self._frame[y1:y2, x1:x2][:, :, ::-1]      # RGB(約定)→ BGR(embedder 預期)
        return _to_unit(np.asarray(self.embedder.extract(crop_bgr), dtype=np.float32))

    def _sim_high(self, sub_tracklets, det_idx, high_boxes, high_scores, predicted_state_boxes):
        # w = 0:完全不用外觀,直接走基準,結果與不加外觀的追蹤器逐位相同
        if self.appearance_w <= 0 or self._frame is None:
            return super()._sim_high(sub_tracklets, det_idx, high_boxes, high_scores,
                                     predicted_state_boxes)

        raw = self._get_iou_matrix(
            sub_tracklets, high_boxes[det_idx] if len(det_idx) else np.empty((0, 4)),
            predicted_state_boxes)
        fused = _fuse_score(self.iou.normalize_for_fusion(raw.copy()), high_scores[det_idx])
        if not len(sub_tracklets) or not len(det_idx):
            return fused, raw

        # 本幀所有高信心偵測的特徵只抽一次(同一個偵測可能出現在多個分層裡)
        if self._det_feat is None:
            self._det_feat = {}
        for j in det_idx:
            if j not in self._det_feat:
                self._det_feat[j] = self._crop_feature(high_boxes[j])

        app = np.full(fused.shape, 0.5, dtype=np.float32)   # 預設中性
        for i, tr in enumerate(sub_tracklets):
            f_t = self._feat.get(id(tr))
            if f_t is None:
                continue
            for k, j in enumerate(det_idx):
                f_d = self._det_feat.get(j)
                if f_d is None:
                    continue
                app[i, k] = 0.5 * (1.0 + float(np.dot(f_t, f_d)))
        w = self.appearance_w
        return (1 - w) * fused + w * app, raw

    def _on_matched(self, track, score, was_lost, det_idx=None, stage=None):
        """配對成功 → 用**配到的那個偵測**的特徵更新 track 的記憶(EMA)。

        ⚠ 只在第一輪(high)與 unconfirmed 更新。低信心偵測的裁圖品質差,
          拿它污染 track 的記憶會讓後續配對更糟 —— 與「外觀只用在第一輪」同一個理由。
        """
        super()._on_matched(track, score, was_lost, det_idx=det_idx, stage=stage)
        if self.appearance_w <= 0 or self._det_feat is None or det_idx is None:
            return
        if stage == "low":
            return
        f_d = self._det_feat.get(det_idx)
        if f_d is None:
            return
        k = id(track)
        prev = self._feat.get(k)
        m = self.ema_momentum
        self._feat[k] = _to_unit(m * prev + (1 - m) * f_d) if prev is not None else f_d

    def stats(self):
        return dict(n_extract=self._n_extract, n_tracks_with_feature=len(self._feat),
                    appearance_w=self.appearance_w, ema_momentum=self.ema_momentum)
