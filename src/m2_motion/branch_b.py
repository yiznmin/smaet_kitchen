"""
M2 分支 B:小/關鍵目標專線(老鼠等小而關鍵的目標)。

設計重點(對應 docs/M2_雙分支重構設計.md §3–§4):
  - 全域 frame-diff(分支 A)看「整張畫面變動量」,小目標佔比太低會被漏掉。
  - 分支 B 改用「背景相減(MOG2)」:對靜止背景上的小前景最敏感,
    且 MOG2 的前景遮罩天生就是「哪裡動了」的定位(region proposal)。
  - 連通元件分析切出「每個 blob 各自的 bbox + 面積」(取代分支 A 的單一合併 bbox)。
  - 依面積分流:面積 ≥ large_blob_area 視為大事件(交給分支 A),小 blob 才走本線。
  - 空間排除:砍掉緊鄰大移動區(人)的小斑塊,孤立的才保留(寧鬆,避免誤砍貼腳邊老鼠)。
  - 時序持續性(方法 C):只確認「連續出現且連貫移動」的斑塊,壓掉閃爍的人碎塊/反光。
  - 在 blob 座標「原生解析度裁切放大」,把小目標放大後交給分類器確認身分。

本檔純 OpenCV/CPU,不含任何模型(分類器在 classifier.py)。
"""
import cv2
import numpy as np


class SmallTargetDetector:
    def __init__(self, min_blob_area=8, large_blob_area=2000,
                 exclusion_margin=24, crop_size=256, max_crops_per_frame=32,
                 mog2_history=500, mog2_var_threshold=16,
                 detect_shadows=False, warmup_frames=30,
                 persistence_frames=3, match_max_dist=60, track_max_miss=2,
                 density_radius=120, density_max_neighbors=3):
        self.min_blob_area = min_blob_area
        self.large_blob_area = large_blob_area
        self.exclusion_margin = exclusion_margin
        self.density_radius = density_radius
        self.density_max_neighbors = density_max_neighbors
        self.crop_size = int(crop_size)
        self.max_crops_per_frame = max_crops_per_frame
        self.warmup_frames = warmup_frames
        self.persistence_frames = persistence_frames
        self.match_max_dist = match_max_dist
        self.track_max_miss = track_max_miss
        self._n = 0
        self._high_load = 0          # 累計「小斑塊數超過警示門檻」的幀數(只記錄,不丟棄)
        self._tracks = []            # 時序追蹤:每筆 {cx, cy, hits, miss}
        self._bg = cv2.createBackgroundSubtractorMOG2(
            history=mog2_history, varThreshold=mog2_var_threshold,
            detectShadows=detect_shadows)
        self._kernel = np.ones((3, 3), np.uint8)

    def process(self, frame_bgr, return_mask=False):
        """回傳 dict:
          - crops:小目標清單,每筆含 bbox(裁切窗)、blob_bbox(原 blob 框)、area、crop(影像)
          - warming:背景模型是否仍在暖機(暖機期不輸出 crop)
          - n_crops:本幀輸出的 crop 數
          - mask(可選):MOG2 前景遮罩,供視覺化
        """
        self._n += 1
        fg = self._bg.apply(frame_bgr)
        # detectShadows=False 時 fg 已是 0/255;保險再二值化 + 開運算去點狀雜訊
        _, fg = cv2.threshold(fg, 127, 255, cv2.THRESH_BINARY)
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, self._kernel)

        crops = []
        n_excluded = 0
        n_dense_excluded = 0
        n_pending = 0
        large_boxes = []             # 大斑塊(人/大事件)bbox,供排除與視覺化
        dense_excluded_boxes = []    # 被密度排除(判為人群聚)的 bbox,供視覺化佐證
        warming = self._n <= self.warmup_frames
        if not warming:
            n_labels, _, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
            # 先分兩堆:大斑塊(人/大事件,作排除依據)與小斑塊候選
            cand = []          # (area, x, y, w, h);label 0 為背景,從 1 起算
            for i in range(1, n_labels):
                area = int(stats[i, cv2.CC_STAT_AREA])
                x = int(stats[i, cv2.CC_STAT_LEFT]); y = int(stats[i, cv2.CC_STAT_TOP])
                w = int(stats[i, cv2.CC_STAT_WIDTH]); h = int(stats[i, cv2.CC_STAT_HEIGHT])
                if area >= self.large_blob_area:
                    large_boxes.append([x, y, x + w, y + h])  # 大事件 → 分支 A,且當人區
                elif area >= self.min_blob_area:
                    cand.append((area, x, y, w, h))            # 小目標候選
                # area < min_blob_area → 雜訊,丟棄

            # 空間排除:砍掉「緊鄰大移動區(人)」的小斑塊(人=單一大斑塊時)
            kept = []
            for area, x, y, w, h in cand:
                if self._near_any([x, y, x + w, y + h], large_boxes):
                    n_excluded += 1
                else:
                    kept.append((area, x, y, w, h))

            # 密度排除:人大動作時散成「一堆疊在一起的小框」→ 密集群聚=人(砍),孤立=老鼠候選(留)
            kept, dense_excl = self._isolate(kept)
            n_dense_excluded = len(dense_excl)
            dense_excluded_boxes = [[x, y, x + w, y + h] for (_, x, y, w, h) in dense_excl]

            kept.sort(reverse=True)  # 面積大→小(僅為輸出順序穩定;不截斷)

            # 時序持續性(方法 C):只確認「連續出現且連貫移動」的斑塊,
            # 壓掉閃爍的人碎塊/反光;真老鼠連貫移動會被保留。
            cents = [(x + w // 2, y + h // 2) for (_, x, y, w, h) in kept]
            confirmed = self._update_tracks(cents)
            n_pending = confirmed.count(False)
            kept = [k for k, ok in zip(kept, confirmed) if ok]

            # 準確優先:不丟棄已確認的。偏多時只記錄(供調參),全部交分類器判。
            if self.max_crops_per_frame and len(kept) > self.max_crops_per_frame:
                self._high_load += 1

            H, W = frame_bgr.shape[:2]
            for area, x, y, w, h in kept:
                crop, box = self._extract_crop(frame_bgr, x, y, w, h, W, H)
                crops.append(dict(bbox=box, blob_bbox=[x, y, x + w, y + h],
                                  area=area, crop=crop))

        out = dict(crops=crops, warming=warming, n_crops=len(crops),
                   n_excluded=n_excluded, n_dense_excluded=n_dense_excluded,
                   n_pending=n_pending, large_boxes=large_boxes,
                   dense_excluded_boxes=dense_excluded_boxes)
        if return_mask:
            out["mask"] = fg
        return out

    def _isolate(self, blobs):
        """砍掉密集群聚的斑塊(人大動作 → 一堆小框疊一起),保留孤立斑塊(老鼠候選)。
        每個斑塊數半徑內的鄰居數;鄰居 ≥ density_max_neighbors 視為群聚 → 排除。
        總數本來就少時全留(孤立的老鼠永遠保得住,符合『寧可多留、別誤砍』)。
        回傳 (kept, excluded):excluded 為被判為群聚(人)的斑塊,供視覺化佐證。"""
        n = len(blobs)
        if n <= self.density_max_neighbors:
            return blobs, []
        cents = [(x + w // 2, y + h // 2) for (_, x, y, w, h) in blobs]
        r2 = self.density_radius * self.density_radius
        kept, excluded = [], []
        for i, (cx, cy) in enumerate(cents):
            neighbors = 0
            for j, (ox, oy) in enumerate(cents):
                if i == j:
                    continue
                if (cx - ox) ** 2 + (cy - oy) ** 2 <= r2:
                    neighbors += 1
                    if neighbors >= self.density_max_neighbors:
                        break
            if neighbors < self.density_max_neighbors:
                kept.append(blobs[i])        # 孤立 → 保留(像老鼠)
            else:
                excluded.append(blobs[i])    # 群聚 → 排除(人)
        return kept, excluded

    def _update_tracks(self, cents):
        """以質心最近鄰更新軌跡;回傳對齊 cents 的 confirmed 布林清單。
        軌跡累積出現 >= persistence_frames 才算確認(壓掉只閃一兩幀的雜訊)。
        persistence_frames <= 1 時等於關閉時序(立即確認)。"""
        gate2 = self.match_max_dist * self.match_max_dist
        for t in self._tracks:
            t["matched"] = False
        confirmed = [False] * len(cents)
        for idx, (cx, cy) in enumerate(cents):
            best, best_d2 = None, gate2 + 1
            for t in self._tracks:
                if t["matched"]:
                    continue
                d2 = (cx - t["cx"]) ** 2 + (cy - t["cy"]) ** 2
                if d2 < best_d2:
                    best, best_d2 = t, d2
            if best is not None and best_d2 <= gate2:
                best["cx"], best["cy"] = cx, cy   # 對到既有軌跡 → 更新位置、累積出現
                best["hits"] += 1
                best["miss"] = 0
                best["matched"] = True
                if best["hits"] >= self.persistence_frames:
                    confirmed[idx] = True
            else:
                # 對不到 → 新軌跡(剛出現,尚未連續,先不確認)
                self._tracks.append({"cx": cx, "cy": cy, "hits": 1,
                                     "miss": 0, "matched": True})
                if self.persistence_frames <= 1:
                    confirmed[idx] = True
        # 本幀沒對到的軌跡累積 miss,超過容忍(短暫遮擋)即移除
        for t in self._tracks:
            if not t["matched"]:
                t["miss"] += 1
        self._tracks = [t for t in self._tracks if t["miss"] <= self.track_max_miss]
        return confirmed

    def _near_any(self, box, large_boxes):
        """小斑塊 box 是否落在「任一大斑塊(人)bbox + margin」範圍內(=屬於人的碎塊)。"""
        m = self.exclusion_margin
        bx1, by1, bx2, by2 = box
        for lx1, ly1, lx2, ly2 in large_boxes:
            # 大框向外擴張 margin 後,與小框是否重疊
            if bx1 <= lx2 + m and bx2 >= lx1 - m and by1 <= ly2 + m and by2 >= ly1 - m:
                return True
        return False

    def _extract_crop(self, frame, x, y, w, h, W, H):
        """以 blob 中心取固定邊長視窗(原生解析度,放大小目標),邊界夾緊不出界。"""
        cx, cy = x + w // 2, y + h // 2
        half = self.crop_size // 2
        if W >= self.crop_size:
            x1 = max(0, min(cx - half, W - self.crop_size))
        else:
            x1 = 0
        if H >= self.crop_size:
            y1 = max(0, min(cy - half, H - self.crop_size))
        else:
            y1 = 0
        x2 = min(W, x1 + self.crop_size)
        y2 = min(H, y1 + self.crop_size)
        return frame[y1:y2, x1:x2].copy(), [x1, y1, x2, y2]

    @property
    def high_load_frames(self):
        """累計「小斑塊數超過警示門檻」的幀數(不丟棄,僅供調參參考:若常偏高代表
        前段過濾太鬆或場景雜訊大,可調 min_blob_area / exclusion_margin)。"""
        return self._high_load
