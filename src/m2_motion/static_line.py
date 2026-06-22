"""
M2 分支 B — 靜態線:偵測「落地後靜止」的目標(水漬、油漬、殘渣)。

為什麼需要獨立一條(對應 docs/M2_雙分支重構設計.md §3.3):
  水漬潑灑瞬間會動,但落地後靜止不動 → MOG2 適應性背景會把它當背景吸收、
  幾秒後就偵測不到;且時序持續性要求「連貫移動」,靜止水漬也過不了。
  故靜態線改用「慢速基準背景比對」:找「與平常場景不同、現在沒在動、又賴著不走」的塊。

判別:
  靜態新增 = (與慢速基準不同) AND (短期沒在動) AND (持續存在 N 幀)
  → 人(會動)被「沒在動」這條排除;一閃而過的東西被「持續存在」排除。

與動態線的差異:
  - 背景:慢速基準(accumulateWeighted)而非 MOG2;
  - 確認:持續「靜止存在」而非「連貫移動」;
  - 面積:大的靜止塊(大灘水漬)要保留,不像動態線把大塊丟給分支 A。

純 OpenCV/CPU,不含模型(多類別分類器共用 classifier.py)。
幾何/追蹤輔助與 branch_b.py 同邏輯,靜態線自含一份(動/靜兩線可獨立調校)。
"""
import cv2
import numpy as np


class StaticTargetDetector:
    def __init__(self, baseline_alpha=0.002, baseline_diff_thresh=30, motion_thresh=25,
                 blur_ksize=5, min_blob_area=60, max_blob_area=120000,
                 crop_size=256, warmup_frames=30, motion_exclusion_margin=61,
                 max_aspect_ratio=2.0,
                 persistence_frames=8, match_max_dist=40, track_max_miss=3,
                 density_radius=120, density_max_neighbors=4, floor_zone=None):
        self.baseline_alpha = baseline_alpha
        self.baseline_diff_thresh = baseline_diff_thresh
        self.motion_thresh = motion_thresh
        self.blur_ksize = blur_ksize if blur_ksize % 2 == 1 else blur_ksize + 1
        self.min_blob_area = min_blob_area
        self.max_blob_area = max_blob_area
        self.crop_size = int(crop_size)
        self.warmup_frames = warmup_frames
        self.max_aspect_ratio = max_aspect_ratio   # 高/寬 > 此值=瘦高(腿/人)→ 排除;扁平(水漬)保留
        self.persistence_frames = persistence_frames
        self.match_max_dist = match_max_dist
        self.track_max_miss = track_max_miss
        self.density_radius = density_radius
        self.density_max_neighbors = density_max_neighbors
        self._baseline = None        # float32 灰階慢速基準背景
        self._prev = None            # 前一幀灰階(算短期移動)
        self._tracks = []
        self._kernel = np.ones((3, 3), np.uint8)
        m = max(1, int(motion_exclusion_margin))     # 動作排除:把「正在動」的區域外擴此範圍後排除
        self._motion_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (m, m))
        # 樓地板 ROI:水漬只在地面 → 只在此多邊形內偵測(牆/檯面/上半部一律排除)。
        # None = 全畫面;否則為 [[x,y],...] 多邊形(每鏡頭各自設定,符合專案 zone 設計)。
        self.floor_zone = floor_zone
        self._floor_mask = None                       # 依首幀尺寸惰性建立
        self._n = 0

    def _gray(self, frame_bgr):
        g = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        return cv2.GaussianBlur(g, (self.blur_ksize, self.blur_ksize), 0)

    def process(self, frame_bgr, return_mask=False):
        """回傳 dict:crops、warming、n_crops、n_dense_excluded、(可選)mask。"""
        self._n += 1
        gray = self._gray(frame_bgr)

        if self._baseline is None:
            self._baseline = gray.astype(np.float32)
            self._prev = gray
            out = dict(crops=[], warming=True, n_crops=0, n_dense_excluded=0)
            if return_mask:
                out["mask"] = np.zeros_like(gray)
            return out

        # 慢速基準背景:水漬會被慢慢吸收 → 提供「偵測時間窗」(alpha 小=窗長)
        cv2.accumulateWeighted(gray.astype(np.float32), self._baseline, self.baseline_alpha)
        baseline_u8 = cv2.convertScaleAbs(self._baseline)

        base_fg = (cv2.absdiff(gray, baseline_u8) > self.baseline_diff_thresh).astype(np.uint8) * 255
        moving = (cv2.absdiff(gray, self._prev) > self.motion_thresh).astype(np.uint8) * 255
        self._prev = gray

        # 動作排除:把「正在動」的區域外擴 margin —— 人就算身體不動,旁邊的手在動,
        # 整個人會落在動作區內被排除;水漬周圍完全沒動作 → 保留。
        motion_zone = cv2.dilate(moving, self._motion_kernel)

        # 靜態新增 = 與基準不同 且 不在(也不靠近)任何動作區
        static_mask = cv2.bitwise_and(base_fg, cv2.bitwise_not(motion_zone))
        static_mask = cv2.morphologyEx(static_mask, cv2.MORPH_OPEN, self._kernel)
        static_mask = cv2.morphologyEx(static_mask, cv2.MORPH_CLOSE, self._kernel)

        # 樓地板 ROI:水漬只在地面 → 只保留地板範圍內的(牆/檯面/上半部排除)
        if self.floor_zone is not None:
            if self._floor_mask is None:
                self._floor_mask = np.zeros(gray.shape, np.uint8)
                cv2.fillPoly(self._floor_mask, [np.array(self.floor_zone, np.int32)], 255)
            static_mask = cv2.bitwise_and(static_mask, self._floor_mask)

        crops, n_dense = [], 0
        warming = self._n <= self.warmup_frames
        if not warming:
            n_labels, _, stats, _ = cv2.connectedComponentsWithStats(static_mask, connectivity=8)
            cand = []
            for i in range(1, n_labels):
                area = int(stats[i, cv2.CC_STAT_AREA])
                if area < self.min_blob_area or area > self.max_blob_area:
                    continue   # 太小=雜訊;太大=全域光線變化(整片),非水漬
                w = int(stats[i, cv2.CC_STAT_WIDTH]); h = int(stats[i, cv2.CC_STAT_HEIGHT])
                if h / max(w, 1) > self.max_aspect_ratio:
                    continue   # 瘦高(腿/人站立)→ 排除;水漬是扁平的(寬≥高)
                cand.append((area, int(stats[i, cv2.CC_STAT_LEFT]), int(stats[i, cv2.CC_STAT_TOP]), w, h))

            cand, excl = self._isolate(cand)          # 砍密集群聚(belt-and-suspenders)
            n_dense = len(excl)
            cents = [(x + w // 2, y + h // 2) for (_, x, y, w, h) in cand]
            confirmed = self._update_tracks(cents)    # 只確認「持續靜止存在」的塊
            cand = [c for c, ok in zip(cand, confirmed) if ok]

            H, W = frame_bgr.shape[:2]
            for area, x, y, w, h in cand:
                crop, box = self._extract_crop(frame_bgr, x, y, w, h, W, H)
                crops.append(dict(bbox=box, blob_bbox=[x, y, x + w, y + h], area=area, crop=crop))

        out = dict(crops=crops, warming=warming, n_crops=len(crops), n_dense_excluded=n_dense)
        if return_mask:
            out["mask"] = static_mask
        return out

    # ---- 幾何/追蹤輔助(與 branch_b.py 同邏輯,靜態線自含一份)----

    def _isolate(self, blobs):
        """砍掉密集群聚(人),保留孤立塊;回傳 (kept, excluded)。總數少時全留。"""
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
            (excluded if neighbors >= self.density_max_neighbors else kept).append(blobs[i])
        return kept, excluded

    def _update_tracks(self, cents):
        """質心最近鄰追蹤;持續存在 >= persistence_frames 才確認(靜止物匹配自身,位移≈0)。"""
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
                best["cx"], best["cy"], best["miss"], best["matched"] = cx, cy, 0, True
                best["hits"] += 1
                if best["hits"] >= self.persistence_frames:
                    confirmed[idx] = True
            else:
                self._tracks.append({"cx": cx, "cy": cy, "hits": 1, "miss": 0, "matched": True})
                if self.persistence_frames <= 1:
                    confirmed[idx] = True
        for t in self._tracks:
            if not t["matched"]:
                t["miss"] += 1
        self._tracks = [t for t in self._tracks if t["miss"] <= self.track_max_miss]
        return confirmed

    def _extract_crop(self, frame, x, y, w, h, W, H):
        """以 blob 中心取固定邊長視窗(原生解析度),邊界夾緊不出界。"""
        cx, cy = x + w // 2, y + h // 2
        half = self.crop_size // 2
        x1 = max(0, min(cx - half, W - self.crop_size)) if W >= self.crop_size else 0
        y1 = max(0, min(cy - half, H - self.crop_size)) if H >= self.crop_size else 0
        x2 = min(W, x1 + self.crop_size)
        y2 = min(H, y1 + self.crop_size)
        return frame[y1:y2, x1:x2].copy(), [x1, y1, x2, y2]
