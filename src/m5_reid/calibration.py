"""跨鏡頭地面平面校正(homography)—— 把影像腳點映射到共同的世界座標。

為什麼需要:第五輪實測顯示,「重疊鏡頭裡有人」不等於「就是這一個人」。
K 位廚師同時在場時,那條證據只說得出「是 K 個之中的一個」→ 誤併 6.8%。
要分辨是誰,必須比對**世界座標**而不是「有沒有人」。

設計上刻意存**點對應**而不是 3×3 矩陣:
  · config 人看得懂、可稽核(矩陣看不出對錯)
  · **標定殘差可以自動算**,而殘差就是 GroundPlaneLR 需要的 σ
    → 不必再猜一個參數,系統自己量得出來

這一點很重要:第六輪 R8 證明證據上限 = log(A / 2πσ²),σ 直接決定這條證據
有沒有用(σ≥0.8m 時上限低於門檻,完全無用)。讓系統自己量 σ,
比讓人填一個數字可靠得多。
"""
import numpy as np


def _max_collinear(pts, tol=1e-6):
    """回傳「最多有幾個點落在同一條直線上」。全部共線時等於點數。"""
    n = len(pts)
    best = 2 if n >= 2 else n
    for i in range(n):
        for j in range(i + 1, n):
            d = pts[j] - pts[i]
            norm = np.hypot(*d)
            if norm < tol:
                continue
            cnt = sum(1 for k in range(n)
                      if abs(np.cross(d, pts[k] - pts[i])) / norm < tol)
            best = max(best, cnt)
    return best


class Homography:
    """單台鏡頭的地面校正。

    image_points: [[u, v], ...]  地面上已知點在畫面中的像素座標(≥4 個)
    world_points: [[x, y], ...]  同樣那些點在廚房平面圖上的座標(公尺)
    """

    def __init__(self, image_points, world_points, camera_id=None):
        self.camera_id = camera_id
        self.img = np.asarray(image_points, dtype=np.float64).reshape(-1, 2)
        self.wld = np.asarray(world_points, dtype=np.float64).reshape(-1, 2)
        if len(self.img) != len(self.wld):
            raise ValueError(f"{camera_id}: 影像點與世界點數量不一致")
        if len(self.img) < 4:
            raise ValueError(f"{camera_id}: homography 至少需要 4 組點對應,"
                             f"目前只有 {len(self.img)}")
        # ⚠ 退化的標定會被 cv2 靜默接受(例如四點共線時仍算得出一個矩陣),
        #   然後在真實資料上給出荒謬的世界座標。明確擋掉,不讓它變成靜默失效。
        for name, pts in (("影像", self.img), ("世界", self.wld)):
            if _max_collinear(pts) >= len(pts) - 0.5:
                raise ValueError(
                    f"{camera_id}: {name}點全部共線 —— homography 需要 4 個"
                    "**不共線**的點(挑地面上散開的四角,不要全排在一條線上)")
        import cv2
        H, mask = cv2.findHomography(self.img, self.wld, method=cv2.RANSAC,
                                     ransacReprojThreshold=0.5)
        if H is None:
            raise ValueError(f"{camera_id}: 無法求出 homography —— "
                             "點是否共線?至少要 4 個不共線的點")
        self.H = H
        self.inliers = int(mask.sum()) if mask is not None else len(self.img)
        self._residuals = self._compute_residuals()

    def _compute_residuals(self):
        """每個標定點的重投影誤差(公尺)。"""
        pred = self.project(self.img)
        return np.linalg.norm(pred - self.wld, axis=1)

    def project(self, image_xy):
        """影像座標 → 世界座標(公尺)。接受 (2,) 或 (N,2)。"""
        pts = np.asarray(image_xy, dtype=np.float64).reshape(-1, 2)
        hom = np.hstack([pts, np.ones((len(pts), 1))])
        out = (self.H @ hom.T).T
        w = out[:, 2:3]
        w = np.where(np.abs(w) < 1e-12, 1e-12, w)      # 避免除以零(點在消失線上)
        xy = out[:, :2] / w
        return xy[0] if np.asarray(image_xy).ndim == 1 else xy

    def foot_to_world(self, bbox):
        """人的腳點(框底中點)→ 世界座標。bbox = (x1,y1,x2,y2)。

        ⚠ 用腳點是因為人站在地面上,腳點才落在校正的那個平面。
          用中心點會隨身高與距離系統性偏移。
        ⚠ 腳被工作台擋住時 bbox 底邊不是真的腳,世界座標會嚴重偏掉 ——
          這個誤差目前併入 σ,沒有單獨偵測。
        """
        if bbox is None:
            return None
        x1, y1, x2, y2 = bbox
        return tuple(float(v) for v in self.project([(x1 + x2) / 2.0, float(y2)]))

    @property
    def sigma_m(self):
        """標定殘差(公尺)= GroundPlaneLR 該用的 σ。

        取 RMS 而非平均:GroundPlaneLR 的高斯模型吃的是標準差。
        """
        return float(np.sqrt((self._residuals ** 2).mean()))

    def report(self):
        r = self._residuals
        return {"camera_id": self.camera_id, "n_points": len(self.img),
                "inliers": self.inliers,
                "rms_m": round(self.sigma_m, 4),
                "max_m": round(float(r.max()), 4),
                "median_m": round(float(np.median(r)), 4)}

    def describe(self):
        rep = self.report()
        return (f"Homography({self.camera_id}: {rep['n_points']} 點 / "
                f"{rep['inliers']} inlier, 殘差 RMS {rep['rms_m']:.3f}m "
                f"最大 {rep['max_m']:.3f}m)")


def load_homographies(cameras_cfg):
    """從 config 的 `cameras` 區段建各鏡頭的 homography。

    cameras:
      cam1:
        homography:
          image_points: [[120, 700], [1150, 690], [980, 400], [300, 405]]
          world_points: [[0.0, 0.0], [6.0, 0.0], [6.0, 4.0], [0.0, 4.0]]

    沒填 homography 的鏡頭就不在回傳的 dict 裡 → 那些鏡頭的 world_xy 是 None,
    重疊路徑退回常數證據(已知過度自信,見第六輪 R8)。
    """
    out = {}
    for cam, cfg in (cameras_cfg or {}).items():
        h = (cfg or {}).get("homography")
        if not h:
            continue
        out[cam] = Homography(h["image_points"], h["world_points"], camera_id=cam)
    return out


def suggest_sigma(homographies, floor_m=0.05):
    """由各鏡頭的殘差推薦 GroundPlaneLR 的 σ。

    取最差的那一台 —— 一對鏡頭的比對精度由較差的那台決定。
    floor 是為了避免標定點太少時殘差假性偏低(過度自信)。
    """
    if not homographies:
        return None
    return max(max(h.sigma_m for h in homographies.values()), floor_m)
