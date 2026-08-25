"""M5 v2 空間+時序(Camera Link Model)核心:相機拓撲 + 轉場時間門 + zone 判定。

跨鏡頭關聯「同一人」的四線索,外觀只是其一:
  相機拓撲(哪台接哪台)、轉場時間(高斯窗)、移動方向(有向連結)、外觀相似度(輔助)。
物理約束:同一人不可能同時出現在兩個不重疊鏡頭;不能比離開更早抵達。

純函式/純邏輯,可單測;point_in_zone 用 cv2(zones.py 無此 helper)。
"""
import numpy as np


def point_in_zone(pt, polygon):
    """點是否在多邊形內(含邊界)。polygon = [[x,y],...]。"""
    import cv2
    poly = np.asarray(polygon, dtype=np.int32)
    return cv2.pointPolygonTest(poly, (float(pt[0]), float(pt[1])), False) >= 0


def foot_point(bbox):
    """人的「腳點」(框底中點)當地面位置,判所在 zone 較準。"""
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, float(y2))


def which_zone(bbox, zones):
    """回傳 bbox 腳點所在的 zone 名稱(zones = [{name, points}]),無則 None。"""
    pt = foot_point(bbox)
    for z in zones:
        if point_in_zone(pt, z["points"]):
            return z["name"]
    return None


def st_prob(dt, mean, std):
    """轉場時間高斯,normalize 到峰值 1(dt=mean 時為 1)。"""
    if std <= 0:
        return 1.0 if dt == mean else 0.0
    return float(np.exp(-0.5 * ((dt - mean) / std) ** 2))


_DEFAULT_FUSION = {
    # 融合模式:llr = v3 對數勝算比(建議);weighted_sum = v2 加權和(保留供消融對照)
    "mode": "llr",
    # ── v3(mode=llr)參數 ────────────────────────────────────────────
    "background_arrival_hz": 1.0 / 600.0,   # 「真正的新人」在單一鏡頭出現的速率
    "cost_false_merge_over_break": 5.0,     # 誤併比碎裂嚴重幾倍 → 換算成 LLR 門檻
    "transit_model": "loiter",              # loiter | gaussian
    "p_loiter": 0.15,                       # 中途停下來做事的比例
    "tau_loiter_s": 20.0,                   # 停留時間尺度
    "loiter_dist": "exp",                   # exp | lognormal(重尾)
    "loiter_log_sigma": 1.0,                # loiter_dist=lognormal 時的 logσ
    "appearance_profile": "dinov2",         # 外觀 LR 用哪組實測分布
    "appearance_clip": None,                # 外觀 LLR 上下限(None=不夾)
    "overlap_llr": 5.0,                     # 重疊鏡頭同時觀測的幾何證據強度(nats)
    # F1 未建模路徑(繞路)、F3 同鏡頭重關聯(M4 斷軌)。
    # ⚠ 兩者**預設關閉** —— 2026-08-25 依預先登記判準實測後的決定,不是尚未啟用。
    #   F1 幾乎無效(碎裂 25.2%→25.0%,在雜訊內);
    #   F3 降碎裂 4.3pp 但誤併率翻倍(4.8%→10.2%),依「誤併比碎裂嚴重 5 倍」的
    #   成本比,加權成本從 49.2 惡化到 71.9。根因:同鏡頭斷軌只用時間無法分辨
    #   「同一台鏡頭裡的哪一位廚師」,需要位置證據(bbox)才收得緊。
    #   詳見 docs/M5_模擬預先登記_20260825.md 與 results/m5_reid/sim_after_fixes.txt。
    "unknown_path": {"enabled": False, "median_multiplier": 2.0,
                     "log_sigma": 0.8, "logprior": -2.0},
    # F3 於 2026-08-25 接上位置證據後改為**預設開啟**:含位置時誤併率回到基準
    # (9.5%→4.5%)且保留碎裂改善(24.4%→19.8%),加權成本 46.9→42.3。
    "same_camera": {"enabled": True, "tau_break_s": 2.0, "max_gap_s": 15.0},
    # 方向證據(出入口 zone)。CLM 四線索裡最後一個;與 Δt 正交,見 DirectionLR。
    "direction": {"enabled": False, "q": 0.85, "n_zones": 3, "clip": 6.0},
    # 跨鏡頭地面校正(homography)。開啟後取代 overlap_llr 常數 —— 它能回答
    # 「重疊視野裡的是哪一個人」,而 overlap_llr 只能說「那裡有人」。
    "ground_plane": {"enabled": False, "sigma_m": 0.4, "area_m2": 30.0, "clip": 8.0},
    # 位置證據(僅同鏡頭適用)。F3 只用時間時誤併率翻倍,位置是收緊它的關鍵。
    "position": {"enabled": True, "speed_bh_per_s": 0.5, "noise_bh": 0.3,
                 "frame_span_bh": 6.0, "clip": 8.0},
    "max_z": 6.0,                           # 轉場分布的遠尾截斷(省算,非決策門)
    # ── v2(mode=weighted_sum)參數 ──────────────────────────────────
    "w_st": 0.7, "w_app": 0.3, "k_sigma": 2.0, "combined_threshold": 0.35,
    # ── 共用 ─────────────────────────────────────────────────────────
    "overlap_window_s": 0.5,
}


class CameraTopology:
    def __init__(self, links, overlapping, fusion=None, cameras=None, clock=None):
        self.links = {(l["from"], l["to"]): (float(l["mean_s"]), float(l["std_s"])) for l in links}
        # 每條連結對應的出入口 zone(方向證據用)。沒填就是 None → 不提供證據。
        self.link_zones = {(l["from"], l["to"]): (l.get("exit_zone"), l.get("entry_zone"))
                           for l in links}
        self.overlapping = set(frozenset(p) for p in overlapping)
        self.fusion = {**_DEFAULT_FUSION, **(fusion or {})}
        # 每台相機的時鐘偏移(秒)。整套 CLM 建立在 t_exit 與 t_enter 同一時基上;
        # NVR 若無 NTP,鏡頭間漂移 1~2 秒是常態,而 σ=1.5s 時 2 秒漂移 = 1.33σ,
        # 足以把幾乎所有真實轉場推出門外 —— 症狀是「chef_id 一直開新的」,
        # 看起來像模型爛,實際是時鐘問題。量化見 analyze_gate_capacity.py §4。
        self.cameras = dict(cameras or {})
        self.clock_offset = {c: float(v.get("clock_offset_s", 0.0) or 0.0)
                             for c, v in self.cameras.items()}
        self.clock = {"max_skew_s": 0.2, **(clock or {})}

        self._build_evidence()

    def offset(self, camera_id):
        """該相機時鐘相對基準的偏移(秒)。校正後時間 = 原始時間 − offset。"""
        return self.clock_offset.get(camera_id, 0.0)

    def corrected(self, camera_id, t_sec):
        return t_sec - self.offset(camera_id)

    def all_cameras(self):
        """config 中出現過的所有 camera_id(links + overlapping + cameras)。"""
        seen = set(self.cameras)
        for a, b in self.links:
            seen |= {a, b}
        for pair in self.overlapping:
            seen |= set(pair)
        return seen

    def _build_evidence(self):
        """建 v3 需要的轉場模型與外觀 LR。mode=weighted_sum 時不會被用到。"""
        from m5_reid.evidence import (AppearanceLR, DirectionLR, GroundPlaneLR, PositionLR,
                                      SameCameraTransit, UnknownPathTransit,
                                      decision_threshold, make_transit)
        f = self.fusion
        kw = {}
        if f["transit_model"] == "loiter":
            kw = dict(p_loiter=f["p_loiter"], tau_loiter_s=f["tau_loiter_s"],
                      loiter_dist=f["loiter_dist"], loiter_log_sigma=f["loiter_log_sigma"])
        elif f["transit_model"] == "gaussian":
            kw = dict(max_z=f["max_z"])
        self.transits = {k: make_transit(mu, sd, kind=f["transit_model"], **kw)
                         for k, (mu, sd) in self.links.items()}

        # F1 未建模路徑(繞路):中位數取「典型直達時間 × 倍率」
        up = f.get("unknown_path") or {}
        if up.get("enabled"):
            typical = float(np.median([mu for mu, _ in self.links.values()])) if self.links else 4.0
            self.unknown_path = UnknownPathTransit(
                median_s=typical * float(up.get("median_multiplier", 2.0)),
                log_sigma=float(up.get("log_sigma", 0.8)),
                logprior=float(up.get("logprior", -2.0)))
        else:
            self.unknown_path = None

        # F3 同鏡頭重關聯(M4 斷軌)
        sc = f.get("same_camera") or {}
        self.same_cam = SameCameraTransit(
            tau_break_s=float(sc.get("tau_break_s", 2.0)),
            max_gap_s=float(sc.get("max_gap_s", 15.0))) if sc.get("enabled") else None
        d = f.get("direction") or {}
        self.dir_lr = DirectionLR(q=float(d.get("q", 0.85)),
                                  n_zones=int(d.get("n_zones", 3)),
                                  clip=float(d.get("clip", 6.0))) if d.get("enabled") else None
        g = f.get("ground_plane") or {}
        self.ground_lr = GroundPlaneLR(sigma_m=float(g.get("sigma_m", 0.4)),
                                       area_m2=float(g.get("area_m2", 30.0)),
                                       clip=float(g.get("clip", 8.0))) if g.get("enabled") else None
        pos = f.get("position") or {}
        self.pos_lr = PositionLR(
            speed_bh_per_s=float(pos.get("speed_bh_per_s", 0.5)),
            noise_bh=float(pos.get("noise_bh", 0.3)),
            frame_span_bh=float(pos.get("frame_span_bh", 6.0)),
            clip=float(pos.get("clip", 8.0))) if pos.get("enabled", True) else None
        self.app_lr = AppearanceLR.measured(f["appearance_profile"], clip=f["appearance_clip"])
        self.log_lambda_bg = np.log(float(f["background_arrival_hz"]))
        self.llr_threshold = decision_threshold(f["cost_false_merge_over_break"])

    def set_transit(self, cam_from, cam_to, model):
        """用實測資料校準後,把某條連結的轉場模型換掉(見 scripts/calibrate_topology.py)。"""
        self.transits[(cam_from, cam_to)] = model

    @classmethod
    def from_config(cls, cfg):
        """cfg = camera_topology 的『內層』dict(links/overlapping/fusion/cameras/clock)。"""
        return cls(cfg.get("links", []), cfg.get("overlapping", []), cfg.get("fusion"),
                   cameras=cfg.get("cameras"), clock=cfg.get("clock"))

    @classmethod
    def from_yaml(cls, path):
        """讀 configs/camera_topology.yaml。

        該檔最外層包了一層 `camera_topology:`,from_config 期望的是內層 dict
        ——直接把整份檔案餵進 from_config 會靜默得到空拓撲(所有轉場都被拒),
        所以在這裡剝掉外層。
        """
        import yaml
        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        cfg = raw.get("camera_topology", raw)
        if not cfg.get("links"):
            raise ValueError(f"{path} 沒有任何 links → 所有跨鏡頭轉場都會被拒絕")
        return cls.from_config(cfg)

    def is_overlapping(self, a, b):
        return frozenset((a, b)) in self.overlapping

    def transition_gate(self, cam_from, t_exit, cam_to, t_enter):
        """v2 加權和用的硬門。回傳 (是否通過, st_prob)。

        ⚠ 實測發現 k_sigma 這道門在真實運作中幾乎從不觸發:只有外觀 cosine ≥ 0.851
          時它才是實際生效的限制,而 DINOv2(0.490)/OSNet(0.618)都達不到。
          真正淘汰候選的一直是 combined_threshold。詳見 analyze_gate_capacity.py §1。
        """
        if cam_from == cam_to:
            return (False, 0.0)
        key = (cam_from, cam_to)
        if key not in self.links:
            return (False, 0.0)
        mean, std = self.links[key]
        dt = t_enter - t_exit
        if dt <= 0:                                       # 不能比離開更早抵達
            return (False, 0.0)
        if abs(dt - mean) > self.fusion["k_sigma"] * std:  # 超出時間窗
            return (False, 0.0)
        return (True, st_prob(dt, mean, std))

    def direction_llr(self, cam_from, cam_to, exit_zone, enter_zone):
        """走對門沒有?只在有拓撲連結、且該連結有標 zone 時提供證據。"""
        if self.dir_lr is None:
            return 0.0
        want = self.link_zones.get((cam_from, cam_to))
        if want is None:
            return 0.0
        return self.dir_lr.llr(exit_zone, want[0], enter_zone, want[1])

    def transit_llr(self, cam_from, t_exit, cam_to, t_enter):
        """v3:轉場時間的對數勝算比 log p(Δt|同一人) − log λ_bg。

        三條路徑(依序嘗試,取第一條適用的):
          1. 同鏡頭極短間隔  → M4 軌跡中斷(SameCameraTransit)
          2. 有拓撲連結      → 正常轉場(config 的 μ/σ)
          3. 無拓撲連結      → 未建模路徑(UnknownPathTransit,帶負先驗)

        回傳 (是否物理可能, llr)。Δt ≤ 0 一律拒絕(不能比離開更早抵達)。
        """
        from m5_reid.evidence import NEG_INF
        dt = t_enter - t_exit
        if dt <= 0:
            return (False, NEG_INF)

        if cam_from == cam_to:
            if self.same_cam is None:
                return (False, NEG_INF)
            model = self.same_cam
        elif (cam_from, cam_to) in self.transits:
            model = self.transits[(cam_from, cam_to)]
        elif self.unknown_path is not None:
            model = self.unknown_path
        else:
            return (False, NEG_INF)

        lp = model.logpdf(dt)
        if lp <= NEG_INF:
            return (False, NEG_INF)
        return (True, lp - self.log_lambda_bg)
