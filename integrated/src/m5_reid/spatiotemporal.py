"""M5 空間+時序(Camera Link Model)核心:相機拓撲 + 轉場時間證據 + zone 判定。

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
    # 融合模式:llr = 對數勝算比(使用中);weighted_sum = 加權和(舊做法,保留供對照)
    "mode": "llr",
    # ── mode=llr 參數 ──────────────────────────────────────────────────
    "background_arrival_hz": 1.0 / 600.0,   # 「真正的新人」在單一鏡頭出現的速率
    "cost_false_merge_over_break": 5.0,     # 誤併比碎裂嚴重幾倍 → 換算成 LLR 門檻
    "transit_model": "loiter",              # loiter | gaussian
    "p_loiter": 0.15,                       # 中途停下來做事的比例
    "tau_loiter_s": 20.0,                   # 停留時間尺度
    "loiter_dist": "exp",                   # exp | lognormal(重尾)
    "loiter_log_sigma": 1.0,                # loiter_dist=lognormal 時的 logσ
    "appearance_profile": "dinov2",         # 外觀 LR 用哪組 embedder 的相似度分布
    "appearance_clip": None,                # 外觀 LLR 上下限(None=不夾)
    "overlap_llr": 5.0,                     # 重疊鏡頭同時觀測的幾何證據強度(nats)
    # 未建模路徑(繞路):走了拓撲沒列出的路線時的保底證據。選用功能,預設關閉。
    "unknown_path": {"enabled": False, "median_multiplier": 2.0,
                     "log_sigma": 0.8, "logprior": -2.0},
    # 同鏡頭重關聯(M4 軌跡中斷後在同一台鏡頭重新出現)。預設開啟。
    # 只用時間分不出「同一台鏡頭裡的哪一位廚師」,要搭配下方 position(位置證據)才收得緊。
    "same_camera": {"enabled": True, "tau_break_s": 2.0, "max_gap_s": 15.0},
    # 方向證據(出入口 zone):與轉場時間互相獨立,見 DirectionLR。選用功能,預設關閉。
    "direction": {"enabled": False, "q": 0.85, "n_zones": 3, "clip": 6.0},
    # 跨鏡頭地面校正(homography)。開啟後取代 overlap_llr 常數 —— 它能回答
    # 「重疊視野裡的是哪一個人」,而 overlap_llr 只能說「那裡有人」。選用功能,預設關閉。
    "ground_plane": {"enabled": False, "sigma_m": 0.4, "area_m2": 30.0, "clip": 8.0},
    # 速度證據:處理「兩人站在同一位置」—— 位置分不出,但他們從不同方向走來。
    # window_s 是觀測窗:越長證據越強,但綁定延遲越久。需要 ground_plane。選用功能,預設關閉。
    "velocity": {"enabled": False, "window_s": 0.5, "max_speed_mps": 1.5, "clip": 6.0},
    # 位置證據(僅同鏡頭適用):同鏡頭重關聯只用時間會誤併,位置是收緊它的關鍵。預設開啟。
    "position": {"enabled": True, "speed_bh_per_s": 0.5, "noise_bh": 0.3,
                 "frame_span_bh": 6.0, "clip": 8.0},
    # ── 以下皆為選用功能,預設關閉;關閉時不影響任何決策 ──────────────────
    #
    # margin:最佳與次佳候選的差距不足時,開新身份而不是賭一個。
    #   誤併(把兩個人併成一個)通常無法事後察覺,代價高於拆開(碎裂),
    #   所以「分不出來」時寧可碎裂。
    #   min_nats=None 時取 log(cost_false_merge_over_break),與 llr_threshold 同源。
    "margin": {"enabled": False, "min_nats": None},
    # same_camera_exclusive:同鏡頭互斥 —— 一個人不可能同時是同一台鏡頭上的兩條 track。
    #   identity_st.py 收集重疊候選時只排除 `c != camera_id`,不會檢查該 chef
    #   是否已經在本鏡頭上綁著另一條 track;開啟後補上這個檢查。
    #   track_ids 會在 removed 時剪除,所以它代表「目前還看得到的」,這個檢查是安全的。
    # iou_exempt:偵測器有時把一個人切成兩塊(遮擋、反光、半身框 + 全身框),
    #   那兩條 track 其實是同一個人。兩個框 IoU ≥ iou_exempt 時視為同一人、不擋。
    # ⚠ iou_exempt = None 時完全不做豁免檢查。
    "same_camera_exclusive": {"enabled": False, "iou_exempt": None},
    # cross_camera_exclusive:跨鏡頭互斥。pairs 列出「從不會同時看到同一個人」的
    #   鏡頭對;一個身份同時出現在這樣的一對上,物理上不可能 → 直接排除該候選。
    # ⚠ 這與 overlapping 是兩回事:overlapping 表示「兩邊的框都夠大、適合做幾何比對」,
    #   不是「看不看得到」。不在 overlapping 裡的兩台鏡頭仍可能隔著門口或縫隙互相看得到,
    #   把它們列為互斥會製造大量碎裂。
    # ⚠ pairs 必須寫在拓撲 yaml,不得寫死在程式碼 —— 它是場地的機位事實,換場地就不同。
    "cross_camera_exclusive": {"enabled": False, "pairs": []},
    # cross_view:跨鏡頭腳點一致性(CrossViewLR),取代重疊路徑的 overlap_llr 常數。
    #   pairs 寫在拓撲 yaml:
    #     {"cam_a|cam_b": {"H": [[...]], "sigma_px": .., "area_px2": ..}}
    #   H 的方向是 a→b(a、b 依字典序),反向自動用反矩陣。
    # ⚠ 沒有某一對的 H 時,那一對自動退回 overlap_llr 常數路徑,不會報錯也不會跳過。
    "cross_view": {"enabled": False, "clip": 8.0, "speed_px_per_s": 0.0, "pairs": {}},
    # transit_place:轉場的出入口位置(TransitPlaceLR),加在 transit_llr 之上,不是取代 ——
    #   一個問「多久」,一個問「從哪個門」。links 寫在拓撲 yaml:
    #     {"cam_a>cam_b": {"mu_exit": [..], "cov_exit": [[..]], "area_from": ..,
    #                      "mu_enter": [..], "cov_enter": [[..]], "area_to": ..}}
    #   ⚠ 有方向性(a>b 與 b>a 不同),與 cross_view 的 "a|b" 記法刻意不同以免混淆。
    # ⚠ 不得與 direction 同時開啟 —— DirectionLR 是同一份資訊的離散粗糙版。
    "transit_place": {"enabled": False, "clip": 6.0, "links": {}},
    # revote:累積投票。M5 預設只在 track 出現時決定一次身份;開啟後在 track 存活期間
    #   每隔 stride_loops 個迴圈重新評分一次,以滑動窗多數決決定是否改判。
    # ⚠ 連續幀高度相關,直接把每幀的 LLR 相加會高估信心,所以:
    #     (a) 用多數決而不是 LLR 相加
    #     (b) 取樣間隔 stride_loops 要拉開到相鄰兩票大致獨立
    # window:滑動窗保留最近幾票。min_votes:票數不足就不改判(短 track 保持原判)。
    # switch_margin:要改判,新的領先者必須比現任多這麼多票 —— 防止幾乎平手就把身份改掉。
    # assignment:同一台鏡頭、同一取樣輪的所有 track 怎麼分配候選。
    #   "greedy"    每條 track 各自搶分數最高的 —— 兩條可能搶到同一位(物理上不可能)
    #   "hungarian" 全域一對一最佳指派,會重新安排整組配對
    # ⚠ assignment 只作用在投票路徑上;on_new_track 逐條事件處理,不做批次指派。
    "revote": {"enabled": False, "stride_loops": 8, "window": 15,
               "min_votes": 3, "switch_margin": 2, "assignment": "greedy"},
    # cue:身份訊號。假設每次偵測都能讀到一個準確率 accuracy 的身份標記
    #   (例如帽頂色塊、編號),用來評估在廚師身上加可辨識標記的效果。關閉時 cue_lr 為 None。
    # ⚠ 一般執行流程不會提供標記讀數;開啟需另外提供 cue_token。
    # n_ids = 讀取器的字彙大小(有幾個不同的標記)。null = 執行時再呼叫 set_cue_n_ids() 指定。
    "cue": {"enabled": False, "accuracy": None, "n_ids": None, "clip": 8.0},
    # floorplan:場地平面圖上的距離(牆是障礙物)。開啟後 ground_plane 與 plan_transit
    #   的距離都改走它;關閉時是兩點直線距離。
    #   distance: rooms(房間 + 出入口連通圖)| grid(可行走網格,grid 給 npz 路徑)
    # ⚠ rooms / doors / grid 是場地的事實,必須寫在拓撲 yaml,不得寫死在程式碼。
    "floorplan": {"enabled": False, "distance": "rooms", "rooms": {}, "doors": {},
                  "grid": None},
    # plan_transit:轉場路徑的位置證據 ——「離開點到進入點,這段時間走得到嗎?」
    #   數學與 ground_plane 相同(同一個 GroundPlaneLR、同一組 σ / 速度 / 面積),
    #   只是比的是「離場點 vs 入場點」而且 dt > 0,所以不確定度隨時間放寬。
    # ⚠ 需要 ground_plane 生效(有 homography 才有平面座標),否則自動不生效。
    # ⚠ 不得與 transit_place 同時開啟 —— 兩者都在問「從哪個位置離開」,會重複計分。
    "plan_transit": {"enabled": False},
    "max_z": 6.0,                          # 轉場分布的遠尾截斷(省算,非決策門)
    # ── mode=weighted_sum 參數 ────────────────────────────────────────
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
        # 看起來像模型不好,實際是時鐘問題。
        self.cameras = dict(cameras or {})
        self.clock_offset = {c: float(v.get("clock_offset_s", 0.0) or 0.0)
                             for c, v in self.cameras.items()}
        self.clock = {"max_skew_s": 0.2, **(clock or {})}
        # 地面校正:有填點對應的鏡頭才有 homography。σ 由標定殘差決定,
        # 不用人填 —— σ 直接決定這條證據有沒有鑑別力,讓系統自己量比讓人猜可靠。
        from m5_reid.calibration import load_homographies, suggest_sigma
        self.homographies = load_homographies(self.cameras)
        self.calib_sigma_m = suggest_sigma(self.homographies)

        self._build_evidence()

    def world_xy(self, camera_id, bbox):
        """影像 bbox → 世界座標(公尺)。該鏡頭沒校正就回 None。"""
        h = self.homographies.get(camera_id)
        return h.foot_to_world(bbox) if h is not None else None

    def ground_dist(self, xy_a, xy_b):
        """兩個世界座標之間的距離。有平面圖時走平面圖(牆是障礙物),否則直線。"""
        if xy_a is None or xy_b is None:
            return None
        if self.plan_map is not None:
            return self.plan_map.distance(xy_a, xy_b)
        import math
        return math.hypot(xy_a[0] - xy_b[0], xy_a[1] - xy_b[1])

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
        """建 mode=llr 需要的轉場模型與外觀 LR。mode=weighted_sum 時不會被用到。"""
        from m5_reid.evidence import (AppearanceLR, CrossViewLR, DirectionLR, GroundPlaneLR,
                                      PositionLR, SameCameraTransit, TransitPlaceLR,
                                      UnknownPathTransit, VelocityLR, decision_threshold,
                                      make_transit)
        f = self.fusion
        kw = {}
        if f["transit_model"] == "loiter":
            kw = dict(p_loiter=f["p_loiter"], tau_loiter_s=f["tau_loiter_s"],
                      loiter_dist=f["loiter_dist"], loiter_log_sigma=f["loiter_log_sigma"])
        elif f["transit_model"] == "gaussian":
            kw = dict(max_z=f["max_z"])
        self.transits = {k: make_transit(mu, sd, kind=f["transit_model"], **kw)
                         for k, (mu, sd) in self.links.items()}

        # 未建模路徑(繞路):中位數取「典型直達時間 × 倍率」
        up = f.get("unknown_path") or {}
        if up.get("enabled"):
            typical = float(np.median([mu for mu, _ in self.links.values()])) if self.links else 4.0
            self.unknown_path = UnknownPathTransit(
                median_s=typical * float(up.get("median_multiplier", 2.0)),
                log_sigma=float(up.get("log_sigma", 0.8)),
                logprior=float(up.get("logprior", -2.0)))
        else:
            self.unknown_path = None

        # 同鏡頭重關聯(M4 軌跡中斷)
        sc = f.get("same_camera") or {}
        self.same_cam = SameCameraTransit(
            tau_break_s=float(sc.get("tau_break_s", 2.0)),
            max_gap_s=float(sc.get("max_gap_s", 15.0))) if sc.get("enabled") else None
        d = f.get("direction") or {}
        self.dir_lr = DirectionLR(q=float(d.get("q", 0.85)),
                                  n_zones=int(d.get("n_zones", 3)),
                                  clip=float(d.get("clip", 6.0))) if d.get("enabled") else None
        g = f.get("ground_plane") or {}
        # σ 優先用標定殘差;config 的 sigma_m 只在沒有 homography 時當退路。
        # ⚠ `sigma_m: null` 表示「自動從標定殘差推算」,
        #   所以這裡不能用 dict.get 的預設值 —— 鍵存在但值是 None 會出錯。
        sigma = self.calib_sigma_m or (g.get("sigma_m") if g.get("sigma_m") else None)
        enabled = g.get("enabled", bool(self.homographies))
        # ⚠ 不可以在「沒標定、也沒填 σ」時偷偷套一個預設 σ:
        #   猜的 σ 下這條證據幾乎沒有鑑別力,但看起來是開著的,
        #   使用者會以為地面校正生效了。寧可明確停用並記下原因,讓自檢報出來。
        self.calib_warning = None
        if enabled and sigma is None:
            self.calib_warning = (
                "ground_plane 已啟用但沒有 homography、也沒填 sigma_m → 停用。"
                "請完成地面校正(手冊步驟二);沒有它多人同時在場會被併成一個身份。")
            enabled = False
        sigma = float(sigma) if sigma is not None else None
        # speed:兩次觀測之間人可能走多遠(預設 0.9 m/s)。
        #   平面圖單位不是公尺時要改成同一個單位。
        _gp_kw = dict(sigma_m=sigma, area_m2=float(g.get("area_m2", 30.0)),
                      clip=float(g.get("clip", 8.0)), speed_mps=float(g.get("speed", 0.9)))
        self.ground_lr = GroundPlaneLR(**_gp_kw) if enabled else None
        # 逐鏡頭對的 σ:單一全域 σ 會被樣本最多的那一對主導,
        # 投影誤差較大的鏡頭對會把真正的配對擋掉。
        # 與 CrossViewLR 同一個原則:逐對估,品質差的那對自己弱。
        #   pairs: {"cam_a|cam_b": 單台 σ}(內部乘 √2,與 sigma_m 同一個慣例)
        #   mode:  absolute = 位置證據自己要過門檻;
        #          share    = 把「那台鏡頭裡有 k 個人」的均勻稀釋 −log k 換成依位置分配
        # ⚠ 沒列在 pairs 的鏡頭對**退回常數路徑**,不是用全域 σ 硬算。
        self.ground_pairs = {}
        if enabled:
            for key, sig in (g.get("pairs") or {}).items():
                a, b = key.split("|")
                self.ground_pairs[frozenset((a, b))] = GroundPlaneLR(
                    **{**_gp_kw, "sigma_m": float(sig)})
        self.ground_mode = str(g.get("mode", "absolute"))
        if self.ground_mode not in ("absolute", "share"):
            raise ValueError(f"ground_plane.mode 只能是 absolute 或 share,收到 {self.ground_mode!r}")
        self.ground_share_eps = float(g.get("share_eps", 0.05))

        # 平面圖距離。關閉時 plan_map 為 None,ground_dist 回直線距離。
        fp = f.get("floorplan") or {}
        self.plan_map = None
        if fp.get("enabled", False):
            from m5_reid.floorplan import PlanMap
            self.plan_map = PlanMap.from_config(fp)
        pt = f.get("plan_transit") or {}
        # ⚠ 另建一個實例而不是共用 ground_lr:兩者之後若要分別調整(例如轉場的 σ
        #   要含離場點外推誤差)不必再拆;目前參數完全相同。
        self.plan_transit_lr = (GroundPlaneLR(**_gp_kw)
                                if (pt.get("enabled", False) and enabled) else None)
        if pt.get("enabled", False) and (f.get("transit_place") or {}).get("enabled", False):
            raise ValueError(
                "plan_transit 與 transit_place 不可同時開啟 —— "
                "兩者都在問「從哪個位置離開」,同時開會重複計算證據。")
        vc = f.get("velocity") or {}
        # 速度是從地面座標差分而來,所以它繼承標定誤差,也繼承「沒標定就不能用」。
        self.vel_lr = VelocityLR(
            sigma_pos_m=sigma,
            window_s=float(vc.get("window_s", 0.5)),
            max_speed_mps=float(vc.get("max_speed_mps", 1.5)),
            clip=float(vc.get("clip", 6.0))) if (vc.get("enabled") and enabled) else None
        pos = f.get("position") or {}
        self.pos_lr = PositionLR(
            speed_bh_per_s=float(pos.get("speed_bh_per_s", 0.5)),
            noise_bh=float(pos.get("noise_bh", 0.3)),
            frame_span_bh=float(pos.get("frame_span_bh", 6.0)),
            clip=float(pos.get("clip", 8.0))) if pos.get("enabled", True) else None
        self.app_lr = AppearanceLR.measured(f["appearance_profile"], clip=f["appearance_clip"])
        self.log_lambda_bg = np.log(float(f["background_arrival_hz"]))
        self.llr_threshold = decision_threshold(f["cost_false_merge_over_break"])

        # margin / same_camera_exclusive。預設關閉 → margin_nats=None、same_cam_exclusive=False,
        # identity_st 不做這兩項檢查。
        mg = f.get("margin") or {}
        self.margin_nats = None
        if mg.get("enabled", False):
            # 預設與 llr_threshold 同源:兩者都是「誤併比碎裂貴幾倍」換算出來的門檻。
            # 差距小於它,表示區分兩個假設的證據還不足以承擔誤併的成本。
            m = mg.get("min_nats")
            self.margin_nats = (self.llr_threshold if m is None else float(m))
        sce = f.get("same_camera_exclusive") or {}
        self.same_cam_exclusive = bool(sce.get("enabled", False))
        # None → 不做豁免檢查,不是 0.0。
        # 0.0 會讓「完全沒有交集的兩個框」也被豁免,等於把同鏡頭互斥整條關掉。
        _iou_x = sce.get("iou_exempt")
        self.same_cam_iou_exempt = None if _iou_x is None else float(_iou_x)
        # 跨鏡頭互斥:與 self.overlapping 用同一種表示法(frozenset),查詢方式也一致。
        cce = f.get("cross_camera_exclusive") or {}
        self.cross_cam_exclusive = bool(cce.get("enabled", False))
        self.exclusive_pairs = {frozenset(p) for p in (cce.get("pairs") or [])}
        # 一對鏡頭不可能既「重疊」又「從未共現」—— 設定寫錯時要當場炸,
        # 不要安靜地讓其中一條規則贏。
        _both = self.exclusive_pairs & self.overlapping
        if _both:
            raise ValueError(
                "同一對鏡頭同時列在 overlapping 與 cross_camera_exclusive.pairs:"
                + ", ".join("+".join(sorted(p)) for p in sorted(map(sorted, _both))))

        # 累積投票。關閉時 revote 為 None,identity_st 據此完全跳過該路徑。
        rv = f.get("revote") or {}
        asg = str(rv.get("assignment", "greedy")).lower()
        if asg not in ("greedy", "hungarian"):
            raise ValueError(f"revote.assignment 只能是 greedy 或 hungarian,收到 {asg!r}")
        self.revote = (dict(stride_loops=int(rv.get("stride_loops", 8)),
                            window=int(rv.get("window", 15)),
                            min_votes=int(rv.get("min_votes", 3)),
                            switch_margin=int(rv.get("switch_margin", 2)),
                            assignment=asg)
                       if rv.get("enabled", False) else None)

        # 身份訊號。關閉時為 None,identity_st 完全跳過。
        # ⚠ n_ids 可以留 null,執行時再呼叫 set_cue_n_ids() 指定 ——
        #   讀取器的字彙大小(有幾個不同的標記)要到執行時才知道,
        #   寫死在拓撲裡會讓同一份設定檔在不同場景代表不同的證據強度。
        cu = f.get("cue") or {}
        self.cue_lr = None
        if cu.get("enabled", False):
            from m5_reid.cue import CueLR
            if cu.get("accuracy") is None:
                raise ValueError("cue.enabled 為 true 時必須指定 accuracy")
            self._cue_cfg = dict(accuracy=float(cu["accuracy"]),
                                 clip=cu.get("clip", 8.0))
            if cu.get("n_ids") is not None:
                self.cue_lr = CueLR(n_ids=int(cu["n_ids"]), **self._cue_cfg)

        # cross_view:建 {目標鏡頭: {來源鏡頭: CrossViewLR}}。
        # 查詢方向是「來源鏡頭的 bbox → 目標鏡頭的 bbox」,所以存進來時
        # 兩個方向都要建 —— 反向用 H 的反矩陣。
        self._cross_view = {}
        cvc = f.get("cross_view") or {}
        if cvc.get("enabled", False):
            clip = float(cvc.get("clip", 8.0))
            spd = float(cvc.get("speed_px_per_s", 0.0))
            for key, m in (cvc.get("pairs") or {}).items():
                a, b = key.split("|")
                H = np.asarray(m["H"], dtype=float).reshape(3, 3)
                sig, area = float(m["sigma_px"]), float(m["area_px2"])
                self._cross_view.setdefault(b, {})[a] = CrossViewLR(
                    H, sig, area, clip=clip, speed_px_per_s=spd)
                # ⚠ 反向的 σ/area 沿用同一組。嚴格說反向的像素尺度不同,
                #   但兩者都是同一組對應點量出來的,而且 area 只進 log ——
                #   誤差落在 log 尺度上遠小於 d²/2σ² 那一項。
                #   若之後發現方向性偏差,應該**分別估**而不是在這裡湊係數。
                try:
                    self._cross_view.setdefault(a, {})[b] = CrossViewLR(
                        np.linalg.inv(H), sig, area, clip=clip, speed_px_per_s=spd)
                except np.linalg.LinAlgError:
                    pass                      # H 退化 → 只留單向,不讓整份拓撲掛掉

        # transit_place:{(cam_from, cam_to): TransitPlaceLR}。有方向性,不建反向。
        self._transit_place = {}
        tpc = f.get("transit_place") or {}
        if tpc.get("enabled", False):
            clip = float(tpc.get("clip", 6.0))
            for key, m in (tpc.get("links") or {}).items():
                a, b = key.split(">")
                self._transit_place[(a, b)] = TransitPlaceLR(
                    m["mu_exit"], m["cov_exit"], m["area_from"],
                    m["mu_enter"], m["cov_enter"], m["area_to"], clip=clip)
            # ⚠ 兩者是同一份資訊(離開時往哪個方向 / 從哪個位置離開),
            #   同時開會把它算兩次。寧可在建構時就吵,也不要靜默雙計。
            if (f.get("direction") or {}).get("enabled", False):
                raise ValueError(
                    "transit_place 與 direction 不可同時開啟 —— "
                    "DirectionLR 是同一份資訊的離散粗糙版,同時開會重複計算證據。")

    def set_cue_n_ids(self, n_ids):
        """補上讀取器的字彙大小(有幾個不同的標記)。cue 沒開就是 no-op。

        ⚠ 這個值要到執行時才知道,寫死在拓撲裡會讓同一份設定檔在不同場景
          代表不同的證據強度 —— 而那種差異不會報錯。所以由呼叫端在執行時指定。
        """
        from m5_reid.cue import CueLR

        if not (self.fusion.get("cue") or {}).get("enabled", False):
            return None
        self.cue_lr = CueLR(n_ids=int(n_ids), **self._cue_cfg)
        return self.cue_lr

    def transit_place(self, cam_from, cam_to):
        """回傳該連結的 TransitPlaceLR;沒有就 None(呼叫端加 0)。"""
        return self._transit_place.get((cam_from, cam_to))

    def cross_view(self, camera_to):
        """回傳 {來源鏡頭: CrossViewLR},把來源的 bbox 投到 camera_to 比對。

        沒有任何單應性時回 None（而不是空 dict）——
        呼叫端據此退回舊的 overlap_llr 常數路徑。
        """
        return self._cross_view.get(camera_to) or None

    def set_transit(self, cam_from, cam_to, model):
        """用實際量到的轉場時間校準後,把某條連結的轉場模型換掉。"""
        self.transits[(cam_from, cam_to)] = model

    @classmethod
    def from_config(cls, cfg):
        """cfg = camera_topology 的『內層』dict(links/overlapping/fusion/cameras/clock)。"""
        return cls(cfg.get("links", []), cfg.get("overlapping", []), cfg.get("fusion"),
                   cameras=cfg.get("cameras"), clock=cfg.get("clock"))

    @classmethod
    def from_yaml(cls, path):
        """讀拓撲 yaml(scripts/build_topology.py 產生,或 run_pipeline.py --topology 指定的檔案)。

        該檔最外層包了一層 `camera_topology:`,from_config 期望的是內層 dict
        ——直接把整份檔案餵進 from_config 會靜默得到空拓撲(所有轉場都被拒),
        所以在這裡剝掉外層。
        """
        import yaml
        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        cfg = raw.get("camera_topology", raw)
        # 這個守衛是為了擋「忘了填 links」的靜默失效。但**全部鏡頭兩兩重疊**時
        # links 本來就該是空的(重疊路徑不需要轉場時間),所以只在同時也沒有
        # overlapping 時才算設定錯誤。
        if not cfg.get("links") and not cfg.get("overlapping"):
            raise ValueError(
                f"{path} 既沒有 links 也沒有 overlapping → 所有跨鏡頭關聯都會被拒絕")
        return cls.from_config(cfg)

    def is_overlapping(self, a, b):
        return frozenset((a, b)) in self.overlapping

    def is_mutually_exclusive(self, a, b):
        """這兩台從不會同時看到同一個人 → 同一瞬間不可能是同一個人。

        ⚠ 不是 `not is_overlapping(a, b)`。「不重疊」只表示不適合做幾何比對,
          兩台仍可能隔著門口互相看得到。
        """
        return frozenset((a, b)) in self.exclusive_pairs

    def transition_gate(self, cam_from, t_exit, cam_to, t_enter):
        """mode=weighted_sum 用的硬門。回傳 (是否通過, st_prob)。

        ⚠ 外觀相似度不高時,k_sigma 這道門幾乎不會是實際生效的限制,
          真正淘汰候選的是 combined_threshold。
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
        """mode=llr:轉場時間的對數勝算比 log p(Δt|同一人) − log λ_bg。

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
