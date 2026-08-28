"""M5 蒙地卡羅的「世界模型」—— 產生 ground truth 事件流。

⚠ 本模組**刻意不 import m5_reid.evidence**,也不共用任何參數。

理由:模擬器的任務不是證明系統能動,而是找出系統的斷點,再回答「真實世界的
偏差落在斷點的哪一側」。若世界用系統的同一個分布族產生資料,量到的只是
「用高斯驗高斯」,無法反駁循環論證的指控。所以:

    世界的轉場時間 = 對數常態(長尾) + 逗留混合 + 未建模繞路
    系統的轉場時間 = 高斯 或 逗留混合(它自己以為的)

掃描的維度是**失配程度**(σ̂/σ_true、μ̂/μ_true、逗留率、繞路率、時鐘漂移),
不是絕對值。這樣量到的是「模型錯多少會壞」,而不是「假設成立時有多好」。

兩個世界參數從**真實資料**校準,不是臆測(報告中要明講這兩個不是我們編的):
  · 外觀:EPFL 449 crops 實測的 same/diff cosine 分布
  · M4 缺陷率:由實際跑 KitchenTracker 的結果量測(見 m4_defect_rates)
"""
import math

import numpy as np


class Chef:
    def __init__(self, gt_id, anchor):
        self.gt_id = gt_id
        self.anchor = anchor


class WorldConfig:
    """世界的真實參數。與系統 config 完全分離。"""

    def __init__(self, n_chefs=4, duration_s=3600.0,
                 # 真實轉場:對數常態(長尾),中位數 = median_s
                 transit_median_s=4.0, transit_log_sigma=0.35,
                 # 逗留:以機率 p 額外加一段指數停留
                 p_loiter=0.15, tau_loiter_s=20.0,
                 # 繞路:走了拓撲沒建模的路徑 → 出現在「無連結」的鏡頭
                 p_detour=0.05,
                 # 每位廚師平均多久轉場一次
                 transition_interval_s=60.0,
                 # 鏡頭時鐘漂移(秒),{cam: offset}
                 clock_skew=None,
                 # M4 缺陷:軌跡中斷率(同鏡頭多開一個 track)、漏偵率
                 m4_fragment_rate=0.05, m4_miss_rate=0.02,
                 # 外觀:EPFL 實測(same 0.490±0.10 / diff 0.465±0.10)
                 app_mu_same=0.490, app_sigma_same=0.10, app_mu_diff=0.465,
                 # γ=0 用實測可分性;γ→1 模擬同制服(可分性趨近 0)
                 gamma_uniform=0.0,
                 # 地面座標:廚房尺寸(公尺)與各鏡頭的標定殘差。
                 # ⚠ 不模擬 homography 的估計過程,直接注入殘差 —— 要問的是
                 #   「殘差多大還能用」,不是「怎麼估 homography」。
                 # ⚠ calib_sigma_m 代表**總殘差**;系統性偏差按比例分配,否則掃描
                 #   會被混淆:固定的偏差在 σ 很小時會主導,讓「校正越準反而越差」。
                 kitchen_m=(10.0, 6.0), calib_sigma_m=0.4, calib_bias_ratio=0.4,
                 # 速度自相關時間(秒)。⚠ 這直接決定軌跡證據有沒有用:
                 #   小 = 隨機遊走(速度不預測下一步)、大 = 直線行走。
                 #   原本的 step_world 等於 tau_v→0,會讓軌跡證據因為錯的理由失效。
                 tau_v_s=3.0, walk_speed_mps=0.9, vel_window_s=0.5,
                 # 全景鏡頭:一台看得到整個廚房的鏡頭。每位廚師全程在它畫面裡,
                 # 唯一的身份遺失路徑是它自己因遮擋而斷軌(master_fragment_rate)。
                 master_camera=None, master_fragment_rate=0.05,
                 # 方向:走連結時使用「對應 zone」的機率;zone_error 是標註/幾何誤差
                 n_zones=3, q_zone=0.85, zone_error_rate=0.0,
                 # 畫面幾何:以「人身高」為單位。frame_span_bh 見 PositionLR。
                 body_px=200.0, frame_span_bh=6.0,
                 dim=64, seed=0):
        self.__dict__.update(locals())
        del self.__dict__["self"]
        self.rng = np.random.RandomState(seed)


def _l2(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def make_anchors(n, mu_same, mu_diff, gamma, dim, rng):
    """造 n 個廚師的外觀錨點,使得

        cos(觀測_i, 錨點_i) ≈ mu_same      (同一人)
        cos(觀測_i, 錨點_j) ≈ mu_diff      (不同人)

    作法:錨點彼此的 cosine 固定為 c = mu_diff / mu_same,則觀測從自己的錨點
    以 mu_same 抽出後,對別人錨點的 cosine 自然落在 mu_same·c = mu_diff。
    c 接近 1 = 錨點幾乎重合 = 外觀幾乎沒有鑑別力,這正是「同制服」的樣子。

    gamma ∈ [0,1] 把 c 往 1 推:γ=0 用 EPFL 實測可分性,γ=1 完全無資訊。
    """
    c = mu_diff / mu_same
    c = c + gamma * (1.0 - c)
    u = _l2(rng.randn(dim))
    anchors = []
    for _ in range(n):
        e = rng.randn(dim)
        e = _l2(e - (e @ u) * u)                    # 與共同分量正交
        anchors.append(_l2(np.sqrt(c) * u + np.sqrt(max(0.0, 1 - c)) * e))
    return anchors


def pick_zone(cfg, rng, expected=None):
    """走連結時以 q_zone 的機率使用「對應的 zone」,否則隨機;再疊上標註誤差。

    zone_error_rate 模擬業主標錯 zone 或 bbox→zone 的幾何誤判 —— 這是決定
    「值不值得要求業主付標註成本」的關鍵參數。
    """
    if expected is not None and rng.rand() < cfg.q_zone:
        z = expected
    else:
        z = int(rng.randint(cfg.n_zones))
    if rng.rand() < cfg.zone_error_rate:          # 觀測被污染
        z = int(rng.randint(cfg.n_zones))
    return z


def rand_world(cfg, rng):
    """廚房地面上的隨機位置(公尺)。"""
    W, H = cfg.kitchen_m
    return (float(rng.rand() * W), float(rng.rand() * H))


def step_world(cfg, state, dt, rng):
    """經過 dt 秒後的世界座標與速度。

    速度用 Ornstein–Uhlenbeck:朝一個方向走一陣子、偶爾轉向 —— 這才是人走路的樣子。
    ⚠ 舊版是純隨機遊走(每步方向獨立重抽),等於 tau_v→0,
      速度完全不預測下一步 → 軌跡證據會因為「這個世界沒有慣性」而失效,
      而不是因為「速度分不出人」。那會得到錯誤的結論。
    """
    W, H = cfg.kitchen_m
    xy, v = state
    dt = max(dt, 0.0)
    a = math.exp(-dt / max(cfg.tau_v_s, 1e-6))
    sd = cfg.walk_speed_mps * math.sqrt(max(1 - a * a, 0.0))
    v = (v[0] * a + rng.randn() * sd, v[1] * a + rng.randn() * sd)
    x = min(max(xy[0] + v[0] * dt, 0.0), W)
    y = min(max(xy[1] + v[1] * dt, 0.0), H)
    if x in (0.0, W):
        v = (-v[0], v[1])                     # 撞牆反彈,速度才不會一直頂著牆
    if y in (0.0, H):
        v = (v[0], -v[1])
    return ((float(x), float(y)), (float(v[0]), float(v[1])))


def observe_vel(cfg, v, rng):
    """觀測到的速度 = 真值 + 噪聲。噪聲由「位置誤差 ÷ 觀測窗」推得
    ——速度是兩個位置點差分出來的,所以它繼承校正誤差。"""
    sv = cfg.calib_sigma_m * math.sqrt(2.0) / max(cfg.vel_window_s, 1e-6)
    return (float(v[0] + rng.randn() * sv), float(v[1] + rng.randn() * sv))


def observe_world(cfg, xy, cam, rng, _bias={}):
    """某台鏡頭「透過 homography 推得」的世界座標 = 真值 + 該台的系統性偏差 + 噪聲。

    系統性偏差每台固定(標定時的殘差),噪聲每次不同(腳點偵測誤差)。
    """
    b = cfg.calib_sigma_m * cfg.calib_bias_ratio          # 系統性(每台固定)
    n = cfg.calib_sigma_m * math.sqrt(max(1 - cfg.calib_bias_ratio ** 2, 0.0))  # 隨機
    key = (cam, round(b, 6))
    if key not in _bias:
        _bias[key] = (rng.randn() * b, rng.randn() * b)
    bx, by = _bias[key]
    return (float(xy[0] + bx + rng.randn() * n),
            float(xy[1] + by + rng.randn() * n))


def rand_pos(cfg, rng):
    """畫面內的隨機落點(以像素表示,身高 = body_px)。"""
    span = cfg.frame_span_bh * cfg.body_px
    return (float(rng.rand() * span), float(rng.rand() * span))


def step_pos(cfg, pos, dt, rng):
    """經過 dt 秒後的新位置。步速 0.5 身高/秒,撞邊界就夾住。"""
    span = cfg.frame_span_bh * cfg.body_px
    sigma = 0.5 * cfg.body_px * max(dt, 0.0)
    x = min(max(pos[0] + rng.randn() * sigma, 0.0), span)
    y = min(max(pos[1] + rng.randn() * sigma, 0.0), span)
    return (float(x), float(y))


def to_bbox(cfg, pos):
    """腳點 → (x1,y1,x2,y2)。寬約身高的 0.4 倍。"""
    h = cfg.body_px
    return (pos[0] - 0.2 * h, pos[1] - h, pos[0] + 0.2 * h, pos[1])


def observe(anchor, mu_same, sigma_same, rng):
    """從錨點抽一個觀測向量,使**兩次觀測之間**的 cosine 服從 N(mu_same, sigma_same)。

    ⚠ 關鍵:AppearanceLR 的 mu_same=0.490 是 EPFL 實測的 **crop 對 crop** cosine,
      不是「crop 對錨點」。而 cos(o1,o2) ≈ cos(o1,錨點)·cos(o2,錨點),
      所以振幅要取 **√mu_same** 才對得上。

      踩過的坑:原本直接用 mu_same 當振幅 → 世界產生的同一人 cosine 只有
      0.49²=0.242,比 AppearanceLR 認定的「不同人 0.465」還低 → **同一個人
      一律拿到負的外觀證據**,把每次綁定都往下拉約 0.4 nats。
    """
    amp = math.sqrt(max(mu_same, 0.0))
    sd = sigma_same / (2.0 * max(amp, 1e-6))      # 讓成對 cosine 的標準差仍為 sigma_same
    a = float(np.clip(rng.normal(amp, sd), -0.99, 0.99))
    n = rng.randn(len(anchor))
    n = _l2(n - (n @ anchor) * anchor)
    return _l2(a * anchor + np.sqrt(max(0.0, 1 - a * a)) * n)


def sample_transit(cfg, rng, want_tag=False):
    """真實轉場時間:對數常態 + 以機率 p_loiter 疊加指數停留。

    系統以為是高斯(或高斯+逗留混合),世界用對數常態 —— 分布族刻意不同。
    """
    dt = rng.lognormal(np.log(cfg.transit_median_s), cfg.transit_log_sigma)
    loitered = rng.rand() < cfg.p_loiter
    if loitered:
        dt += rng.exponential(cfg.tau_loiter_s)
    return (float(dt), loitered) if want_tag else float(dt)


def generate(cfg, links, all_cameras, link_zones=None):
    """產生事件流。

    回傳 [(kind, ..., bbox, zone, world_xy, world_v)]。
    t_observed 已加上該鏡頭的時鐘漂移(系統看到的就是這個,不是真值)。
    tag 標明這次抵達的成因,供失效分解用:
      first    首次入場(不是轉場,不計入碎裂/誤併)
      normal   正常轉場(有拓撲連結、沒逗留)
      loiter   中途逗留後才抵達
      detour   走了拓撲沒建模的路徑
      fragment M4 斷軌造成的同鏡頭重新出現
    """
    rng = cfg.rng
    anchors = make_anchors(cfg.n_chefs, cfg.app_mu_same, cfg.app_mu_diff,
                           cfg.gamma_uniform, cfg.dim, rng)
    chefs = [Chef(i, anchors[i]) for i in range(cfg.n_chefs)]
    skew = cfg.clock_skew or {}
    # ⚠ 全景鏡頭**不參與一般鏡頭的輪替**。它是「一直看得到整個廚房」,
    #   不是「廚師會走過去的另一個station」。
    #   踩過的坑:master 混在 cams 裡 → 廚師會「轉場到 master」,產生
    #   normal/detour 事件,與那條連續的全景 track 互相衝突 →
    #   master 的 detour 與 fragment 100% 碎裂,整體多出約 14% 的假碎裂。
    #   這就是第六輪「近乎完美校正下仍有 14.5% 碎裂」的真正原因。
    cams = sorted(c for c in all_cameras if c != cfg.master_camera)
    out_links = {}
    for (a, b) in links:
        out_links.setdefault(a, []).append(b)
    lz = link_zones or {}

    events = []
    traj = {}          # gt_id -> [(t, 真實世界座標)] 供全景鏡頭取樣同一條軌跡
    for chef in chefs:
        cam = cams[rng.randint(len(cams))]
        t = rng.rand() * cfg.transition_interval_s
        pos = rand_pos(cfg, rng)
        wst = (rand_world(cfg, rng), (0.0, 0.0))
        traj.setdefault(chef.gt_id, []).append((t, wst))
        events.append(("enter", chef.gt_id, cam, t + skew.get(cam, 0.0),
                       observe(chef.anchor, cfg.app_mu_same, cfg.app_sigma_same, rng),
                       "first", to_bbox(cfg, pos), pick_zone(cfg, rng),
                       observe_world(cfg, wst[0], cam, rng),
                       observe_vel(cfg, wst[1], rng)))
        while t < cfg.duration_s:
            stay = rng.exponential(cfg.transition_interval_s)
            t_leave = t + stay
            if t_leave > cfg.duration_s:
                break
            pos = step_pos(cfg, pos, t_leave - t, rng)      # 停留期間有走動
            wst = step_world(cfg, wst, t_leave - t, rng)
            traj[chef.gt_id].append((t_leave, wst))
            nxt = out_links.get(cam, [])
            detour = rng.rand() < cfg.p_detour or not nxt
            if detour:                                  # 走了拓撲沒建模的路徑
                dest = cams[rng.randint(len(cams))]
                dt = sample_transit(cfg, rng) * (1.0 + rng.rand() * 3)   # 繞路比較久
                tag = "detour"
            else:
                dest = nxt[rng.randint(len(nxt))]
                dt, loitered = sample_transit(cfg, rng, want_tag=True)
                tag = "loiter" if loitered else "normal"
            # 離場 zone 取決於「等一下要走哪條連結」→ 這正是方向證據的來源
            want = lz.get((cam, dest), (None, None))
            events.append(("leave", chef.gt_id, cam, t_leave + skew.get(cam, 0.0),
                           None, "", to_bbox(cfg, pos),
                           pick_zone(cfg, rng, None if detour else want[0]),
                           observe_world(cfg, wst[0], cam, rng),
                       observe_vel(cfg, wst[1], rng)))
            t = t_leave + dt
            if t > cfg.duration_s:
                break
            if rng.rand() < cfg.m4_miss_rate:           # M4 漏偵 → 這次抵達沒被看到
                cam = dest
                continue
            cam = dest
            pos = rand_pos(cfg, rng)                        # 換鏡頭 → 新影像座標系,重抽
            wst = step_world(cfg, wst, dt, rng)             # 世界座標是連續的
            traj[chef.gt_id].append((t, wst))
            events.append(("enter", chef.gt_id, cam, t + skew.get(cam, 0.0),
                           observe(chef.anchor, cfg.app_mu_same, cfg.app_sigma_same, rng),
                           tag, to_bbox(cfg, pos),
                           pick_zone(cfg, rng, None if detour else want[1]),
                           observe_world(cfg, wst[0], cam, rng),
                       observe_vel(cfg, wst[1], rng)))
            if rng.rand() < cfg.m4_fragment_rate:       # M4 斷軌 → 同鏡頭再開一個 track
                t_frag = t + rng.exponential(5.0)
                pos = step_pos(cfg, pos, t_frag - t, rng)
                wst = step_world(cfg, wst, t_frag - t, rng)
                events.append(("leave", chef.gt_id, cam, t_frag + skew.get(cam, 0.0),
                               None, "", to_bbox(cfg, pos), pick_zone(cfg, rng),
                               observe_world(cfg, wst[0], cam, rng),
                       observe_vel(cfg, wst[1], rng)))
                # 關鍵:斷軌前後是同一個人,位置只移動了 0.5 秒的距離
                pos = step_pos(cfg, pos, 0.5, rng)
                events.append(("enter", chef.gt_id, cam, t_frag + 0.5 + skew.get(cam, 0.0),
                               observe(chef.anchor, cfg.app_mu_same, cfg.app_sigma_same, rng),
                               "fragment", to_bbox(cfg, pos), pick_zone(cfg, rng),
                               observe_world(cfg, wst[0], cam, rng),
                       observe_vel(cfg, wst[1], rng)))
                t = t_frag + 0.5
    if cfg.master_camera:
        events += _master_events(cfg, events, rng, traj,
                                 {c.gt_id: c.anchor for c in chefs})
    # 同一時刻時,「心跳」必須排在「進場」之前 —— 否則位置資訊到得太晚,
    # 綁定決策當下拿到的還是舊位置,地面校正等於沒接上。
    events.sort(key=lambda e: (e[3], 0 if e[0] == "update" else 1))
    return events


def _world_at(traj_pts, t):
    """在該廚師的真實軌跡上取 t 時刻的位置(最近鄰)。

    全景鏡頭必須觀測到**與其他鏡頭同一個**世界座標,否則地面校正證據就是假的
    ——它會變成在比對兩條無關的軌跡。
    """
    best, bd = traj_pts[0][1], abs(traj_pts[0][0] - t)
    for tt, xy in traj_pts:
        d = abs(tt - t)
        if d < bd:
            bd, best = d, xy
    return best


def _master_events(cfg, events, rng, traj, anchors):
    """全景鏡頭的事件流:每位廚師從首次出現到最後一次出現,全程在畫面裡。

    唯一會中斷的原因是遮擋(中島、抽油煙機、其他人)→ master_fragment_rate。
    斷掉之後立刻重新出現(人沒走,只是被擋住),所以同鏡頭重關聯路徑會接手。
    """
    span = {}
    for kind, gt, cam, t, emb, tag, box, zn, wxy, wv in events:
        lo, hi = span.get(gt, (t, t))
        span[gt] = (min(lo, t), max(hi, t))

    # 別台鏡頭的進場時刻 —— 心跳只需要在這些瞬間之前送到
    enters = sorted({t for kind, gt, cam, t, *_ in events
                     if kind == "enter" and cam != cfg.master_camera})

    def beats(gt, pts, a, b, mc, rng):
        """在 [a,b] 這段「track 確實活著」的區間內,於每個別台進場時刻回報位置。

        ⚠ 只在活著的區間發 —— 斷軌空檔沒有 track,回報了也沒有對應的 chef。
          真實系統就是這樣:M4 每幀輸出當前 active tracks,track 不在就沒有它。
        """
        return [("update", gt, mc, t, None, "", None, None,
                 observe_world(cfg, _world_at(pts, t)[0], mc, rng),
                 observe_vel(cfg, _world_at(pts, t)[1], rng))
                for t in enters if a <= t <= b]

    out = []
    mc = cfg.master_camera
    for gt, (t0, t1) in span.items():
        # ⚠ 必須用**該廚師本人的**錨點。踩過的坑:這裡原本重新抽一個隨機錨點,
        #   於是全景鏡頭與其他鏡頭的外觀完全無關 → 跨鏡頭 cosine ≈ 0 →
        #   外觀證據給 −1.2 nats,把地面證據的 +2.5 拖到門檻以下 → 全部綁不上。
        #   那是模擬的 bug,不是系統的性質。
        anchor = anchors[gt]
        pos = rand_pos(cfg, rng)
        t = t0
        pts = traj.get(gt) or [(t0, rand_world(cfg, rng))]
        out.append(("enter", gt, mc, t, observe(anchor, cfg.app_mu_same,
                                                cfg.app_sigma_same, rng),
                    "first", to_bbox(cfg, pos), pick_zone(cfg, rng),
                    observe_world(cfg, _world_at(pts, t)[0], mc, rng),
                    observe_vel(cfg, _world_at(pts, t)[1], rng)))
        while t < t1:
            # 下一次遮擋斷軌。rate 越高,平均連續時間越短。
            gap = rng.exponential(max(60.0 / max(cfg.master_fragment_rate, 1e-6), 1.0))
            t_break = min(t + gap, t1)
            if t_break >= t1:
                break
            pos = step_pos(cfg, pos, t_break - t, rng)
            out += beats(gt, pts, t, t_break, mc, rng)      # 這段 track 活著
            out.append(("leave", gt, mc, t_break, None, "", to_bbox(cfg, pos),
                        pick_zone(cfg, rng),
                        observe_world(cfg, _world_at(pts, t_break)[0], mc, rng),
                    observe_vel(cfg, _world_at(pts, t_break)[1], rng)))
            t = t_break + rng.uniform(0.3, 2.0)          # 被擋住一下下就又看到
            pos = step_pos(cfg, pos, t - t_break, rng)
            out.append(("enter", gt, mc, t, observe(anchor, cfg.app_mu_same,
                                                    cfg.app_sigma_same, rng),
                        "fragment", to_bbox(cfg, pos), pick_zone(cfg, rng),
                        observe_world(cfg, _world_at(pts, t)[0], mc, rng),
                    observe_vel(cfg, _world_at(pts, t)[1], rng)))
        out += beats(gt, pts, t, t1, mc, rng)              # 最後一段
        out.append(("leave", gt, mc, t1, None, "", to_bbox(cfg, pos), pick_zone(cfg, rng),
                    observe_world(cfg, _world_at(pts, t1)[0], mc, rng),
                    observe_vel(cfg, _world_at(pts, t1)[1], rng)))

    return out


def m4_defect_rates(tracks_csv=None):
    """M4 的斷軌/漏偵率。有實測檔就用實測,否則回保守預設並標明來源。

    ⚠ 預設值是估計,不是量測。報告中必須標示。
    """
    if tracks_csv is None:
        return dict(m4_fragment_rate=0.05, m4_miss_rate=0.02, source="估計值(未實測)")
    import csv
    from collections import Counter
    with open(tracks_csv, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    per_track = Counter(r["track_id"] for r in rows)
    if not per_track:
        return dict(m4_fragment_rate=0.05, m4_miss_rate=0.02, source="檔案為空 → 用估計值")
    short = sum(1 for n in per_track.values() if n < 15)     # 極短軌跡 ≈ 斷軌產物
    return dict(m4_fragment_rate=short / len(per_track), m4_miss_rate=0.02,
                source=f"實測 {tracks_csv}({len(per_track)} 條軌跡)")
