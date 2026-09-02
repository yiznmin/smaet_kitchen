"""幾何式模擬世界:攝影機有固定位置與視野,**看得到誰由幾何決定**。

## 為什麼要有這個模組

`world.py`(事件式)有一個結構性缺陷:「人在哪台鏡頭」與「人在哪個位置」是
**兩條獨立演化的線**——

    cam = cams[rng.randint(len(cams))]      # 鏡頭:照拓撲連結隨機挑
    wst = advance(cfg, wst, dt, rng, ...)   # 位置:OU 過程,與鏡頭無關

後果:①動畫是誤導的(圓點位置與鏡頭標示不一致)②轉場時間是我們自己設的、
系統再用高斯去猜它,有自問自答的味道 ③沒有真正的死角。

這個模組改成**位置驅動**:每個時間步推進位置 → 幾何判定哪些鏡頭看得到 →
可見集合的變化才產生 enter/leave。於是:

  · 走進死角會從所有鏡頭消失,再出現時系統必須靠時空證據接回去
  · **轉場時間變成「走出來的」而不是設定的** —— 這是比事件式強得多的考題
  · 重疊關係由 FOV 相交自然浮現,不必人工指定

⚠ **`world.py` 一個字都不能動** —— 七輪實驗的所有數字建立在它上面。
  這個模組與它**並存**,靠 import 切換(消費腳本的 `--world` 旗標)。
  為此本模組刻意提供**同名的** `generate` / `WorldConfig` / `sample_transit` /
  `m4_defect_rates`,以及同樣的 10 欄事件 tuple 與排序規則。

⚠ **跨模組的數字不可比較** —— 不同的世界模型、不同的隨機數流。
  這是新的一條驗證軸,不是舊結果的延續。
"""
import copy
import math

import numpy as np

from m5_sim.world import (m4_defect_rates, observe, observe_vel,  # noqa: F401
                          observe_world, sample_transit, to_bbox)

NEG = -1e9


# ── 相機 ──────────────────────────────────────────────────────────────
class Camera:
    """扇形視野。⚠ **不要**用 spatiotemporal.point_in_zone 做這件事 ——
    它內部 `np.asarray(polygon, dtype=np.int32)`,是為像素座標寫的;
    公尺尺度的多邊形丟進去會被截斷成整數,FOV 判定完全錯。
    """

    def __init__(self, name, x, y, heading_deg=0.0, fov_deg=60.0, range_m=5.0):
        self.name = name
        self.x, self.y = float(x), float(y)
        self.heading_deg = float(heading_deg)
        self.fov_deg = float(fov_deg)
        self.range_m = float(range_m)

    def sees(self, p):
        dx, dy = p[0] - self.x, p[1] - self.y
        if dx * dx + dy * dy > self.range_m ** 2:
            return False
        if self.fov_deg >= 360.0:
            return True
        a = math.degrees(math.atan2(dy, dx))
        return abs((a - self.heading_deg + 180.0) % 360.0 - 180.0) <= self.fov_deg / 2.0

    def __repr__(self):
        return (f"Camera({self.name}, ({self.x:.1f},{self.y:.1f}), "
                f"朝向 {self.heading_deg:.0f}°, 視角 {self.fov_deg:.0f}°, "
                f"射程 {self.range_m:.1f}m)")


def corner_cameras(kitchen_m, fov_deg, range_m, names=("cam1", "cam2", "cam3", "cam4")):
    """四台在四角、朝向房間中心。雙臂對照就是同擺位只改 fov/range —— 只有一個變因。"""
    W, H = kitchen_m
    cx, cy = W / 2.0, H / 2.0
    out = []
    for name, (x, y) in zip(names, [(0, 0), (W, 0), (W, H), (0, H)]):
        head = math.degrees(math.atan2(cy - y, cx - x))
        out.append(Camera(name, x, y, head, fov_deg, range_m))
    return out


LAYOUTS = {
    # 10×6 m、四角朝向房間中心。實測覆蓋率(verify_world_geom.py S4 會實算):
    #   GAP  死角 30.8% / 單鏡頭 59.6% / 重疊  9.7%
    #   FULL 死角  0.0% / 單鏡頭  0.0% / 重疊 100.0%
    # 關鍵是**單鏡頭比例**:GAP 有六成區域只有一台看得到,人從那裡走到別處
    # 必須靠轉場時間證據接回;FULL 處處重疊,轉場路徑零觸發(等同 EPFL 情境)。
    # ⚠ 這兩組數字會隨 kitchen_m 改變 —— 腳本一律實算,不要引用這裡的註解。
    "GAP": dict(fov_deg=55.0, range_m=5.0),
    "FULL": dict(fov_deg=90.0, range_m=12.0),
}


def coverage_stats(cams, kitchen_m, n=300):
    """格點取樣算覆蓋率。⚠ 腳本要**實算並印出**,不要寫死上面註解的數字。"""
    W, H = kitchen_m
    xs = np.linspace(0, W, n)
    ys = np.linspace(0, H, max(2, int(n * H / W)))
    seen = np.zeros((len(ys), len(xs)), dtype=np.int16)
    for c in cams:
        for i, y in enumerate(ys):
            for j, x in enumerate(xs):
                if c.sees((x, y)):
                    seen[i, j] += 1
    tot = seen.size
    return dict(blind=float((seen == 0).sum() / tot),
                single=float((seen == 1).sum() / tot),
                multi=float((seen >= 2).sum() / tot))


# ── 世界設定 ──────────────────────────────────────────────────────────
class WorldConfig:
    """⚠ 屬性名必須與 world.WorldConfig 一致 —— sim_m5_montecarlo 會用 setattr
    掃描這些名字,render_fake_kitchen 會讀 kitchen_m / duration_s / traj_dt_s。
    真正的耦合面是**屬性名**,不是事件格式。
    """

    def __init__(self, n_chefs=4, duration_s=3600.0,
                 kitchen_m=(10.0, 6.0),
                 # ── 幾何(本模組特有)──
                 layout="GAP", cameras=None,
                 sim_dt_s=0.2,            # 位置推進步長
                 # 可見期間多久回報一次位置。⚠ 預設對齊 sim_dt_s —— 真實的 M4
                 # 每幀都輸出 tracks,m5_track_video 也是每個迴圈呼叫 on_track_update。
                 # 設成 1 秒的話系統手上的位置最多過時 0.9 公尺,而地面校正的 σ 只有
                 # 0.1 公尺 → 等於 6 個標準差,真正的那個人反而被判成「位置對不上」。
                 # 實測:心跳 1.0s → 0.2s 讓 FULL 臂的誤併從 48% 降到 21.7%。
                 heartbeat_dt_s=None,
                 min_visible_s=0.4,       # 短於這個的可見片段視為雜訊,不發事件
                 # ⚠ 離開視野的遲滯。**沒有它模擬會嚴重失真**:人在 FOV 邊緣
                 #   晃動會不斷進進出出,實測 4 人 0.2 小時跑出 1704 次轉場
                 #   (每人每 0.4 秒跨一次),誤併率因此衝到 59%。
                 #   真實的 M4 有 lost_track_buffer(configs/tracker.yaml 是 30 幀
                 #   @30fps = 1.0 秒)正是在做這件事 —— 模擬本來缺了它。
                 #   這不是為了讓數字好看而調的旋鈕,是補一個本來就該有的機制。
                 min_gone_s=1.0,
                 fragment_gap_s=0.5,      # M4 斷軌後多久才可能重新被追上
                 # ── 走動(與 world.py 同義)──
                 tau_v_s=3.0, walk_speed_mps=0.9, vel_window_s=0.5,
                 # ── M4 缺陷 ──
                 m4_fragment_rate=0.05, m4_miss_rate=0.02,
                 # ── 外觀(EPFL 實測)──
                 app_mu_same=0.490, app_sigma_same=0.10, app_mu_diff=0.465,
                 gamma_uniform=0.0,
                 # ── 地面校正 ──
                 calib_sigma_m=0.1, calib_bias_ratio=0.4,
                 # ── 畫面幾何 ──
                 body_px=200.0, frame_span_bh=6.0,
                 # ── 以下為 world.py 的相容欄位 ──
                 # ⚠ 這些在幾何世界裡**不再是輸入而是輸出** —— 轉場時間現在是
                 #   「走出來的」。保留欄位只為了讓 sim_m5_montecarlo 的 setattr
                 #   掃描不會 AttributeError;設了也不影響世界的行為。
                 transit_median_s=4.0, transit_log_sigma=0.35,
                 p_loiter=0.15, tau_loiter_s=20.0, p_detour=0.05,
                 transition_interval_s=60.0, clock_skew=None,
                 n_zones=3, q_zone=0.85, zone_error_rate=0.0,
                 master_camera=None, master_fragment_rate=0.05,
                 traj_dt_s=None, dim=64, seed=0):
        self.__dict__.update(locals())
        del self.__dict__["self"]
        # ⚠ **兩條獨立的隨機數流**。動作(位置)與觀測(缺陷/外觀/噪聲)分開,
        #   這樣同一個 seed 在不同佈局下會產生**完全相同的行走軌跡** ——
        #   兩臂的差異就只剩鏡頭幾何,是真正的配對比較。
        #   共用一條流的話,缺陷判定消耗的隨機數次數不同,軌跡會分岔,
        #   量到的差異裡混進了「走的路不一樣」這個無關變因。
        if heartbeat_dt_s is None:
            self.heartbeat_dt_s = sim_dt_s
        self.rng = np.random.RandomState(seed)            # 觀測流(相容既有介面)
        self.rng_motion = np.random.RandomState(seed ^ 0x5EED)   # 動作流
        if cameras is None:
            self.cameras = corner_cameras(kitchen_m, **LAYOUTS[layout])
            # 全景鏡頭統一成幾何的特例:360° 視角 + 無限射程。
            # 這樣它不再是獨立機制(world.py:269 要特別把它排除在輪替之外),
            # 而是從幾何自然浮現。
            if master_camera:
                W, H = kitchen_m
                self.cameras.append(
                    Camera(master_camera, W / 2, H / 2, 0.0, 360.0, 1e6))

    def camera_names(self):
        return [c.name for c in self.cameras]

    def coverage(self, n=300):
        return coverage_stats(self.cameras, self.kitchen_m, n)


def step(cfg, state, dt, rng):
    """OU 位置推進。數學與 world.step_world 相同(刻意保持一致,
    這樣兩個世界的差異只來自「鏡頭歸屬怎麼決定」這一件事)。"""
    W, H = cfg.kitchen_m
    xy, v = state
    a = math.exp(-dt / max(cfg.tau_v_s, 1e-6))
    sd = cfg.walk_speed_mps * math.sqrt(max(1 - a * a, 0.0))
    v = (v[0] * a + rng.randn() * sd, v[1] * a + rng.randn() * sd)
    x = min(max(xy[0] + v[0] * dt, 0.0), W)
    y = min(max(xy[1] + v[1] * dt, 0.0), H)
    if x in (0.0, W):
        v = (-v[0], v[1])
    if y in (0.0, H):
        v = (v[0], -v[1])
    return ((float(x), float(y)), (float(v[0]), float(v[1])))


def generate(cfg, links=None, all_cameras=None, link_zones=None, return_traj=False):
    """位置驅動的事件流。簽章與 world.generate 相同,但 links/all_cameras 被忽略
    —— 在幾何世界裡鏡頭歸屬由 cfg.cameras 的視野決定,不由拓撲決定。

    回傳 10 欄 tuple:(kind, gt, cam, t, emb, tag, box, zone, world_xy, world_v)
    排序 key=(t, update 優先) —— 心跳必須排在 enter 之前,否則位置資訊到得太晚。
    """
    rng = cfg.rng                                  # 觀測:缺陷、外觀、座標噪聲
    mrng = getattr(cfg, "rng_motion", None) or rng  # 動作:起始位置與 OU 行走
    cams = cfg.cameras
    anchors = [rng.randn(cfg.dim) for _ in range(cfg.n_chefs)]
    anchors = [a / max(np.linalg.norm(a), 1e-9) for a in anchors]

    events, traj = [], {}
    n_steps = int(cfg.duration_s / cfg.sim_dt_s)
    traj_every = (max(1, int(round((cfg.traj_dt_s or 0) / cfg.sim_dt_s)))
                  if cfg.traj_dt_s else None)

    for gt in range(cfg.n_chefs):
        st = ((float(mrng.rand() * cfg.kitchen_m[0]),
               float(mrng.rand() * cfg.kitchen_m[1])), (0.0, 0.0))
        traj[gt] = [(0.0, st)]
        # 每台鏡頭獨立的追蹤狀態:是否正在被追蹤、上次心跳時間、第幾次進場
        active = {c.name: None for c in cams}      # None 或 進場時間
        last_beat = {c.name: -1e9 for c in cams}
        pending = {c.name: None for c in cams}     # 可見但尚未確認(去抖動)
        cooldown = {c.name: -1e9 for c in cams}    # 斷軌後的冷卻,見下方說明
        gone_since = {c.name: None for c in cams}  # 離開視野多久了(遲滯用)

        for i in range(1, n_steps + 1):
            t = i * cfg.sim_dt_s
            st = step(cfg, st, cfg.sim_dt_s, mrng)
            if traj_every and i % traj_every == 0:
                traj[gt].append((t, st))
            xy, v = st

            for c in cams:
                vis = c.sees(xy)
                nm = c.name

                if vis and active[nm] is None and t >= cooldown[nm]:
                    # ── 去抖動:連續可見超過 min_visible_s 才算真的進場。
                    #    人在 FOV 邊界擦過去會製造大量一兩幀的假 track,
                    #    那是幾何造成的假象不是 M4 的問題。
                    if pending[nm] is None:
                        pending[nm] = t
                    elif t - pending[nm] >= cfg.min_visible_s:
                        pending[nm] = None
                        if rng.rand() < cfg.m4_miss_rate:
                            active[nm] = -1.0       # 漏偵:進了但沒被看到
                            continue
                        active[nm] = t
                        events.append(_enter(cfg, gt, nm, t, anchors[gt], "geom",
                                             xy, v, rng))
                        last_beat[nm] = t
                elif vis and active[nm] is not None:
                    gone_since[nm] = None          # 又看到了,取消離場倒數
                    if active[nm] > 0 and t - last_beat[nm] >= cfg.heartbeat_dt_s:
                        events.append(("update", gt, nm, t, None, "", None, None,
                                       observe_world(cfg, xy, nm, rng),
                                       observe_vel(cfg, v, rng)))
                        last_beat[nm] = t
                    # M4 斷軌:追蹤器跟丟,冷卻一段時間後**在人當時的位置**重新進場。
                    # ⚠ 第一版是「立刻在 t+0.5 補一個 enter,但用 t 時刻的位置」——
                    #   人在那 0.5 秒裡可能已經走出視野,於是產生「鏡頭看不到卻發了
                    #   進場事件」的幾何不一致(verify_world_geom.py 的 S1 抓到 4 筆)。
                    #   改成走冷卻期,讓正常的進場邏輯處理,位置自然就對了。
                    if active[nm] > 0 and rng.rand() < cfg.m4_fragment_rate * cfg.sim_dt_s:
                        events.append(_leave(cfg, gt, nm, t, xy, v, rng))
                        active[nm] = None
                        pending[nm] = None
                        cooldown[nm] = t + cfg.fragment_gap_s
                elif not vis:
                    pending[nm] = None
                    if active[nm] is not None:
                        # 遲滯:離開視野要超過 min_gone_s 才算真的離場。
                        # 對應 M4 的 lost_track_buffer —— 短暫看不到會撐住不斷軌。
                        if gone_since[nm] is None:
                            gone_since[nm] = t
                        elif t - gone_since[nm] >= cfg.min_gone_s:
                            if active[nm] > 0:
                                # ⚠ 時間戳用「**實際消失的時刻**」而不是遲滯到期的時刻。
                                #   用到期時刻的話出口時間會晚 min_gone_s,所有跨鏡頭
                                #   Δt 系統性少 1 秒 —— 而估出的 mean_s 只有 0.6 秒,
                                #   於是幾乎所有真實轉場都被判成「還沒離開就到了」而拒絕。
                                #   這正是 M4/M5 當初的教訓(出口時間戳取自 lost_track
                                #   當時而非 removed 當時),同一個坑在新世界又出現一次。
                                events.append(_leave(cfg, gt, nm, gone_since[nm],
                                                     xy, v, rng))
                            active[nm] = None
                            gone_since[nm] = None

        # 收尾:結束時仍在畫面裡的,補一個 leave
        for c in cams:
            if active[c.name] is not None and active[c.name] > 0:
                xy, v = st
                events.append(_leave(cfg, gt, c.name, cfg.duration_s, xy, v, rng))

    # ⚠ 排序規則必須與 world.py 一致:同一時刻心跳排在 enter 之前。
    events.sort(key=lambda e: (e[3], 0 if e[0] == "update" else 1))
    return (events, traj) if return_traj else events


def _enter(cfg, gt, cam, t, anchor, tag, xy, v, rng):
    return ("enter", gt, cam, t,
            observe(anchor, cfg.app_mu_same, cfg.app_sigma_same, rng),
            tag, to_bbox(cfg, (0.5, 0.5)), None,
            observe_world(cfg, xy, cam, rng), observe_vel(cfg, v, rng))


def _leave(cfg, gt, cam, t, xy, v, rng):
    return ("leave", gt, cam, t, None, "", to_bbox(cfg, (0.5, 0.5)), None,
            observe_world(cfg, xy, cam, rng), observe_vel(cfg, v, rng))


# ── 由幾何**估計**拓撲(這是最大的價值)────────────────────────────────
def estimate_topology(cfg, walk_speed_mps=None, sigma_ratio=0.35):
    """從相機幾何估出 links 與 overlapping —— **照業主實際會做的事**。

    ⚠ **絕對不可以把世界的真實轉場分布直接餵給系統**,那就回到自問自答。
      這裡走的是 scripts/build_camera_topology.py:34-35,79-80 同一套換算:
          mean_s = 兩台視野邊界之間的距離 / 0.9 m/s
          std_s  = 0.35 × mean_s
      系統拿到的是**估計值**,世界產生的是**真實值**,兩者的差距正是要量的東西。
    """
    speed = walk_speed_mps or cfg.walk_speed_mps
    cams = [c for c in cfg.cameras if c.fov_deg < 360.0]
    cent = {c.name: _coverage_centroid(c, cfg.kitchen_m) for c in cams}
    links, overlapping = [], []

    for i, a in enumerate(cams):
        for b in cams[i + 1:]:
            if _fov_overlap(a, b, cfg.kitchen_m):
                overlapping.append([a.name, b.name])
                # ⚠ **重疊也要給 link。** 兩台只在一小塊區域重疊時,人可以離開 A、
                #   走幾秒才進 B,全程沒有同時可見 —— 那時重疊路徑幫不上忙,
                #   而沒有 link 的話 transit_llr 會落到停用的 unknown_path,
                #   直接回絕。實測 cam1↔cam4 被判重疊,但真實轉場中位 3.2 秒、
                #   p90 8.0 秒,結果全部被拒。兩條路徑並存才對:
                #   同時可見走幾何、不同時可見走轉場時間。
            ca, cb = cent[a.name], cent[b.name]
            if ca is None or cb is None:
                continue
            # ⚠ 用**兩台涵蓋區域的形心距離**,不是「圓心距 − 兩個射程」。
            #   後者對朝向彼此的扇形會算出接近 0 的距離(實測估出 0.6 秒而
            #   真實中位是 1.2 秒,只有四成的轉場落在窗內)。形心距才是
            #   「人從被 A 看到走到被 B 看到」實際要走的距離。
            d = max(math.hypot(ca[0] - cb[0], ca[1] - cb[1]), 0.5)
            mean_s = round(d / speed, 1)
            std_s = round(mean_s * sigma_ratio, 1)
            links += [{"from": a.name, "to": b.name, "mean_s": mean_s, "std_s": std_s},
                      {"from": b.name, "to": a.name, "mean_s": mean_s, "std_s": std_s}]

    master = [c for c in cfg.cameras if c.fov_deg >= 360.0]
    for m in master:
        overlapping += [[m.name, c.name] for c in cams]
    return links, overlapping


def _coverage_centroid(cam, kitchen_m, n=80):
    """該鏡頭在廚房範圍內實際看得到的區域的形心(公尺)。"""
    W, H = kitchen_m
    pts = [(x, y) for x in np.linspace(0, W, n)
           for y in np.linspace(0, H, max(2, int(n * H / W))) if cam.sees((x, y))]
    if not pts:
        return None
    return (sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts))


def _fov_overlap(a, b, kitchen_m, n=120):
    """兩台的視野在廚房範圍內有沒有交集(格點取樣)。"""
    W, H = kitchen_m
    for x in np.linspace(0, W, n):
        for y in np.linspace(0, H, max(2, int(n * H / W))):
            if a.sees((x, y)) and b.sees((x, y)):
                return True
    return False


def measure_transits(events):
    """量出**真實**的轉場時間 —— 用來與 estimate_topology 的估計對照。

    這直接回答「業主用步行距離估轉場時間會差多少」,對交付套件很有用。
    回傳 {(from, to): [dt, ...]}。
    """
    from collections import defaultdict
    last_leave, out = {}, defaultdict(list)
    for kind, gt, cam, t, *_ in sorted(events, key=lambda e: e[3]):
        if kind == "leave":
            last_leave[gt] = (cam, t)
        elif kind == "enter" and gt in last_leave:
            c0, t0 = last_leave[gt]
            if c0 != cam and t > t0:
                out[(c0, cam)].append(t - t0)
    return dict(out)
