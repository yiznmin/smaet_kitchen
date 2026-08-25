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


def observe(anchor, mu_same, sigma_same, rng):
    """從錨點抽一個觀測向量,cosine 服從 N(mu_same, sigma_same)。"""
    a = float(np.clip(rng.normal(mu_same, sigma_same), -0.99, 0.99))
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


def generate(cfg, links, all_cameras):
    """產生事件流。

    回傳 [(kind, gt_id, camera, t_observed, embedding, tag)],kind ∈ {enter, leave}。
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
    cams = sorted(all_cameras)
    out_links = {}
    for (a, b) in links:
        out_links.setdefault(a, []).append(b)

    events = []
    for chef in chefs:
        cam = cams[rng.randint(len(cams))]
        t = rng.rand() * cfg.transition_interval_s
        events.append(("enter", chef.gt_id, cam, t + skew.get(cam, 0.0),
                       observe(chef.anchor, cfg.app_mu_same, cfg.app_sigma_same, rng), "first"))
        while t < cfg.duration_s:
            stay = rng.exponential(cfg.transition_interval_s)
            t_leave = t + stay
            if t_leave > cfg.duration_s:
                break
            events.append(("leave", chef.gt_id, cam, t_leave + skew.get(cam, 0.0), None, ""))

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
            t = t_leave + dt
            if t > cfg.duration_s:
                break
            if rng.rand() < cfg.m4_miss_rate:           # M4 漏偵 → 這次抵達沒被看到
                cam = dest
                continue
            cam = dest
            events.append(("enter", chef.gt_id, cam, t + skew.get(cam, 0.0),
                           observe(chef.anchor, cfg.app_mu_same, cfg.app_sigma_same, rng), tag))
            if rng.rand() < cfg.m4_fragment_rate:       # M4 斷軌 → 同鏡頭再開一個 track
                t_frag = t + rng.exponential(5.0)
                events.append(("leave", chef.gt_id, cam, t_frag + skew.get(cam, 0.0), None, ""))
                events.append(("enter", chef.gt_id, cam, t_frag + 0.5 + skew.get(cam, 0.0),
                               observe(chef.anchor, cfg.app_mu_same, cfg.app_sigma_same, rng),
                               "fragment"))
                t = t_frag + 0.5
    events.sort(key=lambda e: e[3])
    return events


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
