"""M5 v3:把「轉場時間」與「外觀」轉成同一尺度的對數勝算比(log-likelihood ratio)。

為什麼要換掉 v2 的加權和 `0.7·st + 0.3·app ≥ 0.35`:
  同一個 threshold 同時控制「時間窗多寬」與「外觀能不能單獨過門」,兩個需求互相
  拉扯 —— 放寬窗要壓低 thr,壓低 thr 又得同步壓低 w_app 才能維持「時空是硬門」,
  而壓低 w_app 就削弱了破平手能力。三者共用一個旋鈕,無法各自獨立滿足。
  (量化證據見 scripts/analyze_gate_capacity.py §6)

v3 的作法 —— 兩個互斥假設的證據累加:
  H1:這條新 track 就是那位剛離場的廚師
  H0:這是一位「與該廚師無關」的人(背景到達)

  LLR = log p(Δt | H1) − log λ_bg  +  log p(cos | 同人) − log p(cos | 不同人)
        └──────── 時間證據 ────────┘  └────────── 外觀證據 ──────────┘

  λ_bg = 「真正的新人」在該鏡頭出現的速率(次/秒)。這是**物理上可量測**的量,
  取代了 v2 那個沒有物理意義的魔數 0.35。

解耦效果:
  · 時間窗寬度 由轉場分布自身形狀 + λ_bg 決定
  · 外觀貢獻   由它自己的可分性決定 —— 特徵越沒鑑別力,LLR 自動越接近 0
  · 判定門檻   設在「總證據強度」上,可直接由誤併/碎裂的成本比推導
  三者各自獨立,不再互相犧牲。

外觀權重不再需要人工指定。DINOv2(同人 0.490 / 不同人 0.465,幾乎重疊)算出來的
LLR 本來就接近 0;OSNet(0.618 / 0.488)算出來就大一些。**模型有多少鑑別力,就
自動獲得多少發言權**,這正是 v2 用固定 w_app=0.3 硬塞時做不到的。
"""
import math

import numpy as np

LOG_2PI = math.log(2.0 * math.pi)
NEG_INF = -1e9          # 物理上不可能(不是 -inf,避免下游算術產生 nan)


def _log_gauss_pdf(x, mu, sigma):
    if sigma <= 0:
        return NEG_INF
    z = (x - mu) / sigma
    return -0.5 * (LOG_2PI + 2.0 * math.log(sigma) + z * z)


# ── 轉場時間模型 ──────────────────────────────────────────────────────────

class TransitModel:
    """轉場時間分布。子類實作 _logpdf,物理下限由基底統一把關。

    ⚠ 物理下限必須是**硬約束**,不能被外觀證據推翻。
      v2 靠 k_sigma 擋住荒謬的短轉場;v3 改成 LLR 後若不另外設下限,
      「離開後 0.02 秒就抵達 4 秒路程的鏡頭」會因為外觀夠像而被接受。
      人再快也不可能瞬移 —— 這種約束屬於物理,不屬於證據權衡。
    """

    #: 最快可能的轉場時間 = hard_min_ratio × μ。
    #: μ 是以廚房步速 0.9 m/s 推得,全力奔跑約 3 m/s → 最快約為 μ 的 0.3 倍。
    hard_min_ratio = 0.3

    def hard_min(self):
        return self.hard_min_ratio * getattr(self, "mu", 0.0)

    def logpdf(self, dt):
        if dt <= 0 or dt < self.hard_min():       # 物理不可能,外觀再像也不行
            return NEG_INF
        return self._logpdf(dt)

    def _logpdf(self, dt):
        raise NotImplementedError

    def describe(self):
        return self.__class__.__name__


class GaussianTransit(TransitModel):
    """單高斯。最簡單,但廚房裡人會中途停下來做事,真實分布是長尾的。

    保留它主要是為了與 v2 對照、以及在完全沒有實測資料時當起手值。
    """

    def __init__(self, mean_s, std_s, max_z=6.0, hard_min_ratio=None):
        self.mu, self.sigma, self.max_z = float(mean_s), float(std_s), float(max_z)
        if hard_min_ratio is not None:
            self.hard_min_ratio = float(hard_min_ratio)

    def _logpdf(self, dt):
        if abs(dt - self.mu) > self.max_z * self.sigma:   # 遠尾直接截斷,省算
            return NEG_INF
        return _log_gauss_pdf(dt, self.mu, self.sigma)

    def describe(self):
        return f"Gaussian(μ={self.mu:.1f}s, σ={self.sigma:.1f}s)"


class LoiterMixtureTransit(TransitModel):
    """直走 + 逗留 的混合。

    廚房不是走廊:廚師可能從 cam1 出去、在中途水槽洗個手、30 秒後才進 cam2。
    這種情況在單高斯下是遠尾 → 被判為不同人 → 碎裂。
    混合模型顯式承認它:以機率 p_loiter 額外加一段指數分布的停留時間。

        p(Δt) = (1-p)·N(μ, σ)  +  p·[N(μ, σ) ⊛ Exp(1/τ)]

    第二項用「移位指數」近似(τ >> σ 時誤差可忽略),避免做數值卷積。
    """

    def __init__(self, mean_s, std_s, p_loiter=0.15, tau_loiter_s=20.0,
                 loiter_dist="exp", loiter_log_sigma=1.0, hard_min_ratio=None):
        self.mu, self.sigma = float(mean_s), float(std_s)
        self.p = float(p_loiter)
        self.tau = float(tau_loiter_s)
        self.dist = loiter_dist            # exp | lognormal
        self.log_sigma = float(loiter_log_sigma)
        if hard_min_ratio is not None:
            self.hard_min_ratio = float(hard_min_ratio)

    def peak_density(self):
        """逗留成分能達到的最大密度。

        判定式要求 p(Δt) > exp(門檻 − (−log λ_bg));若這個峰值本身就低於那個值,
        **任何逗留時間都過不了門**,換分布形狀也救不了(重尾改善的是很久之後,
        不是峰值高度)。見 docs/M5_模擬預先登記_逗留_20260825.md §2。
        """
        if self.dist == "lognormal":
            return self.p / (self.tau * self.log_sigma * math.sqrt(2 * math.pi))
        return self.p / self.tau

    def _loiter_pdf(self, gap):
        if gap <= 0:
            return 0.0
        if self.dist == "lognormal":
            z = (math.log(gap) - math.log(self.tau)) / self.log_sigma
            return math.exp(-0.5 * z * z) / (gap * self.log_sigma * math.sqrt(2 * math.pi))
        return math.exp(-gap / self.tau) / self.tau

    def _logpdf(self, dt):
        direct = math.exp(_log_gauss_pdf(dt, self.mu, self.sigma))
        loiter = self._loiter_pdf(dt - self.mu) if dt > self.mu else 0.0
        p = (1.0 - self.p) * direct + self.p * loiter
        return math.log(p) if p > 0 else NEG_INF

    def describe(self):
        tail = (f"對數常態(中位數 {self.tau:.0f}s, logσ {self.log_sigma:.1f})"
                if self.dist == "lognormal" else f"指數(τ={self.tau:.0f}s)")
        return (f"LoiterMixture(μ={self.mu:.1f}s, σ={self.sigma:.1f}s, "
                f"p_逗留={self.p:.2f}, 尾部={tail}, 峰值密度={self.peak_density():.4f})")


class HistogramParzenTransit(TransitModel):
    """直接從實測的轉場時間樣本估分布(st-ReID 的作法)。

    這是最終要用的版本 —— 業主的影片跑過校準模式後,把觀測到的 Δt 餵進來,
    就不必再猜 μ/σ,也不必假設分布形狀。樣本不足時退回 fallback。
    """

    def __init__(self, samples, bandwidth=None, fallback=None, floor_logp=None):
        self.samples = np.asarray([s for s in samples if s > 0], dtype=float)
        self.fallback = fallback
        n = len(self.samples)
        if n >= 2 and bandwidth is None:
            # Silverman 經驗法則
            bandwidth = 1.06 * self.samples.std(ddof=1) * n ** (-0.2)
        self.h = float(bandwidth) if bandwidth else 0.0
        # 樣本外的機率下限,避免單一離群樣本把整段區間判死
        self.floor_logp = floor_logp

    @property
    def usable(self):
        return len(self.samples) >= 5 and self.h > 0

    def hard_min(self):
        return self.fallback.hard_min() if self.fallback is not None else 0.0

    def _logpdf(self, dt):
        if not self.usable:
            if self.fallback is None:
                return NEG_INF
            return self.fallback.logpdf(dt)
        z = (dt - self.samples) / self.h
        dens = np.exp(-0.5 * z * z).sum() / (len(self.samples) * self.h * math.sqrt(2 * math.pi))
        if dens <= 0:
            return self.floor_logp if self.floor_logp is not None else NEG_INF
        lp = math.log(dens)
        return max(lp, self.floor_logp) if self.floor_logp is not None else lp

    def describe(self):
        if not self.usable:
            return f"HistogramParzen(樣本不足 n={len(self.samples)} → 退回 {self.fallback})"
        return f"HistogramParzen(n={len(self.samples)}, h={self.h:.2f}s)"


class UnknownPathTransit(TransitModel):
    """走了拓撲沒建模的路徑(繞路)。

    v2/v3 原本對「沒有連結的鏡頭對」直接回 NEG_INF → 繞路的廚師 100% 開新 chef_id。
    實測顯示繞路佔全部碎裂的 25%(該情境碎裂率 90.4%)。

    但也不能無條件接受:若任何鏡頭對都同樣可信,候選數會爆炸、誤併率上升。
    所以給它「寬分布 + 負先驗」:
      · 分布寬(log_sigma 大)—— 路徑未知,時間變異本來就大
      · median 是典型直達時間的數倍 —— 繞路比直達久
      · logprior < 0 —— 有連結的路徑永遠優先於沒連結的

    ⚠ 這條路徑必然抬高誤併率,是取捨不是免費午餐。代價由模擬量化。
    """

    def __init__(self, median_s, log_sigma=0.8, logprior=-2.0, hard_min_ratio=None):
        self.mu = float(median_s)          # 供 hard_min 用
        self.median = float(median_s)
        self.log_sigma = float(log_sigma)
        self.logprior = float(logprior)
        if hard_min_ratio is not None:
            self.hard_min_ratio = float(hard_min_ratio)

    def _logpdf(self, dt):
        z = (math.log(dt) - math.log(self.median)) / self.log_sigma
        lp = -0.5 * (LOG_2PI + 2 * math.log(self.log_sigma) + z * z) - math.log(dt)
        return lp + self.logprior

    def describe(self):
        return (f"UnknownPath(median={self.median:.1f}s, logσ={self.log_sigma:.2f}, "
                f"先驗 {self.logprior:+.1f} nats)")


class SameCameraTransit(TransitModel):
    """同一台鏡頭、極短間隔重新出現 —— 幾乎一定是 M4 軌跡中斷,不是真的離開又回來。

    v2/v3 原本 `cam_from == cam_to → 拒絕`,所以 M4 斷軌後重現必定開新 chef_id。
    實測顯示這佔全部碎裂的 18%(該情境碎裂率 89.0%),而廚房遮擋極頻繁。

    分布用指數(短間隔機率高、長間隔迅速衰減),並在 max_gap_s 之後硬性截止
    —— 超過那個時間就該當成真的離場再回來,走正常轉場路徑。

    ⚠ 目前只用時間。identity_st.py 的 bbox 參數仍未接上;若接上,加入
      「斷軌前後 bbox 接近」會顯著收緊這條路徑、降低誤併。已登記為可強化點。
    """

    hard_min_ratio = 0.0                   # 同鏡頭斷軌沒有「走路距離」的物理下限

    def __init__(self, tau_break_s=2.0, max_gap_s=15.0):
        self.tau = float(tau_break_s)
        self.max_gap = float(max_gap_s)

    def _logpdf(self, dt):
        if dt > self.max_gap:              # 太久 → 不是斷軌,交給正常轉場路徑判斷
            return NEG_INF
        return -dt / self.tau - math.log(self.tau)

    def describe(self):
        return f"SameCamera(τ={self.tau:.1f}s, 上限 {self.max_gap:.0f}s)"


def make_transit(mean_s, std_s, kind="loiter", **kw):
    """依 config 的 `transit_model` 欄位建轉場模型。"""
    if kind == "gaussian":
        return GaussianTransit(mean_s, std_s, **kw)
    if kind == "loiter":
        return LoiterMixtureTransit(mean_s, std_s, **kw)
    raise ValueError(f"未知的 transit_model: {kind}(可用:gaussian / loiter)")


# ── 外觀證據 ──────────────────────────────────────────────────────────────

class AppearanceLR:
    """外觀相似度的對數概似比:log p(cos|同人) − log p(cos|不同人)。

    兩個分布直接用實測的 cross-view 統計擬合,不需要人工指定權重。
    鑑別力越差,兩個分布越重疊,LLR 越接近 0 → 該特徵自動失去發言權。

    ⚠ 這些統計來自 EPFL(每個場次一人 → 「身份」等於「錄影場次」),同一人的
      crop 全部同衣服同光照,所以 same 的平均偏高、可分性偏樂觀。實際部署要用
      業主現場資料重估。詳見 docs/M5_可行性驗證與模型選型.md。
    """

    # 實測值。⚠ **兩組來源、兩種難度,不可直接比大小**:
    #
    #  · dinov2 / osnet —— EPFL 6 身份 / 449 crops,見 reid_epfl_*.json。
    #    EPFL 有 session confound(一個場次一人 → 同一人所有 crop 同衣服同光照),
    #    而且九台鏡頭**同步且重疊** → 同一時刻不同視角。數字偏樂觀。
    #
    #  · chirla_* —— CHIRLA benchmark 的 **multi_camera** scenario,
    #    gallery×query 的**跨實體相機**配對(closed-set,排除負號 distractor),
    #    18,121 組同人 / 152,659 組不同人。7 台分佈在不同房間的鏡頭,接近 M5 的部署情境。
    #    ⚠ 論文寫「非重疊」但實測不成立(cam2/cam3 同一個房間),所以這組分布
    #      **混合了重疊與非重疊兩種路徑**,比純轉場路徑樂觀。
    #      見 docs/CHIRLA_鏡頭佈局實測_20260903.md。
    #    由 scripts/calib_appearance_chirla.py 產生(2026-09-03),
    #    原始資料在 results/m5_reid/appearance_calib_chirla.json。
    #
    # ⚠ 每一組的 sigma_same 與 sigma_diff **刻意相同(合併 σ)**,不是懶得分開量。
    #   兩個 σ 不等時高斯 LLR 變成二次式,在遠離平均處會外插出幾十 nats
    #   (判定門檻才 1.61),而且 σ_same > σ_diff 時 llr(mu_same) 會變成負的 ——
    #   「剛好等於同人平均的 cosine 反而是不同人的證據」,自相矛盾。
    #   合併 σ 下 llr(mu_same) = d'^2 / 2,乾淨且與世界端的單一 σ 同步。
    #
    # 括號內是 d' = (mu_same - mu_diff) / sigma,**跨資料集比較請只看它**。
    MEASURED = {
        "dinov2": dict(mu_same=0.490, mu_diff=0.465, sigma_same=0.10, sigma_diff=0.10),      # d'=0.25
        "osnet": dict(mu_same=0.618, mu_diff=0.488, sigma_same=0.12, sigma_diff=0.12),       # d'=1.08
        # 可出貨(CC-BY-4.0 資料 + ImageNet 權重)
        "chirla_armS": dict(mu_same=0.3989, mu_diff=0.3769,
                            sigma_same=0.1865, sigma_diff=0.1865),                            # d'=0.118
        "chirla_armS0": dict(mu_same=0.7682, mu_diff=0.7599,
                             sigma_same=0.0634, sigma_diff=0.0634),                           # d'=0.131
        # ⚠ 研究限定(起始權重訓練於 Market-1501),對照用,不可出貨
        "chirla_armR": dict(mu_same=0.7398, mu_diff=0.6833,
                            sigma_same=0.1342, sigma_diff=0.1342),                            # d'=0.421
    }

    def __init__(self, mu_same, sigma_same, mu_diff, sigma_diff, clip=None):
        self.mu_same, self.sigma_same = float(mu_same), float(sigma_same)
        self.mu_diff, self.sigma_diff = float(mu_diff), float(sigma_diff)
        # 夾住 LLR 上下限:外觀是輔助證據,不該單獨壓垮時空證據
        self.clip = float(clip) if clip is not None else None

    @classmethod
    def measured(cls, embedder="dinov2", clip=None):
        if embedder not in cls.MEASURED:
            raise ValueError(f"沒有 {embedder} 的實測分布(可用:{list(cls.MEASURED)})")
        return cls(clip=clip, **cls.MEASURED[embedder])

    @classmethod
    def uninformative(cls):
        """完全沒有鑑別力(兩個分布相同)→ LLR 恆為 0。用於「純時空」消融。"""
        return cls(mu_same=0.5, sigma_same=0.1, mu_diff=0.5, sigma_diff=0.1)

    def llr(self, cos):
        v = (_log_gauss_pdf(cos, self.mu_same, self.sigma_same)
             - _log_gauss_pdf(cos, self.mu_diff, self.sigma_diff))
        if self.clip is not None:
            v = max(-self.clip, min(self.clip, v))
        return v

    def max_abs_llr(self, lo=0.0, hi=1.0, n=401):
        """外觀在**實際會出現的** cosine 範圍內能提供的最大證據量(nats)。

        這個數字就是「這個 embedder 的發言權上限」,可直接與時間證據比大小
        (時間證據峰值約 5 nats,見 scripts/analyze_gate_capacity.py)。

        ⚠ 範圍預設 [0,1] 而非 [-1,1]:L2 正規化的深度特徵之間 cosine 幾乎不會
          到負值,而高斯尾部在 cos=-1 處會給出巨大但永遠用不到的 LLR,把它算進來
          會嚴重高估這個特徵的實際發言權。
        """
        return max(abs(self.llr(c)) for c in np.linspace(lo, hi, n))

    def describe(self):
        return (f"AppearanceLR(同人 {self.mu_same:.3f}±{self.sigma_same:.2f}, "
                f"不同人 {self.mu_diff:.3f}±{self.sigma_diff:.2f}, "
                f"最大證據 {self.max_abs_llr():.2f} nats)")


# ── 位置證據(僅限同一台鏡頭)────────────────────────────────────────────

class PositionLR:
    """同鏡頭前後兩次觀測的位移證據:log p(位移|同一人) − log p(位移|不同人)。

    ⚠ **只在同一台鏡頭內有意義。** 跨鏡頭的影像座標是不同的座標系,沒有外參
      校正就不可比 —— 硬比會得到隨機結果。所以本類只用在 M4 斷軌的重關聯上。

    為什麼需要它:實測(2026-08-25 預先登記那輪)顯示,同鏡頭重關聯**只用時間**
    會讓誤併率翻倍(4.8%→10.2%)—— 因為同一台鏡頭裡有多位廚師時,時間分不出
    斷軌的片段屬於誰。位置可以。

    模型(尺度不變,用「人身高」當單位,所以不需要知道實際公尺數):
      · 同一人:短暫斷軌期間位移小。速度約 0.5 身高/秒(廚房步速 0.9 m/s ÷ 1.7 m),
        位移向量 ~ 各向同性高斯,σ = speed·Δt + 量測雜訊
      · 不同人:進場位置與離場位置無關 → 大致均勻散布在畫面上,密度 = 1/畫面面積

      LLR = log A − log(2πσ²) − d²/(2σ²)      d = 位移 / 平均身高,A = 畫面面積(身高²)
    """

    def __init__(self, speed_bh_per_s=0.5, noise_bh=0.3, frame_span_bh=6.0, clip=8.0,
                 uniform_mix=0.05):
        self.speed = float(speed_bh_per_s)
        self.noise = float(noise_bh)
        self.span = float(frame_span_bh)
        self.area = self.span ** 2                 # 畫面面積,以身高² 為單位
        self.clip = float(clip)
        # 「同一人」的位置分布不可能比「均勻散布在畫面上」更分散 —— 人再怎麼走
        # 也走不出畫面。σ 若無上限地隨 Δt 成長,長間隔時它的密度會低於均勻分布,
        # 反而給出莫名的負證據。用畫面的均勻分布標準差當上限。
        self.sigma_max = self.span / math.sqrt(12.0)
        # 少量均勻成分:偵測跳框、被完全遮擋後從別處出現等。同時把 LLR 的下界
        # 夾在 log(uniform_mix),避免單一離群位置給出無限大的反證。
        self.eps = float(uniform_mix)

    @staticmethod
    def _foot_and_height(bbox):
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2.0, float(y2)), max(abs(y2 - y1), 1e-6)

    def llr(self, bbox_exit, bbox_enter, dt):
        """兩個 bbox 皆為同一鏡頭的 (x1,y1,x2,y2)。任一為 None 則回 0(無證據)。"""
        if bbox_exit is None or bbox_enter is None:
            return 0.0
        (ex, ey), he = self._foot_and_height(bbox_exit)
        (nx, ny), hn = self._foot_and_height(bbox_enter)
        h = 0.5 * (he + hn)
        d = math.hypot(nx - ex, ny - ey) / h        # 以身高為單位的位移
        sigma = min(self.speed * max(dt, 0.0) + self.noise, self.sigma_max)
        gauss = math.exp(-d * d / (2 * sigma * sigma)) / (2 * math.pi * sigma * sigma)
        p_same = (1.0 - self.eps) * gauss + self.eps / self.area
        p_diff = 1.0 / self.area
        v = math.log(p_same / p_diff)
        return max(-self.clip, min(self.clip, v))

    def describe(self):
        return (f"PositionLR(速度 {self.speed:.2f} 身高/秒, 雜訊 {self.noise:.2f} 身高, "
                f"畫面 {math.sqrt(self.area):.1f}×{math.sqrt(self.area):.1f} 身高, "
                f"夾 ±{self.clip:.0f} nats)")


# ── 地面平面證據(跨鏡頭,需 homography 校正)──────────────────────────────

class GroundPlaneLR:
    """兩台鏡頭推得的**世界座標**距離,對數概似比。

    這是唯一能回答「重疊視野裡的是**哪一個**人」的證據。
    在它之前,重疊路徑只能說「那台鏡頭裡有人」,於是 K 位廚師同時在場時
    只能亂猜 —— 第五輪實測誤併 6.8% 就是這麼來的。

        LLR = log A − log(2πσ²) − d²/(2σ²)
        d = 兩台鏡頭推得的世界座標距離(公尺)
        σ = 標定殘差 + 腳點偵測誤差(公尺)
        A = 重疊區面積(m²),「不同人」時位置大致均勻散布的範圍

    與同鏡頭的 PositionLR 是同一套數學,差別只在單位:
    PositionLR 用「人身高」(因為沒有校正,只能求尺度不變);
    本類用真實公尺(因為有 homography)。

    ⚠ σ 必須遠小於「兩人之間的典型距離」(廚房約 1~2 m),否則分不出誰是誰。
      A=30 m² 時:σ=0.2 給 +4.8/−23.3 nats(極強);σ=1.5 給 +0.75/+0.25(幾乎無)。

    ⚠ 它取代 `overlap_llr = 5.0` 這個常數 —— 那是編的,性質同 v2 那個
      被批評的魔數 0.35。這是架構裡最後一個沒有物理依據的參數。
    """

    def __init__(self, sigma_m=0.4, area_m2=30.0, clip=8.0, speed_mps=0.9,
                 pairwise=True):
        # ⚠ sigma_m 是**單台鏡頭**的標定誤差。我們比的是兩台鏡頭的觀測**差值**,
        #   其標準差是 √2 倍。用單台的 σ 會高估鑑別力 → 真的同一人被判成
        #   「位置對不上」。實測時真的那位只拿到 +0.20 nats(應該 +2.5)。
        self.sigma = float(sigma_m) * (math.sqrt(2.0) if pairwise else 1.0)
        self.speed = float(speed_mps)      # 兩次觀測之間人可能走多遠
        self.area = float(area_m2)
        self.clip = float(clip)
        # 與 PositionLR 同樣的物理約束:「同一人」的分布不可能比「均勻散布在
        # 重疊區」更分散,否則長距離時會給出莫名的負證據。
        self.sigma_max = math.sqrt(self.area / 12.0)

    def llr(self, xy_a, xy_b, dt=0.0):
        """兩個世界座標 (x, y),單位公尺。dt = 兩次觀測相隔幾秒。

        dt > 0 時要把「這段時間人可能走多遠」算進不確定度,否則會拿舊位置
        跟新位置比,把真正的同一人判成位置對不上。dt 大到一定程度後
        σ 會被 sigma_max 夾住 → 證據自然趨近 0(我們確實什麼都不知道)。
        """
        if xy_a is None or xy_b is None:
            return 0.0
        d = math.hypot(xy_a[0] - xy_b[0], xy_a[1] - xy_b[1])
        s = min(math.hypot(self.sigma, self.speed * max(dt, 0.0)), self.sigma_max)
        v = math.log(self.area) - math.log(2 * math.pi * s * s) - d * d / (2 * s * s)
        return max(-self.clip, min(self.clip, v))

    def describe(self):
        return (f"GroundPlaneLR(σ={self.sigma:.2f}m, 重疊區 {self.area:.0f}m², "
                f"同人 {self.llr((0,0),(0,0)):+.2f} / 相距1.5m {self.llr((0,0),(1.5,0)):+.2f} nats)")


class CrossViewLR:
    """跨鏡頭腳點一致性 —— `GroundPlaneLR` 的**無標定**版本。

    ## 為什麼需要它

    `GroundPlaneLR` 是唯一能回答「重疊視野裡的是**哪一個**人」的證據,但它要
    homography 才能算世界座標。CHIRLA 沒有相機標定參數 → 退回 `overlap_llr = 5.0`
    這個常數,而那個常數**在數學上不可能拒絕任何候選**:

        5.0 − log(k) + app_llr < 1.609  →  需要 k > 29.7
        而 CHIRLA 單台鏡頭同時最多 9 人

    2026-09-04 實測的後果:重疊路徑 `p_break = 2.71%`(幾乎從不拒絕)、
    `p_false_merge = 80.80%`、`p_correct = 16.49%`(≈ 1/6,與隨機挑一個無法區分)。

    ## 關鍵觀察:沒有標定,但**有真值身份**就能估出單應性

    同一時刻兩台看到同一個人 → 那就是一組地面對應點(取 bbox 底邊中點)。
    夠多組就能用 RANSAC 解出 **H_AB**(A 的像素平面 → B 的像素平面)。

    ⚠ 這**不是**度量標定 —— 沒有公尺、沒有外參。它只回答一個問題:
      「這兩個觀測落在地面的同一點嗎?」而那正是我們要的。

    ⚠ **逐對估,不串接。** H_1→R = H_1→3 · H_3→R 會累積誤差,而 CHIRLA 有些
      鏡頭對的共現只有 3 次,串進來會污染整條鏈。逐對估的代價是沒有全域座標,
      好處是每一對帶著自己實測的 σ,品質差的那對自己弱。

        LLR = log A − log(2πσ²) − d²/(2σ²)

        d = 把 A 的腳點經 H_AB 投到 B 之後,與 B 的腳點的殘差(B 的像素)
        σ = 該鏡頭對的**實測**殘差尺度(像素)
        A = 該對的共視區面積(B 的像素²),「不同人」時腳點大致散布的範圍

    ⚠⚠ **σ 不可以再乘 √2。** 它是直接從「真正同一人」的殘差分布量出來的,
      **已經包含兩台鏡頭的誤差**。GroundPlaneLR 的 σ 是單台標定誤差所以要乘,
      這裡不是。本專案已經在 `GroundPlaneLR`(第六輪)與 `VelocityLR`(第七輪)
      各踩過一次這個坑,不要有第三次。
    """

    def __init__(self, H, sigma_px, area_px2, clip=8.0, speed_px_per_s=0.0):
        import numpy as _np
        self.H = _np.asarray(H, dtype=float).reshape(3, 3)
        # ⚠ 見類別說明:σ 已是 pairwise 殘差,不再乘 √2。
        self.sigma = float(sigma_px)
        self.area = float(area_px2)
        self.clip = float(clip)
        # 兩次觀測相隔 dt 時人可能移動多少(B 的像素/秒)。重疊路徑的 dt 很小
        # (同一時刻),所以預設 0;留著是為了與 GroundPlaneLR 的介面一致。
        self.speed = float(speed_px_per_s)
        # 與 GroundPlaneLR 同樣的物理約束:「同一人」的分布不可能比「均勻散布在
        # 共視區」更分散,否則遠距離時會給出莫名的負證據。
        self.sigma_max = math.sqrt(self.area / 12.0)

    @staticmethod
    def foot(bbox):
        """bbox 底邊中點 —— 站立的人與地面的接觸點,是無標定下最好的地面代理。"""
        if bbox is None:
            return None
        x1, y1, x2, y2 = (float(v) for v in bbox[:4])
        return ((x1 + x2) * 0.5, y2)

    def project(self, xy):
        """把 A 平面的點經 H 投到 B 平面。齊次除法退化時回 None(而不是爆掉)。"""
        if xy is None:
            return None
        h = self.H
        w = h[2, 0] * xy[0] + h[2, 1] * xy[1] + h[2, 2]
        if abs(w) < 1e-9:                       # 點落在消失線上 → 無法投影
            return None
        return ((h[0, 0] * xy[0] + h[0, 1] * xy[1] + h[0, 2]) / w,
                (h[1, 0] * xy[0] + h[1, 1] * xy[1] + h[1, 2]) / w)

    def llr(self, bbox_a, bbox_b, dt=0.0):
        """bbox_a 在鏡頭 A、bbox_b 在鏡頭 B(H 的方向是 A→B)。

        資料不足時回 **0.0**(中性)而不是負值 —— 「我們算不出來」與
        「證據說不是同一人」是兩回事,混淆它們會把缺資料變成反證。
        """
        pa, pb = self.foot(bbox_a), self.foot(bbox_b)
        if pa is None or pb is None:
            return 0.0
        q = self.project(pa)
        if q is None:
            return 0.0
        d = math.hypot(q[0] - pb[0], q[1] - pb[1])
        s = min(math.hypot(self.sigma, self.speed * max(dt, 0.0)), self.sigma_max)
        v = math.log(self.area) - math.log(2 * math.pi * s * s) - d * d / (2 * s * s)
        return max(-self.clip, min(self.clip, v))

    def describe(self):
        return (f"CrossViewLR(σ={self.sigma:.1f}px, 共視區 {self.area:.0f}px², "
                f"殘差0 {self.llr_at(0.0):+.2f} / 殘差{self.sigma*3:.0f}px "
                f"{self.llr_at(self.sigma * 3):+.2f} nats)")

    def llr_at(self, d):
        """給定殘差直接算 LLR —— 供校準與診斷用,不經過投影。"""
        s = min(self.sigma, self.sigma_max)
        v = math.log(self.area) - math.log(2 * math.pi * s * s) - d * d / (2 * s * s)
        return max(-self.clip, min(self.clip, v))


class TransitPlaceLR:
    """轉場的**出入口位置**證據 —— 「你是從通往這裡的那個門出去的嗎?」

    ## 為什麼需要它(2026-09-05 的量化診斷)

    轉場路徑上唯一的實質證據是轉場時間:

        transit_llr = log p(Δt | 同一人) − log λ_bg

    CHIRLA 拓撲實測 `λ_bg = 0.0482`(每 20.7 秒一位新人,誠實估自推導集),
    所以要過門檻 1.609 需要 **pdf(Δt) > 0.24 /秒**。而實際擬合出來的分布:

        camera_6→camera_7   峰值 pdf 0.70  → 只有 Δt∈[0.20, 1.30]s 過得了
        camera_2→camera_6   峰值 pdf 0.198 → **任何 Δt 都過不了**(差一點點)
        camera_7→camera_5   峰值 pdf 0.05  → **任何 Δt 都過不了**
        unknown_path        最佳 −1.33     → **任何 Δt 都過不了**

    **轉場路徑不是「證據弱」,是對多數鏡頭對而言數學上不可能綁定。**
    實測後果:轉場路徑碎裂 54.42% —— 拒掉近半的**真**配對。
    而外觀只能加 ±0.03 nats,補不上這個缺口。

    ## 這條證據的資訊從哪來

    人不是從房間的任意位置消失、再從任意位置出現的 —— 他們**走門**。
    同一條連結 A→B 的真實轉場,離開 A 的腳點會聚在「通往 B 的那個門」,
    進入 B 的腳點會聚在對應的入口。兩者都可從推導集的真值學出來。

        LLR = [log N(退場點; μ_exit, Σ_exit) + log A_from]
            + [log N(入場點; μ_enter, Σ_enter) + log A_to]

    ⚠ **只有退場項能區分候選。** 入場項只取決於這條新 track 自己的位置,
      對所有候選都一樣 → 它影響「綁不綁」但不影響「綁哪一個」。
      這一點必須講清楚,不然會高估它的鑑別力。
      真正在挑人的是退場項:**從通往這裡的門出去的那位,才是他**。

    ⚠ Σ 要用**留出樣本**估,與 `CrossViewLR` 同一個陷阱 —— 用擬合殘差會低估,
      使 LLR 過度自信而把真正的同一人判成「不是從那個門出去的」。

    ⚠ 這是 `DirectionLR` 想做的事的連續版本。DirectionLR 用 n_zones=3 的離散
      分區,粒度太粗且一直是關閉的;這裡直接用位置的二維分布。
      **兩者不應同時開啟** —— 會把同一份資訊算兩次。
    """

    def __init__(self, mu_exit, cov_exit, area_from, mu_enter, cov_enter, area_to,
                 clip=6.0, p_offdoor=0.10):
        import numpy as _np
        # ⚠ p_offdoor:「這次沒走常走的那個門」的比例。沒有它的話,純高斯在
        #   600px 外會給 −90 nats(σ=45px 時 (600/45)²/2),只能靠 clip 硬夾 ——
        #   那讓這條證據變成硬性閘門,一個繞路回來的**真**同一人會被直接否決。
        #   混合一個均勻成分後,下界自然是 log(ε) ≈ −2.3,而不是負無窮。
        #   這與 transit_model="loiter" 的 p_loiter 是同一個慣用法。
        self.eps = min(max(float(p_offdoor), 1e-4), 0.5)
        self.mu_e = _np.asarray(mu_exit, dtype=float).reshape(2)
        self.mu_n = _np.asarray(mu_enter, dtype=float).reshape(2)
        # 正則化:資料少時共變異數可能接近奇異,加一點對角項。
        # 3px 是腳點量化誤差的下限,與 chirla_build_crossview 的 σ 下限同源。
        self.ic_e, self.ld_e = self._inv(cov_exit)
        self.ic_n, self.ld_n = self._inv(cov_enter)
        self.log_area_from = math.log(max(float(area_from), 1.0))
        self.log_area_to = math.log(max(float(area_to), 1.0))
        self.clip = float(clip)

    @staticmethod
    def _inv(cov):
        import numpy as _np
        c = _np.asarray(cov, dtype=float).reshape(2, 2) + _np.eye(2) * 9.0
        return _np.linalg.inv(c), float(_np.log(_np.linalg.det(c)))

    @staticmethod
    def foot(bbox):
        if bbox is None:
            return None
        x1, _y1, x2, y2 = (float(v) for v in bbox[:4])
        return ((x1 + x2) * 0.5, y2)

    def _term(self, p, mu, ic, logdet, log_area):
        """log[(1−ε)·N(p;μ,Σ) + ε/A] − log(1/A) = log[(1−ε)·A·N(p) + ε]

        遠離門口時 N→0,整項 → log(ε):**有界**,不需要靠 clip 救。
        """
        import numpy as _np
        d = _np.asarray(p, dtype=float) - mu
        m = float(d @ ic @ d)
        log_n = -0.5 * logdet - math.log(2 * math.pi) - 0.5 * m     # log N(p)
        # log-sum-exp:log[(1−ε)·exp(log_n + log_area) + ε]
        a = log_n + log_area + math.log(1.0 - self.eps)
        b = math.log(self.eps)
        hi = max(a, b)
        return hi + math.log(math.exp(a - hi) + math.exp(b - hi))

    def llr(self, exit_bbox, enter_bbox):
        """exit_bbox 在 cam_from、enter_bbox 在 cam_to。缺資料回中性 0.0。

        ⚠ 回 0.0 而不是負值 —— 「我們算不出來」與「證據說不是」是兩回事。
        """
        pe, pn = self.foot(exit_bbox), self.foot(enter_bbox)
        v = 0.0
        if pe is not None:
            v += self._term(pe, self.mu_e, self.ic_e, self.ld_e, self.log_area_from)
        if pn is not None:
            v += self._term(pn, self.mu_n, self.ic_n, self.ld_n, self.log_area_to)
        return max(-self.clip, min(self.clip, v))

    def describe(self):
        return (f"TransitPlaceLR(退場 μ=({self.mu_e[0]:.0f},{self.mu_e[1]:.0f}), "
                f"入場 μ=({self.mu_n[0]:.0f},{self.mu_n[1]:.0f}), clip={self.clip})")


class VelocityLR:
    """速度證據 —— 專門解「兩個人站在同一個位置」的情況。

    第六輪量化:剩下的誤併有 **76% 發生在兩人相距 < 0.5 m**(中位數 0.11 m)。
    位置在 σ=0.05m 的近乎完美校正下仍分不出 —— 那是物理極限。

    但那兩個人**是從不同方向走過來的**。速度與位置正交,而且全景鏡頭
    本來就在連續追蹤,軌跡資料是現成的。

        LLR = log V − log(2πσ_v²) − |Δv|² / (2σ_v²)

        Δv  = 兩者速度向量之差(世界座標,m/s)
        σ_v = 速度估計噪聲 ≈ σ_pos·√2 / 觀測窗
        V   = 「不同人」時速度差的散布範圍 ≈ π·(最大步速)²

    ⚠ 這是「延遲換精度」:觀測窗越長 σ_v 越小、證據越強,但綁定要等越久。
      0.25s 只有 1.26 nats(低於門檻);0.5s 有 2.64 nats。
      系統是事後查詢用的,所以延遲 0.5~1 秒可接受。
    ⚠ 速度由地面校正後的世界座標差分而來,**繼承校正誤差**。
    """

    def __init__(self, sigma_pos_m=0.1, window_s=0.5, max_speed_mps=1.5, clip=6.0):
        # 單次速度估計的噪聲:位置差分 → σ_pos·√2 / 窗長
        single = sigma_pos_m * math.sqrt(2.0) / max(window_s, 1e-6)
        # ⚠ 我們比的是**兩個**速度觀測的差,所以再乘 √2。
        #   第六輪的 GroundPlaneLR 犯過同一個錯(用單次觀測的 σ 去比差值),
        #   這裡又犯一次 —— 記錄下來:**凡是比較兩個有噪聲的量,
        #   差值的標準差都是單次的 √2 倍**。少了它會系統性高估鑑別力,
        #   結果把真的同一人判成「對不上」。
        self.sigma_v = single * math.sqrt(2.0)
        self.area = math.pi * float(max_speed_mps) ** 2
        self.clip = float(clip)
        # 同 PositionLR / GroundPlaneLR 的物理約束:「同一人」的速度分布
        # 不可能比「均勻散布在所有合理速度」更分散
        self.sigma_max = math.sqrt(self.area / 12.0)

    def llr(self, v_a, v_b):
        if v_a is None or v_b is None:
            return 0.0
        d = math.hypot(v_a[0] - v_b[0], v_a[1] - v_b[1])
        s = min(self.sigma_v, self.sigma_max)
        v = math.log(self.area) - math.log(2 * math.pi * s * s) - d * d / (2 * s * s)
        return max(-self.clip, min(self.clip, v))

    def describe(self):
        return (f"VelocityLR(σ_v={self.sigma_v:.2f} m/s, 上限 "
                f"{self.llr((0,0),(0,0)):+.2f} / 差 1 m/s 時 {self.llr((0,0),(1,0)):+.2f} nats)")


# ── 方向證據(出入口 zone)────────────────────────────────────────────────

class DirectionLR:
    """走對門沒有?出入口 zone 的對數概似比。

    這是 CLM 四線索(拓撲、轉場時間、移動方向、外觀)裡最後一個被實作的。
    前兩輪失敗的共同模式是:所有旋鈕都沿著**時間軸**放寬,而「逗留的人」與
    「不相干的人剛好出現」在時間軸上就是重疊的 → 必然零和交換。

    方向與 Δt 正交:
      · 走對門的人 **不管走了多久** 都更可能是同一人 → 對逗留者提供固定證據
      · 走錯門的人 **時間再吻合也拿到負證據** → 壓低誤併

        LLR = log[P(觀測 zone | 走這條連結) / P(該 zone | 背景)]   (出口 + 入口各一)
        P(符合 | 走這條連結) = q,其餘 zone 均分 (1−q)
        P(任一 zone | 背景)  = 1/n_zones

    ⚠ 模擬用離散 zone 標籤;真實部署要從 bbox 經 which_zone() 判定,
      會多一層幾何誤差 → 模擬結果是**樂觀上界**。
    """

    def __init__(self, q=0.85, n_zones=3, clip=6.0):
        self.q = float(q)
        self.n = int(n_zones)
        self.clip = float(clip)

    def _endpoint_llr(self, observed, expected):
        if observed is None or expected is None:
            return 0.0                      # 沒標 zone → 不提供證據(不是負證據)
        if observed == expected:
            p = self.q
        else:
            p = (1.0 - self.q) / max(self.n - 1, 1)
        if p <= 0:
            return -self.clip
        return math.log(p * self.n)         # 除以背景機率 1/n

    def llr(self, exit_zone, expected_exit, enter_zone, expected_enter):
        v = (self._endpoint_llr(exit_zone, expected_exit)
             + self._endpoint_llr(enter_zone, expected_enter))
        return max(-self.clip, min(self.clip, v))

    def max_llr(self):
        """兩端都走對時的證據量(nats)。與判定門檻比大小就知道夠不夠。"""
        return 2 * math.log(self.q * self.n)

    def describe(self):
        return (f"DirectionLR(q={self.q:.2f}, {self.n} 個 zone, "
                f"走對兩端 +{self.max_llr():.2f} nats, "
                f"走錯一端 {self._endpoint_llr('a', 'b'):+.2f} nats)")


# ── 決策門檻 ──────────────────────────────────────────────────────────────

def decision_threshold(cost_false_merge_over_break=5.0, prior_odds=1.0):
    """把「誤併比碎裂嚴重幾倍」直接轉成 LLR 門檻(nats)。

    誤併(把兩人綁成一個 chef_id)會替沒洗手的人偽造一筆洗手紀錄 → 靜默漏報,
    下游 M7 只驗物件重疊、不驗身份,M8 直接寫進 DB,沒有任何一關能發現。
    碎裂(同一人被拆成兩個 chef_id)則可被「chef_id 數 > 排班人數」自動偵測。
    這個不對稱就是門檻要往保守側偏的理由,也是這個參數的物理意義。
    """
    return math.log(cost_false_merge_over_break) - math.log(prior_odds)
