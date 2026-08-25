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

    def __init__(self, mean_s, std_s, p_loiter=0.15, tau_loiter_s=20.0, hard_min_ratio=None):
        self.mu, self.sigma = float(mean_s), float(std_s)
        self.p = float(p_loiter)
        self.tau = float(tau_loiter_s)
        if hard_min_ratio is not None:
            self.hard_min_ratio = float(hard_min_ratio)

    def _logpdf(self, dt):
        direct = math.exp(_log_gauss_pdf(dt, self.mu, self.sigma))
        if dt > self.mu:
            loiter = math.exp(-(dt - self.mu) / self.tau) / self.tau
        else:
            loiter = 0.0
        p = (1.0 - self.p) * direct + self.p * loiter
        return math.log(p) if p > 0 else NEG_INF

    def describe(self):
        return (f"LoiterMixture(μ={self.mu:.1f}s, σ={self.sigma:.1f}s, "
                f"p_逗留={self.p:.2f}, τ={self.tau:.0f}s)")


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

    # 實測值(EPFL 6 身份 / 449 crops),見 reid_epfl_*.json
    MEASURED = {
        "dinov2": dict(mu_same=0.490, mu_diff=0.465, sigma_same=0.10, sigma_diff=0.10),
        "osnet": dict(mu_same=0.618, mu_diff=0.488, sigma_same=0.12, sigma_diff=0.12),
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


# ── 決策門檻 ──────────────────────────────────────────────────────────────

def decision_threshold(cost_false_merge_over_break=5.0, prior_odds=1.0):
    """把「誤併比碎裂嚴重幾倍」直接轉成 LLR 門檻(nats)。

    誤併(把兩人綁成一個 chef_id)會替沒洗手的人偽造一筆洗手紀錄 → 靜默漏報,
    下游 M7 只驗物件重疊、不驗身份,M8 直接寫進 DB,沒有任何一關能發現。
    碎裂(同一人被拆成兩個 chef_id)則可被「chef_id 數 > 排班人數」自動偵測。
    這個不對稱就是門檻要往保守側偏的理由,也是這個參數的物理意義。
    """
    return math.log(cost_false_merge_over_break) - math.log(prior_odds)
