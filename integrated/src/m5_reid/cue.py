"""身份訊號(cue):每次偵測都能讀到「這是誰」的證據。

2026-09-16 的上限實驗用。**這不是一個修法,是一把尺** ——
它回答「若每個框都有一個準確率 p 的身份訊號,這套架構最好能做到什麼程度」,
用來決定要不要在廚師身上加可辨識標記(帽頂色塊、編號),以及標記要做到多準。

## 為什麼不走 embedder / AppearanceLR

`AppearanceLR` 的高斯參數是為特定 embedder **實測校正**的(evidence.py 的 MEASURED,
dinov2 d′=0.25)。把合成訊號塞進那條路有兩個問題:
  1. 訊號的鑑別力會被壓成 dinov2 的鑑別力,準確率 p 這個旋鈕失效
  2. 「讀對的機率」與「cosine 分布」是兩種東西,硬套會讓結論無法解釋
所以身份訊號自成一條證據,LLR 直接從讀取器的錯誤模型推導。

## CueLR 的推導

讀取器每次讀出一個 id,**正確的機率是 p**,否則從其他 K−1 個 id 均勻誤讀。
比較的是兩次**獨立的讀數**:這條 track 現在的讀數 a,與候選 chef 身上存的上一個讀數 b。
(⚠ 存的也是一次有雜訊的讀數,不是真值 —— 所以兩邊都要算進錯誤。)

同一個人(真值 g)時兩次讀數相同的機率:

    P_same = p² + (1−p)²/(K−1)
             ↑ 兩次都讀對    ↑ 兩次都讀錯、而且錯成同一個

不同人(真值 g₁ ≠ g₂)時相同的機率:

    P_diff = 2p(1−p)/(K−1) + (K−2)(1−p)²/(K−1)²
             ↑ 一邊讀對、另一邊剛好錯成它   ↑ 兩邊都錯、錯成同一個第三者

於是

    讀數相同 → llr = log(P_same / P_diff)
    讀數不同 → llr = log((1 − P_same) / (1 − P_diff))

檢查兩個端點:
  · p = 1     → P_same = 1、P_diff = 0 → 相同是無限強的證據(由 clip 夾住)
  · p = 1/K   → P_same = P_diff = 1/K  → **llr 恆為 0**(讀數等於亂猜,沒有資訊)
p < 1/K(比亂猜還差)時符號會反過來,公式照樣成立,不特別處理。

## 合成訊號

`synth_token` 是**純函數**:同樣的 (seed, key) 必定得到同樣的讀數,
所以整格實驗可以逐位重現(不是每次跑都重抽)。
`burst` > 0 時錯誤會**成群出現**(同一個桶內的讀數完全相同)——
真實標記被遮住時是連續讀不到,不是每幀獨立擲骰子。
"""
import hashlib
import math


def reading_key(camera_id, track_id, loop_i, burst=0):
    """一次讀取的識別。burst > 0 時把時間軸分桶,同桶內是同一個讀數。"""
    return (str(camera_id), int(track_id),
            int(loop_i) // int(burst) if burst else int(loop_i))


def _unit(seed, key):
    """由 (seed, key) 決定性地產生 [0, 1) 的亂數。

    ⚠ 不用 Python 內建 hash():它對 str 有逐次啟動的隨機化(PYTHONHASHSEED),
      同樣的輸入換一次執行就不同,實驗會無法重現。
    """
    h = hashlib.blake2b(f"{seed}|{key}".encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(h, "big") / float(1 << 64)


def synth_token(gt_id, ids, *, key, accuracy, seed):
    """以 accuracy 的機率回傳 gt_id,否則從其他 id 均勻誤讀一個。

    gt_id 為 None(這條 track 根本不是人)→ 回 None,表示**讀不到標記**。
    誤偵身上沒有標記可讀,給它一個假讀數等於憑空發明證據。
    """
    if gt_id is None:
        return None
    others = [i for i in ids if i != gt_id]
    u = _unit(seed, key)
    if u < accuracy or not others:
        return gt_id
    # 把 [accuracy, 1) 重新攤平到 [0, 1),再均勻選一個其他 id
    v = (u - accuracy) / (1.0 - accuracy)
    return others[min(int(v * len(others)), len(others) - 1)]


class CueLR:
    """兩次讀數比對的對數似然比。推導見檔頭。

    clip:夾住上下限。p → 1 時 LLR 會發散,而**沒有任何單一證據該壓垮其他全部證據**
      —— 與 AppearanceLR / GroundPlaneLR 同一個慣例。
    """

    def __init__(self, accuracy, n_ids, clip=8.0):
        self.p = float(accuracy)
        self.k = int(n_ids)
        if not 0.0 <= self.p <= 1.0:
            raise ValueError(f"accuracy 必須在 [0,1],收到 {accuracy!r}")
        if self.k < 2:
            raise ValueError(f"n_ids 必須 ≥ 2,收到 {n_ids!r}")
        self.clip = float(clip) if clip is not None else None
        p, k = self.p, self.k
        self.p_same = p * p + (1.0 - p) ** 2 / (k - 1)
        self.p_diff = (2.0 * p * (1.0 - p) / (k - 1)
                       + (k - 2) * (1.0 - p) ** 2 / (k - 1) ** 2)

    def _clipped(self, num, den):
        if num <= 0.0:                      # 這個結果在該假設下不可能發生
            return -self.clip if self.clip is not None else -math.inf
        if den <= 0.0:
            return self.clip if self.clip is not None else math.inf
        v = math.log(num / den)
        if self.clip is not None:
            v = max(-self.clip, min(self.clip, v))
        return v

    def llr(self, token_a, token_b):
        """兩個讀數的 LLR。任一邊沒有讀數(None)→ 0.0,表示沒有證據。"""
        if token_a is None or token_b is None:
            return 0.0
        if token_a == token_b:
            return self._clipped(self.p_same, self.p_diff)
        return self._clipped(1.0 - self.p_same, 1.0 - self.p_diff)

    def max_abs_llr(self):
        """這個訊號的發言權上限(nats),可直接與時間證據(峰值約 5)比大小。"""
        return max(abs(self.llr(1, 1)), abs(self.llr(1, 2)))

    def describe(self):
        return (f"CueLR(準確率 {self.p:.3f}, {self.k} 個身份, "
                f"相同 {self.llr(1, 1):+.2f} / 不同 {self.llr(1, 2):+.2f} nats)")
