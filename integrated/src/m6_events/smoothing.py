"""時序平滑(遲滯狀態機)與多視角融合。

遲滯:最近 n 步裡有 k_on 步成立才開(開啟點回推到窗內第一個成立的步),
     連續 k_off 步不成立才關(關閉點回推到最後一個成立的步)。
     這樣區間邊界落在真正有證據的時間步上,不會被平滑參數往外拉。
"""
import numpy as np


def hysteresis(raw, n, k_on, k_off):
    raw = np.asarray(raw, dtype=bool)
    out = np.zeros(len(raw), dtype=bool)
    csum = np.concatenate([[0], np.cumsum(raw)])
    on, falses = False, 0
    for i in range(len(raw)):
        if not on:
            lo = max(0, i - n + 1)
            if csum[i + 1] - csum[lo] >= k_on:
                first = lo + int(np.argmax(raw[lo:i + 1]))
                out[first:i + 1] = True
                on, falses = True, 0
        elif raw[i]:
            out[i], falses = True, 0
        else:
            falses += 1
            if falses >= k_off:
                out[i - falses + 1:i + 1] = False
                on, falses = False, 0
            else:
                out[i] = True
    if on and falses:                      # 片尾還開著:同樣回推到最後一個成立的步
        out[len(raw) - falses:] = False
    return out


def fuse_k_of_n(raws, k):
    """多視角融合:同一時間步至少 k 個鏡頭成立。raws: [n_cam, T]。"""
    return np.asarray(raws, dtype=bool).sum(0) >= k
