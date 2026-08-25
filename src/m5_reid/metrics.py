"""M5 系統級指標:IDF1、碎裂率、誤併率、ID switch。

為什麼不能只用既有的 binding_sweep(scripts/reid_eval_market1501.py):

| | binding_sweep | 這裡的指標 |
|---|---|---|
| 單位 | 單張 crop | 整段軌跡 |
| gallery 代表 | 全體平均(偷看未來) | 系統線上 EMA |
| 能不能失敗在「碎裂」 | **不能** | 能 |
| 量的是 | embedder 的性質 | M5 系統的性質 |

binding_sweep 結構上不可能失敗在碎裂上,而碎裂正是本架構最可能的失效模式
——只拿它給業主看會系統性隱藏最大的風險。兩者並存、角色分工。

IDF1 直接實作(不引入 motmetrics),因為跨鏡頭版需要控制「幀」的定義:
把 (時間, 鏡頭) 串成單一全域序列,預測側一律用全域 chef_id,這樣跨鏡頭的
身份斷裂會自動被 IDF1 懲罰 —— 即 CVIDF1 的近似。
"""
from collections import Counter, defaultdict

import numpy as np
from scipy.optimize import linear_sum_assignment


def idf1(observations):
    """observations = [(gt_id, pred_id), ...],每筆是一次「某人在某鏡頭被看到」。

    標準 IDF1:在 gt_id 與 pred_id 之間解一次全域最佳一對一指派(每個真實身份
    只能對應一個預測身份),被正確涵蓋的觀測數 = IDTP。

        IDF1 = 2·IDTP / (2·IDTP + IDFP + IDFN)

    本情境下每筆觀測都同時有 gt 與 pred(事件層評估,偵測視為完美),
    故 IDFP = IDFN = N − IDTP,IDF1 退化為 IDTP / N。
    """
    if not observations:
        return dict(idf1=None, idtp=0, n=0)
    gts = sorted({g for g, _ in observations})
    preds = sorted({p for _, p in observations})
    gi = {g: i for i, g in enumerate(gts)}
    pi = {p: i for i, p in enumerate(preds)}
    m = np.zeros((len(gts), len(preds)))
    for g, p in observations:
        m[gi[g], pi[p]] += 1
    r, c = linear_sum_assignment(-m)                 # 最大化被涵蓋的觀測數
    idtp = int(m[r, c].sum())
    n = len(observations)
    idfp = idfn = n - idtp
    return dict(idf1=round(2 * idtp / (2 * idtp + idfp + idfn), 4),
                idtp=idtp, n=n, n_gt=len(gts), n_pred=len(preds))


def binding_outcomes(records):
    """逐次綁定決策的結果分類。records 需**按時間排序**,
    每筆 = (gt_id, pred_id, matched, is_transition)。

    ⚠ 不能用 `pred_id != gt_id` 判斷綁錯 —— 系統的 chef_id 是任意編號,
      不可能與 ground-truth 編號重合。正確的判準是「有沒有綁回這位廚師
      上一次拿到的那個 chef_id」:

      · 綁到上次同一個 id            → 正確(身份連續)
      · 綁到別的 id                  → **誤併**(接到別人的身份上)
      · 沒綁、開新 id                → **碎裂**

    碎裂可偵測(chef_id 數 > 排班人數就會告警),誤併完全靜默且下游全盤繼承,
    所以兩者要分開報,不可合併成單一「錯誤率」。
    """
    last_pred = {}
    n_tr = fm = brk = ok = 0
    for gt, pred, matched, is_tr in records:
        if is_tr:
            n_tr += 1
            if not matched:
                brk += 1
            elif last_pred.get(gt) == pred:
                ok += 1
            else:
                fm += 1
        last_pred[gt] = pred
    if n_tr == 0:
        return dict(n_transitions=0, p_break=None, p_false_merge=None, p_correct=None)
    return dict(n_transitions=n_tr,
                p_break=round(brk / n_tr, 4),
                p_false_merge=round(fm / n_tr, 4),
                p_correct=round(ok / n_tr, 4))


def id_switches(observations_in_time_order):
    """同一個真實身份的預測 id 中途改變的次數。

    碎裂與誤併都會造成 switch,但 switch 本身不區分兩者 —— 所以它是輔助指標,
    主指標仍是上面那兩個率。
    """
    last = {}
    n = 0
    for gt, pred in observations_in_time_order:
        if gt in last and last[gt] != pred:
            n += 1
        last[gt] = pred
    return n


def fragmentation(observations):
    """每個真實身份被拆成幾個 chef_id(1 = 完美)。回傳 (平均, 最大, 分布)。"""
    per = defaultdict(set)
    for g, p in observations:
        per[g].add(p)
    counts = [len(v) for v in per.values()]
    if not counts:
        return dict(mean=None, max=None, hist={})
    return dict(mean=round(float(np.mean(counts)), 3), max=max(counts),
                hist=dict(Counter(counts)))


def headcount_alarm(observations, expected_headcount):
    """碎裂的**可偵測性**:預測身份數 > 排班人數就該告警。

    這是唯一能自動抓到的失效訊號 —— 誤併不會讓身份數變多,所以抓不到。
    這個不對稱是「寧可碎裂也不要誤併」的根據。
    """
    n_pred = len({p for _, p in observations})
    return dict(n_pred_ids=n_pred, expected=expected_headcount,
                alarm=n_pred > expected_headcount)


def summarize(records, expected_headcount=None):
    """把一次模擬跑的結果整理成一份指標。records 見 binding_outcomes。"""
    obs = [(g, p) for g, p, _, _ in records]
    out = {}
    out.update(idf1(obs))
    out.update(binding_outcomes(records))
    out["id_switches"] = id_switches(obs)
    out["fragmentation"] = fragmentation(obs)
    if expected_headcount:
        out["headcount"] = headcount_alarm(obs, expected_headcount)
    return out
