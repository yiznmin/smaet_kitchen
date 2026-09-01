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
    # ⚠ 「綁回上次同一個 id」不足以判定正確 —— 若**所有人**都被併成同一個 id,
    #   那每個人的 last_pred 都等於 pred,於是每次都被算成正確,
    #   碎裂與誤併雙雙變成 0%,但系統其實完全失效(實測 4 人併成 1 個,
    #   兩個率都是 0.0% 而 IDF1 只有 0.264)。
    #   所以還要看該 id 的**歸屬**:它主要屬於誰。綁到別人「擁有」的 id 就是誤併。
    owner_votes = defaultdict(Counter)
    for gt, pred, _m, _t in records:
        owner_votes[pred][gt] += 1
    owner = {p: v.most_common(1)[0][0] for p, v in owner_votes.items()}

    # ⚠ 只有一個 ground-truth 身份時,誤併率**結構上不可量測**,不是 0。
    #   上面的 owner 判定會讓 owner[pred] 恆等於那唯一的 gt,於是下面
    #   「綁到別人身上」那條分支永遠不成立;剩下唯一會加到 fm 的是
    #   「matched 但換了個 id」,而那是自己碎裂後又被回收 —— 語意上是回收不是誤併。
    #   回傳 0.0 會讓報表印出「誤併率 0.0% ✅ 通過」,而那正是本專案踩過最嚴重的坑
    #   (4 人被併成 1 個,兩個率同時報 0%,IDF1 卻只有 0.264)的同一種形狀:
    #   指標瞎掉時給出好看的數字。寧可回 None,強迫上游處理「量不到」這件事。
    single_identity = len({g for g, _, _, _ in records}) < 2
    why_fm = ("ground-truth 只有一個身份 → 誤併分支是死碼,任何數字都不是誤併率"
              if single_identity else None)

    last_pred = {}
    n_tr = fm = brk = ok = 0
    for gt, pred, matched, is_tr in records:
        if is_tr:
            n_tr += 1
            if not matched:
                brk += 1
            elif owner.get(pred) != gt:
                fm += 1                       # 綁到別人的身份
            elif last_pred.get(gt) == pred:
                ok += 1
            else:
                fm += 1                       # 換了一個 id,而且不是自己的
        last_pred[gt] = pred
    if n_tr == 0:
        return dict(n_transitions=0, p_break=None, p_false_merge=None, p_correct=None,
                    fm_unmeasurable_reason="沒有任何轉場,三個率都沒有樣本")
    return dict(n_transitions=n_tr,
                p_break=round(brk / n_tr, 4),
                p_false_merge=None if single_identity else round(fm / n_tr, 4),
                p_correct=round(ok / n_tr, 4),
                fm_unmeasurable_reason=why_fm)


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
    """身份數與排班人數對不上就告警 —— **兩個方向都要看**。

    · 多於人數 → 碎裂(同一人被拆成多個)
    · 少於人數 → **整體誤併**(多人被併成一個)

    ⚠ 原本只檢查「多於」,結果 4 人被併成 1 個時 alarm=False,
      完全沒抓到。而那正是最嚴重的失效。
    """
    # ⚠ expected=1(單人資料)時,「整體誤併」方向需要 n_pred < 1 即 n_pred == 0,
    #   而 observations 非空就不可能 → 那個方向是死碼。單人資料上唯一有效的是
    #   「碎裂」方向,而它正好就是這次要用的主判準(chef_id 數應為 1)。
    n_pred = len({p for _, p in observations})
    kind = ("碎裂" if n_pred > expected_headcount else
            "整體誤併" if n_pred < expected_headcount else None)
    return dict(n_pred_ids=n_pred, expected=expected_headcount,
                alarm=kind is not None, kind=kind)


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

    # 把「這個數字能不能拿來下判斷」跟數字本身綁在一起,不靠讀報告的人自己記得。
    # 單人資料上有兩個指標會退化成無意義但**看起來正常**的值,這裡明講。
    n_gt = len({g for g, _ in obs})
    out["measurable"] = {
        "p_break": (out.get("n_transitions") or 0) > 0,
        "p_false_merge": n_gt >= 2 and (out.get("n_transitions") or 0) > 0,
        # 單一 gt 時 IDF1 的匈牙利指派只有一列,退化成「最大單一 chef_id 佔全部
        # 觀測的比例」。那個數字有意義(= 1 − 碎裂造成的觀測損失),但**不是**
        # MTMC 領域的 IDF1,不可拿去跟文獻或別的系統比。
        "idf1": n_gt >= 2,
        "fragmentation": True,          # 單人下唯一完全有效的主指標
        "headcount": True,
    }
    return out
