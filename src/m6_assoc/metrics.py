"""M6 關聯的評估指標(逐時間步、逐段)。

時間軸是偵測快取的取樣點(stride 3 → 每 0.1 秒一步),t = fid / fps。

逐時間步:GT 聯集區間內 = 正例。距任一 GT 邊界 ≤ tol 秒的時間步不計分(標註邊界的不確定)。
逐段:
  · Recall    :GT 段(聯集後)有 ≥ cover 比例的時間步被預測為正 → 抓到
  · Precision :預測區間有 ≥ cover 比例落在 GT(往外擴 tol 秒)內 → 正確
  兩邊都用「覆蓋比例」而不是「有重疊就算」:否則一個貫穿全片的預測區間就能 P=R=1。
輔助:F1@τ(τ = 0.1 / 0.25 / 0.5),與 MS-TCN、EPFL-Smart-Kitchen 基準同定義,供對外比較。
"""
import numpy as np


def mask_from_intervals(t, intervals):
    m = np.zeros(len(t), dtype=bool)
    for s, e in intervals:
        m |= (t >= s) & (t < e)
    return m


def boundary_mask(t, intervals, tol):
    """距任一區間邊界 ≤ tol 的時間步(不計分)。"""
    m = np.zeros(len(t), dtype=bool)
    if tol <= 0:
        return m
    for s, e in intervals:
        m |= np.abs(t - s) <= tol
        m |= np.abs(t - e) <= tol
    return m


def intervals_from_mask(t, mask, step):
    """布林序列 → 區間 [(s, e)],e = 最後一個為真的時間步 + step。"""
    out, start = [], None
    for i, v in enumerate(mask):
        if v and start is None:
            start = t[i]
        if not v and start is not None:
            out.append((float(start), float(t[i - 1] + step)))
            start = None
    if start is not None:
        out.append((float(start), float(t[-1] + step)))
    return out


def _prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return dict(P=round(p, 4), R=round(r, 4), F1=round(f, 4), tp=int(tp), fp=int(fp), fn=int(fn))


def timestep_counts(t, pred, gt_union, tol):
    gt = mask_from_intervals(t, gt_union)
    care = ~boundary_mask(t, gt_union, tol)
    return dict(tp=int((pred & gt & care).sum()), fp=int((pred & ~gt & care).sum()),
                fn=int((~pred & gt & care).sum()))


def segment_counts(t, pred, step, gt_union, tol, cover):
    """回傳 dict(gt_hit, gt_n, pred_ok, pred_n)。"""
    gt_hit = 0
    for s, e in gt_union:
        sel = (t >= s) & (t < e)
        if sel.any() and pred[sel].mean() >= cover:
            gt_hit += 1
    dil = mask_from_intervals(t, [(s - tol, e + tol) for s, e in gt_union])
    pred_iv = intervals_from_mask(t, pred, step)
    pred_ok = 0
    for s, e in pred_iv:
        sel = (t >= s) & (t < e)
        if sel.any() and dil[sel].mean() >= cover:
            pred_ok += 1
    return dict(gt_hit=gt_hit, gt_n=len(gt_union), pred_ok=pred_ok, pred_n=len(pred_iv))


F1_TAUS = (0.1, 0.25, 0.5)


def tiou(a, b):
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union > 0 else 0.0


def f1_at_tau_counts(pred_iv, gt_union, tau):
    """MS-TCN / EPFL 基準的 F1@τ:預測段依序找 tIoU 最大、尚未配對的 GT 段,tIoU ≥ τ 算 TP(一對一)。"""
    used = [False] * len(gt_union)
    tp = fp = 0
    for p in pred_iv:
        best, j = 0.0, -1
        for k, g in enumerate(gt_union):
            if not used[k]:
                v = tiou(p, g)
                if v > best:
                    best, j = v, k
        if j >= 0 and best >= tau:
            used[j], tp = True, tp + 1
        else:
            fp += 1
    return dict(tp=tp, fp=fp, fn=len(gt_union) - tp)


def evaluate(t, preds, gt, step, tol=0.5, cover=0.5):
    """preds[cls] = 布林陣列(與 t 等長);gt[cls]["union"] = 區間。

    回傳 {"per_class": {...}, "pooled": {...}}。pooled = 把 7 類的計數加總(主指標)。
    """
    per, ts_sum, sg_sum = {}, dict(tp=0, fp=0, fn=0), dict(gt_hit=0, gt_n=0, pred_ok=0, pred_n=0)
    f1_sum = {tau: dict(tp=0, fp=0, fn=0) for tau in F1_TAUS}
    for cls, g in gt.items():
        u = g["union"]
        ts = timestep_counts(t, preds[cls], u, tol)
        sg = segment_counts(t, preds[cls], step, u, tol, cover)
        pred_iv = intervals_from_mask(t, preds[cls], step)
        f1s = {tau: f1_at_tau_counts(pred_iv, u, tau) for tau in F1_TAUS}
        for k in ts_sum:
            ts_sum[k] += ts[k]
        for k in sg_sum:
            sg_sum[k] += sg[k]
        for tau, c in f1s.items():
            for k in c:
                f1_sum[tau][k] += c[k]
        per[cls] = dict(timestep=_prf(**ts), segment=_seg(sg),
                        f1_at={str(tau): _prf(**c) for tau, c in f1s.items()})
    return dict(per_class=per, pooled=dict(timestep=_prf(**ts_sum), segment=_seg(sg_sum),
                                           f1_at={str(tau): _prf(**c) for tau, c in f1_sum.items()}))


def _seg(c):
    r = c["gt_hit"] / c["gt_n"] if c["gt_n"] else 0.0
    p = c["pred_ok"] / c["pred_n"] if c["pred_n"] else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return dict(P=round(p, 4), R=round(r, 4), F1=round(f, 4), **c)
