"""依鏡頭角色分工(觀察紀錄:docs/EPFL_鏡頭配置觀察_20261019.md)。

  位置:人在 A 側 / B 側 / 中間(M)
       · 人工:configs/epfl_side_labels/<場次>.csv(output0 每 2 秒目視判讀)
       · 自動:定位鏡頭(output0)上信心最高的人框,腳點 x 落在哪一段
  接觸:只採用該側近拍組的證據;人在中間時兩組都採用
  評估:近拍鏡頭只對「人在該側」那段時間的真值計分(分側真值)
"""
import csv
import json
from pathlib import Path

import numpy as np

from m6_assoc.gt import SCORED
from m6_assoc.metrics import intervals_from_mask, mask_from_intervals

SIDES = ("A", "B", "M")


def load_roles(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_side_labels(path, t):
    """人工位置標記 → 與 t 等長的字串陣列(取最近的標記點)。"""
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    ts = np.array([float(r["t_sec"]) for r in rows])
    lab = np.array([r["side"] for r in rows])
    idx = np.clip(np.searchsorted(ts, t), 0, len(ts) - 1)
    prev = np.clip(idx - 1, 0, len(ts) - 1)
    near = np.where(np.abs(ts[prev] - t) <= np.abs(ts[idx] - t), prev, idx)
    return lab[near]


def auto_side(foot_x, x_a, x_b):
    """腳點 x(像素;NaN = 沒偵測到人)→ A / B / M。x < x_a → A,x > x_b → B,其餘 M。
    沒偵測到人時沿用前一步(片頭則為 M)。"""
    out = np.empty(len(foot_x), dtype="<U1")
    last = "M"
    for i, x in enumerate(foot_x):
        if np.isnan(x):
            out[i] = last
            continue
        last = out[i] = "A" if x < x_a else ("B" if x > x_b else "M")
    return out


def gate_gt(gt, t, step, keep):
    """真值只留 keep(布林,與 t 等長)為真的時間,重新切成區間。"""
    out = {}
    for c in SCORED:
        m = mask_from_intervals(t, gt[c]["union"]) & keep
        out[c] = {"segments": [], "union": intervals_from_mask(t, m, step)}
    return out


def role_fuse(raws_by_cam, side, close, k):
    """依位置挑近拍組再融合。raws_by_cam[cam] = [T, n_cls] 布林;回傳 [T, n_cls]。"""
    T, n = next(iter(raws_by_cam.values())).shape
    out = np.zeros((T, n), dtype=bool)
    groups = {"A": close["A"], "B": close["B"], "M": close["A"] + close["B"]}
    for s, cams in groups.items():
        sel = side == s
        if not sel.any():
            continue
        cnt = sum(raws_by_cam[c][sel].astype(int) for c in cams)
        out[sel] = cnt >= k
    return out
