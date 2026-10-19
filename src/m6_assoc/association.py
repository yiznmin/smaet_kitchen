"""單幀人—物關聯(規則,可審計)。

輸入一幀的 M3 偵測(xyxy, conf, class_id),對每個計分類別算出幾個連續量,
門檻比較放到 decide() —— 這樣 d、IoA 門檻可以直接在同一份連續量上掃,不必重算。

規則:
  1. 人框:class「人」且 conf > thr_person
  2. 手框:class「手」且 conf > thr_hand,且 IoA(手, 某人框) ≥ hand_in_person → 指派給 IoA 最大的那個人
  3. 物件 o 與人 p 有關聯:
       a. p 有手:o 與 p 任一隻手的框間距 ≤ d × p 的框高          → 證據 "hand"
       b. p 沒有偵測到手(退路):IoA(o, p 的上半身框) ≥ x_upper   → 證據 "upper_body"

兩條基準線(同一套輸入,用來證明「用手」有用):
  present:畫面裡有這類物件(conf > thr_obj)就算接觸
  person :IoA(o, 任一人框) ≥ x_person 就算接觸(不用手)

門檻比較一律用 `conf > float32(thr)`,與 rfdetr predict 及 DetCache 的判定一致。
"""
import numpy as np

from m3.classes import NAME_TO_CLASS

PERSON = NAME_TO_CLASS["人"]
HAND = NAME_TO_CLASS["手"]


def ioa(a, b):
    """IoA(a 在 b 內的比例)。a: [N,4], b: [M,4] → [N,M]。"""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = np.clip(a[:, 2] - a[:, 0], 1e-6, None) * np.clip(a[:, 3] - a[:, 1], 1e-6, None)
    return inter / area[:, None]


def gap(a, b):
    """兩組框之間的最短距離(重疊為 0)。[N,4] × [M,4] → [N,M]。"""
    if len(a) == 0 or len(b) == 0:
        return np.full((len(a), len(b)), np.inf)
    dx = np.clip(np.maximum(a[:, None, 0], b[None, :, 0])
                 - np.minimum(a[:, None, 2], b[None, :, 2]), 0, None)
    dy = np.clip(np.maximum(a[:, None, 1], b[None, :, 1])
                 - np.minimum(a[:, None, 3], b[None, :, 3]), 0, None)
    return np.hypot(dx, dy)


def _above(conf, thr):
    return conf > np.float32(thr).astype(conf.dtype)


def frame_features(xyxy, conf, cls, class_ids, thr_person, thr_hand, thr_obj, hand_in_person):
    """回傳 [len(class_ids), 4] 的 float 陣列,欄位:

    0 present    :有沒有這類物件(0/1)
    1 person_ioa :max IoA(物件, 人框);沒有則 0
    2 hand_gap   :min(物件與「有手的人」的手之框間距 / 該人框高);沒有則 inf
    3 upper_ioa  :max IoA(物件, 「沒偵測到手的人」的上半身框);沒有則 0
    """
    xyxy = np.asarray(xyxy, dtype=np.float64).reshape(-1, 4)
    persons = xyxy[(cls == PERSON) & _above(conf, thr_person)]
    hands = xyxy[(cls == HAND) & _above(conf, thr_hand)]
    owner = np.full(len(hands), -1)
    if len(hands) and len(persons):
        m = ioa(hands, persons)
        best = m.argmax(1)
        owner = np.where(m[np.arange(len(hands)), best] >= hand_in_person, best, -1)
    heights = np.clip(persons[:, 3] - persons[:, 1], 1e-6, None)
    has_hand = np.array([(owner == j).any() for j in range(len(persons))], dtype=bool)
    upper = persons[~has_hand].copy()
    if len(upper):
        upper[:, 3] = upper[:, 1] + (upper[:, 3] - upper[:, 1]) / 2

    out = np.zeros((len(class_ids), 4))
    out[:, 2] = np.inf
    obj_ok = _above(conf, thr_obj)
    for i, c in enumerate(class_ids):
        objs = xyxy[(cls == c) & obj_ok]
        if not len(objs):
            continue
        out[i, 0] = 1
        if len(persons):
            out[i, 1] = ioa(objs, persons).max()
        for j in np.flatnonzero(has_hand):
            g = gap(objs, hands[owner == j]).min() / heights[j]
            out[i, 2] = min(out[i, 2], g)
        if len(upper):
            out[i, 3] = ioa(objs, upper).max()
    return out


def decide(feat, d, x_upper):
    """M6 單幀判定。feat: [..., 4] → (hit, evidence),evidence: 1=hand、2=upper_body、0=無。"""
    by_hand = feat[..., 2] <= d
    by_upper = (~by_hand) & (feat[..., 3] >= x_upper) & (feat[..., 0] > 0)
    return by_hand | by_upper, np.where(by_hand, 1, np.where(by_upper, 2, 0))


EVIDENCE = {1: "hand", 2: "upper_body"}
