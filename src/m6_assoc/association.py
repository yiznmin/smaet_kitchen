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


# ── 修正第 1 輪(2026-10-19):一手一物(N)、只算有在動的物件(V)──────────────
# 起因:第一輪誤報約 90% 發生在人正在接觸「另一個」物件時 —— 手邊每個物件都被算進去
# (docs/M6_人物關聯_預先登記_20261019.md §9.4)。

def iou(a, b):
    """IoU。a: [N,4], b: [M,4] → [N,M]。"""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.clip(area_a[:, None] + area_b[None, :] - inter, 1e-6, None)


def motion_iou(objs, obj_cls, prev, prev_cls):
    """每個物件與 Δ 步前同類框的最大 IoU;Δ 步前沒有同類框 → 0(視為在動)。"""
    out = np.zeros(len(objs))
    for i in range(len(objs)):
        p = prev[prev_cls == obj_cls[i]]
        if len(p):
            out[i] = iou(objs[i:i + 1], p).max()
    return out


def _nearest_gap(objs, obj_cls, hands_of, heights, class_ids):
    """一手一物:每隻手只配「框間距最小」的物件,同為最小時取與手 IoU 最大者。
    回傳每個類別被某隻手選中時的最小正規化間距(沒有 → inf)。"""
    out = np.full(len(class_ids), np.inf)
    if not len(objs):
        return out
    pos = {c: i for i, c in enumerate(class_ids)}
    for j, hands in hands_of.items():
        g = gap(hands, objs)                       # [n_hand, n_obj]
        u = iou(hands, objs)
        for h in range(len(hands)):
            best = np.flatnonzero(g[h] <= g[h].min() + 1e-6)
            k = best[u[h, best].argmax()]
            i = pos[obj_cls[k]]
            out[i] = min(out[i], g[h, k] / heights[j])
    return out


def frame_features_r1(xyxy, conf, cls, prev_xyxy, prev_conf, prev_cls, class_ids,
                      thr_person, thr_hand, thr_obj, hand_in_person, iou_moves, motion=None):
    """修正第 1 輪的單幀特徵。回傳 dict:

    upper     [n_cls]          IoA(物件, 沒偵測到手的人的上半身框)最大值(退路,四組共用)
    gap       [n_cls]          原規則:物件與手的最小正規化間距
    nn_gap    [n_cls]          N:一手一物
    mv_gap    [n_move, n_cls]  V:只看「在動」的物件(motion IoU < iou_move)
    nnmv_gap  [n_move, n_cls]  N+V:先濾掉靜止物件,再一手一物

    motion:每個物件的 motion IoU(順序 = 通過 thr_obj 的計分類別物件在本幀的順序);None = 與任一同類框比
    """
    xyxy = np.asarray(xyxy, dtype=np.float64).reshape(-1, 4)
    prev_xyxy = np.asarray(prev_xyxy, dtype=np.float64).reshape(-1, 4)
    persons = xyxy[(cls == PERSON) & _above(conf, thr_person)]
    hands = xyxy[(cls == HAND) & _above(conf, thr_hand)]
    owner = np.full(len(hands), -1)
    if len(hands) and len(persons):
        m = ioa(hands, persons)
        best = m.argmax(1)
        owner = np.where(m[np.arange(len(hands)), best] >= hand_in_person, best, -1)
    heights = np.clip(persons[:, 3] - persons[:, 1], 1e-6, None)
    hands_of = {j: hands[owner == j] for j in range(len(persons)) if (owner == j).any()}
    no_hand = np.array([j not in hands_of for j in range(len(persons))], dtype=bool)
    upper = persons[no_hand].copy()
    if len(upper):
        upper[:, 3] = upper[:, 1] + (upper[:, 3] - upper[:, 1]) / 2

    n = len(class_ids)
    obj_mask = np.isin(cls, class_ids) & _above(conf, thr_obj)
    objs, obj_cls = xyxy[obj_mask], cls[obj_mask]
    prev_mask = np.isin(prev_cls, class_ids) & _above(prev_conf, thr_obj)
    # motion:修正第 3 輪改由物件追蹤給「同一個物件」0.5 秒前的 IoU(順序對齊 obj_mask);
    # 沒給時沿用修正第 1 輪的「與任一同類框比」
    m_iou = (np.asarray(motion, dtype=np.float64) if motion is not None else
             motion_iou(objs, obj_cls, prev_xyxy[prev_mask], prev_cls[prev_mask]))

    out = dict(upper=np.zeros(n), gap=np.full(n, np.inf), nn_gap=np.full(n, np.inf),
               mv_gap=np.full((len(iou_moves), n), np.inf),
               nnmv_gap=np.full((len(iou_moves), n), np.inf))
    if not len(objs):
        return out
    for i, c in enumerate(class_ids):
        sel = obj_cls == c
        if not sel.any():
            continue
        if len(upper):
            out["upper"][i] = ioa(objs[sel], upper).max()
        for j, hs in hands_of.items():
            g = gap(objs[sel], hs).min(1) / heights[j]        # 每個物件到這個人的手
            out["gap"][i] = min(out["gap"][i], g.min())
            for q, thr_m in enumerate(iou_moves):
                mv = m_iou[sel] < thr_m
                if mv.any():
                    out["mv_gap"][q, i] = min(out["mv_gap"][q, i], g[mv].min())
    out["nn_gap"] = _nearest_gap(objs, obj_cls, hands_of, heights, class_ids)
    for q, thr_m in enumerate(iou_moves):
        mv = m_iou < thr_m
        out["nnmv_gap"][q] = _nearest_gap(objs[mv], obj_cls[mv], hands_of, heights, class_ids)
    return out


# ── 修正第 3 輪(2026-10-20):同一個物件的前後對應 ─────────────────────────
# 起因:修正第 2 輪 §6.2 —— 容器漏報 35% 是「在手邊但被判為靜止」,含 Carry Bottle。
# 檯面上同類物件很多(一排調味瓶),拿「任一同類框」比會比到旁邊靜止的另一個。

def track_motion(frames, class_ids, lag, link_iou=0.3, max_gap=3):
    """逐類別貪婪 IoU 追蹤,回傳每幀每個物件與「自己」lag 步前的 IoU。

    frames:每個時間步一筆 (xyxy [n,4], cls [n]),只含通過 thr_obj 的計分類別物件,順序即輸出順序
    規則:與最近 max_gap 步內出現過的同類軌跡,依 IoU 由大到小貪婪配對(≥ link_iou);沒配到 → 新軌跡。
    motion IoU:與同一軌跡在 lag 步前(或其後最早一次出現)的框比;軌跡比 lag 步還新 → 與軌跡第一個框比;
    第一次出現 → 0(視為在動,與修正第 1 輪「找不到同類框」的處理一致)。
    """
    tracks = {}                      # id -> dict(cls, steps[list], boxes[list])
    next_id, out = 0, []
    for i, (xy, cl) in enumerate(frames):
        res = np.zeros(len(xy))
        ids = np.full(len(xy), -1)
        for c in class_ids:
            det = np.flatnonzero(cl == c)
            cand = [k for k, tr in tracks.items() if tr["cls"] == c and i - tr["steps"][-1] <= max_gap]
            if len(det) and cand:
                m = iou(xy[det], np.array([tracks[k]["boxes"][-1] for k in cand]))
                used_d, used_t = set(), set()
                for flat in np.argsort(-m, axis=None):
                    a, b = divmod(int(flat), m.shape[1])
                    if m[a, b] < link_iou:
                        break
                    if a in used_d or b in used_t:
                        continue
                    used_d.add(a)
                    used_t.add(b)
                    ids[det[a]] = cand[b]
            for j in det:
                if ids[j] < 0:
                    ids[j] = next_id
                    tracks[next_id] = dict(cls=c, steps=[], boxes=[])
                    next_id += 1
        for j, k in enumerate(ids):
            tr = tracks[k]
            if tr["steps"]:
                ref = next((b for s, b in zip(tr["steps"], tr["boxes"]) if s >= i - lag), tr["boxes"][-1])
                res[j] = iou(xy[j:j + 1], np.asarray(ref)[None])[0, 0]
            tr["steps"].append(i)
            tr["boxes"].append(xy[j])
        # 清掉用不到的:超過 max_gap 沒出現的軌跡不會再被配對;lag 步以前的框不會再被參考
        for k in [k for k, tr in tracks.items() if i - tr["steps"][-1] > max_gap]:
            del tracks[k]
        for tr in tracks.values():
            while len(tr["steps"]) > 1 and tr["steps"][1] <= i - lag:
                tr["steps"].pop(0)
                tr["boxes"].pop(0)
        out.append(res)
    return out
