"""M7 階段 A:對候選物件跑框提示分割,存遮罩特徵(之後評估只讀這份,不再跑模型)。

預先登記:docs/M7_模型評估_預先登記_20261020.md §3
候選:該側近拍鏡頭上,物件(計分類別、conf > 0.4)與某人的手(conf > 0.2)框間距 ≤ 0.1 × 人高。
  —— 取網格最寬的門檻;評估時再依 thr_obj、thr_hand、d 過濾。
每個候選物件:
  mask_iou :同一個物件框在 t 與 t−0.5 秒各分割一次,扣掉兩幀所有手遮罩後的 IoU
  每隻手一筆:手 conf、框間距 / 人高、手遮罩膨脹 r(0 / 1% / 2% 人高)後與物件遮罩重疊的像素數

輸出:results/m7/stage_a/feat_<model>.npz(本機,EPFL 衍生物不進 git)

第 2 輪 --exclusive(AX):另切其他物件,框較小且 ≥ 50% 落在候選框內者從候選遮罩扣掉 → feat_<model>_x.npz

用法:
    python scripts/m7_stage_a_features.py --model sam2.1_b+ [--exclusive]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import eval_m6_roles as R                                                 # noqa: E402
from m3.classes import NAME_TO_CLASS                                      # noqa: E402
from m4_track.det_cache import DetCache                                   # noqa: E402
from m6_assoc.association import _above, gap, ioa                         # noqa: E402
from m6_assoc.gt import SCORED                                            # noqa: E402
from m6_assoc.roles import auto_side, load_roles                          # noqa: E402
from m7_mask.contact import mask_contact, mask_motion_iou                 # noqa: E402
from m7_mask.segment import BoxSegmenter                                  # noqa: E402

THR_PERSON, HAND_IN_PERSON = 0.3, 0.5
THR_OBJ_MIN, THR_HAND_MIN, D_MAX = 0.4, 0.2, 0.1
R_FRAC = (0.0, 0.01, 0.02)
LAG = 5
CLASS_IDS = [NAME_TO_CLASS[c] for c in SCORED]


def owned_hands(xy, cf, cl):
    """指派給某人框的手:(框, conf, 該人框高)。"""
    pm = (cl == NAME_TO_CLASS["人"]) & _above(cf, THR_PERSON)
    hm = (cl == NAME_TO_CLASS["手"]) & _above(cf, THR_HAND_MIN)
    persons, hands, hconf = xy[pm], xy[hm], cf[hm]
    if not len(hands) or not len(persons):
        return np.zeros((0, 4)), np.zeros(0), np.zeros(0)
    m = ioa(hands, persons)
    best = m.argmax(1)
    ok = m[np.arange(len(hands)), best] >= HAND_IN_PERSON
    return hands[ok], hconf[ok], (persons[best, 3] - persons[best, 1])[ok]


def exclusive_subtract(objs, others, o_t, x_t, o_p, x_p, inside=0.5):
    """每個候選物件:框面積較小、且框有 ≥ inside 落在候選框內的其他物件,遮罩從候選遮罩扣掉(兩幀都扣)。"""
    allb = np.vstack([objs, others])
    all_t, all_p = np.concatenate([o_t, x_t]), np.concatenate([o_p, x_p])
    area = (allb[:, 2] - allb[:, 0]) * (allb[:, 3] - allb[:, 1])
    inn = ioa(allb, objs)                                   # [n_all, n_obj]
    o_t, o_p = o_t.copy(), o_p.copy()
    for k in range(len(objs)):
        sub = np.flatnonzero((area < area[k]) & (inn[:, k] >= inside))
        sub = sub[sub != k]
        if len(sub):
            o_t[k] &= ~all_t[sub].any(0)
            o_p[k] &= ~all_p[sub].any(0)
    return o_t, o_p


def run_cam(seg, c, video, active, exclusive=False):
    """active[i]:第 i 步這台鏡頭是否在該側近拍組。回傳候選與手配對的 list。"""
    stride = int(c.meta["cache_stride"])
    cap = cv2.VideoCapture(str(video))
    frames, embs = {}, {}
    cands, pairs = [], []
    n_steps = len(active)
    fid = 0
    for i in range(n_steps):
        target = i * stride
        while fid <= target:                       # 依序解碼,只留 stride 幀
            ok = cap.grab()
            if fid % stride == 0 and ok:
                frames[fid] = cap.retrieve()[1]
            fid += 1
        for k in [k for k in frames if k < target - LAG * stride]:
            frames.pop(k)
        for k in [k for k in embs if k < target - LAG * stride]:
            embs.pop(k)
        if not active[i] or i < LAG:
            continue
        lo, hi = np.searchsorted(c._fid, target, "left"), np.searchsorted(c._fid, target, "right")
        xy, cf, cl = c._xyxy[lo:hi].astype(np.float64), c._conf[lo:hi], c._cls[lo:hi]
        hands, hconf, heights = owned_hands(xy, cf, cl)
        om = np.isin(cl, CLASS_IDS) & _above(cf, THR_OBJ_MIN)
        objs, ocls, oconf = xy[om], cl[om], cf[om]
        if not len(objs) or not len(hands):
            continue
        g = gap(objs, hands) / heights[None, :]            # [n_obj, n_hand]
        near = g.min(1) <= D_MAX
        if not near.any():
            continue
        # 互斥扣除(第 2 輪 AX):其他物件也要切,之後從候選物件遮罩扣掉
        others = objs if exclusive else np.zeros((0, 4))
        objs, ocls, oconf, g = objs[near], ocls[near], oconf[near], g[near]
        if exclusive:
            others = others[~near]
        prev = target - LAG * stride
        plo, phi = np.searchsorted(c._fid, prev, "left"), np.searchsorted(c._fid, prev, "right")
        phands, _, _ = owned_hands(c._xyxy[plo:phi].astype(np.float64), c._conf[plo:phi], c._cls[plo:phi])
        for f in (target, prev):
            if f not in embs:
                embs[f] = seg.embed(frames[f][:, :, ::-1].copy())
        n_o, n_x = len(objs), len(others)
        m_t = seg.masks(embs[target], np.vstack([objs, others, hands]))
        m_p = seg.masks(embs[prev], np.vstack([objs, others, phands]))
        o_t, x_t, h_t = m_t[:n_o], m_t[n_o:n_o + n_x], m_t[n_o + n_x:]
        o_p, x_p, h_p = m_p[:n_o], m_p[n_o:n_o + n_x], m_p[n_o + n_x:]
        if exclusive:
            o_t, o_p = exclusive_subtract(objs, others, o_t, x_t, o_p, x_p)
        all_hands = h_t.any(0) | (h_p.any(0) if len(h_p) else np.zeros_like(h_t[0]))
        for k in range(len(objs)):
            cid = len(cands)
            cands.append((i, CLASS_IDS.index(int(ocls[k])), float(oconf[k]),
                          mask_motion_iou(o_t[k], o_p[k], all_hands)))
            for h in np.flatnonzero(g[k] <= D_MAX):
                pairs.append((cid, float(hconf[h]), float(g[k, h]),
                              *[mask_contact(o_t[k], h_t[h], r * heights[h]) for r in R_FRAC]))
    cap.release()
    return cands, pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="sam2.1_b+")
    ap.add_argument("--cache-dir", default=str(ROOT / "results/det_cache/epfl_allcls_armA_s3"))
    ap.add_argument("--roles-selected", default=str(ROOT / "results/m6/dev_yh2040_roles/selected_roles.json"))
    ap.add_argument("--out-dir", default=str(ROOT / "results/m7/stage_a"))
    ap.add_argument("--exclusive", action="store_true", help="第 2 輪 AX:遮罩互斥扣除")
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    roles = load_roles(ROOT / "configs/camera_roles.epfl.json")
    close = roles["close"]
    cams = close["A"] + close["B"]
    c0 = DetCache(Path(args.cache_dir) / f"{cams[0]}.npz", expect={"classes": "all", "complete": True})
    fps, stride = float(c0.meta["video_meta"]["fps"]), int(c0.meta["cache_stride"])
    t = np.arange(0, int(c0.meta["max_fid"]) + 1, stride) / fps
    auto_x = json.loads(Path(args.roles_selected).read_text(encoding="utf-8"))["auto_x"]
    side = auto_side(R.foot_x(args.cache_dir, roles["locator"], t), **auto_x)
    seg = BoxSegmenter(args.model)

    C, P, timing = [], [], {}
    for ci, cam in enumerate(cams):
        t1 = time.time()
        c = DetCache(Path(args.cache_dir) / f"{cam}.npz", expect={"classes": "all", "complete": True})
        active = np.array([cam in (close[s] if s in ("A", "B") else cams) for s in side])
        cands, pairs = run_cam(seg, c, ROOT / "data/epfl" / f"{cam}.mp4", active, args.exclusive)
        base = len(C)
        C += [(ci, *x) for x in cands]
        P += [(base + p[0], *p[1:]) for p in pairs]
        timing[cam] = dict(sec=round(time.time() - t1, 1), active_steps=int(active.sum()), cands=len(cands))
        print(f"  {cam}:有效步 {active.sum():,}、候選 {len(cands):,}、{time.time() - t1:.0f} 秒", flush=True)

    C, P = np.array(C, dtype=np.float64), np.array(P, dtype=np.float64)
    name = f"feat_{args.model}{'_x' if args.exclusive else ''}.npz"
    np.savez_compressed(out / name,
                        cand=C, pair=P, t=t, side=side.astype("U1"),
                        meta=json.dumps(dict(model=args.model, exclusive=args.exclusive, cams=cams, r_frac=R_FRAC, lag=LAG,
                                             thr_obj_min=THR_OBJ_MIN, thr_hand_min=THR_HAND_MIN,
                                             d_max=D_MAX, timing=timing,
                                             cand_cols=["cam", "step", "cls", "obj_conf", "mask_iou"],
                                             pair_cols=["cand", "hand_conf", "gap"] +
                                                       [f"contact_r{r}" for r in R_FRAC])))
    print(f"輸出 {out / name},{time.time() - t0:.0f} 秒")


if __name__ == "__main__":
    main()
