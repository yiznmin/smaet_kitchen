"""逐片段獨立推論的指標(預先登記 `docs/難度分級_逐片段獨立推論_預先登記_20260923.md` §5)。

## 為什麼要新寫一套

既有的兩套指標都不合用:

- `m5_reid.metrics`(誤併 / 碎裂 / 正確率)是**逐次綁定決策**,不是逐幀;
  而且它的「正確」定義是「綁回這個人上次拿到的編號」,片段從零開始時沒有「上次」。
- `eval_m4m5_chirla.match_tracks` 用**整段多數決**決定「這條 track 是誰」,
  一條前半是甲後半是乙的 track 只會被記成其中一個 —— 正好把要看的東西藏起來。

所以本模組一律**逐幀**判定,並把所有率的分母寫死在檔頭:

    T = 目標「有標註框」的取樣格數           → 召回、IoU 統計的分母
    B = 目標「被配對且該 track 有編號」的格數 → 三個身份率的分母

## ⚠ 三個率的關係(這是本輪最容易誤讀的地方)

連續率、多數率**都無法否證「整段被綁成同一個錯誤編號」**:那種情況下兩者都是 100%。
能否證它的只有排他率 —— 但排他率需要窗內有第二個真人。
**窗內只有一個身份時,排他率結構上不可量測 → 回 None,由上游印「不可量測」,不可印 100%。**
(與 `m5_reid/metrics.py` 的 `p_false_merge` 同一條紀律。)
"""
import math
from collections import Counter, defaultdict

import numpy as np
from scipy.optimize import linear_sum_assignment

IOU_THR = 0.50
GHOST_MATCH_RATE_MIN = 0.50          # 與 eval_m4m5_chirla.MATCH_RATE_MIN 同值,但**只在窗內**算
UNMEASURABLE = "不可量測(窗內只有 1 個身份)"


def iou_xyxy(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def wilson(k, n, z=1.96):
    """Wilson 95% 區間。⚠ L3 最短的片段只有 18 個取樣點,一格 = 5.6%,
    不附區間的百分比會被當成精確值。"""
    if not n:
        return (None, None)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(max(0.0, c - h), 4), round(min(1.0, c + h), 4))


def assign_frame(gt_dets, trk_dets, iou_thr=IOU_THR):
    """單幀匈牙利配對。

    ⚠ 配對前依 gt_id / track_id 升冪排序 —— 平手時 `linear_sum_assignment` 的選擇
      取決於矩陣順序,不排序的話同一份資料兩次執行可能得到不同配對。
    回傳 (matched{tid: (gid, iou)}, ghost_tids, missed_gids, best_iou_by_gid)
    """
    gt = sorted(gt_dets, key=lambda x: x[0])
    trk = sorted(trk_dets, key=lambda x: x[0])
    best = {g: 0.0 for g, _ in gt}
    if not gt or not trk:
        return {}, [t for t, _ in trk], [g for g, _ in gt], best
    m = np.zeros((len(gt), len(trk)))
    for i, (_g, gb) in enumerate(gt):
        for j, (_t, tb) in enumerate(trk):
            m[i, j] = iou_xyxy(gb, tb)
    for i, (g, _b) in enumerate(gt):
        best[g] = float(m[i].max()) if m.shape[1] else 0.0
    ri, ci = linear_sum_assignment(-m)
    matched, used_g, used_t = {}, set(), set()
    for i, j in zip(ri, ci):
        if m[i, j] >= iou_thr:
            matched[trk[j][0]] = (gt[i][0], float(m[i, j]))
            used_g.add(gt[i][0])
            used_t.add(trk[j][0])
    return (matched,
            [t for t, _ in trk if t not in used_t],
            [g for g, _ in gt if g not in used_g],
            best)


def per_frame_rows(clip, stride, gt, tracks, chefs):
    """逐(鏡頭, 取樣幀)一列。目標不在場的格也要留 —— L2 的間隔與誤偵都靠它。

    gt:     {鏡頭: {真值幀(1-based): [(gt_id, bbox)]}}
    tracks: {(鏡頭, fid): [(track_id, bbox 或 None)]}(None = 該幀沒配到偵測,空轉)
    chefs:  {(鏡頭, track_id): chef_id}
    """
    gid = clip["gt_id"]
    rows = []
    for fid in range(clip["start_fid"], clip["end_fid"] + 1, stride):
        for cam in clip["cameras"]:
            all_trk = tracks.get((cam, fid), [])
            live = [(t, b) for t, b in all_trk if b is not None]
            coast = [t for t, b in all_trk if b is None]
            gts = gt.get(cam, {}).get(fid + 1, [])
            matched, ghosts, _missed, best = assign_frame(gts, live)
            gt_box = next((b for g, b in gts if g == gid), None)
            tid = next((t for t, (g, _i) in matched.items() if g == gid), None)
            iou = matched[tid][1] if tid is not None else 0.0
            chef = chefs.get((cam, tid)) if tid is not None else None
            # 這一幀有沒有別人也戴著同一個編號(排他性的原始資料)
            others_same_chef = sorted(
                {g for t, (g, _i) in matched.items()
                 if g != gid and chefs.get((cam, t)) == chef and chef is not None})
            rows.append(dict(
                clip_id=clip["clip_id"], stride=stride, camera_id=cam, video_fid=fid,
                gt_frame=fid + 1, t_sec=round(fid / 30.0, 3),
                gt_present=int(gt_box is not None),
                gt_box="" if gt_box is None else " ".join(f"{v:.1f}" for v in gt_box),
                matched=int(tid is not None), track_id="" if tid is None else tid,
                iou=round(iou, 4), iou_best=round(best.get(gid, 0.0), 4),
                chef_id="" if chef is None else chef,
                n_tracks=len(live), n_gt=len(gts), n_ghost=len(ghosts),
                n_coasting=len(coast),
                other_gt_ids=" ".join(str(g) for g, _b in sorted(gts) if g != gid),
                others_same_chef=" ".join(str(x) for x in others_same_chef),
                chef_ids_on_frame=" ".join(
                    str(chefs[(cam, t)]) for t, _b in sorted(live)
                    if (cam, t) in chefs)))
    return rows


def _series(rows):
    """目標在場的格,依時間排序。"""
    return sorted((r for r in rows if r["gt_present"]),
                  key=lambda r: (r["video_fid"], r["camera_id"]))


def iou_stats(rows):
    s = _series(rows)
    v = [r["iou"] for r in s]
    hit = [r for r in s if r["matched"]]
    q = (lambda p: round(float(np.percentile(v, p)), 4)) if v else (lambda p: None)
    return dict(
        n_target_cells=len(s), n_matched=len(hit),
        recall_target=round(len(hit) / len(s), 4) if s else None,
        recall_ci=wilson(len(hit), len(s)),
        mean_iou=round(float(np.mean(v)), 4) if v else None,
        median_iou=q(50), p10_iou=q(10), p90_iou=q(90),
        frac_iou_ge_075=round(sum(x >= 0.75 for x in v) / len(v), 4) if v else None,
        mean_iou_matched=round(float(np.mean([r["iou"] for r in hit])), 4) if hit else None)


def identity_rates(rows, exclusive):
    """三個身份率。分母一律是 B(目標被配對且有編號的格數)。"""
    s = _series(rows)
    b = [r for r in s if r["matched"] and r["chef_id"] != ""]
    out = dict(n_B=len(b), bound_rate=round(len(b) / len(s), 4) if s else None,
               first_bound_fid=b[0]["video_fid"] if b else None,
               n_distinct_chef_ids=len({r["chef_id"] for r in b}))
    if not b:
        return {**out, "p_continuity": None, "p_majority": None, "p_exclusive": None,
                "c0": None, "c_star": None, "majority_tie": None,
                "p_continuity_ci": (None, None), "p_majority_ci": (None, None)}
    c0 = b[0]["chef_id"]
    cnt = Counter(r["chef_id"] for r in b)
    top = max(cnt.values())
    tied = sorted(c for c, n in cnt.items() if n == top)
    c_star = tied[0]                       # 平手取較小編號,並記錄有平手
    k_cont = sum(r["chef_id"] == c0 for r in b)
    k_maj = cnt[c_star]
    # 排他:同一幀上沒有別人戴著目標這一格的編號
    k_exc = sum(not r["others_same_chef"] for r in b)
    return {**out,
            "c0": c0, "c_star": c_star, "majority_tie": len(tied) > 1,
            "p_continuity": round(k_cont / len(b), 4),
            "p_continuity_ci": wilson(k_cont, len(b)),
            "p_majority": round(k_maj / len(b), 4),
            "p_majority_ci": wilson(k_maj, len(b)),
            # ⚠ 獨佔窗:結構上不可量測,回 None(不可回 1.0)
            "p_exclusive": None if exclusive else round(k_exc / len(b), 4),
            "p_exclusive_ci": (None, None) if exclusive else wilson(k_exc, len(b))}


def switches(rows):
    """編號切換與 track 切換。

    ⚠ **2026-09-24 更正,使用者指出**:原本把所有鏡頭的格子依 `(fid, camera)` 混在一起
      排序後數相鄰變化(登記 §5.4 的字面定義),對**多鏡頭片段**量到的其實是
      「兩台鏡頭彼此不一致」,不是「身份隨時間改變」——
      例:`L3S-seq_004-c6+c7-8-008115` 的 camera_7 全程 chef 1、camera_6 全程 chef 3,
      逐鏡頭各 0 / 1 次切換,混排後卻報 359 次(每一幀都算一次)。
      跨鏡頭的不一致已經由 `l3_p_handoff` / `l3_p_simul_agree` 在量,不該在這裡重複計。
      → 主數字改成 **`chef_switches`(逐鏡頭沿時間算再相加)**;
        混排版留成 `chef_switches_interleaved` 備查,單鏡頭片段兩者相同。
    """
    b = [r for r in _series(rows) if r["matched"] and r["chef_id"] != ""]
    inter = [dict(at_fid=y["video_fid"], frm=x["chef_id"], to=y["chef_id"])
             for x, y in zip(b, b[1:]) if x["chef_id"] != y["chef_id"]]
    per_cam = defaultdict(list)
    for r in b:
        per_cam[r["camera_id"]].append(r)
    chef_sw, trk_sw, n_pairs = [], 0, 0
    for cam, rs in per_cam.items():
        rs = sorted(rs, key=lambda r: r["video_fid"])
        chef_sw += [dict(camera=cam, at_fid=y["video_fid"], frm=x["chef_id"], to=y["chef_id"],
                         gap_cells=(y["video_fid"] - x["video_fid"]) // max(y["stride"], 1) - 1)
                    for x, y in zip(rs, rs[1:]) if x["chef_id"] != y["chef_id"]]
        trk_sw += sum(x["track_id"] != y["track_id"] for x, y in zip(rs, rs[1:]))
        n_pairs += max(len(rs) - 1, 0)
    return dict(chef_switches=len(chef_sw),
                chef_switch_rate=round(len(chef_sw) / n_pairs, 4) if n_pairs else None,
                chef_switch_detail=chef_sw,
                chef_switches_interleaved=len(inter),
                track_switches=trk_sw,
                track_switch_rate=round(trk_sw / n_pairs, 4) if n_pairs else None,
                n_track_ids=len({r["track_id"] for r in b}))


def ghost_stats(rows, tracks, chefs, clip, c_star):
    """誤偵。分母是**這段畫出來的系統框總數** —— 看影片時數得到的那個數字。"""
    n_box = sum(r["n_tracks"] for r in rows)
    n_ghost = sum(r["n_ghost"] for r in rows)
    # track 層:窗內配對率 < 0.5 的 track(⚠ 只在窗內算,與 B 套整段定義不同)
    seen, hit = Counter(), Counter()
    for (cam, fid), lst in tracks.items():
        if not (clip["start_fid"] <= fid <= clip["end_fid"]) or cam not in clip["cameras"]:
            continue
        for t, b in lst:
            if b is not None:
                seen[(cam, t)] += 1
    for r in rows:
        if r["matched"]:
            hit[(r["camera_id"], r["track_id"])] += 1
    for r in rows:                      # 也要算「配到別人」的次數
        pass
    ghost_tracks = [k for k, n in seen.items()
                    if hit.get(k, 0) / n < GHOST_MATCH_RATE_MIN]
    stolen = sum(1 for r in rows
                 if r["chef_id"] == c_star and r["gt_present"] and r["matched"]
                 and any(chefs.get((r["camera_id"], t)) == c_star
                         for t, b in tracks.get((r["camera_id"], r["video_fid"]), [])
                         if b is not None and t != r["track_id"]))
    b_n = sum(1 for r in _series(rows) if r["matched"] and r["chef_id"] != "")
    return dict(n_boxes=n_box, n_ghost_boxes=n_ghost,
                ghost_box_rate=round(n_ghost / n_box, 4) if n_box else None,
                n_tracks_in_window=len(seen), n_ghost_tracks=len(ghost_tracks),
                ghost_track_rate=round(len(ghost_tracks) / len(seen), 4) if seen else None,
                p_chef_shared_with_other_box=round(stolen / b_n, 4) if b_n else None)


def _majority(rs, k):
    rs = [r for r in rs if r["matched"] and r["chef_id"] != ""]
    if not rs:
        return None
    take = rs[:k] if k > 0 else rs
    return Counter(r["chef_id"] for r in take).most_common(1)[0][0]


def l2_reacquire(rows, appearances, k=5):
    """離開再回來,有沒有接回同一個編號。

    分母 = **兩側都至少有一個綁定格**的間隔對數;被排除的對數要報出來。
    """
    s = _series(rows)
    segs = [[r for r in s if a <= r["gt_frame"] <= b] for a, b in appearances]
    pairs, ok, dropped = 0, 0, 0
    detail = []
    for (a, b) in zip(segs, segs[1:]):
        before = _majority(list(reversed(a)), k)      # 前段的最後 k 個
        after = _majority(b, k)                       # 後段的最前 k 個
        if before is None or after is None:
            dropped += 1
            continue
        pairs += 1
        ok += int(before == after)
        detail.append(dict(before=before, after=after, same=before == after))
    return dict(n_pairs=pairs, n_pairs_dropped=dropped,
                p_reacquire=round(ok / pairs, 4) if pairs else None,
                p_reacquire_ci=wilson(ok, pairs), detail=detail)


def l3_handoff(rows, cameras):
    """跨鏡頭有沒有給同一個編號。分母 = 兩台都有綁定的鏡頭對數。"""
    s = _series(rows)
    by_cam = {c: [r for r in s if r["camera_id"] == c] for c in cameras}
    maj = {c: _majority(rs, 0) for c, rs in by_cam.items()}
    pairs = [(a, b) for i, a in enumerate(sorted(cameras)) for b in sorted(cameras)[i + 1:]]
    usable = [(a, b) for a, b in pairs if maj[a] is not None and maj[b] is not None]
    agree = [(a, b) for a, b in usable if maj[a] == maj[b]]
    # 同時出現的幀:各台編號是否一致
    by_fid = defaultdict(dict)
    for r in s:
        if r["matched"] and r["chef_id"] != "":
            by_fid[r["video_fid"]][r["camera_id"]] = r["chef_id"]
    simul = [v for v in by_fid.values() if len(v) >= 2]
    simul_ok = [v for v in simul if len(set(v.values())) == 1]
    return dict(chef_per_camera={c: maj[c] for c in sorted(cameras)},
                n_camera_pairs=len(pairs), n_pairs_usable=len(usable),
                n_pairs_dropped=len(pairs) - len(usable),
                p_handoff=round(len(agree) / len(usable), 4) if usable else None,
                p_handoff_ci=wilson(len(agree), len(usable)),
                n_simul_frames=len(simul),
                p_simul_agree=round(len(simul_ok) / len(simul), 4) if simul else None,
                p_simul_agree_ci=wilson(len(simul_ok), len(simul)),
                n_chef_ids_per_camera={c: len({r["chef_id"] for r in rs
                                               if r["matched"] and r["chef_id"] != ""})
                                       for c, rs in by_cam.items()})


def clip_metrics(clip, stride, gt, tracks, chefs):
    """回傳 (逐幀列, 這一段的純量)。所有率都由逐幀列算出,沒有第二條計算路徑。"""
    rows = per_frame_rows(clip, stride, gt, tracks, chefs)
    ident = identity_rates(rows, clip["exclusive"])
    out = dict(clip_id=clip["clip_id"], rule=clip["rule"], level=clip["level"],
               seq=clip["seq"], cameras="+".join(clip["cameras"]), gt_id=clip["gt_id"],
               stride=stride, duration_s=clip["duration_s"],
               exclusive=clip["exclusive"], n_other_ids=clip["n_other_ids"],
               n_sampled_cells=len(rows),
               visibility=round(sum(r["gt_present"] for r in rows) / len(rows), 4) if rows else None,
               **iou_stats(rows), **ident, **switches(rows),
               **ghost_stats(rows, tracks, chefs, clip, ident["c_star"]))
    # 目標拿到編號之前,系統已經開了幾個編號(誤偵搶號的證據)
    before = {c for (cam, t), c in chefs.items()
              if ident["c0"] is not None and c < ident["c0"]}
    out["n_chefs_created_before_target_bound"] = len(before)
    if clip["level"] == "L2":
        for k in (5, 1):
            r = l2_reacquire(rows, [tuple(x) for x in clip["appearances"]], k=k)
            out.update({f"l2_{key}_k{k}": v for key, v in r.items() if key != "detail"})
            out[f"l2_detail_k{k}"] = r["detail"]
        out["gap_band"] = clip["gap_band"]
        out["gaps_s"] = clip["gaps_s"]
    if clip["level"] == "L3":
        out.update({f"l3_{k}": v for k, v in l3_handoff(rows, clip["cameras"]).items()})
        out["handoff_type"] = clip["handoff_type"]
    return rows, out
