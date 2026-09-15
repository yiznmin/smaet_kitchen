"""M4「② 貼著人但框不準」的誤偵 track:框為什麼不準 —— 探索性診斷,非預先登記(2026-09-15)。

## 為什麼

排除 camera_5、6 後,5 台 `cbiou` 的誤偵 506 條裡 **212 條(41.9%)是 ②**:
框在真人身上(IoU ≥ 0.1 的幀過半)卻配不到(IoU < 0.5)。這類不能用過濾處理,
要先知道框為什麼不準,才知道修法打哪裡。

## 量什麼(規則執行前寫死)

對每條 ② track、每一幀「與某人框 IoU ≥ 0.1」的幀,取 IoU 最大的那個人框 G、track 的框 T:

  框與人的關係(依序判定)
    偏小     T 面積 < 0.5 × G 面積,且 T 有 ≥ 70% 落在 G 內
             依 T 中心相對 G 中心的垂直位移(÷ G 高):< −0.15 上半身、> +0.15 下半身、其餘中段
    偏大     T 面積 > 2 × G 面積,且 G 有 ≥ 70% 被 T 包住
    位置偏移 其餘

  遮擋     同一幀有別人的標註框蓋住 G 的 ≥ 10% 面積
  跨兩人   T 與另一個人的標註框 IoU ≥ 0.1
  切邊     框距畫面邊緣 ≤ 5 px(G 與 T 分開算)

track 層級:關係取多數幀;遮擋、跨兩人、切邊取「≥ 50% 的幀」。
**對照組**:真人 track 在配對幀上的遮擋比例 —— 沒有對照就分不出「② 常被遮擋」與「本來就常有遮擋」。

## 口徑

② 的判定**直接 import `diag_m4_ghosts`**(分類、逐幀配對與它完全相同),並自檢條數與既有結果一致。

用法:
    .venv/Scripts/python.exe scripts/diag_m4_boxes.py --root <CHIRLA根> \\
        --tracks-dir results/m4_5cam/cbiou_dump --seqs seq_004 ... \\
        --ghosts-json results/m4_5cam/ghosts_cbiou.json --out results/m4_5cam/boxes_cbiou.json
"""
import argparse
import glob
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from diag_m4_ghosts import CATS, NEAR_IOU, features, load_rows, per_frame_assign   # noqa: E402
from eval_m4m5_chirla import iou, load_gt, match_tracks                         # noqa: E402

SMALL_AREA = 0.5
LARGE_AREA = 2.0
INSIDE = 0.7
UPPER = 0.15
OCC_COVER = 0.1
SPAN_IOU = 0.1
BORDER_PX = 5.0
KINDS = ("偏小:上半身", "偏小:中段", "偏小:下半身", "偏大:包住人", "位置偏移")


def area(b):
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def inter(a, b):
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    return w * h if w > 0 and h > 0 else 0.0


def relation(T, G):
    aT, aG, it = area(T), area(G), inter(T, G)
    r = aT / aG if aG > 0 else float("inf")
    dy = ((T[1] + T[3]) / 2 - (G[1] + G[3]) / 2) / max(G[3] - G[1], 1e-9)
    if aT > 0 and r < SMALL_AREA and it / aT >= INSIDE:
        return ("偏小:上半身" if dy < -UPPER else "偏小:下半身" if dy > UPPER else "偏小:中段"), r, dy
    if aG > 0 and r > LARGE_AREA and it / aG >= INSIDE:
        return "偏大:包住人", r, dy
    return "位置偏移", r, dy


def at_border(b, w, h):
    return b[0] <= BORDER_PX or b[1] <= BORDER_PX or b[2] >= w - BORDER_PX or b[3] >= h - BORDER_PX


def frame_size(root, seq, cam):
    import cv2
    vid = sorted(glob.glob(f"{root}/videos/{seq}/{cam}_*.avi"))[0]
    cap = cv2.VideoCapture(vid)
    w, h = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    cap.release()
    return w, h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--tracks-dir", required=True)
    ap.add_argument("--seqs", nargs="+", required=True)
    ap.add_argument("--ghosts-json", required=True, help="diag_m4_ghosts.py 的輸出,自檢 ② 條數用")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    per_track = []                 # 每條 ② track 的彙總
    frame_kind = Counter()         # 所有 ② 幀的關係
    occ_frames = Counter()         # ②:遮擋幀 / 總幀
    real_occ = Counter()           # 真人 track 配對幀:遮擋幀 / 總幀
    sizes = {}
    for seq in args.seqs:
        gt = load_gt(args.root, seq)
        rows = load_rows(Path(args.tracks_dir) / seq / "tracks.csv")
        track_gt, *_ = match_tracks(gt, rows)
        info, _ = per_frame_assign(gt, rows)
        by = defaultdict(list)
        for r in rows:
            by[(r["cam"], r["tid"])].append(r)
        for (cam, tid), rs in by.items():
            if (seq, cam) not in sizes:
                sizes[(seq, cam)] = frame_size(args.root, seq, cam)
            W, H = sizes[(seq, cam)]
            box_at = {r["fid"]: r["bbox"] for r in rs}
            fr = sorted(info[(cam, tid)])
            gid = track_gt[(cam, tid)]

            if gid is not None:
                # 對照組:真人 track 在「有配到人」的幀上,那個人被別人蓋住的比例
                for fid, _bi, _bg, mg in fr:
                    if mg is None:
                        continue
                    boxes = gt[cam][fid + 1]
                    G = dict(boxes)[mg]
                    occ = any(inter(G, O) / area(G) >= OCC_COVER for og, O in boxes if og != mg) if area(G) > 0 else False
                    real_occ["frames"] += 1
                    real_occ["occluded"] += occ
                continue

            f = features(fr, [r["bbox"] for r in rs], [r["conf"] for r in rs])
            if f["cat"] != CATS[1]:
                continue
            kinds, occ_n, span_n, cutG_n, cutT_n, n = Counter(), 0, 0, 0, 0, 0
            ratios, dys = [], []
            for fid, bi, bg, _mg in fr:
                if bi < NEAR_IOU:
                    continue
                boxes = gt[cam][fid + 1]
                G, T = dict(boxes)[bg], box_at[fid]
                k, r, dy = relation(T, G)
                kinds[k] += 1
                ratios.append(r)
                dys.append(dy)
                others = [O for og, O in boxes if og != bg]
                occ = area(G) > 0 and any(inter(G, O) / area(G) >= OCC_COVER for O in others)
                occ_n += occ
                span_n += any(iou(T, O) >= SPAN_IOU for O in others)
                cutG_n += at_border(G, W, H)
                cutT_n += at_border(T, W, H)
                n += 1
            frame_kind.update(kinds)
            occ_frames["frames"] += n
            occ_frames["occluded"] += occ_n
            ratios.sort()
            dys.sort()
            per_track.append(dict(
                seq=seq, cam=cam, tid=tid, n_near=n, kind=kinds.most_common(1)[0][0],
                occluded=occ_n / n >= 0.5, spans_two=span_n / n >= 0.5,
                cut_gt=cutG_n / n >= 0.5, cut_track=cutT_n / n >= 0.5,
                area_ratio_median=ratios[len(ratios) // 2], dy_median=dys[len(dys) // 2],
                conf_mean=f["conf_mean"], seconds=f["n"] * 5 / 30))

    ref = json.loads(Path(args.ghosts_json).read_text(encoding="utf-8"))["ghost_cats"][CATS[1]]
    if len(per_track) != ref:
        raise SystemExit(f"[FAIL] ② 條數 {len(per_track)} ≠ {args.ghosts_json} 的 {ref}")
    n = len(per_track)
    print(f"② 貼著人但框不準:{n} 條(與 {args.ghosts_json} 相同 [OK]);畫面尺寸 {sorted(set(sizes.values()))}")

    print("\n1. 框與人的關係(track 層級取多數幀)")
    tk = Counter(t["kind"] for t in per_track)
    for k in KINDS:
        print(f"   {k:<10} {tk[k]:>4}({tk[k] / n:.1%})")
    tf = sum(frame_kind.values())
    print("   幀層級:" + "、".join(f"{k} {frame_kind[k] / tf:.1%}" for k in KINDS))

    print("\n2. 遮擋、跨兩人、切邊(≥ 50% 的幀)")
    occ_t = sum(t["occluded"] for t in per_track)
    span_t = sum(t["spans_two"] for t in per_track)
    cutg = sum(t["cut_gt"] for t in per_track)
    cutt = sum(t["cut_track"] for t in per_track)
    ro = real_occ["occluded"] / real_occ["frames"] if real_occ["frames"] else 0.0
    po = occ_frames["occluded"] / occ_frames["frames"] if occ_frames["frames"] else 0.0
    print(f"   被別人遮擋的 track     {occ_t:>4}({occ_t / n:.1%})")
    print(f"   幀層級遮擋率:② {po:.1%}  vs  真人 track 配對幀 {ro:.1%}(對照)")
    print(f"   框跨到另一個人         {span_t:>4}({span_t / n:.1%})")
    print(f"   人框被畫面邊緣切到     {cutg:>4}({cutg / n:.1%})")
    print(f"   track 框被畫面邊緣切到 {cutt:>4}({cutt / n:.1%})")

    print("\n3. 各鏡頭")
    for cam in sorted({t["cam"] for t in per_track}):
        ts = [t for t in per_track if t["cam"] == cam]
        c = Counter(t["kind"] for t in ts)
        print(f"   {cam:<10} {len(ts):>4} 條  遮擋 {sum(t['occluded'] for t in ts) / len(ts):>5.1%}  "
              + "  ".join(f"{k}:{c[k]}" for k in KINDS))

    print("\n4. 假說對照(執行前寫下)")
    small = tk["偏小:上半身"] + tk["偏小:中段"] + tk["偏小:下半身"]
    print(f"   H1 大多數在被別人遮擋時發生:遮擋 track {occ_t / n:.1%} → {'成立' if occ_t / n > 0.5 else '不成立'}"
          f"(對照:② 幀遮擋率 {po:.1%} vs 真人 {ro:.1%})")
    print(f"   H2 偏小(含上半身)多於偏大:偏小 {small}({small / n:.1%})vs 偏大 {tk['偏大:包住人']}"
          f"({tk['偏大:包住人'] / n:.1%}) → {'成立' if small > tk['偏大:包住人'] else '不成立'}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(
        args=vars(args),
        rules=dict(SMALL_AREA=SMALL_AREA, LARGE_AREA=LARGE_AREA, INSIDE=INSIDE, UPPER=UPPER,
                   OCC_COVER=OCC_COVER, SPAN_IOU=SPAN_IOU, BORDER_PX=BORDER_PX, NEAR_IOU=NEAR_IOU),
        n=n, kinds_track={k: tk[k] for k in KINDS}, kinds_frame={k: frame_kind[k] for k in KINDS},
        occluded_tracks=occ_t, spans_two_tracks=span_t, cut_gt_tracks=cutg, cut_track_tracks=cutt,
        occlusion_frame_rate=dict(ghost2=po, real=ro, ghost2_frames=occ_frames["frames"],
                                  real_frames=real_occ["frames"]),
        tracks=per_track), ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n已存 {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
