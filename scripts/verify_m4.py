"""M4 追蹤器合成驗證(免 rfdetr、免 GPU,現在就能跑)。

手工造 supervision Detections 序列餵 KitchenTracker,斷言追蹤行為:
  S1 單框移動:1 個 new_track、id 全程不變。
  S2 兩框交叉:2 個 new_track、交叉後各自延續、ID-switch==0(EPFL 單人給不了的多目標檢查)。
  S3 遮擋恢復:消失 5–8 幀(< lost_track_buffer)→ 同 id 復現;負例 gap > buffer → removed + 新 id。

任何一項失敗 → exit 1(CI 友善)。
"""
import sys
from pathlib import Path

import numpy as np
import supervision as sv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from m4_track import KitchenTracker   # noqa: E402


def mk(boxes, cls=0, conf=0.9):
    if not boxes:
        return sv.Detections.empty()
    xyxy = np.array(boxes, dtype=float)
    return sv.Detections(xyxy=xyxy,
                         confidence=np.full(len(boxes), conf, dtype=float),
                         class_id=np.full(len(boxes), cls, dtype=int))


def new_tracker(**kw):
    base = dict(track_activation_threshold=0.25, lost_track_buffer=8,
                minimum_matching_threshold=0.8, frame_rate=30, minimum_consecutive_frames=1)
    base.update(kw)
    return KitchenTracker(**base)


def _id_at(tracks, cx, cy):
    """回傳中心最接近 (cx,cy) 的軌跡 id。"""
    best, bd = None, 1e9
    for t in tracks:
        if t.bbox is None:
            continue
        tx, ty = (t.bbox[0] + t.bbox[2]) / 2, (t.bbox[1] + t.bbox[3]) / 2
        d = (tx - cx) ** 2 + (ty - cy) ** 2
        if d < bd:
            bd, best = d, t.track_id
    return best


def check(name, cond, extra=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f" — {extra}" if extra else ""))
    return cond


def s1_single_moving():
    print("S1 單框移動")
    tk = new_tracker()
    new_ct, ids = 0, set()
    for f in range(20):
        x = 10 + f * 20
        out = tk.update(mk([[x, 100, x + 40, 140]]), f)
        new_ct += sum(1 for e in out.events if e.kind == "new_track")
        ids |= {t.track_id for t in out.tracks}
    ok = check("只 1 個 new_track", new_ct == 1, f"got {new_ct}")
    ok &= check("全程單一 id", len(ids) == 1, f"ids={ids}")
    return ok


def s2_two_crossing():
    print("S2 兩框交叉(核心多目標檢查)")
    tk = new_tracker()
    new_ct = 0
    id_left_start = id_right_start = None
    id_left_end = id_right_end = None
    N = 20
    for f in range(N):
        # A 由左往右、B 由右往左,中段交會
        ax = 10 + f * 18
        bx = 370 - f * 18
        out = tk.update(mk([[ax, 100, ax + 36, 140], [bx, 100, bx + 36, 140]]), f)
        new_ct += sum(1 for e in out.events if e.kind == "new_track")
        if f == 0:
            id_left_start = _id_at(out.tracks, 10 + 18, 120)      # 起點左邊那顆(A)
            id_right_start = _id_at(out.tracks, 370 + 18, 120)    # 起點右邊那顆(B)
        if f == N - 1:
            axl = 10 + (N - 1) * 18
            bxl = 370 - (N - 1) * 18
            id_left_end = _id_at(out.tracks, bxl + 18, 120)       # 終點左邊(此時是 B)
            id_right_end = _id_at(out.tracks, axl + 18, 120)      # 終點右邊(此時是 A)
    ok = check("2 個 new_track", new_ct == 2, f"got {new_ct}")
    # A 從左到右:起點左=A、終點右=A → 同 id;B 反之
    ok &= check("A 交叉後仍同 id(無 ID-switch)", id_left_start == id_right_end,
                f"A start={id_left_start} end={id_right_end}")
    ok &= check("B 交叉後仍同 id(無 ID-switch)", id_right_start == id_left_end,
                f"B start={id_right_start} end={id_left_end}")
    return ok


def s3_occlusion():
    print("S3 遮擋恢復")
    tk = new_tracker(lost_track_buffer=10)
    ids, lost_evt, new_ct = [], 0, 0
    # 出現 0-9,消失 10-16(7 幀 < buffer 10),復現 17-26 於原位附近
    for f in range(27):
        if f < 10 or f >= 17:
            x = 100 + (f if f < 10 else f - 7) * 2
            out = tk.update(mk([[x, 100, x + 40, 140]]), f)
        else:
            out = tk.update(mk([]), f)          # 空幀
        new_ct += sum(1 for e in out.events if e.kind == "new_track")
        lost_evt += sum(1 for e in out.events if e.kind == "lost_track")
        if out.tracks:
            ids.append(out.tracks[0].track_id)
    ok = check("gap 間發 lost_track", lost_evt >= 1, f"lost={lost_evt}")
    ok &= check("復現沿用同 id(不新增)", len(set(ids)) == 1 and new_ct == 1,
                f"ids={set(ids)} new={new_ct}")

    print("S3b 遮擋過久(gap > buffer)→ 新 id")
    tk2 = new_tracker(lost_track_buffer=5)
    ids2, rem_evt, new_ct2 = [], 0, 0
    for f in range(40):
        if f < 8 or f >= 30:                    # 消失 8-29(22 幀 >> buffer 5)
            out = tk2.update(mk([[120, 100, 160, 140]]), f)
        else:
            out = tk2.update(mk([]), f)
        rem_evt += sum(1 for e in out.events if e.kind == "removed")
        new_ct2 += sum(1 for e in out.events if e.kind == "new_track")
        if out.tracks:
            ids2.append(out.tracks[0].track_id)
    ok &= check("發 removed 且復現為新 id", rem_evt >= 1 and new_ct2 == 2,
                f"removed={rem_evt} new={new_ct2} ids={set(ids2)}")
    return ok


def main():
    results = [s1_single_moving(), s2_two_crossing(), s3_occlusion()]
    print()
    if all(results):
        print("[ALL PASS] 全部通過")
        sys.exit(0)
    print("[FAILED] 有失敗項")
    sys.exit(1)


if __name__ == "__main__":
    main()
