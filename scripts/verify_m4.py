"""M4 追蹤器合成驗證(免 rfdetr、免 GPU,現在就能跑)。

手工造 supervision Detections 序列餵 KitchenTracker,斷言追蹤行為:
  S1 單框移動:1 個 new_track、id 全程不變。
  S2 兩框交叉:2 個 new_track、交叉後各自延續、ID-switch==0(EPFL 單人給不了的多目標檢查)。
  S3 遮擋恢復:消失 5–8 幀(< lost_track_buffer)→ 同 id 復現;負例 gap > buffer → removed + 新 id。
  S4 M5 事件契約:復現要發 reacquired、事件要帶 camera_id / t_sec(M5 跨鏡頭與轉場時間窗依賴)。
  S5 從未確認的誤偵不發 removed。
  S6 rf_cbiou 的 buffer 設 0 與 rf_botsort 逐位相同(網格裡兩格才只差框的放大)。

S1~S5 對每個 backend 各跑一次(預設全部;`--backends bytetrack` 只跑 supervision)。
rf_* 需要 `pip install trackers==2.6.0`;裝不起來會直接 FAIL,不會悄悄跳過。

任何一項失敗 → exit 1(CI 友善)。
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import supervision as sv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from m4_track import KitchenTracker   # noqa: E402
from m4_track.tracker import RF_BACKENDS   # noqa: E402

BACKENDS = ("bytetrack",) + RF_BACKENDS
_BACKEND = "bytetrack"          # main() 逐一切換


def mk(boxes, cls=0, conf=0.9):
    if not boxes:
        return sv.Detections.empty()
    xyxy = np.array(boxes, dtype=float)
    confs = np.full(len(boxes), conf, dtype=float) if np.isscalar(conf) else np.asarray(conf, float)
    return sv.Detections(xyxy=xyxy, confidence=confs,
                         class_id=np.full(len(boxes), cls, dtype=int))


def new_tracker(**kw):
    base = dict(backend=_BACKEND, track_activation_threshold=0.25, lost_track_buffer=8,
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
        if id_left_start is None and len(out.tracks) == 2:
            # ⚠ 不寫死第 0 幀:OC-SORT 第 0 幀只開未確認 track,第 1 幀才發 id
            id_left_start = _id_at(out.tracks, ax + 18, 120)      # 起點左邊那顆(A)
            id_right_start = _id_at(out.tracks, bx + 18, 120)     # 起點右邊那顆(B)
        if f == N - 1:
            id_left_end = _id_at(out.tracks, bx + 18, 120)        # 終點左邊(此時是 B)
            id_right_end = _id_at(out.tracks, ax + 18, 120)       # 終點右邊(此時是 A)
    ok = check("2 個 new_track", new_ct == 2, f"got {new_ct}")
    # A 從左到右:起點左=A、終點右=A → 同 id;B 反之
    ok &= check("A 交叉後仍同 id(無 ID-switch)",
                id_left_start is not None and id_left_start == id_right_end,
                f"A start={id_left_start} end={id_right_end}")
    ok &= check("B 交叉後仍同 id(無 ID-switch)",
                id_right_start is not None and id_right_start == id_left_end,
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


def s4_reacquired_and_m5_contract():
    """M5 依賴的事件契約:短暫遮擋要發 reacquired,且事件要帶 camera_id / t_sec。

    沒有 reacquired 的話,M5 會在 lost_track 把該 chef 標成 gone 之後永遠卡住
    ——因為 new_track 只認 active_ids - _seen_ids,復現的 track 不會再發 new_track。
    """
    print("S4 遮擋復現發 reacquired(M5 離場語意)")
    tk = new_tracker(lost_track_buffer=10, camera_id="cam1")
    kinds, all_events = [], []
    for f in range(27):                        # 同 S3:出現 0-9、消失 10-16、復現 17-26
        if f < 10 or f >= 17:
            x = 100 + (f if f < 10 else f - 7) * 2
            out = tk.update(mk([[x, 100, x + 40, 140]]), f)
        else:
            out = tk.update(mk([]), f)
        kinds += [e.kind for e in out.events]
        all_events += out.events
    ok = check("復現發 1 個 reacquired", kinds.count("reacquired") == 1, f"kinds={kinds}")
    ok &= check("未超 buffer 不發 removed", kinds.count("removed") == 0, f"kinds={kinds}")
    ok &= check("順序為 lost_track → reacquired",
                "lost_track" in kinds and "reacquired" in kinds
                and kinds.index("lost_track") < kinds.index("reacquired"), f"kinds={kinds}")
    ok &= check("所有事件都帶 camera_id",
                all(e.camera_id == "cam1" for e in all_events),
                f"{[e.camera_id for e in all_events]}")
    ok &= check("t_sec 由 frame_id/frame_rate 換算",
                all(abs(e.t_sec - e.frame_id / 30.0) < 1e-9 for e in all_events),
                f"{[(e.frame_id, e.t_sec) for e in all_events]}")

    print("S4b 遮擋過久 → 只發 removed,不發 reacquired")
    tk2 = new_tracker(lost_track_buffer=5, camera_id="cam2")
    kinds2 = []
    for f in range(40):
        out = tk2.update(mk([[120, 100, 160, 140]]) if (f < 8 or f >= 30) else mk([]), f)
        kinds2 += [e.kind for e in out.events]
    ok &= check("不發 reacquired", kinds2.count("reacquired") == 0, f"kinds={kinds2}")
    ok &= check("發 removed", kinds2.count("removed") >= 1, f"kinds={kinds2}")

    print("S4c 顯式 timestamp 覆蓋換算值")
    # ⚠ 餵兩幀:OC-SORT 第一幀只開未確認 track、不發事件
    tk3 = new_tracker(camera_id="cam3")
    got = []
    for f, ts in ((5, 123.5), (6, 123.6)):
        got += [(e.kind, e.t_sec, ts) for e in tk3.update(mk([[10, 10, 50, 50]]), f, timestamp=ts).events]
    ok &= check("t_sec 用傳入的 timestamp",
                len(got) == 1 and all(t == ts for _k, t, ts in got),
                f"{[(k, t) for k, t, _ in got]}")
    return ok


def _unconfirmed(tr):
    """未確認 track 的身分集合。bytetrack 回 None(改看 removed_tracks 的 NO_ID)。"""
    if tr._rf is None:
        return None
    return {id(t) for t in tr._rf.tracks if t.tracker_id == -1}


def s5_unconfirmed_not_removed():
    """S5:從未啟動的偵測不得發 removed 事件(2026-09-14 的 bug)。

    supervision 對只出現一幀、沒被確認的 track 給 external id = NO_ID(-1),
    而 removed_tracks 逐幀覆寫 → -1 反覆進出集合 → 舊版每次都發 removed -1。
    CHIRLA 實測 7,779 次 removed 裡 4,192 次是這種。

    ⚠ 尾端多跑 buffer + 2 幀:trackers 的 OC-SORT 不刪未確認 track,要等緩衝到期。
    ⚠ A 全程都在,所以**任何** removed 都是錯的(比只查 -1 更嚴)。
    """
    print("S5 只出現一幀的誤偵 → 不發 removed")
    tr = new_tracker()
    raw_unconfirmed_removed = removed_ct = new_ct = 0
    for f in range(16 + 8 + 2):
        boxes = [[100 + f, 100, 150 + f, 250]]            # A:穩定存在
        if f in (4, 8, 12):
            boxes.append([600, 100, 650, 250])            # 遠處只出現一幀的誤偵
        before = _unconfirmed(tr)
        out = tr.update(mk(boxes), frame_id=f)
        # ⚠ 先確認情境真的觸發了未確認 track 的移除,否則下面的斷言是空轉
        if before is None:
            no_id = tr._bt.external_id_counter.NO_ID
            raw_unconfirmed_removed += sum(1 for t in tr._bt.removed_tracks
                                           if t.external_track_id == no_id)
        else:
            raw_unconfirmed_removed += len(before - {id(t) for t in tr._rf.tracks})
        removed_ct += sum(1 for e in out.events if e.kind == "removed")
        new_ct += sum(1 for e in out.events if e.kind == "new_track")
    ok = check("情境確實產生未確認 track 的移除(否則此測試無效)", raw_unconfirmed_removed > 0,
               f"backend 內部移除 {raw_unconfirmed_removed} 次")
    ok &= check("不發 removed", removed_ct == 0, f"got {removed_ct}")
    ok &= check("誤偵從未成為 track → 只有 A 一個 new_track", new_ct == 1, f"got {new_ct}")
    return ok


def s6_cbiou_zero_buffer_equals_botsort():
    """rf_cbiou 的 buffer 設 0 必須與 rf_botsort 逐位相同。

    這是網格裡 `cbiou` 對 `rf_botsort` 的前提:兩格只差框的放大。
    情境要夠亂才有意義:四個人、隨機漏偵、高低分混雜 —— 並斷言真的出現了 lost 與 reacquired。
    """
    print("S6 rf_cbiou(buffer 0)≡ rf_botsort")
    rng = np.random.default_rng(20260915)
    frames = []
    for f in range(80):
        boxes, confs = [], []
        for k in range(4):
            if rng.random() < 0.25:
                continue
            x = 40 + k * 90 + 30 * np.sin(f / (6 + k)) + rng.normal(0, 2)
            y = 60 + 10 * k + f * (1 + k * 0.5) + rng.normal(0, 2)
            boxes.append([x, y, x + 45, y + 110])
            confs.append(float(rng.uniform(0.12, 0.95)))
        frames.append((boxes, confs))

    def run(backend, params):
        tk = new_tracker(backend=backend, backend_params=params)
        log = []
        for f, (boxes, confs) in enumerate(frames):
            out = tk.update(mk(boxes, conf=confs) if boxes else mk([]), f)
            log.append(([(e.kind, e.track_id, e.bbox) for e in out.events],
                        [(t.track_id, t.bbox, t.confidence) for t in out.tracks]))
        return log

    a = run("rf_botsort", None)
    b = run("rf_cbiou", dict(buffer_ratio_first=0.0, buffer_ratio_second=0.0))
    kinds = [k for evs, _ in a for k, _t, _b in evs]
    ok = check("情境有 lost 與 reacquired(否則此測試無效)",
               "lost_track" in kinds and "reacquired" in kinds,
               f"new={kinds.count('new_track')} lost={kinds.count('lost_track')} "
               f"reacq={kinds.count('reacquired')} removed={kinds.count('removed')}")
    diff = next((f for f, (x, y) in enumerate(zip(a, b)) if x != y), None)
    ok &= check("80 幀的事件與軌跡逐位相同", diff is None, f"第一個不同在第 {diff} 幀")
    return ok


def main():
    global _BACKEND
    ap = argparse.ArgumentParser()
    ap.add_argument("--backends", nargs="+", default=list(BACKENDS), choices=BACKENDS)
    args = ap.parse_args()

    results = []
    for backend in args.backends:
        _BACKEND = backend
        print(f"\n===== backend = {backend} =====")
        try:
            new_tracker()
        except ImportError as e:
            print(f"  [FAIL] 無法建立 {backend}:{e}\n"
                  f"         → pip install trackers==2.6.0,或加 --backends bytetrack")
            results.append(False)
            continue
        results += [s1_single_moving(), s2_two_crossing(), s3_occlusion(),
                    s4_reacquired_and_m5_contract(), s5_unconfirmed_not_removed()]
    if {"rf_botsort", "rf_cbiou"} <= set(args.backends):
        _BACKEND = "rf_botsort"
        print("\n===== 跨 backend =====")
        try:
            results.append(s6_cbiou_zero_buffer_equals_botsort())
        except ImportError as e:
            print(f"  [FAIL] {e}")
            results.append(False)
    print()
    if all(results):
        print("[ALL PASS] 全部通過")
        sys.exit(0)
    print("[FAILED] 有失敗項")
    sys.exit(1)


if __name__ == "__main__":
    main()
