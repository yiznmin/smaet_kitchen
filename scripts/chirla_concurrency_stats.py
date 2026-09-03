"""CHIRLA 到底能不能量「誤併率」—— 量同時在場的身份數。

## 為什麼這支非跑不可

整套系統的驗收門檻是「**誤併率 ≤ 1%**」,但我們**從來沒有在真實影片上量過一次**。
EPFL 全片只有一個人 → 誤併(把兩個人併成一個)**結構上不可能發生**,
所以七輪的誤併數字全部來自模擬。

CHIRLA 有 21 個身份,看起來能解決這件事。**但「資料集有 21 個身份」不等於
「同一時刻有多個人在場」** —— 如果它也是一次只拍一個人走動,誤併照樣量不到,
跟 EPFL 同一個下場。

⚠ 已有的 `chirla_overlap_stats.py` 量的是「(幀, 身份) 被幾台相機看到」,
  那是**每個人被幾台看到**,不是**每幀有幾個人**。這兩件事完全不同,
  而後者從來沒有人量過。

## 判準

| 量 | 決定什麼 |
|---|---|
| 同一相機、同一幀 ≥2 個身份 | **M4 的多目標追蹤**能不能測(EPFL 完全測不到) |
| 同一時刻、跨相機 ≥2 個身份 | **M5 的誤併**能不能測 |
| 轉場窗內有幾個「其他人也在動」 | 誤併的**真正**條件 —— 候選要競爭才可能併錯 |
| 5 對真正非重疊的相機對上的轉場 | **跨時轉場路徑**的覆蓋量(我們目前零覆蓋) |

⚠ distractor 用負號 id,`-1` 與 `1` 是同一個人的兩種角色 → 一律取絕對值。

用法:
    python scripts/chirla_concurrency_stats.py --root "D:/.../CHIRLA"
"""
import argparse
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

# 遠端 2026-09-03 實測:21 對相機裡真正從未共現的 5 對。
# 這 5 對之間的移動**只能走跨時轉場路徑** —— 正是我們零覆蓋的那條。
TRULY_DISJOINT = {frozenset(x) for x in
                  [("camera_1", "camera_6"), ("camera_1", "camera_7"),
                   ("camera_2", "camera_6"), ("camera_3", "camera_6"),
                   ("camera_4", "camera_7")]}
FPS = 30.0


def phys(name):
    return "_".join(name.split("_")[:2])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--transit-window-s", type=float, default=60.0,
                    help="判定『轉場』時,離開與再出現之間的最大間隔")
    ap.add_argument("--out", default="results/m5_reid/chirla_concurrency.json")
    args = ap.parse_args()

    aroot = Path(args.root) / "annotations"
    if not aroot.exists():
        raise SystemExit(f"找不到 {aroot}")

    tot_cam_frames = defaultdict(int)     # 同相機同幀的身份數 -> 次數
    tot_world = defaultdict(int)          # 同一時刻全場的身份數 -> 次數
    per_seq, transits, disjoint_transits = {}, [], []
    competing = []                        # 每次轉場時,有幾個其他身份同時在場

    for seq in sorted(p for p in aroot.iterdir() if p.is_dir()):
        cam_frame = defaultdict(set)      # (cam, frame) -> {id}
        world = defaultdict(set)          # frame -> {id}
        seen = defaultdict(list)          # (id, cam) -> [frame]
        for f in sorted(seq.glob("*.json")):
            cam = phys(f.stem)
            for fr, dets in json.loads(f.read_text(encoding="utf-8")).items():
                fr = int(fr)
                for o in dets:
                    i = abs(int(o["id"]))          # distractor 負號 → 同一個人
                    cam_frame[(cam, fr)].add(i)
                    world[fr].add(i)
                    seen[(i, cam)].append(fr)

        s_cam = defaultdict(int)
        for v in cam_frame.values():
            s_cam[len(v)] += 1
            tot_cam_frames[len(v)] += 1
        s_world = defaultdict(int)
        for v in world.values():
            s_world[len(v)] += 1
            tot_world[len(v)] += 1

        # 轉場:同一身份在 cam A 最後出現 → 在 cam B 首次出現,間隔在窗內
        iv = defaultdict(list)
        for (i, cam), frs in seen.items():
            frs.sort()
            start, prev = frs[0], frs[0]
            for x in frs[1:]:
                if x - prev > FPS * 2:            # 中斷超過 2 秒算另一段
                    iv[i].append((cam, start, prev)); start = x
                prev = x
            iv[i].append((cam, start, prev))
        for i, segs in iv.items():
            segs.sort(key=lambda s: s[1])
            for a, b in zip(segs, segs[1:]):
                if a[0] == b[0]:
                    continue
                dt = (b[1] - a[2]) / FPS
                if not (0 <= dt <= args.transit_window_s):
                    continue
                transits.append(dt)
                # 轉場當下,除了本人還有幾個身份在全場?這才是誤併的必要條件
                others = {x for fr in range(a[2], b[1] + 1) for x in world.get(fr, ())} - {i}
                competing.append(len(others))
                if frozenset((a[0], b[0])) in TRULY_DISJOINT:
                    disjoint_transits.append(dt)

        per_seq[seq.name] = {
            "cam_frames_with_multi": sum(v for k, v in s_cam.items() if k >= 2),
            "cam_frames_total": sum(s_cam.values()),
            "world_frames_with_multi": sum(v for k, v in s_world.items() if k >= 2),
            "world_frames_total": sum(s_world.values()),
            "max_concurrent": max(s_world) if s_world else 0,
        }

    def show(title, hist):
        tot = sum(hist.values())
        multi = sum(v for k, v in hist.items() if k >= 2)
        print(f"\n  {title}")
        for k in sorted(hist):
            print(f"    {k} 個身份: {hist[k]:>9,}  ({hist[k]/tot:>5.1%})")
        print(f"    → ≥2 個身份 **{multi:,} / {tot:,} = {multi/tot:.1%}**")
        return multi / tot if tot else 0.0

    print("=" * 74)
    print("CHIRLA 同時在場身份數 —— 決定誤併率能不能在真實影片上量到")
    print("=" * 74)
    r_cam = show("同一相機、同一幀(決定 M4 多目標追蹤能不能測):", tot_cam_frames)
    r_world = show("同一時刻、全場跨相機(決定 M5 誤併能不能測):", tot_world)

    print(f"\n  跨相機轉場(間隔 ≤ {args.transit_window_s:.0f} 秒):{len(transits):,} 次")
    if transits:
        print(f"    間隔中位 {st.median(transits):.1f}s"
              f" / p90 {sorted(transits)[int(len(transits)*.9)]:.1f}s")
        print(f"    ⚠ 對照:驗收假設是「每 2~5 分鐘跨一次」(120~300s)")
    print(f"  其中發生在**5 對真正非重疊**相機上的:{len(disjoint_transits):,} 次"
          f"  ← 跨時轉場路徑的覆蓋量(目前零覆蓋)")
    if competing:
        n0 = sum(1 for c in competing if c == 0)
        print(f"\n  轉場當下同時在場的**其他**身份數:中位 {st.median(competing):.0f}"
              f" / 最多 {max(competing)}")
        print(f"    其中「全場只有這一個人」的轉場:{n0:,} / {len(competing):,}"
              f" = {n0/len(competing):.1%}")
        print("    ⚠ 這種轉場**量不到誤併** —— 沒有別人可以併錯,與 EPFL 同一個限制")

    print("\n" + "─" * 74)
    ok = r_world >= 0.10 and len(transits) > 100
    print("判定:" + ("✅ CHIRLA 可以量誤併率 —— 這是 EPFL 做不到的"
                     if ok else
                     "❌ 同時在場的身份太少,誤併率在 CHIRLA 上也量不到"))
    print("─" * 74)

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"same_camera_hist": dict(tot_cam_frames), "world_hist": dict(tot_world),
         "per_seq": per_seq, "n_transits": len(transits),
         "n_disjoint_transits": len(disjoint_transits),
         "transit_dt_s": {"median": st.median(transits) if transits else None},
         "competing_identities": {"median": st.median(competing) if competing else None,
                                  "n_alone": sum(1 for c in competing if c == 0)},
         "verdict_can_measure_false_merge": ok, "args": vars(args)},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已存 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
