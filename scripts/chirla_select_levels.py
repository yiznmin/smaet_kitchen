"""挑出三個難度層級的評估窗 —— 只讀真值標註,不看任何追蹤結果。

## 三個 Level(2026-09-16 交付需求)

    Level 1  單人、單一鏡頭、**持續**出現
    Level 2  單人、單一鏡頭、**非持續**出現(離開後再回來)
    Level 3  單人、**跨 2~3 台鏡頭**出現

L3 有兩種,**分開報**(2026-09-16 使用者決定):

    L3S  同時出現 —— 同一時刻被 2~3 台看到(重疊視野)
    L3T  先後出現 —— 離開 A 鏡頭、過一陣子出現在 B 鏡頭(轉場)

兩者的失敗方向相反(實測:重疊路徑高誤併低碎裂、轉場路徑相反),平均起來不描述任何真實部署。

## ⚠ 為什麼只讀真值

難度必須是**影片本身的性質**,不能是受測系統的函數。
`match_tracks` 是用 IoU 多數決把 track 配到真值身份的,而 track id 會隨追蹤器改變
(`make_track_gt.py` 檔頭就寫了這件事)。若挑窗時看了追蹤輸出,換一次追蹤器就會換一批
「Level 1 窗」,兩個版本的數字**不再可比**,而且錯得很安靜:每一版自己看都很合理。

只讀真值還有一個好處:manifest 可以在算任何指標之前先提交,符合本專案的預先登記紀律。

## ⚠ 全部定義在取樣格點上

跑的時候是 stride 5,所以 `tracks.csv` 只有 `video_fid ∈ {0,5,10,…}`,
而標註幀號是 **1-based**(GT 幀 = video_fid + 1)。若在原始幀上定義窗,
窗內可能只有寥寥幾個可評估的取樣點,算出來的指標是雜訊卻看起來像分數。

## ⚠ 窗有長度上限

人可能在同一台鏡頭裡待好幾分鐘。不設上限的話一個窗會變成十幾分鐘的影片,沒有人會看完,
而且一段極長的窗會主導該層級的彙總數字。所以一律截到 `--max-window-s`,
**但關鍵事件(L2 的消失期間、L3T 的轉場間隔)一定完整保留**,前後平均分配剩下的時間。

## 判定

  獨佔  窗內每個取樣幀、每台選中的鏡頭,標註裡只有目標這一個人
  L1    單鏡頭、獨佔、合併後只有一個在場區段,長度 20~60 秒
  L2    單鏡頭、獨佔(**連消失期間也不能有別人**)、≥2 個區段,
        每段 ≥3 秒,間隔 2~30 秒;依間隔分兩帶:≤5 秒(M4 的 lost buffer 內)
        與 >5 秒(M4 救不回,只能靠 M5)
  L3S   2~3 台鏡頭的在場區段在時間上重疊 ≥ 5 秒
  L3T   相鄰兩段分屬不同鏡頭、時間不重疊、Δt ≤ 30 秒
        ⚠ 獨佔在 L3 是**屬性不是條件** —— CHIRLA 全場只有一個人的時刻只有 1 個,
          硬性要求會挑不到任何窗。獨佔的優先拿去拍影片,非獨佔的才算得出誤併率
          (誤併需要場上有 ≥2 個身份才存在)。

用法:
    python scripts/chirla_select_levels.py --root "D:/.../CHIRLA" \\
        --train-seqs seq_000 seq_001 seq_002 \\
        --test-seqs seq_004 seq_006 seq_007 seq_024 seq_025 seq_026 \\
        --extra-seqs seq_020 --out results/levels/level_manifest.json
"""
import argparse
import json
import subprocess
import sys
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from chirla_build_topology import segments                      # noqa: E402
from chirla_concurrency_stats import TRULY_DISJOINT             # noqa: E402
from eval_m4m5_chirla import load_gt                            # noqa: E402

FPS = 30.0


def sampled_frames(stride, lo, hi):
    """窗內的取樣真值幀號(1-based)。video_fid 是 stride 的倍數 → GT 幀 ≡ 1 (mod stride)。"""
    first = lo + ((1 - lo) % stride)
    return range(first, hi + 1, stride)


def trim(lo, hi, keep_lo, keep_hi, max_frames, stride):
    """把 [lo, hi] 截到 max_frames 內,但完整保留 [keep_lo, keep_hi]。

    ⚠ 起點必須留在取樣格點上(與 lo 同餘),否則窗內就沒有可評估的幀。
    keep 本身就超過上限時不截 —— 寧可影片長一點,也不要把要展示的事件切掉。
    """
    if hi - lo <= max_frames:
        return lo, hi, False
    room = max_frames - (keep_hi - keep_lo)
    if room <= 0:
        return keep_lo, keep_hi, True
    nlo = max(lo, keep_lo - room // 2)
    nlo = lo + ((nlo - lo) // stride) * stride          # 對回取樣格點
    return nlo, min(hi, nlo + max_frames), True


def build_tables(gt, stride):
    """回傳 (occ, pres):(鏡頭,幀)->{身份} 與 (身份,鏡頭)->[取樣幀]。

    occ 只收取樣幀 —— 獨佔是在「評估看得到的幀」上判定的。非取樣幀上有別人
    不影響任何指標,把它算進來只會讓可用素材無謂變少。
    """
    occ = defaultdict(set)
    pres = defaultdict(list)
    for cam, per_frame in gt.items():
        for fr, dets in per_frame.items():
            if (fr - 1) % stride:
                continue
            for gid, _bbox in dets:
                occ[(cam, fr)].add(gid)
                pres[(gid, cam)].append(fr)
    for v in pres.values():
        v.sort()
    return occ, pres


def exclusive(occ, cams, gid, lo, hi, stride):
    """窗內每個取樣幀、每台鏡頭都只有 gid 一個人。"""
    for f in sampled_frames(stride, lo, hi):
        for cam in cams:
            ids = occ.get((cam, f))
            if ids and ids - {gid}:
                return False
    return True


def others_in(occ, cams, gid, lo, hi, stride):
    out = set()
    for f in sampled_frames(stride, lo, hi):
        for cam in cams:
            out |= occ.get((cam, f), set()) - {gid}
    return out


def pair_class(a, b, overlapping):
    if frozenset((a, b)) in overlapping:
        return "overlapping"
    if frozenset((a, b)) in TRULY_DISJOINT:
        return "truly_disjoint"
    return "other"


def make_window(level, rule, split, seq, cams, gid, lo, hi, stride, occ, **extra):
    others = others_in(occ, cams, gid, lo, hi, stride)
    n = len(list(sampled_frames(stride, lo, hi)))
    return dict(
        window_id=f"{rule}-{split}-{seq}-{'+'.join(cams)}-{lo - 1:06d}",
        level=level, selection_rule=rule, split=split, seq=seq,
        cameras=list(cams), gt_id=gid,
        start_frame=lo, end_frame=hi,           # 1-based 標註幀
        start_fid=lo - 1, end_fid=hi - 1,       # 0-based,tracks.csv 的鍵
        t_start_s=round((lo - 1) / FPS, 3), t_end_s=round((hi - 1) / FPS, 3),
        duration_s=round((hi - lo) / FPS, 3), n_sampled_frames=n,
        exclusive=not others, n_other_ids=len(others), rendered=False, **extra)


def find_l1(seq, split, pres, occ, p):
    out = []
    for (gid, cam), frames in pres.items():
        segs = segments(frames, p["merge_gap"])
        if len(segs) != 1:
            continue
        s, e = segs[0]
        if (e - s) / FPS < p["l1_min_s"]:
            continue
        # ⚠ 從區段**開頭**截斷,不是挑中間「最好看」的一段
        hi = min(e, s + int(p["l1_max_s"] * FPS))
        if (hi - s) / FPS < p["l1_min_s"]:
            continue
        if not exclusive(occ, [cam], gid, s, hi, p["stride"]):
            continue
        out.append(make_window("L1", "L1", split, seq, [cam], gid, s, hi,
                               p["stride"], occ, segments_s=[round((e - s) / FPS, 3)],
                               gaps_s=[], gap_band=None, trimmed=hi < e))
    return out


def find_l2(seq, split, pres, occ, p):
    out = []
    mx = int(p["max_window_s"] * FPS)
    for (gid, cam), frames in pres.items():
        segs = segments(frames, p["merge_gap"])
        if len(segs) < 2:
            continue
        for (s1, e1), (s2, e2) in zip(segs, segs[1:]):
            gap = (s2 - e1) / FPS
            if not (p["l2_min_gap_s"] <= gap <= p["l2_max_gap_s"]):
                continue
            if min((e1 - s1), (e2 - s2)) / FPS < p["l2_min_seg_s"]:
                continue
            lo, hi, cut = trim(s1, e2, e1, s2, mx, p["stride"])   # 消失期間完整保留
            if not exclusive(occ, [cam], gid, lo, hi, p["stride"]):
                continue
            band = "within_buffer" if gap <= p["l2_buffer_s"] else "beyond_buffer"
            out.append(make_window("L2", "L2", split, seq, [cam], gid, lo, hi,
                                   p["stride"], occ,
                                   segments_s=[round((e1 - s1) / FPS, 3),
                                               round((e2 - s2) / FPS, 3)],
                                   gaps_s=[round(gap, 3)], gap_band=band, trimmed=cut))
    return out


def find_l3(seq, split, pres, occ, p, overlapping):
    """L3S 同時出現在 2~3 台;L3T 先後出現(轉場)。兩種分開標記。"""
    mx = int(p["max_window_s"] * FPS)
    by_id_cam = defaultdict(dict)
    for (gid, cam), frames in pres.items():
        by_id_cam[gid][cam] = segments(frames, p["merge_gap"])
    out = []

    for gid, by_cam in by_id_cam.items():
        # ── L3S:兩台的在場區段在時間上重疊夠久 ──
        for ca, cb in combinations(sorted(by_cam), 2):
            for sa, ea in by_cam[ca]:
                for sb, eb in by_cam[cb]:
                    lo, hi = max(sa, sb), min(ea, eb)
                    if (hi - lo) / FPS < p["l3_min_overlap_s"]:
                        continue
                    full = (hi - lo) / FPS
                    hi = min(hi, lo + mx)              # 同時出現沒有「關鍵瞬間」,取開頭
                    cams = [ca, cb]
                    # 第三台若整段都在,升級成三鏡頭(需求寫的是 2~3 台)
                    for cc in sorted(by_cam):
                        if cc in cams:
                            continue
                        if any(s <= lo and e >= hi for s, e in by_cam[cc]):
                            cams = sorted(cams + [cc])
                            break
                    out.append(make_window(
                        "L3", "L3S", split, seq, cams, gid, lo, hi, p["stride"], occ,
                        segments_s=[], gaps_s=[], gap_band=None, transitions=[],
                        overlap_s=round(full, 3), trimmed=full > p["max_window_s"],
                        pair_classes=sorted({pair_class(a, b, overlapping)
                                             for a, b in combinations(cams, 2)})))

        # ── L3T:相鄰兩段分屬不同鏡頭、時間不重疊 ──
        segs = sorted((s, e, c) for c, ss in by_cam.items() for s, e in ss)
        for (s1, e1, c1), (s2, e2, c2) in zip(segs, segs[1:]):
            if c1 == c2 or s2 <= e1:            # 同機、或時間重疊 → 不是轉場
                continue
            dt = (s2 - e1) / FPS
            if not (0 < dt <= p["l3_max_dt_s"]):
                continue
            lo, hi, cut = trim(s1, e2, e1, s2, mx, p["stride"])   # 轉場間隔完整保留
            out.append(make_window(
                "L3", "L3T", split, seq, [c1, c2], gid, lo, hi, p["stride"], occ,
                segments_s=[], gaps_s=[round(dt, 3)], gap_band=None, overlap_s=None,
                trimmed=cut,
                transitions=[dict(**{"from": c1}, to=c2, dt_s=round(dt, 3),
                                  pair_class=pair_class(c1, c2, overlapping))],
                pair_classes=[pair_class(c1, c2, overlapping)]))
    return out


def pick(windows, take):
    """固定排序鍵取前 N,且同一 (seq, 鏡頭組) 最多一個。

    ⚠ 排序鍵**不含任何品質指標** —— 用結果好壞挑要拍的片段,等於用測試集選樣本。
    ⚠ 同一組鏡頭限一個:否則幾段影片可能全來自同一分鐘的畫面。
    ⚠ 獨佔的優先(字面符合「單人」),但**只作為排序的第一鍵**,不是過濾條件 ——
      沒有獨佔的窗時仍要挑得出東西。
    """
    seen, chosen = set(), []
    for w in sorted(windows, key=lambda x: (not x["exclusive"], x["seq"],
                                            tuple(x["cameras"]), x["start_frame"])):
        key = (w["seq"], tuple(w["cameras"]))
        if key in seen:
            continue
        seen.add(key)
        chosen.append(w)
        if len(chosen) >= take:
            break
    return chosen


def selfcheck(windows, splits, p):
    bad = []
    for w in windows:
        if w["start_fid"] != w["start_frame"] - 1 or w["start_fid"] % p["stride"]:
            bad.append(f"{w['window_id']}:取樣格點對不上")
        if w["n_sampled_frames"] < 2:
            bad.append(f"{w['window_id']}:窗內取樣幀不足 2")
        r = w["selection_rule"]
        if r == "L1" and (len(w["gaps_s"]) or len(w["segments_s"]) != 1):
            bad.append(f"{w['window_id']}:L1 不該有間隔")
        if r == "L2" and (len(w["gaps_s"]) != 1 or len(w["segments_s"]) != 2):
            bad.append(f"{w['window_id']}:L2 必須剛好兩段一隔")
        if r == "L2" and not w["exclusive"]:
            bad.append(f"{w['window_id']}:L2 必須獨佔")
        if r.startswith("L3") and not 2 <= len(w["cameras"]) <= 3:
            bad.append(f"{w['window_id']}:L3 的鏡頭數必須是 2~3")
        if r == "L3S" and (w["transitions"] or not w["overlap_s"]):
            bad.append(f"{w['window_id']}:L3S 應該有重疊長度、沒有轉場")
        if r == "L3T" and len(w["transitions"]) != 1:
            bad.append(f"{w['window_id']}:L3T 應該剛好一次轉場")
    a, b, c = (set(splits["train"]), set(splits["test"]), set(splits["extra"]))
    if a & b or a & c or b & c:
        bad.append("訓練 / 測試 / 第三桶的序列有重疊")
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--merge-gap", type=int, default=15,
                    help="合併在場區段的最大斷點(幀)。15 幀 = 0.5 秒,容忍標註掉幀,不容忍真的離開")
    ap.add_argument("--max-window-s", type=float, default=60.0,
                    help="窗的長度上限(秒)。關鍵事件一定完整保留,前後平均分配剩下的時間")
    ap.add_argument("--l1-min-s", type=float, default=20.0)
    ap.add_argument("--l1-max-s", type=float, default=60.0)
    ap.add_argument("--l2-min-gap-s", type=float, default=2.0)
    ap.add_argument("--l2-max-gap-s", type=float, default=30.0)
    ap.add_argument("--l2-min-seg-s", type=float, default=3.0)
    ap.add_argument("--l2-buffer-s", type=float, default=5.0,
                    help="M4 的 lost_track_buffer 秒數;間隔在這之內追蹤器自己該救回")
    ap.add_argument("--l3-max-dt-s", type=float, default=30.0, help="L3T:轉場的最大間隔")
    ap.add_argument("--l3-min-overlap-s", type=float, default=5.0,
                    help="L3S:同時出現在多台鏡頭的最短重疊時間")
    ap.add_argument("--take", type=int, default=2, help="每個 (規則, 桶) 要拍幾段影片")
    ap.add_argument("--topology", default=str(ROOT / "configs" / "camera_topology.chirla.yaml"))
    ap.add_argument("--train-seqs", nargs="+", default=[])
    ap.add_argument("--test-seqs", nargs="+", default=[])
    ap.add_argument("--extra-seqs", nargs="+", default=[],
                    help="兩邊都算、所以兩邊都不放的序列(seq_020),單獨報")
    ap.add_argument("--out", default=str(ROOT / "results" / "levels" / "level_manifest.json"))
    args = ap.parse_args()

    p = dict(stride=args.stride, merge_gap=args.merge_gap, max_window_s=args.max_window_s,
             l1_min_s=args.l1_min_s, l1_max_s=args.l1_max_s,
             l2_min_gap_s=args.l2_min_gap_s, l2_max_gap_s=args.l2_max_gap_s,
             l2_min_seg_s=args.l2_min_seg_s, l2_buffer_s=args.l2_buffer_s,
             l3_max_dt_s=args.l3_max_dt_s, l3_min_overlap_s=args.l3_min_overlap_s,
             take=args.take)
    splits = dict(train=args.train_seqs, test=args.test_seqs, extra=args.extra_seqs)
    if not any(splits.values()):
        raise SystemExit("至少要給 --train-seqs / --test-seqs / --extra-seqs 其中一個")

    with open(args.topology, encoding="utf-8") as f:
        topo = (yaml.safe_load(f) or {}).get("camera_topology", {})
    overlapping = {frozenset(x) for x in topo.get("overlapping", [])}

    windows = []
    for split, seqs in splits.items():
        for seq in seqs:
            gt = load_gt(args.root, seq)
            empty = [c for c, v in gt.items() if not v]
            if empty:
                # seq_006/camera_4 整段沒有人,標註是 {} —— 結構性排除,不是錯誤
                print(f"  ⚠ {seq}:{empty} 全序列無標註,已排除")
                gt = {c: v for c, v in gt.items() if v}
            occ, pres = build_tables(gt, args.stride)
            found = (find_l1(seq, split, pres, occ, p)
                     + find_l2(seq, split, pres, occ, p)
                     + find_l3(seq, split, pres, occ, p, overlapping))
            windows += found
            print(f"  {seq:<10}({split:<5}) 身份 {len({g for g, _ in pres})} 個 → 窗 {len(found)}")

    # ⚠ 去重。L3S 的「第三台升級」會讓 (A,B)、(A,C)、(B,C) 三個配對收斂到**同一組鏡頭、
    #   同一段時間** → 產生 window_id 完全相同的窗。不去重的話:每級產量被灌水、
    #   同一段影片被渲染好幾次,而且彙總指標會把同一段資料重複計入。
    seen, uniq = set(), []
    for w in windows:
        if w["window_id"] in seen:
            continue
        seen.add(w["window_id"])
        uniq.append(w)
    if len(uniq) != len(windows):
        print(f"\n  去重:{len(windows)} → {len(uniq)} 個窗(重複的鏡頭組合)")
    windows = uniq

    rules = ("L1", "L2", "L3S", "L3T")
    for rule in rules:
        for split in splits:
            pool = [w for w in windows if w["selection_rule"] == rule and w["split"] == split]
            for w in pick(pool, args.take):
                w["rendered"] = True

    bad = selfcheck(windows, splits, p)
    sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                         text=True, cwd=ROOT).stdout.strip()
    summary = Counter(f"{w['selection_rule']}|{w['split']}" for w in windows)
    out = dict(params=p, splits=splits, topology=str(args.topology),
               git_sha=sha, argv=sys.argv[1:],
               counts=dict(summary), n_rendered=sum(w["rendered"] for w in windows),
               selfcheck_failures=bad, windows=windows)

    print("\n每個規則的產量(⚠ 0 就是 0,不從另一側補):")
    for rule in rules:
        row = "  ".join(f"{s}={summary.get(f'{rule}|{s}', 0)}" for s in splits)
        print(f"  {rule:<4} {row}")
    print(f"  要渲染的窗:{out['n_rendered']}")

    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已存 {path}")
    if bad:
        print(f"[FAIL] 自檢 {len(bad)} 項:")
        for b in bad[:10]:
            print(f"  - {b}")
        return 1
    print("[OK] 自檢全部通過")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
