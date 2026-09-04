"""用 CHIRLA 的逐幀標註評估 M4/M5 —— 專案第一次在真實多人影片上量誤併率。

實作 `docs/CHIRLA_M4M5驗證_預先登記_20260903.md` §4 的評估協定。

**為什麼不能用 `scripts/eval_m5_longrun.py`**:那支的判準是「全片只有一個人」,
從硬事實推出「chef_id 數 − 1 = 碎裂次數」。CHIRLA 同一時刻中位有 6~7 個人,
那個判準完全不適用。這裡改用真正的逐幀 GT 框。

⚠ **幀號差一格會靜默毀掉整份結果。** CHIRLA 的標註是 **1-based**
  (實測 min=1、max=影片幀數),而 `m5_track_video.py` 的 `video_fid` 是 **0-based**。
  所以 `GT 幀 = video_fid + 1`。這件事沒有任何錯誤訊息會提醒你 ——
  對錯了只會看到「配對率很低」,而那看起來像模型爛。自檢區塊會擋。

⚠ **自檢的斷言一律寫成不變量。** 9/3 的教訓:`eval_m5_longrun.py` 把重疊對數
  硬編成 3,擴到九台就誤判成失敗。這裡不寫任何「應該有 N 個」的常數。

用法:
    python scripts/eval_m4m5_chirla.py --root <CHIRLA根> \
        --run-dir results/chirla_m4m5/coco_none/seq_004 [更多 run-dir ...]
"""
import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

IOU_THR = 0.5           # §4.1 先寫死。換門檻會改變誤偵的判定邊界,配對結果有存檔可重算
MATCH_RATE_MIN = 0.5    # §4.1:一條 track 的配對率低於此值 → 標記為「無 GT」(誤偵)


def phys(name):
    return "_".join(name.split("_")[:2])


def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / ua if ua > 0 else 0.0


def load_gt(root, seq):
    """{camera: {frame(1-based): [(gt_id, bbox), ...]}}。distractor 負號取絕對值(§4.1)。

    ⚠ **有標註檔但內容為空的相機也要建鍵。** 實測 seq_006 的 camera_4 全程沒有人,
      標註是 `{}`。第一版只在有偵測時才建鍵,於是自檢誤判成「追蹤輸出的相機
      沒有 GT」而中止 —— 但那台相機上偵測到的東西**依定義就是幽靈**,
      正是本輪要量的目標,不是中止的理由。
      要擋的是「標註檔根本不存在」,不是「標註檔是空的」。
    """
    out = {}
    for f in sorted((Path(root) / "annotations" / seq).glob("*.json")):
        cam = phys(f.stem)
        per = out.setdefault(cam, defaultdict(list))
        for fr, dets in json.loads(f.read_text(encoding="utf-8")).items():
            for o in dets:
                per[int(fr)].append((abs(int(o["id"])), tuple(map(float, o["BboxP"]))))
    return out


def load_run(run_dir):
    d = Path(run_dir)
    meta = json.loads((d / "run_meta.json").read_text(encoding="utf-8"))
    tracks = []
    with open(d / "tracks.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            tracks.append(dict(fid=int(r["video_fid"]), cam=r["camera_id"],
                               tid=int(r["track_id"]),
                               bbox=(float(r["x1"]), float(r["y1"]),
                                     float(r["x2"]), float(r["y2"]))))
    events = [json.loads(l) for l in
              (d / "chef_events.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    return meta, tracks, events


# ── §0 自檢 ───────────────────────────────────────────────────────────
def selfcheck(gt, tracks, events, meta, seq):
    """回傳 [(是否致命, 通過與否, 訊息)]。⚠ 全部寫成不變量,不含硬編數量。"""
    out = []

    def chk(fatal, ok, msg):
        out.append((fatal, ok, msg))

    cams_tr = {t["cam"] for t in tracks}
    cams_gt = set(gt)
    chk(True, bool(tracks), f"tracks.csv 非空({len(tracks):,} 列)")
    chk(True, bool(events), f"chef_events 非空({len(events):,} 筆)")
    chk(True, cams_tr <= cams_gt,
        f"追蹤輸出的相機都有 GT 標註(追蹤 {len(cams_tr)} 台 / GT {len(cams_gt)} 台)"
        + ("" if cams_tr <= cams_gt else f" —— 多出 {sorted(cams_tr - cams_gt)}"))

    # 幀號對齊 —— 本腳本最容易出錯、而且錯了不會報錯的一項。
    #
    # ⚠ 第一版寫成「每個 fid+1 都要落在該相機的 GT 幀範圍內」,結果誤報 FAIL:
    #   GT 是逐相機的,人不在那台相機的時段本來就沒有標註,fid 落在範圍外是正常的。
    #   要擋的是**編號慣例對錯**,不是「每一幀都有人」。
    #
    # 所以改成直接檢驗對齊假設本身:把追蹤框與 GT 框在位移 -1 / 0 / +1 三種
    # 對應下各算一次 IoU 總量,**+1 必須是最大的**(GT 1-based vs fid 0-based)。
    # 這是經驗不變量,不含任何硬編數量,而且對錯了會立刻現形。
    fids = sorted({t["fid"] for t in tracks})
    per_ft = defaultdict(list)
    for t in tracks:
        per_ft[(t["cam"], t["fid"])].append(t["bbox"])
    sample = fids[:: max(1, len(fids) // 200)]
    mass = {}
    for off in (-1, 0, 1):
        s = 0.0
        for cam in cams_tr & cams_gt:
            for f in sample:
                gts = gt[cam].get(f + off, [])
                for tb in per_ft.get((cam, f), []):
                    s += max((iou(tb, gb) for _g, gb in gts), default=0.0)
        mass[off] = round(s, 1)
    best = max(mass, key=mass.get)
    # ⚠ 這個檢查的訊號很弱:30fps 下相鄰幀幾乎一樣,三個位移的 IoU 總量實測只差
    #   約 1%(2375.7 / 2389.2 / 2408.0)。它擋得住「差了好幾幀」或「用錯欄位」,
    #   擋不住細微的時基問題。不要把它當成對齊已經被證明。
    chk(True, best == 1 and mass[1] > 0,
        f"幀號對齊:IoU 總量 位移-1={mass[-1]} / 0={mass[0]} / +1={mass[1]} "
        f"→ 最佳位移 {best:+d}(應為 +1,因為 GT 是 1-based、video_fid 是 0-based;"
        f"⚠ 三者差距僅約 1%,這是弱訊號)")

    # stride 一致性:tracks 的相鄰 fid 間距應等於 run_meta 的 stride
    sf = sorted(fids)
    gaps = {b - a for a, b in zip(sf, sf[1:])} or {0}
    stride = meta.get("stride")
    chk(False, stride in gaps or len(sf) < 2,
        f"fid 間距 {sorted(gaps)[:3]} 與 run_meta.stride={stride} 一致")

    # 事件裡的 (cam, fid, tid) 都要能在 tracks.csv 找到,否則兩份輸出對不起來
    key = {(t["cam"], t["fid"], t["tid"]) for t in tracks}
    miss = sum(1 for e in events
               if (e["camera_id"], e["video_fid"], e["track_id"]) not in key)
    chk(True, miss == 0, f"chef_events 的 (相機,幀,track) 都在 tracks.csv 裡(缺 {miss} 筆)")

    # 覆蓋率:處理到的幀數 / GT 的幀數(受 stride 與 max_frames 影響,只是資訊)
    n_gt_frames = sum(len(v) for v in gt.values())
    chk(False, n_gt_frames > 0, f"GT 標註幀數 {n_gt_frames:,}(序列 {seq})")

    # ⚠ 不寫「應該有 N 台相機 / N 對重疊」這類斷言。9/3 就是把重疊對數硬編成 3,
    #   擴到九台時誤判成失敗。斷言只能是不變量。
    return out


# ── §1 GT 配對 ────────────────────────────────────────────────────────
def match_tracks(gt, tracks):
    """逐相機逐幀 IoU>=0.5 匈牙利配對。回傳 (track 的 GT 身份, 每條 track 的配對統計)。"""
    from scipy.optimize import linear_sum_assignment

    per_frame = defaultdict(list)
    for t in tracks:
        per_frame[(t["cam"], t["fid"])].append(t)

    votes = defaultdict(Counter)      # (cam,tid) -> Counter(gt_id)
    seen = Counter()                  # (cam,tid) -> 出現幀數
    gt_matched = Counter()            # (cam,frame) 命中的 GT 數,用來算漏偵
    gt_total = Counter()

    for (cam, fid), ts in per_frame.items():
        for t in ts:
            seen[(cam, t["tid"])] += 1
        gts = gt.get(cam, {}).get(fid + 1, [])      # ⚠ GT 是 1-based
        gt_total[cam] += len(gts)
        if not gts:
            continue
        m = np.zeros((len(ts), len(gts)))
        for i, t in enumerate(ts):
            for j, (_gid, gb) in enumerate(gts):
                m[i, j] = iou(t["bbox"], gb)
        r, c = linear_sum_assignment(-m)
        for i, j in zip(r, c):
            if m[i, j] >= IOU_THR:
                votes[(cam, ts[i]["tid"])][gts[j][0]] += 1
                gt_matched[cam] += 1

    track_gt, stats = {}, {}
    for k, n in seen.items():
        v = votes.get(k)
        hit = sum(v.values()) if v else 0
        rate = hit / n if n else 0.0
        gid = v.most_common(1)[0][0] if v and rate >= MATCH_RATE_MIN else None
        track_gt[k] = gid
        stats[k] = dict(n_frames=n, n_matched=hit, match_rate=round(rate, 4),
                        gt_id=gid, is_ghost=gid is None)
    return track_gt, stats, gt_matched, gt_total


# ── §4 M5 指標 ───────────────────────────────────────────────────────
def build_records(events, track_gt, overlapping):
    """把 chef_events 轉成 metrics.summarize 需要的 records。

    §4.2:**分母不含初始化那一次** —— 所以每個 GT 身份的第一次綁定決策
    `is_transition=False`,之後才算。

    另外分出「這次決策走的是重疊路徑還是轉場路徑」(預測 2 要用):
    看該 chef 上一次出現的相機與這次的相機是否在 overlapping 裡。
    """
    recs, ghost_bind, seen_gt = [], 0, set()
    last_cam = {}
    path_recs = {"overlap": [], "transit": []}
    for e in sorted(events, key=lambda x: (x["video_fid"], x["camera_id"])):
        gid = track_gt.get((e["camera_id"], e["track_id"]))
        if gid is None:                       # 綁到「無 GT」的 track = 誤偵綁定
            ghost_bind += 1
            continue
        is_tr = gid in seen_gt
        seen_gt.add(gid)
        rec = (gid, e["chef_id"], bool(e.get("matched")), is_tr)
        recs.append(rec)
        if is_tr:
            # ⚠ 用「這個**真實身份**上次出現在哪台相機」,不是「這個 chef_id 上次
            #   出現在哪台」。第一版用 chef_id,結果碎裂事件全部被漏掉:
            #   綁不上時會開一個**全新的** chef_id,而新 id 沒有上一台相機
            #   → prev 是 None → 整筆被跳過。症狀是「整體碎裂率 31.33%,
            #   但兩條路徑都是 0.00%」—— 自相矛盾,而那正是指標寫錯的訊號。
            #   真實身份的上一台相機與綁定成功與否無關,才是路徑的正確定義。
            prev = last_cam.get(gid)
            if prev and prev != e["camera_id"]:
                kind = ("overlap" if frozenset((prev, e["camera_id"])) in overlapping
                        else "transit")
                path_recs[kind].append(rec)
        last_cam[gid] = e["camera_id"]
    return recs, ghost_bind, path_recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="CHIRLA 根目錄")
    ap.add_argument("--run-dir", nargs="+", required=True,
                    help="m5_track_video.py 的輸出目錄,每個對應一個序列")
    ap.add_argument("--seq", nargs="+", default=None,
                    help="每個 run-dir 對應的序列名;不給則從 run_meta 的影片路徑推")
    ap.add_argument("--topology", default=str(ROOT / "configs" / "camera_topology.chirla.yaml"))
    ap.add_argument("--out", default=None, help="報告輸出目錄(預設寫到第一個 run-dir)")
    args = ap.parse_args()

    from m5_reid import metrics
    from m5_reid.spatiotemporal import CameraTopology
    topo = CameraTopology.from_yaml(args.topology)

    L, fatal_fail = [], False
    all_recs, all_path = [], {"overlap": [], "transit": []}
    tot_ghost = 0
    m4 = defaultdict(lambda: dict(n_tracks=0, n_ghost=0, gt_matched=0, gt_total=0))
    per_run = {}

    for i, rd in enumerate(args.run_dir):
        meta, tracks, events = load_run(rd)
        seq = (args.seq[i] if args.seq else
               next(s for s in Path(next(iter(meta["camera_video_map"].values()))).parts
                    if s.startswith("seq_")))
        gt = load_gt(args.root, seq)

        L.append(f"\n{'=' * 74}\n{rd}  ·  {seq}\n{'=' * 74}")
        L.append("\n§0 自檢 —— 不全過就不印指標\n")
        checks = selfcheck(gt, tracks, events, meta, seq)
        for is_fatal, ok, msg in checks:
            tag = "PASS" if ok else ("FAIL" if is_fatal else "WARN")
            L.append(f"  [{tag}] {msg}")
            if is_fatal and not ok:
                fatal_fail = True
        if fatal_fail:
            continue

        track_gt, tstats, gtm, gtt = match_tracks(gt, tracks)
        recs, ghost, path = build_records(events, track_gt, topo.overlapping)
        all_recs += recs
        tot_ghost += ghost
        for k in all_path:
            all_path[k] += path[k]

        for (cam, _tid), s in tstats.items():
            m4[cam]["n_tracks"] += 1
            m4[cam]["n_ghost"] += int(s["is_ghost"])
        for cam in gtt:
            m4[cam]["gt_matched"] += gtm[cam]
            m4[cam]["gt_total"] += gtt[cam]

        per_run[rd] = dict(seq=seq, n_tracks=len(tstats),
                           n_ghost_tracks=sum(s["is_ghost"] for s in tstats.values()),
                           n_bindings=len(recs), n_ghost_bindings=ghost)
        L.append(f"\n  track {len(tstats)} 條,其中無 GT(誤偵){per_run[rd]['n_ghost_tracks']} 條")

    if fatal_fail:
        L.append("\n自檢有致命項未通過。這次的資料不足以支撐任何結論,不印指標。")
        print("\n".join(L))
        return 2

    # ── M4 ──
    L.append(f"\n{'=' * 74}\n§3  M4:偵測與追蹤(以 GT 框為準)\n{'=' * 74}")
    L.append(f"\n  {'相機':<12}{'track 數':>9}{'無 GT':>8}{'誤偵率':>9}"
             f"{'GT 框':>10}{'配對到':>10}{'漏偵率':>9}")
    for cam in sorted(m4):
        d = m4[cam]
        gr = d["n_ghost"] / d["n_tracks"] if d["n_tracks"] else 0
        miss = 1 - d["gt_matched"] / d["gt_total"] if d["gt_total"] else None
        L.append(f"  {cam:<12}{d['n_tracks']:>9}{d['n_ghost']:>8}{gr * 100:>8.1f}%"
                 f"{d['gt_total']:>10,}{d['gt_matched']:>10,}"
                 f"{(miss * 100 if miss is not None else float('nan')):>8.1f}%")

    # ── M5 ──
    L.append(f"\n{'=' * 74}\n§4  M5:跨鏡頭身份(§4.2 的指標)\n{'=' * 74}")
    n_gt = len({g for g, _, _, _ in all_recs})
    s = metrics.summarize(all_recs, expected_headcount=n_gt)
    L.append(f"\n  綁定決策 {len(all_recs):,} 次(不含初始化 {len(all_recs) - s['n_transitions']:,} 次)"
             f",GT 身份 {n_gt} 個")
    L.append(f"  誤併率      {_p(s['p_false_merge'])}"
             f"      ← 專案第一次在真實多人影片上量到")
    L.append(f"  碎裂率      {_p(s['p_break'])}")
    L.append(f"  正確率      {_p(s['p_correct'])}")
    L.append(f"  IDF1        {s['idf1'] if s['idf1'] is not None else '—'}")
    L.append(f"  ID switch   {s['id_switches']}")
    L.append(f"  碎裂分布    平均 {s['fragmentation']['mean']} / 最大 {s['fragmentation']['max']}")
    if s.get("fm_unmeasurable_reason"):
        L.append(f"  ⚠ {s['fm_unmeasurable_reason']}")

    # 誤偵綁定率(§4.2 的新指標,§6 的次判準)
    denom = len(all_recs) + tot_ghost
    gb = tot_ghost / denom if denom else 0
    L.append(f"\n  誤偵綁定率  {gb * 100:.2f}%   ({tot_ghost:,} / {denom:,})")
    L.append(f"    ← 9/3 在 EPFL 上量到 22~41%,那時沒有 GT 只能看畫面判斷")
    if gb > 0.10:
        L.append("    ⚠ 超過 §6 次判準的 10% → 判定「M5 入口缺少『這個框是不是人』的把關」,"
                 "處置是做靜止/非人過濾,**不是**調 LLR 門檻")

    # 路徑拆分(預測 2)
    L.append(f"\n{'=' * 74}\n§5  重疊路徑 vs 轉場路徑(§7 預測 2)\n{'=' * 74}\n")
    L.append(f"  {'路徑':<10}{'決策數':>8}{'碎裂率':>10}{'誤併率':>10}")
    row = {}
    for k, label in (("overlap", "重疊"), ("transit", "轉場")):
        if not all_path[k]:
            L.append(f"  {label:<10}{'0':>8}      (沒有樣本)")
            continue
        b = metrics.binding_outcomes(all_path[k])
        row[k] = b
        L.append(f"  {label:<10}{b['n_transitions']:>8}{_p(b['p_break']):>10}"
                 f"{_p(b['p_false_merge']):>10}")
    if "overlap" in row and "transit" in row and row["overlap"]["p_break"]:
        r = row["transit"]["p_break"] / row["overlap"]["p_break"]
        L.append(f"\n  轉場 / 重疊 的碎裂率比值 = {r:.2f}  "
                 f"(預測 2 說 >= 2.0 → {'成立' if r >= 2 else '**被否證**'})")

    # ── 判準 ──
    L.append(f"\n{'=' * 74}\n§6  對照先寫死的判準\n{'=' * 74}\n")
    fm, br = s["p_false_merge"], s["p_break"]
    if fm is None:
        L.append("  誤併率不可量測 —— 見上方原因。不下級別判定。")
    else:
        lvl = "A" if (fm <= 0.01 and br is not None and br <= 0.05) else \
              "B" if fm <= 0.05 else "C"
        L.append(f"  誤併 {fm * 100:.2f}% / 碎裂 {br * 100:.2f}%  →  **{lvl} 級**")
        if fm > 0.30:
            L.append("  ⚠ 誤併 > 30% → 觸發 §6 先寫死的失敗條件:判定為 overlap_llr 的"
                     "結構性缺陷。處置是**對重疊證據加上幾何一致性條件**,"
                     "而不是把 overlap_llr 調小。")

    text = "\n".join(L)
    out_dir = Path(args.out or args.run_dir[0])
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "eval_m4m5_report.txt").write_text(text, encoding="utf-8")
    (out_dir / "eval_m4m5_report.json").write_text(json.dumps(dict(
        per_run=per_run, m4={k: dict(v) for k, v in m4.items()}, m5=s,
        ghost_binding_rate=gb, n_ghost_bindings=tot_ghost,
        path_split={k: (metrics.binding_outcomes(v) if v else None)
                    for k, v in all_path.items()},
        iou_thr=IOU_THR, match_rate_min=MATCH_RATE_MIN),
        ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(text)
    print(f"\n→ {out_dir/'eval_m4m5_report.txt'}")
    return 0


def _p(v):
    return "—" if v is None else f"{v * 100:.2f}%"


if __name__ == "__main__":
    sys.exit(main())
