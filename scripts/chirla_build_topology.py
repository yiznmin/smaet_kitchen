"""從 CHIRLA 的逐幀標註推導相機拓撲,輸出 configs/camera_topology.chirla.yaml。

實作 `docs/CHIRLA_M4M5驗證_預先登記_20260903.md` §9 的交付物之一。

⚠⚠ **時間參數只能從推導集(seq_000/001/002)估,不得碰評估集。**
   這是本輪最重要的設計。9/1 的消融(`docs/M4_M5_實片全長評估_20260901.md`)證明
   「能力來自知道鏡頭怎麼擺,不是來自演算法」—— 拿評估集自己的統計去建拓撲,
   等於把答案餵給系統,量到的數字沒有意義。
   本腳本**硬性拒絕**把評估集序列納入參數估計,見 `EVAL_SEQS` 的守衛。

⚠ **重疊關係是例外,而且這個例外要照實揭露。**
   `scripts/chirla_overlap_stats.py` 已在全部 10 個序列跑過(2026-09-03),
   所以「哪幾對相機重疊」我們看過評估集了。處置照 §3:相機是固定的、10 個序列間
   機位沒動過,所以重疊關係視為**物理事實**,沿用全域結果並在 YAML 裡註明洩漏。

轉場的定義(逐幀標註 → 有向轉場):
  1. 同一序列、同一身份,先把每台相機的出現幀合併成「在場區段」
     (幀距 ≤ `--merge-gap` 幀視為同一段,容忍偵測/標註的零星缺幀)
  2. 把所有區段依開始幀排序,相鄰兩段若**分屬不同相機**且**時間不重疊**
     (後段開始 > 前段結束),記一次 A→B 轉場,Δt = 間隔秒數
  3. 只收 Δt ≤ `--window` 的
  ⚠ 時間重疊的不算轉場 —— 那是重疊視野下的同時出現,走的是另一條證據路徑。

用法:
    python scripts/chirla_build_topology.py --root "D:/新增資料夾/CHIRLA/CHIRLA_data/CHIRLA"
"""
import argparse
import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

# §3 定死的切分。⚠ 不得在看到結果後調整。
DERIV_SEQS = ("seq_000", "seq_001", "seq_002")
EVAL_SEQS = ("seq_004", "seq_006", "seq_007", "seq_020",
             "seq_024", "seq_025", "seq_026")

FPS = 30.0


def phys(name):
    return "_".join(name.split("_")[:2])


def load_seq(ann_dir):
    """回傳 {identity: {camera: [frame, ...]}}。distractor 的負號取絕對值(§4.1)。"""
    out = defaultdict(lambda: defaultdict(list))
    for f in sorted(Path(ann_dir).glob("*.json")):
        cam = phys(f.stem)
        for fr, dets in json.loads(f.read_text(encoding="utf-8")).items():
            for o in dets:
                out[abs(int(o["id"]))][cam].append(int(fr))
    return out


def segments(frames, merge_gap):
    """把出現幀合併成在場區段 [(start, end), ...]。"""
    fs = sorted(frames)
    segs, s, p = [], fs[0], fs[0]
    for f in fs[1:]:
        if f - p > merge_gap:
            segs.append((s, p))
            s = f
        p = f
    segs.append((s, p))
    return segs


def transitions(seqs_dir, seq_names, merge_gap, window_s):
    """回傳 {(camA, camB): [Δt 秒, ...]} 與新身份到達事件數、總時長。"""
    dts = defaultdict(list)
    n_arrivals, total_frames = 0, 0
    for name in seq_names:
        per_id = load_seq(Path(seqs_dir) / name)
        seq_max = 0
        for ident, by_cam in per_id.items():
            n_arrivals += 1                       # 該身份在該序列的首次出現
            segs = []
            for cam, frames in by_cam.items():
                seq_max = max(seq_max, max(frames))
                segs += [(s, e, cam) for s, e in segments(frames, merge_gap)]
            segs.sort()
            for (s1, e1, c1), (s2, e2, c2) in zip(segs, segs[1:]):
                if c1 == c2 or s2 <= e1:          # 同機、或時間重疊 → 不是轉場
                    continue
                dt = (s2 - e1) / FPS
                if 0 < dt <= window_s:
                    dts[(c1, c2)].append(dt)
        total_frames += seq_max
    return dts, n_arrivals, total_frames / FPS


def fit_loiter(all_dts):
    """從轉場間隔分布估 p_loiter 與 tau_loiter_s。

    做法刻意簡單且可複述:以中位數 + 2×MAD 當「直達」與「逗留」的分界,
    超過的視為逗留成分。p_loiter = 其比例,tau = 超出部分的平均超額。
    ⚠ 這不是最大概似擬合,是一個保守的起手值 —— 目的是不要人工猜,
      而不是求最佳。實際分布存進 JSON,之後要換更好的擬合隨時可重算。
    """
    x = np.asarray(sorted(all_dts), dtype=float)
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med))) or 1e-6
    cut = med + 2.0 * 1.4826 * mad
    tail = x[x > cut]
    p = float(len(tail) / len(x)) if len(x) else 0.0
    tau = float((tail - cut).mean()) if len(tail) else 1.0
    return round(max(p, 0.01), 4), round(max(tau, 1.0), 2), round(cut, 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", default=str(ROOT / "configs" / "camera_topology.chirla.yaml"))
    ap.add_argument("--stats-out", default="results/m5_reid/chirla_topology_fit.json")
    ap.add_argument("--merge-gap", type=int, default=15,
                    help="幀距 <= 此值視為同一在場區段(15 幀 = 0.5 秒)")
    ap.add_argument("--window", type=float, default=60.0, help="轉場間隔上限(秒)")
    ap.add_argument("--min-transits", type=int, default=3,
                    help="一條有向連結至少要觀察到幾次才寫進 links")
    args = ap.parse_args()

    root = Path(args.root)
    ann = root / "annotations"

    # 守衛:推導集與評估集不得有交集,且推導集必須存在
    assert not (set(DERIV_SEQS) & set(EVAL_SEQS)), "推導集與評估集重疊 —— 切分寫錯了"
    missing = [s for s in DERIV_SEQS if not (ann / s).is_dir()]
    if missing:
        raise SystemExit(f"推導集缺序列:{missing}")

    print("=" * 78)
    print("CHIRLA 相機拓撲推導")
    print("=" * 78)
    print(f"  推導集(可以看):{', '.join(DERIV_SEQS)}")
    print(f"  評估集(絕不碰):{', '.join(EVAL_SEQS)}")

    dts, n_arrivals, dur_s = transitions(ann, DERIV_SEQS, args.merge_gap, args.window)
    all_dts = [d for v in dts.values() for d in v]
    if not all_dts:
        raise SystemExit("推導集裡一次轉場都沒觀察到 —— 檢查 --merge-gap / --window")

    # 重疊關係:沿用全域結果(§3 的例外,已揭露洩漏)
    from chirla_overlap_stats import GOOD_H, GOOD_W, collect
    _per_seq, pair, _hist = collect(root)
    overlapping = []
    for (a, b), v in pair.items():
        arr = np.array(v)
        good = ((arr[:, 0] >= GOOD_W) & (arr[:, 1] >= GOOD_H)).mean() * 100
        if len(arr) >= 500 and good >= 60:
            overlapping.append(sorted((a, b)))
    overlapping.sort()
    ovl_set = {frozenset(p) for p in overlapping}

    print(f"\n  重疊相機對(視為物理事實,沿用全域結果):{len(overlapping)} 對")
    for a, b in overlapping:
        print(f"      {a} + {b}")

    print(f"\n  推導集觀察到的有向轉場(Δt <= {args.window:.0f}s):")
    print(f"      {'連結':<26}{'次數':>6}{'中位':>8}{'平均':>8}{'標準差':>8}")
    links, skipped = [], []
    for (a, b), v in sorted(dts.items(), key=lambda kv: -len(kv[1])):
        tag = ""
        if frozenset((a, b)) in ovl_set:
            tag = "  (重疊對,不建 link)"
        elif len(v) < args.min_transits:
            tag = f"  (< {args.min_transits} 次,略過)"
            skipped.append((a, b, len(v)))
        mean = st.mean(v)
        sd = st.pstdev(v) if len(v) > 1 else mean * 0.35
        print(f"      {a + ' -> ' + b:<26}{len(v):>6}{st.median(v):>8.2f}"
              f"{mean:>8.2f}{sd:>8.2f}{tag}")
        if not tag:
            links.append(dict(**{"from": a, "to": b},
                              mean_s=round(mean, 3), std_s=round(max(sd, 0.2), 3),
                              n=len(v)))

    p_loiter, tau_loiter, cut = fit_loiter(all_dts)
    bg_hz = n_arrivals / dur_s

    print(f"\n  參數(只從推導集估):")
    print(f"      background_arrival_hz  {bg_hz:.5f}   "
          f"({n_arrivals} 個身份到達 / {dur_s:.0f} 秒)")
    print(f"      p_loiter               {p_loiter}     (分界 {cut:.2f}s 之後的比例)")
    print(f"      tau_loiter_s           {tau_loiter}")
    print(f"      轉場 Δt 中位            {st.median(all_dts):.2f}s   "
          f"p90 {np.percentile(all_dts, 90):.2f}s   n={len(all_dts)}")
    print(f"\n  寫入 links:{len(links)} 條;略過(次數不足):{len(skipped)} 條")

    cams = sorted({c for p in overlapping for c in p} |
                  {c for k in dts for c in k})

    # ── 產生 YAML(手寫以保留註解,格式對齊 configs/camera_topology.epfl9.yaml)──
    L = []
    L.append("# CHIRLA 七鏡頭的相機拓撲(2026-09-04 由 scripts/chirla_build_topology.py 產生)")
    L.append("#")
    L.append("# ⚠ 時間參數**只從推導集 seq_000/001/002 估**,評估集")
    L.append("#   (seq_004/006/007/020/024/025/026)完全沒有參與。")
    L.append("#   理由見 docs/CHIRLA_M4M5驗證_預先登記_20260903.md §3。")
    L.append("#")
    L.append("# ⚠ **重疊關係有資訊洩漏,照實揭露**:overlapping 清單來自")
    L.append("#   scripts/chirla_overlap_stats.py 在**全部 10 個序列**上的結果。")
    L.append("#   處置:相機固定、10 個序列間機位沒動過 → 視為物理事實。")
    L.append("#   但報告必須寫明「重疊拓撲有洩漏,時間參數沒有」。")
    L.append("#")
    L.append("# ⚠ 論文說 CHIRLA 是「7 台非重疊」,實測不成立 ——")
    L.append("#   見 docs/CHIRLA_鏡頭佈局實測_20260903.md。這裡是混合佈局。")
    L.append("#")
    L.append(f"# 推導依據:{len(all_dts)} 次轉場、{n_arrivals} 個身份到達、"
             f"{dur_s:.0f} 秒推導集影像")
    L.append("camera_topology:")
    L.append("")
    L.append("  fusion:")
    L.append("    mode: llr")
    L.append(f"    background_arrival_hz: {bg_hz:.5f}   "
             f"# 實測:{n_arrivals} 個身份 / {dur_s:.0f}s(EPFL 用的是 0.00167)")
    L.append("    cost_false_merge_over_break: 5.0")
    L.append("    transit_model: loiter")
    L.append(f"    p_loiter: {p_loiter}")
    L.append(f"    tau_loiter_s: {tau_loiter}")
    L.append("    appearance_profile: dinov2")
    L.append("    appearance_clip: null")
    L.append("    overlap_llr: 5.0")
    L.append("    overlap_window_s: 0.5")
    L.append("    same_camera: { enabled: true, tau_break_s: 2.0, max_gap_s: 15.0 }")
    # ⚠ 與 EPFL 那兩輪(unknown_path: false)不同,報告要註明這個差異。
    #   理由:EPFL 是 36 對全重疊,任兩台之間永遠有重疊路徑,unknown_path 用不到;
    #   CHIRLA 是混合佈局,而 539 秒的推導集**不足以觀察到每一條真實通道** ——
    #   實測 camera_1 相關的轉場只有 4 次且全是單次,低於 min-transits。
    #   沒有 unknown_path 的話 camera_1 兩條路徑都走不到 → 進出必定開新 chef_id,
    #   量到的碎裂率會是「推導集太短」的假象,不是系統的性質。
    #   ⚠ 這個決定在**看到任何評估集結果之前**做成,而且是**統一套用到所有沒有
    #     link 的配對**,不是針對 camera_1 挖的例外。
    L.append("    unknown_path: { enabled: true, median_multiplier: 2.0, "
             "log_sigma: 0.8, logprior: -2.0 }")
    L.append("    direction: { enabled: false }")
    L.append("    # ⚠ ground_plane 不啟用:CHIRLA 沒有提供相機標定參數,")
    L.append("    #   與 EPFL 同樣的限制(預先登記 §8 第 2 條)。")
    L.append("")
    if links:
        L.append("  links:")
        for l in links:
            L.append(f"    - {{ from: {l['from']}, to: {l['to']}, "
                     f"mean_s: {l['mean_s']}, std_s: {l['std_s']} }}   "
                     f"# 觀察到 {l['n']} 次")
    else:
        L.append("  links: []")
    L.append("")
    L.append("  overlapping:")
    for a, b in overlapping:
        L.append(f"    - [{a}, {b}]")
    L.append("")
    L.append("  clock:")
    L.append("    max_skew_s: 0.2   # 同序列七支影片起始時間戳相同、幀數差 10 幀內")
    L.append("")
    L.append("  cameras:")
    for c in cams:
        L.append(f"    {c}: {{ clock_offset_s: 0.0 }}")
    out = Path(args.out)
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"\n  → {out}")

    s = Path(args.stats_out)
    s.parent.mkdir(parents=True, exist_ok=True)
    s.write_text(json.dumps(dict(
        deriv_seqs=list(DERIV_SEQS), eval_seqs=list(EVAL_SEQS),
        n_transits=len(all_dts), n_arrivals=n_arrivals, duration_s=dur_s,
        background_arrival_hz=bg_hz, p_loiter=p_loiter, tau_loiter_s=tau_loiter,
        loiter_cut_s=cut,
        dt_median=st.median(all_dts), dt_p90=float(np.percentile(all_dts, 90)),
        links=links, skipped=[dict(a=a, b=b, n=n) for a, b, n in skipped],
        overlapping=[list(p) for p in overlapping],
        per_link={f"{a}->{b}": v for (a, b), v in dts.items()}),
        ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  → {s}")

    # 自檢:載得起來、而且沒有評估集的痕跡
    from m5_reid.spatiotemporal import CameraTopology
    topo = CameraTopology.from_yaml(out)
    print(f"\n  自檢:載入成功,{len(topo.links)} 條 link、"
          f"{len(topo.overlapping)} 對重疊、{len(topo.all_cameras())} 台相機")

    # 自檢:每一台相機都要至少有一條可用路徑,否則進出它必定開新 chef_id,
    # 量到的碎裂率會是拓撲推導的假象。unknown_path 開著時所有配對都有退路。
    reachable = set()
    for a, b in topo.links:
        reachable |= {a, b}
    for p in topo.overlapping:
        reachable |= set(p)
    orphan = sorted(set(topo.all_cameras()) - reachable)
    if topo.unknown_path is not None:
        print(f"  自檢:unknown_path 已啟用 → 所有配對都有退路"
              f"{'(否則孤立的會是 ' + ', '.join(orphan) + ')' if orphan else ''}  [OK]")
    else:
        print(f"  自檢:孤立相機(無重疊也無 link)→ {orphan or '無'}"
              f"{'  [FAIL] 進出這些相機必定開新 chef_id' if orphan else '  [OK]'}")
        if orphan:
            return 1
    # ⚠ 只掃**非註解**的行。第一版掃了整份檔案,結果被自己寫在檔頭的
    #   「評估集(seq_004/…)完全沒有參與」那句揭露文字觸發 → 誤報 FAIL。
    #   要擋的是「評估集的資料流進參數」,不是「文件裡提到評估集的名字」。
    payload = "\n".join(ln for ln in out.read_text(encoding="utf-8").splitlines()
                        if not ln.lstrip().startswith("#"))
    leaked = [s_ for s_ in EVAL_SEQS if s_ in payload]
    print(f"  自檢:YAML 的非註解內容出現評估集序列名 → {leaked or '無'}"
          f"{'  [FAIL]' if leaked else '  [OK]'}")
    return 1 if leaked else 0


if __name__ == "__main__":
    sys.exit(main())
