"""EPFL 單場次全長跑的分析 —— M4 在真實影片上的第一份量化 + M5 的長時間行為。

**為什麼不用 acceptance_m5.py**:那支的契約是「吃區間級真值 CSV,判三個率」。
我們沒有逐幀真值,而且三個率有兩個在單人資料上結構性地不可量測。硬塞會讓
未來的人以為單人資料可以走驗收流程 —— 那正是本專案最想避免的誤導。

這支改用一個不需要標註的判準:**全片自始至終只有一個人**(已由資料集文件與
九視角對照圖確認)。於是
  · 系統應該全程只開 1 個 chef_id → `total_chefs - 1` 就是碎裂次數
  · 每幀 active track 數應為 1 → 0 是漏偵、≥2 是幽靈軌跡
兩者都是硬事實推導出來的,比任何需要標註的指標都直接。

⚠ 誤併率在這裡**永遠是 null**。單一 gt 身份時它結構上量不到(見 metrics.py 的
  說明),填 0 會變成假的好成績 —— 那是本專案踩過最嚴重的坑的同一種形狀。

用法:
    python scripts/eval_m5_longrun.py --run-dir results/m5_full
    python scripts/eval_m5_longrun.py --run-dir results/m5_full --emit-truth
"""
import argparse
import csv
import json
import statistics as st
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

TRUTH_CSV = """# ⚠ 這**不是**標註。EPFL 沒有任何逐幀人框或 person_id 真值。
# 這份檔案只編碼一個已確認的事實:全片自始至終只有一個人,三台鏡頭同步。
# 唯一目的是把 acceptance_m5.py 走一遍,證明它在單人資料上會誠實地報
# 「不完整」(exit 2),而不是像修復前那樣印「✅ 通過驗收」exit 0。
chef,camera,t_start,t_end
單人,cam1,0,780
單人,cam2,0,780
單人,cam3,0,780
"""


def read_csv(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def pct(xs, q):
    if not xs:
        return None
    s = sorted(xs)
    return s[min(len(s) - 1, int(len(s) * q))]


# ── §0 自檢 ───────────────────────────────────────────────────────────
def selfcheck(meta, tracks, tevents, resident, events):
    """每一條都對應一個已踩過或已識別的具體失效。任一項失敗就不准印指標。"""
    C = []

    def chk(ok, name, detail):
        C.append((bool(ok), name, detail))

    chk(not meta.get("truncated") and meta.get("coverage", 0) >= 0.999,
        "跑完整支影片",
        f"coverage={meta.get('coverage')} truncated={meta.get('truncated')} "
        f"—— 防的是 --max-frames 預設 120 的靜默 0.3% 截斷")
    chk(meta.get("n_det", 0) > 0 and all(v > 0 for v in meta.get("n_det_per_cam", {}).values()),
        "每台鏡頭都偵測到人",
        f"{meta.get('n_det_per_cam')} —— 防的是 person_cls 接錯(COCO=1/微調=0)"
        f"時靜默輸出 0 筆")
    chk(len(set(meta.get("per_cam_loops", {}).values())) == 1,
        "各鏡頭迴圈數一致",
        f"{meta.get('per_cam_loops')} —— 防的是某支影片提前耗盡、t_sec 靜默倒退")
    chk(tracks and tevents and events and len(resident) >= 10,
        "四個輸出檔都有資料",
        f"tracks={len(tracks)} events={len(tevents)} bindings={len(events)} "
        f"resident={len(resident)} —— 防的是空評估被當成完美")
    # EPFL 的 9 支影片是同一場次同步錄製、視野全部重疊,所以正確的 config
    # 必然是「links 為空、所有鏡頭對都 overlapping」。
    # ⚠ 第一版把數字硬編成 3(為三台跑寫的),擴到九台就誤判成失敗。
    #   斷言要寫成**不變量**(C(n,2) 對全重疊)而不是某一次的實際值。
    n_cam = len(meta.get("per_cam_loops") or {})
    want_pairs = n_cam * (n_cam - 1) // 2
    chk(meta.get("n_links") == 0 and meta.get("n_overlapping_pairs") == want_pairs,
        "讀到正確的拓撲 config(全重疊、無 links)",
        f"{n_cam} 台 → 應有 {want_pairs} 個重疊對,實際 "
        f"overlapping={meta.get('n_overlapping_pairs')} links={meta.get('n_links')}"
        f" —— 讀成範本 camera_topology.yaml 會讓多數鏡頭必然碎裂,"
        f"而症狀看起來像演算法爛")
    chk(meta.get("n_homographies") == 0
        and meta.get("ground_plane_effective") is False
        and meta.get("velocity_effective") is False,
        "地面校正/軌跡證據的狀態自洽",
        "沒有 homography 就必須兩條都不生效 —— 「開了卻不生效」與"
        "「沒開卻聲稱生效」兩個方向都要擋")
    ms = meta.get("ms_per_loop") or {}
    chk(ms.get("max", 0) < 20 * max(ms.get("p50", 1), 1),
        "沒有異常長停頓",
        f"max={ms.get('max')} vs p50={ms.get('p50')} —— 防的是 swap/記憶體壓力")
    return C


# ── §1 M4 ─────────────────────────────────────────────────────────────
def analyse_m4(tracks, tevents, meta):
    fps, stride = 30.0, meta["stride"]
    sec_per_loop = stride / fps
    out = {}
    by_cam = defaultdict(list)
    for r in tracks:
        by_cam[r["camera_id"]].append(r)

    for cam, rs in sorted(by_cam.items()):
        # 每個 loop 有幾條 active track。全片單人 → 1 是理想、0 是漏偵、≥2 是幽靈。
        per_loop = Counter()
        for r in rs:
            per_loop[int(r["loop_i"])] += 1
        n_loops = meta["n_loops"]
        hist = Counter(per_loop.get(i, 0) for i in range(n_loops))

        # track 存活長度:同一 track_id 出現的 loop 範圍
        span = defaultdict(lambda: [10 ** 9, -1])
        for r in rs:
            t, i = int(r["track_id"]), int(r["loop_i"])
            span[t][0], span[t][1] = min(span[t][0], i), max(span[t][1], i)
        lens = sorted((b - a + 1) * sec_per_loop for a, b in span.values())
        idle = sum(1 for r in rs if not r["conf"])      # Kalman 預測幀 = 沒配對上

        ev = Counter(e["kind"] for e in tevents if e["camera_id"] == cam)
        out[cam] = {
            "n_tracks": len(span),
            "tracks_per_min": round(len(span) / (n_loops * sec_per_loop / 60), 2),
            "active_per_loop": {str(k): hist.get(k, 0) for k in sorted(hist)},
            "miss_rate": round(hist.get(0, 0) / n_loops, 4),
            "ghost_rate": round(sum(v for k, v in hist.items() if k >= 2) / n_loops, 4),
            "lifetime_s": {"median": round(st.median(lens), 1) if lens else None,
                           "p10": round(pct(lens, .10), 1) if lens else None,
                           "p90": round(pct(lens, .90), 1) if lens else None,
                           "max": round(lens[-1], 1) if lens else None},
            "short_lived_pct": round(sum(1 for x in lens if x < 2.0) / len(lens), 4) if lens else None,
            "kalman_idle_pct": round(idle / len(rs), 4) if rs else None,
            "events": dict(ev),
            # 進了遮擋又被找回來的比例。M4 的 lost→reacquired 就是「撐過短暫遮擋」,
            # 而撐不過的那些會變成 removed,再出現時是新 track → M5 的碎裂來源。
            "occlusion_recovery": (round(ev.get("reacquired", 0) / ev["lost_track"], 3)
                                   if ev.get("lost_track") else None),
        }
    return out


# ── §2 M5 ─────────────────────────────────────────────────────────────
def analyse_m5(events, meta):
    thr = meta.get("llr_threshold") or 1.609
    # ⚠ **第一次**綁定必然「開新身份」—— 那是建立初始身份,不是碎裂。
    #   第一版把它算進分母與分子,於是同一份報告同時印出「碎裂 0 次」與
    #   「碎裂率 5.6%」(= 1/18)。自相矛盾就是指標寫錯的訊號。
    #   metrics.binding_outcomes 用 `gt in seen` 表達同一件事;這裡沒有 gt,
    #   但「全片單人」意味著整場只有第一次是初始化,其餘都是轉場。
    decisions = events[1:] if events else []
    opened = [e for e in decisions if not e["matched"]]
    scores = [e["score"] for e in decisions if e["matched"]]
    margins = [s - thr for s in scores]
    return {
        # 主判準:全片一人 → 理想是 1 個 chef_id。不需要任何標註。
        "total_chefs": meta["total_chefs"],
        "expected_chefs": 1,
        "fragmentation_count": meta["total_chefs"] - 1,
        "n_bindings": len(events),
        "n_transitions": len(decisions),      # 扣掉第一次的初始化
        "n_opened_new": len(opened),
        "p_break": round(len(opened) / len(decisions), 4) if decisions else None,
        # ⚠ 永遠是 None。單一 gt 身份時誤併分支是死碼(見 metrics.binding_outcomes)。
        "p_false_merge": None,
        "p_false_merge_reason": "資料只有一個人 → 誤併結構上不可量測,任何數字都不是誤併率",
        "candidate_histogram": meta["candidate_histogram"],
        "llr_threshold": round(thr, 3),
        "score_margin": {"median": round(st.median(margins), 2) if margins else None,
                         "min": round(min(margins), 2) if margins else None,
                         "within_1nat_pct": (round(sum(1 for m in margins if m < 1.0)
                                                   / len(margins), 4) if margins else None)},
        "opened_new_detail": [{"t_sec": e["t_sec"], "camera_id": e["camera_id"],
                               "track_id": e["track_id"], "chef_id": e["chef_id"]}
                              for e in opened],
    }


# ── §3 記憶體 ─────────────────────────────────────────────────────────
def analyse_memory(resident, meta):
    def col(name):
        return [float(r[name]) for r in resident if r.get(name) not in (None, "")]

    def slope_per_hour(name):
        """用後半段擬合,避開暖機。單位:每小時影片時間的增量。"""
        half = resident[len(resident) // 2:]
        if len(half) < 3:
            return None
        xs = [float(r["t_sec"]) for r in half]
        ys = [float(r[name]) for r in half]
        n, sx, sy = len(xs), sum(xs), sum(ys)
        sxx = sum(x * x for x in xs)
        sxy = sum(x * y for x, y in zip(xs, ys))
        den = n * sxx - sx * sx
        return round((n * sxy - sx * sy) / den * 3600, 3) if den else None

    keys = [k for k in resident[0] if k not in ("loop_i", "t_sec")]
    out = {}
    for k in keys:
        vals = col(k)
        if not vals:
            continue
        out[k] = {"max": max(vals), "final": vals[-1], "per_hour": slope_per_hour(k)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default=str(ROOT / "results" / "m5_full"))
    ap.add_argument("--emit-truth", action="store_true",
                    help="產生單人真值 CSV(用途見檔頭註解)")
    args = ap.parse_args()
    d = Path(args.run_dir)

    if args.emit_truth:
        p = d / "truth_singleperson.csv"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(TRUTH_CSV, encoding="utf-8")
        print(f"已寫出 {p}")
        return 0

    meta = json.loads((d / "run_meta.json").read_text(encoding="utf-8"))
    tracks = read_csv(d / "tracks.csv")
    tevents = read_csv(d / "track_events.csv")
    resident = read_csv(d / "resident.csv")
    events = [json.loads(l) for l in (d / "chef_events.jsonl").read_text(
        encoding="utf-8").splitlines() if l.strip()]

    L = []
    L.append("=" * 74)
    L.append("EPFL 單場次全長 M4+M5 端到端評估")
    L.append("=" * 74)

    # ── §0 ──
    checks = selfcheck(meta, tracks, tevents, resident, events)
    L.append("\n§0 自檢 —— 不全過就不印任何指標\n")
    for ok, name, detail in checks:
        L.append(f"  {'✅' if ok else '❌'} {name}")
        L.append(f"       {detail}")
    if not all(ok for ok, _, _ in checks):
        L.append("\n❌ 自檢未通過。這次的資料不足以支撐任何結論,不印指標。")
        print("\n".join(L))
        return 1

    v = meta["videos"][list(meta["videos"])[0]]
    L.append(f"\n跑法:{meta['n_loops']} 迴圈 × {len(meta['per_cam_loops'])} 鏡頭 · "
             f"stride={meta['stride']} · 覆蓋 {meta['coverage']*100:.1f}% · "
             f"{v['nb_frames']/v['fps']:.0f} 秒影片 · 耗時 {meta['wall_seconds']/60:.1f} 分")

    m4 = analyse_m4(tracks, tevents, meta)
    L.append("\n" + "─" * 74)
    L.append("§1 M4 追蹤(真實影片上的第一份量化)")
    L.append("─" * 74)
    L.append("  全片只有一個人 → 每迴圈的 active track 數理想值是 1。")
    L.append("  0 = 漏偵(M3 recall 的代理);≥2 = 幽靈軌跡(誤偵/重複框的上界)。\n")
    L.append(f"  {'鏡頭':<7}{'track數':>8}{'漏偵':>8}{'幽靈':>8}"
             f"{'存活中位':>10}{'短命<2s':>9}{'空轉':>8}{'遮擋救回':>9}")
    L.append("  " + "-" * 66)
    for cam, s in m4.items():
        rec = s["occlusion_recovery"]
        rec_s = "—" if rec is None else f"{rec * 100:.0f}%"
        life = s["lifetime_s"]["median"]
        L.append(f"  {cam:<7}{s['n_tracks']:>8}{s['miss_rate']*100:>7.1f}%"
                 f"{s['ghost_rate']*100:>7.1f}%{f'{life}s':>10}"
                 f"{(s['short_lived_pct'] or 0)*100:>8.0f}%"
                 f"{(s['kalman_idle_pct'] or 0)*100:>7.0f}%{rec_s:>9}")
    for cam, s in m4.items():
        L.append(f"  {cam} 事件:{s['events']}")

    m5 = analyse_m5(events, meta)
    L.append("\n" + "─" * 74)
    L.append("§2 M5 跨鏡頭身份")
    L.append("─" * 74)
    L.append(f"  chef_id 總數        {m5['total_chefs']}(理想 1,全片單人)"
             f"  → 碎裂 {m5['fragmentation_count']} 次")
    L.append(f"  綁定決策            {m5['n_bindings']} 次(首次為初始化,"
             f"其餘 {m5['n_transitions']} 次是真正的重新綁定)")
    L.append(f"  開新身份            {m5['n_opened_new']} 次(不含初始化)")
    L.append(f"  碎裂率              {m5['p_break']*100:.1f}%"
             f"(分母是 {m5['n_transitions']} 次重新綁定)")
    L.append(f"  誤併率              **不可量測** —— {m5['p_false_merge_reason']}")
    L.append(f"  候選數分布          {m5['candidate_histogram']}")
    if (m5["candidate_histogram"].get("2", 0) + m5["candidate_histogram"].get("3", 0)) == 0:
        L.append("      → 從來沒有兩個候選同時競爭 → 外觀 embedder 的品質"
                 "在這次評估中**完全沒有影響結果的機會**")
    sm = m5["score_margin"]
    L.append(f"  決策餘裕(距門檻 {m5['llr_threshold']} nats)"
             f"  中位 {sm['median']} / 最小 {sm['min']} / "
             f"1 nat 內佔 {(sm['within_1nat_pct'] or 0)*100:.0f}%")
    if m5["opened_new_detail"]:
        L.append("\n  開新身份的每一次(要肉眼稽核的就是這些幀):")
        for e in m5["opened_new_detail"][:20]:
            L.append(f"    t={e['t_sec']:>7.2f}s  {e['camera_id']}  "
                     f"track {e['track_id']} → chef {e['chef_id']}")

    mem = analyse_memory(resident, meta)
    L.append("\n" + "─" * 74)
    L.append("§3 記憶體有界性(規格驗收標準之一)")
    L.append("─" * 74)
    L.append(f"  {'項目':<22}{'最大':>9}{'最終':>9}{'每小時增量':>12}")
    L.append("  " + "-" * 52)
    for k, s in mem.items():
        rate = s["per_hour"]
        rate_s = "—" if rate is None else f"{rate:+.2f}"
        L.append(f"  {k:<22}{s['max']:>9.1f}{s['final']:>9.1f}{rate_s:>12}")
    L.append("\n  ⚠ M4 的 _seen_ids 是單調累積的集合,結構上是 O(累計 track 數) 不是 O(1)。"
             "\n    數量級可能可忽略,但形狀要照實報。")

    ms = meta["ms_per_loop"]
    L.append("\n" + "─" * 74)
    L.append("§4 吞吐量")
    L.append("─" * 74)
    L.append(f"  每迴圈 p50 {ms['p50']:.0f} / p95 {ms['p95']:.0f} / max {ms['max']:.0f} ms"
             f"({len(meta['per_cam_loops'])} 鏡頭,純 CPU)")
    L.append(f"  等效處理速度:{1000/ms['p50']*meta['stride']:.1f} 影片幀/秒 "
             f"= 即時的 {1000/ms['p50']*meta['stride']/30:.2f} 倍")

    L.append("\n" + "─" * 74)
    L.append("§5 這次評估**不能**回答什麼(必讀)")
    L.append("─" * 74)
    L.append("""
  1. 誤併率 —— 資料只有一個人,誤併分支結構上是死碼。M5 唯一剩下的失效模式
     (模擬量到 5.6%,預算 1%)這次完全沒有被觸碰。

  2. 轉場時間路徑 —— EPFL 九台鏡頭是**全重疊**佈局,人同時出現在所有鏡頭裡。
     跨鏡頭綁定全走 overlap 路徑的常數 overlap_llr=5.0(已知過度自信)。
     CLM 的主機制「不重疊鏡頭靠轉場時間推測」在這次評估中**零覆蓋**。

  3. 地面校正與軌跡證據 —— 沒有 homography → world_xy=None → 這兩條把誤併
     從 72% 壓到 5.6% 的關鍵證據**零貢獻**。結果好看不能歸功於它們。

  4. MOTA / ID-switch —— 沒有逐幀人框與 track_id 真值,只能用「全片單人」
     這個事實推導間接指標。§1 的漏偵率與幽靈率是代理值,不是標準指標。

  要補上這四項,需要的是**多人、同步、跨鏡頭、帶 chef_id 真值**的自錄資料
  —— 也就是交付套件裡走動測試要取得的東西。""")

    txt = "\n".join(L)
    (d / "eval_report.txt").write_text(txt, encoding="utf-8")
    (d / "eval_report.json").write_text(json.dumps(
        {"meta": meta, "selfcheck": [{"ok": o, "name": n, "detail": dd}
                                     for o, n, dd in checks],
         "m4": m4, "m5": m5, "memory": mem},
        ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(txt)
    print(f"\n已存 {d/'eval_report.txt'} 與 eval_report.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
