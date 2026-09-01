"""M5 驗收:拿業主標的真值,評 chef_id 的正確性。

業主不會逐幀標框。這支腳本吃的是**區間級**真值 —— 走動測試時記下
「誰、在哪台鏡頭、從幾秒到幾秒」,這是人拿手機就能記的粒度。

輸出的是可以寫進驗收報告的三個數字:

  IDF1        身份保持的整體品質(MTMC 領域的標準指標,第三方認得)
  碎裂率      同一人被拆成多個 chef_id → 事件序列斷掉、計時歸零
  誤併率      綁到別人身上 → **靜默漏報**,下游 M7/M8 都驗不出來

⚠ 兩者不可加總成單一「錯誤率」:碎裂可被「chef_id 數 ≠ 排班人數」自動抓到,
  誤併完全靜默。誤併嚴重約 5 倍,驗收門檻據此推導。

真值 CSV(有 header):
    chef,camera,t_start,t_end
    張師傅,cam1,0,45.5
    張師傅,cam2,52,90
    李師傅,cam1,10,88

系統輸出:scripts/m5_track_video.py 產生的 chef_events.jsonl

用法:
    python scripts/acceptance_m5.py --truth walktest.csv \
        --events results/m5_track/chef_events.jsonl --headcount 3
    python scripts/acceptance_m5.py --template walktest.csv
"""
import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from m5_reid import metrics                                   # noqa: E402

TEMPLATE = """chef,camera,t_start,t_end
# 走動測試的紀錄。一列 = 「某人在某鏡頭出現的一個連續區間」。
# chef:用人看得懂的代號(張師傅、A、1 都可以),同一個人請用同一個代號
# camera:要與 config 的 camera_id 一致
# t_start / t_end:該區間的起訖秒數(從影片開頭算)
張師傅,cam1,0,45.5
張師傅,cam2,52,90
李師傅,cam1,10,88
"""

# 驗收門檻。推導見 docs/M5_v3_重設計與可行性驗證_20260824.md Phase 1:
# 「未洗手」事件鏈約含 3 次跨鏡頭交接,要 recall ≥ 0.80 → (1-p)³ ≥ 0.8 → p ≤ 7.2%,
# 留 M3/M4/M6 的誤差預算後訂 5%。誤併是靜默漏報,加嚴 5 倍。
BUDGET = {"p_break": 0.05, "p_false_merge": 0.01, "idf1": 0.90}


def read_truth(path):
    rows = []
    with open(path, encoding="utf-8-sig") as f:
        for r in csv.DictReader(l for l in f if not l.lstrip().startswith("#")):
            if not (r.get("chef") or "").strip():
                continue
            rows.append((r["chef"].strip(), r["camera"].strip(),
                         float(r["t_start"]), float(r["t_end"])))
    return rows


def read_events(path):
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def match(events, truth, tol_s=1.0):
    """把每筆綁定決策對到真值區間。回傳 (records, 未對上的筆數)。

    records = [(真實身份, 系統給的 chef_id, matched, 是不是轉場)],供 metrics 用。
    """
    by_cam = {}
    for chef, cam, a, b in truth:
        by_cam.setdefault(cam, []).append((a, b, chef))
    recs, seen, unmatched = [], set(), 0
    for e in sorted(events, key=lambda x: x["t_sec"]):
        t, cam = e["t_sec"], e["camera_id"]
        hit = [c for a, b, c in by_cam.get(cam, [])
               if a - tol_s <= t <= b + tol_s]
        if not hit:
            unmatched += 1
            continue
        gt = hit[0]
        recs.append((gt, e["chef_id"], bool(e["matched"]), gt in seen))
        seen.add(gt)
    return recs, unmatched


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--truth")
    ap.add_argument("--events", default=str(ROOT / "results" / "m5_track" / "chef_events.jsonl"))
    ap.add_argument("--headcount", type=int, default=None, help="該次測試的實際人數")
    ap.add_argument("--tol", type=float, default=1.0, help="時間對齊容差(秒)")
    ap.add_argument("--template", help="產生空白真值範本")
    ap.add_argument("--out", default=str(ROOT / "results" / "m5_track" / "acceptance.json"))
    args = ap.parse_args()

    if args.template:
        Path(args.template).write_text(TEMPLATE, encoding="utf-8")
        print(f"真值範本已寫到 {args.template}")
        return
    if not args.truth:
        raise SystemExit("要給 --truth(標好的真值)或 --template")

    truth = read_truth(args.truth)
    events = read_events(args.events)
    recs, unmatched = match(events, truth, args.tol)
    n_people = args.headcount or len({c for c, _, _, _ in truth})
    s = metrics.summarize(recs, expected_headcount=n_people)

    print("=" * 68)
    print("M5 驗收結果")
    print("=" * 68)
    print(f"  真值:{len(truth)} 個區間、{n_people} 位人員")
    print(f"  系統:{len(events)} 次綁定決策,對上真值 {len(recs)} 筆"
          + (f"、**{unmatched} 筆對不上**" if unmatched else ""))
    if unmatched:
        print("    ⚠ 對不上表示系統在真值沒涵蓋的時段/鏡頭偵測到人 ——")
        print("      可能是真值漏標,也可能是誤偵測。先釐清再看下面的數字。")
    if not recs:
        raise SystemExit("沒有任何決策對得上真值,無法評估")

    print()
    print(f"  {'指標':<14}{'實測':>10}{'門檻':>10}   判定")
    print("  " + "-" * 46)
    rows = [("IDF1", s["idf1"], BUDGET["idf1"], "ge"),
            ("碎裂率", s["p_break"], BUDGET["p_break"], "le"),
            ("誤併率", s["p_false_merge"], BUDGET["p_false_merge"], "le")]
    # ⚠ 「沒量到」必須與「通過」分開,而且要是不同的 exit code。
    #   舊版在 val is None 時只印「無資料」然後 continue,**完全不碰 passed**
    #   (初始化為 True)→ 印出「✅ 通過驗收」exit 0,實際什麼都沒量到。
    #   results/m5_track/acceptance.json 就是這樣來的:4 筆事件裡 2 筆對不上真值,
    #   兩個率都 None,卻 passed: true。單人資料會 100% 觸發這條路徑。
    passed, unmeasurable = True, []
    for name, val, thr, direction in rows:
        if val is None:
            why = s.get("fm_unmeasurable_reason") or "沒有樣本"
            print(f"  {name:<14}{'不可量測':>10}{thr:>10.2f}   ⚠ {why}")
            unmeasurable.append(name)
            continue
        ok = (val >= thr) if direction == "ge" else (val <= thr)
        passed &= ok
        fmt = f"{val:.3f}" if name == "IDF1" else f"{val*100:.1f}%"
        tfm = f"≥{thr:.2f}" if direction == "ge" else f"≤{thr*100:.0f}%"
        print(f"  {name:<14}{fmt:>10}{tfm:>10}   {'✅ 通過' if ok else '❌ 未達標'}")

    hc = s.get("headcount") or {}
    print()
    print(f"  身份數:系統 {hc.get('n_pred_ids')} 個 vs 實際 {n_people} 位"
          + (f"  ← ⚠ {hc['kind']}" if hc.get("kind") else "  ✅ 相符"))
    print(f"  每人被拆成幾個身份:{s['fragmentation']}")
    print(f"  ID 切換次數:{s['id_switches']}")

    print()
    print("  怎麼讀:")
    print("    碎裂 = 同一人被拆成多個 chef_id → 事件序列斷掉、計時歸零")
    print("    誤併 = 綁到別人身上 → **靜默漏報**,下游 M7/M8 都驗不出來")
    print("    兩者不可加總:碎裂可被「身份數 ≠ 人數」抓到,誤併完全靜默,")
    print("    所以誤併的門檻嚴 5 倍。")

    # 對不上真值的事件也讓結果不完整 —— 舊版只印警告,不影響判定。
    if unmatched:
        unmeasurable.append(f"{unmatched} 筆事件對不上真值")

    verdict = "incomplete" if unmeasurable else ("pass" if passed else "fail")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(
        {"n_truth_intervals": len(truth), "n_people": n_people,
         "n_events": len(events), "n_matched": len(recs), "n_unmatched": unmatched,
         "metrics": s, "budget": BUDGET, "verdict": verdict,
         "unmeasurable": unmeasurable,
         # ⚠ passed 只在真的量到全部指標時才是布林;不完整時是 null,
         #   否則任何讀 JSON 的下游都會把「沒量到」讀成「沒通過」或「通過」。
         "passed": None if verdict == "incomplete" else bool(passed)},
        ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"\n  已存 {args.out}")

    if verdict == "incomplete":
        print(f"\n  總判定:⚠ **不完整** —— {'、'.join(unmeasurable)}")
        print("    這**不是**通過驗收。要判定通過,必須三個指標都真的量到。")
        print("    (單人資料一定會走到這裡:誤併率結構上量不到,見 metrics.py 的說明。)")
    else:
        print(f"\n  總判定:{'✅ 通過驗收' if passed else '❌ 未達驗收標準'}")
    sys.exit({"pass": 0, "fail": 1, "incomplete": 2}[verdict])


if __name__ == "__main__":
    main()
