"""M5 部署前 config 自檢 —— 業主佈署時第一個該跑的東西。

把「靜默失效」變成「開機就報錯」:拓撲填錯、σ 不合理、M4 的 lost_track_buffer
與轉場時間衝突、鏡頭時鐘沒同步 —— 這些都會讓 chef_id 一直開新的,但現場看不出
原因。這支腳本在還沒錄任何影片之前就能抓到。

用法:
  python scripts/audit_m5_config.py
  python scripts/audit_m5_config.py --headcount 4
  python scripts/audit_m5_config.py --estimate-skew coobs.csv   # 反推殘餘時鐘偏移

--estimate-skew 的 CSV 格式(有 header):
  cam_a,t_a,cam_b,t_b        ← 同一人在兩台**視野重疊**鏡頭上被同時看到的時間戳
有 ERROR 時以 exit 1 結束,方便接進部署流程。
"""
import argparse
import csv
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from m5_reid.audit import (ERROR, audit_topology, effective_window,   # noqa: E402
                           estimate_clock_skew, format_findings)
from m5_reid.spatiotemporal import CameraTopology                     # noqa: E402


def _load_tracker_cfg(path):
    if not Path(path).exists():
        return None
    with open(path, encoding="utf-8") as f:
        return (yaml.safe_load(f) or {}).get("tracker")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topology", default=str(ROOT / "configs" / "camera_topology.yaml"))
    ap.add_argument("--tracker", default=str(ROOT / "configs" / "tracker.yaml"))
    ap.add_argument("--headcount", type=int, default=None, help="該班排班人數")
    ap.add_argument("--estimate-skew", default=None, help="重疊鏡頭同時觀測的 CSV")
    args = ap.parse_args()

    topo = CameraTopology.from_yaml(args.topology)
    tracker_cfg = _load_tracker_cfg(args.tracker)

    print("=" * 74)
    print("M5 config 自檢")
    print("=" * 74)
    print(f"  拓撲          : {args.topology}")
    print(f"  M4 設定       : {args.tracker}"
          f"{'' if tracker_cfg else '  (讀不到 → 跳過跨模組檢查)'}")
    print(f"  融合模式      : {topo.fusion['mode']}")
    print(f"  相機          : {', '.join(sorted(topo.all_cameras()))}")
    print(f"  時鐘偏移      : {topo.clock_offset or '(未設定)'}")
    print()

    print("有效時間窗(實際會接受的 Δt 範圍)")
    print("  連結            | 外觀中性        | 外觀最有利      | μ / σ")
    print("  " + "-" * 68)
    for (a, b), (mu, sd) in sorted(topo.links.items()):
        lo0, hi0 = effective_window(topo, a, b, app_llr=0.0)
        lo1, hi1 = effective_window(topo, a, b, app_llr=topo.app_lr.max_abs_llr())
        f0 = f"{lo0:.2f}~{hi0:.2f}s" if lo0 is not None else "(全拒)"
        f1 = f"{lo1:.2f}~{hi1:.2f}s" if lo1 is not None else "(全拒)"
        print(f"  {a}→{b:<10} | {f0:<15} | {f1:<15} | {mu:.1f} / {sd:.1f}")
    print()

    findings = audit_topology(topo, tracker_cfg, expected_headcount=args.headcount)
    print(format_findings(findings))

    if args.estimate_skew:
        print()
        print("=" * 74)
        print("殘餘時鐘偏移估計(從重疊鏡頭的同時觀測反推)")
        print("=" * 74)
        rows = []
        with open(args.estimate_skew, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                rows.append((r["cam_a"], float(r["t_a"]), r["cam_b"], float(r["t_b"])))
        est = estimate_clock_skew(rows)
        if not est:
            print("  樣本不足(每組需 ≥5 筆)。")
        for (a, b), (med, n, iqr) in sorted(est.items()):
            tol = float(topo.clock["max_skew_s"])
            flag = "✗ 超出容忍" if abs(med) > tol else "✅"
            print(f"  {a} vs {b}: 偏移 {med:+.3f}s (n={n}, IQR={iqr:.3f}s)  {flag}")
            if abs(med) > tol:
                print(f"      → 在 cameras.{b}.clock_offset_s 填 {-med:+.3f}"
                      f"(或校正 {b} 的系統時間)")

    sys.exit(1 if any(f.level == ERROR for f in findings) else 0)


if __name__ == "__main__":
    main()
