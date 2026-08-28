"""從平面圖量到的距離,產生 M5 的相機拓撲 config。

業主只有平面圖,沒有轉場時間的實測值。這支腳本讓他們**只回答看得懂的問題**
(哪台走得到哪台、走過去幾公尺、視野有沒有重疊),其餘由腳本換算。

換算依據:
    mean_s = 步行路徑距離 / 0.9        0.9 m/s 是廚房內的實際步速
    std_s  = 0.35 × mean_s             由「趕時間 1.6 m/s ~ 慢走 0.6 m/s」推得

⚠ 必須填「**步行路徑距離**」不是直線距離。兩台鏡頭直線 3 公尺,但中間隔著
  中島要繞過去走 8 公尺 —— 要填 8。這是最容易填錯、而且錯了會系統性低估的一項。

⚠ 這些是**起手值**。取得實際影片後應改用 scripts/calibrate_topology.py
  從真實轉場時間重估(st-ReID 的作法),取代估算。

輸入 CSV(有 header,順序不拘):
    from,to,walk_distance_m,overlapping
    cam1,cam2,8.0,no
    cam2,cam1,8.0,no
    cam1,cam3,,yes            # 重疊的話距離留空

用法:
    python scripts/build_camera_topology.py --csv links.csv --out configs/camera_topology.yaml
    python scripts/build_camera_topology.py --template links.csv     # 產生空白範本
"""
import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

WALK_SPEED_MPS = 0.9        # 廚房內實際步速(平地 1.2~1.4,扣掉繞障礙、端東西、閃人)
SIGMA_RATIO = 0.35          # 由 0.6~1.6 m/s 的速度區間推得

TEMPLATE = """from,to,walk_distance_m,overlapping
# 每一條「可以走過去的路徑」填一列。雙向請各填一列(來回距離可以不同)。
# walk_distance_m:兩台鏡頭視野邊界之間的**步行路徑距離**(繞過障礙,不是直線!)
# overlapping:兩台鏡頭的視野有沒有重疊(yes/no)。重疊的話距離留空。
cam1,cam2,8.0,no
cam2,cam1,8.0,no
cam1,cam3,,yes
"""


def read_rows(path):
    rows = []
    with open(path, encoding="utf-8-sig") as f:
        for r in csv.DictReader(l for l in f if not l.lstrip().startswith("#")):
            if not (r.get("from") or "").strip():
                continue
            rows.append({k: (v or "").strip() for k, v in r.items()})
    return rows


def build(rows, lost_track_buffer=30, fps=30.0):
    links, overlapping, warnings = [], [], []
    seen = set()
    for r in rows:
        a, b = r["from"], r["to"]
        if (a, b) in seen:
            warnings.append(f"{a}→{b} 重複填寫,只取第一筆")
            continue
        seen.add((a, b))
        if r.get("overlapping", "").lower() in ("yes", "y", "true", "1"):
            pair = sorted([a, b])
            if pair not in overlapping:
                overlapping.append(pair)
            continue
        d = r.get("walk_distance_m", "")
        if not d:
            warnings.append(f"{a}→{b} 沒填距離且不是 overlapping → 略過")
            continue
        d = float(d)
        if d <= 0:
            warnings.append(f"{a}→{b} 距離 {d} 不合理 → 略過")
            continue
        mean_s = round(d / WALK_SPEED_MPS, 1)
        std_s = round(mean_s * SIGMA_RATIO, 1)
        links.append({"from": a, "to": b, "mean_s": mean_s, "std_s": std_s,
                      "_distance_m": d})

    # 與 M4 的跨模組約束:走得太快的路徑,人在進入候選名單前就抵達了
    gallery_delay = lost_track_buffer / fps
    for l in links:
        earliest = l["mean_s"] - 2 * l["std_s"]
        if earliest < gallery_delay:
            warnings.append(
                f"⚠ {l['from']}→{l['to']} 只要 {l['mean_s']:.1f}s,太短 —— "
                f"M4 需要 {gallery_delay:.1f}s 才會把離場的人放進候選名單。"
                f"建議把這兩台改列為 overlapping,或降低 lost_track_buffer。")

    # 單向連結
    pairs = {(l["from"], l["to"]) for l in links}
    for (a, b) in sorted(pairs):
        if (b, a) not in pairs and sorted([a, b]) not in overlapping:
            warnings.append(f"⚠ {a}→{b} 有填但 {b}→{a} 沒有 —— "
                            "廚師走回頭路時會被判為新人。若可雙向通行請補一列。")
    return links, overlapping, warnings


def to_yaml(links, overlapping, cameras):
    L = ["# 由 scripts/build_camera_topology.py 從平面圖距離產生。",
         "# ⚠ mean_s / std_s 是**估算值**(距離 ÷ 0.9 m/s)。取得實際影片後應改用",
         "#   scripts/calibrate_topology.py 從真實轉場時間重估,取代這裡的估算。",
         "camera_topology:", "", "  fusion:", "    mode: llr",
         "    background_arrival_hz: 0.00167      # 「真正的新人」出現的速率(次/秒)",
         "    cost_false_merge_over_break: 5.0    # 誤併比碎裂嚴重幾倍",
         "    transit_model: loiter",
         "    p_loiter: 0.15", "    tau_loiter_s: 20.0",
         "    appearance_profile: dinov2", "    appearance_clip: null",
         "    overlap_llr: 5.0", "    overlap_window_s: 0.5",
         "    same_camera: { enabled: true, tau_break_s: 2.0, max_gap_s: 15.0 }",
         "    unknown_path: { enabled: false }",
         "    direction: { enabled: false, q: 0.85, n_zones: 3 }",
         "", "  links:"]
    for l in links:
        L.append(f"    # {l['from']}→{l['to']}:步行 {l['_distance_m']:.1f} m "
                 f"÷ {WALK_SPEED_MPS} m/s")
        L.append(f"    - {{ from: {l['from']}, to: {l['to']}, "
                 f"mean_s: {l['mean_s']}, std_s: {l['std_s']} }}")
    L += ["", "  overlapping:"]
    if overlapping:
        for a, b in overlapping:
            L.append(f"    - [{a}, {b}]")
    else:
        L.append("    []   # ⚠ 沒有任何重疊鏡頭 → 跨鏡頭只能靠猜轉場時間,誤差大得多")
    L += ["", "  clock:", "    max_skew_s: 0.2   # 部署需求:鏡頭間 NTP 殘餘偏移上限",
          "", "  cameras:"]
    for c in cameras:
        L += [f"    {c}:", "      clock_offset_s: 0.0",
              "      # homography:   # 地面校正:挑 ≥4 個地面上不共線的點",
              "      #   image_points: [[u1,v1], [u2,v2], [u3,v3], [u4,v4]]",
              "      #   world_points: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]"]
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", help="填好的連結表")
    ap.add_argument("--template", help="產生空白範本到這個路徑")
    ap.add_argument("--out", default=str(ROOT / "configs" / "camera_topology.generated.yaml"))
    ap.add_argument("--lost-track-buffer", type=int, default=30)
    ap.add_argument("--fps", type=float, default=30.0)
    args = ap.parse_args()

    if args.template:
        Path(args.template).write_text(TEMPLATE, encoding="utf-8")
        print(f"範本已寫到 {args.template}")
        print("填好之後跑:python scripts/build_camera_topology.py --csv "
              f"{args.template} --out configs/camera_topology.yaml")
        return
    if not args.csv:
        raise SystemExit("要給 --csv(填好的表)或 --template(產生空白範本)")

    rows = read_rows(args.csv)
    links, overlapping, warnings = build(rows, args.lost_track_buffer, args.fps)
    cameras = sorted({c for l in links for c in (l["from"], l["to"])}
                     | {c for p in overlapping for c in p})

    print("=" * 68)
    print(f"讀到 {len(rows)} 列 → {len(links)} 條連結、{len(overlapping)} 對重疊鏡頭")
    print("=" * 68)
    if links:
        print(f"  {'連結':<16}{'步行距離':>10}{'轉場時間 μ':>12}{'變異 σ':>10}")
        print("  " + "-" * 48)
        for l in links:
            print(f"  {l['from']+'→'+l['to']:<16}{l['_distance_m']:>9.1f}m"
                  f"{l['mean_s']:>11.1f}s{l['std_s']:>9.1f}s")
    if overlapping:
        print(f"\n  重疊鏡頭對:{', '.join('↔'.join(p) for p in overlapping)}")
        print("  (重疊是最可靠的關聯方式 —— 不必猜走了多久,直接看幾何)")
    if warnings:
        print("\n提醒:")
        for w in warnings:
            print(f"  {w}")

    Path(args.out).write_text(to_yaml(links, overlapping, cameras), encoding="utf-8")
    print(f"\n已產生 {args.out}")
    print("\n下一步:")
    print("  1. python scripts/audit_m5_config.py --topology " + args.out)
    print("  2. 每台鏡頭補上 homography(地面校正)—— 沒有它,多人同時在場時")
    print("     系統分不出誰是誰(實測誤併率會從 14% 惡化到 72%)")


if __name__ == "__main__":
    main()
