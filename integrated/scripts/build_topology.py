"""由「鏡頭之間怎麼走、走多遠」的表格,產生人員追蹤用的相機拓撲檔(run_pipeline.py --topology)。

步驟:
  1. 產生空白表格:python scripts/build_topology.py --template configs/topology_links.csv
  2. 依平面圖填表(每條可走的路徑填一列,雙向各一列):
       from,to,walk_distance_m,overlapping
       cam1,cam2,8.0,no        cam1 視野邊界走到 cam2 視野邊界的「步行路徑距離」(繞過障礙,不是直線)
       cam2,cam1,8.0,no
       cam1,cam3,,yes          兩台畫面有重疊:距離留空、填 yes
  3. 產生拓撲檔:python scripts/build_topology.py --csv configs/topology_links.csv --out configs/topology.yaml
  4. 執行時指定:python run_pipeline.py ... --topology configs/topology.yaml

換算方式:移動時間平均 = 距離 / 0.9 m/s(廚房內步速),標準差 = 平均 x 0.35。
融合參數(fusion、clock)沿用 pipeline_config.yaml 的 camera_topology。
"""
import argparse
import csv
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
WALK_SPEED_MPS = 0.9
SIGMA_RATIO = 0.35

TEMPLATE = """from,to,walk_distance_m,overlapping
# 每條可以走過去的路徑填一列,雙向各填一列。以 # 開頭的列會被忽略。
# walk_distance_m:兩台鏡頭視野邊界之間的步行路徑距離(公尺,繞過障礙,不是直線)
# overlapping:兩台鏡頭畫面有重疊填 yes(距離留空),否則填 no
cam1,cam2,8.0,no
cam2,cam1,8.0,no
cam1,cam3,,yes
"""


def read_rows(path):
    with open(path, encoding="utf-8-sig") as f:
        rows = csv.DictReader(l for l in f if not l.lstrip().startswith("#"))
        return [{k: (v or "").strip() for k, v in r.items()} for r in rows if (r.get("from") or "").strip()]


def build(rows):
    links, overlapping, warnings, seen = [], [], [], set()
    for r in rows:
        a, b = r["from"], r["to"]
        if (a, b) in seen:
            warnings.append(f"{a}→{b} 重複填寫,只取第一列")
            continue
        seen.add((a, b))
        if r.get("overlapping", "").lower() in ("yes", "y", "true", "1"):
            if sorted([a, b]) not in overlapping:
                overlapping.append(sorted([a, b]))
            continue
        d = float(r["walk_distance_m"]) if r.get("walk_distance_m") else 0.0
        if d <= 0:
            warnings.append(f"{a}→{b} 沒有填距離且不是重疊鏡頭,略過")
            continue
        mean_s = round(d / WALK_SPEED_MPS, 2)
        links.append({"from": a, "to": b, "mean_s": mean_s, "std_s": round(mean_s * SIGMA_RATIO, 2)})
    pairs = {(l["from"], l["to"]) for l in links}
    for a, b in sorted(pairs):
        if (b, a) not in pairs and sorted([a, b]) not in overlapping:
            warnings.append(f"{a}→{b} 有填、{b}→{a} 沒填:往回走的人會被當成新的人。可雙向通行請補一列")
    return links, overlapping, warnings


def main():
    ap = argparse.ArgumentParser(description="由路徑表格產生相機拓撲檔")
    ap.add_argument("--template", help="產生空白表格到這個路徑")
    ap.add_argument("--csv", help="填好的表格")
    ap.add_argument("--out", default="configs/topology.yaml")
    args = ap.parse_args()

    if args.template:
        Path(args.template).parent.mkdir(parents=True, exist_ok=True)
        Path(args.template).write_text(TEMPLATE, encoding="utf-8")
        print(f"空白表格:{args.template}")
        return
    if not args.csv:
        raise SystemExit("請給 --template(產生空白表格)或 --csv(填好的表格)")

    links, overlapping, warnings = build(read_rows(args.csv))
    if not links and not overlapping:
        raise SystemExit("表格裡沒有任何路徑或重疊鏡頭")
    cams = sorted({c for l in links for c in (l["from"], l["to"])} | {c for p in overlapping for c in p})
    base = yaml.safe_load((ROOT / "pipeline_config.yaml").read_text(encoding="utf-8"))["camera_topology"]
    topo = {"camera_topology": {
        "fusion": base["fusion"],
        "links": links,
        "overlapping": overlapping,
        "clock": base["clock"],
        "cameras": {c: {"clock_offset_s": 0.0} for c in cams},
    }}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("# 由 scripts/build_topology.py 產生;鏡頭名稱須與 run_pipeline.py --cameras 相同\n"
                   + yaml.safe_dump(topo, allow_unicode=True, sort_keys=False, default_flow_style=None),
                   encoding="utf-8")
    print(f"鏡頭 {len(cams)} 台、路徑 {len(links)} 條、重疊 {len(overlapping)} 組")
    for l in links:
        print(f"  {l['from']}→{l['to']}:平均 {l['mean_s']} 秒,標準差 {l['std_s']} 秒")
    for w in warnings:
        print(f"注意:{w}")
    print(f"拓撲檔:{out}")


if __name__ == "__main__":
    main()
