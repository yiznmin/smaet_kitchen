"""把 CHIRLA 交接包的場地幾何轉成 M5 的拓撲設定(每序列一份)與校正報告。

預先登記:docs/M5_平面圖幾何接入_預先登記_20261013.md §4、§5。

## 讀什麼(全部來自交接包 level_search/data,**只讀**)

  floorplan/landmarks.json              人工地標(影像點 ↔ 示意圖點)與出入口在示意圖上的位置
  floorplan/door_zones.csv              各鏡頭畫面中出入口區域的中位點
  floorplan/camera_to_plan_refined.json 交接包重新擬合的投影矩陣(walk_pairs_final5h,逐序列留一)
  floorplan/site_floorplan.png          場地示意圖 → 可行走網格
  sync/camera_align.json                各序列畫面 → 參考畫面(seq_020)的對齊矩陣

## 產出

  configs/plan_grid/{P1,P2,G2}/<seq>.yaml   拓撲(base_overlapfix + 平面圖座標)
  configs/plan_grid/plan_map.npz            可行走網格
  results/m5_plan/calib_report.json         σ、速度、面積、證據上限、各鏡頭落在可行走區外的比例

## 為什麼自己重寫而不 import 交接包的模組

交接包的 floorplan 模組一 import 就會連帶載入評估程式,並在它自己的資料夾底下建立
results/ 目錄 —— 建置我們的設定不該在別人的交付物裡留下東西。
下面的常數與作法逐一對應交接包原始碼,出處寫在各段註解。

⚠ σ、速度**只用推導集 seq_000/001/002 的標註、原始時間戳**估(§5),評估集不參與。
⚠ 單位是示意圖像素。示意圖不按比例,不換算成公尺。
"""
import argparse
import copy
import csv
import glob
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml
from scipy import ndimage

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from m5_reid.floorplan import PlanMap                      # noqa: E402

CAMS = [f"camera_{i}" for i in range(1, 8)]
SEQS = ["seq_000", "seq_001", "seq_002", "seq_004", "seq_006", "seq_007", "seq_020",
        "seq_024", "seq_025", "seq_026"]
DERIVE = ["seq_000", "seq_001", "seq_002"]

# ── 交接包 floorplan_cues.py 的場地常數(示意圖像素)──────────────────
ROOMS = {"O1": (130, 40, 470, 127), "L1": (110, 127, 262, 226), "L2": (262, 127, 348, 226),
         "L3": (348, 127, 465, 226), "H1": (100, 226, 520, 262)}
DOOR_LINKS = {"L2-H1 door": ("L2", "H1"), "L2-O1 door": ("L2", "O1"),
              "L2-L3 passage": ("L2", "L3"), "L3-H1 door": ("L3", "H1"),
              "L1-H1 door": ("L1", "H1")}
# ── 交接包 floorplan_global.py:(鏡頭, door_zones 區域) → 示意圖上的門 ──
DOOR_ZONES = {("camera_1", "Z1"): "L2-H1 door", ("camera_2", "Z1"): "L2-H1 door",
              ("camera_3", "Z2"): "L2-O1 door", ("camera_4", "Z1"): "L3-H1 door",
              ("camera_4", "Z2"): "L3-H1 door", ("camera_6", "Z2"): "L1-H1 door",
              ("camera_7", "Z1"): "L1-H1 door"}
# ── 交接包 floorplan_grid_cues.py 的可行走網格參數 ────────────────────
CELL, DOOR_RADIUS = 4, 8
FLOOR_GRAY, PINK_HUE, HUE_TOL, GREY_SAT = 170, 170, 12, 20
MAX_LABEL_AREA, WALL_GROW = 400, 2
REFINED_KEY = "walk_pairs_final5h"

STEP = 5                  # 每 5 影格取 1 格(與系統 stride 一致)
FPS = 30
MOVE_WIN, MOVE_RATIO = 15, 0.15   # 交接包的走動判定:前後 0.5 秒接地點位移 ≥ 0.15 倍框高
MAX_PAIRS, SEED = 3000, 20261013
MEDIAN_TO_SIGMA = math.sqrt(2.0 * math.log(2.0))   # 二維等向高斯:距離中位數 = σ × 1.1774


def walkable_mask(plan_png, doors):
    """示意圖 → 可行走網格 (ny, nx)。逐行對應交接包 floorplan_grid_cues.walkable_mask。"""
    img = cv2.imdecode(np.fromfile(str(plan_png), np.uint8), cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    pink = np.abs(hsv[..., 0].astype(int) - PINK_HUE) <= HUE_TOL
    floor = (gray >= FLOOR_GRAY) & (pink | (hsv[..., 1] < GREY_SAT))
    obstacle, n = ndimage.label(~floor)
    small = np.bincount(obstacle.ravel(), minlength=n + 1) < MAX_LABEL_AREA
    wall = ~floor & ~small[obstacle]
    wall = ndimage.binary_dilation(wall, iterations=WALL_GROW)
    edges = np.zeros(wall.shape, np.uint8)
    for x0, y0, x1, y1 in ROOMS.values():
        cv2.rectangle(edges, (x0, y0), (x1, y1), 1, 2 * WALL_GROW + 1)
    wall |= edges.astype(bool)
    yy, xx = np.mgrid[:wall.shape[0], :wall.shape[1]]
    for x, y in doors.values():
        wall[(xx - x) ** 2 + (yy - y) ** 2 <= DOOR_RADIUS ** 2] = False
    labels, _ = ndimage.label(~wall)
    seeds = {labels[(y0 + y1) // 2, (x0 + x1) // 2] for x0, y0, x1, y1 in ROOMS.values()} - {0}
    free = np.isin(labels, list(seeds))
    ny, nx = free.shape[0] // CELL, free.shape[1] // CELL
    cells = free[:ny * CELL, :nx * CELL].reshape(ny, CELL, nx, CELL)
    blocked = wall[:ny * CELL, :nx * CELL].reshape(ny, CELL, nx, CELL).any(axis=(1, 3))
    walk = (cells.mean(axis=(1, 3)) >= 0.5) & ~blocked
    comp, n = ndimage.label(walk, structure=np.ones((3, 3)))
    return comp == np.argmax(np.bincount(comp.ravel(), minlength=n + 1)[1:]) + 1


def landmark_homographies(landmarks, zones_csv):
    """人工地標 + 出入口區域中位點的最小平方解(交接包 landmark_homographies / anchors)。"""
    doors = landmarks["plan_doors"]
    pts = {c: ([p["img"] for p in landmarks.get(c, [])],
               [p["plan"] for p in landmarks.get(c, [])]) for c in CAMS}
    with open(zones_csv, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            door = DOOR_ZONES.get((r["camera"], r["zone"]))
            if door in doors:
                pts[r["camera"]][0].append([float(r["median_x"]), float(r["median_y"])])
                pts[r["camera"]][1].append(doors[door])
    out, n = {}, {}
    for c, (img, plan) in pts.items():
        h, _ = cv2.findHomography(np.float32(img), np.float32(plan), 0)
        out[c], n[c] = h, len(img)
    return out, n


def load_boxes(path):
    """{影格(1-based): {人員: 框}};只留正的編號(負號是 distractor)。"""
    with open(path) as f:
        d = json.load(f)
    frames = [{} for _ in range(max(int(k) for k in d) + 1)]
    for k, persons in d.items():
        frames[int(k)] = {p["id"]: p["BboxP"] for p in persons if p["id"] > 0}
    return frames


def foot(box):
    return ((box[0] + box[2]) / 2.0, float(box[3]))


def project(m, pt):
    v = m @ np.array([pt[0], pt[1], 1.0])
    if abs(v[2]) < 1e-12:
        return np.array([np.inf, np.inf])
    return v[:2] / v[2]


def calibrate(ann_root, mats, pmap, walk_map):
    """推導集上估 σ、速度與落在可行走區外的比例。mats: {序列: {鏡頭: 3×3}}。"""
    rng = np.random.default_rng(SEED)
    pair_d, speeds, off = [], [], {c: [0, 0] for c in CAMS}
    n_pairs_total = 0
    for seq in DERIVE:
        boxes = {}
        for c in CAMS:
            hit = glob.glob(f"{ann_root}/{seq}/{c}_*.json")
            boxes[c] = load_boxes(hit[0]) if hit else []
        n = max(len(f) for f in boxes.values())
        pairs = []
        for t in range(1, n, STEP):
            feet = {}
            for c in CAMS:
                if t >= len(boxes[c]):
                    continue
                for pid, b in boxes[c][t].items():
                    p = project(mats[seq][c], foot(b))
                    feet[(c, pid)] = p
                    off[c][1] += 1
                    off[c][0] += int(not (np.isfinite(p).all() and walk_map.on_walkable(p)))
                    # 走動者的 1 秒位移(平面圖單位/秒)
                    fr = boxes[c]
                    if t - MOVE_WIN >= 1 and t + MOVE_WIN < len(fr):
                        a, z = fr[t - MOVE_WIN].get(pid), fr[t + MOVE_WIN].get(pid)
                        if a is not None and z is not None:
                            fa, fz = foot(a), foot(z)
                            if math.hypot(fz[0] - fa[0], fz[1] - fa[1]) >= MOVE_RATIO * (b[3] - b[1]):
                                pa, pz = project(mats[seq][c], fa), project(mats[seq][c], fz)
                                if np.isfinite(pa).all() and np.isfinite(pz).all():
                                    speeds.append(float(np.linalg.norm(pz - pa)))
            for i, a in enumerate(CAMS):
                for b_ in CAMS[i + 1:]:
                    for (c, pid), p in feet.items():
                        if c == a and (b_, pid) in feet:
                            pairs.append((p, feet[(b_, pid)]))
        n_pairs_total += len(pairs)
        pick = rng.choice(len(pairs), min(MAX_PAIRS, len(pairs)), replace=False)
        for k in pick:
            d = pmap.distance(*pairs[k])
            if math.isfinite(d):
                pair_d.append(d)
    d = np.array(pair_d)
    sigma_pair = float(np.median(d)) / MEDIAN_TO_SIGMA
    return dict(
        n_pairs_total=n_pairs_total, n_pairs_used=len(d),
        pair_dist_median=round(float(np.median(d)), 2),
        pair_dist_p90=round(float(np.percentile(d, 90)), 2),
        sigma_pair=round(sigma_pair, 3),
        sigma_single=round(sigma_pair / math.sqrt(2.0), 3),
        n_walk_samples=len(speeds),
        speed=round(float(np.median(speeds)), 3),
        speed_p95=round(float(np.percentile(speeds, 95)), 3),
        off_walkable_ratio={c: round(v[0] / v[1], 3) if v[1] else None for c, v in off.items()})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--handover", default=str(
        ROOT.parent / "CHIRLA_輔助線索與2D平面和網格程式" / "CHIRLA_L3_handover"))
    ap.add_argument("--ann-root", default="D:/yizhen/CHIRLA/CHIRLA_data/CHIRLA/annotations")
    ap.add_argument("--base", default="configs/fix_grid/base_overlapfix.yaml")
    ap.add_argument("--out-dir", default="configs/plan_grid")
    ap.add_argument("--report", default="results/m5_plan/calib_report.json")
    args = ap.parse_args()

    data = Path(args.handover) / "level_search" / "data"
    landmarks = json.loads((data / "floorplan" / "landmarks.json").read_text(encoding="utf-8"))
    refined = json.loads((data / "floorplan" / "camera_to_plan_refined.json")
                         .read_text(encoding="utf-8"))[REFINED_KEY]
    align = json.loads((data / "sync" / "camera_align.json").read_text(encoding="utf-8"))
    doors = landmarks["plan_doors"]

    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    walk = walkable_mask(data / "floorplan" / "site_floorplan.png", doors)
    npz = out_dir / "plan_map.npz"
    np.savez_compressed(npz, walk=walk, cell=CELL)
    area = float(walk.sum()) * CELL * CELL

    rooms_cfg = {k: list(v) for k, v in ROOMS.items()}
    doors_cfg = {k: {"xy": [float(v) for v in doors[k]], "rooms": list(DOOR_LINKS[k])}
                 for k in DOOR_LINKS if k in doors}
    npz_rel = npz.relative_to(ROOT).as_posix()
    fp_rooms = {"enabled": True, "distance": "rooms", "rooms": rooms_cfg, "doors": doors_cfg}
    fp_grid = {"enabled": True, "distance": "grid", "grid": npz_rel}
    grid_map = PlanMap("grid", walk=walk, cell=CELL)

    lm_h, lm_n = landmark_homographies(landmarks, data / "floorplan" / "door_zones.csv")

    def seq_mats(source):
        """{序列: {鏡頭: 平面圖投影 × 該序列的畫面對齊}}。"""
        out = {}
        for s in SEQS:
            out[s] = {}
            for c in CAMS:
                h = lm_h[c] if source == "landmarks" else np.array(refined[s][c], dtype=float)
                m = h @ np.array(align[s][c], dtype=float)
                out[s][c] = m / m[2, 2]
        return out

    mats = {"landmarks": seq_mats("landmarks"), "refined": seq_mats("refined")}
    cal = {"landmarks": calibrate(args.ann_root, mats["landmarks"],
                                  PlanMap("rooms", rooms_cfg, doors_cfg), grid_map),
           "refined": calibrate(args.ann_root, mats["refined"], grid_map, grid_map)}
    thr = math.log(5.0)
    for k, c in cal.items():
        c["area"] = area
        c["evidence_ceiling_nats"] = round(
            math.log(area / (2 * math.pi * c["sigma_pair"] ** 2)), 3)
        c["v2_pass"] = c["evidence_ceiling_nats"] > thr

    base = yaml.safe_load((ROOT / args.base).read_text(encoding="utf-8"))["camera_topology"]
    cells = {"P1": ("landmarks", fp_rooms, False), "P2": ("landmarks", fp_rooms, True),
             "G2": ("refined", fp_grid, True)}
    for name, (source, fp, transit) in cells.items():
        c = cal[source]
        (out_dir / name).mkdir(exist_ok=True)
        for s in SEQS:
            cfg = copy.deepcopy(base)
            cfg["fusion"]["ground_plane"] = {"enabled": True, "sigma_m": c["sigma_single"],
                                             "area_m2": area, "clip": 8.0, "speed": c["speed"]}
            cfg["fusion"]["floorplan"] = fp
            cfg["fusion"]["plan_transit"] = {"enabled": transit}
            for cam in CAMS:
                cfg["cameras"][cam]["homography"] = {
                    "matrix": [[float(v) for v in row] for row in mats[source][s][cam]]}
            head = (f"# {name} · {s} —— 由 scripts/chirla_build_plan_topology.py 產生,請勿手改\n"
                    f"# 預先登記:docs/M5_平面圖幾何接入_預先登記_20261013.md\n"
                    f"# 基底 {args.base};投影來源 {source};單位為示意圖像素(欄位名沿用 _m)\n")
            (out_dir / name / f"{s}.yaml").write_text(
                head + yaml.safe_dump({"camera_topology": cfg}, allow_unicode=True,
                                      sort_keys=False), encoding="utf-8")

    report = {"unit": "plan_px", "cell": CELL, "walkable_cells": int(walk.sum()),
              "area": area, "llr_threshold": round(thr, 4), "derive_seqs": DERIVE,
              "landmark_points_per_camera": lm_n, "calibration": cal,
              "cells": {k: {"source": v[0], "distance": v[1]["distance"],
                            "plan_transit": v[2]} for k, v in cells.items()}}
    rp = ROOT / args.report
    rp.parent.mkdir(parents=True, exist_ok=True)
    rp.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
