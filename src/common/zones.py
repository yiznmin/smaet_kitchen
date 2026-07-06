"""
Zone 設定的讀寫與繪製(全專案共用)。

zones.json 格式:
{
  "source": "data/epfl/Boutput0.mp4", "frame": 0, "width": 1280, "height": 720,
  "zones": [
    {"name": "floor",      "points": [[x,y], ...]},
    {"name": "zoneA_raw",  "points": [[x,y], ...]}
  ]
}

固定鏡頭裝機時標一次、存檔,開機載入。需要區域的模組共用:
  - 靜態線地板 ROI(floor)
  - M6 事件判斷各 zone(生食區/熟食區/洗手台/垃圾桶區…)
"""
import json
from pathlib import Path

import cv2
import numpy as np

_COLORS = [(0, 255, 255), (0, 255, 0), (255, 128, 0), (255, 0, 255),
           (0, 128, 255), (128, 255, 0), (0, 0, 255), (255, 255, 0)]


def load_zones(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


_ROOT = Path(__file__).resolve().parents[2]


def get_zone(path, name):
    """取某個命名 zone 的多邊形點(找不到回 None)。"""
    for z in load_zones(path).get("zones", []):
        if z.get("name") == name:
            return z["points"]
    return None


def resolve_floor(cfg_static):
    """依 m2_static_line 設定解析地板 ROI 多邊形。
    優先 floor_zone_file + floor_zone_name(標註工具產出);
    找不到檔/zone 時退回 floor_zone 內嵌點;再不然 None(全畫面)。"""
    f = cfg_static.get("floor_zone_file")
    name = cfg_static.get("floor_zone_name", "floor")
    if f:
        p = Path(f)
        if not p.is_absolute():
            p = _ROOT / f
        if p.exists():
            pts = get_zone(p, name)
            if pts:
                return pts
    return cfg_static.get("floor_zone")


def save_zones(path, zones, source=None, width=None, height=None, frame=0):
    data = {"source": str(source) if source is not None else None,
            "frame": frame, "width": width, "height": height, "zones": zones}
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def draw_zones(frame, zones, current=None):
    """把已完成 zones(半透明填色 + 名稱)與進行中多邊形(白點白線)畫到畫面上。"""
    vis = frame.copy()
    for i, z in enumerate(zones):
        pts = np.array(z["points"], np.int32)
        if len(pts) < 2:
            continue
        color = _COLORS[i % len(_COLORS)]
        overlay = vis.copy()
        cv2.fillPoly(overlay, [pts], color)
        vis = cv2.addWeighted(overlay, 0.25, vis, 0.75, 0)
        cv2.polylines(vis, [pts], True, color, 2)
        cx, cy = pts.mean(axis=0).astype(int)
        cv2.putText(vis, z.get("name", ""), (int(cx) - 20, int(cy)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    if current:
        for p in current:
            cv2.circle(vis, (int(p[0]), int(p[1])), 4, (255, 255, 255), -1)
        if len(current) >= 2:
            cv2.polylines(vis, [np.array(current, np.int32)], False, (255, 255, 255), 1)
    return vis
