"""M5 v2 空間+時序(Camera Link Model)核心:相機拓撲 + 轉場時間門 + zone 判定。

跨鏡頭關聯「同一人」的四線索,外觀只是其一:
  相機拓撲(哪台接哪台)、轉場時間(高斯窗)、移動方向(有向連結)、外觀相似度(輔助)。
物理約束:同一人不可能同時出現在兩個不重疊鏡頭;不能比離開更早抵達。

純函式/純邏輯,可單測;point_in_zone 用 cv2(zones.py 無此 helper)。
"""
import numpy as np


def point_in_zone(pt, polygon):
    """點是否在多邊形內(含邊界)。polygon = [[x,y],...]。"""
    import cv2
    poly = np.asarray(polygon, dtype=np.int32)
    return cv2.pointPolygonTest(poly, (float(pt[0]), float(pt[1])), False) >= 0


def foot_point(bbox):
    """人的「腳點」(框底中點)當地面位置,判所在 zone 較準。"""
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, float(y2))


def which_zone(bbox, zones):
    """回傳 bbox 腳點所在的 zone 名稱(zones = [{name, points}]),無則 None。"""
    pt = foot_point(bbox)
    for z in zones:
        if point_in_zone(pt, z["points"]):
            return z["name"]
    return None


def st_prob(dt, mean, std):
    """轉場時間高斯,normalize 到峰值 1(dt=mean 時為 1)。"""
    if std <= 0:
        return 1.0 if dt == mean else 0.0
    return float(np.exp(-0.5 * ((dt - mean) / std) ** 2))


_DEFAULT_FUSION = {"w_st": 0.7, "w_app": 0.3, "k_sigma": 2.0,
                   "combined_threshold": 0.35, "overlap_window_s": 0.5}


class CameraTopology:
    def __init__(self, links, overlapping, fusion=None):
        self.links = {(l["from"], l["to"]): (float(l["mean_s"]), float(l["std_s"])) for l in links}
        self.overlapping = set(frozenset(p) for p in overlapping)
        self.fusion = {**_DEFAULT_FUSION, **(fusion or {})}

    @classmethod
    def from_config(cls, cfg):
        return cls(cfg.get("links", []), cfg.get("overlapping", []), cfg.get("fusion"))

    def is_overlapping(self, a, b):
        return frozenset((a, b)) in self.overlapping

    def transition_gate(self, cam_from, t_exit, cam_to, t_enter):
        """回傳 (是否通過, st_prob)。需:有向連結存在、Δt>0、Δt 在時間窗 μ±kσ 內。"""
        if cam_from == cam_to:
            return (False, 0.0)
        key = (cam_from, cam_to)
        if key not in self.links:
            return (False, 0.0)
        mean, std = self.links[key]
        dt = t_enter - t_exit
        if dt <= 0:                                       # 不能比離開更早抵達
            return (False, 0.0)
        if abs(dt - mean) > self.fusion["k_sigma"] * std:  # 超出時間窗
            return (False, 0.0)
        return (True, st_prob(dt, mean, std))
