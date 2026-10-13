"""場地平面圖上的距離:牆是障礙物,兩點之間不能直接量直線。

為什麼需要它:`GroundPlaneLR` 原本拿兩個世界座標算直線距離。同一個房間裡那沒問題,
但隔著一道牆的兩台鏡頭(CHIRLA 的 camera_2 與 camera_7 在示意圖上只差 14 像素,
卻分屬 L2 與 L1 兩個房間)會被判成「幾乎站在同一點」。

兩種距離(2026-10-13,作法移植自 CHIRLA 交接包的 floorplan_cues / floorplan_grid_cues):
  rooms  平面點歸入房間,經由出入口的連通圖求最短路徑;同房間為直線距離
  grid   可行走網格上的 8 鄰接最短路徑(網格由建置腳本預先算好存成 npz,
         執行期不解析圖片 —— 部署端不該相依於一張示意圖的顏色門檻)

⚠ 單位是**平面圖自己的單位**(CHIRLA 為示意圖像素,且示意圖不按比例)。
  σ、速度、面積必須用同一個單位估,不可與公尺制的預設值混用。
"""
import heapq
import math

import numpy as np


class PlanMap:
    def __init__(self, kind="rooms", rooms=None, doors=None, walk=None, cell=1.0):
        if kind not in ("rooms", "grid"):
            raise ValueError(f"floorplan.distance 只能是 rooms 或 grid,收到 {kind!r}")
        self.kind = kind
        self.rooms = {k: tuple(float(v) for v in r) for k, r in (rooms or {}).items()}
        # doors: {名稱: {"xy": [x, y], "rooms": [房間A, 房間B]}}
        self.doors = {k: (np.asarray(d["xy"], dtype=float), frozenset(d["rooms"]))
                      for k, d in (doors or {}).items()}
        self.cell = float(cell)
        self._rows = {}
        if kind == "grid":
            if walk is None:
                raise ValueError("floorplan.distance = grid 需要可行走網格(floorplan.grid)")
            from scipy import ndimage
            self.walk = np.asarray(walk, dtype=bool)
            self.nx = self.walk.shape[1]
            self._near = ndimage.distance_transform_edt(~self.walk, return_indices=True)[1]
            self._graph = self._grid_graph()
        elif not self.rooms:
            raise ValueError("floorplan.distance = rooms 需要 floorplan.rooms")

    @classmethod
    def from_config(cls, cfg):
        kind = cfg.get("distance", "rooms")
        walk, cell = None, 1.0
        if kind == "grid":
            d = np.load(cfg["grid"])
            walk, cell = d["walk"], float(d["cell"])
        return cls(kind, cfg.get("rooms"), cfg.get("doors"), walk, cell)

    # ── rooms ────────────────────────────────────────────────────────
    def room_of(self, p):
        """點所在的房間;落在所有房間外時取最近的(投影誤差會把牆邊的人丟到牆外)。"""
        best = None
        for name, (x0, y0, x1, y1) in self.rooms.items():
            if x0 <= p[0] <= x1 and y0 <= p[1] <= y1:
                return name
            d = max(x0 - p[0], 0, p[0] - x1) + max(y0 - p[1], 0, p[1] - y1)
            if best is None or d < best[0]:
                best = (d, name)
        return best[1]

    def _room_path(self, a, b):
        ra, rb = self.room_of(a), self.room_of(b)
        if ra == rb:
            return float(np.linalg.norm(a - b))
        nodes = {"A": (a, frozenset((ra,))), "B": (b, frozenset((rb,)))}
        nodes.update(self.doors)
        dist, heap = {"A": 0.0}, [(0.0, "A")]
        while heap:
            d, u = heapq.heappop(heap)
            if u == "B":
                return d
            if d > dist.get(u, math.inf):
                continue
            for v in nodes:
                if v != u and nodes[u][1] & nodes[v][1]:
                    nd = d + float(np.linalg.norm(nodes[u][0] - nodes[v][0]))
                    if nd < dist.get(v, math.inf):
                        dist[v] = nd
                        heapq.heappush(heap, (nd, v))
        return float(np.linalg.norm(a - b))     # 連通圖走不到 → 退回直線

    # ── grid ─────────────────────────────────────────────────────────
    def _grid_graph(self):
        from scipy.sparse import coo_matrix
        walk = self.walk
        ny, nx = walk.shape
        idx = np.arange(ny * nx).reshape(ny, nx)
        rows, cols, w = [], [], []
        for dy, dx in ((0, 1), (1, 0), (1, 1), (1, -1)):
            sa = (slice(0, ny - dy), slice(max(0, -dx), nx - max(0, dx)))
            sb = (slice(dy, ny), slice(max(0, dx), nx - max(0, -dx)))
            ok = walk[sa] & walk[sb]
            rows += [idx[sa][ok], idx[sb][ok]]
            cols += [idx[sb][ok], idx[sa][ok]]
            w.append(np.full(int(ok.sum()) * 2, self.cell * math.hypot(dy, dx)))
        n = ny * nx
        return coo_matrix((np.concatenate(w), (np.concatenate(rows), np.concatenate(cols))),
                          shape=(n, n)).tocsr()

    def snap(self, p):
        """最近的可行走格 (列, 欄)。"""
        ny, nx = self.walk.shape
        cy = int(min(max(p[1] // self.cell, 0), ny - 1))
        cx = int(min(max(p[0] // self.cell, 0), nx - 1))
        return int(self._near[0, cy, cx]), int(self._near[1, cy, cx])

    def on_walkable(self, p):
        """點是否落在可行走格內(校正報告用)。"""
        ny, nx = self.walk.shape
        cy, cx = int(p[1] // self.cell), int(p[0] // self.cell)
        return 0 <= cy < ny and 0 <= cx < nx and bool(self.walk[cy, cx])

    def _grid_path(self, a, b):
        from scipy.sparse.csgraph import dijkstra
        (ay, ax), (by, bx) = self.snap(a), self.snap(b)
        u, v = ay * self.nx + ax, by * self.nx + bx
        if u not in self._rows:
            # ⚠ 快取以「來源格」為鍵,上界 = 可行走格數(CHIRLA 約數千格),不會無限成長。
            self._rows[u] = dijkstra(self._graph, indices=u)
        d = self._rows[u][v]
        return float(d) if np.isfinite(d) else float(np.linalg.norm(a - b))

    def distance(self, xy_a, xy_b):
        a, b = np.asarray(xy_a, dtype=float), np.asarray(xy_b, dtype=float)
        # 投影在消失線附近會爆成 inf/nan → 當成「算不出來」,由呼叫端的 clip 收斂。
        if not (np.isfinite(a).all() and np.isfinite(b).all()):
            return math.inf
        return self._grid_path(a, b) if self.kind == "grid" else self._room_path(a, b)

    def describe(self):
        if self.kind == "grid":
            return (f"PlanMap(grid, {int(self.walk.sum())} 個可行走格, "
                    f"每格 {self.cell:g} 單位)")
        return f"PlanMap(rooms, {len(self.rooms)} 個房間, {len(self.doors)} 個出入口)"
