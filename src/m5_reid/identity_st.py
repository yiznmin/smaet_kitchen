"""M5 v2 空間+時序身份管理:時空為主、外觀(DINOv2 可商用)為輔破平手。

擴充 IdentityManager:on_new_track/on_track_lost 帶入 camera_id + bbox + 時間,
用 CameraTopology 的「轉場時間門」先把候選縮到少數,再用外觀 cosine 破平手。
→ 即使外觀弱(DINOv2 Rank-1 僅 11%),時空門也能綁對;且全條可商用。

track_id 各鏡頭不全域唯一 → 內部一律用 (camera_id, track_id) 當 key。
"""
from m5_reid.embedder import l2norm
from m5_reid.identity import ChefIdentity, IdentityManager, MatchResult, cosine
from m5_reid.spatiotemporal import CameraTopology


class SpatioTemporalIdentityManager(IdentityManager):
    def __init__(self, topology, embedder=None, fps=30.0,
                 recently_disappeared_ttl=600, embedding_ema=0.5):
        super().__init__(similarity_threshold=0.0, recently_disappeared_ttl=recently_disappeared_ttl,
                         embedding_ema=embedding_ema, embedder=embedder)
        self.topo = topology
        self.fps = fps
        self.f = topology.fusion
        self._exit = {}         # chef_id -> (camera_id, t_sec) 離場資訊
        self._cam = {}          # chef_id -> (camera_id, t_sec) 目前所在(active)

    @classmethod
    def from_config(cls, topo_cfg, embedder=None, fps=30.0):
        return cls(CameraTopology.from_config(topo_cfg), embedder=embedder, fps=fps)

    def _t(self, frame_id, t_sec):
        return float(t_sec) if t_sec is not None else frame_id / self.fps

    def on_new_track(self, track_id, camera_id, frame_id=0, bbox=None,
                     embedding=None, crop=None, t_sec=None):
        self.tick(frame_id)
        t = self._t(frame_id, t_sec)
        emb = l2norm(embedding if embedding is not None else self.embedder.extract(crop))
        w_st, w_app = self.f["w_st"], self.f["w_app"]
        thr = self.f["combined_threshold"]

        best_id, best_score, best_app = None, -1.0, 0.0

        # (a) recently_disappeared:時空轉場門(不重疊跨時的主路徑)
        for cid, chef in self.gone.items():
            ex = self._exit.get(cid)
            if ex is None:
                continue
            passed, sp = self.topo.transition_gate(ex[0], ex[1], camera_id, t)
            if not passed:
                continue
            app = cosine(emb, chef.embedding)
            score = w_st * sp + w_app * app
            if score > best_score:
                best_id, best_score, best_app = cid, score, app

        # (b) active + 重疊相機 + 同時刻:幾何關聯(st_prob=1)
        for cid, chef in self.active.items():
            ct = self._cam.get(cid)
            if ct is None or ct[0] == camera_id:
                continue
            if self.topo.is_overlapping(ct[0], camera_id) and abs(t - ct[1]) <= self.f["overlap_window_s"]:
                app = cosine(emb, chef.embedding)
                score = w_st * 1.0 + w_app * app
                if score > best_score:
                    best_id, best_score, best_app = cid, score, app

        gk = (camera_id, track_id)
        if best_id is not None and best_score >= thr:          # 綁到既有廚師
            chef = self.gone.pop(best_id, None) or self.active[best_id]
            chef.state = "active"
            self.active[best_id] = chef
            if gk not in chef.track_ids:
                chef.track_ids.append(gk)
            chef.last_seen = frame_id
            chef.embedding = l2norm(self.ema * chef.embedding + (1 - self.ema) * emb)
            self.track_to_chef[gk] = best_id
            self._cam[best_id] = (camera_id, t)
            return MatchResult(track_id, best_id, True, round(best_score, 4), frame_id)

        cid = self._next                                        # 開新廚師
        self._next += 1
        self.active[cid] = ChefIdentity(cid, emb, [gk], "active", frame_id, frame_id)
        self.track_to_chef[gk] = cid
        self._cam[cid] = (camera_id, t)
        return MatchResult(track_id, cid, False, round(max(best_score, 0.0), 4), frame_id)

    def on_track_lost(self, track_id, camera_id, frame_id, bbox=None, t_sec=None):
        t = self._t(frame_id, t_sec)
        gk = (camera_id, track_id)
        cid = self.track_to_chef.pop(gk, None)
        if cid is None:
            return None
        chef = self.active.get(cid)
        if chef is None:
            return None
        if gk in chef.track_ids:
            chef.track_ids.remove(gk)
        chef.last_seen = frame_id
        if not chef.track_ids:                                  # 無任何鏡頭還看得到 → 離場
            chef.state = "gone"
            self.active.pop(cid, None)
            self.gone[cid] = chef
            self._exit[cid] = (camera_id, t)
            self._cam.pop(cid, None)
        return cid
