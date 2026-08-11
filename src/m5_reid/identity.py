"""M5 廚師身份管理:把 M4 的本地 track_id 綁定到全域 chef_id。

由 M4 的 new_track 事件觸發:抽外觀特徵 → 用餘弦相似度比對「目前活躍(其他鏡頭仍在)+
最近消失(recently_disappeared)」的廚師 → 相似度夠高就沿用該 chef_id,否則開新 chef_id。
track 消失 → 該 chef 若無其他鏡頭仍看得到,移入 recently_disappeared,保留 TTL 幀等再現。

本檔是我方自有邏輯(授權乾淨);外觀特徵抽取抽象在 embedder(可插拔),見 embedder.py。
"""
from dataclasses import dataclass, field

import numpy as np

from m5_reid.embedder import l2norm


def cosine(a, b):
    return float(np.dot(a, b))          # a、b 皆已 L2-normalized


@dataclass
class ChefIdentity:
    chef_id: int
    embedding: np.ndarray               # 代表特徵(EMA 更新)
    track_ids: list                     # 目前綁定、仍活躍的 track_id(可跨鏡頭多個)
    state: str                          # "active" | "gone"
    first_seen: int
    last_seen: int


@dataclass
class MatchResult:
    track_id: int
    chef_id: int
    matched: bool                       # True=綁到既有 chef;False=開新 chef
    similarity: float
    frame_id: int


class IdentityManager:
    def __init__(self, similarity_threshold=0.5, recently_disappeared_ttl=150,
                 embedding_ema=0.5, embedder=None):
        self.thr = similarity_threshold
        self.ttl = recently_disappeared_ttl
        self.ema = embedding_ema
        self.embedder = embedder
        self.active = {}                # chef_id -> ChefIdentity
        self.gone = {}                  # chef_id -> ChefIdentity(最近消失,TTL 內)
        self.track_to_chef = {}         # track_id -> chef_id
        self._next = 1

    @classmethod
    def from_config(cls, cfg, embedder=None):
        return cls(similarity_threshold=cfg.get("similarity_threshold", 0.5),
                   recently_disappeared_ttl=cfg.get("recently_disappeared_ttl", 150),
                   embedding_ema=cfg.get("embedding_ema", 0.5), embedder=embedder)

    def tick(self, frame_id):
        """讓 recently_disappeared 逾 TTL 的廚師永久移除。回傳被移除的 chef_id。"""
        expired = [cid for cid, c in self.gone.items() if frame_id - c.last_seen > self.ttl]
        for cid in expired:
            self.gone.pop(cid, None)
        return expired

    def on_new_track(self, track_id, crop=None, frame_id=0, embedding=None):
        """M4 出現新 track 時呼叫。回傳 MatchResult(含指派的 chef_id)。"""
        self.tick(frame_id)
        emb = l2norm(embedding if embedding is not None else self.embedder.extract(crop))

        best_id, best_sim = None, -1.0
        for cid, chef in list(self.active.items()) + list(self.gone.items()):
            s = cosine(emb, chef.embedding)
            if s > best_sim:
                best_sim, best_id = s, cid

        if best_id is not None and best_sim >= self.thr:      # 綁到既有廚師
            chef = self.gone.pop(best_id, None) or self.active[best_id]
            chef.state = "active"
            self.active[best_id] = chef
            if track_id not in chef.track_ids:
                chef.track_ids.append(track_id)
            chef.last_seen = frame_id
            chef.embedding = l2norm(self.ema * chef.embedding + (1 - self.ema) * emb)
            self.track_to_chef[track_id] = best_id
            return MatchResult(track_id, best_id, True, best_sim, frame_id)

        cid = self._next                                       # 開新廚師
        self._next += 1
        self.active[cid] = ChefIdentity(cid, emb, [track_id], "active", frame_id, frame_id)
        self.track_to_chef[track_id] = cid
        return MatchResult(track_id, cid, False, best_sim, frame_id)

    def on_track_lost(self, track_id, frame_id):
        """M4 track 消失時呼叫。該 chef 若無其他鏡頭仍看得到 → 移入 recently_disappeared。"""
        cid = self.track_to_chef.pop(track_id, None)
        if cid is None:
            return None
        chef = self.active.get(cid)
        if chef is None:
            return None
        if track_id in chef.track_ids:
            chef.track_ids.remove(track_id)
        chef.last_seen = frame_id
        if not chef.track_ids:                                 # 無任何鏡頭還看得到
            chef.state = "gone"
            self.active.pop(cid, None)
            self.gone[cid] = chef
        return cid

    def chef_of(self, track_id):
        return self.track_to_chef.get(track_id)

    def stats(self):
        return {"active": len(self.active), "gone": len(self.gone),
                "total_chefs": self._next - 1}
