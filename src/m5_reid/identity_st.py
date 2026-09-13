"""M5 v2 空間+時序身份管理:時空為主、外觀(DINOv2 可商用)為輔破平手。

擴充 IdentityManager:on_new_track/on_track_lost 帶入 camera_id + bbox + 時間,
用 CameraTopology 的「轉場時間門」先把候選縮到少數,再用外觀 cosine 破平手。
→ 即使外觀弱(DINOv2 Rank-1 僅 11%),時空門也能綁對;且全條可商用。

track_id 各鏡頭不全域唯一 → 內部一律用 (camera_id, track_id) 當 key。
"""
from collections import Counter, deque
import math

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
        self._exit = {}         # chef_id -> (camera_id, t_sec, bbox) 離場資訊
        self._cam = {}          # chef_id -> (camera_id, t_sec) 目前所在(active)
        self._pending_exit = {} # (cam, track) -> (cam, t_sec, frame_id, bbox)
        self._world = {}        # chef_id -> (世界座標, 時間)
        self._vel = {}          # chef_id -> 最近觀測到的速度向量(m/s)
        # F4 CrossViewLR 用:(chef_id, camera) -> (bbox, t)。
        # ⚠ 鍵含 camera 是必要的 —— 一位 chef 可能同時被多台重疊鏡頭看著,
        #   而我們要比的是「他在**那一台**上的腳點」。
        # ⚠ 記憶體有界性:上界是 活躍 chef 數 × 鏡頭數。track 移除與 _forget
        #   都會清,見 on_track_removed 與 _forget。
        self._bbox = {}
        # 診斷:每次綁定決策有幾個候選通過物理可能性檢查。索引 5 代表「5 個以上」。
        # 這是本架構最關鍵的觀測量 —— 若幾乎總是 1,外觀品質對結果沒有影響。
        self._cand_hist = [0] * 6
        self.last_candidates = 0
        # F1 margin test 的診斷。_n_margin_blocked 是「本來會綁、被 margin 擋下改開新
        # 身份」的次數 —— 它就是這個修法把多少誤併換成了碎裂,必須看得見。
        self._n_margin_blocked = 0
        self.last_margin = None
        # F2 同鏡頭互斥的驗收量:開啟後這個數字必須是 0。
        self._n_same_cam_conflicts = 0
        # P1 累積投票的狀態與診斷。全部是計數器或有界 deque(maxlen=window),
        # 所以不影響「活躍清單記憶體不無限成長」那條規格驗收。
        self._votes = {}                 # (cam, track) -> deque[chef_id]
        self._vote_calls = {}            # (cam, track) -> 收到幾次 update
        self._n_revotes = 0              # 實際投出的票數
        self._n_revote_switch = 0        # 因投票而改判的次數
        self._n_revote_no_emb = 0        # 到了取樣點卻沒有 embedding → 沒投成
        # P2 全域指派用的緩衝。camera -> [待指派的 track]。resolve_frame() 會清空。
        # ⚠ runner 沒呼叫 resolve_frame 的話票投不出去 —— revote_pending 讓它看得見。
        self._pending_vote = {}
        self._n_resolved = 0

    @classmethod
    def from_config(cls, topo_cfg, embedder=None, fps=30.0):
        return cls(CameraTopology.from_config(topo_cfg), embedder=embedder, fps=fps)

    def _key(self, track_id, camera_id=None):
        """track_id 各鏡頭不全域唯一 → 一律用 (camera_id, track_id) 當鍵。"""
        return (camera_id, track_id)

    def _forget(self, chef_id):
        """chef 逾 TTL 被永久移除時,連帶清掉時空狀態。

        沒有這個的話 _exit 會隨累計人次單調成長(tick() 只清 self.gone),
        直接違反 spec「連續運作下活躍清單記憶體不無限成長」。
        """
        self._exit.pop(chef_id, None)
        self._cam.pop(chef_id, None)
        self._world.pop(chef_id, None)
        self._vel.pop(chef_id, None)
        # F4:_bbox 的鍵是 (chef, camera),要把該 chef 的所有鏡頭一起清掉,
        # 否則它會隨累計人次單調成長 —— 與上面那段註解防的是同一件事。
        for k in [k for k in self._bbox if k[0] == chef_id]:
            del self._bbox[k]

    def resident_stats(self):
        s = super().resident_stats()
        s.update({"_exit": len(self._exit), "_cam": len(self._cam),
                  "_pending_exit": len(self._pending_exit), "_world": len(self._world),
                  "_vel": len(self._vel),
                  # F1/F2 的診斷。都是計數器不是集合,所以不影響記憶體有界性驗收。
                  "margin_blocked": self._n_margin_blocked,
                  "same_cam_conflicts": self._n_same_cam_conflicts,
                  # P1:_votes 的每個 deque 有 maxlen,_vote_calls 隨活躍 track 數
                  # 成長並在 track 移除時清掉 —— 兩者都不違反記憶體有界性。
                  "revotes": self._n_revotes,
                  "revote_switch": self._n_revote_switch,
                  "revote_no_emb": self._n_revote_no_emb,
                  "_votes": len(self._votes),
                  # P2:pending 應該在每輪 resolve_frame 後歸零。持續 > 0 表示
                  # runner 沒呼叫 resolve_frame,票根本沒投出去。
                  "revote_pending": sum(len(v) for v in self._pending_vote.values()),
                  "revote_resolved": self._n_resolved})
        return s

    def _t(self, frame_id, t_sec, camera_id=None):
        """事件時間(秒),已套用該相機的時鐘偏移校正。

        所有跨鏡頭的 Δt 都由這裡產出,所以校正只需做在這一個點。
        未校正的話,鏡頭間 2 秒漂移在 σ=1.5s 下等於 1.33σ,足以把幾乎所有
        真實轉場推出時間窗 —— 見 analyze_gate_capacity.py §4。
        """
        raw = float(t_sec) if t_sec is not None else frame_id / self.fps
        return self.topo.corrected(camera_id, raw) if camera_id is not None else raw


    def _score_candidates(self, camera_id, t, emb, bbox, zone, world_xy, world_v,
                          exclude_key=None):
        """算出所有通過物理可能性檢查的候選 [(score, chef_id, app_cos)]。

        ⚠ **這段從 `on_new_track` 抽出來,一個字都沒改。** 抽出來的唯一理由是
          讓 P1 的重新投票走**完全相同**的評分邏輯 —— 兩條路若分歧,
          投票結果就不能和出生時的決策相比,整個累積投票失去意義。

        exclude_key:(camera, track) —— 計算某位 chef「此刻被哪些鏡頭看著」時
          要忽略的那一條 track。重新投票時必須把**被評分的這條 track 自己**
          排除,否則它會把自己算成「該 chef 正在這台鏡頭上」而自我佐證。
        """
        w_st, w_app = self.f["w_st"], self.f["w_app"]
        llr_mode = self.f["mode"] == "llr"
        cands = []          # [(score, chef_id, app_cos)] 所有通過物理可能性檢查的候選

        # (a) recently_disappeared:跨時轉場(不重疊鏡頭的主路徑)
        for cid, chef in self.gone.items():
            ex = self._exit.get(cid)
            if ex is None:
                continue
            app = None
            if llr_mode:
                ok, llr_t = self.topo.transit_llr(ex[0], ex[1], camera_id, t)
                if not ok:
                    continue
                app = cosine(emb, chef.embedding)
                score = llr_t + self.topo.app_lr.llr(app)
                # 位置證據只在**同一台鏡頭**適用:跨鏡頭的影像座標是不同座標系,
                # 沒有外參校正就不可比。這條專門用來收緊 M4 斷軌的重關聯。
                if ex[0] == camera_id and self.topo.pos_lr is not None:
                    score += self.topo.pos_lr.llr(ex[2] if len(ex) > 2 else None,
                                                  bbox, t - ex[1])
                # 方向證據:與 Δt 正交,所以能幫到逗留者(時間軸上救不了的那些)
                score += self.topo.direction_llr(ex[0], camera_id,
                                                 ex[3] if len(ex) > 3 else None, zone)
                # F5 出入口位置(2026-09-05):「你是從通往這裡的那個門出去的嗎?」
                # ⚠ 轉場路徑上除了 Δt 幾乎沒有別的證據,而實測顯示 3 條連結有 2 條
                #   在**任何 Δt** 都過不了門檻(pdf 峰值 0.198 < 需要的 0.24)。
                #   碎裂 54.42% 就是這麼來的 —— 不是證據弱,是不可能綁定。
                # ⚠ 只有**退場項**能區分候選(每位候選有自己的退場點);
                #   入場項對所有候選相同,只影響綁不綁。見 TransitPlaceLR 說明。
                tp = self.topo.transit_place(ex[0], camera_id)
                if tp is not None:
                    score += tp.llr(ex[2] if len(ex) > 2 else None, bbox)
            else:
                ok, sp = self.topo.transition_gate(ex[0], ex[1], camera_id, t)
                if not ok:
                    continue
                app = cosine(emb, chef.embedding)
                score = w_st * sp + w_app * app
            cands.append((score, cid, app))

        # (b) 該 chef 此刻正被某台「與本鏡頭重疊」的相機看著 → 幾何關聯
        #
        # ⚠ 判斷依據是 chef.track_ids(目前綁定中的所有 (鏡頭, track)),**不是**
        #   _cam 的時間戳。踩過的坑:原本用 `_cam` 並要求 |Δt| ≤ 0.5s,但 _cam 只在
        #   「綁定事件」發生時更新,而「這個人現在正在畫面裡」是一個**持續狀態**,
        #   不是事件。M4 每幀都會輸出當前 active 的 tracks,但 M5 只吃事件,
        #   所以它無從得知誰此刻在畫面上 —— 重疊路徑因此幾乎從不觸發。
        #   全景鏡頭方案第一次跑時碎裂率完全沒降,根因就是這個。
        overlap_pool = {}          # 重疊鏡頭 -> 此刻在該鏡頭裡的 chef 數
        for chef in self.active.values():
            for c, _t in chef.track_ids:
                if exclude_key is not None and (c, _t) == exclude_key:
                    continue
                if c != camera_id and self.topo.is_overlapping(c, camera_id):
                    overlap_pool[c] = overlap_pool.get(c, 0) + 1

        for cid, chef in self.active.items():
            # F2 同鏡頭互斥(2026-09-05,預設關閉):這位 chef 此刻**已經**在本鏡頭上
            # 綁著另一條 track → 他不可能同時又是這條新 track。
            # ⚠ track_ids 在 on_track_removed(:283)會被剪除,所以它確實代表
            #   「目前還看得到的」,不是歷史累積 —— 這個檢查才安全。
            own = [k for k in chef.track_ids if k != exclude_key]
            if (getattr(self.topo, "same_cam_exclusive", False)
                    and any(c == camera_id for c, _t in own)):
                continue
            cams = {c for c, _t in own
                    if c != camera_id and self.topo.is_overlapping(c, camera_id)}
            if not cams:
                continue
            # ⚠ 「重疊鏡頭裡有人」不等於「就是這一個人」。若該鏡頭此刻有 K 位廚師,
            #   這條證據只說得出「是那 K 個之中的一個」→ 證據量要除以 K。
            #   沒有這一項的話,多人同時在場時重疊路徑會無差別地給滿分,誤併爆增。
            #   真正要收緊它需要**跨鏡頭的地面平面校正(homography)**,才能比對
            #   「兩台鏡頭看到的是不是地面上同一個點」。本版尚未實作,故只做稀釋。
            app = cosine(emb, chef.embedding)
            if not llr_mode:
                cands.append((w_st * 1.0 + w_app * app, cid, app))
                continue
            if self.topo.ground_lr is not None and world_xy is not None:
                # 有地面校正:直接問「是不是站在同一個位置」,不必稀釋。
                # ⚠ 但要把「上次觀測到現在」的可能移動算進不確定度 —— 否則會拿
                #   舊位置跟新位置比,把對的綁定當成位置對不上而擋掉。
                g = self._world.get(cid)
                if g is None:
                    continue
                score = (self.topo.ground_lr.llr(g[0], world_xy, dt=t - g[1])
                         + self.topo.app_lr.llr(app))
                # 速度證據:位置相同時唯一還能分辨的線索
                if self.topo.vel_lr is not None:
                    score += self.topo.vel_lr.llr(self._vel.get(cid), world_v)
            else:
                # F4 CrossViewLR(2026-09-05):無度量標定,但用真值身份估出的
                # 逐對單應性,一樣能回答「這兩個觀測落在地面的同一點嗎」。
                # ⚠ 它**取代**下面那個常數,不是相加 —— 兩者回答同一個問題,
                #   相加會把同一份證據算兩次。
                cv = self.topo.cross_view(camera_id) if hasattr(self.topo, "cross_view") else None
                cv_llr = None
                if cv is not None:
                    # 挑「這位 chef 此刻被哪一台重疊鏡頭看著、而且我們有那一對的
                    # 單應性」。多台都可用時取證據最強的那台 —— 它是資訊最多的觀測,
                    # 不是「對我們最有利的」:負證據一樣會被選中並擋下綁定。
                    best_cv, best_abs = None, -1.0
                    for c in cams:
                        m = cv.get(c)
                        rec = self._bbox.get((cid, c))
                        if m is None or rec is None:
                            continue
                        v = m.llr(rec[0], bbox, dt=max(0.0, t - rec[1]))
                        if abs(v) > best_abs:
                            best_cv, best_abs = v, abs(v)
                    cv_llr = best_cv
                if cv_llr is not None:
                    score = cv_llr + self.topo.app_lr.llr(app)
                else:
                    # 無校正也無單應性:只知道「那台鏡頭裡有人」。若該鏡頭此刻有
                    # K 位廚師,這條證據只說得出「是 K 個之中的一個」→ 除以 K。
                    # ⚠ 2026-09-04 實測:這條在數學上**不可能拒絕**任何候選 ——
                    #   5.0 − log(k) + app < 1.609 需要 k > 29.7,而 CHIRLA 單台
                    #   最多 9 人。重疊路徑因此誤併 80.80%、正確率 16.49%(≈1/6)。
                    #   保留它只是為了讓沒有單應性的鏡頭對仍能運作,不是認可它。
                    k = max(min(overlap_pool.get(c, 1) for c in cams), 1)
                    score = self.f["overlap_llr"] - math.log(k) + self.topo.app_lr.llr(app)
            cands.append((score, cid, app))
        return cands

    def on_new_track(self, track_id, *, camera_id=None, frame_id=0, t_sec=None,
                     crop=None, embedding=None, bbox=None, zone=None, world_xy=None,
                     world_v=None):
        self.tick(frame_id)
        t = self._t(frame_id, t_sec, camera_id)
        emb = l2norm(embedding if embedding is not None else self.embedder.extract(crop))
        llr_mode = self.f["mode"] == "llr"
        thr = self.topo.llr_threshold if llr_mode else self.f["combined_threshold"]

        cands = self._score_candidates(camera_id, t, emb, bbox, zone, world_xy, world_v)

        self.last_candidates = len(cands)
        self._cand_hist[min(len(cands), 5)] += 1

        best_score, best_id, _ = max(cands, default=(float("-inf"), None, None))

        # F1 margin test(2026-09-05,預設關閉)。
        # 每位 chef 在 cands 裡最多出現一次(gone 與 active 互斥),所以「次佳」
        # 必然是**另一個人**,差距才有「區分兩個假設」的意義。
        #
        # ⚠ 只有一個候選時不套用 —— 那不是「分不出來」,是「沒有別人可混淆」。
        #   把它也擋掉會把單人場景的正確綁定全部打成碎裂。
        margin_nats = getattr(self.topo, "margin_nats", None)
        self.last_margin = None
        if margin_nats is not None and len(cands) >= 2:
            second = sorted((c[0] for c in cands), reverse=True)[1]
            self.last_margin = best_score - second
            if self.last_margin < margin_nats:
                # 分不出來 → 依 5:1 成本比,寧可碎裂也不要誤併。
                self._n_margin_blocked += 1
                best_id = None

        gk = self._key(track_id, camera_id)
        if best_id is not None and best_score >= thr:          # 綁到既有廚師
            chef = self.gone.pop(best_id, None) or self.active[best_id]
            # F2 的驗收量。**不論修法開關都計數** —— 關閉時量到的是基線的問題規模,
            # 開啟後必須是 0。一個人不可能同時是同一台鏡頭上的兩條 track。
            if any(c == camera_id for c, _t in chef.track_ids):
                self._n_same_cam_conflicts += 1
            chef.state = "active"
            self.active[best_id] = chef
            if gk not in chef.track_ids:
                chef.track_ids.append(gk)
            chef.last_seen = frame_id
            chef.embedding = l2norm(self.ema * chef.embedding + (1 - self.ema) * emb)
            self.track_to_chef[gk] = best_id
            self._cam[best_id] = (camera_id, t)
            if bbox is not None:                       # F4:見 on_track_update
                self._bbox[(best_id, camera_id)] = (tuple(bbox[:4]), t)
            if world_xy is not None:
                self._world[best_id] = (world_xy, t)
            if world_v is not None:
                self._vel[best_id] = world_v
            return MatchResult(track_id, best_id, True, round(best_score, 4), frame_id)

        cid = self._next                                        # 開新廚師
        self._next += 1
        self.active[cid] = ChefIdentity(cid, emb, [gk], "active", frame_id, frame_id)
        self.track_to_chef[gk] = cid
        self._cam[cid] = (camera_id, t)
        if bbox is not None:                           # F4:見 on_track_update
            self._bbox[(cid, camera_id)] = (tuple(bbox[:4]), t)
        if world_xy is not None:
            self._world[cid] = (world_xy, t)
        if world_v is not None:
            self._vel[cid] = world_v
        # 無候選時 best_score 是 -inf;回 0.0 表示「沒有任何證據」而非「證據為負」
        shown = 0.0 if best_score == float("-inf") else round(best_score, 4)
        return MatchResult(track_id, cid, False, shown, frame_id)

    def on_track_update(self, track_id, *, camera_id=None, frame_id=0, t_sec=None,
                        world_xy=None, bbox=None, world_v=None,
                        embedding=None, crop=None, zone=None):
        """M4 每幀(或每隔幾幀)回報「這條 track 還在,現在在這裡」。

        ⚠ 這個介面是**地面校正的前提**,不是可有可無的優化。
          M5 原本只吃事件(new_track / lost / removed),所以它只知道每位 chef
          **上次綁定時**在哪裡。全景鏡頭的 track 是連續的、中間沒有綁定事件,
          於是比對「30 秒前的位置」與「現在的位置」→ 距離很大 → 強負證據 →
          把對的綁定擋掉(實測碎裂從 0.2% 暴增到 22.7%)。

          這與第五輪踩到的 `_cam` 時間戳問題是同一個結構性缺口:
          **M4 每幀輸出 active tracks,但 M5 只吃事件。**
          第五輪用 chef.track_ids 繞過了「誰在畫面裡」,但「在哪個位置」繞不過。
        """
        cid = self.track_to_chef.get(self._key(track_id, camera_id))
        if cid is None:
            return None
        t = self._t(frame_id, t_sec, camera_id)
        self._cam[cid] = (camera_id, t)
        if world_xy is not None:
            self._world[cid] = (world_xy, t)
        if world_v is not None:
            self._vel[cid] = world_v
        # F4:CrossViewLR 要知道候選此刻在**那一台重疊鏡頭**上的 bbox。
        # 與 world_xy 同一個理由 —— M4 每幀輸出 active tracks,但 M5 只吃事件,
        # 所以不在這裡記的話,拿到的會是「上次綁定時」的位置。
        if bbox is not None:
            self._bbox[(cid, camera_id)] = (tuple(bbox[:4]), t)

        # ── P1 累積投票(2026-09-13,預設關閉)────────────────────────────
        rv = getattr(self.topo, "revote", None)
        if rv is None:
            return cid
        gk = self._key(track_id, camera_id)
        self._vote_calls[gk] = self._vote_calls.get(gk, 0) + 1
        if self._vote_calls[gk] % rv["stride_loops"] != 0:
            return cid                       # 還沒到取樣點

        # ⚠ 沒有 embedding 就**不投票**,而不是用預設值硬投 —— 外觀是候選排序的
        #   唯一依據之一,拿不到它等於投一張沒有根據的票。
        #   計數器讓「整輪都沒投到票」這件事看得見,不會靜默失效。
        if embedding is None and crop is None:
            self._n_revote_no_emb += 1
            return cid
        emb = l2norm(embedding if embedding is not None else self.embedder.extract(crop))

        # ⚠ exclude_key 把**這條 track 自己**排除:不排除的話,這位 chef 會因為
        #   「他正在這台鏡頭上」而自我佐證,投票永遠投給現任。
        cands = self._score_candidates(camera_id, t, emb, bbox, zone, world_xy,
                                       world_v, exclude_key=gk)

        if rv["assignment"] == "hungarian":
            # P2:先緩衝,等這台鏡頭這一輪的 track 都進來了再一次解全域指派。
            # 由 runner 呼叫 resolve_frame() 觸發;沒呼叫的話票不會投出去,
            # 而 revote_pending 這個診斷會讓那件事看得見(不是靜默失效)。
            self._pending_vote.setdefault(camera_id, []).append(
                dict(gk=gk, cid=cid, cands=cands, frame_id=frame_id, t=t,
                     emb=emb, bbox=bbox))
            return cid

        self._cast_vote(gk, cid, cands, frame_id, t, camera_id, emb, bbox)
        votes = self._votes[gk]
        return self._settle(gk, cid, frame_id, t, camera_id, emb, bbox)

    def _cast_vote(self, gk, cid, cands, frame_id, t, camera_id, emb, bbox,
                   forced=None):
        """投一票。forced 不為 None 時用它(P2 的全域指派結果)。

        ⚠ 過不了門檻就投「維持現狀」而不是「開新身份」—— 那是「沒有證據要改」
          不是「證據說要改」。投成開新身份會讓每一輪都想拆掉現有綁定。
        """
        rv = self.topo.revote
        if forced is None:
            thr = (self.topo.llr_threshold if self.f["mode"] == "llr"
                   else self.f["combined_threshold"])
            best = max((c for c in cands if c[0] >= thr), default=None)
            forced = best[1] if best is not None else cid
        self._votes.setdefault(gk, deque(maxlen=rv["window"])).append(forced)
        self._n_revotes += 1

    def _settle(self, gk, cid, frame_id, t, camera_id, emb, bbox):
        """看票箱決定要不要改判。"""
        rv = self.topo.revote
        votes = self._votes.get(gk)
        if votes is None or len(votes) < rv["min_votes"]:
            return cid                       # 票數不足 → 短 track 保持原判
        tally = Counter(votes)
        top, n_top = tally.most_common(1)[0]
        # ⚠ 要改判,領先者必須比現任多 switch_margin 票。5:4 這種幾乎平手就改判
        #   會製造新的 ID switch —— 那是把一種錯誤換成另一種。
        if top != cid and n_top - tally.get(cid, 0) >= rv["switch_margin"]:
            self._rebind(gk, cid, top, frame_id, t, camera_id, emb, bbox)
            self._n_revote_switch += 1
            return top
        return cid

    def resolve_frame(self, camera_id):
        """P2:把這台鏡頭這一輪緩衝的 track 一次做**全域一對一指派**,再投票。

        沒開 hungarian 時是 no-op,所以 runner 可以無條件呼叫。

        ⚠ 為什麼要一對一:同一時刻一台鏡頭上的兩條 track **不可能是同一個人**。
          貪心讓每條各自搶最高分,於是兩條可能搶到同一位;Hungarian 會重新
          安排整組配對,總分最佳且滿足這個物理約束。
          F2(同鏡頭互斥)是它的粗糙版 —— F2 只能說「不准選」,不會重排。
        """
        rv = getattr(self.topo, "revote", None)
        pend = self._pending_vote.pop(camera_id, None)
        if rv is None or not pend:
            return 0
        import numpy as np
        from scipy.optimize import linear_sum_assignment

        thr = (self.topo.llr_threshold if self.f["mode"] == "llr"
               else self.f["combined_threshold"])
        chefs = sorted({c[1] for it in pend for c in it["cands"] if c[0] >= thr})
        if chefs:
            idx = {ch: j for j, ch in enumerate(chefs)}
            # ⚠ 不可行的配對給一個「比任何可行解都差」的大成本,而不是 inf ——
            #   inf 會讓 linear_sum_assignment 在無完美匹配時直接丟例外。
            BIG = 1e6
            cost = np.full((len(pend), len(chefs)), BIG, dtype=float)
            for i, it in enumerate(pend):
                for sc, ch, _a in it["cands"]:
                    if sc >= thr:
                        cost[i, idx[ch]] = -float(sc)     # 最大化分數 = 最小化 −分數
            rows, cols = linear_sum_assignment(cost)
            picked = {int(r): chefs[int(c)] for r, c in zip(rows, cols)
                      if cost[r, c] < BIG}                # 被指到不可行格 = 沒指派到
        else:
            picked = {}

        for i, it in enumerate(pend):
            # 沒被指派到任何 chef → 投「維持現狀」,與 greedy 的語意一致
            self._cast_vote(it["gk"], it["cid"], it["cands"], it["frame_id"], it["t"],
                            camera_id, it["emb"], it["bbox"],
                            forced=picked.get(i, it["cid"]))
        for it in pend:
            self._settle(it["gk"], it["cid"], it["frame_id"], it["t"], camera_id,
                         it["emb"], it["bbox"])
        self._n_resolved += len(pend)
        return len(pend)

    def _rebind(self, gk, old_cid, new_cid, frame_id, t, camera_id, emb, bbox):
        """把一條已綁定的 track 從 old_cid 搬到 new_cid。

        ⚠ 兩邊的 track_ids 都要維護 —— 只改 track_to_chef 會讓舊 chef 永遠以為
          自己還在這台鏡頭上,污染後續所有重疊路徑的候選判斷。
        """
        old = self.active.get(old_cid)
        if old is not None and gk in old.track_ids:
            old.track_ids.remove(gk)
            if not old.track_ids:            # 舊 chef 沒有任何鏡頭看得到了
                old.state = "gone"
                self.active.pop(old_cid, None)
                self.gone[old_cid] = old
                self._exit[old_cid] = (camera_id, t, bbox, None)
        self._bbox.pop((old_cid, camera_id), None)

        new = self.gone.pop(new_cid, None) or self.active.get(new_cid)
        if new is None:                      # 候選在這期間被 TTL 清掉了
            return
        new.state = "active"
        self.active[new_cid] = new
        if gk not in new.track_ids:
            new.track_ids.append(gk)
        new.last_seen = frame_id
        new.embedding = l2norm(self.ema * new.embedding + (1 - self.ema) * emb)
        self.track_to_chef[gk] = new_cid
        self._cam[new_cid] = (camera_id, t)
        if bbox is not None:
            self._bbox[(new_cid, camera_id)] = (tuple(bbox[:4]), t)

    def candidate_histogram(self):
        """回傳 {候選數: 次數}。'5' 代表 5 個以上。

        典型廚房情境應該絕大多數落在 0(開新人)與 1(唯一候選)。
        若 2 以上佔比高,表示外觀真的在仲裁,embedder 品質才會影響結果。
        """
        return {i: n for i, n in enumerate(self._cand_hist) if n}

    # ── 離場語意三段式 ────────────────────────────────────────────────
    # M4 的 lost_track 在「短暫遮擋」就會觸發,而 track 被救回時不會再發 new_track
    # (new_track 只認 active_ids - _seen_ids,而 _seen_ids 是累積的)。
    # 若在 lost 就標 gone,該 chef 會永遠卡在 gone、綁定永久遺失。
    # 因此:lost → 只記預備出口;reacquired → 撤銷;removed → 才真的離場。

    def on_track_lost(self, track_id, *, camera_id=None, frame_id=0, t_sec=None,
                      bbox=None, zone=None):
        """進 ByteTrack lost。只記「預備出口」(含離場位置),不動 chef 狀態。

        bbox 是 M4 該 track 最後已知的位置,同鏡頭重關聯要靠它分辨斷軌前後是不是
        同一個人 —— 只有時間的話,同鏡頭有多人時會綁錯(實測誤併率會翻倍)。
        """
        gk = self._key(track_id, camera_id)
        cid = self.track_to_chef.get(gk)
        if cid is None:
            return None
        self._pending_exit[gk] = (camera_id, self._t(frame_id, t_sec, camera_id),
                                  frame_id, bbox, zone)
        return cid

    def on_track_reacquired(self, track_id, *, camera_id=None, frame_id=0, t_sec=None,
                            bbox=None, zone=None):
        """從 lost 找回(人沒走,只是被擋住)→ 撤銷預備出口,chef 維持 active。"""
        gk = self._key(track_id, camera_id)
        self._pending_exit.pop(gk, None)
        cid = self.track_to_chef.get(gk)
        if cid is None:
            return None
        chef = self.active.get(cid)
        if chef is not None:
            chef.last_seen = frame_id
            self._cam[cid] = (camera_id, self._t(frame_id, t_sec, camera_id))
        return cid

    def on_track_removed(self, track_id, *, camera_id=None, frame_id=0, t_sec=None,
                         bbox=None, zone=None):
        """lost buffer 到期 → 真正離場。

        ⚠ 出口時間戳取自 lost_track 當時,不是 removed 當時。用 removed 的話,
        離場時間會系統性延後約 lost_track_buffer/fps,使所有轉場 Δt 偏小,
        直接讓 transition_gate 的時間窗判斷失準。
        """
        gk = self._key(track_id, camera_id)
        pending = self._pending_exit.pop(gk, None)
        if pending is not None:
            exit_cam, exit_t, exit_frame, exit_box, exit_zone = pending
        else:                              # 沒收到 lost 就直接 removed(理論上不該發生)
            exit_cam, exit_t, exit_frame, exit_box, exit_zone = (
                camera_id, self._t(frame_id, t_sec, camera_id), frame_id, bbox, zone)

        cid = self.track_to_chef.pop(gk, None)
        if cid is None:
            return None
        chef = self.active.get(cid)
        if chef is None:
            return None
        if gk in chef.track_ids:
            chef.track_ids.remove(gk)
        # F4:這位 chef 在這台鏡頭上已經看不到了 → 他的腳點也不再有效。
        # 留著的話會拿「離場前的位置」去跟現在比,製造假的位置證據
        # (與 :316 那個出口時間戳的坑同一種形狀)。
        self._bbox.pop((cid, camera_id), None)
        # P1:track 沒了,它的投票紀錄也該清 —— 不清的話 _votes 與 _vote_calls
        # 會隨累計人次單調成長,違反「活躍清單記憶體不無限成長」那條規格。
        self._votes.pop(gk, None)
        self._vote_calls.pop(gk, None)
        chef.last_seen = exit_frame
        if not chef.track_ids:                                  # 無任何鏡頭還看得到 → 離場
            chef.state = "gone"
            self.active.pop(cid, None)
            self.gone[cid] = chef
            self._exit[cid] = (exit_cam, exit_t, exit_box, exit_zone)
            self._cam.pop(cid, None)
        return cid
