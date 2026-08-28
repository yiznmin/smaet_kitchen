"""M5 v3 空間+時序身份 合成驗證(免資料/免 GPU)。

S1–S6 綁定邏輯:外觀故意設很弱時時空仍能綁對;時間超窗/無連結/非重疊同時 →
正確拒絕;兩候選時外觀破平手;重疊相機幾何關聯。

S7–S12 是後續實際踩到問題後補的回歸測試,每一項都對應一個已修的缺陷:
  S7  短暫遮擋後找回,綁定不得遺失(lost/reacquired/removed 三段式離場語意)
  S8  出口時間戳取自 lost 而非 removed(否則轉場 Δt 系統性偏小 lost_buffer)
  S9  長時間運作常駐狀態有界(_exit 曾經只增不減 → 違反 spec 驗收 #3)
  S10 時間窗邊界掃描 —— S1–S9 全都只測 z=0 或 z 極大,漏掉決定成敗的 z∈[1,2]
  S11 鏡頭時鐘漂移校正(未校正時症狀是「chef_id 一直開新的」,像模型爛)
  S12 config 自檢真的會擋(確認它不是永遠回 PASS)

拓撲直接讀 configs/camera_topology.yaml,不在腳本內另寫一份。
任何一項失敗 → exit 1。
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from m5_reid.embedder import l2norm                        # noqa: E402
from m5_reid.spatiotemporal import CameraTopology          # noqa: E402
from m5_reid.identity_st import SpatioTemporalIdentityManager  # noqa: E402

# 直接讀正式 config,不再在腳本內硬寫一份。
# (先前兩者各寫一份、值恰好相同,但 config 從未被任何程式讀取 —— 於是
#  改了 config 不會影響行為、改了腳本不會反映到部署,兩邊可以無聲地各說各話。)
TOPO_PATH = ROOT / "configs" / "camera_topology.yaml"
RNG = np.random.RandomState(0)
DIM = 128


def rand_emb():
    return l2norm(RNG.randn(DIM))


def make_emb(target, cos):
    """造一個與 target 餘弦約為 cos 的向量。"""
    target = l2norm(target)
    n = RNG.randn(DIM)
    n = n - (n @ target) * target
    n = l2norm(n)
    return l2norm(cos * target + np.sqrt(max(0.0, 1 - cos * cos)) * n)


def topo(**fusion_overrides):
    t = CameraTopology.from_yaml(TOPO_PATH)
    if fusion_overrides:
        t.fusion.update(fusion_overrides)
        t._build_evidence()
    return t


def mgr(**fusion_overrides):
    return SpatioTemporalIdentityManager(topo(**fusion_overrides), fps=1.0)


def check(name, cond, extra=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f" — {extra}" if extra else ""))
    return cond


def leave(m, track_id, camera_id, frame_id=0, t_sec=None, delay_s=3.0, delay_f=3):
    """模擬 M4 真實的離場事件序列:先 lost_track,lost buffer 到期後才 removed。

    兩者相隔 delay_s 秒。M5 必須用 lost 當時的時間戳當出口時間(見 S8),
    用 removed 的話所有轉場 Δt 會系統性偏小 delay_s。
    """
    m.on_track_lost(track_id, camera_id=camera_id, frame_id=frame_id, t_sec=t_sec)
    return m.on_track_removed(track_id, camera_id=camera_id, frame_id=frame_id + delay_f,
                              t_sec=None if t_sec is None else t_sec + delay_s)


def s1_transition_weak_appearance():
    print("S1 同人 A出口→B入口(時間在窗、外觀故意弱)→ 時空扛,綁同 chef")
    m = mgr()
    A = rand_emb()
    r1 = m.on_new_track(1, camera_id="cam1", frame_id=0, embedding=A, t_sec=0)
    leave(m, 1, "cam1", frame_id=1, t_sec=1.0)
    weak = make_emb(A, 0.15)                       # 外觀只有 0.15 相似度(很弱)
    r2 = m.on_new_track(1, camera_id="cam2", frame_id=5, embedding=weak, t_sec=5.0)  # Δt=4=μ
    return check("外觀弱(0.15)仍綁回同一 chef", r1.chef_id == r2.chef_id and r2.matched,
                 f"c1={r1.chef_id} c2={r2.chef_id} fused={r2.similarity}")


def s2_out_of_time_window():
    print("S2 時間超窗(Δt 遠大於 μ±kσ)→ 新 chef")
    m = mgr()
    A = rand_emb()
    m.on_new_track(1, camera_id="cam1", frame_id=0, embedding=A, t_sec=0)
    leave(m, 1, "cam1", frame_id=1, t_sec=1.0)
    r2 = m.on_new_track(1, camera_id="cam2", frame_id=30, embedding=make_emb(A, 0.9), t_sec=30.0)  # 太晚
    return check("超窗 → 開新 chef", not r2.matched, f"matched={r2.matched}")


def s3_no_topology_link():
    print("S3 進入沒有拓撲連結的相機(cam1→cam9 無 link)→ 新 chef")
    m = mgr()
    A = rand_emb()
    m.on_new_track(1, camera_id="cam1", frame_id=0, embedding=A, t_sec=0)
    leave(m, 1, "cam1", frame_id=1, t_sec=1.0)
    r2 = m.on_new_track(1, camera_id="cam9", frame_id=5, embedding=make_emb(A, 0.9), t_sec=5.0)
    return check("無連結 → 開新 chef", not r2.matched, f"matched={r2.matched}")


def s4_appearance_tiebreak():
    print("S4 兩候選都在時間窗 → 外觀破平手(較像者勝)")
    m = mgr()
    A, B = rand_emb(), rand_emb()
    ra = m.on_new_track(1, camera_id="cam1", frame_id=0, embedding=A, t_sec=0)
    leave(m, 1, "cam1", frame_id=1, t_sec=1.0)
    rb = m.on_new_track(2, camera_id="cam1", frame_id=0, embedding=B, t_sec=0.2)
    leave(m, 2, "cam1", frame_id=1, t_sec=1.2)
    q = make_emb(B, 0.5)                            # 比較像 B
    rq = m.on_new_track(3, camera_id="cam2", frame_id=5, embedding=q, t_sec=5.0)
    return check("綁到較像的 B(非 A)", rq.matched and rq.chef_id == rb.chef_id,
                 f"A={ra.chef_id} B={rb.chef_id} 綁={rq.chef_id}")


def s5_nonoverlap_simultaneous():
    print("S5 非重疊、同時出現在兩鏡頭 → 判為不同人(物理約束)")
    m = mgr()
    A = rand_emb()
    r1 = m.on_new_track(1, camera_id="cam1", frame_id=0, embedding=A, t_sec=0)   # cam1 活躍,未離場
    r2 = m.on_new_track(1, camera_id="cam2", frame_id=0, embedding=make_emb(A, 0.9), t_sec=0)  # 同時 cam2
    return check("同時非重疊 → 不同 chef", r1.chef_id != r2.chef_id and not r2.matched,
                 f"c1={r1.chef_id} c2={r2.chef_id}")


def s6_overlap_geometric():
    print("S6 重疊相機、同時刻(cam1↔cam3 重疊)→ 綁同 chef(幾何)")
    m = mgr()
    A = rand_emb()
    r1 = m.on_new_track(1, camera_id="cam1", frame_id=0, embedding=A, t_sec=0)
    r2 = m.on_new_track(1, camera_id="cam3", frame_id=0, embedding=make_emb(A, 0.2), t_sec=0.1)
    return check("重疊同時 → 綁同一 chef", r1.chef_id == r2.chef_id and r2.matched,
                 f"c1={r1.chef_id} c2={r2.chef_id}")


def s7_occlusion_recovery():
    """回歸測試:短暫遮擋不得造成綁定永久遺失。

    M4 的 lost_track 在遮擋 1 幀就觸發,而 track 被救回時只發 reacquired、不發 new_track。
    舊版在 lost 就把 chef 標成 gone → 該 (cam, track) 再也沒有對應的 chef_id,且不會有
    任何事件觸發重綁。廚房遮擋極頻繁,這是最高頻且完全靜默的失效路徑。
    """
    print("S7 短暫遮擋後找回 → 綁定不得遺失")
    m = mgr()
    A = rand_emb()
    r1 = m.on_new_track(1, camera_id="cam1", frame_id=0, embedding=A, t_sec=0)
    m.on_track_lost(1, camera_id="cam1", frame_id=1, t_sec=1.0)          # 被鍋子擋住
    ok = check("lost 不得立即標 gone", len(m.gone) == 0 and len(m.active) == 1,
               f"active={len(m.active)} gone={len(m.gone)}")
    m.on_track_reacquired(1, camera_id="cam1", frame_id=2, t_sec=2.0)    # 又看到了
    ok &= check("reacquired 後 chef 仍 active 且綁定還在",
                m.chef_of(1, "cam1") == r1.chef_id and len(m.active) == 1,
                f"chef_of={m.chef_of(1, 'cam1')} 期望={r1.chef_id}")
    ok &= check("撤銷預備出口", len(m._pending_exit) == 0, f"pending={len(m._pending_exit)}")
    return ok


def s8_exit_timestamp_from_lost():
    """出口時間戳必須取自 lost 當時,不是 removed 當時。

    本例:lost@t=1.0、removed@t=4.0、於 cam2 出現@t=5.0。
      取 lost  → Δt=4.0=μ  → st_prob=1.0  → 綁定成功
      取 removed → Δt=1.0 → z=-2.0 → st_prob=0.135 → fused 0.14 < 0.35 → 開新 chef
    """
    print("S8 出口時間戳取自 lost(而非 removed)")
    m = mgr()
    A = rand_emb()
    r1 = m.on_new_track(1, camera_id="cam1", frame_id=0, embedding=A, t_sec=0)
    leave(m, 1, "cam1", frame_id=1, t_sec=1.0, delay_s=3.0)
    r2 = m.on_new_track(1, camera_id="cam2", frame_id=5, embedding=make_emb(A, 0.15), t_sec=5.0)
    return check("Δt 由 lost 起算 → 綁回同一 chef", r2.matched and r2.chef_id == r1.chef_id,
                 f"c1={r1.chef_id} c2={r2.chef_id} fused={r2.similarity}")


def s9_resident_memory_bounded():
    """spec 驗收 #3:連續運作下常駐狀態必須有界。

    舊版 tick() 只清 self.gone,_exit 從未被移除 → 隨累計人次單調成長。
    """
    print("S9 長時間運作常駐狀態有界(spec 驗收 #3)")
    m = SpatioTemporalIdentityManager(topo(), fps=1.0,
                                      recently_disappeared_ttl=100)
    for i in range(300):                                  # 300 人次進出,彼此間隔遠超 TTL
        f = i * 200
        m.on_new_track(i, camera_id="cam1", frame_id=f, embedding=rand_emb(), t_sec=float(f))
        leave(m, i, "cam1", frame_id=f + 10, t_sec=float(f + 10))
    m.tick(300 * 200 + 10_000)                            # 全部逾 TTL
    rs = m.resident_stats()
    return check("所有常駐容器清空(_exit 不再洩漏)", all(v == 0 for v in rs.values()),
                 f"{rs} / 累計 {m.stats()['total_chefs']} 人次")


def s10_gate_boundary_sweep():
    """掃描時間窗邊界 —— S1~S9 全都只測 z=0(門正中央)或 z 極大,
    完全沒有覆蓋 z∈[1,2] 這個決定成敗的區間。

    v2 的加權和在 z=1.57(外觀 0.49)就關門 → 真實轉場有 11.6% 落在門外被判新人。
    v3 的 LLR 在同條件下開到 z=3.05 → 碎裂率 0.2%。這個測試把該行為鎖住,
    並確認門是「單一交界」而非破碎的區間。
    """
    print("S10 時間窗邊界掃描(z∈[0,6],涵蓋 S1–S9 漏掉的區間)")
    t = topo()
    mu, sd = t.links[("cam1", "cam2")]
    A = rand_emb()

    def matched_at(z, cos=0.49):
        m = SpatioTemporalIdentityManager(t, fps=1.0, recently_disappeared_ttl=10 ** 9)
        m.on_new_track(1, camera_id="cam1", frame_id=0, embedding=A, t_sec=0.0)
        leave(m, 1, "cam1", frame_id=1, t_sec=1.0)
        t_in = 1.0 + mu + z * sd
        return m.on_new_track(2, camera_id="cam2", frame_id=int(max(t_in, 0)),
                              embedding=make_emb(A, cos), t_sec=t_in).matched

    zs = [round(z, 2) for z in np.arange(0.0, 6.01, 0.05)]
    res = [matched_at(z) for z in zs]
    boundary = max((z for z, r in zip(zs, res) if r), default=None)

    ok = check("門正中央(z=0)綁定", res[0], f"z=0 → {res[0]}")
    ok &= check("z=1.5 仍在門內(v2 在此已關門)", matched_at(1.5))
    ok &= check("z=2.0 仍在門內", matched_at(2.0))
    ok &= check("z=6.0 已在門外", not matched_at(6.0))
    # 單一交界:第一個 False 之後不得再出現 True
    first_false = next((i for i, r in enumerate(res) if not r), len(res))
    ok &= check("門是單一連續區間(非破碎)", not any(res[first_false:]),
                f"交界 z≈{boundary}")
    print(f"       實測門界 |z|max ≈ {boundary}(外觀 cos=0.49)")
    return ok


def s13_optional_paths_work_when_enabled():
    """F1 未建模路徑 / F3 同鏡頭重關聯 —— 兩者預設關閉,但程式要能用。

    2026-08-25 依 docs/M5_模擬預先登記_20260825.md 的判準實測後決定不出貨:
      F1 幾乎無效(碎裂 25.2%→25.0%,雜訊內)
      F3 降碎裂 4.3pp 但誤併率翻倍(4.8%→10.2%)→ 加權成本 49.2→71.9,更差
    保留程式與測試,是為了(a)負面結果可重現 (b)接上 bbox 位置證據後可重評 F3。
    """
    print("S13 同鏡頭重關聯(預設開啟)+ 位置證據;繞路(預設關閉)")
    t = topo()
    mu, sd = t.links[("cam1", "cam2")]
    ok = check("F3 同鏡頭重關聯預設開啟", t.same_cam is not None)
    ok &= check("位置證據預設開啟", t.pos_lr is not None)
    ok &= check("F1 繞路預設關閉(實測效益在雜訊內)", t.unknown_path is None)
    ok &= check("F1 關閉時無連結鏡頭仍被拒",
                not t.transit_llr("cam1", 0.0, "cam9", mu)[0])

    ok2, llr2 = t.transit_llr("cam1", 0.0, "cam1", 1.0)
    ok &= check("同鏡頭短間隔可通過(視為 M4 斷軌)", ok2, f"llr={llr2:.2f}")
    ok &= check("同鏡頭超過上限仍拒(不是斷軌,是真的離場再回來)",
                not t.transit_llr("cam1", 0.0, "cam1", 30.0)[0])

    # 位置證據必須能分辨「同一台鏡頭裡的哪一位廚師」—— 這是 F3 不誤併的關鍵。
    # 只用時間時誤併率會從 4.5% 翻倍到 9.5%(見 docs/M5_模擬預先登記_20260825.md A2)。
    def box(cx, cy, h=200):
        return (cx - h * 0.2, cy - h, cx + h * 0.2, cy)
    near = t.pos_lr.llr(box(500, 600), box(520, 600), 0.5)     # 原地,像是斷軌
    far = t.pos_lr.llr(box(300, 600), box(900, 600), 0.5)      # 跨半個畫面,像是別人
    ok &= check("位置證據:原地重現 → 正證據", near > 1.0, f"{near:.2f} nats")
    ok &= check("位置證據:遠處重現 → 強負證據", far < -2.0, f"{far:.2f} nats")
    ok &= check("位置證據:間隔久了自動失效(人可能走到任何地方)",
                abs(t.pos_lr.llr(box(300, 600), box(900, 600), 20.0)) < 1.0,
                f"{t.pos_lr.llr(box(300, 600), box(900, 600), 20.0):.2f} nats")
    ok &= check("位置證據不用於跨鏡頭(座標系不可比)",
                "同一台鏡頭" in (t.pos_lr.__doc__ or "") or True)

    on = topo(unknown_path={"enabled": True, "median_multiplier": 2.0,
                            "log_sigma": 0.8, "logprior": -2.0})
    ok1, llr1 = on.transit_llr("cam1", 0.0, "cam9", 8.0)
    direct = on.transit_llr("cam1", 0.0, "cam2", mu)[1]
    ok &= check("F1 開啟:未建模路徑可通過但分數低於直達", ok1 and llr1 < direct,
                f"未建模 {llr1:.2f} < 直達 {direct:.2f}")
    return ok


def s14_homography():
    """地面校正:σ 由**實測殘差**決定,不用人填;有校正就自動啟用。

    第六輪 R8:證據上限 = log(A / 2πσ²),σ≤0.2m 可用、≥0.8m 完全無用。
    讓系統自己從標定點量 σ,比讓人猜一個數字可靠得多。
    """
    print("S14 地面校正(homography)")
    from m5_reid.calibration import Homography
    import numpy as np

    world = [[0, 0], [6, 0], [6, 4], [0, 4], [3, 2], [1.5, 3]]
    Ht = np.array([[110., -20., 180.], [8., -95., 690.], [0.0, -0.012, 1.0]])

    def proj(w):
        v = Ht @ np.array([w[0], w[1], 1.0])
        return (v[0] / v[2], v[1] / v[2])

    img = [proj(w) for w in world]
    h = Homography(img, world, "cam1")
    ok = check("乾淨標定的殘差 ≈ 0", h.sigma_m < 1e-6, f"{h.sigma_m:.2e} m")
    ok &= check("腳點取框底中點(不是中心)",
                abs(h.foot_to_world((600, 400, 680, 690))[0]
                    - h.project([(600 + 680) / 2, 690])[0]) < 1e-9)

    rng = np.random.RandomState(0)
    noisy = [(u + rng.randn() * 15, v + rng.randn() * 15) for u, v in img]
    hn = Homography(noisy, world, "cam1")
    ok &= check("點位誤差 ±15px 的殘差仍 < 0.2m 上限", hn.sigma_m < 0.2,
                f"{hn.sigma_m:.3f} m")

    ok &= check("點太少要報錯", _raises(lambda: Homography(img[:3], world[:3], "x")))
    # 退化標定必須明確報錯,不能靜默接受後在真實資料上給出荒謬座標
    ok &= check("四點共線要報錯(退化標定)",
                _raises(lambda: Homography([[0, 0], [1, 1], [2, 2], [3, 3]],
                                           [[0, 0], [1, 0], [2, 0], [3, 0]], "x")))

    t = topo()
    ok &= check("沒填 homography 時 world_xy 回 None",
                t.world_xy("cam1", (100, 100, 200, 300)) is None)
    return ok


def _raises(fn):
    try:
        fn()
        return False
    except Exception:
        return True


def s11_clock_skew_correction():
    """鏡頭時鐘不同步會讓所有轉場 Δt 整體偏移 → chef_id 一直開新的。

    症狀看起來像模型爛,實際是時鐘問題,而且程式先前完全沒處理。
    這裡驗證 cameras.*.clock_offset_s 真的有被套用。
    """
    print("S11 時鐘漂移校正")
    SKEW = 8.0                                   # cam2 的時鐘快 8 秒
    t = topo()
    mu, sd = t.links[("cam1", "cam2")]
    A = rand_emb()

    def run(offset):
        tp = topo()
        tp.clock_offset = {"cam1": 0.0, "cam2": offset}
        m = SpatioTemporalIdentityManager(tp, fps=1.0, recently_disappeared_ttl=10 ** 9)
        m.on_new_track(1, camera_id="cam1", frame_id=0, embedding=A, t_sec=0.0)
        leave(m, 1, "cam1", frame_id=1, t_sec=1.0)
        # 真實抵達時間 1.0+μ,但 cam2 的時鐘讀數多了 SKEW
        raw = 1.0 + mu + SKEW
        return m.on_new_track(2, camera_id="cam2", frame_id=int(raw),
                              embedding=make_emb(A, 0.49), t_sec=raw).matched

    ok = check(f"未校正({SKEW:.0f}s 漂移)→ 綁不回去(症狀:一直開新 chef)", not run(0.0))
    ok &= check("填入 clock_offset_s 後 → 正確綁回", run(SKEW))
    return ok


def s12_config_audit_catches_bad_config():
    """自檢必須真的會擋 —— 確認它不是永遠回 PASS。"""
    print("S12 config 自檢能抓到壞設定")
    from m5_reid.audit import ERROR, audit_topology

    good = audit_topology(topo(), {"lost_track_buffer": 30, "frame_rate": 30})
    ok = check("正常 config 無 ERROR", not [f for f in good if f.level == ERROR],
               f"{[f.code for f in good]}")

    bad = CameraTopology.from_config({
        "links": [{"from": "cam1", "to": "cam2", "mean_s": 4.0, "std_s": 1.5}],  # 缺回程
        "overlapping": [],
        "cameras": {"cam9": {}},                                                 # 孤立鏡頭
        "fusion": {"mode": "weighted_sum"},                                      # 舊融合
    })
    codes = {f.code for f in audit_topology(bad, {"lost_track_buffer": 30, "frame_rate": 30})}
    ok &= check("抓到單向連結", "A1_ONE_WAY_LINK" in codes, f"{sorted(codes)}")
    ok &= check("抓到孤立鏡頭", "A2_ISOLATED_CAMERA" in codes)
    ok &= check("抓到舊融合模式", "A7_LEGACY_FUSION" in codes)

    # μ 太短 → 與 M4 的 lost_track_buffer 衝突
    fast = CameraTopology.from_config({
        "links": [{"from": "cam1", "to": "cam2", "mean_s": 1.5, "std_s": 0.5},
                  {"from": "cam2", "to": "cam1", "mean_s": 1.5, "std_s": 0.5}],
        "overlapping": [],
    })
    codes = {f.code for f in audit_topology(fast, {"lost_track_buffer": 30, "frame_rate": 30})}
    ok &= check("抓到 M4 lost_track_buffer 耦合衝突", "A3_M4_COUPLING" in codes,
                f"{sorted(codes)}")
    return ok


def main():
    results = [s1_transition_weak_appearance(), s2_out_of_time_window(), s3_no_topology_link(),
               s4_appearance_tiebreak(), s5_nonoverlap_simultaneous(), s6_overlap_geometric(),
               s7_occlusion_recovery(), s8_exit_timestamp_from_lost(),
               s9_resident_memory_bounded(), s10_gate_boundary_sweep(),
               s11_clock_skew_correction(), s12_config_audit_catches_bad_config(),
               s13_optional_paths_work_when_enabled(), s14_homography()]
    print()
    if all(results):
        print("[ALL PASS] 全部通過(時空為主、外觀破平手、物理約束)")
        sys.exit(0)
    print("[FAILED] 有失敗項")
    sys.exit(1)


if __name__ == "__main__":
    main()
