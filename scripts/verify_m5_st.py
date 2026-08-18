"""M5 v2 空間+時序身份 合成驗證(免資料/免 GPU)。

驗證「時空為主、外觀為輔」的綁定邏輯:即使外觀故意設很弱,時空門也能綁對;
時間超窗/無連結/非重疊同時 → 正確拒絕;兩候選時外觀破平手;重疊相機幾何關聯。

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

TOPO_CFG = {
    "fusion": {"w_st": 0.7, "w_app": 0.3, "k_sigma": 2.0,
               "combined_threshold": 0.35, "overlap_window_s": 0.5},
    "links": [{"from": "cam1", "to": "cam2", "mean_s": 4.0, "std_s": 1.5},
              {"from": "cam2", "to": "cam1", "mean_s": 4.5, "std_s": 1.6}],
    "overlapping": [["cam1", "cam3"]],
}
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


def mgr():
    return SpatioTemporalIdentityManager(CameraTopology.from_config(TOPO_CFG), fps=1.0)


def check(name, cond, extra=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f" — {extra}" if extra else ""))
    return cond


def s1_transition_weak_appearance():
    print("S1 同人 A出口→B入口(時間在窗、外觀故意弱)→ 時空扛,綁同 chef")
    m = mgr()
    A = rand_emb()
    r1 = m.on_new_track(1, "cam1", frame_id=0, embedding=A, t_sec=0)
    m.on_track_lost(1, "cam1", frame_id=1, t_sec=1.0)
    weak = make_emb(A, 0.15)                       # 外觀只有 0.15 相似度(很弱)
    r2 = m.on_new_track(1, "cam2", frame_id=5, embedding=weak, t_sec=5.0)  # Δt=4=μ
    return check("外觀弱(0.15)仍綁回同一 chef", r1.chef_id == r2.chef_id and r2.matched,
                 f"c1={r1.chef_id} c2={r2.chef_id} fused={r2.similarity}")


def s2_out_of_time_window():
    print("S2 時間超窗(Δt 遠大於 μ±kσ)→ 新 chef")
    m = mgr()
    A = rand_emb()
    m.on_new_track(1, "cam1", frame_id=0, embedding=A, t_sec=0)
    m.on_track_lost(1, "cam1", frame_id=1, t_sec=1.0)
    r2 = m.on_new_track(1, "cam2", frame_id=30, embedding=make_emb(A, 0.9), t_sec=30.0)  # 太晚
    return check("超窗 → 開新 chef", not r2.matched, f"matched={r2.matched}")


def s3_no_topology_link():
    print("S3 進入沒有拓撲連結的相機(cam1→cam9 無 link)→ 新 chef")
    m = mgr()
    A = rand_emb()
    m.on_new_track(1, "cam1", frame_id=0, embedding=A, t_sec=0)
    m.on_track_lost(1, "cam1", frame_id=1, t_sec=1.0)
    r2 = m.on_new_track(1, "cam9", frame_id=5, embedding=make_emb(A, 0.9), t_sec=5.0)
    return check("無連結 → 開新 chef", not r2.matched, f"matched={r2.matched}")


def s4_appearance_tiebreak():
    print("S4 兩候選都在時間窗 → 外觀破平手(較像者勝)")
    m = mgr()
    A, B = rand_emb(), rand_emb()
    ra = m.on_new_track(1, "cam1", frame_id=0, embedding=A, t_sec=0)
    m.on_track_lost(1, "cam1", frame_id=1, t_sec=1.0)
    rb = m.on_new_track(2, "cam1", frame_id=0, embedding=B, t_sec=0.2)
    m.on_track_lost(2, "cam1", frame_id=1, t_sec=1.2)
    q = make_emb(B, 0.5)                            # 比較像 B
    rq = m.on_new_track(3, "cam2", frame_id=5, embedding=q, t_sec=5.0)
    return check("綁到較像的 B(非 A)", rq.matched and rq.chef_id == rb.chef_id,
                 f"A={ra.chef_id} B={rb.chef_id} 綁={rq.chef_id}")


def s5_nonoverlap_simultaneous():
    print("S5 非重疊、同時出現在兩鏡頭 → 判為不同人(物理約束)")
    m = mgr()
    A = rand_emb()
    r1 = m.on_new_track(1, "cam1", frame_id=0, embedding=A, t_sec=0)   # cam1 活躍,未離場
    r2 = m.on_new_track(1, "cam2", frame_id=0, embedding=make_emb(A, 0.9), t_sec=0)  # 同時 cam2
    return check("同時非重疊 → 不同 chef", r1.chef_id != r2.chef_id and not r2.matched,
                 f"c1={r1.chef_id} c2={r2.chef_id}")


def s6_overlap_geometric():
    print("S6 重疊相機、同時刻(cam1↔cam3 重疊)→ 綁同 chef(幾何)")
    m = mgr()
    A = rand_emb()
    r1 = m.on_new_track(1, "cam1", frame_id=0, embedding=A, t_sec=0)
    r2 = m.on_new_track(1, "cam3", frame_id=0, embedding=make_emb(A, 0.2), t_sec=0.1)
    return check("重疊同時 → 綁同一 chef", r1.chef_id == r2.chef_id and r2.matched,
                 f"c1={r1.chef_id} c2={r2.chef_id}")


def main():
    results = [s1_transition_weak_appearance(), s2_out_of_time_window(), s3_no_topology_link(),
               s4_appearance_tiebreak(), s5_nonoverlap_simultaneous(), s6_overlap_geometric()]
    print()
    if all(results):
        print("[ALL PASS] 全部通過(時空為主、外觀破平手、物理約束)")
        sys.exit(0)
    print("[FAILED] 有失敗項")
    sys.exit(1)


if __name__ == "__main__":
    main()
