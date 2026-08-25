"""M5 v2 時空門「逐步判斷」可視化 demo(免資料/免 GPU)。

模擬多鏡頭多廚師的一段情節,對每個「有人出現」事件,印出:
  對每個「最近消失的廚師」候選 → 拓撲有連結嗎?Δt 在時間窗嗎?st_prob、外觀、融合分數 → 決定。
讓你直觀看到「走得到走不到 + 時間對不對」是怎麼綁定/拒絕的。

用法:python scripts/demo_m5_st.py
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from m5_reid.embedder import l2norm                            # noqa: E402
from m5_reid.identity import cosine as _cos                    # noqa: E402
from m5_reid.spatiotemporal import CameraTopology, st_prob     # noqa: E402
from m5_reid.identity_st import SpatioTemporalIdentityManager  # noqa: E402

TOPO = {
    "fusion": {"w_st": 0.7, "w_app": 0.3, "k_sigma": 2.0, "combined_threshold": 0.35,
               "overlap_window_s": 0.5},
    "links": [{"from": "cam1", "to": "cam2", "mean_s": 4.0, "std_s": 1.5},
              {"from": "cam2", "to": "cam1", "mean_s": 4.5, "std_s": 1.6}],
    "overlapping": [["cam1", "cam3"]],
}
RNG = np.random.RandomState(1)
DIM = 128


def make_emb(target, cos):
    target = l2norm(target)
    n = RNG.randn(DIM)
    n = l2norm(n - (n @ target) * target)
    return l2norm(cos * target + np.sqrt(max(0.0, 1 - cos * cos)) * n)


def enter(m, topo, tid, cam, t, emb, note=""):
    """印出時空門對每個候選的逐步判斷,再實際綁定。"""
    f = topo.fusion
    print(f"\n▶ t={t:>4.1f}s  cam={cam}  有人出現(track {tid}){'  【' + note + '】' if note else ''}")
    cands = []
    for cid, chef in list(m.gone.items()):
        ex = m._exit.get(cid)
        if ex is None:
            continue
        passed, sp = topo.transition_gate(ex[0], ex[1], cam, t)
        app = _cos(emb, chef.embedding)
        fused = f["w_st"] * sp + f["w_app"] * app if passed else None
        dt = t - ex[1]
        mark = "✅通過" if passed else "✗擋掉"
        reason = ""
        if not passed:
            if (ex[0], cam) not in topo.links:
                reason = f"(無 {ex[0]}→{cam} 連結)"
            elif dt <= 0:
                reason = "(不能比離開早到)"
            else:
                mean, std = topo.links[(ex[0], cam)]
                reason = f"(Δt={dt:.1f}s 超出窗 {mean}±{f['k_sigma']}×{std})"
        fs = f"{fused:.3f}" if fused is not None else "  -  "
        print(f"   候選 chef{cid}: 從 {ex[0]} 於 {ex[1]:.1f}s 離開 | Δt={dt:>4.1f}s "
              f"時空門 {mark}{reason} st={sp:.2f} 外觀={app:.2f} → 融合 {fs}")
        if passed:
            cands.append((fused, cid))
    r = m.on_new_track(tid, camera_id=cam, t_sec=t, frame_id=int(t * 30), embedding=emb)
    if r.matched:
        print(f"   ➜ 決定:融合最高且 ≥{f['combined_threshold']} → **綁定 chef{r.chef_id}**(同一位廚師)")
    else:
        why = "無候選通過時空門" if not cands else f"最高融合 <{f['combined_threshold']}"
        print(f"   ➜ 決定:{why} → **開新 chef{r.chef_id}**")
    return r


def leave(m, tid, cam, t, buffer_s=1.0):
    """M4 真實序列:lost_track 先發,lost buffer 到期才發 removed。

    出口時間戳由 M5 內部取自 lost 當時(t),不是 removed 當時(t+buffer_s)。
    """
    m.on_track_lost(tid, camera_id=cam, frame_id=int(t * 30), t_sec=t)
    m.on_track_removed(tid, camera_id=cam, frame_id=int((t + buffer_s) * 30),
                       t_sec=t + buffer_s)
    print(f"\n◀ t={t:>4.1f}s  cam={cam}  track {tid} 離開 → 記入「最近消失」")


def main():
    topo = CameraTopology.from_config(TOPO)
    m = SpatioTemporalIdentityManager(topo, fps=30.0)
    A = l2norm(RNG.randn(DIM))          # 廚師 A 的外觀

    print("=" * 70)
    print("情節:廚師 A 從 cam1 走到 cam2、再走回 cam1;中間有『太快的冒牌』與『無連結鏡頭』")
    print("拓撲:cam1↔cam2(走 ~4 秒);外觀刻意設很弱(制服相似)")
    print("=" * 70)

    enter(m, topo, 1, "cam1", 0.0, A, note="A 第一次出現")
    leave(m, 1, "cam1", 2.0)
    enter(m, topo, 1, "cam2", 6.0, make_emb(A, 0.18), note="走到 cam2,外觀只有 0.18")
    leave(m, 1, "cam2", 8.0)
    enter(m, topo, 9, "cam1", 8.5, make_emb(A, 0.9), note="太快(0.5s)冒牌者")
    enter(m, topo, 1, "cam1", 13.0, make_emb(A, 0.18), note="A 走回 cam1(Δt=5s 合理)")
    enter(m, topo, 7, "cam2", 30.0, make_emb(A, 0.9), note="無人剛離開/超時")

    print("\n" + "=" * 70)
    print(f"結果:共 {m.stats()['total_chefs']} 位不同 chef_id;A 全程維持同一號(靠時空,外觀僅 0.18)")
    print("=" * 70)


if __name__ == "__main__":
    main()
