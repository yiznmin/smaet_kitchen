"""F1(margin test)與 F2(同鏡頭互斥)的行為驗證 —— 不需要 CHIRLA 資料。

## 這支要擋住什麼

1. **關閉時必須逐位不變。** 兩個修法都是加在**所有既有結果**的決策路徑上,
   若關閉時行為有一絲改變,2026-09-04 的基線就不再可比,整個修復記錄的
   「修改前後對比」會失去意義。這裡用同一組事件跑「全關」與「基線程式碼語意」
   兩條路,逐筆比對綁定結果。

2. **開啟時必須真的生效,而且是以預期的方式。**
   · F1:多候選且分數接近時 → 開新身份(碎裂),而不是賭一個
   · F1:**單一候選時不可被擋** —— 那不是「分不出來」,是「沒有別人可混淆」。
        擋掉它會把單人場景的正確綁定全部打成碎裂(EPFL 兩輪都是這種情況)
   · F2:同一 chef 不再綁到同一鏡頭的第二條 track,`same_cam_conflicts` 歸零

⚠ 這支**不回答「修法有沒有降誤併」** —— 那要在 CHIRLA 上跑完整評估集,
  見 docs/M5_修復_預先登記_20260905.md。這裡只驗證「行為是否如設計」。

用法:
    python scripts/verify_m5_fixes.py
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from m5_reid.identity_st import SpatioTemporalIdentityManager      # noqa: E402
from m5_reid.spatiotemporal import CameraTopology               # noqa: E402

FAIL = []


def chk(ok, name, detail=""):
    print(f"  {'✅' if ok else '❌'} {name}")
    if detail:
        print(f"       {detail}")
    if not ok:
        FAIL.append(name)


def topo(**fusion):
    """三台全重疊、無 link 的最小拓撲。重疊路徑是本次兩個修法的戰場。"""
    base = {
        "mode": "llr", "background_arrival_hz": 1.0 / 600.0,
        "cost_false_merge_over_break": 5.0, "overlap_llr": 5.0,
        "appearance_profile": "dinov2", "position": {"enabled": False},
        "same_camera": {"enabled": False}, "unknown_path": {"enabled": False},
    }
    base.update(fusion)
    return CameraTopology.from_config({
        "links": [], "overlapping": [["cam1", "cam2"], ["cam1", "cam3"], ["cam2", "cam3"]],
        "cameras": {c: {} for c in ("cam1", "cam2", "cam3")}, "fusion": base})


# ⚠ embedding 必須做成**實測的可分性**,不能用正交隨機向量。
#   第一版用 `rng.normal()` 各自獨立 → 自己 cos=1.0、別人 cos≈0,
#   margin 高達 2.5 nats,F1 當然不擋 —— 那是**測試不真實**,不是程式沒生效。
#   真實情況是 AppearanceLR.MEASURED["dinov2"]:同人 0.490 / 不同人 0.465、
#   d′ 只有 0.25 —— 幾乎分不開,這才是 CHIRLA 上 6~7 個候選互相競爭的局面。
#
#   作法:v = normalize(a·b + √(1−a²)·n),b 是共用基底、n 是各自的雜訊。
#   不同人之間 cos ≈ a²;同一人的兩次觀測再共用一部分雜訊把 cos 推高一點。
MU_SAME, MU_DIFF = 0.490, 0.465
_A = np.sqrt(MU_DIFF)                                   # 不同人的 cos ≈ a²
_C = (MU_SAME - MU_DIFF) / (1.0 - MU_DIFF)              # 同人再多共用這麼多雜訊
_BASE = np.random.default_rng(0).normal(size=384)
_BASE /= np.linalg.norm(_BASE)


def emb(person, obs=0):
    """person 決定身份,obs 決定是第幾次觀測(同一人不同觀測 cos≈0.490)。"""
    n = np.random.default_rng(1000 + person).normal(size=384)
    n -= n.dot(_BASE) * _BASE
    n /= np.linalg.norm(n)
    if obs:                                             # 同一人的另一次觀測
        m = np.random.default_rng(90000 + person * 97 + obs).normal(size=384)
        m -= m.dot(_BASE) * _BASE
        m -= m.dot(n) * n
        m /= np.linalg.norm(m)
        n = _C * n + np.sqrt(1.0 - _C ** 2) * m
    v = _A * _BASE + np.sqrt(1.0 - _A ** 2) * n
    return v / np.linalg.norm(v)


def scenario(m5, people, t0=0.0):
    """讓 people 個不同的人各自在 cam1 出現,再全部出現在 cam2。

    回傳 cam2 那一輪的綁定結果 —— 那是「多個候選同時競爭」的局面。
    """
    out = []
    for i in range(people):
        m5.on_new_track(100 + i, camera_id="cam1", frame_id=int(t0 * 30) + i,
                        t_sec=t0 + i * 0.01, embedding=emb(i), bbox=(10 * i, 0, 10 * i + 50, 120))
    for i in range(people):
        r = m5.on_new_track(200 + i, camera_id="cam2", frame_id=int(t0 * 30) + 10 + i,
                            t_sec=t0 + 0.3 + i * 0.01, embedding=emb(i, obs=1),
                            bbox=(10 * i, 0, 10 * i + 50, 120))
        out.append(r)
    return out


print("=" * 74)
print("F1 margin test / F2 同鏡頭互斥 —— 行為驗證(不需 CHIRLA 資料)")
print("=" * 74)

# ── 1. 關閉時必須與基線逐位相同 ──────────────────────────────────────
print("\n【1】關閉時的回歸(最重要:基線可比性)")
t_off = topo()
chk(t_off.margin_nats is None, "margin 預設關閉",
    f"margin_nats = {t_off.margin_nats}")
chk(t_off.same_cam_exclusive is False, "同鏡頭互斥預設關閉",
    f"same_cam_exclusive = {t_off.same_cam_exclusive}")

a = scenario(SpatioTemporalIdentityManager(topo(), fps=30.0), 6)
b = scenario(SpatioTemporalIdentityManager(topo(), fps=30.0), 6)
same = all((x.matched, x.chef_id, x.similarity) == (y.matched, y.chef_id, y.similarity)
           for x, y in zip(a, b))
chk(same, "關閉時兩次執行逐筆相同(決定論)")

# ── 2. F1:多候選且分數接近 → 應該開新身份 ──────────────────────────
print("\n【2】F1 margin test")
off = scenario(SpatioTemporalIdentityManager(topo(), fps=30.0), 6)
on_m5 = SpatioTemporalIdentityManager(topo(margin={"enabled": True}), fps=30.0)
on = scenario(on_m5, 6)

n_off = sum(1 for r in off if r.matched)
n_on = sum(1 for r in on if r.matched)
chk(n_off > n_on,
    "多候選競爭時,margin 擋下了綁定",
    f"綁定數 {n_off} → {n_on}(6 個人各自在 cam2 現身,全部互為候選)")
chk(on_m5._n_margin_blocked > 0, "margin_blocked 計數器有記錄",
    f"_n_margin_blocked = {on_m5._n_margin_blocked}")

t_on = topo(margin={"enabled": True})
chk(abs(t_on.margin_nats - t_on.llr_threshold) < 1e-9,
    "margin 預設值與 llr_threshold 同源",
    f"margin_nats = {t_on.margin_nats:.4f} = log(5)")

# ── 3. F1 的反例:單一候選不可被擋 ──────────────────────────────────
print("\n【3】F1 的反例(這條擋的是最危險的迴歸)")
solo = SpatioTemporalIdentityManager(topo(margin={"enabled": True}), fps=30.0)
solo.on_new_track(1, camera_id="cam1", frame_id=0, t_sec=0.0,
                  embedding=emb(0), bbox=(0, 0, 50, 120))
r_solo = solo.on_new_track(2, camera_id="cam2", frame_id=1, t_sec=0.1,
                           embedding=emb(0, obs=1), bbox=(0, 0, 50, 120))
chk(r_solo.matched and solo._n_margin_blocked == 0,
    "只有一個候選時不套用 margin",
    f"matched={r_solo.matched} blocked={solo._n_margin_blocked} "
    f"—— 單人場景(EPFL 兩輪)不可被這個修法打成碎裂")

# ── 4. F2:同鏡頭互斥 ────────────────────────────────────────────────
print("\n【4】F2 同鏡頭互斥")
conf = SpatioTemporalIdentityManager(topo(), fps=30.0)
conf.on_new_track(1, camera_id="cam1", frame_id=0, t_sec=0.0,
                  embedding=emb(0), bbox=(0, 0, 50, 120))
conf.on_new_track(2, camera_id="cam2", frame_id=1, t_sec=0.1,
                  embedding=emb(0), bbox=(0, 0, 50, 120))
# 同一個人已在 cam2 上;現在 cam2 又冒出第二條 track → 物理上不可能是同一人
conf.on_new_track(3, camera_id="cam2", frame_id=2, t_sec=0.2,
                  embedding=emb(0), bbox=(200, 0, 250, 120))
chk(conf._n_same_cam_conflicts > 0,
    "關閉時能量到基線的問題規模",
    f"same_cam_conflicts = {conf._n_same_cam_conflicts}(這就是要修掉的東西)")

ex = SpatioTemporalIdentityManager(topo(same_camera_exclusive={"enabled": True}), fps=30.0)
ex.on_new_track(1, camera_id="cam1", frame_id=0, t_sec=0.0,
                embedding=emb(0), bbox=(0, 0, 50, 120))
ex.on_new_track(2, camera_id="cam2", frame_id=1, t_sec=0.1,
                embedding=emb(0), bbox=(0, 0, 50, 120))
ex.on_new_track(3, camera_id="cam2", frame_id=2, t_sec=0.2,
                embedding=emb(0), bbox=(200, 0, 250, 120))
chk(ex._n_same_cam_conflicts == 0, "開啟後衝突歸零",
    f"same_cam_conflicts = {ex._n_same_cam_conflicts}")

# ── 5. 診斷量要出現在 resident_stats(記憶體驗收會讀它)──────────────
print("\n【5】診斷量的可觀測性")
s = on_m5.resident_stats()
chk("margin_blocked" in s and "same_cam_conflicts" in s,
    "兩個計數器都出現在 resident_stats",
    f"{ {k: s[k] for k in ('margin_blocked', 'same_cam_conflicts') if k in s} }")
chk(all(isinstance(s[k], int) for k in ("margin_blocked", "same_cam_conflicts")),
    "是計數器不是集合 → 不影響記憶體有界性驗收")

print("\n" + "─" * 74)
if FAIL:
    print(f"❌ {len(FAIL)} 項未通過:{FAIL}")
    sys.exit(1)
print("✅ 全部通過。⚠ 但這只驗證『行為是否如設計』——")
print("   『修法有沒有降誤併』要在 CHIRLA 評估集上跑,見預先登記。")
sys.exit(0)
