"""P1 累積投票的行為驗證 —— 不需要 CHIRLA 資料。

## 這支要擋住什麼

1. **關閉時必須逐位不變。** 投票加在 `on_track_update` 上,而那是所有既有結果
   都會走的路徑。關閉時若有一絲改變,2026-09-12 的 `base` 就不再可比。

2. **開啟時要真的能翻轉錯誤的初判。** 這是 P1 存在的唯一理由 ——
   出生那一刻證據弱、綁錯了,後續的觀測應該把它投回來。

3. **不該翻的不能翻:**
   · 票數不足(短 track)→ 保持原判。實測 53.8% 的 track 只能投 1 票
   · 幾乎平手(差距 < switch_margin)→ 不動。5:4 就改判會製造新的 ID switch
   · 拿不到 embedding → **不投票**而不是用預設值硬投,且計數器要看得見

4. **記憶體有界**:track 移除後投票紀錄要清乾淨。

⚠ 這支**不回答「投票有沒有提升 IDF1」** —— 那要在 CHIRLA 上跑完整評估集。
  這裡只驗證「行為是否如設計」。

用法:
    python scripts/verify_m5_revote.py
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from m5_reid.identity_st import SpatioTemporalIdentityManager      # noqa: E402
from m5_reid.spatiotemporal import CameraTopology                  # noqa: E402

FAIL = []


def chk(ok, name, detail=""):
    print(f"  {'✅' if ok else '❌'} {name}")
    if detail:
        print(f"       {detail}")
    if not ok:
        FAIL.append(name)


def topo(**fusion):
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


# 與 verify_m5_fixes.py 同一套:embedding 必須做成**實測的可分性**,
# 不能用正交隨機向量(那會讓外觀強到不真實,投票的價值被高估)。
MU_SAME, MU_DIFF = 0.490, 0.465
_A = np.sqrt(MU_DIFF)
_C = (MU_SAME - MU_DIFF) / (1.0 - MU_DIFF)
_BASE = np.random.default_rng(0).normal(size=384)
_BASE /= np.linalg.norm(_BASE)


def emb(person, obs=0):
    n = np.random.default_rng(1000 + person).normal(size=384)
    n -= n.dot(_BASE) * _BASE
    n /= np.linalg.norm(n)
    if obs:
        m = np.random.default_rng(90000 + person * 97 + obs).normal(size=384)
        m -= m.dot(_BASE) * _BASE
        m -= m.dot(n) * n
        m /= np.linalg.norm(m)
        n = _C * n + np.sqrt(1.0 - _C ** 2) * m
    v = _A * _BASE + np.sqrt(1.0 - _A ** 2) * n
    return v / np.linalg.norm(v)


def bb(x, y, w=50, h=120):
    return (x - w / 2.0, y - h, x + w / 2.0, y)


# 用 CrossViewLR 當證據軸:它是唯一能真正區分候選的(外觀只有 0.062 nats),
# 所以「投票能不能翻轉」必須在有鑑別力的證據下測,否則測到的是雜訊。
_H = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]     # cam1→cam2 恆等
CV = {"enabled": True, "clip": 8.0,
      "pairs": {"cam1|cam2": {"H": _H, "sigma_px": 20.0, "area_px2": 200000.0}}}
RV = {"enabled": True, "stride_loops": 1, "window": 9, "min_votes": 3,
      "switch_margin": 2}

print("=" * 74)
print("P1 累積投票 —— 行為驗證(不需 CHIRLA 資料)")
print("=" * 74)

# ── 1. 關閉時 ──────────────────────────────────────────────────────
print("\n【1】關閉時的回歸")
chk(topo().revote is None, "revote 預設關閉", f"revote = {topo().revote}")

m = SpatioTemporalIdentityManager(topo(cross_view=CV), fps=30.0)
m.on_new_track(1, camera_id="cam1", frame_id=0, t_sec=0.0, embedding=emb(0), bbox=bb(300, 400))
for k in range(20):
    m.on_track_update(1, camera_id="cam1", frame_id=k + 1, t_sec=0.03 * (k + 1),
                      bbox=bb(300, 400), embedding=emb(0, obs=1))
s = m.resident_stats()
chk(s["revotes"] == 0 and s["_votes"] == 0,
    "關閉時完全不投票、不佔記憶體", f"revotes={s['revotes']} _votes={s['_votes']}")

# ── 2. 開啟時能翻轉錯誤的初判 ───────────────────────────────────────
print("\n【2】投票能翻轉錯誤的初判(P1 存在的唯一理由)")
m = SpatioTemporalIdentityManager(topo(cross_view=CV, revote=RV), fps=30.0)
# 兩個人都在 cam1:A 站在 (300,400),B 站在 (900,400)
m.on_new_track(10, camera_id="cam1", frame_id=0, t_sec=0.0, embedding=emb(0), bbox=bb(300, 400))
m.on_new_track(11, camera_id="cam1", frame_id=0, t_sec=0.0, embedding=emb(1), bbox=bb(900, 400))
chef_a = m.track_to_chef[("cam1", 10)]
chef_b = m.track_to_chef[("cam1", 11)]
chk(chef_a != chef_b, "兩個人各自開了身份", f"chef {chef_a} / {chef_b}")

# cam2 出現一條新 track,位置在 (900,400) → 幾何上是 B。
# 但出生那一刻我們讓它的 bbox 落在 A 附近,製造一個**錯誤的初判**。
r = m.on_new_track(20, camera_id="cam2", frame_id=1, t_sec=0.03,
                   embedding=emb(1, obs=1), bbox=bb(300, 400))
first = r.chef_id
chk(first == chef_a, "初判(故意)綁錯人", f"綁到 chef {first},正確應為 {chef_b}")

# 之後每一幀它都待在 B 的位置 → 幾何證據持續指向 B
last = first
for k in range(12):
    last = m.on_track_update(20, camera_id="cam2", frame_id=2 + k, t_sec=0.03 * (2 + k),
                             bbox=bb(900, 400), embedding=emb(1, obs=2 + k))
s = m.resident_stats()
chk(last == chef_b, "**投票把身份翻回正確的那位**",
    f"chef {first} → {last}(正確 {chef_b});改判 {s['revote_switch']} 次")
chk(s["revotes"] >= 10, "票數有累積", f"revotes = {s['revotes']}")

# ── 3. 不該翻的不能翻 ───────────────────────────────────────────────
print("\n【3】不該翻的不能翻")
m = SpatioTemporalIdentityManager(topo(cross_view=CV, revote=RV), fps=30.0)
m.on_new_track(10, camera_id="cam1", frame_id=0, t_sec=0.0, embedding=emb(0), bbox=bb(300, 400))
m.on_new_track(11, camera_id="cam1", frame_id=0, t_sec=0.0, embedding=emb(1), bbox=bb(900, 400))
a = m.track_to_chef[("cam1", 10)]
r = m.on_new_track(20, camera_id="cam2", frame_id=1, t_sec=0.03,
                   embedding=emb(1, obs=1), bbox=bb(300, 400))
# 只給 2 次 update → 票數 < min_votes(3)
for k in range(2):
    last = m.on_track_update(20, camera_id="cam2", frame_id=2 + k, t_sec=0.03 * (2 + k),
                             bbox=bb(900, 400), embedding=emb(1, obs=2 + k))
chk(last == r.chef_id and m.resident_stats()["revote_switch"] == 0,
    "票數不足時保持原判",
    f"2 票 < min_votes=3 —— 實測 53.8% 的 track 只能投 1 票,不可被亂改")

# ── 4. 拿不到 embedding → 不投票,而且看得見 ──────────────────────────
print("\n【4】拿不到 embedding 時的行為")
m = SpatioTemporalIdentityManager(topo(cross_view=CV, revote=RV), fps=30.0)
m.on_new_track(1, camera_id="cam1", frame_id=0, t_sec=0.0, embedding=emb(0), bbox=bb(300, 400))
for k in range(10):
    m.on_track_update(1, camera_id="cam1", frame_id=k + 1, t_sec=0.03 * (k + 1),
                      bbox=bb(300, 400))            # ← 不給 embedding
s = m.resident_stats()
chk(s["revotes"] == 0 and s["revote_no_emb"] == 10,
    "沒有 embedding 就不投票,且計數器看得見",
    f"revotes={s['revotes']} revote_no_emb={s['revote_no_emb']} "
    f"—— 靜默失效是這個專案最在意的那種錯")

# ── 5. 取樣間隔 ────────────────────────────────────────────────────
print("\n【5】取樣間隔(stride_loops)")
m = SpatioTemporalIdentityManager(
    topo(cross_view=CV, revote={**RV, "stride_loops": 8}), fps=30.0)
m.on_new_track(1, camera_id="cam1", frame_id=0, t_sec=0.0, embedding=emb(0), bbox=bb(300, 400))
for k in range(40):
    m.on_track_update(1, camera_id="cam1", frame_id=k + 1, t_sec=0.03 * (k + 1),
                      bbox=bb(300, 400), embedding=emb(0, obs=1))
s = m.resident_stats()
chk(s["revotes"] == 5, "每 8 次 update 才投一票",
    f"40 次 update → {s['revotes']} 票(預期 5)"
    f"—— 間隔取自 measure_evidence_decorrelation.py 的實測,不是猜的")

# ── 6. 記憶體有界 ──────────────────────────────────────────────────
print("\n【6】記憶體有界性")
m = SpatioTemporalIdentityManager(topo(cross_view=CV, revote=RV), fps=30.0)
m.on_new_track(1, camera_id="cam1", frame_id=0, t_sec=0.0, embedding=emb(0), bbox=bb(300, 400))
for k in range(20):
    m.on_track_update(1, camera_id="cam1", frame_id=k + 1, t_sec=0.03 * (k + 1),
                      bbox=bb(300, 400), embedding=emb(0, obs=1))
mid = m.resident_stats()["_votes"]
m.on_track_lost(1, camera_id="cam1", frame_id=30, t_sec=1.0, bbox=bb(300, 400))
m.on_track_removed(1, camera_id="cam1", frame_id=31, t_sec=1.03, bbox=bb(300, 400))
end = m.resident_stats()["_votes"]
chk(mid == 1 and end == 0, "track 移除後投票紀錄清乾淨",
    f"存活中 {mid} → 移除後 {end}")
chk(len(m._votes) == 0 and len(m._vote_calls) == 0,
    "_votes 與 _vote_calls 都清了",
    "不清的話會隨累計人次單調成長,違反規格的記憶體有界性")

print("\n" + "─" * 74)
if FAIL:
    print(f"❌ {len(FAIL)} 項未通過:{FAIL}")
    sys.exit(1)
print("✅ 全部通過。⚠ 但這只驗證『行為是否如設計』——")
print("   『投票有沒有提升 IDF1』要在 CHIRLA 評估集上跑,見預先登記。")
sys.exit(0)
