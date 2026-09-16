"""身份訊號(cue)的行為驗證 —— 不需要 CHIRLA 資料。

## 這支要擋住什麼

1. **關閉時必須逐位不變。** cue 加在 `_score_candidates` 上,那是所有既有結果都會走的路徑。
2. **推導要對。** p=1 → 相同讀數是最強證據;p=1/K → 恆為 0(讀數等於亂猜)。
   這兩個端點錯了,整個上限實驗量到的數字就沒有意義。
3. **合成訊號要可重現且真的是那個準確率。** 不可重現 → 同一格跑兩次結果不同;
   實際準確率與登記值不符 → 量到的是別的 p。
4. **「持續重新確認能把出生時的錯改回來」要真的發生。** 這是整輪實驗的主張本身:
   出生時綁錯了,之後每次讀到訊號都投一票,票夠了就要改回正確的人。

⚠ 這支**不回答「上限是多少」** —— 那要在 CHIRLA 上跑完整評估集。

用法:
    python scripts/verify_m5_cue.py
"""
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from m5_reid.cue import CueLR, reading_key, synth_token          # noqa: E402
from m5_reid.identity_st import SpatioTemporalIdentityManager    # noqa: E402
from m5_reid.spatiotemporal import CameraTopology                # noqa: E402

FAIL = []


def chk(ok, name, detail=""):
    print(f"  {'✅' if ok else '❌'} {name}")
    if detail:
        print(f"       {detail}")
    if not ok:
        FAIL.append(name)


class _Zero:
    """與 runner 的 `--embedder none` 相同:全零向量 → 外觀項是常數,不參與區分。

    本輪要量的就是「在沒有外觀證據的條件下,身份訊號能做到什麼」,
    所以這裡刻意不給可分辨的外觀,與評估集的設定一致。
    """
    dim = 64

    def extract(self, crop):
        return np.zeros(self.dim, dtype=np.float32)


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


def mgr(t):
    return SpatioTemporalIdentityManager(t, embedder=_Zero(), fps=30.0)


def bb(x, y, w=50, h=120):
    return (x - w / 2.0, y - h, x + w / 2.0, y)


CUE = {"enabled": True, "accuracy": 0.9, "n_ids": 10, "clip": 8.0}
RV = {"enabled": True, "stride_loops": 1, "window": 9, "min_votes": 3, "switch_margin": 2}

print("=" * 74)
print("身份訊號(cue)—— 行為驗證(不需 CHIRLA 資料)")
print("=" * 74)

# ── 1. 關閉時 ──────────────────────────────────────────────────────
print("\n【1】關閉時的回歸")
chk(topo().cue_lr is None, "cue 預設關閉", f"cue_lr = {topo().cue_lr}")

m_off = mgr(topo())
m_off.on_new_track(1, camera_id="cam2", frame_id=0, t_sec=0.0, bbox=bb(300, 400))
m_off.on_new_track(2, camera_id="cam2", frame_id=0, t_sec=0.0, bbox=bb(900, 400))
zero = np.zeros(64, dtype=np.float32)
base_scores = sorted(s for s, _c, _a in m_off._score_candidates(
    "cam1", 1.0, zero, bb(300, 400), None, None, None))

m_tok = mgr(topo())
m_tok.on_new_track(1, camera_id="cam2", frame_id=0, t_sec=0.0, bbox=bb(300, 400), cue_token=7)
m_tok.on_new_track(2, camera_id="cam2", frame_id=0, t_sec=0.0, bbox=bb(900, 400), cue_token=8)
tok_scores = sorted(s for s, _c, _a in m_tok._score_candidates(
    "cam1", 1.0, zero, bb(300, 400), None, None, None, cue_token=7))
chk(base_scores == tok_scores,
    "關閉時就算給了 cue_token 也完全不影響分數", f"{base_scores} vs {tok_scores}")

# ── 2. 推導的兩個端點 ──────────────────────────────────────────────
print("\n【2】CueLR 的推導")
perfect = CueLR(1.0, 10, clip=8.0)
chk(perfect.llr(3, 3) == 8.0 and perfect.llr(3, 4) == -8.0,
    "p=1:相同 = +clip、不同 = −clip",
    f"{perfect.llr(3, 3):+.2f} / {perfect.llr(3, 4):+.2f}")

chance = CueLR(1.0 / 10, 10)
chk(abs(chance.llr(3, 3)) < 1e-12 and abs(chance.llr(3, 4)) < 1e-12,
    "p=1/K:讀數等於亂猜 → LLR 恆為 0",
    f"{chance.llr(3, 3):+.3e} / {chance.llr(3, 4):+.3e}")

mid = CueLR(0.9, 10)
chk(mid.llr(3, 3) > 0 > mid.llr(3, 4),
    "p=0.9:相同是正證據、不同是負證據", mid.describe())

incr = [CueLR(p, 10).llr(1, 1) for p in (0.2, 0.4, 0.6, 0.8, 0.95)]
chk(all(a < b for a, b in zip(incr, incr[1:])),
    "準確率越高,相同讀數的證據越強", f"{[round(v, 3) for v in incr]}")

probs = CueLR(0.8, 10)
chk(abs(probs.p_same - (0.8 ** 2 + 0.2 ** 2 / 9)) < 1e-12
    and abs(probs.p_diff - (2 * 0.8 * 0.2 / 9 + 8 * 0.2 ** 2 / 81)) < 1e-12,
    "P_same / P_diff 與檔頭的推導式相符",
    f"P_same={probs.p_same:.6f} P_diff={probs.p_diff:.6f}")

# 證據要能壓過重疊路徑的常數必綁規則(3.806 − ln k),否則上限實驗量不到東西
chk(CueLR(0.9, 10).llr(1, 2) < -1.6094,
    "p=0.9 時「讀數不同」足以否決重疊路徑的必綁規則",
    f"{CueLR(0.9, 10).llr(1, 2):.3f} < −1.609(門檻)")

# ── 3. 合成訊號 ────────────────────────────────────────────────────
print("\n【3】合成訊號")
IDS = tuple(range(1, 11))
k1 = reading_key("cam1", 5, 100)
chk(synth_token(3, IDS, key=k1, accuracy=0.8, seed=7)
    == synth_token(3, IDS, key=k1, accuracy=0.8, seed=7),
    "同樣的 (seed, key) 必得同樣的讀數(可重現)")

N = 100_000
for p in (1.0, 0.9, 0.6):
    hit = sum(synth_token(3, IDS, key=reading_key("cam1", 1, i), accuracy=p, seed=11) == 3
              for i in range(N))
    chk(abs(hit / N - p) < 0.005, f"實際讀對率 ≈ 登記的 p={p}",
        f"{hit / N:.4f}(N={N:,})")

wrong = Counter(synth_token(3, IDS, key=reading_key("cam1", 2, i), accuracy=0.5, seed=13)
                for i in range(N))
others = [wrong[i] for i in IDS if i != 3]
chk(wrong[3] > 0 and min(others) > 0 and max(others) / min(others) < 1.2,
    "誤讀大致均勻分布在其他 id 上", f"其他 id 次數 {min(others)}~{max(others)}")

chk(synth_token(None, IDS, key=k1, accuracy=0.9, seed=7) is None,
    "誤偵(沒有真值)→ 讀不到標記,不憑空發明證據")

burst = [synth_token(3, IDS, key=reading_key("cam1", 9, i, burst=30), accuracy=0.5, seed=17)
         for i in range(90)]
chk(len(set(burst[:30])) == 1 and len(set(burst[30:60])) == 1,
    "burst:同一桶內的讀數完全相同(錯誤成群出現)",
    f"三桶分別是 {burst[0]} / {burst[30]} / {burst[60]}")

# ── 4. 出生時就能選對人 ────────────────────────────────────────────
print("\n【4】有訊號時,出生當下就選對人")
m = mgr(topo(cue=CUE))
m.on_new_track(1, camera_id="cam2", frame_id=0, t_sec=0.0, bbox=bb(300, 400), cue_token=1)
m.on_new_track(2, camera_id="cam2", frame_id=0, t_sec=0.0, bbox=bb(900, 400), cue_token=2)
chef_a, chef_b = m.track_to_chef[("cam2", 1)], m.track_to_chef[("cam2", 2)]
r = m.on_new_track(9, camera_id="cam1", frame_id=1, t_sec=0.1, bbox=bb(300, 400), cue_token=2)
chk(r.matched and r.chef_id == chef_b,
    "新 track 的讀數是 B → 綁給 B(而不是任選一個同分的)",
    f"綁到 chef {r.chef_id}(A={chef_a} B={chef_b})")

# ── 5. 出生時綁錯,持續確認要能改回來 ──────────────────────────────
print("\n【5】出生時綁錯 → 持續重新確認把它改回來(本輪主張)")
m = mgr(topo(cue=CUE, revote=RV))
m.on_new_track(1, camera_id="cam2", frame_id=0, t_sec=0.0, bbox=bb(300, 400), cue_token=1)
m.on_new_track(2, camera_id="cam2", frame_id=0, t_sec=0.0, bbox=bb(900, 400), cue_token=2)
chef_a, chef_b = m.track_to_chef[("cam2", 1)], m.track_to_chef[("cam2", 2)]
# 出生當下讀不到標記(cue_token=None)→ 只剩常數規則 → 綁給同分的其中一個
r = m.on_new_track(9, camera_id="cam1", frame_id=1, t_sec=0.1, bbox=bb(300, 400))
born = r.chef_id
chk(r.matched, "出生時沒有訊號,照舊被常數規則綁走", f"綁到 chef {born}")

for i in range(8):                       # 之後每一次都讀到「我是 B」
    m.on_track_update(9, camera_id="cam1", frame_id=2 + i, t_sec=0.2 + 0.1 * i,
                      bbox=bb(300, 400), embedding=zero, cue_token=2)
    m.on_track_update(1, camera_id="cam2", frame_id=2 + i, t_sec=0.2 + 0.1 * i,
                      bbox=bb(300, 400), embedding=zero, cue_token=1)
    m.on_track_update(2, camera_id="cam2", frame_id=2 + i, t_sec=0.2 + 0.1 * i,
                      bbox=bb(900, 400), embedding=zero, cue_token=2)
now = m.track_to_chef[("cam1", 9)]
chk(now == chef_b, "持續確認後改判到正確的人 B",
    f"出生時 {born} → 現在 {now}(A={chef_a} B={chef_b})")
old = m.active.get(chef_a)
chk(m.resident_stats()["revotes"] > 0 and (old is None or ("cam1", 9) not in old.track_ids),
    "舊 chef 的 track_ids 有一併清乾淨(不會污染後續候選)",
    f"revotes={m.resident_stats()['revotes']}")

# 負對照:沒有訊號時同一情境不會改判 —— 證明【5】改判的是訊號,不是別的東西
m2 = mgr(topo(revote=RV))
m2.on_new_track(1, camera_id="cam2", frame_id=0, t_sec=0.0, bbox=bb(300, 400))
m2.on_new_track(2, camera_id="cam2", frame_id=0, t_sec=0.0, bbox=bb(900, 400))
born2 = m2.on_new_track(9, camera_id="cam1", frame_id=1, t_sec=0.1, bbox=bb(300, 400)).chef_id
for i in range(8):
    for tid, cam, x in ((9, "cam1", 300), (1, "cam2", 300), (2, "cam2", 900)):
        m2.on_track_update(tid, camera_id=cam, frame_id=2 + i, t_sec=0.2 + 0.1 * i,
                           bbox=bb(x, 400), embedding=zero)
chk(m2.track_to_chef[("cam1", 9)] == born2,
    "負對照:沒有訊號時同一情境不會改判(改判來自訊號)", f"維持 chef {born2}")

print()
if FAIL:
    print(f"[FAILED] {len(FAIL)} 項失敗:{FAIL}")
    sys.exit(1)
print("[ALL PASS] 全部通過")
