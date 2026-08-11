"""M5 身份管理合成驗證(免真實 Re-ID 模型、免資料,現在就能跑)。

用顏色直方圖 embedder + 合成的「不同衣色人物」crop,驗證 chef_id 綁定邏輯:
  S1 同一人離開再現(TTL 內)      → 沿用同一 chef_id
  S2 不同人(不同衣色)           → 開新 chef_id
  S3 離開太久(超過 TTL)再現     → 開新 chef_id(舊身份已永久移除)
  S4 跨鏡頭同時出現(同一人兩鏡頭)→ 綁到同一 chef_id、該 chef 有兩個 track

任何一項失敗 → exit 1。
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from m5_reid import ColorHistogramEmbedder, IdentityManager   # noqa: E402

RED = (40, 40, 200)
BLUE = (200, 40, 40)


def crop(bgr, seed=0):
    """合成一張某衣色的人物 crop(加一點雜訊,不完全相同但同色)。"""
    rng = np.random.RandomState(seed)
    img = np.full((64, 32, 3), bgr, dtype=np.int16) + rng.randint(-8, 8, (64, 32, 3))
    return np.clip(img, 0, 255).astype(np.uint8)


def mgr(ttl=100, thr=0.5):
    return IdentityManager(similarity_threshold=thr, recently_disappeared_ttl=ttl,
                           embedding_ema=0.5, embedder=ColorHistogramEmbedder())


def check(name, cond, extra=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f" — {extra}" if extra else ""))
    return cond


def s1_reappear_within_ttl():
    print("S1 同一人離開再現(TTL 內)")
    m = mgr(ttl=100)
    r1 = m.on_new_track(1, crop(RED, 1), frame_id=0)
    m.on_track_lost(1, 10)
    r2 = m.on_new_track(2, crop(RED, 2), frame_id=30)
    ok = check("再現綁回同一 chef_id", r1.chef_id == r2.chef_id and r2.matched,
               f"c1={r1.chef_id} c2={r2.chef_id} sim={r2.similarity:.3f}")
    ok &= check("最終只有 1 位廚師", m.stats()["total_chefs"] == 1, str(m.stats()))
    return ok


def s2_different_person():
    print("S2 不同人(不同衣色)")
    m = mgr()
    r1 = m.on_new_track(1, crop(RED, 1), frame_id=0)
    r2 = m.on_new_track(2, crop(BLUE, 1), frame_id=1)
    return check("不同衣色 → 不同 chef_id", r1.chef_id != r2.chef_id and not r2.matched,
                 f"c1={r1.chef_id} c2={r2.chef_id} sim={r2.similarity:.3f}")


def s3_reappear_after_ttl():
    print("S3 離開太久(> TTL)再現")
    m = mgr(ttl=20)
    r1 = m.on_new_track(1, crop(RED, 1), frame_id=0)
    m.on_track_lost(1, 5)
    r2 = m.on_new_track(2, crop(RED, 2), frame_id=5 + 20 + 10)   # 超過 TTL
    return check("舊身份已過期 → 開新 chef_id", r1.chef_id != r2.chef_id and not r2.matched,
                 f"c1={r1.chef_id} c2={r2.chef_id}")


def s4_cross_camera():
    print("S4 跨鏡頭同時出現(同一人兩鏡頭)")
    m = mgr()
    r1 = m.on_new_track(1, crop(RED, 1), frame_id=0)     # cam1
    r2 = m.on_new_track(2, crop(RED, 3), frame_id=1)     # cam2,同人、A 仍活躍
    ok = check("第二鏡頭綁到同一 chef", r1.chef_id == r2.chef_id and r2.matched,
               f"c1={r1.chef_id} c2={r2.chef_id}")
    ok &= check("該 chef 綁定兩個 track", set(m.active[r1.chef_id].track_ids) == {1, 2},
                str(m.active[r1.chef_id].track_ids))
    return ok


def main():
    results = [s1_reappear_within_ttl(), s2_different_person(),
               s3_reappear_after_ttl(), s4_cross_camera()]
    print()
    if all(results):
        print("[ALL PASS] 全部通過")
        sys.exit(0)
    print("[FAILED] 有失敗項")
    sys.exit(1)


if __name__ == "__main__":
    main()
