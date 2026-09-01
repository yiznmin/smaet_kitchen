"""Re-ID 指標的合成驗證 —— 用已知答案的假特徵檢查指標實作本身有沒有錯。

⚠ `docs/M5_可行性驗證與模型選型.md:58` 聲稱「metric/binding 為純函式,已用合成特徵
  單測通過(完美可分離 → rank1=mAP=1.0)」,但**那份程式碼從未進版控**。
  接下來要拿這些函式去評估 CHIRLA 訓練出來的模型,指標本身錯了整批數字都白算,
  所以先把這個測試補回來。

沿用 repo 的 verify_*.py 風格:直跑、印 PASS/FAIL、失敗 exit 1,不引入 pytest。

用法:python scripts/verify_reid_metrics.py
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import reid_eval_market1501 as rm          # noqa: E402
from m5_reid.embedder import l2norm        # noqa: E402

FAILED = []


def check(name, cond, detail=""):
    print(f"  {'✅' if cond else '❌'} {name}" + (f"  —— {detail}" if detail else ""))
    if not cond:
        FAILED.append(name)


def onehot(pids, n=None):
    n = n or int(max(pids)) + 1
    return np.eye(n, dtype=np.float32)[np.asarray(pids)]


def s1_perfect():
    """完美可分離:同身份特徵完全相同、不同身份正交 → rank1 = mAP = 1.0。"""
    print("\nS1 完美可分離")
    pid = np.array([0, 0, 1, 1, 2, 2, 3, 3])
    cam = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    f = onehot(pid)
    r = rm.evaluate_cmc_map(f, pid, cam, f, pid, cam, topk=(1, 5))
    check("rank1 = 1.0", r["rank1"] == 1.0, f"實得 {r['rank1']}")
    check("mAP = 1.0", r["mAP"] == 1.0, f"實得 {r['mAP']}")
    check("所有 query 都有效", r["num_query_valid"] == len(pid), str(r))


def s2_random():
    """完全隨機特徵 → rank1 應該接近隨機水準,而不是 0 或 1。

    這條擋的是「指標永遠回 1.0」或「永遠回 0」這類實作錯誤 —— 只測完美可分離
    抓不到那種錯。
    """
    print("\nS2 隨機特徵(擋『指標永遠給滿分』的實作錯誤)")
    rng = np.random.RandomState(0)
    n_id, per = 20, 4
    pid = np.repeat(np.arange(n_id), per)
    cam = np.tile(np.arange(per), n_id)
    f = l2norm_rows(rng.randn(len(pid), 128).astype(np.float32))
    r = rm.evaluate_cmc_map(f, pid, cam, f, pid, cam, topk=(1, 5))
    # gallery 排除同 pid+同 cam 後,每個 query 仍有 3 個同身份正樣本、76 個負樣本
    chance = 3 / (3 + (n_id - 1) * per)
    check("rank1 明顯低於 1.0", r["rank1"] < 0.5, f"實得 {r['rank1']:.3f}")
    check("rank1 落在隨機水準的量級", r["rank1"] < chance * 6,
          f"實得 {r['rank1']:.3f},隨機約 {chance:.3f}")
    check("mAP 不為 0", r["mAP"] > 0, f"實得 {r['mAP']:.3f}")


def s3_same_cam_excluded():
    """Market-1501 協定要排除「同身份且同相機」的 gallery。

    若沒排除,單相機內的自我匹配會讓分數虛高 —— 這是 Re-ID 評估最經典的錯誤。
    構造:同身份在同相機的特徵完全相同,跨相機則完全正交(即跨相機毫無資訊)。
    正確實作應該得到很低的 rank1;沒排除的話會是 1.0。
    """
    print("\nS3 排除同身份同相機(擋『分數虛高』的協定錯誤)")
    pid = np.array([0, 0, 1, 1, 2, 2])
    cam = np.array([0, 1, 0, 1, 0, 1])
    # 每個 (pid, cam) 一個獨立方向 → 同 pid 跨 cam 的相似度是 0
    f = onehot(np.arange(len(pid)))
    r = rm.evaluate_cmc_map(f, pid, cam, f, pid, cam, topk=(1,))
    check("跨相機無資訊時 rank1 不應是 1.0", r["rank1"] < 1.0,
          f"實得 {r['rank1']:.3f} —— 若為 1.0 表示沒排除同 pid+同 cam")


def s4_gallery_reps():
    """gallery_reps 應該把每個身份的特徵平均後重新 L2-normalize。"""
    print("\nS4 gallery_reps")
    pid = np.array([0, 0, 1, 1])
    f = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]], dtype=np.float32)
    reps, rpid = rm.gallery_reps(f, pid)
    check("每個身份一個代表", len(rpid) == 2, f"實得 {len(rpid)}")
    norms = np.linalg.norm(reps, axis=1)
    check("代表特徵已 L2-normalize", np.allclose(norms, 1.0, atol=1e-5), str(norms))


def s5_binding_sweep():
    """binding_sweep 的三個率必須加總為 1,且門檻拉滿時應全部 reject。"""
    print("\nS5 binding_sweep")
    pid = np.array([0, 0, 1, 1, 2, 2])
    f = onehot(pid)
    reps, rpid = rm.gallery_reps(f, pid)
    rows = rm.binding_sweep(f, pid, reps, rpid, [0.5, 0.99, 1.01])
    for r in rows:
        tot = r["accuracy"] + r["false_merge"] + r["reject"]
        check(f"門檻 {r['thr']}:三個率加總 = 1", abs(tot - 1.0) < 1e-6, f"實得 {tot}")
    check("門檻 0.5 時完美特徵應全綁對", rows[0]["accuracy"] == 1.0, str(rows[0]))
    check("門檻 > 1 時應全部 reject", rows[2]["reject"] == 1.0, str(rows[2]))


def s6_l2norm():
    """embedder 契約:extract 的輸出必須是 L2-normalized。順帶測零向量安全。"""
    print("\nS6 l2norm 契約")
    v = l2norm(np.array([3.0, 4.0], dtype=np.float32))
    check("正常向量正規化", abs(np.linalg.norm(v) - 1.0) < 1e-6, str(v))
    z = l2norm(np.zeros(4, dtype=np.float32))
    check("零向量不炸且不含 NaN", np.isfinite(z).all(), str(z))


def l2norm_rows(m):
    return m / np.maximum(np.linalg.norm(m, axis=1, keepdims=True), 1e-12)


def main():
    print("=" * 66)
    print("Re-ID 指標合成驗證")
    print("=" * 66)
    for fn in (s1_perfect, s2_random, s3_same_cam_excluded,
               s4_gallery_reps, s5_binding_sweep, s6_l2norm):
        fn()
    print("\n" + "=" * 66)
    if FAILED:
        print(f"[FAIL] {len(FAILED)} 項未通過:{FAILED}")
        return 1
    print("[ALL PASS] 指標實作可信,可以拿去評估 CHIRLA 模型")
    return 0


if __name__ == "__main__":
    sys.exit(main())
