"""在 Market-1501 上驗證 Re-ID 可行性:標準 Rank-1/mAP + 我們 M5 的 chef_id 綁定正確率。

用途:比較「可商用的 DINOv2(Apache-2.0)」vs「Re-ID 專用的 OSNet(研究權重、僅對照)」,
     判斷通用可商用模型夠不夠用。⚠ Market-1501 與 OSNet 權重皆研究限定,僅供驗證、不可出貨。

metric / binding 為純函式(吃特徵陣列),可用合成特徵單測,不需 torch。
抽特徵需 torch(dinov2)或 torchreid(osnet)。

用法:
  python scripts/reid_eval_market1501.py --data data/market1501 --embedder dinov2
  python scripts/reid_eval_market1501.py --data data/market1501 --embedder osnet --model-name osnet_x1_0
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

_FN = re.compile(r"(-?\d+)_c(\d+)")     # 0001_c1s1_... → pid=1, cam=1


def parse_market1501(split_dir):
    """回傳 [(path, pid, cam)];跳過 junk(pid == -1)。"""
    out = []
    for p in sorted(Path(split_dir).glob("*.jpg")):
        m = _FN.match(p.name)
        if not m:
            continue
        pid, cam = int(m.group(1)), int(m.group(2))
        if pid == -1:
            continue
        out.append((str(p), pid, cam))
    return out


# ---------- 純函式:可用合成特徵單測 ----------

def evaluate_cmc_map(qf, q_pid, q_cam, gf, g_pid, g_cam, topk=(1, 5)):
    """標準 Market-1501 協定:排除同 pid+同 cam 的 gallery。qf/gf 需 L2-normalized。"""
    q_pid, q_cam, g_pid, g_cam = map(np.asarray, (q_pid, q_cam, g_pid, g_cam))
    sims = qf @ gf.T                                     # 餘弦(已正規化)
    all_cmc, all_ap, valid = [], [], 0
    max_rank = max(topk)
    for i in range(qf.shape[0]):
        order = np.argsort(-sims[i])
        keep = ~((g_pid[order] == q_pid[i]) & (g_cam[order] == q_cam[i]))
        matches = (g_pid[order] == q_pid[i])[keep].astype(np.int32)
        if not matches.any():
            continue
        valid += 1
        cmc = matches.cumsum()
        cmc[cmc > 1] = 1
        all_cmc.append(cmc[:max_rank])
        num_rel = matches.sum()
        tmp = matches.cumsum()
        prec = tmp / (np.arange(len(tmp)) + 1.0)
        all_ap.append((prec * matches).sum() / num_rel)
    cmc = np.asarray(all_cmc).mean(0)
    res = {f"rank{k}": round(float(cmc[k - 1]), 4) for k in topk}
    res["mAP"] = round(float(np.mean(all_ap)), 4)
    res["num_query_valid"] = valid
    return res


def gallery_reps(gf, g_pid):
    """每個 pid 的代表特徵 = 該 pid gallery 特徵平均(再 L2-normalized)。回傳 (reps[N,D], pids[N])。"""
    from m5_reid.embedder import l2norm
    pids = sorted(set(int(x) for x in g_pid))
    g_pid = np.asarray(g_pid)
    reps = np.stack([l2norm(gf[g_pid == pid].mean(0)) for pid in pids])
    return reps, np.asarray(pids)


def binding_sweep(qf, q_pid, reps, rep_pids, thresholds):
    """模擬 IdentityManager 決策(比對最相似的已知廚師,cosine>=門檻才綁)。

    closed-set(query pid 均在 gallery):
      correct    = 綁對(pred pid==query pid)
      false_merge= 綁錯(綁到別人)→ 身份污染
      reject     = 未達門檻(漏綁,此情境算失敗)
    回傳每個門檻的 {thr, accuracy, false_merge, reject}。
    """
    q_pid = np.asarray(q_pid)
    sims = qf @ reps.T
    best = sims.argmax(1)
    best_sim = sims[np.arange(len(qf)), best]
    pred_pid = rep_pids[best]
    rows = []
    n = len(q_pid)
    for thr in thresholds:
        accept = best_sim >= thr
        correct = int((accept & (pred_pid == q_pid)).sum())
        false_merge = int((accept & (pred_pid != q_pid)).sum())
        reject = int((~accept).sum())
        rows.append({"thr": round(float(thr), 2),
                     "accuracy": round(correct / n, 4),
                     "false_merge": round(false_merge / n, 4),
                     "reject": round(reject / n, 4)})
    return rows


# ---------- 抽特徵(需 torch / torchreid) ----------

def build_embedder(name, model_name, device):
    if name == "dinov2":
        from m5_reid.dino_embedder import DINOv2Embedder
        return DINOv2Embedder(model_name=model_name or "dinov2_vits14", device=device)
    if name == "osnet":
        from m5_reid.osnet_embedder import OSNetEmbedder
        return OSNetEmbedder(model_name=model_name or "osnet_x1_0", device=device)
    raise ValueError(name)


def extract_features(embedder, samples, batch=32):
    import cv2
    feats = []
    for i in range(0, len(samples), batch):
        crops = [cv2.imread(s[0]) for s in samples[i:i + batch]]
        crops = [c for c in crops if c is not None]
        feats.append(embedder.extract_batch(crops))
        print(f"  抽特徵 {min(i + batch, len(samples))}/{len(samples)}", end="\r")
    print()
    return np.concatenate(feats, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data" / "market1501"),
                    help="Market-1501 根目錄(含 query/、bounding_box_test/)")
    ap.add_argument("--embedder", required=True, choices=["dinov2", "osnet"])
    ap.add_argument("--model-name", default="")
    ap.add_argument("--device", default=None)
    ap.add_argument("--max-ids", type=int, default=0, help=">0 只取前 N 個 pid 加速")
    ap.add_argument("--out", default=str(ROOT / "reid_eval_results.json"))
    args = ap.parse_args()

    data = Path(args.data)
    q = parse_market1501(data / "query")
    g = parse_market1501(data / "bounding_box_test")
    if args.max_ids:
        keep = set(sorted(set(pid for _, pid, _ in g))[:args.max_ids])
        q = [s for s in q if s[1] in keep]
        g = [s for s in g if s[1] in keep]
    print(f"query {len(q)} 張、gallery {len(g)} 張、pid {len(set(s[1] for s in g))} 人")

    emb = build_embedder(args.embedder, args.model_name, args.device)
    print(f"embedder = {args.embedder}({args.model_name or 'default'}), dim={emb.dim}")
    qf = extract_features(emb, q)
    gf = extract_features(emb, g)
    q_pid = [s[1] for s in q]; q_cam = [s[2] for s in q]
    g_pid = [s[1] for s in g]; g_cam = [s[2] for s in g]

    cmc = evaluate_cmc_map(qf, q_pid, q_cam, gf, g_pid, g_cam)
    reps, rep_pids = gallery_reps(gf, g_pid)
    sweep = binding_sweep(qf, q_pid, reps, rep_pids,
                          [round(0.05 * i, 2) for i in range(4, 19)])   # 0.20..0.90
    best = max(sweep, key=lambda r: r["accuracy"])

    print("\n=== 標準 Re-ID 指標 ===")
    print(cmc)
    print("\n=== chef_id 綁定(掃門檻)===")
    print(f"{'門檻':>6}{'綁定正確':>10}{'綁錯(污染)':>12}{'漏綁':>8}")
    for r in sweep:
        print(f"{r['thr']:>6}{r['accuracy']:>10}{r['false_merge']:>12}{r['reject']:>8}")
    print(f"\n最佳門檻 = {best['thr']}(綁定正確 {best['accuracy']})")

    out = {"embedder": args.embedder, "model_name": args.model_name or "default",
           "cmc_map": cmc, "binding_sweep": sweep, "best_binding": best}
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已輸出 {args.out}")


if __name__ == "__main__":
    main()
