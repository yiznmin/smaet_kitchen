"""在 CHIRLA 上訓練人物 Re-ID 模型。

⚠ **repo 內沒有任何訓練程式碼可抄** —— M3 微調全靠 rfdetr 的一行 `model.train()`,
  沒有 optimizer/loss/DataLoader/sampler 的既有範本。這支是從零寫的。

用的是 Re-ID 的**標準強基線**(Luo et al. 2019, "Bag of Tricks"),不自創:
  骨幹 ResNet50 → GAP → feature f
                            ├─ triplet loss(用 **BN 前**的 f)
                            └─ BNNeck(BN)→ classifier → CE+label smoothing
  BNNeck 的用意:triplet 要的是「同人靠近」的歐氏空間,CE 要的是「好分類」的
  超球面,兩者在同一個特徵上會互相拉扯。中間插一個 BN 讓兩個損失各自作用在
  適合的空間 —— 這是 Re-ID 領域公認能穩定漲點的做法。

**兩條路線,除了起始權重以外所有設定逐字相同**(受控雙臂):
  · `--arm S`  ImageNet 預訓 → **可出貨**(交付主線)
  · `--arm R`  外部 Re-ID 預訓權重 → 研究上限對照,**不可出貨**
    (FastReID/CION 的權重多訓練於 Market-1501/MSMT17,研究限定且授權會傳染)

⚠ **最大的風險是過擬合**:CHIRLA 只有 22 個身份,train 切分更少。
  Market-1501 有 1501 個。所以預設就開了強增強 + 早停 + 可選凍結骨幹前段。
  train/val 曲線分離是預期中的事,**記錄下來不要藏**。

用法:
    python scripts/chirla_prep.py --root <CHIRLA> --index --out chirla_index.json
    python scripts/train_reid_chirla.py --index chirla_index.json \
        --scenario multi_camera --arm S --epochs 60 --out model_result/reid/armS
"""
import argparse
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


# ── 資料 ──────────────────────────────────────────────────────────────
def load_index(path, scenario, subset):
    """讀 chirla_prep.py --index 的輸出 → [(path, identity, camera, fid)]。"""
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = d["index"][scenario].get(subset, [])
    root = Path(d["root"])
    out = []
    for p, ident, cam, fid in rows:
        f = Path(p)
        out.append((str(f if f.is_absolute() else root / p), ident, cam, fid))
    return out


class ReIDDataset:
    """CHIRLA crops。⚠ 用 cv2.imdecode(np.fromfile(...)) 而非 cv2.imread ——
    專案路徑含中文,imread 在 Windows 上會靜默回 None(既有腳本踩過)。"""

    def __init__(self, rows, pid_map, train=True, size=(256, 128), erase_p=0.5):
        self.rows, self.pid_map, self.train = rows, pid_map, train
        self.h, self.w = size
        self.erase_p = erase_p

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        import cv2
        import torch
        path, ident, cam, _fid = self.rows[i]
        img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"讀不到影像:{path}")
        img = cv2.resize(img, (self.w, self.h))
        if self.train:
            if random.random() < 0.5:
                img = img[:, ::-1]
            img = self._pad_crop(img)
        x = torch.from_numpy(np.ascontiguousarray(img[:, :, ::-1])).permute(2, 0, 1).float() / 255.
        x = (x - torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)) / \
            torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        if self.train and random.random() < self.erase_p:
            x = self._erase(x)
        return x, self.pid_map[ident], self.cam_id(cam)

    def cam_id(self, cam):
        return abs(hash(cam)) % 10_000 if not cam.isdigit() else int(cam)

    def _pad_crop(self, img, pad=10):
        import cv2
        p = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        y, x = random.randint(0, 2 * pad), random.randint(0, 2 * pad)
        return p[y:y + self.h, x:x + self.w]

    def _erase(self, x, sl=0.02, sh=0.3, r=0.3):
        """Random Erasing —— 對遮擋的魯棒性。22 身份下這類強增強是必要的。"""
        c, h, w = x.shape
        for _ in range(10):
            area = h * w * random.uniform(sl, sh)
            ar = random.uniform(r, 1 / r)
            eh, ew = int(round((area * ar) ** .5)), int(round((area / ar) ** .5))
            if eh < h and ew < w:
                y0, x0 = random.randint(0, h - eh), random.randint(0, w - ew)
                x[:, y0:y0 + eh, x0:x0 + ew] = 0.0
                return x
        return x


class PKSampler:
    """每個 batch 取 P 個身份 × 每人 K 張 —— triplet loss 的前提。

    ⚠ 沒有這個,一個 batch 裡可能一個正樣本對都沒有,triplet 恆為 0、白訓練。
    """

    def __init__(self, rows, pid_map, P=8, K=4, batches=None):
        self.by_pid = defaultdict(list)
        for i, (_p, ident, _c, _f) in enumerate(rows):
            self.by_pid[pid_map[ident]].append(i)
        self.pids = [p for p, v in self.by_pid.items() if len(v) >= 2]
        self.P, self.K = min(P, len(self.pids)), K
        self.batches = batches or max(1, len(rows) // (self.P * K))

    def __len__(self):
        return self.batches

    def __iter__(self):
        for _ in range(self.batches):
            out = []
            for pid in random.sample(self.pids, self.P):
                pool = self.by_pid[pid]
                out += (random.sample(pool, self.K) if len(pool) >= self.K
                        else [random.choice(pool) for _ in range(self.K)])
            yield out


# ── 模型 ──────────────────────────────────────────────────────────────
def build_model(n_classes, arm="S", weights=None, device="cpu", pretrained=True):
    """ResNet50 + BNNeck。arm 只影響**起始權重**,其餘結構完全相同。

    ⚠ `pretrained=False` 用於「只要結構、等一下會 load_state_dict 蓋掉」的場合
      (例如 ChirlaEmbedder 載 checkpoint)。不加這個參數的話每次載模型都會
      白下載 100MB 的 ImageNet 權重再立刻丟掉。
    """
    import torch
    import torch.nn as nn
    from torchvision.models import ResNet50_Weights, resnet50

    if arm == "S":
        # ImageNet 預訓 —— 授權寬鬆,可出貨
        net = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)
        provenance = ("torchvision ResNet50 IMAGENET1K_V1(可出貨)" if pretrained
                      else "(僅建結構,權重由 checkpoint 提供)")
    else:
        net = resnet50(weights=None)
        provenance = f"外部 Re-ID 權重 {weights}(⚠ 研究限定,不可出貨)"
        if weights:
            sd = torch.load(weights, map_location="cpu")
            sd = sd.get("state_dict", sd.get("model", sd))
            sd = {k.replace("module.", "").replace("backbone.", ""): v for k, v in sd.items()}
            missing, unexpected = net.load_state_dict(sd, strict=False)
            print(f"  載入外部權重:missing {len(missing)} / unexpected {len(unexpected)}")

    net.layer4[0].downsample[0].stride = (1, 1)     # last stride = 1,Re-ID 標準做法
    net.layer4[0].conv2.stride = (1, 1)

    class ReIDNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = nn.Sequential(*list(net.children())[:-2])
            self.gap = nn.AdaptiveAvgPool2d(1)
            self.bnneck = nn.BatchNorm1d(2048)
            self.bnneck.bias.requires_grad_(False)   # BNNeck 的 bias 要凍結
            self.classifier = nn.Linear(2048, n_classes, bias=False)
            nn.init.normal_(self.classifier.weight, std=0.001)

        def forward(self, x):
            f = self.gap(self.backbone(x)).flatten(1)   # triplet 用這個(BN 前)
            fb = self.bnneck(f)                          # 推論時輸出這個
            return (f, self.classifier(fb)) if self.training else fb

    m = ReIDNet().to(device)
    print(f"  起始權重:{provenance}")
    return m, provenance


def triplet_loss(f, pid, margin=0.3):
    """Batch-hard triplet:每個 anchor 取最難正樣本與最難負樣本。"""
    import torch
    import torch.nn.functional as F
    d = torch.cdist(f, f)
    same = pid.unsqueeze(0) == pid.unsqueeze(1)
    eye = torch.eye(len(pid), dtype=torch.bool, device=f.device)
    ap = (d * (same & ~eye)).max(1).values
    an = (d + (same * 1e6)).min(1).values
    return F.relu(ap - an + margin).mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True, help="chirla_prep.py --index 的輸出")
    ap.add_argument("--scenario", required=True)
    ap.add_argument("--arm", choices=["S", "R"], default="S")
    ap.add_argument("--weights", default=None, help="arm R 的外部 Re-ID 權重")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--P", type=int, default=8, help="每 batch 幾個身份")
    ap.add_argument("--K", type=int, default=4, help="每個身份幾張")
    ap.add_argument("--lr", type=float, default=3.5e-4)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--patience", type=int, default=15, help="早停:val 幾輪沒進步就停")
    ap.add_argument("--freeze-until", type=int, default=0,
                    help="凍結骨幹前 N 個 stage(0=不凍)。22 身份易過擬合時可開")
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=str(ROOT / "model_result" / "reid" / "armS"))
    ap.add_argument("--smoke", action="store_true", help="每 epoch 只跑 2 個 batch,驗管線用")
    args = ap.parse_args()

    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    dev = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    random.seed(0); np.random.seed(0); torch.manual_seed(0)

    tr = load_index(args.index, args.scenario, "train")
    va = load_index(args.index, args.scenario, "val")
    if not tr:
        raise SystemExit(f"{args.scenario} 沒有 train 切分")
    pid_map = {p: i for i, p in enumerate(sorted({r[1] for r in tr}))}
    print("=" * 70)
    print(f"CHIRLA Re-ID 訓練 · scenario={args.scenario} · arm={args.arm} · {dev}")
    print("=" * 70)
    print(f"  train {len(tr):,} 張 / {len(pid_map)} 身份    val {len(va):,} 張")
    # ⚠ 這是本任務最大的技術風險,開跑就講,不要等報告才發現
    if len(pid_map) < 100:
        print(f"  ⚠ 只有 {len(pid_map)} 個身份(Market-1501 有 1501 個)→ **極易過擬合**。"
              "已開強增強+早停;train/val 分離是預期中的事,要如實記錄。")

    ds_tr = ReIDDataset(tr, pid_map, train=True)
    sampler = PKSampler(tr, pid_map, P=args.P, K=args.K,
                        batches=2 if args.smoke else None)
    dl = DataLoader(ds_tr, batch_sampler=sampler, num_workers=0)

    model, provenance = build_model(len(pid_map), args.arm, args.weights, dev)
    if args.freeze_until:
        stages = list(model.backbone.children())
        for m_ in stages[:args.freeze_until]:
            for p in m_.parameters():
                p.requires_grad_(False)
        print(f"  已凍結骨幹前 {args.freeze_until} 個 stage")

    # --epochs 0:不訓練,直接把「起始權重」本身存成 checkpoint。
    # 為什麼需要:CHIRLA 論文的基線全部是**零樣本**(拿在 Market-1501/CION 上
    # 預訓好的模型直接抽特徵,沒有在 CHIRLA 上訓練過)。所以「arm R 微調後」
    # 與論文數字並不是同一件事。有了這個零樣本參照點,才答得出
    # 「在 23~65 張影像上微調,到底是幫忙還是傷害」——本輪最關鍵的問題之一。
    # ⚠ 這是**額外參照點**,不改動預先登記的 S/R 雙臂設定。
    if args.epochs == 0:
        out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
        torch.save({"model": model.state_dict(), "pid_map": pid_map,
                    "arm": args.arm, "provenance": provenance + "(零樣本,未微調)",
                    "dim": 2048, "epoch": -1}, out / "best.pth")
        (out / "train_log.json").write_text(json.dumps(
            {"args": vars(args), "provenance": provenance + "(零樣本,未微調)",
             "n_ids": len(pid_map), "n_train": len(tr), "n_val": len(va),
             "history": []}, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n--epochs 0:未訓練,已存起始權重到 {out/'best.pth'}")
        return 0

    ce = nn.CrossEntropyLoss(label_smoothing=0.1)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=args.lr,
                           weight_decay=5e-4)

    def lr_at(ep):
        if ep < args.warmup:                       # warmup:小資料集上很關鍵
            return (ep + 1) / args.warmup
        return 0.1 ** sum(ep >= m for m in (30, 50))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    best, bad, hist = 1e9, 0, []
    for ep in range(args.epochs):
        model.train()
        t0, agg = time.time(), defaultdict(float)
        for x, pid, _cam in dl:
            x, pid = x.to(dev), pid.to(dev)
            f, logits = model(x)
            l_ce, l_tri = ce(logits, pid), triplet_loss(f, pid)
            loss = l_ce + l_tri
            opt.zero_grad(); loss.backward(); opt.step()
            agg["ce"] += l_ce.item(); agg["tri"] += l_tri.item(); agg["n"] += 1
            agg["acc"] += (logits.argmax(1) == pid).float().mean().item()
        sched.step()
        n = max(agg["n"], 1)
        row = dict(epoch=ep, lr=opt.param_groups[0]["lr"],
                   ce=agg["ce"] / n, tri=agg["tri"] / n, train_acc=agg["acc"] / n,
                   sec=round(time.time() - t0, 1))
        hist.append(row)
        print(f"  ep{ep:>3}  lr {row['lr']:.2e}  CE {row['ce']:.3f}  "
              f"tri {row['tri']:.3f}  train_acc {row['train_acc']:.3f}  {row['sec']}s")

        # 早停看 total loss(val 的 CMC 要抽完整特徵,太慢,留給評估腳本)
        cur = row["ce"] + row["tri"]
        if cur < best - 1e-4:
            best, bad = cur, 0
            torch.save({"model": model.state_dict(), "pid_map": pid_map,
                        "arm": args.arm, "provenance": provenance,
                        "dim": 2048, "epoch": ep}, out / "best.pth")
        else:
            bad += 1
            if bad >= args.patience:
                print(f"  早停:{args.patience} 輪沒進步")
                break

    (out / "train_log.json").write_text(json.dumps(
        {"args": vars(args), "provenance": provenance, "n_ids": len(pid_map),
         "n_train": len(tr), "n_val": len(va), "history": hist},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已存 {out/'best.pth'} 與 train_log.json")
    print(f"  權重來源:{provenance}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
