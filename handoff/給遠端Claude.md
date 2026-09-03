# 給遠端 Claude 的交接單

> ## 🛑 2026-09-03:下面的步驟 1~7 **已經全部跑完了,不要重跑**
>
> 結果與結論在 `docs/CHIRLA_預先登記_20260901.md` §8~§11,
> 過程與踩到的坑在 [紀錄/2026-09-03_遠端.md](紀錄/2026-09-03_遠端.md)。**先讀那兩份。**
>
> 一句話:**在 CHIRLA 官方 `_train` 切分(23~65 張、單一實體相機)上訓練,
> 產不出對 M5 有用的可出貨外觀模型。** 預先登記 §5 寫死的「過擬合主導」失敗條件成立,
> §5 也先寫死了處置:**不要繼續調參**。
>
> 下面這頁保留原文不動(它是 9/1 定稿的交接單),當作背景與指令參考。
> 現在還沒做的事只剩:寄兩封資料集申請信、9/10~9/11 的報告彙整。見 [待辦清單.md](待辦清單.md)。

> **你正在學校的遠端 GPU 機器上。這份文件讓你不必問就知道現在該做什麼。**
> 讀完這頁再讀 [背景與架構.md](背景與架構.md)(專案脈絡)與
> [遠端操作手冊.md](遠端操作手冊.md)(逐步指令)。

---

## 一句話任務

**用 CHIRLA 資料集訓練一個人物 Re-ID 模型,9/11 向工研院報告。**
使用者是宜蓁,負責模型訓練/測試/外部驗證;曾薪負責資料下載。

今天是第一階段(9/3~9/11)。**剩不到 9 天,緩衝很薄。**

---

## 為什麼做這件事(一分鐘版)

智慧廚房食安追責系統。M5 模組要判斷「cam2 這個人是不是剛從 cam1 走掉的那位」,
它綜合多種證據:轉場時間、地面幾何、位置、軌跡、**還有長得像不像**。

最後那條一直是空的,**卡的不是技術是授權**:

| 模型 | 跨視角 Rank-1 | 能出貨? |
|---|---|---|
| DINOv2(通用) | **11%**(隨機約 17%) | ✅ |
| OSNet(Re-ID 專用) | **92.7%** | ❌ 權重訓練於 Market-1501/MSMT17,研究限定且**授權會傳染** |

**CHIRLA 是 CC-BY-4.0 可商用 → 用它訓練 = 第一個能出貨的外觀模型。**

---

## ⛔ 已經做好的,不要重做

**第一階段需要的程式在 2026-09-01 全部寫完並驗過了。** 你的工作是**執行**,不是重寫。

| 檔案 | 用途 | 狀態 |
|---|---|---|
| `scripts/check_reid_env.py` | 環境自檢(5 項,含實跑 ResNet50 驗 GPU) | ✅ |
| `scripts/chirla_prep.py` | `--verify` 交接驗收 / `--index` 建索引 | ✅ |
| `scripts/train_reid_chirla.py` | 訓練(ResNet50+BNNeck+triplet+PK sampler) | ✅ 假資料跑通 |
| `src/m5_reid/chirla_embedder.py` | 讓模型插回 M5 | ✅ 契約驗過 |
| `scripts/export_chirla_embeddings.py` | 匯出 CHIRLA 官方 HDF5 格式 | ✅ |
| `scripts/verify_reid_metrics.py` | 指標實作可信度(6 組) | ✅ |
| `requirements-reid.txt` | 額外相依 | ✅ |
| `docs/CHIRLA_預先登記_20260901.md` | **訓練前的預先登記** | ✅ **已提交,不可修改 §2~§6** |

整條路徑已用「真影像的假資料集」端到端驗過:
索引 → 訓練(CE 1.78→0.87)→ checkpoint → embedder(2048 維、L2 範數 1.0)→ CMC。

---

## 現在該做什麼 —— 依序,每步有判斷點

### 步驟 1:環境(不需要資料,現在就能做)

```bash
conda create -n reid python=3.11 -y && conda activate reid   # ⚠ 不要 3.13+
nvidia-smi                                                    # 先看 CUDA 版本
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124  # 版本依上一行
pip install -r requirements.txt -r requirements-reid.txt
python scripts/check_reid_env.py --data-dir <打算放 CHIRLA 的路徑>
```

**判斷點:必須 0 個 ❌。** 它會實跑一次 ResNet50 forward+backward
(列得出 GPU ≠ 算得動),並量出 VRAM 峰值告訴你 batch 能開多大。

順便跑這個確認環境正常(不需要資料):

```bash
python scripts/verify_reid_metrics.py   # 應該 [ALL PASS]
```

### 步驟 2:資料驗收(拿到資料後的**第一件事**)

```bash
python scripts/chirla_prep.py --root <CHIRLA路徑> --verify
```

**判斷點:exit code 必須是 0。不通過就不要開始訓練。**
資料是別人下載的,不完整或是 Git LFS 空殼會白白吃掉三天。

它會對帳論文規格(22 身份 / 7 相機 / 10 序列)、檢查四種切分 CSV、
identity leakage、以及**全掃 Git LFS pointer**。

| 症狀 | 修法 |
|---|---|
| 大量影像小於 1KB | `git lfs pull` 或重跑 `huggingface-cli download` |
| 身份數 ≠ 22 | 可能只下載了部分 scenario |
| 找不到 CSV | 確認下載的是 `benchmark/` 不是只有 `videos/` |

### 步驟 3:建索引

```bash
python scripts/chirla_prep.py --root <CHIRLA路徑> --index --out chirla_index.json
```

輸出會列出每個 scenario 的 train/val/gallery/query 各有多少影像與身份。**挑一個 scenario 記下來**,後面都用它。

### 步驟 4:先 smoke 再正式訓練

```bash
# smoke:每 epoch 只跑 2 個 batch,確認能跑
python scripts/train_reid_chirla.py --index chirla_index.json \
    --scenario <上一步挑的> --arm S --epochs 2 --smoke --out /tmp/smoke

# S 臂(可出貨,交付主線)
python scripts/train_reid_chirla.py --index chirla_index.json \
    --scenario <同上> --arm S --epochs 60 --out model_result/reid/armS
```

**VRAM 不夠時**(`check_reid_env.py` 會告訴你):

| VRAM | 參數 |
|---|---|
| ≥10 GB | 預設 `--P 8 --K 4` |
| 6~10 GB | `--P 6 --K 4` |
| <6 GB | `--P 4 --K 4 --freeze-until 6` |

### 步驟 5:R 臂(研究上限對照)

```bash
python scripts/train_reid_chirla.py --index chirla_index.json \
    --scenario <同上> --arm R --weights <外部 Re-ID 權重> \
    --epochs 60 --out model_result/reid/armR
```

⚠ **除了 `--arm` 與 `--weights`,其他參數必須與 S 臂逐字相同** —— 這是受控雙臂,
不然比不出差異。R 臂的權重要從 FastReID 或 CION_ReIDZoo 另外抓。

### 步驟 6:評估

```bash
python scripts/export_chirla_embeddings.py --index chirla_index.json \
    --scenario <同上> --ckpt model_result/reid/armS/best.pth \
    --method chirla_armS --out embeddings/
```

然後在 CHIRLA 的 repo 裡跑官方的 `evaluate_reid.py --topk 1 5 10 --per-subset`。

⚠ **最終數字只能報 `_gallery` vs `_query`。** 開發階段只能用 `_train`/`_val` ——
看過 gallery/query 之後回頭調參,等於在測試集上調參,結果作廢。

### 步驟 7:接回 M5(最容易漏掉的一步)

⚠ **必須重新校準 `src/m5_reid/evidence.py:275` 的 `AppearanceLR.MEASURED`**,
加入新模型的 `(mu_same, sigma_same, mu_diff, sigma_diff)`。沒有這組數字,
`AppearanceLR.measured()` 會直接 raise。
數字由 `scripts/reid_eval_epfl.py` 的 `cross_view_consistency()` 產生。

然後用 `scripts/sim_m5_montecarlo.py` 重跑,對照舊的 DINOv2 數字 ——
這直接回答「這個模型對整套系統值多少」。

---

## 📉 期望值:數字不會好看,那不是你做壞了

**CHIRLA 論文自己的最佳基線 ResNet101 只有 CMC@1 18.81% / mAP 23.24%**(最難的長期情境)。

這個 benchmark 很難 —— 橫跨 7 個月,同一個人會換衣服換髮型。
而且**只有 22 個身份**可訓練(Market-1501 有 1501 個)。

**最大的技術風險是過擬合。** train_acc 衝很高但 CMC 沒跟上是**預期中的事**。
`train_reid_chirla.py` 開跑就會印警告。對策:`--freeze-until 5`、降 `--lr`、
早停(預設 `--patience 15`)。

⚠ **train/val 曲線分離要如實記錄進報告,不要藏。** 那是資料規模的限制不是做壞了。

---

## ⛔ 絕對不要做的事

1. **不要用 MMPTRACK 訓練。** 它是研究限定授權,訓練了模型就不能出貨。
   只能拿來做外部驗證。
2. **不要修改 `docs/CHIRLA_預先登記_20260901.md` 的 §2~§6。** 那是訓練前提交的
   設定與判準,跑完只能追加 §7 執行紀錄與結果。
3. **不要動 `src/m5_sim/world.py`。** 七輪實驗的所有數字建立在它上面。
4. **不要 commit 模型權重。** `.gitignore` 已排除 `model_result/reid/`。
5. **不要在看過 gallery/query 之後回頭調參。**

---

## 這個 repo 的工作文化(請沿用)

1. **預先登記**:實驗前把設定、判準、可否證的預測寫成文件提交 git,跑完只追加結果。
2. **誠實優先**:寧可報「量不到」也不要報好看的數字。
   踩過最嚴重的坑是「指標同時瞎掉,報出 0%/0%/A 級,實際 4 個人被併成 1 個」。
3. **每份報告都要有「這次不能回答什麼」段落,而且放在結論之前。**
4. **註解寫「為什麼」不寫「做什麼」**,把踩過的坑記在程式碼裡。繁體中文。
5. **自相矛盾就是指標寫錯的訊號** —— 例如同時印出「碎裂 0 次」與「碎裂率 5.6%」。
6. **當一個修法讓所有指標同時變完美時,先懷疑指標。**

---

## 收工前一定要做

```bash
# 1. 寫紀錄(格式見 紀錄/_範本.md)
#    第一次在這台機器操作時,「環境資訊」表格一定要填(GPU 型號/VRAM/磁碟/CHIRLA 路徑)
vim handoff/紀錄/$(date +%Y-%m-%d)_遠端.md

# 2. 更新待辦清單的勾選狀態
vim handoff/待辦清單.md

# 3. 推回去
git add handoff/ docs/ && git commit && git push
```

**本機那邊靠 `handoff/紀錄/` 知道遠端發生什麼事。** 不寫紀錄 = 本機完全不知道進度。

---

## 遇到問題時

| 症狀 | 先看 |
|---|---|
| 環境相關 | `check_reid_env.py` 的輸出,每一項都附了修法 |
| 資料相關 | `chirla_prep.py --verify` 的輸出 |
| 訓練跑不動 | 先跑 `--smoke`;VRAM 不夠就調 `--P`/`--K`/`--freeze-until` |
| 不確定為什麼要這樣做 | [背景與架構.md](背景與架構.md) |
| 指令細節 | [遠端操作手冊.md](遠端操作手冊.md) |

**卡住超過半小時就寫進紀錄並 push**,不要自己硬撐 —— 本機這邊看得到就能幫忙。

---

## 順帶一提:有一件事還沒解決,但不影響第一階段

2026-09-01 發現 M5 有個架構缺口:**地面校正與軌跡這兩條把誤併從 72% 壓到 5.6% 的證據,
只在重疊路徑生效,跨時轉場路徑完全沒有**。

⚠ 原文寫「**CHIRLA 是非重疊佈局 → 一切都走轉場路徑**」。
**2026-09-03 實測後這句要更正** —— CHIRLA 是**混合佈局**:40% 的(幀,身份)被 2 台以上
同時看到,cam2 與 cam3 根本在拍同一個房間,21 對相機裡只有 5 對從未共現。
所以 CHIRLA 上**兩條路徑都會被走到**,反而是同時測兩條路徑的好素材。
細節見 `docs/CHIRLA_鏡頭佈局實測_20260903.md`。

缺口本身仍然存在(真正非重疊的那 5 對、以及 60% 只被一台看到的觀測都走轉場路徑)。
所以如果你把訓練好的模型接回 M5(步驟 7)後看到誤併率仍然很高,**那不一定是模型的問題**,
可能是那個缺口。詳見 `docs/M5_模擬預先登記_幾何世界_20260901.md` §9.3 與 §11。

修它的計畫在 `~/.claude/plans/reid-playful-backus.md`(第八/九輪),**排在 9/11 之後**。
