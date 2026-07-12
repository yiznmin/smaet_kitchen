# M3 微調 Colab 實戰紀錄(bootstrap 成功 + 踩雷修法)

> 記錄第一次在 Colab 微調 RF-DETR 的**完整結果**與**踩到的雷 + 修法**。
> 下次(尤其真實資料)照這份走,不會再卡。版本 v1(2026-07-12)。

---

## 1. 成果:bootstrap 成功 ✅

用 EPFL 多視角多類別資料(90圖/732框/11類)微調 RF-DETR-**Nano**,60 epoch。

### 訓練曲線(最直接證據)
`mAP@50:95` 從 **0.0001 → 0.4579**(epoch 0→36),穩定爬升 → **微調有效**。

### 各類別結果(val, epoch 60)
整體 **mAP@50 = 0.85、mAP@50:95 = 0.43**。

| 類別 | AP@50:95 | Recall | 類別 | AP@50:95 | Recall |
|---|---|---|---|---|---|
| 人 | 0.87 | 1.00 | 容器 | 0.46 | 1.00 |
| **刀具** | **0.45** | **1.00** | 抹布 | 0.55 | 0.90 |
| 鍋鏟 | 0.45 | 1.00 | 手套 | 0.43 | 0.86 |
| 食材 | 0.30 | 0.70 | 手 | 0.32 | 0.71 |
| 砧板 | 0.22 | 0.67 | 鍋子 | 0.27 | 0.67 |

### test 集實測(未看過的圖)
微調後在 test 圖上抓到 **刀具、砧板、食材、手、容器…**(中文自訂類別)。
對比未微調:只認 COCO 的 person/bowl/bottle。→ **微調把自訂類別教會了。**

### 結論
**微調確實把俯視小物件(刀)從「抓不到」救到 Recall 1.0。** 方向驗證成功 → 值得投入真實資料做可出貨模型。
(EPFL 為 CC-NC,此模型丟棄用。)

---

## 2. 踩雷 + 修法(重要,下次照做)

| # | 症狀 | 原因 | 修法 |
|---|---|---|---|
| 1 | `No module named 'pytorch_lightning'` | 只裝了推理版 rfdetr | 裝 **`pip install "rfdetr[train,loggers]"`** |
| 2 | `resolution=728 not divisible by 32` | 解析度需為 32 倍數 | 改 **`resolution=704`**(或 640) |
| 3 | 第一次 predict 卡住不動 | 訓練後 GPU 記憶體/狀態 | **重啟工作階段** → 載入權重(下方) |
| 4 | **每張圖偵測結果一模一樣** | `optimize_for_inference()` 把模型 trace 成常數 | **不要呼叫 optimize_for_inference**,直接 predict |
| 5 | 類別顯示成 COCO 英文(person/bowl…) | predict 的 class_name 預設用 COCO | 用 `det.class_id` + 自己的 NAMES 對照 |
| 6 | ONNX 匯出 `onnx has no attribute load_model_from_string` | onnx/torch 版本不合 | 改下載 `.pth`;或 `pip install --force-reinstall onnx==1.16.2` |
| 7 | `files.upload()` 一直沒反應 | 在等你選檔 | 點「選擇檔案」選 dataset.zip |
| 8 | 訓練被重跑一次 | 按了「全部執行」 | **一格一格點 ▶**,不要全部執行 |

### 關鍵修正碼

**安裝(第 1 格)**
```python
!pip -q install "rfdetr[train,loggers]" supervision
```

**載入微調權重 + 正確預測(不優化、中文類別)**
```python
from rfdetr import RFDETRNano   # medium 換 RFDETRMedium
model = RFDETRNano(pretrain_weights='/content/out_nano/checkpoint_best_regular.pth', num_classes=11)
import glob, os
from PIL import Image
NAMES={0:'人',1:'刀具',2:'砧板',3:'食材',4:'鍋鏟',5:'鍋子',6:'手',7:'容器',8:'抹布',9:'夾子',10:'手套'}
for p in sorted(glob.glob('/content/m3ds/test/*.jpg'))[:8]:
    det = model.predict(Image.open(p).convert('RGB'), threshold=0.3)
    print(os.path.basename(p),'->',[NAMES.get(int(c),int(c)) for c in det.class_id])
```

---

## 3. Nano vs Medium(變體選擇,尚未鎖定)

變體要**兩個都微調、比 AP** 才決定。改法:

| 位置 | Nano → Medium |
|---|---|
| 類別 | `RFDETRNano` → `RFDETRMedium` |
| 輸出 | `output_dir='/content/out_nano'` → `'/content/out_medium'` |
| batch | 4 → **2**(medium 較吃記憶體) |
| 權重路徑 | `out_nano/...` → `out_medium/...` |

跑完把兩者的 AP 表比較(尤其刀具/砧板/食材),AP 高又省的贏。

---

## 4. 下一步(真實資料 = 可出貨)

流程一模一樣,只換成**你們自己的廚房影片**:
```
你的影片 → 抽幀(隔N秒)→ bbox 標註器標(可動物件)→ zone 工具框(固定區域)
        → 切 train/valid/test → Colab 微調(nano+medium 比較)→ 匯出 ONNX → 本機部署
```

## 相關檔案
- Colab 手冊:[M3_微調bootstrap_Colab.md](M3_微調bootstrap_Colab.md)
- 筆記本:`notebooks/m3_finetune.ipynb`
- 評估方式:[M3_物件層評估方式說明.md](M3_物件層評估方式說明.md)
- 事件↔物件↔指標:[M3_事件物件對照與評估指標.md](M3_事件物件對照與評估指標.md)
