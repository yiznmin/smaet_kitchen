# M3 微調 bootstrap — Colab 操作手冊

> 目的:用 EPFL 標好的小資料(person + knife,37 張)在 Colab 微調 RF-DETR,
> **驗證「微調能不能把之前抓不到的刀救回來」**。這是丟棄的驗證(EPFL CC-NC,不出貨)。
>
> 需要:一個 Google 帳號。Colab 免費 T4 GPU 即可。
> 資料(**多視角多類別版**):`data/m3_finetune_mv/dataset.zip`(~16 MB,90圖/732框/**11類**:人/刀具/砧板/食材/鍋鏟/鍋子/手/容器/抹布/夾子/手套;含 train/valid/test)。
> (舊的 person+knife bootstrap 在 `data/m3_finetune/`,已被此多類別版取代。)
> **重點觀察類別**:刀具、砧板、食材、手(食安關鍵)的 AP 有沒有從低往上。

---

## 步驟

### 0. 開 Colab + 開 GPU
1. 到 https://colab.research.google.com → 新增筆記本。
2. 上方選單 **執行階段 → 變更執行階段類型 → 硬體加速器選 GPU(T4)**。

### 1. 檢查 GPU、裝 rfdetr
```python
!nvidia-smi -L
!pip -q install rfdetr supervision
```

### 2. 上傳資料集 zip 並解壓
```python
from google.colab import files
up = files.upload()          # 選 dataset.zip
!unzip -q dataset.zip -d /content/m3ds
!ls /content/m3ds            # 應看到 train valid test
```

### 3.(對照組)先看「未微調」在 test 上抓不抓得到刀
```python
from rfdetr import RFDETRNano
from PIL import Image
import glob, os

base = RFDETRNano()          # COCO 預訓練(未微調)
for p in sorted(glob.glob('/content/m3ds/test/*.jpg'))[:3]:
    det = base.predict(Image.open(p).convert('RGB'), threshold=0.3)
    names = det.data.get('class_name') if det.data else None
    labs = [str(names[i]) for i in range(len(det))] if names is not None else []
    print(os.path.basename(p), '→', [l for l in labs])   # 幾乎不會有 'knife'
```

### 4. 微調(重點)
```python
from rfdetr import RFDETRNano
model = RFDETRNano()
model.train(
    dataset_dir='/content/m3ds',
    epochs=60,
    batch_size=4,
    grad_accum_steps=4,
    lr=1e-4,
    resolution=728,           # 高解析度(救小物件)
    output_dir='/content/output',
)
# 訓練過程會印出每個 epoch 的 validation mAP —— 看 knife 的 AP 有沒有往上動
```
> ⏱ 60 epoch 在 T4 約 15–40 分鐘(視資料/解析度)。
> ⚠ 若 OOM:把 `batch_size` 降到 2、`resolution` 降到 640。
> 📌 **變體尚未鎖定**:先跑 `RFDETRNano`;之後也用 `RFDETRMedium` 跑一次,**比較兩者微調後的 knife AP**,才決定用哪個(見 M3_完整發現與決策 §2.6、§4.2)。微調會洗牌,不能只用未微調表現選。

### 5.(驗證)看「微調後」在 test 上抓不抓得到刀
```python
for p in sorted(glob.glob('/content/m3ds/test/*.jpg'))[:6]:
    det = model.predict(Image.open(p).convert('RGB'), threshold=0.3)
    names = det.data.get('class_name') if det.data else None
    labs = [str(names[i]) for i in range(len(det))] if names is not None else []
    print(os.path.basename(p), '→', labs)   # 期待:開始出現 'knife'
```
```python
# 視覺化一張(存圖下載)
import supervision as sv, numpy as np, cv2
p = sorted(glob.glob('/content/m3ds/test/*.jpg'))[0]
img = Image.open(p).convert('RGB')
det = model.predict(img, threshold=0.3)
ann = sv.BoxAnnotator().annotate(np.array(img).copy(), det)
cv2.imwrite('/content/pred.jpg', cv2.cvtColor(ann, cv2.COLOR_RGB2BGR))
files.download('/content/pred.jpg')
```

### 6. 匯出 ONNX + 下載權重
```python
model.export(output_dir='/content/onnx')   # 產生 ONNX
!ls -la /content/onnx
from google.colab import files
files.download('/content/output/checkpoint_best_total.pth')   # 微調權重(檔名以實際輸出為準)
# 若有 onnx:files.download('/content/onnx/xxx.onnx')
```

---

## 怎麼判斷 bootstrap「成功」

| 現象 | 意義 |
|---|---|
| 步驟 3(未微調)→ 幾乎沒有 `knife` | 印證問題(現成模型漏刀) |
| 步驟 5(微調後)→ **開始出現 `knife`** | ✅ **微調有效,能救小物件** → 值得投入真實資料 |
| 訓練時 val 的 knife AP 從 ~0 往上 | ✅ 量化證據 |
| 微調後仍幾乎沒 knife | 資料太少/標註問題,需檢討(但至少流程跑通了) |

> 37 張很少,**別期待很準**;只要刀「從無到有」開始被偵測,就達到 bootstrap 目的。

---

## 之後(真實資料)

流程一模一樣,只是換成**你們自己的廚房資料**(幾百張、多樣、含 person/knife/cutting_board/raw_meat),用同一個 bbox 標註器標、同樣上 Colab 微調 → 產出**可出貨**的模型。

## 相關檔案
- 資料集:`data/m3_finetune/dataset.zip`
- 標註器:`results/m2/bbox_labeler.html`
- 抽幀/切分腳本:`scripts/extract_finetune_frames.py`、`split_coco.py`
