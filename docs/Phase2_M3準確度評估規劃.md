# Phase 2 規劃:M3 物件偵測「準確度」初步評估

> 新階段文件(獨立於 Phase 1)。Phase 1 已完成 M2 與 M3 **速度**選型;本階段補上**準確度**這一半。
> 本步定位:用公開資料做**初步準確度探針**,非最終依據。

| 項目 | 內容 |
|---|---|
| 階段 | Phase 2:M3 準確度評估(起步) |
| 前提 | 已有 8 個 Apache 模型(D-FINE N/S/M、RF-DETR nano/small/medium、RT-DETRv2 r18/r34) |
| 資料 | Roboflow `kitchen-object-detection-acyvk` v1 |
| 環境 | RTX 3050 4GB / Python 3.14 / transformers + rfdetr + supervision(皆已裝) |

---

## 1. 為什麼做這步

最終選型 = **速度 × 準確度**。Phase 1 只做了速度(EPFL 無 bbox 標註,量不了準確度)。
本步用一份**有 bbox、可商用授權**的廚房資料集,先量「現成模型對廚房物件、尤其**刀(小物件)**的偵測準確度」,
作為選型的第二個維度起點。

---

## 2. 資料集分析(已查證)

| 項目 | 內容 |
|---|---|
| 來源 | https://universe.roboflow.com/kitchenobjectdetection/kitchen-object-detection-acyvk/dataset/1 |
| 授權 | **CC BY 4.0**(可商用、需標註來源)— 正式使用前於頁面再確認一次 |
| 規模 | 389 張、含 bounding box 標註 |
| 類別(14) | blender, bottle, bowl, countertop stone, countertop wood, cup, fork, **knife**, microwave, plate, refrigerator, **sink**, spoon, wineglass |

### 與本專案需求的落差(關鍵)
| 我們要的 | 此資料集 |
|---|---|
| person(廚師,主目標) | ❌ 無 |
| knife(刀,小物件) | ✅ 有 |
| cutting_board(砧板) | ❌ 無 |
| 生肉 / chicken(交叉污染核心) | ❌ 無 |
| sink(洗手台)、countertop(檯面) | ✅ 有 |
| 餐具/容器(fork/spoon/bowl/cup…) | ✅ 有 |

→ **缺主目標(人)與事件核心物件(砧板、生肉)**;有的多是靜物廚房用品。因此本步只能評「部分」。

### 另一隱憂:場景域
此類資料多為**家用廚房近景/靜物照**,與本案**商用廚房、固定俯視攝影機**不同視角 →
量到的準確度**不等於**部署現場的準確度。**僅供初步參考。**

---

## 3. 評估設計

### 3.1 模型與推論方式
沿用 Phase 1 的 8 個模型。**準確度用各自原生框架推論(非 ONNX),以正確解碼輸出**:
- **D-FINE / RT-DETR**:`transformers` 的 `AutoModelForObjectDetection` + `AutoImageProcessor.post_process_object_detection`。
- **RF-DETR**:`rfdetr` 的 `.predict()`(回傳 supervision `Detections`)。

> 速度評估用 ONNX(貼近部署);準確度評估用原生框架(解碼正確、省去重寫後處理)。兩者目的不同,各取所需。

### 3.2 類別映射(模型為 COCO 預訓練)
模型輸出 COCO 80 類。與資料集**可評的重疊類別(10)**:
`knife, fork, spoon, sink, bowl, cup, bottle, wine glass, microwave, refrigerator`。
**不可評**:countertop stone/wood、blender、plate(COCO 無對應類別)→ 報告中明確跳過、不灌水。

### 3.3 指標
用 **supervision** 的 mAP metric:
- 每類 **AP**(Average Precision),整體 **mAP@50** 與 **mAP@50:95**。
- **重點看 `knife`**(代表小物件偵測能力,呼應「能不能抓到廚師手上的刀」)。
- 可附 precision/recall。

### 3.4 產物
- 腳本:`scripts/eval_accuracy.py`
- 結果:`results/m3_acc/m3_accuracy.csv`(模型 × 類別 AP)+ `results/m3_acc/m3_accuracy.md`(報告)
- 報告開頭標明:資料集、授權、評估限制。

---

## 4. 取得資料(需使用者協助)
Roboflow 公開資料下載需帳號 / API key(直連被 403)。請二擇一:
1. 提供 Roboflow **API key**,我用 `roboflow` 套件下載 **COCO 格式**到 `data/roboflow_kitchen/`;或
2. 手動在頁面下載 **COCO 格式 zip**,解壓到 `data/roboflow_kitchen/`。

---

## 5. 誠實限制(本步不能宣稱的事)
- 只 ~10 重疊類別、389 圖、單一域 → **初步參考,非最終**。
- **缺 person / cutting_board / 生肉** → 主目標與交叉污染事件物件未評。
- 模型為 **COCO 預訓練、未在廚房微調** → 數字代表「現成模型」表現,微調後會不同。
- 場景域(家用近景 vs 商用俯視)可能不符。

## 6. 後續(非本步)
- **自標場域資料**:含 person、cutting_board、raw_meat,且為**實際部署鏡頭視角**;這才是最終 M3 準確度依據。
- 速度(Phase 1)× 準確度(本步起)合併 → 最終選型;最後於**目標主機**複核。

---

## 7. 驗證方式
- 跑 `python scripts/eval_accuracy.py` → 產出 per-class AP 表。
- Sanity:`knife` 有 AP 值;`countertop/blender/plate` 正確標為「不可評/跳過」;各模型都有結果或明確錯誤。
