# 智慧廚房影像分析系統 — Phase 1 結案報告

| 項目 | 內容 |
|---|---|
| 階段 | Phase 1:M2 動態偵測 + M3 物件偵測「速度選型」 |
| 架構 | 集中式單主機、純規則邏輯、**不引入 LLM/VLM** |
| 開發機 | RTX 3050 Laptop 4GB / Python 3.14 / onnxruntime-gpu 1.26(CUDA) |
| 日期 | 2026-05-31 |
| 狀態 | ✅ 工具鏈與初步選型完成;⏳ 準確度評估與目標主機複核待後續 |

---

## 1. 目標與範圍

在投入完整管線前,先解決最關鍵的可行性問題:**「哪個可商用偵測模型,在目標主機上能即時處理多路鏡頭?」**
此問題只需計時計量、不需標註,故先用公開資料(EPFL-Smart-Kitchen)驗證,準確度留待有標註資料時做。

- **做**:M2 動態偵測(省電過濾)、M3 速度/資源評估、可重現工具鏈。
- **不做(本階段)**:M3 準確度(無標註)、M4 追蹤、M5 Re-ID、M6+ 事件;不引入 LLM/VLM。

---

## 2. 完成項與關鍵結果

### 2.1 資料(EPFL-Smart-Kitchen)
- 以 **HTTP Range** 只取單支 ~80MB clip(免下載 192.7GB)實測規格:**1280×720 @30fps、H.264、單人場景**。
- 釐清此資料集只能做 M2/M3 速度(無 bbox);M4/M5 不適用(單人)。授權 **CC BY-NC 4.0 → 僅作 benchmark**。

### 2.2 M2 動態偵測（`src/m2_motion/detector.py`)
- `cv2.absdiff` 背景差分,純 CPU。實測:觸發率 41.1%、**對 M3 省電 58.9%**、**2.5ms/幀(400FPS)**。
- 正確性已驗證:單元測試(相同→靜、變化→動、雜訊不誤判)全 PASS + 真實影格視覺抽檢。
- 已知特性:M2 偵測「動作」非「存在」,門檻需依場域與後續事件需求調校。

### 2.3 M3 速度工具鏈（`scripts/bench_m3_speed.py`）
- 後端 **ONNX Runtime + CUDA**(授權乾淨、貼近部署);量延遲(mean/p95)、吞吐、GPU 記憶體峰值、可支援鏡頭數。
- 在 Python 3.14 踩通 GPU 環境:onnxruntime-gpu 1.26 + pip 安裝 CUDA12.9/cuDNN9.23 + `cuda_env` 掛載 DLL。
- 自動偵測模型輸入尺寸、OOM/錯誤容錯標記、環境資訊寫入報告。

### 2.4 評估了 8 個可商用(Apache-2.0)模型
D-FINE(N/S/M)、RF-DETR(nano/small/medium)、RT-DETRv2(r18/r34);全用官方來源匯出(pip / transformers)。

**速度排名(公平比較,統一 672×672,batch=1,RTX 3050):**

| 名次 | 模型 | 延遲 (ms) | 吞吐 (img/s) | GPU (MB) |
|---|---|---|---|---|
| 🥇 | dfine_n | 17.3 | 57.9 | 214 |
| 2 | dfine_s | 25.1 | 39.9 | 346 |
| 3 | rtdetrv2_r18 | 29.3 | 34.2 | 354 |
| 4 | dfine_m | 38.7 | 25.9 | 346 |
| 5 | rtdetrv2_r34 | 38.9 | 25.7 | 346 |
| 6-8 | rfdetr nano/small/medium | 43–46 | ~22 | ~600 |

**關鍵發現**:同解析度下 **D-FINE 系列速度/記憶體全面領先**;RF-DETR 因 ViT(DINOv2)backbone,
高解析度運算量大、在 4GB 上 batch 多會撞記憶體牆。原生尺寸時 RF-DETR-nano(384)雖快,
但那是低解析度的結果,對小物件偵測未必有利 —— 須由準確度評估定論。

> 詳見 `docs/M3_選型結果_公平672.md`(同條件)與 `docs/M3_選型結果_3050初步.md`(各原生尺寸)。

---

## 3. 授權合規(產學合作要出貨)
- **資料**:EPFL CC BY-NC 4.0 → 僅 benchmark、不訓練交付模型、使用引用論文。
- **模型**:全部 **Apache-2.0**(D-FINE/RF-DETR/RT-DETR),**避開 AGPL 的 Ultralytics YOLO**。
- 推論後端 ONNX Runtime(MIT)。詳見 `docs/模型選型評估.md`。

---

## 4. 誠實限制(尚未完成 / 數據效力)
1. **未做準確度評估**:EPFL 無 bbox 標註。**最終選型仍缺「準確度」這一半**,不能只憑速度拍板。
2. **數字在 RTX 3050(非部署主機)**:相對排名可信,**絕對數字(能否 30fps、接幾顆鏡頭)須在目標主機複核**。
3. **YOLOX/DAMO-YOLO/LW-DETR 刻意略過**:較舊、紙面準確度已輸現有模型,匯出成本高、預期墊底。
   例外:若目標主機**無 GPU**,DAMO-YOLO 的 CPU 變體才值得補。
4. **M4/M5/M6+ 尚未開始**:追蹤、Re-ID、事件判斷為後續 Phase。

---

## 5. 後續建議(優先序)
1. **準確度評估**(選型另一半):用 Roboflow 廚房資料 + 自標場域畫面,量各模型 precision/recall/mAP,
   並回答「RF-DETR@低解析度 vs D-FINE@640 對刀具等小物件的準確度」。**速度×準確度合併才是最終依據**。
2. **確認目標部署主機 GPU** → 在該機複核絕對數字、決定可支援鏡頭數。
3. 推進 **M4 追蹤(ByteTrack)→ M5 Re-ID → M6 事件規則**,並把 M2→M3 實際串接。

---

## 6. 產物索引

| 類別 | 路徑 |
|---|---|
| 計畫書 / 評估 | `docs/Phase1_M2-M3_評估計畫書.md`、`docs/模型選型評估.md` |
| 選型結果 | `docs/M3_選型結果_公平672.md`、`docs/M3_選型結果_3050初步.md` |
| 設定 | `configs/benchmark.yaml`、`configs/benchmark_fair672.yaml` |
| 取樣/規格 | `scripts/fetch_sample.py`、`scripts/probe_video.py` |
| M2 | `src/m2_motion/detector.py`、`scripts/bench_m2.py`、`scripts/verify_m2.py` |
| M3 | `scripts/bench_m3_speed.py`、`scripts/check_gpu.py`、`src/common/cuda_env.py` |
| 模型匯出 | `scripts/export_rfdetr.py`、`export_dfine.py`、`export_rtdetr.py`、`export_torchvision.py` |
| 數據/報告 | `results/probe/`、`results/m2/`、`results/m3/` |
| 模型檔 | `models/*.onnx`(原生 + `_672` 公平版) |
