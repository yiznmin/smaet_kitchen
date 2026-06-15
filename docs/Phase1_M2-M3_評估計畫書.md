# 智慧廚房影像分析系統 — Phase 1：M2 動態偵測與 M3 物件偵測速度評估計畫書

| 項目 | 內容 |
|---|---|
| 計畫階段 | Phase 1（基礎建置：偵測管線速度與動態前過濾評估） |
| 對應模組 | M2 動態偵測、M3 物件偵測（YOLO）— 速度/資源面 |
| 主要資料來源 | EPFL-Smart-Kitchen-30（固定相機 exocentric 影片） |
| 架構版本 | 集中式單主機（純規則邏輯，不引入 LLM/VLM） |
| 文件版本 | v1（2026-05-31） |

---

## 1. 背景與目的

本專案為導入餐飲廚房現場的電腦視覺食安監控系統，採**集中式單主機**架構，
所有事件判斷以可解釋、可審計的規則邏輯實作，明確不引入 LLM/VLM。

整體系統分九大模組（M1–M9）。在投入完整管線開發前，**Phase 1 先解決最關鍵的可行性問題**：

> 「在目標主機上，哪一個 YOLO 版本／解析度／批次設定，能即時處理多路鏡頭？
>   加上 M2 動態前過濾後，能替 GPU 省下多少無效推論？」

此問題不需要任何物件標註，只需計時計量，因此可用公開資料集先行驗證演算法效能，
再以自錄資料驗證實際場域可用性。

### 目的

1. 實測並確定 EPFL 固定相機影片的規格（解析度、幀率、同畫面人數），補齊測試前置資訊。
2. 建立 **M3 速度選型表**：各 YOLO 版本在目標主機上的延遲、吞吐、GPU 記憶體、可支援鏡頭數。
3. 量化 **M2 動態前過濾** 對 GPU 負載的實際節省比例。
4. 產出可重現、設定檔驅動的 benchmark 工具，作為後續 Phase 的基礎。

---

## 2. 資料來源與授權（已查證）

採用 **EPFL-Smart-Kitchen-30**（論文 arXiv:2506.01608）的固定相機影片。
經實際查證，釐清其定位與限制如下，後續開發一律以此為準，不憑記憶引用數字。

### 2.1 資料集定位

EPFL-Smart-Kitchen-30 本質是一個「多模態人類行為理解 benchmark」學術專案，
定義 Action Recognition、Action Segmentation、Full-Body Motion Generation、
Lemonade(VLM-QA) 四個任務。**這些 benchmark、預訓練模型與動作標註本專案皆不使用**，
本專案只取其「原始固定相機影片」作為偵測管線的測試素材。

### 2.2 規格與結構（以 HTTP Range 讀取 zip central directory 實測）

| 項目 | 內容 |
|---|---|
| 相機 | 9 顆 Microsoft Kinect Azure 固定 RGB-D 相機（另有 1 個 HoloLens 頭戴，本專案不用） |
| 內容 | 16 位受試者、4 道食譜、共 29.7 小時 |
| 封裝 | 單一檔 `Public_release_videos.zip`（192.7 GB），**無單獨小檔可下載** |
| 結構 | `Public_release_videos/{train,test}/{受試者}/{session時間戳}/videos/*.mp4` |
| 相機檔名 | `output0, Aoutput0~3, Boutput0~3`（深度影片另置於 `videos_depth/`） |
| 單支 clip | 約 80–90 MB（最小 79.8 MB），壓縮方式 deflate |
| 解析度/幀率 | **公開文件未載明**，需取樣本以 ffprobe/OpenCV 實測（見 §4 步驟一） |

### 2.3 授權與使用界線（重要）

本案為產學合作，**最終產品要交付業者（商用部署）**，因此「資料集」與「模型」兩種授權都須避開非商用/AGPL。

**(a) 資料集授權**
- EPFL 資料集為 **CC BY-NC 4.0（禁止商業用途）**；GitHub 上的程式碼才是 MIT。
- 決議：**EPFL 影片僅作為 benchmark 測試素材，不用於訓練最終交付的模型**，使用時須引用論文（arXiv:2506.01608）。
- 對應原始說明文件 C.4 第 3 點。

**(b) 模型授權 — 關鍵限制**
- **Ultralytics YOLOv8 / YOLO11 為 AGPL-3.0**：研究/內部 benchmark 可用，但**包進商用閉源產品交付須開源整個應用或付費購買商業授權**，因此**不作為出貨候選**。
- 出貨候選改採 **Apache-2.0 授權**的偵測模型：**RTMDet（OpenMMLab）、YOLOX（Megvii）、RT-DETR（Baidu 原版 / lyuwenyu PyTorch 移植）**。
  - ⚠ 注意：RT-DETR 須用**原版 repo**；Ultralytics 套件內的 RT-DETR 實作仍屬 AGPL，不可使用。
- Ultralytics YOLO 僅保留為**速度參考基準對照欄**，標明不可出貨。
- 測速統一以 **ONNX Runtime（MIT）** 為後端：授權乾淨，且貼近實際部署形式（ONNX/TensorRT）。

---

## 3. 範圍界定

### 3.1 本階段「做」

- 以 EPFL 固定相機影片，實測影片規格。
- M2 動態偵測（背景差分）觸發率與 GPU 省電比例量測。
- M3 物件偵測**速度/資源**評估（延遲、FPS、GPU 記憶體、可支援鏡頭數）。
- 產出設定檔驅動、可重現的 benchmark 腳本與選型表。

### 3.2 本階段「不做」（明確排除）

| 不做 | 原因 |
|---|---|
| M3 偵測**準確度**（precision/recall/mAP） | EPFL 無物件 bbox 標註，改用 Roboflow + 自標資料於後續階段 |
| M4 多人追蹤驗證 | EPFL 每 session 為單人煮菜，無法驗證多目標 ID switch |
| M5 跨鏡頭 Re-ID | EPFL 為重疊同步多視角，與「跨區跨時重識別」情境不符 |
| 引入任何 LLM/VLM | 集中式版設計原則 |
| 使用 EPFL 的 benchmark／預訓練模型／動作標註 | 與本專案目標無關 |
| 下載完整 192.7 GB | 僅以 HTTP Range 取單支 clip（~80 MB）確認規格，測試視需要再多取數支 |

---

## 4. 技術方法與執行步驟

工具採極簡、可重現原則；模型版本不寫死於程式碼，一律由 `configs/benchmark.yaml` 切換。

### 步驟一：取樣本並實測規格（解掉 C.4 待確認 #1、#2）

- `scripts/fetch_sample.py`：用 HTTP Range 從 zip 取出單支 ~80 MB 固定相機 clip + zlib 還原，
  不下載整包 192.7 GB。
- `scripts/probe_video.py`：以 ffprobe + OpenCV 輸出解析度、幀率、總幀數、codec，
  並抽數張影格供目視確認同畫面人數；結果寫入 `results/video_probe.md`。
- **此步驟結果決定測試矩陣的解析度欄位**（若原生 < 1280，則測 1280 屬上採樣，需註記）。

### 步驟二：M2 動態偵測 benchmark（`scripts/bench_m2.py`）

- 以 `cv2.absdiff` 對相鄰影格（或固定參考幀）做差分 → 二值化 → 計算變動像素佔比 → `has_motion`。
- 統計整段影片的**觸發比例**（有動幀數佔比）；可選輸出動態區域 bbox（供 ROI 加速）。
- 比較「有 M2 前過濾 vs 無前過濾」送入 M3 的影格數差，估算 **GPU 省電比例**。
- 閾值由設定檔管理。

### 步驟三：M3 速度選型（`scripts/bench_m3_speed.py`）— 本階段主產出

依原始說明文件 C.3 的測試矩陣：

| 維度 | 取值 |
|---|---|
| 模型（出貨候選，Apache-2.0） | RTMDet（tiny/s）、YOLOX（tiny/s）、RT-DETR（r18） |
| 模型（參考基準，AGPL，不出貨） | Ultralytics YOLOv8n、YOLO11n |
| 推論後端 | ONNX Runtime（MIT），對齊實際部署 |
| 輸入解析度 | 640、1280（依步驟一決定 1280 是否保留） |
| 批次大小 | 1、4、8（模擬多鏡頭同時進來） |

每個組合記錄並輸出至 `results/m3_speed_table.csv`：

| 欄位 | 說明 |
|---|---|
| 模型版本/尺寸 | rtmdet_tiny、yolox_s… |
| 授權 / 可出貨 | Apache-2.0(可) / AGPL(僅參考) |
| 輸入解析度 | 640 / 1280 |
| 批次大小 | 1 / 4 / 8 |
| 單張推論延遲 (ms) | 平均 + p95 |
| 吞吐率 (FPS) | 批次下有效幀率 |
| GPU 記憶體峰值 (MB) | `torch.cuda.max_memory_allocated` |
| 可支援鏡頭數估算 | 依 FPS 與 target_fps 推算 |

附帶輸出「有/無 M2 前過濾」的 GPU 負載差異。

---

## 5. 預期產出

1. `results/video_probe.md` — EPFL 影片規格實測報告。
2. `results/m3_speed_table.csv` — YOLO 速度/資源選型表。
3. M2 觸發率與 GPU 省電比例量測結果。
4. 設定檔驅動、可重現的 benchmark 工具（`scripts/`、`src/`）。
5. 選型建議：在目標主機上即時處理 N 路鏡頭的推薦 YOLO 設定。

---

## 6. 驗收標準

| 項目 | 標準 |
|---|---|
| 規格實測 | 能明確報出解析度、幀率、同畫面人數，並反映到測試矩陣設計 |
| M2 | 空檔正確判無動、有人活動不漏觸發；能量出對 GPU 的省電比例 |
| M3 速度 | 各組合延遲/FPS/GPU 記憶體齊全；batch 增大時數值變化符合預期 |
| 可重現性 | 同設定重跑數值穩定；結果檔標明執行主機 GPU 型號與數據來源 |

---

## 7. 風險與待確認事項

| # | 事項 | 影響 | 處置 |
|---|---|---|---|
| 1 | **目標部署主機 GPU 規格未知** | 速度結論的有效性 | 開工前向業主／指導教授確認；開發機數據僅供管線驗證，不作選型結論（C.4 #4） |
| 2 | EPFL 解析度/幀率未公開 | 1280/高幀率測試是否有意義 | 步驟一實測解決 |
| 3 | EPFL 為單人場景 | M4/M5 無法以此驗證 | 後續以自錄資料驗證（C.5） |
| 4 | 資料集 CC BY-NC 授權 | 商用合規 | 僅作 benchmark，不訓練交付模型，使用引用論文 |
| 4b | **模型授權 AGPL（Ultralytics YOLO）** | 商用出貨合規 | 出貨候選改用 Apache-2.0（RTMDet/YOLOX/RT-DETR）；Ultralytics YOLO 僅作參考基準，不出貨 |
| 5 | M3 準確度無法用 EPFL 驗證 | 準確度評估缺口 | 後續以 Roboflow + 自標資料補（C.1） |

---

## 8. 時程與里程碑（建議）

| 里程碑 | 內容 | 前置 |
|---|---|---|
| M1 | 取樣本 + 規格實測完成 | 無 |
| M2 | M2 動態偵測 benchmark 完成 | 規格確認 |
| M3 | M3 速度選型表完成 | 確認目標主機 GPU |
| M4 | Phase 1 結案報告（選型建議） | 上述完成 |

---

## 9. 後續階段銜接

Phase 1 完成後，依原始說明文件 D 的優先順序推進：
Phase 2（+M2/M6/M7 事件驅動與精細分析）、Phase 3（+M5 跨鏡頭身份）、
Phase 4（Neo4j 關聯圖 + M9 調查查詢前端）。
M3 準確度、M4、M5 的驗證將改用對應的標註資料集與自錄整合資料。

---

## 參考

- EPFL-Smart-Kitchen-30：https://github.com/amathislab/EPFL-Smart-Kitchen ；論文 arXiv:2506.01608
- 資料下載：Zenodo records 15535461（影片）、15551913（姿態與標註）
- 原始專案說明：`original_info/智慧廚房系統_專案說明.md`
