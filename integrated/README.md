# 智慧廚房影像分析系統

以多台廚房監視鏡頭的影片為輸入,提供兩條獨立的處理流程:

| 流程 | 進入點 | 做什麼 | 輸出 |
|---|---|---|---|
| 人員追蹤 | `run_pipeline.py` | 偵測畫面中的人與物件,單鏡頭追蹤,跨鏡頭給同一個人同一個 chef_id | 逐幀人員軌跡、身份綁定紀錄 |
| 違規判定 | `run_violations.py` | 判斷單一廚師是否發生 NG1~NG4 食安違規 | 每件違規的類別、起訖時間、原因 |

所有路徑皆相對於本資料夾。

---

## 1. 資料夾結構

```
integrated/
├── run_pipeline.py            人員追蹤進入點(M1~M5)
├── run_violations.py          違規判定進入點(M6,NG1~NG4)
├── pipeline_config.yaml       人員追蹤的設定(含相機拓撲)
├── configs/
│   ├── violation_rules.yaml   違規判定的規則參數
│   └── zones/                 各鏡頭的區域多邊形(垃圾桶區、洗手台)
├── requirements.txt           相依套件
├── src/                       各模組原始碼
│   ├── common/                影片讀取、區域、繪圖
│   ├── m2_motion/             M2 動態偵測
│   ├── m3/                    M3 物件偵測(RF-DETR)
│   ├── m4_track/              M4 單鏡頭追蹤、偵測快取
│   ├── m5_reid/               M5 跨鏡頭身份
│   └── m6_events/             M6 違規判定(NG1~NG4、人體姿態)
├── scripts/                   M3 維護工具(訓練、掃門檻、單張推論)
└── weights/
    ├── m3_nano_r2_armA/       M3 微調權重(廚房 11 類)
    └── pretrained/            RF-DETR COCO 預訓權重、DINOv2
```

`weights/` 底下的權重檔不在 git 版本控制內(單檔超過上限),需與本資料夾一起以檔案方式交付。

M3 可偵測的 11 類:人、刀具、砧板、食材、鍋鏟、鍋子、手、容器、抹布、夾子、手套(夾子、手套目前沒有訓練樣本,實際偵測不到)。

---

## 2. 環境安裝

1. Python 3.11(3.10~3.12 可,不要用 3.13 以上)。
2. 依機器的 CUDA 版本安裝 torch / torchvision,例如 CUDA 12.8:
   ```
   pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
   ```
   確認 `python -c "import torch; print(torch.cuda.is_available())"` 輸出 `True`。裝到 CPU 版不會報錯,但會慢數十倍。
3. 安裝其餘套件:
   ```
   pip install -r requirements.txt
   ```
4. Windows 終端機執行前先設定 `set PYTHONIOENCODING=utf-8`,否則中文輸出會出錯。
5. 第一次跑 NG3 時會從 Hugging Face 下載人體姿態模型 `usyd-community/vitpose-base-simple`,需要連網一次;其餘預訓權重已放在 `weights/pretrained/`,不需下載。

---

## 3. 流程一:人員追蹤(run_pipeline.py)

### 3.1 處理流程

```
M1 讀影片 → M2 動態偵測 → M3 物件偵測 → M4 單鏡頭追蹤 → M5 跨鏡頭身份(chef_id)
```

| 模組 | 說明 |
|---|---|
| M1 | 讀影片檔,每 `stride` 幀取一幀(預設 5,即 30 fps 影片取 6 fps)。目前只支援影片檔,不支援 RTSP 串流 |
| M2 | 畫面差異判斷有無動態;可選擇無動態時沿用上一次偵測以節省運算(預設關閉) |
| M3 | RF-DETR 偵測人與廚房物件 |
| M4 | 每台鏡頭各自追蹤,給每個人一個 track_id |
| M5 | 依相機拓撲(鏡頭間移動時間、畫面重疊關係)與外觀,把不同鏡頭的 track 綁成同一個 chef_id |

### 3.2 執行

```
python run_pipeline.py \
    --videos <cam1.mp4> <cam2.mp4> <cam3.mp4> \
    --cameras camera_1 camera_2 camera_3 \
    --weights weights/m3_nano_r2_armA/checkpoint_best_total.pth \
    --topology <該場地的拓撲檔.yaml> \
    --out outputs/run
```

- `--cameras` 與 `--videos` 一一對應,名稱要與拓撲檔的鏡頭名稱相同。
- `--profile`:
  - `delivery`(預設):多人場景。
  - `single_person`:畫面中只有一個人時使用,加入 DINOv2 外觀特徵,追蹤較穩定;不可用於多人場景。
- 不給 `--weights` 時使用 COCO 預訓權重,只偵測人。
- 不給 `--topology` 時使用 `pipeline_config.yaml` 內建的示範場地拓撲,換場地一定要給自己的拓撲。
- 其他參數:`--stride`、`--start-fid`、`--end-fid`、`--max-loops`、`--thr`、`--resolution`、`--embedder`、`--gate-m3`、`--save-frames`。沒給的一律用 `pipeline_config.yaml` 的值。

### 3.3 輸出

| 檔案 | 內容 |
|---|---|
| `tracks.csv` | 逐幀、逐 track 的框座標與 chef_id |
| `chef_events.jsonl` | 每次身份綁定決策一筆:鏡頭、track_id、chef_id、分數、候選數 |
| `track_events.csv` | 追蹤事件:new_track、reacquired、lost_track、removed |
| `detections.csv` | M3 的 11 類物件偵測(有給 `--weights` 時才有) |
| `m2_motion.csv` | 每台每次取樣的動態判斷 |
| `resident.csv` | 常駐狀態與記憶體取樣 |
| `run_meta.json` | 本次執行的完整設定與統計 |

---

## 4. 流程二:違規判定(run_violations.py)

### 4.1 處理流程

```
各鏡頭影片 → M3 偵測(每 0.1 秒一次,全 11 類) → 垃圾桶接觸 / 洗手 / 手碰物件 / 清洗與放置 → NG1、NG2、NG4
output0、Aoutput1 影片 → ViTPose 人體姿態(臉、肩、手腕) + M3 手框 → NG3
```

只用規則判斷(偵測結果 + 區域 + 時序),不使用大型語言模型。只判斷畫面中的一位廚師。

### 4.2 執行

影片資料夾內放各鏡頭影片,檔名為 `<鏡頭名稱>.mp4`,且各鏡頭時間需同步:

```
python run_violations.py --video-dir <影片資料夾> --out outputs/ng
```

- 需要的鏡頭:`output0`(俯視全景)、`Aoutput1`、`Aoutput2`、`Aoutput3`、`Boutput1`、`Boutput2`。
- M3 偵測結果存在 `<out>/det_cache/<鏡頭>.npz`,人體姿態存在 `<out>/pose_cache/<鏡頭>.npz`,重跑時直接讀取、不重算。
- `--det-cache <資料夾>`:指定偵測快取位置(可重複使用已算好的快取)。
- `--t0 <秒>`:影片第 0 幀對應原始錄影的秒數,輸出的時間會加上此值。
- `--config`:改用別的規則設定檔(預設 `configs/violation_rules.yaml`)。

### 4.3 輸出

| 檔案 | 內容 |
|---|---|
| `ng_events.jsonl` | 每件違規一筆:`type`(NG1~NG4)、`start`、`end`(秒)、`reason`;NG2 附碰觸物件,NG3 附成立鏡頭,NG4 附物件、放置位置與 `status`(確定 / 待確認) |
| `ng_summary.json` | 各類違規次數、輸入、使用的設定 |

### 4.4 違規判斷規則

| NG | 違規 | 判斷方式 |
|---|---|---|
| NG1 | 碰垃圾桶後沒洗手 | Aoutput1 看到手在垃圾桶區、且 output0 看到人站在垃圾桶前,持續 1 秒算碰到;5 秒內再碰合併成一次;離開後 5 秒內洗手累計不到 1 秒即違規 |
| NG2 | 髒手碰食材、器具、容器 | 碰垃圾桶後、洗手前,手碰到食材 / 刀具 / 砧板 / 鍋鏟 / 鍋子 / 容器,每碰一次(≥ 0.3 秒)算一次;碰垃圾桶期間與離開後 1 秒內不算。與 NG1 各自獨立判斷 |
| NG3 | 手碰臉 | output0、Aoutput1 任一台看到手在臉旁(手框到臉中心 ≤ 0.5 倍肩寬)持續 ≥ 0.2 秒即觸發,每碰一次算一次;另一台清楚看到兩隻手都離臉很遠時否決。摸頭髮、隔著衛生紙擦鼻子、用湯匙等餐具吃東西不算 |
| NG4 | 洗過的東西直接放桌面 | Boutput1、Boutput2 看到東西在水槽中清洗;之後放下的位置由多數鏡頭判定:放在砧板 / 容器 / 鍋子上合格,放桌面違規,看不出位置時輸出 `status = 待確認` |

所有門檻與秒數都在 `configs/violation_rules.yaml`,可依企業的食安規範調整(例如洗手秒數、NG1 洗手期限、緩衝時間、碰觸物件類別)。

不在判斷範圍內:手機、衛生紙等 M3 沒有偵測類別的污染源;多人同時出現的畫面。

---

## 5. 導入新場地需要做的事

本系統的鏡頭配置、區域、相機拓撲與 M3 權重都是依示範廚房建立的。導入企業自己的廚房時,依序完成:

### 5.1 鏡頭配置與時間同步

- 違規判定需要:一台俯視全景(看得到垃圾桶與洗手台、人站的位置)、看得到臉與手的近拍鏡頭、水槽上方的近拍鏡頭。
- 各鏡頭影片時間必須同步。
- 鏡頭名稱若與示範不同,修改 `configs/violation_rules.yaml` 的 `cameras`、`zones`、`trash_event`、`hand_hygiene`、`ng3.views`、`ng4` 各節中的鏡頭名稱,以及 `run_violations.py` 開頭的 `CAMERAS`。

### 5.2 重畫區域

`configs/zones/epfl_<鏡頭>.json` 是示範廚房的區域,換場地必須依新畫面重畫。格式:

```json
{
  "width": 1280, "height": 720,
  "zones": [
    {"name": "垃圾桶區", "points": [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]},
    {"name": "洗手台",   "points": [[...], ...]}
  ]
}
```

- 座標為影片原始解析度的像素。
- 區域名稱需與 `violation_rules.yaml` 使用的名稱相同:`垃圾桶區`、`洗手台`、`洗手台_NG4`。
- `violation_rules.yaml` 的 `cameras.side_x` 是俯視鏡頭中 A / B 兩側工作區的分界 x 座標,也要依新畫面重設。

### 5.3 重建相機拓撲(人員追蹤用)

拓撲描述鏡頭之間的移動時間與畫面重疊關係,格式與 `pipeline_config.yaml` 的 `camera_topology` 區塊相同:

- `links`:相鄰鏡頭之間走路所需時間(平均與標準差,秒),由實際量測人員在鏡頭間移動的時間得到。
- `overlapping`:畫面有重疊的鏡頭組。
- `cameras`:各鏡頭名稱與時鐘偏移。

寫成獨立 yaml 檔,執行時用 `--topology` 指定,或寫入 `pipeline_config.yaml` 的 `camera_topology`。

### 5.4 重新訓練 M3(建議)

目前的微調權重只在示範廚房的畫面上訓練,換場地後偵測準確度可能下降,且有授權限制(見第 6 節)。建議用企業自己的影片標註後重新訓練:

1. 從影片截圖,以 COCO 格式標註 11 類(類別順序見 `src/m3/classes.py`)。
2. 另建訓練用虛擬環境(與推論環境的 supervision 版本衝突):
   ```
   pip install "rfdetr[train,loggers]==1.9.4"
   ```
3. 訓練(資料集格式:`<資料集>/{train,valid,test}/_annotations.coco.json`):
   ```
   python scripts/train_m3_rfdetr.py --dataset <資料集資料夾> --out <輸出資料夾>
   ```
4. 掃各類門檻,把結果填回 `pipeline_config.yaml` 的 `m3.class_thresholds`:
   ```
   python scripts/sweep_thresholds.py --weights <權重.pth> --variant nano --valid <資料集>/valid
   ```
5. 抽查效果,畫框輸出供目視檢查:
   ```
   python scripts/m3_on_frame.py --weights <權重.pth> --images <圖片>
   python scripts/m3_on_frame.py --weights <權重.pth> --video <影片> --frames <幀號…>
   ```
6. 換掉 `run_violations.py` 開頭的 `DET_WEIGHTS` 與執行 `run_pipeline.py` 時的 `--weights`。

### 5.5 調整規則參數

依企業的食安規範修改 `configs/violation_rules.yaml`,常見項目:

| 參數 | 位置 | 預設 |
|---|---|---|
| 洗手需累計秒數 | `hand_hygiene.wash_s` | 1.0 秒 |
| 碰垃圾桶後須在幾秒內洗手 | `ng.ng1_wash_within_s` | 5.0 秒 |
| 碰垃圾桶需持續秒數 | `trash_event.min_s` | 1.0 秒 |
| 髒手碰觸需持續秒數 | `ng.touch_min_s` | 0.3 秒 |
| 視為「碰到食材 / 器具」的類別 | `touch_classes` | 食材、刀具、砧板、鍋鏟、鍋子、容器 |
| 洗過後放在哪些物件上算合格 | `ng4.supports` | 砧板、容器、鍋子 |

建議先在幾段已知有無違規的影片上試跑,確認結果符合預期後再正式使用。

---

## 6. 授權注意事項

| 項目 | 授權 | 商業使用 |
|---|---|---|
| RF-DETR、Roboflow trackers、DINOv2 | Apache-2.0 | 可 |
| supervision | MIT | 可 |
| ViTPose(`usyd-community/vitpose-base-simple`) | Apache-2.0 | 可 |
| `weights/m3_nano_r2_armA/` M3 微調權重 | 訓練資料含 CC BY-NC 影片 | **不可**,商業使用前必須以自有資料重新訓練(見 5.4) |
