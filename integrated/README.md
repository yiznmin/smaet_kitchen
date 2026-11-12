# 智慧廚房影像分析系統

以多台廚房監視鏡頭的影片為輸入,提供兩條獨立的處理流程:

| 流程 | 進入點 | 做什麼 | 輸出 |
|---|---|---|---|
| 人員追蹤 | `run_pipeline.py` | 偵測畫面中的人與物件,單鏡頭追蹤,跨鏡頭給同一個人同一個 chef_id | 逐幀人員軌跡、身份綁定紀錄 |
| 違規判定 | `run_violations.py` | 判斷單一廚師是否發生 NG1~NG4 食安違規 | 每件違規的類別、起訖時間、原因 |

所有路徑皆相對於本資料夾。各模組的原理、參數與修改方式見 [模組說明.md](模組說明.md)。

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
├── scripts/                   導入新場地用的工具(見第 5 節)
│   ├── draw_zones.py          畫各鏡頭區域、量座標
│   ├── extract_frames.py      從影片抽出要標註的畫面
│   ├── make_labeler.py        產生瀏覽器版物件標註頁
│   ├── build_dataset.py       合併標註、切分訓練資料集
│   ├── train_m3_rfdetr.py     訓練 M3
│   ├── sweep_thresholds.py    掃 M3 各類門檻
│   ├── m3_on_frame.py         對圖片 / 影片幀跑 M3 並畫框
│   ├── build_topology.py      產生人員追蹤用的相機拓撲
│   └── render_violations.py   把每件違規剪成對照影片
└── weights/
    ├── m3_nano_r2_armA/       M3 微調權重(廚房 11 類)
    └── pretrained/            RF-DETR COCO 預訓權重、DINOv2、ViTPose 人體姿態模型
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
5. 所有模型權重都已放在 `weights/` 底下(含 NG3 用的人體姿態模型 `weights/pretrained/vitpose-base-simple/`),執行時不需連網。

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
- 處理時間參考:6 台鏡頭從影片開始跑,約為影片長度的 3 倍(NVIDIA RTX A6000)。大部分時間花在 M3 偵測與姿態估計;已有快取時重跑只需數秒。
- `--det-cache <資料夾>`:指定偵測快取位置(可重複使用已算好的快取)。
- `--t0 <秒>`:影片第 0 幀對應原始錄影的秒數,輸出的時間會加上此值。
- `--config`:改用別的規則設定檔(預設 `configs/violation_rules.yaml`)。

### 4.3 輸出

| 檔案 | 內容 |
|---|---|
| `ng_events.jsonl` | 每件違規一筆:`type`(NG1~NG4)、`start`、`end`(秒)、`reason`;NG2 附碰觸物件,NG3 附成立鏡頭,NG4 附物件、放置位置與 `status`(確定 / 待確認) |
| `ng_summary.json` | 各類違規次數、輸入、使用的設定 |

要看每件違規的畫面,接著執行:

```
python scripts/render_violations.py --video-dir <影片資料夾> --out outputs/ng
```

每件違規輸出一支影片到 `<out>/report/<編號>_<NG 類別>.mp4`:6 台鏡頭並排,疊上偵測框與區域;上方是違規類別、原因、時間,下方時間軸標出違規區間,播放到違規區間內時外框變紅。中文字型預設用 Windows 的微軟正黑體,其他系統用 `--font` 指定。

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

本系統的鏡頭配置、區域、相機拓撲與 M3 權重都是依示範廚房建立的。導入企業自己的廚房時,依下列順序完成:

| 步驟 | 項目 | 必要性 | 使用工具 |
|---|---|---|---|
| 5.1 | 鏡頭配置與時間同步 | 必要 | — |
| 5.2 | 畫各鏡頭的區域 | 必要 | `scripts/draw_zones.py` |
| 5.3 | 設定俯視鏡頭的 A / B 側分界 | 必要 | `scripts/draw_zones.py`(量座標) |
| 5.4 | 重新訓練 M3 | 必要(授權與準確度) | `extract_frames.py` → `make_labeler.py` → `build_dataset.py` → `train_m3_rfdetr.py` |
| 5.5 | 調整規則參數 | 選擇性 | 編輯 `configs/violation_rules.yaml` |
| 5.6 | 建立相機拓撲 | 只有人員追蹤需要 | `scripts/build_topology.py` |
| 5.7 | 試跑驗收 | 必要 | `run_violations.py` |

工具產生的標註頁都是本機 HTML,用瀏覽器打開即可操作,不需安裝其他軟體,也不會上傳任何畫面。

### 5.1 鏡頭配置與時間同步

違規判定需要以下鏡頭(名稱為示範廚房的命名):

| 鏡頭 | 位置 | 用途 |
|---|---|---|
| `output0` | 俯視全景,看得到垃圾桶與人站的位置 | 人站在哪一側、是否站在垃圾桶前 |
| `Aoutput1` | 近拍,看得到垃圾桶與廚師的臉、手 | 手碰垃圾桶、手碰臉 |
| `Aoutput1`~`Aoutput3`、`Boutput1`~`Boutput2` | A / B 兩側工作區的近拍 | 手碰食材、器具;放置位置 |
| `Boutput1`、`Boutput2` | 水槽上方 | 洗手、清洗食材 / 器具 |

- 影片為 30 fps,檔名為 `<鏡頭名稱>.mp4`。
- 各鏡頭的時間必須同步:同一個動作在各支影片中要出現在同一秒。檢查方式是找一個明顯的瞬間(例如開關燈、拍手),確認各支影片中該瞬間的時間相同;不同時,先用剪輯工具(例如 ffmpeg)把各支裁成同一個起點。
- 鏡頭名稱或數量與示範不同時,修改:
  - `configs/violation_rules.yaml` 的 `cameras`、`zones`、`trash_event`、`hand_hygiene.close_cams`、`ng3.views`、`ng4.views`、`ng4.sink_cams`、`ng4.wash_views`。
  - `run_violations.py` 開頭的 `CAMERAS`。

### 5.2 畫各鏡頭的區域

需要畫的區域:

| 鏡頭 | 區域名稱 | 範圍 |
|---|---|---|
| `output0` | `垃圾桶區` | 垃圾桶本體及前方站立處 |
| `Aoutput1` | `垃圾桶區` | 垃圾桶開口(手伸進去丟東西的位置) |
| `Aoutput1`、`Aoutput2` | `洗手台_NG4` | 畫面中的水槽(排除還在水槽內清洗的東西) |
| `Boutput1`、`Boutput2` | `洗手台`、`洗手台_NG4` | 畫面中的水槽 |

步驟(每台鏡頭各做一次):

1. 產生標註頁。`--frame` 挑一張區域沒被人擋住的畫面:
   ```
   python scripts/draw_zones.py <影片資料夾>/output0.mp4 --frame 300 --camera output0
   ```
2. 用瀏覽器打開 `outputs/zones/output0_zones.html`:
   - 左鍵沿區域邊界點一圈(至少 3 點)。
   - 從「名稱」下拉選單選區域名稱,按「完成區域」。
   - 該鏡頭的區域都畫完後按「匯出」,瀏覽器會下載 `output0.json`。
3. 把下載的檔案放到 `configs/zones/`,並修改 `configs/violation_rules.yaml` 的 `zones`,讓該鏡頭指向它:
   ```yaml
   zones:
     output0: configs/zones/output0.json
   ```
4. 檢查:把區域疊回畫面輸出圖片,確認位置正確:
   ```
   python scripts/draw_zones.py <影片資料夾>/output0.mp4 --frame 300 --preview configs/zones/output0.json
   ```
   輸出 `outputs/zones/output0_preview.png`,區域以編號標示,編號對應的名稱會印在終端機。

### 5.3 設定俯視鏡頭的 A / B 側分界

系統用俯視鏡頭(`output0`)中人框中心的 x 座標判斷廚師在哪一側工作,再挑該側的近拍鏡頭判斷碰觸:

- x 小於 `x_a`:A 側(用 `Aoutput1`~`Aoutput3`)。
- x 大於 `x_b`:B 側(用 `Boutput1`、`Boutput2`)。
- 介於兩者之間:兩側的近拍鏡頭都用。

量法:

1. 從 `output0` 影片中各挑一張廚師站在 A 側工作區最靠中間處、以及 B 側工作區最靠中間處的畫面(記下幀號)。
2. 用 5.2 的標註頁打開這兩幀(`--frame <幀號>`),把游標移到廚師身體中線,讀畫面上方顯示的 x 座標。
3. A 側那張的 x 填入 `x_a`,B 側那張的 x 填入 `x_b`:
   ```yaml
   cameras:
     side_x: {x_a: 480.9, x_b: 729.3}
   ```

### 5.4 重新訓練 M3

目前的權重只用示範廚房的畫面訓練,而且訓練資料的授權不允許商業使用(見第 6 節),所以必須用企業自己的影片重新訓練。參考:目前權重用了 540 張標註圖(約 5,000 個框)。

**1. 抽圖**:從各鏡頭、不同時段、不同廚師的影片抽出要標註的畫面。

```
python scripts/extract_frames.py --videos <a.mp4> <b.mp4> ... --n 50 --out datasets/kitchen/images
```

每支影片均勻挑 `--n` 張有動作的畫面,檔名為 `<影片名>_<幀號>.jpg`。

**2. 標註**:產生標註頁,在瀏覽器裡畫框。

```
python scripts/make_labeler.py --img-dir datasets/kitchen/images --out outputs/labeler/labeler.html --batch 200
```

- 每頁 200 張,產生 `labeler_1.html`、`labeler_2.html`…,可分給多人同時標。
- 操作方式:
  - 拖拉滑鼠畫框。
  - 數字鍵 1~9 或下拉選單切換類別。
  - 左右方向鍵換張。
  - 右鍵點框可刪除。
- 每張圖中看得到的 11 類物件都要標;漏標的物件會被模型學成「不是物件」。
- 標完按「匯出」,下載 `coco_1.json`、`coco_2.json`…。關掉頁面前一定要匯出,頁面不會自動存檔。

**3. 建立資料集**:合併所有匯出檔,切成 train / valid / test。

```
python scripts/build_dataset.py --labels coco_1.json coco_2.json ... --img-dir datasets/kitchen/images --out datasets/kitchen/m3
```

終端機會列出各類別的框數;沒有任何標註的類別訓練後偵測不到。

**4. 訓練**:另建訓練用虛擬環境(與推論環境的 supervision 版本衝突,不可裝在同一個環境),需要 GPU。

```
pip install "rfdetr[train,loggers]==1.9.4"
python scripts/train_m3_rfdetr.py --dataset datasets/kitchen/m3 --out weights/m3_kitchen
```

完成後權重為 `weights/m3_kitchen/checkpoint_best_total.pth`。

**5. 掃各類門檻**:回到推論環境執行,結果填回 `pipeline_config.yaml` 的 `m3.class_thresholds`。

```
python scripts/sweep_thresholds.py --weights weights/m3_kitchen/checkpoint_best_total.pth --variant nano --valid datasets/kitchen/m3/valid
```

**6. 抽查效果**:對幾張圖或影片幀畫框,目視檢查。

```
python scripts/m3_on_frame.py --weights weights/m3_kitchen/checkpoint_best_total.pth --video <影片> --frames 300 600 900
```

**7. 換上新權重**:
- 修改 `run_violations.py` 開頭的 `DET_WEIGHTS`。
- 刪除舊的偵測快取 `<out>/det_cache/`,否則會沿用舊權重的結果。
- 執行 `run_pipeline.py` 時改用 `--weights weights/m3_kitchen/checkpoint_best_total.pth`。

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

只改規則參數不需要重算偵測快取與姿態,重跑 `run_violations.py` 會直接讀既有快取,很快就有結果。

### 5.6 建立相機拓撲(只有人員追蹤需要)

拓撲描述鏡頭之間走路要多久、哪些鏡頭畫面重疊,M5 用它判斷不同鏡頭裡的人是不是同一個。只做違規判定可跳過這一步。

1. 產生空白表格:
   ```
   python scripts/build_topology.py --template configs/topology_links.csv
   ```
2. 依平面圖填表,每條可以走過去的路徑填一列,雙向各填一列:
   ```
   from,to,walk_distance_m,overlapping
   cam1,cam2,8.0,no
   cam2,cam1,8.0,no
   cam1,cam3,,yes
   ```
   - `walk_distance_m`:兩台鏡頭視野邊界之間的步行路徑距離(公尺)。要繞過中島、桌子等障礙物,不是直線距離。
   - `overlapping`:兩台鏡頭的畫面有重疊就填 `yes`,距離留空。
3. 產生拓撲檔(移動時間 = 距離 / 0.9 公尺每秒):
   ```
   python scripts/build_topology.py --csv configs/topology_links.csv --out configs/topology.yaml
   ```
4. 執行人員追蹤時指定:`python run_pipeline.py ... --topology configs/topology.yaml`。鏡頭名稱要與 `--cameras` 相同。

### 5.7 試跑驗收

1. 準備幾段已知內容的影片:有違規的(碰垃圾桶後沒洗手、手碰臉、洗過的東西放桌面等)和沒有違規的各幾段。
2. 執行 `run_violations.py` 與 `scripts/render_violations.py`,看每件違規的對照影片,判斷是否符合實際情況;也要確認沒有違規的影片沒有輸出事件。
3. 結果不符時,依序檢查:
   - 區域預覽圖的位置是否正確(5.2)。
   - `side_x` 是否正確(5.3)。
   - 用 `m3_on_frame.py` 看 M3 有沒有偵測到相關物件(5.4)。
   - 規則參數是否符合規範(5.5)。

---

## 6. 授權注意事項

| 項目 | 授權 | 商業使用 |
|---|---|---|
| RF-DETR、Roboflow trackers、DINOv2 | Apache-2.0 | 可 |
| supervision | MIT | 可 |
| ViTPose(`usyd-community/vitpose-base-simple`) | Apache-2.0 | 可 |
| `weights/m3_nano_r2_armA/` M3 微調權重 | 訓練資料含 CC BY-NC 影片 | **不可**,商業使用前必須以自有資料重新訓練(見 5.4) |
