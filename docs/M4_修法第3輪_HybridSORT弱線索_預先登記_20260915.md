# 預先登記:M4 修法第 3 輪 —— Hybrid-SORT 弱線索(2026-09-15)

> **本文件在評估集上跑 `tcm`、`hmiou`、`hybrid` 任何一格之前定稿並提交 git。**
> 跑完只追加 §8 之後,§1–§7 不得修改。
> 過程紀錄:`docs/M4_修法輪次_20260915.md`。**本輪是三輪中的最後一輪:依使用者規則,無效就停下交給使用者決定。**

---

## 1. 為什麼做

| 輪 | 做法 | 結果 |
|---|---|---|
| 1 | McByte 遮罩 | 未達標(IDF1 +0.0199、IDSW −5.1%);`mcbyte` IDF1 0.4710 目前最高 |
| 2 | SparseTrack 偽深度分層 | 無效(IDF1 上升來自 FP 減少,IDSW 反而 +4.2%) |

第 3 輪依原計畫試 **Hybrid-SORT**(AAAI 2024,MIT)的**弱線索**:框重疊時位置與外觀都變模糊,
改用「偵測信心分數的變化」與「框的高度」輔助分辨。

讀官方原始碼後確認:
- **TCM(信心分數建模)**:第 1 階段成本加 `|clip(track 的分數卡爾曼預測, track_thresh, 1) − 偵測分數| × 權重`;
  第 2 階段(低分)成本加 `|clip(score − (pre_score − score), 0.1, track_thresh) − 偵測分數| × 權重`
- **HMIoU(高度調整 IoU)**:IoU × 兩框垂直方向重疊比例
- 官方 MOT17、MOT20 設定:`asso = Height_Modulated_IoU`、`TCM_first_step_weight = 1.0`、`TCM_byte_step_weight = 1.0`
- 官方 README 明列「ByteTrack + TCM」(`byte_tracker_score.py`)—— 與我們的基準(McByte 不開遮罩 = ByteTrack 式配對)最接近

---

## 2. 工具(本輪新增或修改)

| 檔案 | 改動 | 驗證 |
|---|---|---|
| `src/m4_track/hybrid_cues.py`(新) | `HMIoU`、`ScoreKalman`(逐行移植官方)、`HybridMcByteTracker` | **已完成**:HMIoU 與官方 `hmiou` 5,000 組隨機框相同;分數卡爾曼與官方 `KalmanFilter_score` 39,600 步相同 |
| `src/m4_track/sparse_dcm.py` | 第 1、2 階段相似度抽成可覆寫方法,加「配對前」「配對後」掛勾;預設行為不變 | **已完成**:重跑第 2 輪 `l1`,仍與第 1 輪 `nomask` 14 檔逐位相同 |
| `src/m4_track/tracker.py` | 新增 `rf_mcbyte_nomask_hybrid`;參數 `tcm_first_weight`、`tcm_byte_weight`、`use_hmiou` | **已完成**:權重 0、IoU 時(`off`)與 `nomask` 14 檔逐位相同 |
| `configs/tracker_hybrid_{off,tcm,hmiou,hybrid}.yaml`(新) | 由 `configs/tracker.yaml` 衍生 | **已完成**:解析後只差 `backend`、`backend_params` |

**官方的時序照抄**:配對到時,分數卡爾曼先用**舊的**分數更新,再換成新偵測分數;lost 找回時 `pre_score` = 新分數。

⚠ **刻意與官方不同**:
- 基底是 McByte(ByteTrack 式兩階段),**沒有** Hybrid-SORT 的 OC-SORT 部分(四角速度方向、OCR 第二次配對)
- TCM 從**相似度扣除**(McByte 最大化相似度),門檻判斷在扣除之後 —— 與官方在距離上加 TCM 後比門檻等價
- `track_thresh`(分數夾取邊界)對映為 McByte 的 `high_conf_det_threshold` = **0.25**(官方 0.6)

---

## 3. 執行網格(定死)

共同設定同第 1、2 輪:評估集 7 序列、camera_1、2、3、4、7、stride 5、跟丟緩衝 5 秒、門檻 0.10、不開遮罩。

| 格 | 設定檔 | TCM 權重(第 1 / 第 2 階段) | HMIoU | 性質 |
|---|---|---|---|---|
| `nomask` | — | 0 / 0 | 否 | 基準(第 1 輪既有;= `off`) |
| **`tcm`** | `tracker_hybrid_tcm.yaml` | 1.0 / 1.0 | 否 | 只加信心分數線索 |
| **`hmiou`** | `tracker_hybrid_hmiou.yaml` | 0 / 0 | 是 | 只換高度調整 IoU |
| **`hybrid`** | `tracker_hybrid_hybrid.yaml` | 1.0 / 1.0 | 是 | **官方 MOT17/MOT20 的弱線索組合** |

跑法與評估工具同第 2 輪(每格 5 台同時跑、合併、`make_track_gt` → `eval_m4_idf1` → `diag_m4_ghosts`)。

---

## 4. 判準(先寫死)

### 4.1 主判準(`tcm`、`hmiou`、`hybrid` 各自對 `nomask`)

| # | 判準 | 門檻 |
|---|---|---|
| M1 | 單鏡頭 IDF1 | **≥ +0.02** |
| M2 | IDSW | **少 ≥ 10%** |

### 4.2 對目前結果

| # | 判準 | 門檻 |
|---|---|---|
| B1 | IDF1 | > `base` 0.4599 |
| B2 | IDF1(記錄用,不參與判定) | 與第 1 輪 `mcbyte` 0.4710 比較 |

### 4.3 硬性約束(對 `nomask`)

| # | 約束 |
|---|---|
| H1 | 召回(CLR)下降不超過 1pp |
| H2 | 誤偵率上升不超過 3pp |

### 4.4 決策表

| 三格中至少一格 M1、M2 都過 | H1、H2 | 判定 | 下一步 |
|---|---|---|---|
| 是 | 過 | **弱線索有效** | 回報使用者;建議把通過的線索疊到遮罩上(另寫預先登記) |
| 是 | 不過 | **有效但代價超限** | 回報使用者決定 |
| **否** | — | **弱線索在 CHIRLA 上無效** | **三輪都沒有有效解決 → 依使用者規則停下**,整理三輪結果、檢討與下一步選項交給使用者 |

⚠ 門檻沿用第 1、2 輪,是判斷值不是實驗結果。

---

## 5. 事先寫下的預測(可被否證)

| # | 預測 | 依據 |
|---|---|---|
| 1 | **三格都不過 M1** | 前兩輪都未能讓 IDF1 +0.02;弱線索只在候選之間分數接近時才改變選擇 |
| 2 | 三格的誤偵率相對 `nomask` 變化都 < 1pp | 只改配對,不改偵測與開新 track 的門檻 |
| 3 | `hmiou` 的 IDSW 少於 `nomask` | 前後站的人框高度通常不同,HMIoU 會壓低高度差大的配對 |
| 4 | `tcm` 的 IDSW 變化絕對值 < 5% | RF-DETR nano 在 CHIRLA 的信心分數跳動大,分數連續性作為線索的鑑別力有限 |

⚠ 預測 4 的依據(分數跳動大)沒有量過。

---

## 6. 已知限制(報告要照抄,放在結論之前)

1. **基底是 McByte 不是 OC-SORT**:沒有 Hybrid-SORT 的四角速度方向與 OCR;結果代表「在 McByte 上加弱線索」,不等於 Hybrid-SORT 本身
2. 分數夾取邊界用 0.25(McByte 的高分門檻),官方 0.6;權重用官方值,**沒有調參**
3. 偵測器是 RF-DETR nano(COCO),官方是 YOLOX;兩者信心分數的分布不同
4. stride 5;只有 5 台鏡頭;**CHIRLA 是辦公室不是廚房**
5. 召回用 CLR 定義;每格只跑一次(CPU 決定性計算,`off` 的逐位相同檢查支持)

---

## 7. 硬性驗收(不通過就是實作或資料錯)

| # | 條件 |
|---|---|
| V1 | HMIoU 與官方等價(**已完成**,5,000 組) |
| V2 | 分數卡爾曼與官方等價(**已完成**,39,600 步) |
| V3 | `off` 與 `nomask` 逐位相同(**已完成**,14 檔) |
| V4 | 重構後第 2 輪 `l1` 仍與 `nomask` 逐位相同(**已完成**,14 檔) |
| V5 | 三格 × 5 台結果 json 的 `tracker_kwargs` 的 `tcm_first_weight`、`tcm_byte_weight`、`use_hmiou` 與 §3 相同 |
| V6 | 四格 `eval_m4_idf1.py` 自檢通過;`nomask` 重算 IDF1 0.4511、IDSW 2,619 |
| V7 | 各台紀錄檔無 Traceback |
| V8 | `src/m5_sim/world.py` 未動;既有預先登記 §1–§7 未動 |

### 7.1 遠端指令

```bash
export PYTHONIOENCODING=utf-8
R="D:/新增資料夾/CHIRLA/CHIRLA_data/CHIRLA"
S="seq_004 seq_006 seq_007 seq_020 seq_024 seq_025 seq_026"
K="camera_1 camera_2 camera_3 camera_4 camera_7"
O=results/m4_round3
C="--cache-dir results/det_cache/coco_nano --root $R --seqs $S --stride 5 --lost-buffer-seconds 5 --thr 0.10"

for L in tcm hmiou hybrid; do
  for c in $K; do
    .venv/Scripts/python.exe scripts/eval_m4_chirla.py $C --cameras $c --tracker configs/tracker_hybrid_$L.yaml \
        --label hyb_${L}_$c --dump-dir $O/parts/hyb_${L}_$c --out $O/parts/hyb_${L}_$c.json > logs/m4_round3_hyb_${L}_$c.log 2>&1 &
  done
  wait
  .venv/Scripts/python.exe scripts/merge_m4_dumps.py --parts $(for c in $K; do echo $O/parts/hyb_${L}_$c; done) \
      --seqs $S --out-dir $O/hyb_${L}_dump
done

for L in nomask tcm hmiou hybrid; do
  D=$O/hyb_${L}_dump; [ $L = nomask ] && D=results/m4_round1/nomask_dump
  .venv/Scripts/python.exe scripts/make_track_gt.py --root "$R" --tracks-dir $D --seqs $S --out-dir $O/${L}_track_gt
  .venv/Scripts/python.exe scripts/eval_m4_idf1.py --root "$R" --tracks-dir $D --seqs $S --cameras $K \
      --stride 5 --label $L --out $O/idf1_$L.json
  .venv/Scripts/python.exe scripts/diag_m4_ghosts.py --root "$R" --tracks-dir $D --seqs $S \
      --track-gt-dir $O/${L}_track_gt --out $O/ghosts_$L.json
done
```

---

*本文件於在評估集上跑 `tcm`、`hmiou`、`hybrid` 任何一格之前定稿。跑完只追加 §8「結果」與其後章節。*
