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

---

# 8. 結果(2026-09-15,遠端 Venus / RTX A6000)

## 8.1 一句話結論

> **依 §4.4 決策表:三格都沒過 M1、M2 → 「弱線索在 CHIRLA 上無效」→ 三輪都沒有有效解決,依使用者規則停下。**
> 三格的 IDF1 都**下降**、ID 切換都**增加**、召回都跌破約束;混人 track 雖然變少,換來的是更多斷裂與新開 track。

## 8.2 硬性驗收

| # | 結果 |
|---|---|
| V1~V4 | 通過(提交前完成):HMIoU 5,000 組、分數卡爾曼 39,600 步與官方相同;`off` 與重構後 `l1` 都與 `nomask` 14 檔逐位相同 |
| V5 | **通過**:三格 × 5 台的 `tcm_first_weight`、`tcm_byte_weight`、`use_hmiou` 與 §3 相同,`enable_mask_manager` = false |
| V6 | **通過**:四格自檢 OK;`nomask` 重算 IDF1 0.4511、IDSW 2,619 |
| V7 | **通過**:15 個紀錄檔無 Traceback |
| V8 | 通過:`world.py` 未動;第 1、2、3 輪預先登記 §1–§7 未動 |

執行時間:三格各約 75 秒;含評估共 272 秒。

## 8.3 結果(5 台 × 7 序列)

| 格 | track | 誤偵率 | 混人 | **IDF1** | IDR / IDP | **IDSW** | 召回(CLR) | FN | FP | Frag |
|---|---|---|---|---|---|---|---|---|---|---|
| `nomask` | 2,320 | 23.32% | 254 | 0.4511 | 0.4845 / 0.4220 | 2,619 | 91.78% | 11,138 | 31,212 | 2,891 |
| `tcm` | 3,946 | 24.68% | 235 | 0.4419 | 0.4556 / 0.4291 | 4,349 | 88.84% | 15,118 | 23,501 | 4,118 |
| `hmiou` | 2,796 | 23.14% | 186 | 0.4419 | 0.4646 / 0.4214 | 2,912 | 90.57% | 12,783 | 26,667 | 3,068 |
| `hybrid` | 4,443 | 22.10% | 169 | 0.4275 | 0.4285 / 0.4265 | 4,828 | 87.07% | 17,519 | 18,140 | 4,564 |

各鏡頭 IDF1:

| 鏡頭 | `nomask` | `tcm` | `hmiou` | `hybrid` |
|---|---|---|---|---|
| camera_1 | 0.3927 | 0.4004 | 0.3895 | 0.3785 |
| camera_2 | 0.5337 | 0.5126 | 0.5166 | 0.5017 |
| camera_3 | 0.4588 | 0.4451 | 0.4483 | 0.4297 |
| camera_4 | 0.3989 | 0.3906 | 0.3875 | 0.3832 |
| camera_7 | 0.4305 | 0.4163 | 0.4283 | 0.4112 |

## 8.4 判準與決策

| 格 | M1 IDF1 ≥ +0.02 | M2 IDSW 少 ≥ 10% | B1 > 0.4599 | H1 召回 | H2 誤偵率 |
|---|---|---|---|---|---|
| `tcm` | −0.0091 **不過** | **多 66.1%,不過** | 不過 | −2.94pp **不過** | +1.36pp 過 |
| `hmiou` | −0.0091 **不過** | **多 11.2%,不過** | 不過 | −1.21pp **不過** | −0.18pp 過 |
| `hybrid` | −0.0236 **不過** | **多 84.3%,不過** | 不過 | −4.71pp **不過** | −1.22pp 過 |

**決策表第 3 列:三格都沒過 → 弱線索在 CHIRLA 上無效 → 依使用者規則停下,整理三輪交給使用者。**

## 8.5 預測逐條

| # | 預測 | 結果 |
|---|---|---|
| 1 | 三格都不過 M1 | **成立**(但方向是 IDF1 下降,不是小幅上升) |
| 2 | 三格誤偵率變化都 < 1pp | **不成立**(`tcm` +1.36pp、`hybrid` −1.22pp;`hmiou` −0.18pp) |
| 3 | `hmiou` 的 IDSW 少於 `nomask` | **不成立**(多 11.2%) |
| 4 | `tcm` 的 IDSW 變化絕對值 < 5% | **不成立**(多 66.1%) |

## 8.6 判準沒涵蓋、必須照實記下的事

1. **`tcm` 與 `hybrid` 讓 track 數暴增**(2,320 → 3,946、4,443)、斷裂(Frag)大增:信心分數差距從相似度扣掉後,
   許多原本會配上的配對掉到門檻下,人沒有換、track 卻斷掉重開
2. **可能原因(未驗證)**:官方 `track_thresh` 0.6 → 分數夾在 [0.6, 1],差距最多 0.4;本輪對映為 0.25 → 夾在 [0.25, 1],
   差距最多 0.75,而 McByte 第 1 階段的融合相似度門檻只有 0.2。**官方權重搬到不同偵測器與門檻上,不能直接適用**
3. **`hmiou` 混人 −26.8%,但 IDSW +11.2%、召回 −1.21pp**:HMIoU ≤ IoU(乘上 ≤ 1 的垂直重疊比例),
   在同樣的配對門檻下更容易配不上 —— 官方 OC-SORT 版本門檻是 0.25,本輪沒有重調(未驗證)
4. **三輪共同的形狀**:凡是讓混人變少的改動(遮罩、HMIoU、TCM),都伴隨斷裂或 ID 切換增加;**沒有任何一個同時改善兩者**
5. `tcm` 在 camera_1 的 IDF1 反而上升(0.3927 → 0.4004),其餘鏡頭下降

## 8.7 這一輪不能回答什麼(§6 照抄並補充)

1. 基底是 McByte 不是 OC-SORT;結果不等於 Hybrid-SORT 本身
2. 分數夾取邊界 0.25(官方 0.6)、權重與門檻都沒調參 —— **本輪不能回答「調過參數的弱線索有沒有用」**
3. 偵測器是 RF-DETR nano(COCO),官方是 YOLOX
4. stride 5;只有 5 台鏡頭;**CHIRLA 是辦公室不是廚房**

## 8.8 下一步

**三輪都沒有有效解決 → 停下。** 三輪總結、檢討與下一步選項見 `docs/M4_修法輪次_20260915.md` 的「三輪總結」,交給使用者決定。
