# 預先登記:M4 修法第 2 輪 —— SparseTrack 偽深度分層配對(2026-09-15)

> **本文件在評估集上跑 `l3`、`l8`、`h3l8` 任何一格之前定稿並提交 git。**
> 跑完只追加 §8 之後,§1–§7 不得修改。
> 過程紀錄:`docs/M4_修法輪次_20260915.md`(工作方式:3 輪無效就停;本輪是第 2 輪)。

---

## 1. 為什麼做

第 1 輪(McByte 遮罩)**未達標**:IDF1 +0.0199(門檻 +0.02)、IDSW −5.1%(門檻 10%);但 IDF1 0.4710 是四格最高、混人 −30.7%
(`docs/M4_修法第1輪_McByte_預先登記_20260915.md` §8)。

第 2 輪依原計畫試 **SparseTrack**(TCSVT 2024,MIT)的**偽深度分層配對(DCM)**:
框底越低 = 越靠近鏡頭;把偵測與 track 依偽深度分層,**由近到遠逐層配對**,讓前後站的人不互搶。
只用框、不需模型,跑得快。

**讀官方原始碼後確認的事實**(`docs/M4_修法輪次_20260915.md` 第 2 輪準備):
官方所有設定檔的高分偵測都只有 **1 層**(不分層),**只在第 2 階段(低分偵測)分層**:MOT17 3 層、MOT20 8 層、DanceTrack 12 層。

---

## 2. 工具(本輪新增或修改)

| 檔案 | 改動 | 驗證 |
|---|---|---|
| `src/m4_track/sparse_dcm.py`(新) | `DCMMcByteTracker`:McByte + 官方分層規則;每層內部仍用 McByte 的相似度、分數融合、門檻、遮罩條件 | **已完成**:分層函式與官方原始碼原樣擷取的版本比對 100,000 組隨機資料,每層遮罩全部相同 |
| `src/m4_track/tracker.py` | 新增 `rf_mcbyte_dcm`、`rf_mcbyte_nomask_dcm`;層數由 `backend_params` 給 | **已完成**:層數 1/1 時 5 台 × 7 序列的匯出與第 1 輪 `nomask` **逐位相同**(14 檔) |
| `configs/tracker_dcm_{l1,l3,l8,h3l8}.yaml`(新) | 由 `configs/tracker.yaml` 衍生 | **已完成**:解析後與原檔只差 `backend`、`backend_params` |

⚠ **刻意與官方不同**:每層的配對門檻用 McByte 的(第 1 階段 IoU 融合分數 ≥ 0.2、第 2 階段 IoU ≥ 0.5),
不用 SparseTrack 的 `match_thresh` 與第 2 階段 0.3 距離。目的是**只測「分不分層」這一件事**。

---

## 3. 執行網格(定死)

共同設定與第 1 輪相同:評估集 7 序列、camera_1、2、3、4、7、stride 5、跟丟緩衝 5 秒、門檻 0.10、偵測快取 `results/det_cache/coco_nano`、**不開遮罩**。

| 格 | 設定檔 | 高分層數 | 低分層數 | 性質 |
|---|---|---|---|---|
| `nomask` | — | 1 | 1 | 基準(第 1 輪既有,`results/m4_round1/nomask_dump`;= `l1`) |
| **`l3`** | `tracker_dcm_l3.yaml` | 1 | 3 | **官方 MOT17** |
| **`l8`** | `tracker_dcm_l8.yaml` | 1 | 8 | **官方 MOT20(擁擠場景)** |
| `h3l8` | `tracker_dcm_h3l8.yaml` | 3 | 8 | **探索格**(官方沒有高分分層);不參與判定 |

跑法:每格 5 台鏡頭各一個程序同時跑,`merge_m4_dumps.py` 依鏡頭順序合併;
每格合併後 `make_track_gt.py` → `eval_m4_idf1.py` → `diag_m4_ghosts.py`(與第 1 輪同一套)。

---

## 4. 判準(先寫死)

### 4.1 主判準(`l3`、`l8` 各自對 `nomask`)

| # | 判準 | 門檻 |
|---|---|---|
| M1 | 單鏡頭 IDF1 | **≥ +0.02** |
| M2 | IDSW | **少 ≥ 10%** |

### 4.2 對目前最好的結果

| # | 判準 | 門檻 |
|---|---|---|
| B1 | IDF1 | **> `base` 的 0.4599** |
| B2 | IDF1(記錄用,不參與判定) | 與第 1 輪 `mcbyte` 的 0.4710 比較 |

### 4.3 硬性約束(對 `nomask`)

| # | 約束 |
|---|---|
| H1 | 召回(CLR)下降不超過 1pp |
| H2 | 誤偵率上升不超過 3pp |

### 4.4 決策表

| `l3` 或 `l8` 中至少一格 M1、M2 都過 | H1、H2 | 判定 | 下一步 |
|---|---|---|---|
| 是 | 過 | **分層有效** | 把通過的層數疊到遮罩上(`rf_mcbyte_dcm`,另寫預先登記),看能否合起來達標 |
| 是 | 不過 | **有效但代價超限** | 回報使用者決定 |
| **否** | — | **分層在 CHIRLA 上無效** | 檢討後進**第 3 輪**(Hybrid-SORT 弱線索);第 3 輪仍無效就停下 |

`h3l8` 不論結果都不改變判定;若它通過而官方兩格不通過,只記為「高分分層值得另外登記」的假說。

⚠ 門檻 +0.02、10% 沿用第 1 輪,是判斷值不是實驗結果。

---

## 5. 事先寫下的預測(可被否證)

| # | 預測 | 依據 |
|---|---|---|
| 1 | **`l3`、`l8` 都不過 M1**(IDF1 變化 < +0.02) | 官方只在低分階段分層;低分偵測(0.10~0.25)只接到還在追的 track,不開新 track,而第 1 輪查到的換人多發生在兩人都看得到、高分框重疊時 |
| 2 | `l3`、`l8` 的誤偵率相對 `nomask` 變化 < 1pp | 分層只改第 2 階段(低分偵測接到還在追的 track);開新 track 只用高分偵測,本輪沒有改動那一步 |
| 3 | `l8` 的 IDF1 ≥ `l3` | 官方在擁擠的 MOT20 用 8 層 |
| 4 | `h3l8` 相對 `nomask` 的 IDF1 變化絕對值大於 `l8` | 高分分層會影響主要的第 1 階段配對 |

⚠ 預測 2 的依據只是推論:第 2 階段配對結果會改變哪些 track 被標為 lost,間接影響之後的幀,可能不成立。

---

## 6. 已知限制(報告要照抄,放在結論之前)

1. **每層的配對門檻用 McByte 的,不是 SparseTrack 的**;結果代表「在 McByte 上加分層」,不等於「SparseTrack 本身」
2. **偽深度假設人站在地上、框底 = 腳**:坐著、被桌子擋住下半身時,框底不是腳,深度會錯
3. stride 5(每秒 6 次更新)
4. 只有 5 台鏡頭;**CHIRLA 是辦公室不是廚房**
5. 召回用 CLR 定義;每格只跑一次(不開遮罩時是 CPU 決定性計算,層數 1 的逐位相同檢查支持這點)

---

## 7. 硬性驗收(不通過就是實作或資料錯)

| # | 條件 |
|---|---|
| V1 | 分層函式與官方原始碼等價(**已完成**,100,000 組相同) |
| V2 | 層數 1/1 與第 1 輪 `nomask` 逐位相同(**已完成**,14 檔相同) |
| V3 | 每格各台結果 json 的 `tracker_kwargs` 的 `depth_levels_high` / `depth_levels_low` 與 §3 相同、`enable_mask_manager` = false |
| V4 | 每格 `eval_m4_idf1.py` 自檢通過;對 `results/m4_round1/nomask_dump` 重算的 IDF1 重現 **0.4511**、IDSW **2,619** |
| V5 | 各台紀錄檔無 Traceback |
| V6 | `src/m5_sim/world.py` 未動;既有預先登記 §1–§7 未動 |

### 7.1 遠端指令

```bash
export PYTHONIOENCODING=utf-8
R="D:/新增資料夾/CHIRLA/CHIRLA_data/CHIRLA"
S="seq_004 seq_006 seq_007 seq_020 seq_024 seq_025 seq_026"
K="camera_1 camera_2 camera_3 camera_4 camera_7"
O=results/m4_round2
C="--cache-dir results/det_cache/coco_nano --root $R --seqs $S --stride 5 --lost-buffer-seconds 5 --thr 0.10"

for L in l3 l8 h3l8; do
  for c in $K; do
    .venv/Scripts/python.exe scripts/eval_m4_chirla.py $C --cameras $c --tracker configs/tracker_dcm_$L.yaml \
        --label dcm_${L}_$c --dump-dir $O/parts/dcm_${L}_$c --out $O/parts/dcm_${L}_$c.json > logs/m4_round2_dcm_${L}_$c.log 2>&1 &
  done
  wait
  .venv/Scripts/python.exe scripts/merge_m4_dumps.py --parts $(for c in $K; do echo $O/parts/dcm_${L}_$c; done) \
      --seqs $S --out-dir $O/dcm_${L}_dump
done

for L in nomask l3 l8 h3l8; do
  D=$O/dcm_${L}_dump; [ $L = nomask ] && D=results/m4_round1/nomask_dump
  .venv/Scripts/python.exe scripts/make_track_gt.py --root "$R" --tracks-dir $D --seqs $S --out-dir $O/${L}_track_gt
  .venv/Scripts/python.exe scripts/eval_m4_idf1.py --root "$R" --tracks-dir $D --seqs $S --cameras $K \
      --stride 5 --label $L --out $O/idf1_$L.json
  .venv/Scripts/python.exe scripts/diag_m4_ghosts.py --root "$R" --tracks-dir $D --seqs $S \
      --track-gt-dir $O/${L}_track_gt --out $O/ghosts_$L.json
done
```

---

*本文件於在評估集上跑 `l3`、`l8`、`h3l8` 任何一格之前定稿。跑完只追加 §8「結果」與其後章節。*
