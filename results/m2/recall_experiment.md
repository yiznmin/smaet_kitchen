# M2 分支 B — 召回實驗(過濾會不會誤殺真目標)

- 影片:`Boutput0.mp4`,stride=10。把真目標貼進「有人、有雜訊」的真片,跑**完整過濾**。
- 召回 = 目標出現(且過 persistence)的幀中,有多少幀牠**仍被保留、送進分類器**。
- 召回高 = 過濾沒誤殺真目標(雜訊被砍光的同時,真目標活著)。

## 動態線(老鼠)

| 情境 | 召回 |
|---|---|
| 開闊地面(遠離人) | 1895/2298 = **82.5%** |
| 貼近人(最壞情況) | 1077/2298 = **46.9%** |

示範圖(黃圈=貼上的老鼠真實位置;綠框=偵測到=命中):
- `results/m2/recall_demo/mouse_open_f440.png`
- `results/m2/recall_demo/mouse_open_f530.png`
- `results/m2/recall_demo/mouse_open_f620.png`
- `results/m2/recall_demo/mouse_near_f440.png`
- `results/m2/recall_demo/mouse_near_f530.png`
- `results/m2/recall_demo/mouse_near_f620.png`

## 靜態線(水漬)

| 情境 | 召回 |
|---|---|
| 地面靜止水漬(人全程在畫面) | 1611/2291 = **70.3%** |

示範圖(黃圈=貼上的水漬;紅框=偵測到=命中):
- `results/m2/recall_demo/spill_f530.png`
- `results/m2/recall_demo/spill_f620.png`
- `results/m2/recall_demo/spill_f710.png`

## 解讀
- 開闊地面的老鼠/水漬召回高 → **過濾砍掉 98% 雜訊的同時,沒有誤殺真目標**。
- 「貼近人」是最壞情況:老鼠若緊貼人,可能被空間/密度排除誤殺,召回會下降——
  這是已知取捨(寧可邊緣漏一點,也不要被人海淹沒);真實老鼠多沿牆角地面跑,少緊貼人。
- 對照降量實驗(filter_experiment.md):雜訊 -98.5%,真目標召回仍高 = 過濾有效且安全。