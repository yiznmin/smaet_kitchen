# M2 動態偵測 benchmark 結果

- 影片:`Boutput0.mp4`(1280x720 @30fps,EPFL,單人場景)
- 參數:diff_threshold=25, min_motion_ratio=0.002, reference_frame=False

| 指標 | 值 |
|---|---|
| 總幀數 | 23400 |
| 觸發(有動態)幀 | 9608（41.1%） |
| **GPU 可略過幀（省電比例）** | 13792（**58.9%**） |
| M2 處理速度 | 2.501 ms/幀（400 FPS, 純 CPU） |

## 疊框影格
- `results/m2_frames/Boutput0_motion_f3.png`
- `results/m2_frames/Boutput0_motion_f4.png`
- `results/m2_frames/Boutput0_motion_f5.png`
- `results/m2_frames/Boutput0_motion_f6.png`

> 省電比例 = M3(YOLO)因 M2 前過濾而可略過的幀數佔比。
> 注意:此片為持續烹飪動作,觸發率偏高屬正常;空檔多的真實場域省電比例會更高。