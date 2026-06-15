# EPFL 樣本影片規格實測

- 檔案:`Boutput0.mp4`
- 解析度:**1280 x 720**
- 幀率:**30.0 fps**
- codec:h264
- 總幀數:23400
- 時長:780.0 秒

## 抽出的影格(目視確認同畫面人數)
- `results/probe/frames/Boutput0_f2340.png`
- `results/probe/frames/Boutput0_f7020.png`
- `results/probe/frames/Boutput0_f11700.png`
- `results/probe/frames/Boutput0_f16379.png`
- `results/probe/frames/Boutput0_f21060.png`

## 對測試矩陣的影響
- 原始解析度為 1280x720：可保留 640 與 1280 兩檔測試。
- 原始幀率 30.0 fps：可支援鏡頭數推算的 target_fps 應 ≤ 此值。

> 資料來源:EPFL-Smart-Kitchen-30 (CC BY-NC 4.0),引用 arXiv 2506.01608。