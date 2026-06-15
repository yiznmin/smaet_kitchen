# M2 動態偵測 — 正確性驗證

## A. 單元測試(人造輸入,答案已知)
- 結果:**全部通過 ✅**
- A1 相同影格→無動態;A2 出現方塊→有動態;A3 微弱雜訊→不誤判;A4 bbox 框住變化區。

## B/C. 真實資料抽檢(人眼核對)
「有動態」抽檢圖(紅色遮罩應落在移動處):
- `results/m2/verify/frames/motion_f15.png`
- `results/m2/verify/frames/motion_f30.png`
- `results/m2/verify/frames/motion_f45.png`
- `results/m2/verify/frames/motion_f60.png`

「無動態」抽檢圖(應幾乎無紅色):
- `results/m2/verify/frames/still_f0.png`
- `results/m2/verify/frames/still_f1695.png`
- `results/m2/verify/frames/still_f1725.png`
- `results/m2/verify/frames/still_f1740.png`

> 說明:EPFL 無動靜的 ground-truth 標註,故以「決定性單元測試 + 人眼抽檢」驗證,
> 對應說明文件 M2 驗收標準『空檔正確判無動、有人活動不漏觸發』。