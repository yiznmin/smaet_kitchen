# M3 物件偵測速度 benchmark

## 執行環境(數據來源)
- 日期:2026-05-31 16:30
- GPU:**NVIDIA GeForce RTX 3050 Laptop GPU**
- onnxruntime:1.26.0(device=GPU)
- 要求 providers:['CUDAExecutionProvider', 'CPUExecutionProvider']
- 掛載的 CUDA DLL 目錄數:7

> ⚠ 本機若為 RTX 3050(開發機),數字**僅供流程驗證,非選型依據**;
> 最終選型須於目標部署主機重跑(說明文件 C.4 #4)。

## 結果摘要
- 總組合:24;成功 24;失敗 0(OOM/ERROR/缺檔)。
- 完整數據見 `m3_speed_table.csv`。

| model | license | imgsz | batch | lat_mean(ms) | p95 | img/s | gpu(MB) | 鏡頭數 | provider |
|---|---|---|---|---|---|---|---|---|---|
| dfine_n | Apache-2.0 | 672 | 1 | 17.26 | 23.162 | 57.9 | 214.5 | 3 | CUDAExecutionProvider |
| dfine_n | Apache-2.0 | 672 | 4 | 43.429 | 44.901 | 92.1 | 600.5 | 6 | CUDAExecutionProvider |
| dfine_n | Apache-2.0 | 672 | 8 | 81.319 | 83.394 | 98.4 | 1112.5 | 6 | CUDAExecutionProvider |
| dfine_s | Apache-2.0 | 672 | 1 | 25.069 | 27.437 | 39.9 | 346.5 | 2 | CUDAExecutionProvider |
| dfine_s | Apache-2.0 | 672 | 4 | 80.696 | 82.776 | 49.6 | 1114.5 | 3 | CUDAExecutionProvider |
| dfine_s | Apache-2.0 | 672 | 8 | 153.817 | 156.427 | 52.0 | 2138.5 | 3 | CUDAExecutionProvider |
| dfine_m | Apache-2.0 | 672 | 1 | 38.658 | 40.714 | 25.9 | 346.5 | 1 | CUDAExecutionProvider |
| dfine_m | Apache-2.0 | 672 | 4 | 131.729 | 134.246 | 30.4 | 1114.5 | 2 | CUDAExecutionProvider |
| dfine_m | Apache-2.0 | 672 | 8 | 253.196 | 255.537 | 31.6 | 2138.5 | 2 | CUDAExecutionProvider |
| rfdetr_nano | Apache-2.0 | 672 | 1 | 43.287 | 44.027 | 23.1 | 602.5 | 1 | CUDAExecutionProvider |
| rfdetr_nano | Apache-2.0 | 672 | 4 | 151.327 | 152.085 | 26.4 | 2138.5 | 1 | CUDAExecutionProvider |
| rfdetr_nano | Apache-2.0 | 672 | 8 | 12756.467 | 12877.442 | 0.6 | 3802.5 | 0 | CUDAExecutionProvider |
| rfdetr_small | Apache-2.0 | 672 | 1 | 45.242 | 46.771 | 22.1 | 600.5 | 1 | CUDAExecutionProvider |
| rfdetr_small | Apache-2.0 | 672 | 4 | 154.835 | 156.003 | 25.8 | 2136.5 | 1 | CUDAExecutionProvider |
| rfdetr_small | Apache-2.0 | 672 | 8 | 11550.273 | 11654.948 | 0.7 | 3864.5 | 0 | CUDAExecutionProvider |
| rfdetr_medium | Apache-2.0 | 672 | 1 | 45.577 | 46.252 | 21.9 | 602.5 | 1 | CUDAExecutionProvider |
| rfdetr_medium | Apache-2.0 | 672 | 4 | 157.952 | 158.821 | 25.3 | 2138.5 | 1 | CUDAExecutionProvider |
| rfdetr_medium | Apache-2.0 | 672 | 8 | 12811.641 | 12942.084 | 0.6 | 3866.5 | 0 | CUDAExecutionProvider |
| rtdetrv2_r18 | Apache-2.0 | 672 | 1 | 29.277 | 31.856 | 34.2 | 354.5 | 2 | CUDAExecutionProvider |
| rtdetrv2_r18 | Apache-2.0 | 672 | 4 | 97.79 | 99.628 | 40.9 | 1122.5 | 2 | CUDAExecutionProvider |
| rtdetrv2_r18 | Apache-2.0 | 672 | 8 | 189.274 | 191.683 | 42.3 | 2146.5 | 2 | CUDAExecutionProvider |
| rtdetrv2_r34 | Apache-2.0 | 672 | 1 | 38.931 | 40.226 | 25.7 | 346.5 | 1 | CUDAExecutionProvider |
| rtdetrv2_r34 | Apache-2.0 | 672 | 4 | 130.929 | 132.824 | 30.6 | 1114.5 | 2 | CUDAExecutionProvider |
| rtdetrv2_r34 | Apache-2.0 | 672 | 8 | 253.319 | 255.388 | 31.6 | 2138.5 | 2 | CUDAExecutionProvider |