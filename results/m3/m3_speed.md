# M3 物件偵測速度 benchmark

## 執行環境(數據來源)
- 日期:2026-05-31 14:57
- GPU:**NVIDIA GeForce RTX 3050 Laptop GPU**
- onnxruntime:1.26.0(device=GPU)
- 要求 providers:['CUDAExecutionProvider', 'CPUExecutionProvider']
- 掛載的 CUDA DLL 目錄數:7

> ⚠ 本機若為 RTX 3050(開發機),數字**僅供流程驗證,非選型依據**;
> 最終選型須於目標部署主機重跑(說明文件 C.4 #4)。

## 結果摘要
- 總組合:30;成功 25;失敗 5(OOM/ERROR/缺檔)。
- 完整數據見 `m3_speed_table.csv`。

| model | license | imgsz | batch | lat_mean(ms) | p95 | img/s | gpu(MB) | 鏡頭數 | provider |
|---|---|---|---|---|---|---|---|---|---|
| dfine_n | Apache-2.0 | 640 | 1 | 17.064 | 22.404 | 58.6 | 214.5 | 3 | CUDAExecutionProvider |
| dfine_n | Apache-2.0 | 640 | 4 | 40.037 | 41.546 | 99.9 | 600.5 | 6 | CUDAExecutionProvider |
| dfine_n | Apache-2.0 | 640 | 8 | 75.283 | 77.105 | 106.3 | 1112.5 | 7 | CUDAExecutionProvider |
| dfine_s | Apache-2.0 | 640 | 1 | 23.687 | 25.799 | 42.2 | 346.5 | 2 | CUDAExecutionProvider |
| dfine_s | Apache-2.0 | 640 | 4 | 74.384 | 76.248 | 53.8 | 1114.5 | 3 | CUDAExecutionProvider |
| dfine_s | Apache-2.0 | 640 | 8 | 142.199 | 145.242 | 56.3 | 2138.5 | 3 | CUDAExecutionProvider |
| dfine_m | Apache-2.0 | 640 | 1 | 37.112 | 39.77 | 26.9 | 346.5 | 1 | CUDAExecutionProvider |
| dfine_m | Apache-2.0 | 640 | 4 | 121.906 | 123.943 | 32.8 | 1114.5 | 2 | CUDAExecutionProvider |
| dfine_m | Apache-2.0 | 640 | 8 | 232.537 | 235.417 | 34.4 | 2138.5 | 2 | CUDAExecutionProvider |
| rfdetr_nano | Apache-2.0 | 384 | 1 | 15.636 | 16.631 | 64.0 | 348.5 | 4 | CUDAExecutionProvider |
| rfdetr_nano | Apache-2.0 | 384 | 4 | 50.731 | 51.429 | 78.8 | 604.5 | 5 | CUDAExecutionProvider |
| rfdetr_nano | Apache-2.0 | 384 | 8 | 95.416 | 95.895 | 83.8 | 604.5 | 5 | CUDAExecutionProvider |
| rfdetr_small | Apache-2.0 | 512 | 1 | 27.702 | 28.653 | 36.1 | 346.5 | 2 | CUDAExecutionProvider |
| rfdetr_small | Apache-2.0 | 512 | 4 | 91.768 | 92.314 | 43.6 | 602.5 | 2 | CUDAExecutionProvider |
| rfdetr_small | Apache-2.0 | 512 | 8 | 170.207 | 170.803 | 47.0 | 1114.5 | 3 | CUDAExecutionProvider |
| rfdetr_medium | Apache-2.0 | 576 | 1 | 35.84 | 37.814 | 27.9 | 602.5 | 1 | CUDAExecutionProvider |
| rfdetr_medium | Apache-2.0 | 576 | 4 | 117.464 | 118.043 | 34.1 | 1114.5 | 2 | CUDAExecutionProvider |
| rfdetr_medium | Apache-2.0 | 576 | 8 | 217.615 | 218.213 | 36.8 | 2138.5 | 2 | CUDAExecutionProvider |
| rtdetrv2_r18 | Apache-2.0 | 640 | 1 | 27.573 | 28.454 | 36.3 | 350.5 | 2 | CUDAExecutionProvider |
| rtdetrv2_r18 | Apache-2.0 | 640 | 4 | 90.416 | 91.892 | 44.2 | 606.5 | 2 | CUDAExecutionProvider |
| rtdetrv2_r18 | Apache-2.0 | 640 | 8 | 172.312 | 174.246 | 46.4 | 2142.5 | 3 | CUDAExecutionProvider |
| rtdetrv2_r34 | Apache-2.0 | 640 | 1 | 36.982 | 38.136 | 27.0 | 346.5 | 1 | CUDAExecutionProvider |
| rtdetrv2_r34 | Apache-2.0 | 640 | 4 | 120.899 | 123.227 | 33.1 | 1114.5 | 2 | CUDAExecutionProvider |
| rtdetrv2_r34 | Apache-2.0 | 640 | 8 | 230.014 | 231.815 | 34.8 | 2138.5 | 2 | CUDAExecutionProvider |
| ssdlite_ref | BSD-3 | 320 | 1 | 35.568 | 38.349 | 28.1 | 258.5 | 1 | CUDAExecutionProvider |
| ssdlite_ref | BSD-3 | 320 | 4 | — | — | — | — | — | **ERROR** |
| ssdlite_ref | BSD-3 | 320 | 8 | — | — | — | — | — | **ERROR** |
| yolox_s | Apache-2.0 | - | - | — | — | — | — | — | **MISSING_ONNX** |
| damoyolo_t | Apache-2.0 | - | - | — | — | — | — | — | **MISSING_ONNX** |
| lwdetr_small | Apache-2.0 | - | - | — | — | — | — | — | **MISSING_ONNX** |