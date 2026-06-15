# M3 物件偵測準確度(初步)

- 資料集:Roboflow kitchen-object-detection v1(CC BY 4.0),330 張、重疊標註 392 個。
- 只評與 COCO 重疊的 10 類;模型為 **COCO 預訓練、未在廚房微調**。
- ⚠ 初步參考:資料小、單一域、缺 person/cutting_board/生肉;非最終依據。

| model | mAP@50 | mAP@50:95 | knife AP | 小物件 mAP |
|---|---|---|---|---|
| dfine_n | 0.5458 | 0.3535 | 0.1736 | 0.0758 |
| dfine_s | 0.5697 | 0.3668 | 0.1972 | 0.083 |
| dfine_m | 0.6002 | 0.3873 | 0.1944 | 0.0671 |
| rtdetrv2_r18 | 0.5479 | 0.3522 | 0.1774 | 0.103 |
| rtdetrv2_r34 | 0.5684 | 0.3581 | 0.2065 | 0.1161 |
| rfdetr_nano | 0.6253 | 0.4036 | 0.2086 | 0.0797 |
| rfdetr_small | 0.6209 | 0.3976 | 0.187 | 0.0896 |
| rfdetr_medium | 0.6331 | 0.4067 | 0.1837 | 0.0843 |

完整 per-class AP 見 `m3_accuracy.csv`。

> 資料來源:Roboflow kitchen-object-detection (CC BY 4.0)。