"""M3 物件偵測的 11 類定義。

兩套 id,差 1:
- COCO 標註檔的 category_id = 1..11(scripts/make_labeler.py 產生、scripts/build_dataset.py 沿用)
- 模型輸出的 class_id     = 0..10
以 CAT2CLS / CLASS2CAT 互轉。類別順序不可更動;新增類別只能加在最後並重新訓練。"""

# 順序 = 模型 class_id 0..10(不可改動)
CLASSES = ["人", "刀具", "砧板", "食材", "鍋鏟", "鍋子", "手", "容器", "抹布", "夾子", "手套"]

NAMES = {i: n for i, n in enumerate(CLASSES)}            # class_id(0..10) -> 名
NAME_TO_CLASS = {n: i for i, n in enumerate(CLASSES)}    # 名 -> class_id
CAT2CLS = {i: i - 1 for i in range(1, len(CLASSES) + 1)}  # COCO cat_id(1..11) -> class_id
CLASS2CAT = {v: k for k, v in CAT2CLS.items()}           # class_id -> COCO cat_id
NAME_TO_CAT = {n: i + 1 for i, n in enumerate(CLASSES)}  # 名 -> COCO cat_id(1..11)

# COCO categories 陣列(資料集/標註用,id 1..11)
COCO_CATEGORIES = [{"id": i + 1, "name": n} for i, n in enumerate(CLASSES)]

# 食安關鍵類(門檻策略:Recall 優先;掃描時用 precision 下限換最大 recall)
FOOD_SAFETY = ["刀具", "食材", "砧板", "人", "手"]

# 畫框時用的 ASCII 標籤(cv2 無法畫中文)。順序對齊 class_id 0..10。
ROMAN = {i: r for i, r in enumerate(
    ["person", "knife", "board", "food", "spatula", "pot",
     "hand", "container", "cloth", "tongs", "glove"])}
