"""M3 的 RF-DETR 模型載入。

推論解析度很重要:微調權重以 704 訓練,若建構時沒傳 resolution,rfdetr 會用變體預設值
(nano 384、medium 576)並內插位置編碼 —— 不報錯,但準確度明顯下降。

因此預設 resolution="auto":
  · 沒給權重(COCO 預訓)→ 變體預設值
  · 有給微調權重        → 權重自己的訓練解析度
"""
from m3.classes import NAMES


def parse_resolution(text):
    """命令列參數 → load_rfdetr 的 resolution。接受 auto / default / 整數。"""
    if text is None:
        return "auto"
    s = str(text).strip().lower()
    if s == "auto":
        return "auto"
    if s in ("default", "none", "null"):
        return None
    return int(s)


def trained_resolution(weights):
    """微調權重的訓練解析度;判斷不出來回 None。

    先讀權重檔的 model_config;rfdetr 的 checkpoint_best_total.pth 沒有這個欄位,
    所以再從位置編碼的格數反推(格數 = 解析度 / patch 邊長,另加 1 個 cls token)。
    只靠 model_config 的話,那種權重會安靜地退回變體預設值。
    """
    import torch
    blob = torch.load(str(weights), map_location="cpu", weights_only=False)
    mc = blob.get("model_config")
    res = mc.get("resolution") if isinstance(mc, dict) else getattr(mc, "resolution", None)
    if res:
        return int(res)
    sd = blob.get("model") or {}
    pos = next((v for k, v in sd.items() if k.endswith("embeddings.position_embeddings")), None)
    patch = next((v for k, v in sd.items()
                  if k.endswith("patch_embeddings.projection.weight")), None)
    if pos is None or patch is None:
        return None
    grid = round((pos.shape[1] - 1) ** 0.5)
    return int(grid * patch.shape[-1]) if grid * grid == pos.shape[1] - 1 else None


def load_rfdetr(variant="nano", weights=None, resolution="auto"):
    """載入 RF-DETR。weights=None 用 COCO 預訓;否則載入 11 類微調權重。

    resolution:"auto"(見檔頭)| None(變體預設值)| 整數(須為 32 的倍數)。
    """
    from rfdetr import RFDETRMedium, RFDETRNano, RFDETRSmall
    ctor = {"nano": RFDETRNano, "small": RFDETRSmall, "medium": RFDETRMedium}[variant]
    if resolution == "auto":
        resolution = trained_resolution(weights) if weights else None
        if weights and resolution is None:
            print("注意:讀不出這顆權重的訓練解析度,退回變體預設值。"
                  "解析度不符時準確度會明顯下降,請明確指定 resolution。")
    kw = {} if resolution is None else {"resolution": int(resolution)}
    if weights:
        return ctor(pretrain_weights=str(weights), num_classes=len(NAMES), **kw)
    return ctor(**kw)


def model_resolution(model):
    """模型實際採用的推論解析度(寫進 run_meta 與 log 用)。"""
    return int(model.model_config.resolution)
