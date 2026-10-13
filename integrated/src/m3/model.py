"""M3 的 RF-DETR 模型載入(集中處,取代散落在各腳本的 load_model)。

集中的理由是推論解析度(2026-10-13):

微調權重是以 704 訓練的,但原本每支腳本都只寫
`ctor(pretrain_weights=..., num_classes=11)`、沒傳 `resolution`。rfdetr 這時會用變體的
預設值(nano 384、medium 576),並把權重裡的位置編碼內插過去 —— **不報錯,只是變不準**。
同一顆 Arm A 權重在 round2 test 上:384 推論 F1 0.824、704 推論 F1 0.922
(results/m3_acc/resolution_compare_armA_20261013.json)。
文件上早期的 F1 0.81 就是這樣在 384 下量出來的。

所以這裡的預設是 `resolution="auto"`:
  · 沒給權重(COCO 預訓)→ 變體預設值,與改動前完全相同
  · 有給微調權重        → 權重自己的訓練解析度
要重現 2026-10-13 之前用微調權重跑出來的結果,請明確傳 `resolution=384`。
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
