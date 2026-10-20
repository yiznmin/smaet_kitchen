"""框提示分割(SAM 2.1 / EdgeTAM,transformers 版)。

一張圖的影像編碼只算一次(embed),之後可用多組框提示重複解碼(masks)。
"""
import numpy as np
import torch
import transformers

MODELS = {
    "sam2.1_b+": ("facebook/sam2.1-hiera-base-plus", "Sam2Model", "Sam2Processor"),
    # EdgeTAM 官方 repo(facebook/EdgeTAM)只有原始 .pt;transformers 文件指定的轉換版
    "edgetam": ("yonigozlan/EdgeTAM-hf", "EdgeTamModel", "Sam2Processor"),
}


class BoxSegmenter:
    def __init__(self, name, device="cuda"):
        repo, model_cls, proc_cls = MODELS[name]
        self.model = getattr(transformers, model_cls).from_pretrained(repo).to(device).eval()
        self.proc = getattr(transformers, proc_cls).from_pretrained(repo)
        self.device = device

    @torch.inference_mode()
    def embed(self, rgb):
        """rgb: [H,W,3] uint8 → (影像特徵, 原始尺寸)。"""
        inp = self.proc(images=rgb, return_tensors="pt").to(self.device)
        return self.model.get_image_embeddings(inp["pixel_values"]), inp["original_sizes"]

    @torch.inference_mode()
    def masks(self, emb, boxes):
        """boxes: [N,4] xyxy(原圖座標)→ bool [N,H,W];N = 0 時回傳空陣列。"""
        feats, sizes = emb
        h, w = sizes[0].tolist()
        if len(boxes) == 0:
            return np.zeros((0, h, w), bool)
        b = self.proc(images=None, input_boxes=[np.asarray(boxes, float).tolist()],
                      original_sizes=sizes.cpu(), return_tensors="pt")
        out = self.model(image_embeddings=feats, input_boxes=b["input_boxes"].to(self.device),
                         multimask_output=False)
        m = self.proc.post_process_masks(out.pred_masks.cpu(), sizes.cpu())[0]
        return m[:, 0].numpy().astype(bool)
