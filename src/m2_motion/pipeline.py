"""
M2 雙分支管線:整合分支 A(大事件)與分支 B(小目標專線)。
對應 docs/M2_雙分支重構設計.md §3。

  - 分支 A:MotionDetector(全域 frame-diff)→ has_motion;有動時整張(縮放)送 M3。
            管人員/潑灑/跨區等大動作,分支本身不掛模型。
  - 分支 B:SmallTargetDetector(MOG2 定位 + 連通元件 + 裁切)→ 小目標 crop,
            再經 TargetClassifier 確認「是不是目標」才保留。

設計觀念:觸發運算 ≠ 觸發存檔。分支 B 常時跑只吃算力;只有分類器確認後才落地。
"""
from m2_motion.detector import MotionDetector
from m2_motion.branch_b import SmallTargetDetector
from m2_motion.classifier import TargetClassifier


class M2Pipeline:
    def __init__(self, branch_a=None, branch_b=None, classifier=None):
        self.branch_a = branch_a or MotionDetector()
        self.branch_b = branch_b or SmallTargetDetector()
        self.classifier = classifier or TargetClassifier()

    def process(self, frame_bgr):
        a = self.branch_a.process(frame_bgr)
        b = self.branch_b.process(frame_bgr)

        targets = []
        for c in b["crops"]:
            is_t, conf = self.classifier.predict(c["crop"])
            # 無分類器(尚無權重)時 is_t=None → 標「待確認」,保留定位結果供檢視/標註
            if is_t is None or is_t:
                targets.append(dict(bbox=c["bbox"], blob_bbox=c["blob_bbox"],
                                    area=c["area"], is_target=is_t, confidence=conf,
                                    crop=c["crop"]))

        return dict(
            send_to_m3=a["has_motion"],          # 分支 A:整張是否送 M3(RF-DETR)
            motion_ratio=a["motion_ratio"],
            branch_b_warming=b["warming"],        # 分支 B 背景模型是否仍暖機中
            candidates=b["n_crops"],              # 分支 B 定位到幾個小目標候選
            targets=targets,                      # 經分類器確認(或待確認)的小目標
            classifier_ready=self.classifier.available,
        )
