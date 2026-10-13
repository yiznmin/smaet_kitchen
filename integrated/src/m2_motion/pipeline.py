"""
M2 雙分支管線:整合分支 A(大事件)與分支 B(小目標專線)。
對應 docs/M2_雙分支重構設計.md §3。

  - 分支 A:MotionDetector(全域 frame-diff)→ has_motion;有動時整張(縮放)送 M3。
            管人員/潑灑/跨區等大動作,分支本身不掛模型。
  - 分支 B 動態線:SmallTargetDetector(MOG2)→ 移動小目標(老鼠)crop。
  - 分支 B 靜態線:StaticTargetDetector(慢速基準比對)→ 靜止新增物(水漬)crop。
  - 兩線 crop 都經同一個多類別 TargetClassifier 確認(老鼠/水漬/拒絕)才保留。

設計觀念:觸發運算 ≠ 觸發存檔。分支 B 常時跑只吃算力;只有分類器確認後才落地。
"""
from m2_motion.detector import MotionDetector
from m2_motion.branch_b import SmallTargetDetector
from m2_motion.static_line import StaticTargetDetector
from m2_motion.classifier import TargetClassifier


class M2Pipeline:
    def __init__(self, branch_a=None, dynamic_line=None, static_line=None, classifier=None):
        self.branch_a = branch_a or MotionDetector()
        self.dynamic_line = dynamic_line or SmallTargetDetector()   # 老鼠(動)
        self.static_line = static_line or StaticTargetDetector()    # 水漬(靜)
        self.classifier = classifier or TargetClassifier()

    def _classify_crops(self, crops, source):
        out = []
        for c in crops:
            is_t, conf = self.classifier.predict(c["crop"])
            # 無分類器(尚無權重)時 is_t=None → 標「待確認」,保留定位結果供檢視/標註
            if is_t is None or is_t:
                out.append(dict(bbox=c["bbox"], blob_bbox=c["blob_bbox"], area=c["area"],
                                source=source, is_target=is_t, confidence=conf, crop=c["crop"]))
        return out

    def process(self, frame_bgr):
        a = self.branch_a.process(frame_bgr)
        dyn = self.dynamic_line.process(frame_bgr)
        sta = self.static_line.process(frame_bgr)

        targets = (self._classify_crops(dyn["crops"], "dynamic")   # 老鼠候選
                   + self._classify_crops(sta["crops"], "static"))  # 水漬候選

        return dict(
            send_to_m3=a["has_motion"],          # 分支 A:整張是否送 M3(RF-DETR)
            motion_ratio=a["motion_ratio"],
            warming=dyn["warming"] or sta["warming"],
            candidates_dynamic=dyn["n_crops"],   # 動態線(老鼠)候選數
            candidates_static=sta["n_crops"],    # 靜態線(水漬)候選數
            targets=targets,                      # 兩線經分類器確認(或待確認)的目標
            classifier_ready=self.classifier.available,
        )
