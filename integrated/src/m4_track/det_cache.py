"""M3 偵測結果快取:同一批偵測可重複用來跑追蹤或違規判定,不必每次重跑 RF-DETR。

## 等價性

RF-DETR 的門檻只在最後 `keep = scores > threshold` 套用,所以「以低門檻快取、事後再過濾」
與「直接用高門檻偵測」邏輯上等價。

⚠ torch 比較 `float32 分數 > threshold` 時會把門檻轉成 float32(0.3 → 0.30000001192…)。
  事後過濾一律用 `np.float32(thr)` 轉成分數的 dtype 再比,與 numpy 版本無關。
⚠ 邏輯等價不等於數值等價:GPU 推論若非決定性,兩次偵測仍可能有微小差異。

## 檔案格式

每支影片一個 npz:
    fid       int64  [N]     影片幀號(非迴圈號),已排序
    xyxy      [N,4]          保留 detect_person 回傳的原始 dtype
    conf      [N]
    class_id  [N]
    meta      0 維字串       JSON,欄位見 REQUIRED_META

以影片幀號為索引,所以 stride 1 的快取可以服務任何 stride(stride 5 取 fid 0,5,10…)。
"""
import hashlib
import json
import os
from pathlib import Path

import numpy as np

CACHE_VERSION = 1
REQUIRED_META = ("variant", "weights_id", "person_cls", "cache_thr",
                 "cache_stride", "n_scanned", "max_fid", "complete")

_WEIGHTS_ID = {}


def weights_id(weights):
    """權重檔的身分。None = COCO 預訓。

    ⚠ 用內容雜湊不用路徑:本機與遠端的路徑寫法不同(相對/絕對),
      比路徑會把同一份權重誤判成不同,或把換過內容的同名檔誤判成相同。
    """
    if not weights:
        return None
    p = Path(weights)
    key = str(p.resolve())
    if key not in _WEIGHTS_ID:
        h = hashlib.md5()
        with p.open("rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        _WEIGHTS_ID[key] = h.hexdigest()
    return _WEIGHTS_ID[key]


def save(path, fids, xyxy, conf, class_id, meta):
    """原子寫入:先寫暫存檔再 rename,中途被砍不會留下半截的快取。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = dict(meta, cache_version=CACHE_VERSION)
    missing = [k for k in REQUIRED_META if k not in meta]
    if missing:
        raise ValueError(f"快取 meta 缺欄位 {missing}")
    fids = np.asarray(fids, dtype=np.int64)
    if len(fids) and np.any(np.diff(fids) < 0):
        raise ValueError("fid 必須已排序")
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez(tmp, fid=fids, xyxy=xyxy, conf=conf, class_id=class_id,
             meta=np.array(json.dumps(meta, ensure_ascii=False)))
    os.replace(tmp, path)


class DetCache:
    """讀一支影片的偵測快取,依幀號與門檻回傳 supervision Detections。"""

    def __init__(self, path, expect=None):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"找不到偵測快取 {self.path}")
        with np.load(self.path, allow_pickle=False) as z:
            self.meta = json.loads(str(z["meta"]))
            self._fid = z["fid"]
            self._xyxy = z["xyxy"]
            self._conf = z["conf"]
            self._cls = z["class_id"]
        if self.meta.get("cache_version") != CACHE_VERSION:
            raise ValueError(f"{self.path} 的快取版本 {self.meta.get('cache_version')} "
                             f"≠ {CACHE_VERSION}")
        # ⚠ 拿錯快取(別的模型、別的權重、別的 person_cls)不會報錯,只會安靜地
        #   產出錯的追蹤結果 —— 與 person_cls 接錯會安靜輸出 0 筆同一類的坑。
        for k, v in (expect or {}).items():
            if self.meta.get(k) != v:
                raise ValueError(f"快取 {self.path} 的 {k}={self.meta.get(k)!r},"
                                 f"但本次執行要 {v!r}")

    def get(self, fid, thr):
        import supervision as sv

        cache_thr = float(self.meta["cache_thr"])
        if thr < cache_thr:
            raise ValueError(f"要求門檻 {thr} 低於快取門檻 {cache_thr} "
                             f"—— 被快取濾掉的框救不回來")
        stride = int(self.meta["cache_stride"])
        if fid % stride != 0 or fid > int(self.meta["max_fid"]):
            raise KeyError(f"{self.path} 沒有掃過 fid={fid}"
                           f"(cache_stride={stride}, max_fid={self.meta['max_fid']})")
        lo = int(np.searchsorted(self._fid, fid, side="left"))
        hi = int(np.searchsorted(self._fid, fid, side="right"))
        if hi == lo:
            return sv.Detections.empty()
        conf = self._conf[lo:hi]
        # ⚠ 見檔頭:在 float32 下比較,與 torch 在 predict 裡的判定一致
        keep = conf > np.float32(thr).astype(conf.dtype)
        if not keep.any():
            return sv.Detections.empty()
        return sv.Detections(xyxy=self._xyxy[lo:hi][keep],
                             confidence=conf[keep],
                             class_id=self._cls[lo:hi][keep])
