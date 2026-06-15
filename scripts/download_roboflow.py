"""
下載 Roboflow kitchen-object-detection 資料集(COCO 格式)到 data/roboflow_kitchen/。
API key 由環境變數 ROBOFLOW_API_KEY 提供(不寫死於檔案,避免進版控)。

用法(PowerShell):
  $env:ROBOFLOW_API_KEY="你的key"; python scripts/download_roboflow.py
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEST = ROOT / "data" / "roboflow_kitchen"

key = os.environ.get("ROBOFLOW_API_KEY")
if not key:
    sys.exit("請先設定環境變數 ROBOFLOW_API_KEY")

from roboflow import Roboflow

rf = Roboflow(api_key=key)
project = rf.workspace("kitchenobjectdetection").project("kitchen-object-detection-acyvk")
version = project.version(1)
dataset = version.download("coco", location=str(DEST))
print("下載完成:", dataset.location)
