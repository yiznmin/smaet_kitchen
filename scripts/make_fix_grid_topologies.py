"""產生 M5 修復網格各格子用的拓撲 YAML。

修法的開關在**拓撲 YAML 的 `fusion` 區塊**,不是命令列參數
(見 `src/m5_reid/spatiotemporal.py` 的 `_DEFAULT_FUSION`):

    F1 margin test      margin: {enabled, min_nats}
    F2 同鏡頭互斥       same_camera_exclusive: {enabled}
    F4 CrossViewLR      cross_view: {enabled, clip, speed_px_per_s, pairs}

七格的定義出自三份預先登記(`docs/M5_修復_預先登記_20260905.md` §3、
`docs/M5_修復F4_預先登記_20260905.md` §3),**已寫死,不得在這裡改**。

⚠ `base` 格**直接複製 `configs/camera_topology.chirla.yaml` 原檔**,不從 f4 檔派生。
  那才是真正的回歸對照 —— 硬性驗收 V1 要求它與 2026-09-04 的 `coco_none` 逐位相同,
  若從 f4 檔派生再把開關關掉,萬一派生過程動到別的欄位就驗不出來了。

⚠ F5(`transit_place`)本輪**不啟用**:`chirla_build_crossview.py` 實測 16 條有向連結
  一條都沒建立(推導集最多的 camera_4→camera_2 只有 14 次,門檻 tp_min=15),
  F5 預先登記的硬性驗收 V2「建立的連結數 > 0」未通過。詳見該份 §8。

用法:
    python scripts/make_fix_grid_topologies.py
"""
import shutil
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
BASE_YAML = ROOT / "configs" / "camera_topology.chirla.yaml"
F4_YAML = ROOT / "configs" / "camera_topology.chirla_f4.yaml"
OUT_DIR = ROOT / "configs" / "fix_grid"

# 三份預先登記 §3 寫死的七格。(F1, F2, F4)
CELLS = {
    "base":   (False, False, False),
    "f1":     (True,  False, False),
    "f2":     (False, True,  False),
    "f1f2":   (True,  True,  False),
    "f4":     (False, False, True),
    "f4f2":   (False, True,  True),
    "f4f1f2": (True,  True,  True),
}


def main():
    for p in (BASE_YAML, F4_YAML):
        if not p.exists():
            raise SystemExit(f"缺 {p.relative_to(ROOT)} —— 先跑 chirla_build_topology.py "
                             f"與 chirla_build_crossview.py")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # ⚠ cross_view / transit_place 在 **fusion 底下**,不是 camera_topology 底下。
    #   第一版少讀一層 → 數到 0 對,誤報「F4 會空轉」而中止。
    src = yaml.safe_load(F4_YAML.read_text(encoding="utf-8"))["camera_topology"]["fusion"]
    n_pairs = len((src.get("cross_view") or {}).get("pairs") or {})
    n_links = len((src.get("transit_place") or {}).get("links") or {})
    print(f"  來源 {F4_YAML.relative_to(ROOT)}:cross_view 有 {n_pairs} 對、"
          f"transit_place 有 {n_links} 條")
    if n_pairs == 0:
        raise SystemExit("cross_view 一對都沒有 —— F4 的格子會是空轉,先查 crossview_fit.json")

    print(f"\n  {'格':<10}{'F1':>5}{'F2':>5}{'F4':>5}   輸出")
    print("  " + "-" * 58)
    for cell, (f1, f2, f4) in CELLS.items():
        out = OUT_DIR / f"{cell}.yaml"
        if cell == "base":
            # 原檔複製,不經過 yaml 來回 —— 保證是 V1 要的那個回歸對照
            shutil.copyfile(BASE_YAML, out)
            print(f"  {cell:<10}{'-':>5}{'-':>5}{'-':>5}   {out.relative_to(ROOT)}  (原檔複製)")
            continue

        cfg = yaml.safe_load(F4_YAML.read_text(encoding="utf-8"))
        fu = cfg["camera_topology"]["fusion"]
        # min_nats 留 None → spatiotemporal.py 會用 llr_threshold(1.609),
        # 這是 F1/F2 預先登記 §2 寫死的預設,不在這裡另外指定。
        fu["margin"] = {"enabled": bool(f1), "min_nats": None}
        fu["same_camera_exclusive"] = {"enabled": bool(f2)}
        fu.setdefault("cross_view", {})["enabled"] = bool(f4)
        # F5 一律關閉(V2 未通過),但保留空的 links 以便日後重算
        fu.setdefault("transit_place", {})["enabled"] = False
        out.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False),
                       encoding="utf-8")
        m = {True: "on", False: "off"}
        print(f"  {cell:<10}{m[f1]:>5}{m[f2]:>5}{m[f4]:>5}   {out.relative_to(ROOT)}")

    # 自檢:七份都要載得起來,而且開關要與上表逐格相符
    sys.path.insert(0, str(ROOT / "src"))
    from m5_reid.spatiotemporal import CameraTopology
    print(f"\n  自檢:{'格':<10}{'載入':>6}{'margin':>9}{'同鏡頭互斥':>12}{'cross_view 對數':>16}")
    bad = 0
    for cell, (f1, f2, f4) in CELLS.items():
        t = CameraTopology.from_yaml(OUT_DIR / f"{cell}.yaml")
        got = (t.margin_nats is not None, bool(t.same_cam_exclusive), len(t._cross_view) > 0)
        ok = got == (f1, f2, f4)
        bad += not ok
        print(f"        {cell:<10}{'OK':>6}{str(got[0]):>9}{str(got[1]):>12}"
              f"{len(t._cross_view):>16}{'' if ok else '   [FAIL] 與定義不符'}")
    print(f"\n  {'[OK] 七格的開關與預先登記 §3 逐格相符' if not bad else f'[FAIL] {bad} 格不符'}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
