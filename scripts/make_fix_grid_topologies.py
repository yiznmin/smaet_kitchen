"""產生 M5 修復網格各格子用的拓撲 YAML。

修法的開關在**拓撲 YAML 的 `fusion` 區塊**,不是命令列參數
(見 `src/m5_reid/spatiotemporal.py` 的 `_DEFAULT_FUSION`):

    F1 margin test      margin: {enabled, min_nats}
    F2 同鏡頭互斥       same_camera_exclusive: {enabled}
    F4 CrossViewLR      cross_view: {enabled, clip, speed_px_per_s, pairs}
    P1 累積投票         revote: {enabled, stride_loops, window, min_votes,
                                switch_margin, assignment}
    P2 Hungarian        revote.assignment: greedy | hungarian

七格的定義出自三份預先登記(`docs/M5_修復_預先登記_20260905.md` §3、
`docs/M5_修復F4_預先登記_20260905.md` §3),**已寫死,不得在這裡改**。
2026-09-13 再加三格(`docs/M5_修復P1_預先登記_20260913.md` §3):p1 / p1f4 / p1p2f4。

⚠ `p1` 從 **base 原檔**派生而不是從 f4 檔 —— 它要與 `base` 只差 `revote` 這一個鍵,
  「P1 單獨的效果」才乾淨。f1/f2/f1f2 沿用 9/12 的作法(從 f4 檔派生再把 cross_view 關掉)。

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

# P1 的取樣間隔**不得用猜的**(P1 預先登記 §2 的 R2)。
# 2026-09-13 在 CHIRLA 七個評估序列自己的 tracks.csv 上重量
# (`measure_evidence_decorrelation.py`),建議值 [1, 2, 2, 3, 80, 120, 120],
# 彙總規則在跑之前先寫死為「中位數,無條件進位」→ **3**。
# ⚠ EPFL 量到的是 8;§6 限制 2 寫死「若與 8 不同就照 CHIRLA 的」。
# ⚠ 七格之間差到 120 倍(seq_006/007/024 的證據幾乎不隨時間變),
#   代表其中三個序列會投出高度相關的票 —— 這正是 R2 警告的那種假自信,
#   結果要照實記進 §8,不得講成「獨立」。
STRIDE_LOOPS = 3

# window / min_votes / switch_margin 是工程判斷不是實驗結果(§6 限制 5),
# 與 `_DEFAULT_FUSION` 的預設一致,在這裡明寫是為了讓拓撲檔自我說明。
def _revote(assignment):
    return {"enabled": True, "stride_loops": STRIDE_LOOPS, "window": 15,
            "min_votes": 3, "switch_margin": 2, "assignment": assignment}


# 四份預先登記 §3 寫死的十格。(F1, F2, F4, revote.assignment 或 None)
CELLS = {
    "base":   (False, False, False, None),
    "f1":     (True,  False, False, None),
    "f2":     (False, True,  False, None),
    "f1f2":   (True,  True,  False, None),
    "f4":     (False, False, True,  None),
    "f4f2":   (False, True,  True,  None),
    "f4f1f2": (True,  True,  True,  None),
    # ── P1 累積投票(2026-09-13)。F1/F2 不進本輪,一律 off。────────────
    "p1":     (False, False, False, "greedy"),
    "p1f4":   (False, False, True,  "greedy"),
    "p1p2f4": (False, False, True,  "hungarian"),
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

    print(f"\n  {'格':<10}{'F1':>5}{'F2':>5}{'F4':>5}{'revote':>11}   輸出")
    print("  " + "-" * 68)
    for cell, (f1, f2, f4, asg) in CELLS.items():
        out = OUT_DIR / f"{cell}.yaml"
        if cell == "base":
            # 原檔複製,不經過 yaml 來回 —— 保證是 V1 要的那個回歸對照
            shutil.copyfile(BASE_YAML, out)
            print(f"  {cell:<10}{'-':>5}{'-':>5}{'-':>5}{'-':>11}   "
                  f"{out.relative_to(ROOT)}  (原檔複製)")
            continue

        # ⚠ p1 從 base 原檔派生,只加 revote 一個鍵 —— 這樣它與 base 的差異就只有
        #   P1 本身。從 f4 檔派生再把 cross_view 關掉行為應該一樣,但檔案裡會多帶
        #   六對單應性,「只差一個開關」這件事就驗不出來了。
        src_yaml = BASE_YAML if cell == "p1" else F4_YAML
        cfg = yaml.safe_load(src_yaml.read_text(encoding="utf-8"))
        fu = cfg["camera_topology"]["fusion"]
        if cell != "p1":
            # min_nats 留 None → spatiotemporal.py 會用 llr_threshold(1.609),
            # 這是 F1/F2 預先登記 §2 寫死的預設,不在這裡另外指定。
            fu["margin"] = {"enabled": bool(f1), "min_nats": None}
            fu["same_camera_exclusive"] = {"enabled": bool(f2)}
            fu.setdefault("cross_view", {})["enabled"] = bool(f4)
            # F5 一律關閉(V2 未通過),但保留空的 links 以便日後重算
            fu.setdefault("transit_place", {})["enabled"] = False
        if asg:
            fu["revote"] = _revote(asg)
        out.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False),
                       encoding="utf-8")
        m = {True: "on", False: "off"}
        print(f"  {cell:<10}{m[f1]:>5}{m[f2]:>5}{m[f4]:>5}{(asg or '-'):>11}   "
              f"{out.relative_to(ROOT)}")

    # 自檢:十份都要載得起來,而且開關要與上表逐格相符
    sys.path.insert(0, str(ROOT / "src"))
    from m5_reid.spatiotemporal import CameraTopology
    print(f"\n  自檢:{'格':<10}{'載入':>6}{'margin':>9}{'同鏡頭互斥':>12}"
          f"{'cross_view 對數':>16}{'revote':>12}{'stride':>8}")
    bad = 0
    for cell, (f1, f2, f4, asg) in CELLS.items():
        t = CameraTopology.from_yaml(OUT_DIR / f"{cell}.yaml")
        rv = t.revote or {}
        got = (t.margin_nats is not None, bool(t.same_cam_exclusive),
               len(t._cross_view) > 0, rv.get("assignment"))
        ok = got == (f1, f2, f4, asg)
        # 取樣間隔是 R2 的核心,寫錯了票數就全部不對 —— 一起檢
        if asg and rv.get("stride_loops") != STRIDE_LOOPS:
            ok = False
        bad += not ok
        print(f"        {cell:<10}{'OK':>6}{str(got[0]):>9}{str(got[1]):>12}"
              f"{len(t._cross_view):>16}{(got[3] or '-'):>12}"
              f"{str(rv.get('stride_loops', '-')):>8}"
              f"{'' if ok else '   [FAIL] 與定義不符'}")

    # p1 必須與 base 只差 revote 一個鍵 —— 這是「P1 單獨的效果」乾不乾淨的前提
    bf = yaml.safe_load((OUT_DIR / "base.yaml").read_text(encoding="utf-8"))["camera_topology"]
    pf = yaml.safe_load((OUT_DIR / "p1.yaml").read_text(encoding="utf-8"))["camera_topology"]
    diff = sorted({k for k in set(bf["fusion"]) | set(pf["fusion"])
                   if bf["fusion"].get(k) != pf["fusion"].get(k)})
    same_rest = ({k: v for k, v in pf.items() if k != "fusion"}
                 == {k: v for k, v in bf.items() if k != "fusion"})
    p1_ok = diff == ["revote"] and same_rest
    bad += not p1_ok
    print(f"\n  p1 vs base:fusion 差異 = {diff}、其餘區塊相同 = {same_rest}"
          f"{'' if p1_ok else '   [FAIL] p1 不是「base + revote」'}")

    print(f"\n  {'[OK] 十格的開關與預先登記 §3 逐格相符' if not bad else f'[FAIL] {bad} 項不符'}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
