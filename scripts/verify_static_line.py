"""
驗證 M2 靜態線(水漬偵測):靜止的新增物(水漬)被確認;移動物(人)被排除。

核心對照(對應 docs/M2_雙分支重構設計.md §3.3,與動態線剛好相反):
  情境1 靜止水漬 — 固定位置放一塊「不動」的新增物,持續多幀 → 應被確認(靜態目標)
  情境2 移動物(人)— 一塊每幀移動的物體 → 應不被確認(現在在動,被靜態遮罩排除)

另含:(可選)真實影片:靜態線在「全程有人走動」的片上應幾乎不誤觸發(人一直在動)。

用法:
  python scripts/verify_static_line.py
  python scripts/verify_static_line.py data/epfl/Boutput0.mp4
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from common.video_io import iter_frames, imwrite_unicode    # noqa: E402
from m2_motion.static_line import StaticTargetDetector       # noqa: E402

CONFIG = ROOT / "configs" / "benchmark.yaml"
OUT = ROOT / "results" / "m2" / "static_line"

# EPFL Boutput0.mp4(1280×720)視野的「樓地板」概略多邊形(僅供示範;實際部署每鏡頭精算)。
# 排除左側牆面反光、檯面、上半部設備,只留中下方地面。
EPFL_FLOOR = [[330, 715], [1080, 715], [950, 380], [470, 380]]


def _make(cfg, floor_zone=None):
    s = cfg["m2_static_line"]
    return StaticTargetDetector(
        baseline_alpha=s["baseline_alpha"], baseline_diff_thresh=s["baseline_diff_thresh"],
        motion_thresh=s["motion_thresh"], blur_ksize=s["blur_ksize"],
        min_blob_area=s["min_blob_area"], max_blob_area=s["max_blob_area"],
        crop_size=s["crop_size"], warmup_frames=s["warmup_frames"],
        motion_exclusion_margin=s["motion_exclusion_margin"],
        max_aspect_ratio=s["max_aspect_ratio"],
        persistence_frames=s["persistence_frames"], match_max_dist=s["match_max_dist"],
        track_max_miss=s["track_max_miss"],
        density_radius=s["density_radius"], density_max_neighbors=s["density_max_neighbors"],
        floor_zone=floor_zone if floor_zone is not None else s.get("floor_zone"))


def test_synthetic(cfg):
    print("=== 靜態線對照:靜止水漬 vs 移動物(人) ===")
    H, W = 720, 1280
    rng = np.random.RandomState(2)
    bg = (np.full((H, W, 3), 110, np.uint8)
          + rng.randint(-6, 7, (H, W, 3)).astype(np.int16)).clip(0, 255).astype(np.uint8)
    P = cfg["m2_static_line"]["persistence_frames"]
    warm = cfg["m2_static_line"]["warmup_frames"]
    bw, bh = 46, 32                       # 物體尺寸(水漬/人皆用此塊)

    # 情境1:靜止水漬 — 固定位置不動,持續 20 幀
    d1 = _make(cfg)
    for _ in range(warm + 5):
        d1.process(bg.copy())
    spill_hits = 0
    for _ in range(20):
        f = bg.copy()
        f[500:500 + bh, 600:600 + bw] = 40   # 固定位置的暗塊(落地後不動的水漬)
        if d1.process(f)["n_crops"] > 0:
            spill_hits += 1

    # 情境2:移動物(人)— 每幀換位置,持續 20 幀
    d2 = _make(cfg)
    for _ in range(warm + 5):
        d2.process(bg.copy())
    move_hits = 0
    for k in range(20):
        f = bg.copy()
        x = 200 + k * 45                     # 每幀水平移動
        f[400:400 + bh, x:x + bw] = 40
        if d2.process(f)["n_crops"] > 0:
            move_hits += 1

    results = [
        ("靜止水漬被確認", spill_hits >= 5,
         f"靜止塊確認 {spill_hits}/20 幀(persistence={P} 後應持續確認)"),
        ("移動物(人)被排除", move_hits == 0,
         f"移動塊確認 {move_hits}/20 幀(預期 0 — 正在動,被靜態遮罩擋掉)"),
    ]
    all_ok = True
    for name, ok, detail in results:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}  -  {detail}")
        all_ok = all_ok and ok
    print(f"  => 靜態線對照:{'全部通過' if all_ok else '有失敗'}")
    print(f"     (關鍵結論:靜止水漬 {spill_hits}/20 確認、移動物 {move_hits}/20 確認)")
    return all_ok, (spill_hits, move_hits)


def test_aspect(cfg):
    """長寬比過濾:扁平水漬保留(不誤砍)、瘦高的腿排除。

    同時放兩塊靜止物:扁平水漬(寬>高)+ 瘦高腿(高>寬),持續多幀。
    應只保留扁平那塊;若扁平的被砍 → 即為誤砍(FAIL)。
    """
    print("\n=== 長寬比過濾:扁平水漬(留) vs 瘦高腿(砍),測誤砍 ===")
    H, W = 720, 1280
    rng = np.random.RandomState(3)
    bg = (np.full((H, W, 3), 110, np.uint8)
          + rng.randint(-6, 7, (H, W, 3)).astype(np.int16)).clip(0, 255).astype(np.uint8)
    warm = cfg["m2_static_line"]["warmup_frames"]
    d = _make(cfg)
    for _ in range(warm + 5):
        d.process(bg.copy())

    # 扁平水漬:寬 80 × 高 24(ratio 0.3);瘦高腿:寬 24 × 高 90(ratio 3.75)
    spill_c, leg_c = (600 + 80 // 2, 500 + 24 // 2), (300 + 24 // 2, 430 + 90 // 2)
    spill_kept = leg_kept = False
    for _ in range(20):
        f = bg.copy()
        f[500:524, 600:680] = 40        # 扁平水漬
        f[430:520, 300:324] = 40        # 瘦高腿
        for c in d.process(f)["crops"]:
            x1, y1, x2, y2 = c["bbox"]
            if x1 <= spill_c[0] < x2 and y1 <= spill_c[1] < y2:
                spill_kept = True
            if x1 <= leg_c[0] < x2 and y1 <= leg_c[1] < y2:
                leg_kept = True

    results = [
        ("扁平水漬被保留(沒誤砍)", spill_kept, f"水漬保留={spill_kept}(必須 True,否則即誤砍)"),
        ("瘦高腿被排除", not leg_kept, f"腿保留={leg_kept}(應 False)"),
    ]
    all_ok = True
    for name, ok, detail in results:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}  -  {detail}")
        all_ok = all_ok and ok
    print(f"  => 長寬比過濾:{'全部通過(扁平水漬沒被誤砍)' if all_ok else '有失敗'}")
    return all_ok, (spill_kept, leg_kept)


def visual_real(cfg, video):
    print("\n=== 真實資料:靜態線(含樓地板 ROI)在『全程有人』片上的誤觸發 ===")
    OUT.mkdir(parents=True, exist_ok=True)
    d = _make(cfg, floor_zone=EPFL_FLOOR)            # 套用 EPFL 概略地板多邊形
    floor_pts = np.array(EPFL_FLOOR, np.int32)
    saved = []
    trig_frames = 0
    total = 0
    for fid, ts, frame in iter_frames(video, stride=10):
        r = d.process(frame, return_mask=True)
        total += 1
        if r["warming"] or r["n_crops"] == 0:
            continue
        trig_frames += 1
        if len(saved) < 6:
            vis = frame.copy()
            grn = np.zeros_like(frame); grn[:, :, 1] = r["mask"]
            vis = cv2.addWeighted(vis, 1.0, grn, 0.4, 0)
            cv2.polylines(vis, [floor_pts], True, (255, 255, 0), 2)   # 青=樓地板 ROI
            for c in r["crops"]:
                x1, y1, x2, y2 = c["bbox"]
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(vis, f"STATIC (floor ROI) kept={r['n_crops']}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            fp = OUT / f"static_f{fid}.png"
            if imwrite_unicode(fp, vis):
                saved.append(fp.relative_to(ROOT).as_posix())
    print(f"  觸發幀:{trig_frames}/{total}(此片無水漬,人一直走動 → 理想應很低)")
    print(f"  已存 {len(saved)} 張:{saved}")
    return saved, trig_frames, total


def main():
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    ok, (spill, move) = test_synthetic(cfg)
    ok_a, (sp_kept, leg_kept) = test_aspect(cfg)
    ok = ok and ok_a

    saved, trig, total = [], 0, 0
    video = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    if video and video.exists():
        saved, trig, total = visual_real(cfg, video)
    elif video:
        print(f"\n(略過真實視覺化:找不到影片 {video})")

    OUT.mkdir(parents=True, exist_ok=True)
    report = OUT / "static_verify.md"
    report.write_text("\n".join([
        "# M2 靜態線(水漬偵測)— 驗證",
        "",
        "## 合成對照:靜止水漬 vs 移動物(人)",
        f"- 結果:**{'全部通過 ✅' if ok else '有失敗 ❌'}**",
        f"- 靜止水漬(固定不動):確認 **{spill}/20** 幀(應持續確認)",
        f"- 移動物(人,每幀換位):確認 **{move}/20** 幀(預期 0 — 正在動被排除)",
        "",
        "## 長寬比過濾(測誤砍)",
        f"- 扁平水漬保留:**{sp_kept}**(必須 True,否則誤砍);瘦高腿排除:**{not leg_kept}**",
        "",
        "## 真實資料(此片無水漬,人全程走動)",
        f"- 靜態線觸發幀:**{trig}/{total}**(理想很低;殘留多為人偶爾站定,待分類器拒絕)" if total
        else "- (未提供影片,略過)",
        *([f"- `{x}`" for x in saved]),
        "",
        "> 結論:靜態線用『慢速基準比對 + 沒在動 + 持續存在』抓落地後靜止的水漬,",
        "> 與動態線(老鼠=連貫移動)互補。多類別分類器接上後做最終 老鼠/水漬/拒絕 判斷。",
    ]), encoding="utf-8")
    print(f"\n報告已寫入 {report.relative_to(ROOT).as_posix()}")
    print(f"整體:靜態線合成對照 {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
