"""
驗證 M2 動態偵測「是否正確無誤」。三層驗證:

  A. 單元測試(決定性,答案已知):用人造輸入確認演算法邏輯正確
     - 兩張完全相同的影格 → 必須判「無動態」
     - 同一張影格但貼上一塊白方塊(模擬有東西出現)→ 必須判「有動態」
     - 雜訊微擾(±2 亮度)→ 在合理閾值下不應誤判為動態
  B. 視覺抽檢(真實資料):各存幾張被判「有動/無動」的影格,人眼核對分類是否正確
  C. 看內部:把 M2 偵測到的「變化遮罩」疊在畫面上,確認它框的就是真正在動的區域

用法:
  python scripts/verify_m2.py data/epfl/Boutput0.mp4
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from common.video_io import iter_frames, imwrite_unicode   # noqa: E402
from m2_motion.detector import MotionDetector              # noqa: E402

CONFIG = ROOT / "configs" / "benchmark.yaml"
OUT = ROOT / "results" / "m2" / "verify" / "frames"


def make_detector(cfg):
    m = cfg["m2_motion"]
    return MotionDetector(m["diff_threshold"], m["min_motion_ratio"],
                          m["blur_ksize"], m["use_reference_frame"])


def test_A_synthetic(cfg):
    print("=== A. 單元測試(人造輸入,答案已知) ===")
    rng = np.random.RandomState(0)
    base = rng.randint(0, 255, (720, 1280, 3), dtype=np.uint8)

    results = []

    # A1: 兩張完全相同 → 無動態
    d = make_detector(cfg)
    d.process(base)                       # 第一幀建立基準
    r = d.process(base.copy())
    ok = (r["has_motion"] is False)
    results.append(("A1 相同影格→無動態", ok, f"motion_ratio={r['motion_ratio']}"))

    # A2: 貼一塊 200x200 白方塊 → 有動態
    d = make_detector(cfg)
    d.process(base)
    moved = base.copy()
    moved[200:400, 500:700] = 255
    r = d.process(moved)
    ok = (r["has_motion"] is True and r["bbox"] is not None)
    results.append(("A2 出現白方塊→有動態", ok, f"motion_ratio={r['motion_ratio']}, bbox={r['bbox']}"))

    # A3: 全畫面 ±2 亮度雜訊 → 不應誤判(雜訊低於 diff_threshold=25)
    d = make_detector(cfg)
    d.process(base)
    noise = rng.randint(-2, 3, base.shape)            # 每像素 -2..+2 的微弱擾動
    noisy = np.clip(base.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    r = d.process(noisy)
    ok = (r["has_motion"] is False)
    results.append(("A3 微弱雜訊→不誤判", ok, f"motion_ratio={r['motion_ratio']}"))

    # A4: bbox 是否確實框住白方塊(中心應落在 [500,700]x[200,400] 內)
    d = make_detector(cfg)
    d.process(base)
    r = d.process(moved)
    bb = r["bbox"]
    cx, cy = (bb[0] + bb[2]) / 2, (bb[1] + bb[3]) / 2
    ok = (480 < cx < 720 and 180 < cy < 420)
    results.append(("A4 bbox 框住變化區", ok, f"bbox 中心=({cx:.0f},{cy:.0f}),預期≈(600,300)"))

    all_ok = True
    for name, ok, detail in results:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}  —  {detail}")
        all_ok = all_ok and ok
    print(f"  => A 總結:{'全部通過' if all_ok else '有失敗!'}")
    return all_ok


def test_BC_real(cfg, video):
    print("\n=== B/C. 真實資料抽檢 + 內部遮罩視覺化 ===")
    OUT.mkdir(parents=True, exist_ok=True)
    det = make_detector(cfg)

    motion_saved, still_saved = [], []
    NEED = 4

    for fid, ts, frame in iter_frames(video, stride=15):   # 每 0.5 秒抽一張,加速
        r = det.process(frame, return_mask=True)
        label = "motion" if r["has_motion"] else "still"
        bucket = motion_saved if r["has_motion"] else still_saved
        if len(bucket) >= NEED:
            if len(motion_saved) >= NEED and len(still_saved) >= NEED:
                break
            continue

        # 把變化遮罩上紅色疊在畫面上,並標註判定結果
        mask = r["mask"]
        vis = frame.copy()
        red = np.zeros_like(frame); red[:, :, 2] = mask
        vis = cv2.addWeighted(vis, 1.0, red, 0.5, 0)
        if r["bbox"]:
            x1, y1, x2, y2 = r["bbox"]
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        txt = f"{label}  ratio={r['motion_ratio']:.4f}  (thr={cfg['m2_motion']['min_motion_ratio']})"
        cv2.putText(vis, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        fp = OUT / f"{label}_f{fid}.png"
        imwrite_unicode(fp, vis)
        bucket.append(fp.relative_to(ROOT).as_posix())

    print(f"  已存「有動態」抽檢圖:{motion_saved}")
    print(f"  已存「無動態」抽檢圖:{still_saved}")
    print("  → 請人眼核對:motion_* 圖中紅色遮罩應落在真正移動處;still_* 圖應幾乎無紅色")
    return motion_saved, still_saved


def main():
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    video = Path(sys.argv[1]) if len(sys.argv) > 1 else \
        ROOT / cfg["dataset"]["sample_dir"] / Path(cfg["dataset"]["default_sample"]).name

    a_ok = test_A_synthetic(cfg)
    m, s = ([], [])
    if video.exists():
        m, s = test_BC_real(cfg, video)
    else:
        print(f"\n(略過 B/C:找不到影片 {video})")

    report = ROOT / "results" / "m2" / "verify" / "m2_verify.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join([
        "# M2 動態偵測 — 正確性驗證",
        "",
        "## A. 單元測試(人造輸入,答案已知)",
        f"- 結果:**{'全部通過 ✅' if a_ok else '有失敗 ❌'}**",
        "- A1 相同影格→無動態;A2 出現方塊→有動態;A3 微弱雜訊→不誤判;A4 bbox 框住變化區。",
        "",
        "## B/C. 真實資料抽檢(人眼核對)",
        "「有動態」抽檢圖(紅色遮罩應落在移動處):",
        *[f"- `{x}`" for x in m],
        "",
        "「無動態」抽檢圖(應幾乎無紅色):",
        *[f"- `{x}`" for x in s],
        "",
        "> 說明:EPFL 無動靜的 ground-truth 標註,故以「決定性單元測試 + 人眼抽檢」驗證,",
        "> 對應說明文件 M2 驗收標準『空檔正確判無動、有人活動不漏觸發』。",
    ]), encoding="utf-8")
    print(f"\n報告已寫入 {report.relative_to(ROOT).as_posix()}")
    print(f"整體:A 單元測試 {'PASS' if a_ok else 'FAIL'}")


if __name__ == "__main__":
    main()
