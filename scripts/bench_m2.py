"""
M2 動態偵測 benchmark。

對一支影片逐幀跑 M2,統計:
  - 觸發比例:被判定 has_motion 的幀數佔比
  - GPU 省電比例:M3 可略過的幀數佔比(= 1 - 觸發比例),即 M2 前過濾替 YOLO 省下的無效推論
  - M2 本身的 CPU 處理速度(每幀 ms、FPS)— 確認 M2 夠輕量
並抽存數張「有動態 + bbox 疊框」影格供目視。

用法:
  python scripts/bench_m2.py                       # 用設定檔 default_sample 取下的檔
  python scripts/bench_m2.py data/epfl/Boutput0.mp4
"""
import sys
import time
from pathlib import Path

import cv2
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from common.video_io import iter_frames, imwrite_unicode   # noqa: E402
from m2_motion.detector import MotionDetector              # noqa: E402

CONFIG = ROOT / "configs" / "benchmark.yaml"


def main():
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    m2cfg = cfg["m2_motion"]

    if len(sys.argv) > 1:
        video = Path(sys.argv[1])
    else:
        name = Path(cfg["dataset"]["default_sample"]).name
        video = ROOT / cfg["dataset"]["sample_dir"] / name
    if not video.exists():
        sys.exit(f"找不到影片 {video};請先跑 scripts/fetch_sample.py")

    det = MotionDetector(
        diff_threshold=m2cfg["diff_threshold"],
        min_motion_ratio=m2cfg["min_motion_ratio"],
        blur_ksize=m2cfg["blur_ksize"],
        use_reference_frame=m2cfg["use_reference_frame"],
    )

    total = 0
    motion = 0
    proc_time = 0.0
    saved = []
    out_dir = ROOT / "results" / "m2" / "bench"
    frame_dir = out_dir / "frames"

    print(f"跑 M2 於 {video.name} …")
    for fid, ts, frame in iter_frames(video):
        t0 = time.perf_counter()
        r = det.process(frame)
        proc_time += time.perf_counter() - t0
        total += 1
        if r["has_motion"]:
            motion += 1
            # 只存前 4 張有動態的疊框影格供目視
            if len(saved) < 4 and r["bbox"]:
                x1, y1, x2, y2 = r["bbox"]
                vis = frame.copy()
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(vis, f"motion {r['motion_ratio']:.3f}", (x1, max(0, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                fp = frame_dir / f"{video.stem}_motion_f{fid}.png"
                if imwrite_unicode(fp, vis):
                    saved.append(fp.relative_to(ROOT).as_posix())

    trigger_ratio = motion / total if total else 0.0
    savings = 1 - trigger_ratio
    ms_per_frame = proc_time / total * 1000 if total else 0.0
    m2_fps = total / proc_time if proc_time else 0.0

    print("\n=== M2 動態偵測結果 ===")
    print(f"總幀數          : {total}")
    print(f"觸發(有動態)幀 : {motion}  ({trigger_ratio:.1%})")
    print(f"GPU 可略過幀    : {total - motion}  (省電比例 {savings:.1%})")
    print(f"M2 處理速度     : {ms_per_frame:.3f} ms/幀  ({m2_fps:.0f} FPS, 純 CPU)")
    print(f"疊框影格        : {saved}")

    out_dir.mkdir(parents=True, exist_ok=True)
    report = out_dir / "m2_motion.md"
    report.write_text("\n".join([
        "# M2 動態偵測 benchmark 結果",
        "",
        f"- 影片:`{video.name}`(1280x720 @30fps,EPFL,單人場景)",
        f"- 參數:diff_threshold={m2cfg['diff_threshold']}, "
        f"min_motion_ratio={m2cfg['min_motion_ratio']}, "
        f"reference_frame={m2cfg['use_reference_frame']}",
        "",
        "| 指標 | 值 |",
        "|---|---|",
        f"| 總幀數 | {total} |",
        f"| 觸發(有動態)幀 | {motion}（{trigger_ratio:.1%}） |",
        f"| **GPU 可略過幀（省電比例）** | {total - motion}（**{savings:.1%}**） |",
        f"| M2 處理速度 | {ms_per_frame:.3f} ms/幀（{m2_fps:.0f} FPS, 純 CPU） |",
        "",
        "## 疊框影格",
        *[f"- `{f}`" for f in saved],
        "",
        "> 省電比例 = M3(YOLO)因 M2 前過濾而可略過的幀數佔比。",
        "> 注意:此片為持續烹飪動作,觸發率偏高屬正常;空檔多的真實場域省電比例會更高。",
    ]), encoding="utf-8")
    print(f"\n報告已寫入 {report.relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    main()
