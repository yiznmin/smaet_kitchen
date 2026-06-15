"""
實測一支影片的規格,解掉說明文件 C.4 的待確認 #1(解析度/幀率)與 #2(同畫面人數)。

輸出:
  - 解析度、幀率、總幀數、時長、codec(優先用 ffprobe,fallback 用 OpenCV)
  - 抽數張影格存成 png(供目視確認同畫面人數)
  - 把結果寫進 results/video_probe.md

用法:
  python scripts/probe_video.py data/epfl/Boutput0.mp4
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results" / "probe"


def ffprobe_info(path):
    exe = shutil.which("ffprobe")
    if not exe:
        return None
    cmd = [exe, "-v", "error", "-print_format", "json",
           "-show_streams", "-show_format", str(path)]
    out = subprocess.run(cmd, capture_output=True, text=True)
    if out.returncode != 0 or not out.stdout or not out.stdout.strip():
        return None
    try:
        data = json.loads(out.stdout)
    except (json.JSONDecodeError, TypeError):
        return None
    v = next((s for s in data.get("streams", []) if s.get("codec_type") == "video"), None)
    if not v:
        return None
    num, den = (v.get("avg_frame_rate", "0/1").split("/") + ["1"])[:2]
    fps = float(num) / float(den) if float(den) else 0.0
    return {
        "source": "ffprobe",
        "width": v.get("width"),
        "height": v.get("height"),
        "fps": round(fps, 3),
        "codec": v.get("codec_name"),
        "nb_frames": v.get("nb_frames"),
        "duration_s": round(float(data.get("format", {}).get("duration", 0)), 1),
        "pix_fmt": v.get("pix_fmt"),
    }


def opencv_info(path, dump_frames=5):
    import cv2
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV 無法開啟 {path}")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = round(cap.get(cv2.CAP_PROP_FPS), 3)
    nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_dir = RESULTS / "frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    if nframes > 0 and dump_frames > 0:
        idxs = [int(nframes * r) for r in [0.1, 0.3, 0.5, 0.7, 0.9][:dump_frames]]
        for k, idx in enumerate(idxs):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if ok:
                fp = frame_dir / f"{Path(path).stem}_f{idx}.png"
                # cv2.imwrite 無法處理含非 ASCII(中文)的路徑會靜默失敗,
                # 改用 imencode 取得位元組再以 Python 寫檔(可正確處理 Unicode 路徑)。
                success, buf = cv2.imencode(".png", frame)
                if success:
                    fp.write_bytes(buf.tobytes())
                    saved.append(fp.relative_to(ROOT).as_posix())
    cap.release()
    return {"source": "opencv", "width": w, "height": h, "fps": fps,
            "nb_frames": nframes, "frames_saved": saved}


def main():
    if len(sys.argv) < 2:
        sys.exit("用法: python scripts/probe_video.py <video_path>")
    path = Path(sys.argv[1])
    if not path.exists():
        sys.exit(f"檔案不存在: {path}")

    ff = ffprobe_info(path)
    cv = opencv_info(path)

    print("=== ffprobe ===")
    print(json.dumps(ff, ensure_ascii=False, indent=2) if ff else "(ffprobe 不可用)")
    print("\n=== OpenCV ===")
    print(json.dumps(cv, ensure_ascii=False, indent=2))

    # 寫報告
    RESULTS.mkdir(parents=True, exist_ok=True)
    info = ff or cv
    lines = [
        "# EPFL 樣本影片規格實測",
        "",
        f"- 檔案:`{path.name}`",
        f"- 解析度:**{info['width']} x {info['height']}**",
        f"- 幀率:**{info['fps']} fps**",
        f"- codec:{ff['codec'] if ff else 'n/a'}",
        f"- 總幀數:{info.get('nb_frames')}",
        f"- 時長:{ff['duration_s'] if ff else 'n/a'} 秒",
        "",
        "## 抽出的影格(目視確認同畫面人數)",
    ]
    for f in cv.get("frames_saved", []):
        lines.append(f"- `{f}`")
    lines += [
        "",
        "## 對測試矩陣的影響",
        f"- 原始解析度為 {info['width']}x{info['height']}：" +
        ("測 1280 屬上採樣,意義有限,建議只測 ≤ 原生解析度。"
         if info["width"] and info["width"] < 1280
         else "可保留 640 與 1280 兩檔測試。"),
        f"- 原始幀率 {info['fps']} fps：可支援鏡頭數推算的 target_fps 應 ≤ 此值。",
        "",
        "> 資料來源:EPFL-Smart-Kitchen-30 (CC BY-NC 4.0),引用 arXiv 2506.01608。",
    ]
    report = RESULTS / "video_probe.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n報告已寫入 {report.relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    main()
