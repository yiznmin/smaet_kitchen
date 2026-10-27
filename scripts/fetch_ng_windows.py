"""NG 準則評估的資料準備:下載團隊事件場次、切事件視窗、建立 M3 全類別快取。

預先登記:docs/M6_NG準則評估_預先登記_20261027.md §2
- 影片:data/epfl_sessions/<參與者>/<場次>/<鏡頭>.mp4(本機,不進 git;EPFL CC BY-NC 4.0 僅內部驗證)
- 視窗:data/ng_windows/<事件>/<鏡頭>.mp4,範圍 [事件起點 − 30, 事件終點 + 60] 秒
- 快取:results/det_cache/ng_windows_s3/<事件>/<鏡頭>.npz
- NG4(食材衛生)場次另做整場快取:results/det_cache/epfl_sessions_full_s3/<參與者>_<場次>/<鏡頭>.npz
已存在的檔案會跳過。

用法:
    python scripts/fetch_ng_windows.py
    python scripts/fetch_ng_windows.py --skip-download
"""
import argparse
import csv
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from fetch_sample import extract_one, read_central_directory   # noqa: E402

URL = "https://zenodo.org/records/15535461/files/Public_release_videos.zip?download=1"
EVENTS = ROOT / "results/ng_clips/team_events_vs_ng.csv"
INDEX = ROOT / "results/ng_clips/ng_windows_index.csv"
EXCLUDE = {"test/YH2003/2023_05_17_09_08_58"}               # 疑為 M3 訓練場次
VIEWS = ["output0", "Aoutput0", "Aoutput1", "Aoutput2", "Aoutput3", "Boutput1", "Boutput2"]
PRE_S, POST_S, FPS = 30.0, 60.0, 30
WEIGHTS = ROOT / "integrated/weights/m3_nano_r2_armA/checkpoint_best_total.pth"
VID_ROOT = ROOT / "data/epfl_sessions"
WIN_ROOT = ROOT / "data/ng_windows"
WIN_CACHE = ROOT / "results/det_cache/ng_windows_s3"
FULL_CACHE = ROOT / "results/det_cache/epfl_sessions_full_s3"
PY = str(ROOT / ".venv/Scripts/python")


def load_events():
    rows = [r for r in csv.DictReader(open(EVENTS, encoding="utf-8-sig")) if r["session_id"] not in EXCLUDE]
    for r in rows:
        pid, sess = r["session_id"].split("/")[1:]
        r["key"] = f"{pid}_{sess}_{r['annotation_id'][:8]}"
        # 起點對齊到整數幀:視窗第 k 幀 = 原影片第 win_start_frame + k 幀
        r["win_start_frame"] = int(max(0.0, float(r["annot_start_sec"]) - PRE_S) * FPS)
        r["win_start"] = r["win_start_frame"] / FPS
        r["win_end"] = float(r["annot_end_sec"]) + POST_S
    return rows


def cache(videos, cams, out_dir, max_frames=-1):
    """一次呼叫快取多支影片(模型只載一次);失敗重試最多 3 次。"""
    todo = [(v, c) for v, c in zip(videos, cams) if not (out_dir / f"{c}.npz").exists() and v.exists()]
    if not todo:
        return
    for attempt in range(3):
        cmd = [PY, str(ROOT / "scripts/cache_detections.py"), "--all-classes", "--weights", str(WEIGHTS),
               "--out-dir", str(out_dir), "--stride", "3", "--max-frames", str(max_frames), "--spot-check", "5",
               "--videos", *[str(v) for v, _ in todo], "--cameras", *[c for _, c in todo]]
        r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")
        ok = all((out_dir / f"{c}.npz").exists() for _, c in todo)
        print(f"  快取 {out_dir.name} 第 {attempt + 1} 次:{'完成' if ok else '失敗 ' + r.stderr[-300:]}", flush=True)
        if ok:
            return


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-download", action="store_true")
    args = ap.parse_args()
    events = load_events()
    sessions = sorted({e["session_id"] for e in events})
    ng4_sessions = sorted({e["session_id"] for e in events if e["event_category"] == "食材衛生"})
    print(f"事件 {len(events)}、場次 {len(sessions)}、NG4 整場 {len(ng4_sessions)}", flush=True)

    if not args.skip_download:
        entries = {e["name"]: e for e in read_central_directory(URL)[1]}
        for s in sessions:
            for v in VIEWS:
                dest = VID_ROOT / s.split("/", 1)[1] / f"{v}.mp4"
                if dest.exists():
                    continue
                e = entries.get(f"Public_release_videos/{s}/videos/{v}.mp4")
                if e is None:
                    print(f"  找不到 {s} {v}", flush=True)
                    continue
                n = extract_one(URL, e, dest)
                print(f"  下載 {s} {v} {n / 1e6:.0f} MB", flush=True)

    # 切視窗(重新編碼;ffmpeg 取 pts ≥ -ss 的第一幀,所以 -ss 給該幀前半幀)
    INDEX.parent.mkdir(parents=True, exist_ok=True)
    with open(INDEX, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["key", "session_id", "annotation_id", "event_category", "ng",
                                          "annot_start_sec", "annot_end_sec", "win_start_frame", "win_start", "win_end"])
        w.writeheader()
        w.writerows({k: e[k] for k in w.fieldnames} for e in events)
    for e in events:
        src = VID_ROOT / e["session_id"].split("/", 1)[1]
        for v in VIEWS:
            out = WIN_ROOT / e["key"] / f"{v}.mp4"
            if out.exists() or not (src / f"{v}.mp4").exists():
                continue
            out.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{(e['win_start_frame'] - 0.5) / FPS:.4f}", "-i", str(src / f"{v}.mp4"),
                            "-t", f"{e['win_end'] - e['win_start']:.3f}", "-an", "-c:v", "libx264",
                            "-preset", "veryfast", "-crf", "18", "-r", "30", str(out)], check=True)
        cache([WIN_ROOT / e["key"] / f"{v}.mp4" for v in VIEWS], VIEWS, WIN_CACHE / e["key"])

    for s in ng4_sessions:
        sid = s.split("/", 1)[1]
        cache([VID_ROOT / sid / f"{v}.mp4" for v in VIEWS], VIEWS, FULL_CACHE / sid.replace("/", "_"))
    print("全部完成", flush=True)


if __name__ == "__main__":
    main()
