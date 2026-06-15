"""
驗證 M2 分支 B(小目標專線)能抓到分支 A 會漏掉的小目標。

核心對照(這正是雙分支存在的理由,對應 docs/M2_雙分支重構設計.md §1):
  造一段「靜止背景 + 一隻很小的移動目標(模擬老鼠)」的合成序列:
    - 分支 A(全域 frame-diff,現有門檻)→ 應「漏抓」(motion_ratio 過不了門檻)
    - 分支 B(MOG2 + 連通元件)→ 應「抓到」並裁切出含目標的 crop

另含:
  - 暖機期不輸出 crop 的檢查
  - (可選)真實影片抽存幾張分支 B 的前景遮罩 + 裁切框供目視

用法:
  python scripts/verify_branch_b.py                # 只跑合成對照
  python scripts/verify_branch_b.py data/epfl/Boutput0.mp4   # 加跑真實資料視覺化
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
from m2_motion.branch_b import SmallTargetDetector         # noqa: E402

CONFIG = ROOT / "configs" / "benchmark.yaml"
OUT = ROOT / "results" / "m2" / "branch_b"


def _make_branch_b(cfg):
    b = cfg["m2_branch_b"]
    return SmallTargetDetector(
        min_blob_area=b["min_blob_area"], large_blob_area=b["large_blob_area"],
        exclusion_margin=b["exclusion_margin"],
        crop_size=b["crop_size"], max_crops_per_frame=b["max_crops_per_frame"],
        mog2_history=b["mog2_history"], mog2_var_threshold=b["mog2_var_threshold"],
        warmup_frames=b["warmup_frames"],
        persistence_frames=b["persistence_frames"], match_max_dist=b["match_max_dist"],
        track_max_miss=b["track_max_miss"],
        density_radius=b["density_radius"], density_max_neighbors=b["density_max_neighbors"])


def _make_branch_a(cfg):
    m = cfg["m2_motion"]
    return MotionDetector(m["diff_threshold"], m["min_motion_ratio"],
                          m["blur_ksize"], m["use_reference_frame"])


def test_synthetic(cfg):
    """靜止背景 + 14x9 小目標移動;比對分支 A(漏)vs 分支 B(抓)。"""
    print("=== 合成對照:小目標 分支 A vs 分支 B ===")
    H, W = 720, 1280
    rng = np.random.RandomState(0)
    # 固定的弱紋理背景(非純色,較貼近真實;但全程不變 → MOG2 學得乾淨)
    bg = (np.full((H, W, 3), 110, np.uint8)
          + rng.randint(-6, 7, (H, W, 3)).astype(np.int16)).clip(0, 255).astype(np.uint8)

    mw, mh = 14, 9                      # 小目標尺寸(像素),佔畫面 ~0.014%
    warmup = cfg["m2_branch_b"]["warmup_frames"]
    a = _make_branch_a(cfg)
    b = _make_branch_b(cfg)

    # 1) 暖機:餵 warmup+5 張純背景(不應有任何 crop)
    crops_during_warmup = 0
    for _ in range(warmup + 5):
        a.process(bg.copy())
        r = b.process(bg.copy())
        crops_during_warmup += r["n_crops"]

    # 2) 目標進場並逐幀移動,記錄兩分支的命中
    a_hits = b_hits = b_contains = n = 0
    for k in range(20):
        x = 200 + k * 40               # 目標水平移動
        y = 360 + (k % 5) * 6
        frame = bg.copy()
        frame[y:y + mh, x:x + mw] = 30  # 暗色小目標
        cx, cy = x + mw // 2, y + mh // 2

        ra = a.process(frame)
        rb = b.process(frame)
        n += 1
        if ra["has_motion"]:
            a_hits += 1
        if rb["n_crops"] > 0:
            b_hits += 1
            # 目標中心是否落在某個 crop 框內
            for c in rb["crops"]:
                x1, y1, x2, y2 = c["bbox"]
                if x1 <= cx < x2 and y1 <= cy < y2:
                    b_contains += 1
                    break

    results = [
        ("暖機期不輸出 crop", crops_during_warmup == 0,
         f"暖機共輸出 {crops_during_warmup} 個 crop(應為 0)"),
        ("分支 A 漏抓小目標(符合預期)", a_hits <= n * 0.2,
         f"分支 A 命中 {a_hits}/{n} 幀(全域門檻下小目標應大多漏掉)"),
        ("分支 B 抓到小目標", b_hits >= n * 0.8,
         f"分支 B 命中 {b_hits}/{n} 幀"),
        ("分支 B 裁切框含目標", b_contains >= n * 0.8,
         f"目標落在 crop 內 {b_contains}/{n} 幀"),
    ]
    all_ok = True
    for name, ok, detail in results:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}  -  {detail}")
        all_ok = all_ok and ok
    print(f"  => 合成對照:{'全部通過' if all_ok else '有失敗'}")
    print(f"     (關鍵結論:分支 A {a_hits}/{n} 漏抓,分支 B {b_hits}/{n} 救回)")
    return all_ok, (a_hits, b_hits, n)


def test_temporal(cfg):
    """時序持續性:連貫移動的目標(老鼠)會被確認;每幀亂跳的斑塊(閃爍)被拒絕。

    做法:餵兩段合成序列給分支 B,看它「確認」了哪些幀(n_crops > 0):
      情境1 連貫移動 — 小目標每幀位移 12px(< match_max_dist)→ 應在第 persistence 幀起被確認
      情境2 閃爍亂跳 — 小斑塊每幀瞬移數百 px(> match_max_dist)→ 應從頭到尾都不被確認
    """
    print("\n=== 時序持續性:連貫移動 vs 閃爍亂跳 ===")
    H, W = 720, 1280
    rng = np.random.RandomState(1)
    bg = (np.full((H, W, 3), 110, np.uint8)
          + rng.randint(-6, 7, (H, W, 3)).astype(np.int16)).clip(0, 255).astype(np.uint8)
    mw, mh = 14, 9
    P = cfg["m2_branch_b"]["persistence_frames"]
    warm = cfg["m2_branch_b"]["warmup_frames"]

    def run(positions):
        d = _make_branch_b(cfg)
        for _ in range(warm + 5):                  # 暖機:餵純背景
            d.process(bg.copy())
        confirms = []
        for (x, y) in positions:
            f = bg.copy()
            f[y:y + mh, x:x + mw] = 30             # 暗色小目標
            confirms.append(d.process(f)["n_crops"] > 0)
        return confirms

    # 情境1:連貫移動(每幀僅位移 12px)
    coherent = run([(150 + k * 12, 400) for k in range(10)])
    first = next((i for i, c in enumerate(coherent) if c), None)
    n1 = sum(coherent)

    # 情境2:每幀瞬移數百 px,模擬閃爍的人碎塊/反光
    flicker = run([(100 + (k * 350) % 1000, 100 + (k * 270) % 500) for k in range(20)])
    n2 = sum(flicker)

    results = [
        (f"連貫移動目標被確認(延遲 ~{P} 幀後)",
         first == P - 1 and n1 >= 10 - P,
         f"首次確認在第 {first} 幀(預期第 {P - 1} 幀),共確認 {n1}/10 幀"),
        ("閃爍亂跳斑塊被拒絕",
         n2 == 0,
         f"亂跳斑塊確認 {n2}/20 幀(預期 0)"),
    ]
    all_ok = True
    for name, ok, detail in results:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}  -  {detail}")
        all_ok = all_ok and ok
    print(f"  => 時序持續性:{'全部通過' if all_ok else '有失敗'}")
    print(f"     (關鍵結論:連貫移動 {n1}/10 確認、閃爍亂跳 {n2}/20 確認)")
    return all_ok, (first, n1, n2)


def test_density(cfg):
    """密度排除:一堆疊在一起的小框(人)被砍掉,孤立的小框(老鼠)被保留。

    直接測 _isolate():給「6 個密集群聚 + 1 個孤立」的斑塊清單,
    應只留下那個孤立的。這正是修正『把人的碎塊當老鼠』的反向判斷。
    """
    print("\n=== 密度排除:密集群聚(人)vs 孤立(老鼠) ===")
    det = _make_branch_b(cfg)
    # 模擬人大動作散成的 6 個密集小框(都擠在 (600,400) 附近 50px 內)
    cluster = [(120, 600 + dx, 400 + dy, 14, 9)
               for dx, dy in [(0, 0), (20, 10), (-15, 8), (10, -12), (30, 5), (-8, -6)]]
    # 一個遠離人群的孤立小框(模擬老鼠,在 (1150, 650))
    lone = [(120, 1150, 650, 14, 9)]
    out, _excl = det._isolate(cluster + lone)
    kept_centers = [(x + w // 2, y + h // 2) for (_, x, y, w, h) in out]
    lone_kept = (1157, 654) in kept_centers
    cluster_dropped = all((cx, cy) == (1157, 654) for (cx, cy) in kept_centers)

    results = [
        ("孤立小框(老鼠)被保留", lone_kept, f"保留中心={kept_centers}(應含孤立的)"),
        ("密集群聚(人)被砍掉", cluster_dropped and len(out) == 1,
         f"保留 {len(out)} 個(應只剩 1 個孤立的)"),
    ]
    all_ok = True
    for name, ok, detail in results:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}  -  {detail}")
        all_ok = all_ok and ok
    print(f"  => 密度排除:{'全部通過' if all_ok else '有失敗'}")
    print("     (修正了『密集=老鼠候選』的反向判斷:密集=人砍掉、孤立=老鼠保留)")
    return all_ok, len(out)


def visual_branch_a(cfg, video):
    """分支 A(全域 frame-diff)抓到什麼:有大動作就整張送 M3。
    畫出 has_motion 與全域變動 bbox(青色),供與分支 B 對照。"""
    print("\n=== 分支 A 視覺化(大事件 → 整張送 M3) ===")
    OUT.mkdir(parents=True, exist_ok=True)
    a = _make_branch_a(cfg)
    saved = []
    motion_frames = 0
    total = 0
    for fid, ts, frame in iter_frames(video, stride=10):
        r = a.process(frame, return_mask=True)
        total += 1
        if r["has_motion"]:
            motion_frames += 1
        # 只存與分支 B 相同的幾個幀號(300~350)方便對照
        if 300 <= fid <= 350 and fid % 10 == 0:
            vis = frame.copy()
            red = np.zeros_like(frame); red[:, :, 2] = r["mask"]
            vis = cv2.addWeighted(vis, 1.0, red, 0.4, 0)
            tag = "send-to-M3" if r["has_motion"] else "skip (no motion)"
            if r["bbox"]:
                x1, y1, x2, y2 = r["bbox"]
                cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 255, 0), 3)  # 青=全域變動框
            cv2.putText(vis, f"BRANCH A: {tag}  motion_ratio={r['motion_ratio']:.4f}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            fp = OUT / f"branch_a_f{fid}.png"
            if imwrite_unicode(fp, vis):
                saved.append(fp.relative_to(ROOT).as_posix())
    print(f"  已存分支 A 標註圖 {len(saved)} 張:{saved}")
    print(f"  分支 A 觸發率:{motion_frames}/{total}(= 會整張送 M3 的比例)")
    print("  → 青框=全域變動範圍(整張都會送 M3);分支 A 抓的是『大動作/有人活動』,不細分小目標")
    return saved, motion_frames, total


def visual_real(cfg, video):
    """真實影片:抽存幾張分支 B 的前景遮罩 + 裁切框,供人眼核對。"""
    print("\n=== 真實資料視覺化(分支 B 前景 + 裁切框) ===")
    OUT.mkdir(parents=True, exist_ok=True)
    crop_dir = OUT / "crops"
    crop_dir.mkdir(parents=True, exist_ok=True)
    b = _make_branch_b(cfg)
    saved, crops_saved = [], []
    excluded_total = 0
    for fid, ts, frame in iter_frames(video, stride=10):
        r = b.process(frame, return_mask=True)
        excluded_total += r["n_excluded"]
        if r["warming"] or (r["n_crops"] == 0 and r["n_excluded"] == 0):
            continue
        vis = frame.copy()
        red = np.zeros_like(frame); red[:, :, 2] = r["mask"]
        vis = cv2.addWeighted(vis, 1.0, red, 0.4, 0)
        # 紅框=被密度排除的「密集群聚」(判定為人 → 砍掉);這就是「密集框選」的佐證
        for ex1, ey1, ex2, ey2 in r["dense_excluded_boxes"]:
            cv2.rectangle(vis, (ex1, ey1), (ex2, ey2), (0, 0, 255), 1)
        # 藍框=被當成「人/大事件」的大區域(排除依據)
        for lx1, ly1, lx2, ly2 in r["large_boxes"]:
            cv2.rectangle(vis, (lx1, ly1), (lx2, ly2), (255, 0, 0), 2)
        # 綠框=保留的小目標裁切窗;黃框=blob 本身
        for i, c in enumerate(r["crops"]):
            x1, y1, x2, y2 = c["bbox"]
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
            bx1, by1, bx2, by2 = c["blob_bbox"]
            cv2.rectangle(vis, (bx1, by1), (bx2, by2), (0, 255, 255), 1)
            # 把實際裁出的小圖單獨存檔(就是會送進分類器的內容)
            if len(crops_saved) < 12:
                cfp = crop_dir / f"crop_f{fid}_c{i}.png"
                if imwrite_unicode(cfp, c["crop"]):
                    crops_saved.append(cfp.relative_to(ROOT).as_posix())
        cv2.putText(vis, f"RED=dense person cluster={r['n_dense_excluded']}  "
                         f"GREEN=kept(mouse cand)={r['n_crops']}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        fp = OUT / f"branch_b_f{fid}.png"
        if imwrite_unicode(fp, vis):
            saved.append(fp.relative_to(ROOT).as_posix())
        if len(saved) >= 6:
            break
    print(f"  已存標註全圖 {len(saved)} 張:{saved}")
    print(f"  已存單獨裁切小圖 {len(crops_saved)} 張(送進分類器的實際內容):{crops_saved}")
    print(f"  空間排除:此次累計排除(人的碎塊){excluded_total} 個小斑塊")
    print("  → 藍框=人區域;綠框=小目標候選(沒有老鼠時,這些多半也是人的碎塊,待分類器拒絕)")
    if b.high_load_frames:
        print(f"  [負載] 有 {b.high_load_frames} 幀小斑塊偏多(已全部保留交分類器,未丟棄);"
              f"若常偏高可調 min_blob_area/exclusion_margin。")
    return saved


def main():
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    ok, (a_hits, b_hits, n) = test_synthetic(cfg)
    ok_t, (first, n1, n2) = test_temporal(cfg)
    ok_d, n_iso = test_density(cfg)
    ok = ok and ok_t and ok_d

    saved = []
    video = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    if video and video.exists():
        visual_branch_a(cfg, video)
        saved = visual_real(cfg, video)
    elif video:
        print(f"\n(略過真實視覺化:找不到影片 {video})")

    report = OUT / "branch_b_verify.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join([
        "# M2 分支 B(小目標專線)— 驗證",
        "",
        "## 合成對照:分支 A vs 分支 B",
        f"- 結果:**{'全部通過 ✅' if ok else '有失敗 ❌'}**",
        f"- 14×9px 小目標(佔畫面 ~0.014%)逐幀移動,共 {n} 幀:",
        f"  - 分支 A(全域 frame-diff):命中 **{a_hits}/{n}**(預期大多漏抓)",
        f"  - 分支 B(MOG2 + 連通元件):命中 **{b_hits}/{n}**(救回小目標)",
        "- 另檢查:暖機期不輸出 crop、分支 B 裁切框確實包住目標。",
        "",
        "## 時序持續性(方法 C):連貫移動 vs 閃爍亂跳",
        f"- 連貫移動小目標:第 **{first}** 幀起被確認(persistence 後),共 **{n1}/10** 幀確認",
        f"- 閃爍亂跳斑塊:**{n2}/20** 幀確認(預期 0 — 證明人碎塊/反光被擋掉)",
        "",
        "## 密度排除:密集群聚(人)vs 孤立(老鼠)",
        f"- 給「6 密集 + 1 孤立」→ 只保留 **{n_iso}** 個(應為 1,孤立的那個)",
        "- 修正『密集小框=老鼠候選』的反向判斷:密集=人(砍)、孤立=老鼠(留)。",
        "",
        "## 真實資料視覺化(分支 B 前景 + 裁切框)",
        *([f"- `{x}`" for x in saved] or ["- (未提供影片,略過)"]),
        "",
        "> 結論:全域門檻會漏掉小目標,改用 MOG2 背景相減 + 連通元件即可救回,",
        "> 驗證雙分支設計(docs/M2_雙分支重構設計.md §1)的必要性。",
        "> 分類器確認層(MobileNet)待權重就緒後接上(設計文件 §5)。",
    ]), encoding="utf-8")
    print(f"\n報告已寫入 {report.relative_to(ROOT).as_posix()}")
    print(f"整體:合成對照 {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
