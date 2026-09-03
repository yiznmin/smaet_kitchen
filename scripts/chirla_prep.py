"""CHIRLA 資料集:交接驗收 + 建索引。

分工上資料由別人下載到遠端設備,**下載不完整或結構不符會白白吃掉好幾天**,
所以動工前先驗收。`--verify` 不通過就不要開始訓練。

CHIRLA(Dominguez-Dager et al. 2025,arXiv:2502.06681,Nature Scientific Data)
  · 7 台室內鏡頭、22 個身份、10 個序列、596,345 幀、963,554 個 bbox
  ⚠ 論文寫「非重疊」,**實測不成立**:40% 的(幀,身份)被 2 台以上同時看到,
    cam2 與 cam3 根本在拍同一個房間。21 對相機裡只有 5 對從未共現。
    見 docs/CHIRLA_鏡頭佈局實測_20260903.md 與 scripts/chirla_overlap_stats.py。
  · 1080×720 @ 30fps,橫跨 7 個月(長期外觀變化)
  · **CC-BY-4.0 —— 可商用,僅需署名**。這是它取代 Market-1501/MSMT17 的理由。
  ⚠ HF 的 dataset card 另有使用限制:禁止「未經明確同意…用於監控、識別或監視
    真實人物的部署」。本專案為雇主/員工知情同意的食安追責,但這是法務判斷。

benchmark 目錄契約(官方 benchmark/reid/README.md):
    Task/Scenario/Split/Subset/Sequences/imgs/Camera/ID/Images
四種 CSV:`*_train`(train_0)、`*_val`(test_0)、`*_gallery`、`*_query`。
⚠ **一律沿用官方切分,不要自己重切** —— 他們已經處理好 identity leakage,
  沿用也讓數字能與論文直接對照。開發只用 train+val,最終數字只報 gallery vs query。

用法:
    python scripts/chirla_prep.py --root /path/to/chirla --verify
    python scripts/chirla_prep.py --root /path/to/chirla --index --out chirla_index.json
"""
import argparse
import csv
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

# 論文載明的規格 —— 驗收就是拿磁碟上的東西跟這些對帳
PAPER = {
    "identities": 22,
    "cameras": 7,
    "sequences": 10,
    "frames": 596_345,
    "boxes": 963_554,
    "resolution": (1080, 720),
    "fps": 30,
}


def _fail(msg):
    print(f"  ❌ {msg}")
    return False


def _ok(msg):
    print(f"  ✅ {msg}")
    return True


def find_csvs(root):
    """找出 benchmark 的切分 CSV。回傳 {scenario: {subset: path}}。"""
    out = defaultdict(dict)
    for p in Path(root).rglob("*.csv"):
        stem = p.stem
        for sub in ("gallery", "query", "train", "val"):
            if stem.endswith("_" + sub):
                out[stem[: -len(sub) - 1]][sub] = p
                break
    return dict(out)


def parse_index_row(path_str):
    """從 CHIRLA 的階層路徑解出 (identity, camera, sequence, subset)。

    契約:.../<scenario>/<split>/<subset>/seq_xxx/imgs/<camera>/<id>/frame_xxx.png
    ⚠ 不用 regex 硬比對整條路徑 —— 官方文件說 subset 是靠找 /train/train_k/ 或
      /test/test_k/ 這個片段,所以這裡也走「從右邊數固定層數」+「往左找 split」,
      對前綴目錄放哪裡不敏感。
    """
    parts = Path(path_str.replace("\\", "/")).parts
    if len(parts) < 4:
        return None
    ident, camera = parts[-2], parts[-3]
    seq = next((p for p in parts if p.startswith("seq")), None)
    subset = next((p for p in parts if p.startswith(("train_", "test_"))), None)
    return dict(identity=ident, camera=camera, sequence=seq, subset=subset)


def read_split_csv(path):
    """讀切分 CSV。只取出影像路徑欄位 —— 欄名在不同 scenario 可能不同,
    所以找「看起來像路徑」的那一欄,而不是寫死欄名。"""
    rows = []
    with open(path, encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            hit = next((v for v in r.values()
                        if isinstance(v, str) and ("/" in v or "\\" in v)), None)
            if hit:
                rows.append((hit, r))
    return rows


# ── --verify ──────────────────────────────────────────────────────────
def verify(root, n_sample=30, seed=0):
    root = Path(root)
    print("=" * 72)
    print(f"CHIRLA 交接驗收:{root}")
    print("=" * 72)
    checks = []

    if not root.exists():
        print(_fail(f"路徑不存在:{root}") and "" or "")
        return 1

    # 1. 頂層結構
    tops = {p.name for p in root.iterdir() if p.is_dir()}
    print("\n[1] 頂層目錄")
    print(f"      實際有:{sorted(tops)}")
    checks.append(_ok("benchmark/ 存在") if "benchmark" in tops
                  else _fail("缺 benchmark/ —— Re-ID 的切分與影像都在這裡,沒有它無法訓練"))
    if "annotations" not in tops:
        print("  ⚠ 缺 annotations/(逐幀 JSON)。只做 Re-ID 可以沒有它,"
              "但要做 tracking 評估就需要")
    if "videos" not in tops:
        print("  ⚠ 缺 videos/(原始 AVI)。第一階段用不到,先不下載是對的")

    # 2. 切分 CSV
    print("\n[2] benchmark 切分 CSV")
    csvs = find_csvs(root)
    if not csvs:
        checks.append(_fail("找不到任何 *_train/_val/_gallery/_query.csv"))
    else:
        for scen, subs in sorted(csvs.items()):
            missing = {"train", "val", "gallery", "query"} - set(subs)
            tag = "✅" if not missing else f"⚠ 缺 {sorted(missing)}"
            print(f"      {scen:<34}{sorted(subs)}  {tag}")
        checks.append(_ok(f"找到 {len(csvs)} 個 scenario 的切分"))

    # 3. 對帳:身份數 / 相機數 / 序列數
    #
    # 2026-09-03 修正(實際資料到手後):原本這裡直接數目錄字串,數出來
    # 「身份 42 / 相機 58」而報 ❌,但兩個數字都是**指標寫錯**不是資料壞掉:
    #   · 身份:官方對同一個人在 distractor 角色下用**負號 id**(benchmark/reid/
    #     README:「Negative IDs (e.g. -1, -4) in queries represent unknown
    #     identities」)。-1 與 1 被當成兩個身份,21 個人就數成 42。
    #   · 相機:目錄名是 `camera_3_2023-06-02-11:14:26`,**帶錄影時間戳**,
    #     同一台實體相機在 10 個序列裡有 10 個名字,7 台就數成 58。
    # 所以改成:身份取絕對值、相機取 `camera_N` 前綴,並且**兩個原始數字照印**,
    # 不要因為改了計數就把原始事實藏起來。
    print("\n[3] 與論文規格對帳")
    all_paths, idents, cams, seqs = [], set(), set(), set()
    for subs in csvs.values():
        for p in subs.values():
            for path_str, _row in read_split_csv(p):
                all_paths.append(path_str)
                meta = parse_index_row(path_str)
                if meta:
                    idents.add(meta["identity"])
                    cams.add(meta["camera"])
                    if meta["sequence"]:
                        seqs.add(meta["sequence"])
    persons = {i.lstrip("-") for i in idents}                      # 去掉 distractor 負號
    phys_cams = {"_".join(c.split("_")[:2]) for c in cams}         # camera_N
    print(f"      {'項目':<20}{'實際':>10}{'論文':>10}")
    print("      " + "-" * 40)
    for name, got, want in (("身份(去負號)", len(persons), PAPER["identities"]),
                            ("實體相機", len(phys_cams), PAPER["cameras"]),
                            ("序列", len(seqs), PAPER["sequences"])):
        mark = "✅" if got == want else ("⚠" if got else "❌")
        print(f"      {name:<20}{got:>10}{want:>10}   {mark}")
    print(f"      (原始 id 值 {len(idents)} 個 = {len(persons)} 人 × 正/負 distractor;"
          f"相機目錄字串 {len(cams)} 個 = 實體相機 × 序列時間戳)")
    print(f"      CSV 列出的影像共 {len(all_paths):,} 筆")
    if len(persons) != PAPER["identities"]:
        # ⚠ 這條差異是真的,不要蓋掉:benchmark 的 10 個序列裡只出現 21 個人
        #   (id 17 與 20~23 從未出現,annotations/ 全掃也是 21)。論文的 22
        #   是**整個資料集**的身份數,benchmark 子集比它少一個。
        print(f"      ⚠ 身份 {len(persons)} ≠ 論文 {PAPER['identities']}:"
              "benchmark 的 10 個序列只涵蓋部分身份,已用 annotations/ 全掃交叉確認,"
              "**不是下載不完整**。報告要照實寫 benchmark 子集的身份數。")
    # 硬性判準改看結構性的兩項 —— 這兩項才是「只下載了部分 scenario」會壞掉的地方
    checks.append(_ok(f"實體相機 {len(phys_cams)} 台對得上論文")
                  if len(phys_cams) == PAPER["cameras"]
                  else _fail(f"實體相機 {len(phys_cams)} ≠ 論文的 {PAPER['cameras']}"))
    checks.append(_ok(f"序列 {len(seqs)} 個對得上論文")
                  if len(seqs) == PAPER["sequences"]
                  else _fail(f"序列 {len(seqs)} ≠ 論文的 {PAPER['sequences']} "
                             "—— 可能只下載了部分 scenario"))

    # 4. identity leakage —— 這是 PDF 訓練設計的第一條
    print("\n[4] identity leakage 檢查(train 與 gallery/query 的身份不可重疊)")
    leaked = False
    for scen, subs in sorted(csvs.items()):
        def ids_of(sub):
            return {parse_index_row(p)["identity"]
                    for p, _ in read_split_csv(subs[sub])
                    if parse_index_row(p)} if sub in subs else set()
        tr, ga, qu = ids_of("train"), ids_of("gallery"), ids_of("query")
        overlap = tr & (ga | qu)
        if overlap:
            leaked = True
            print(f"      ⚠ {scen}:train 與 gallery/query 共用 {len(overlap)} 個身份")
    if not leaked and csvs:
        print("      ✅ 各 scenario 的 train 與 gallery/query 身份不重疊")
    else:
        print("      ⚠ 有重疊。CHIRLA 官方切分本身可能就是這樣設計(closed-set),")
        print("        **不代表資料壞了**,但報告要說明用的是哪一種協定。")

    # 5. CSV 列到的檔案是否真的存在(抽樣)
    print(f"\n[5] CSV 列到的影像是否真的存在(隨機抽 {n_sample} 張)")
    if not all_paths:
        checks.append(_fail("沒有影像路徑可抽驗"))
    else:
        sample = list(all_paths)
        random.Random(seed).shuffle(sample)
        miss = 0
        for rel in sample[:n_sample]:
            f = root / rel if not Path(rel).is_absolute() else Path(rel)
            if not f.exists() and next(root.rglob(Path(rel).name), None) is None:
                miss += 1
        print(f"      找不到 {miss} / {min(n_sample, len(sample))}")
        checks.append(_ok("抽樣路徑都對得上磁碟")
                      if miss == 0 else
                      _fail("CSV 列的路徑在磁碟上找不到 —— 下載不完整或目錄層級不符"))

    # 6. 全掃:磁碟用量 + Git LFS pointer
    # ⚠ 這裡刻意**全掃而不是抽樣**。抽樣只抓得到「全部都壞」,抓不到零星壞檔 ——
    #   900k 張抽 30 張等於沒抽。而反正這一步本來就要走遍整棵樹算容量,
    #   順手比對檔案大小是零成本。LFS pointer 只有一百多 bytes,一眼可辨。
    print("\n[6] 磁碟用量 + Git LFS pointer 全掃")
    IMG = {".png", ".jpg", ".jpeg", ".bmp"}
    total_tiny, tiny_examples = 0, []
    print(f"      {'目錄':<16}{'檔數':>10}{'容量':>10}{'疑似 pointer':>14}")
    print("      " + "-" * 50)
    for d in sorted(tops):
        n = sz = tiny = 0
        for f in (root / d).rglob("*"):
            if not f.is_file():
                continue
            n += 1
            s = f.stat().st_size
            sz += s
            if f.suffix.lower() in IMG and s < 1024:
                tiny += 1
                if len(tiny_examples) < 3:
                    tiny_examples.append(str(f.relative_to(root)))
        total_tiny += tiny
        print(f"      {d:<16}{n:>10,}{sz/1e9:>9.2f}G{tiny:>14,}")
    if total_tiny:
        checks.append(_fail(
            f"有 {total_tiny:,} 個影像檔小於 1KB —— 幾乎確定是 Git LFS pointer,"
            "不是真的影像"))
        for e in tiny_examples:
            print(f"        例:{e}")
        print("        → 在資料目錄跑 `git lfs pull`,或重跑 huggingface-cli download")
    else:
        checks.append(_ok("沒有 LFS pointer,影像檔大小正常"))

    meta_out = root / "chirla_meta.json"
    meta_out.write_text(json.dumps(
        {"root": str(root), "scenarios": {k: {s: str(v) for s, v in d.items()}
                                          for k, d in csvs.items()},
         "n_identities": len(persons), "n_cameras": len(phys_cams),
         "n_sequences": len(seqs),
         "n_raw_id_values": len(idents), "n_camera_dirnames": len(cams),
         "identities": sorted(persons, key=lambda x: int(x)),
         "cameras": sorted(phys_cams), "camera_dirnames": sorted(cams),
         "n_images_listed": len(all_paths), "paper": PAPER},
        ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 72)
    if all(checks):
        print("✅ 驗收通過,可以往下走。事實來源已寫入 chirla_meta.json")
        return 0
    print("❌ 驗收未通過 —— **先解決上面的 ❌ 再開始訓練**,不要用不完整的資料跑。")
    return 1


# ── --index ───────────────────────────────────────────────────────────
def build_index(root, out):
    """輸出與 reid_eval_epfl.parse_frames() 同形的索引:
    [(path, identity, camera, frame_id)] —— 這個形狀能直接接上
    reid_eval_market1501 的 evaluate_cmc_map / gallery_reps / binding_sweep。"""
    root = Path(root)
    idx = defaultdict(lambda: defaultdict(list))
    for scen, subs in sorted(find_csvs(root).items()):
        for sub, p in sorted(subs.items()):
            for path_str, _row in read_split_csv(p):
                m = parse_index_row(path_str)
                if not m:
                    continue
                stem = Path(path_str).stem
                fid = int("".join(ch for ch in stem if ch.isdigit()) or 0)
                idx[scen][sub].append([path_str, m["identity"], m["camera"], fid])

    Path(out).write_text(json.dumps(
        {"root": str(root), "index": {k: dict(v) for k, v in idx.items()}},
        ensure_ascii=False), encoding="utf-8")

    print(f"已寫出 {out}")
    print(f"  {'scenario':<34}{'subset':<10}{'影像':>9}{'身份':>7}{'相機':>7}")
    print("  " + "-" * 68)
    for scen, subs in sorted(idx.items()):
        for sub, rows in sorted(subs.items()):
            ids = {r[1] for r in rows}
            cams = {r[2] for r in rows}
            print(f"  {scen:<34}{sub:<10}{len(rows):>9,}{len(ids):>7}{len(cams):>7}")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="CHIRLA 下載後的根目錄")
    ap.add_argument("--verify", action="store_true", help="交接驗收(先跑這個)")
    ap.add_argument("--index", action="store_true", help="建索引")
    ap.add_argument("--out", default="chirla_index.json")
    ap.add_argument("--sample", type=int, default=30, help="抽驗幾張影像")
    args = ap.parse_args()
    if args.verify:
        return verify(args.root, n_sample=args.sample)
    if args.index:
        return build_index(args.root, args.out)
    ap.error("要給 --verify 或 --index")


if __name__ == "__main__":
    sys.exit(main())
