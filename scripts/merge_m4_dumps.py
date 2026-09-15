"""把「每台鏡頭各跑一個程序」的 `eval_m4_chirla.py --dump-dir` 匯出,合併成一次跑完的格式(2026-09-15)。

## 為什麼

McByte 開遮罩時每次更新要跑 SAM / Cutie,5 台鏡頭串列估計 2.4 小時;M4 每台鏡頭各自追蹤、互不影響,
所以可以每台一個程序同時跑,再合併。

`eval_m4_chirla.py` 一次跑多台時,是依 `--cameras` 的順序逐台把 track 與事件接在後面寫出。
這支照**同樣的順序**把各台的 `tracks.csv`、`track_events.csv` 接起來(表頭只留一次),
結果應與一次跑完**逐位相同** —— 由預先登記 V2 驗證。

用法:
    python scripts/merge_m4_dumps.py --parts <camera_1 的匯出目錄> <camera_2 的> ... \\
        --seqs seq_004 ... --out-dir <合併後目錄>
"""
import argparse
import sys
from pathlib import Path

FILES = ("tracks.csv", "track_events.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", nargs="+", required=True, help="依鏡頭順序排列的各台匯出目錄")
    ap.add_argument("--seqs", nargs="+", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    for seq in args.seqs:
        out = Path(args.out_dir) / seq
        out.mkdir(parents=True, exist_ok=True)
        for name in FILES:
            header, body = None, []
            for part in args.parts:
                p = Path(part) / seq / name
                if not p.exists():
                    raise SystemExit(f"[FAIL] 找不到 {p}")
                lines = p.read_bytes().splitlines(keepends=True)
                if not lines:
                    raise SystemExit(f"[FAIL] {p} 是空檔")
                if header is None:
                    header = lines[0]
                elif lines[0] != header:
                    raise SystemExit(f"[FAIL] {p} 的表頭與第一個部分不同")
                body += lines[1:]
            (out / name).write_bytes(header + b"".join(body))
        print(f"{seq}:合併 {len(args.parts)} 台 → {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
