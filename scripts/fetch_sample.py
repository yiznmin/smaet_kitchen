"""
從 EPFL-Smart-Kitchen 的 192.7 GB zip 中,只用 HTTP Range 取出「單一支」影片 clip。

原理:
  - ZIP 的 central directory 在檔尾,可用 Range 抓下來解析(本檔約 314 KB)。
  - 解析出目標 clip 的 local header offset、壓縮大小、壓縮方式(deflate=8)。
  - 再 Range 取該 clip 的壓縮位元組,zlib 原始 inflate 還原成 .mp4。
  - 全程只傳輸目標那一支(~80 MB),不下載整包 192.7 GB。

授權提醒:EPFL 資料集為 CC BY-NC 4.0(禁商用),本專案僅作 benchmark 素材,使用須引用論文 arXiv 2506.01608。

用法:
  python scripts/fetch_sample.py                      # 取設定檔的 default_sample
  python scripts/fetch_sample.py --inner <zip內路徑>   # 指定某一支
  python scripts/fetch_sample.py --smallest           # 取最小的非深度 clip
  python scripts/fetch_sample.py --list               # 只列出 clip 清單,不下載
"""
import argparse
import os
import struct
import sys
import urllib.request
import zlib
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / "configs" / "benchmark.yaml"


def http_range(url, start, end, retries=5):
    """抓 [start, end](含)位元組。分塊讀滿 + 中斷自動重試(對大範圍讀取穩健)。"""
    expected = end - start + 1
    last_err = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
            with urllib.request.urlopen(req, timeout=300) as r:
                buf = bytearray()
                while len(buf) < expected:
                    chunk = r.read(min(1 << 20, expected - len(buf)))   # 每次最多 1MB
                    if not chunk:
                        break
                    buf.extend(chunk)
            if len(buf) == expected:
                return bytes(buf)
            last_err = f"讀到 {len(buf)}/{expected} bytes"
        except Exception as ex:                      # 連線中斷/逾時 → 重試整段
            last_err = repr(ex)
        print(f"    range 讀取重試 {attempt + 1}/{retries}（{last_err}）")
    raise RuntimeError(f"range 讀取失敗:{url} [{start}-{end}] — {last_err}")


def http_size(url):
    req = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(req, timeout=60) as r:
        return int(r.headers.get("Content-Length", 0))


def read_central_directory(url):
    """回傳 zip 內所有項目:list of dict(name, usize, csize, method, local_offset)。"""
    total = http_size(url)
    tail = http_range(url, max(0, total - 200000), total - 1)

    i = tail.rfind(b"PK\x05\x06")  # End Of Central Directory
    if i == -1:
        raise RuntimeError("找不到 EOCD,zip 結構異常")
    eocd = tail[i:i + 22]
    cd_size = struct.unpack("<I", eocd[12:16])[0]
    cd_off = struct.unpack("<I", eocd[16:20])[0]

    # ZIP64(本檔 >4GB 必為 ZIP64)
    if cd_off == 0xFFFFFFFF or cd_size == 0xFFFFFFFF:
        li = tail.rfind(b"PK\x06\x07")  # ZIP64 EOCD locator
        z64_off = struct.unpack("<Q", tail[li + 8:li + 16])[0]
        z64 = http_range(url, z64_off, z64_off + 56)
        cd_size = struct.unpack("<Q", z64[40:48])[0]
        cd_off = struct.unpack("<Q", z64[48:56])[0]

    cd = http_range(url, cd_off, cd_off + cd_size - 1)
    entries = []
    p = 0
    while p + 46 <= len(cd) and cd[p:p + 4] == b"PK\x01\x02":
        method = struct.unpack("<H", cd[p + 10:p + 12])[0]
        csize = struct.unpack("<I", cd[p + 20:p + 24])[0]
        usize = struct.unpack("<I", cd[p + 24:p + 28])[0]
        nlen = struct.unpack("<H", cd[p + 28:p + 30])[0]
        elen = struct.unpack("<H", cd[p + 30:p + 32])[0]
        clen = struct.unpack("<H", cd[p + 32:p + 34])[0]
        loff = struct.unpack("<I", cd[p + 42:p + 46])[0]
        name = cd[p + 46:p + 46 + nlen].decode("utf-8", "replace")
        extra = cd[p + 46 + nlen:p + 46 + nlen + elen]

        # ZIP64 extra field:當欄位為 0xFFFFFFFF 時真值在這
        q = 0
        while q + 4 <= len(extra):
            hid, dsz = struct.unpack("<HH", extra[q:q + 4])
            if hid == 0x0001:
                d = extra[q + 4:q + 4 + dsz]
                dp = 0
                if usize == 0xFFFFFFFF:
                    usize = struct.unpack("<Q", d[dp:dp + 8])[0]; dp += 8
                if csize == 0xFFFFFFFF:
                    csize = struct.unpack("<Q", d[dp:dp + 8])[0]; dp += 8
                if loff == 0xFFFFFFFF:
                    loff = struct.unpack("<Q", d[dp:dp + 8])[0]; dp += 8
            q += 4 + dsz

        entries.append(dict(name=name, usize=usize, csize=csize,
                            method=method, local_offset=loff))
        p += 46 + nlen + elen + clen
    return total, entries


def extract_one(url, entry, dest_path):
    """Range 取單一 entry 的壓縮資料並還原,寫到 dest_path。"""
    loff = entry["local_offset"]
    # local file header:固定 30 bytes + filename + extra(長度可能與 central 不同)
    lh = http_range(url, loff, loff + 30 - 1)
    if lh[:4] != b"PK\x03\x04":
        raise RuntimeError(f"local header 簽章錯誤 @ {loff}")
    n = struct.unpack("<H", lh[26:28])[0]
    e = struct.unpack("<H", lh[28:30])[0]
    data_start = loff + 30 + n + e
    comp = http_range(url, data_start, data_start + entry["csize"] - 1)

    if entry["method"] == 0:        # stored
        raw = comp
    elif entry["method"] == 8:      # deflate
        raw = zlib.decompress(comp, -15)
    else:
        raise RuntimeError(f"不支援的壓縮方式 method={entry['method']}")

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    dest_path.write_bytes(raw)
    return len(raw)


def main():
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    ds = cfg["dataset"]
    url = ds["zip_url"]

    ap = argparse.ArgumentParser()
    ap.add_argument("--inner", help="zip 內的影片路徑")
    ap.add_argument("--smallest", action="store_true", help="取最小的非深度 clip")
    ap.add_argument("--list", action="store_true", help="只列清單不下載")
    args = ap.parse_args()

    print(f"讀取 zip central directory … ({url})")
    total, entries = read_central_directory(url)
    mp4 = [e for e in entries if e["name"].lower().endswith(".mp4")]
    exo = [e for e in mp4 if "depth" not in e["name"].lower()]
    print(f"zip 總大小 {total/1e9:.1f} GB,項目 {len(entries)},mp4 {len(mp4)},非深度 {len(exo)}")

    if args.list:
        for e in sorted(exo, key=lambda x: x["usize"])[:20]:
            print(f"  {e['usize']/1e6:8.1f} MB  method={e['method']}  {e['name']}")
        return

    if args.inner:
        target = next((e for e in mp4 if e["name"] == args.inner), None)
        if target is None:
            sys.exit(f"找不到 {args.inner}")
    elif args.smallest:
        target = min(exo, key=lambda x: x["usize"])
    else:
        target = next((e for e in mp4 if e["name"] == ds["default_sample"]), None)
        if target is None:
            print("找不到 default_sample,改取最小非深度 clip")
            target = min(exo, key=lambda x: x["usize"])

    dest = ROOT / ds["sample_dir"] / Path(target["name"]).name
    print(f"\n取出:{target['name']}")
    print(f"  大小 {target['usize']/1e6:.1f} MB,method={target['method']} → {dest}")
    n = extract_one(url, target, dest)
    print(f"完成,寫入 {n/1e6:.1f} MB")
    print(f"\n下一步:python scripts/probe_video.py \"{dest}\"")


if __name__ == "__main__":
    main()
