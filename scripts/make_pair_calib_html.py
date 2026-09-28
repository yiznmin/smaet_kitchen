"""產生「兩台鏡頭上下並排、人工點選同一個實體地面點」的本機瀏覽器頁面。

## 為什麼是瀏覽器版

沿用 repo 既有的三支 `make_*_html.py` 模式(`make_zone_html.py`、
`make_bbox_labeler_html.py`、`make_walktest_html.py`):很多環境裝的是
opencv-python-headless,`cv2.imshow` 會直接報錯。畫面以 base64 內嵌成單一 HTML,
**全程本機、不上傳**,而且單檔可以搬到別台機器上點。

## 為什麼上下排列而不是左右

CHIRLA 是 1080 寬。左右並排時兩張各縮到約 540 寬 → 地磚角、門檻這種細節點不準,
而點選精度直接決定 σ。上下排列可以各自維持接近原尺寸,代價是要捲動。

## ⚠ 取幀一律用 `iter_frames(start=N)`,不可用 CAP_PROP_POS_FRAMES

`src/common/video_io.py` 檔頭明令:這些影片的跳轉不可靠,幀號一偏整支影片的標註
就錯位。`scripts/chirla_pair_figure.py:72` 用了 `CAP_PROP_POS_FRAMES`(它只是畫圖
佐證所以無妨),**這裡不照抄**。

## 點選原則(頁面上也會提示)

- 挑**固定不動**的地面特徵:門檻兩端、地磚接縫交點、固定設備的螺栓、排水孔、地面標線
- **不要**用會移動的東西:推車、椅子、垃圾桶
- 要**散開、不共線**(不要三點以上落在同一條地板接縫上)、近處遠處都要有
- 標記必須在**同一個平面**上 —— 單應性假設只有一個平面,檯面/台階上的點不能混用
- 8~12 組;不足 5 組 `fit_pair_calib.py` 會拒絕(leave-one-out 需要 N−1 ≥ 4)

用法:
    python scripts/make_pair_calib_html.py --root "D:/.../CHIRLA" \\
        --seq seq_000 --pair camera_2+camera_5 --frame 0

    # 本機用 EPFL 驗工具本身(⚠ CC-BY-NC,僅內部驗證,產物不提交)
    python scripts/make_pair_calib_html.py --videos data/epfl/output0.mp4 data/epfl/output1.mp4 \\
        --names cam0 cam1 --frame 0
"""
import argparse
import base64
import json
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from common.video_io import iter_frames                 # noqa: E402

OUT_DIR = ROOT / "results" / "m5_reid" / "pair_calib"

# ⚠ 用 .replace("__DATA__", ...) 注入而不是 str.format ——
#   沿用 make_bbox_labeler_html.py:166,這樣 CSS/JS 的大括號不必雙寫。
TEMPLATE = r"""<!doctype html>
<html lang="zh-Hant"><head><meta charset="utf-8">
<title>跨鏡頭地面點對應</title>
<style>
 body{font-family:system-ui,"Microsoft JhengHei",sans-serif;margin:10px;background:#1e1e1e;color:#eee}
 h3{margin:4px 0}
 .hint{color:#aaa;font-size:13px;margin:6px 0;line-height:1.6}
 #bar{position:sticky;top:0;background:#1e1e1e;padding:6px 0;z-index:9;border-bottom:1px solid #444}
 button{padding:6px 12px;margin-right:6px;font-size:14px;cursor:pointer}
 .pane{margin:8px 0;border:1px solid #555;display:inline-block}
 .lbl{font-weight:bold;color:#ffd54a;margin:6px 0 2px}
 canvas{display:block;cursor:crosshair}
 textarea{width:100%;height:120px;margin-top:8px;background:#111;color:#6f6;font-family:monospace;font-size:12px}
 b{color:#ffd54a} .warn{color:#ff8a65}
</style></head><body>
<h3>跨鏡頭地面點對應 —— <span id="ttl"></span></h3>
<div class="hint">
 <b>上圖點一下 → 下圖點對應的那一下</b>,成為一組(編號會同時出現在兩邊)。<br>
 挑<b>固定不動的地面特徵</b>:門檻兩端、地磚接縫交點、設備螺栓、排水孔、地面標線。
 <span class="warn">不要用推車、椅子、垃圾桶這種會移動的東西。</span><br>
 要<b>散開、不共線</b>(不要三點以上落在同一條地板接縫上)、<b>近處遠處都要有</b>,
 而且都必須在<b>同一個地面平面</b>上(檯面、台階上的點不能混用)。<br>
 目標 <b>8~12 組</b>;少於 5 組會被拒絕。按 <b>u</b> 退回上一步。
</div>
<div id="bar">
 <button onclick="undo()">↶ 退回上一步 (u)</button>
 <button onclick="exportJson()">⬇ 匯出 points.json</button>
 <span id="status" style="margin-left:12px;color:#aaa"></span>
</div>
<div class="lbl" id="lblA"></div><div class="pane"><canvas id="cA"></canvas></div>
<div class="lbl" id="lblB"></div><div class="pane"><canvas id="cB"></canvas></div>
<textarea id="out" placeholder="按「匯出」後內容會出現在這裡(也會自動下載)"></textarea>
<script>
const DATA = __DATA__;
const COLORS=["#00e5ff","#4caf50","#ff9100","#e040fb","#40a4ff","#c6ff00","#ff5252",
              "#ffeb3b","#69f0ae","#ff80ab","#b388ff","#ffd180"];
document.getElementById("ttl").textContent =
  DATA.a.name + " + " + DATA.b.name + "  (" + DATA.seq + " frame " + DATA.frame + ")";
document.getElementById("lblA").textContent = "① 上:" + DATA.a.name;
document.getElementById("lblB").textContent = "② 下:" + DATA.b.name;

let pairs=[], pending=null;

function mk(side, spec, canvasId){
  const cv=document.getElementById(canvasId), ctx=cv.getContext("2d");
  const img=new Image();
  const st={cv:cv, ctx:ctx, img:img, scale:1, spec:spec, side:side};
  img.onload=()=>{
    const maxW=Math.min(1280, window.innerWidth-40);
    st.scale=Math.min(1, maxW/spec.w);
    cv.width=Math.round(spec.w*st.scale);
    cv.height=Math.round(spec.h*st.scale);
    redraw();
  };
  cv.addEventListener("click", e=>{
    const r=cv.getBoundingClientRect();
    // ⚠ 不取整 —— 手點精度本來只有一兩像素,取整會白白引入 ±0.5px。
    const x=(e.clientX-r.left)/st.scale, y=(e.clientY-r.top)/st.scale;
    click(side, [x, y]);
  });
  img.src=spec.src;
  return st;
}

function click(side, xy){
  if(side==="a"){
    pending=xy;                       // 重複點上圖 = 改點,不會累積
  }else{
    if(pending===null){ alert("請先點上面那張圖的對應點"); return; }
    pairs.push({a:pending, b:xy}); pending=null;
  }
  redraw();
}
function undo(){ if(pending!==null){pending=null;} else {pairs.pop();} redraw(); }
document.addEventListener("keydown", e=>{ if(e.key==="u") undo(); });

function drawOne(st){
  const {ctx,cv,img,scale,side}=st;
  ctx.clearRect(0,0,cv.width,cv.height);
  ctx.drawImage(img,0,0,cv.width,cv.height);
  pairs.forEach((p,i)=>mark(ctx, p[side], scale, COLORS[i%COLORS.length], String(i+1)));
  if(side==="a" && pending!==null) mark(ctx, pending, scale, "#ffffff", "?");
}
function mark(ctx,xy,scale,color,label){
  const x=xy[0]*scale, y=xy[1]*scale;
  ctx.strokeStyle=color; ctx.lineWidth=1.5;
  ctx.beginPath(); ctx.moveTo(x-9,y); ctx.lineTo(x+9,y);
                   ctx.moveTo(x,y-9); ctx.lineTo(x,y+9); ctx.stroke();
  ctx.beginPath(); ctx.arc(x,y,3,0,7); ctx.stroke();
  ctx.fillStyle=color; ctx.font="bold 14px sans-serif"; ctx.fillText(label,x+11,y-6);
}
function redraw(){
  drawOne(A); drawOne(B);
  const need = pairs.length<5 ? ("  ⚠ 還需要至少 "+(5-pairs.length)+" 組") : "";
  document.getElementById("status").textContent =
    "已配對 " + pairs.length + " 組" +
    (pending!==null ? "(上圖已點,等下圖)" : "") + need;
}

const A=mk("a", DATA.a, "cA"), B=mk("b", DATA.b, "cB");

function exportJson(){
  if(pairs.length<5){ if(!confirm("只有 "+pairs.length+" 組,少於 5 組會被 fit_pair_calib.py 拒絕。仍要匯出?")) return; }
  const data={pair:[DATA.a.name, DATA.b.name], seq:DATA.seq, frame:DATA.frame,
              size:{}, points:[]};
  data.size[DATA.a.name]=[DATA.a.w, DATA.a.h];
  data.size[DATA.b.name]=[DATA.b.w, DATA.b.h];
  pairs.forEach((p,i)=>{
    const row={i:i+1};
    row[DATA.a.name]=[Math.round(p.a[0]*100)/100, Math.round(p.a[1]*100)/100];
    row[DATA.b.name]=[Math.round(p.b[0]*100)/100, Math.round(p.b[1]*100)/100];
    data.points.push(row);
  });
  const txt=JSON.stringify(data,null,2);
  document.getElementById("out").value=txt;
  const blob=new Blob([txt],{type:"application/json"});
  const a=document.createElement("a");
  a.href=URL.createObjectURL(blob); a.download="points.json"; a.click();
}
</script></body></html>
"""


def grab(path, frame):
    """取第 frame 幀。⚠ 走 iter_frames(start=)，不用 CAP_PROP_POS_FRAMES —— 見檔頭。"""
    for _fid, _ts, img in iter_frames(str(path), start=frame):
        return img
    return None


def encode(img, quality):
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise SystemExit("JPEG 編碼失敗")
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None, help="CHIRLA 根目錄(與 --seq/--pair 併用)")
    ap.add_argument("--seq", default=None, help="序列,例如 seq_000")
    ap.add_argument("--pair", default=None, help="camera_2+camera_5")
    ap.add_argument("--videos", nargs=2, default=None,
                    help="直接給兩支影片(本機驗工具用),需搭配 --names")
    ap.add_argument("--names", nargs=2, default=None)
    ap.add_argument("--frame", type=int, default=0)
    # 95:標定要看得清地磚角與門檻,壓縮雜訊會直接變成點選誤差 → σ 變大
    ap.add_argument("--quality", type=int, default=95)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.videos:
        if not args.names:
            ap.error("--videos 要搭配 --names")
        names = list(args.names)
        paths = [Path(p) for p in args.videos]
        seq = "(直接給檔)"
    else:
        if not (args.root and args.seq and args.pair):
            ap.error("要給 --root --seq --pair,或改用 --videos + --names")
        if "+" not in args.pair:
            ap.error("--pair 格式是 camera_2+camera_5")
        names = args.pair.split("+", 1)
        seq = args.seq
        vdir = Path(args.root) / "videos" / args.seq
        paths = []
        for cam in names:
            # 沿用 chirla_pair_figure.py:63 的解析方式
            hit = sorted(vdir.glob(cam + "_*.avi"))
            if not hit:
                raise SystemExit(f"找不到 {vdir / (cam + '_*.avi')}")
            paths.append(hit[0])

    specs = []
    for cam, p in zip(names, paths):
        img = grab(p, args.frame)
        if img is None:
            raise SystemExit(f"取不到第 {args.frame} 幀:{p}")
        h, w = img.shape[:2]
        specs.append(dict(name=cam, w=int(w), h=int(h), src=encode(img, args.quality)))
        print(f"  {cam}  {w}x{h}  ← {p.name}")

    data = dict(a=specs[0], b=specs[1], seq=seq, frame=args.frame)
    out = Path(args.out) if args.out else OUT_DIR / f"{names[0]}+{names[1]}.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(TEMPLATE.replace("__DATA__", json.dumps(data)), encoding="utf-8")

    print(f"\n已產生:{out}")
    print("用瀏覽器打開 → 上圖點一下、下圖點對應的那一下 → 湊 8~12 組 → 匯出 points.json")
    print("然後:python scripts/fit_pair_calib.py <points.json>")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
