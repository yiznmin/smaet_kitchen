"""
產生一個「本機瀏覽器版」zone 標註器 HTML(不需 opencv GUI)。

背景:很多環境裝的是 opencv-python-headless(無視窗),cv2.imshow 會報錯。
改用瀏覽器點選:把鏡頭畫面嵌進 HTML,用滑鼠在瀏覽器點多邊形 → 匯出 zones.json。
全程本機、不上傳(EPFL 畫面只留在本機 HTML)。

用法:
  python scripts/make_zone_html.py data/epfl/Boutput0.mp4            # 用第 0 幀
  python scripts/make_zone_html.py data/epfl/Boutput0.mp4 --frame 100
輸出:results/m2/zone_annotator.html(用瀏覽器打開)
"""
import argparse
import base64
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from common.video_io import iter_frames        # noqa: E402

OUT = ROOT / "results" / "m2" / "zone_annotator.html"

HTML = """<!doctype html>
<html lang="zh-Hant"><head><meta charset="utf-8">
<title>Zone 標註器</title>
<style>
  body{{font-family:system-ui,"Microsoft JhengHei",sans-serif;margin:12px;background:#1e1e1e;color:#eee}}
  #bar{{margin-bottom:8px}}
  button{{padding:6px 12px;margin-right:6px;font-size:14px;cursor:pointer}}
  #wrap{{position:relative;display:inline-block;border:1px solid #555}}
  canvas{{display:block;cursor:crosshair}}
  #hint{{color:#aaa;font-size:13px;margin:6px 0}}
  textarea{{width:100%;height:160px;margin-top:8px;background:#111;color:#6f6;font-family:monospace;font-size:12px}}
  b{{color:#ffd54a}}
</style></head><body>
<h3>Zone 標註器(瀏覽器版)</h3>
<div id="hint">
  <b>左鍵</b>=加點(沿地板邊界點一圈) &nbsp;|&nbsp;
  <b>完成區域</b>=按鈕(會問名稱,如 floor) &nbsp;|&nbsp;
  點完所有區域→<b>匯出 zones.json</b>
</div>
<div id="bar">
  <button onclick="finishZone()">✓ 完成這個區域(命名)</button>
  <button onclick="undoPoint()">↶ 退回上一點</button>
  <button onclick="delZone()">🗑 刪最後一個區域</button>
  <button onclick="exportJson()">⬇ 匯出 zones.json</button>
  <span id="status" style="margin-left:12px;color:#aaa"></span>
</div>
<div id="wrap"><canvas id="c"></canvas></div>
<textarea id="out" placeholder="按「匯出」後,zones.json 內容會出現在這裡(也會自動下載)"></textarea>
<script>
const IMG_W={W}, IMG_H={H}, SRC={SRC}, FRAME={FRAME};
const COLORS=["#00e5ff","#4caf50","#ff9100","#e040fb","#40a4ff","#c6ff00","#ff5252","#ffeb3b"];
const cv=document.getElementById("c"), ctx=cv.getContext("2d");
const img=new Image();
let scale=1, zones=[], current=[];

img.onload=()=>{{
  const maxW=Math.min(1100, window.innerWidth-40);
  scale=Math.min(1, maxW/IMG_W);
  cv.width=Math.round(IMG_W*scale); cv.height=Math.round(IMG_H*scale);
  draw();
}};
img.src=SRC;

cv.addEventListener("click",e=>{{
  const r=cv.getBoundingClientRect();
  const x=Math.round((e.clientX-r.left)/scale), y=Math.round((e.clientY-r.top)/scale);
  current.push([x,y]); draw();
}});

function finishZone(){{
  if(current.length<3){{alert("至少要 3 個點");return;}}
  const name=prompt("這個區域的名稱(如 floor / zoneA_raw / wash / trash):","floor");
  if(!name)return;
  zones.push({{name:name.trim(), points:current.slice()}});
  current=[]; draw();
}}
function undoPoint(){{ current.pop(); draw(); }}
function delZone(){{ zones.pop(); draw(); }}

function draw(){{
  ctx.clearRect(0,0,cv.width,cv.height);
  ctx.drawImage(img,0,0,cv.width,cv.height);
  zones.forEach((z,i)=>{{
    const c=COLORS[i%COLORS.length];
    poly(z.points,c,true);
    const cx=z.points.reduce((a,p)=>a+p[0],0)/z.points.length*scale;
    const cy=z.points.reduce((a,p)=>a+p[1],0)/z.points.length*scale;
    ctx.fillStyle=c; ctx.font="16px sans-serif"; ctx.fillText(z.name,cx-20,cy);
  }});
  poly(current,"#ffffff",false);
  current.forEach(p=>dot(p,"#fff"));
  document.getElementById("status").textContent=
    `zones=${{zones.length}}  目前點數=${{current.length}}`;
}}
function poly(pts,color,close){{
  if(pts.length<1)return;
  ctx.strokeStyle=color; ctx.lineWidth=2; ctx.beginPath();
  ctx.moveTo(pts[0][0]*scale,pts[0][1]*scale);
  for(let i=1;i<pts.length;i++)ctx.lineTo(pts[i][0]*scale,pts[i][1]*scale);
  if(close){{ctx.closePath(); ctx.fillStyle=color+"33"; ctx.fill();}}
  ctx.stroke();
}}
function dot(p,color){{ctx.fillStyle=color;ctx.beginPath();ctx.arc(p[0]*scale,p[1]*scale,4,0,7);ctx.fill();}}

function exportJson(){{
  if(current.length>=3 && !confirm("目前有未完成的區域("+current.length+"點),要忽略它直接匯出嗎?"))return;
  const data={{source:"data/epfl/Boutput0.mp4", frame:FRAME, width:IMG_W, height:IMG_H, zones:zones}};
  const txt=JSON.stringify(data,null,2);
  document.getElementById("out").value=txt;
  const blob=new Blob([txt],{{type:"application/json"}});
  const a=document.createElement("a");
  a.href=URL.createObjectURL(blob); a.download="zones.json"; a.click();
}}
</script></body></html>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source")
    ap.add_argument("--frame", type=int, default=0)
    args = ap.parse_args()

    frame = None
    for fid, ts, fr in iter_frames(args.source):
        if fid >= args.frame:
            frame = fr
            break
    if frame is None:
        sys.exit(f"無法取得畫面:{args.source}")
    H, W = frame.shape[:2]
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 88])
    src = "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    html = HTML.format(W=W, H=H, SRC='"' + src + '"', FRAME=args.frame)
    OUT.write_text(html, encoding="utf-8")
    print(f"已產生:{OUT.relative_to(ROOT).as_posix()}")
    print("用瀏覽器打開它 → 滑鼠點多邊形 → 完成區域(命名 floor)→ 匯出 zones.json")
    print("匯出的 zones.json 會下載到你的『下載』資料夾,再搬到 configs/zones.json 即可")


if __name__ == "__main__":
    main()
