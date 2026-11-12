"""畫各鏡頭的區域(垃圾桶區、洗手台、洗手台_NG4),輸出 configs/zones/ 用的 JSON。

產生一個本機 HTML,用瀏覽器打開後以滑鼠點多邊形,匯出 JSON。全程本機,不上傳。
畫面右上方會即時顯示游標的像素座標,也可用來量 violation_rules.yaml 的 side_x。

用法:
  # 1. 產生標註頁(取影片第 300 幀;挑一張區域沒被人擋住的畫面)
  python scripts/draw_zones.py <影片或圖片> --frame 300 --camera output0
  #    → 用瀏覽器打開 outputs/zones/output0_zones.html,畫完按「匯出」,下載 output0.json
  # 2. 檢查:把 JSON 疊回畫面,輸出 PNG
  python scripts/draw_zones.py <影片或圖片> --frame 300 --preview configs/zones/output0.json
"""
import argparse
import base64
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from common.video_io import imwrite_unicode, iter_frames   # noqa: E402
from common.zones import draw_zones, load_zones             # noqa: E402

ZONE_NAMES = ["垃圾桶區", "洗手台", "洗手台_NG4"]

HTML = r"""<!doctype html>
<html lang="zh-Hant"><head><meta charset="utf-8"><title>區域標註 __CAMERA__</title>
<style>
 body{font-family:system-ui,"Microsoft JhengHei",sans-serif;margin:12px;background:#1e1e1e;color:#eee}
 button,select{padding:6px 12px;margin-right:6px;font-size:14px;cursor:pointer}
 #wrap{position:relative;display:inline-block;border:1px solid #555}
 canvas{display:block;cursor:crosshair}
 .hint{color:#aaa;font-size:13px;margin:6px 0}
 #xy{color:#ffd54a;font-family:monospace;margin-left:12px}
 textarea{width:100%;height:140px;margin-top:8px;background:#111;color:#6f6;font-family:monospace;font-size:12px}
</style></head><body>
<h3>區域標註:__CAMERA__</h3>
<div class="hint">左鍵 = 加一個點(沿區域邊界點一圈)。點完按「完成區域」,選名稱。全部畫完按「匯出」。</div>
<div>
 名稱:<select id="name"></select>
 <button onclick="finishZone()">完成區域</button>
 <button onclick="undoPoint()">退回上一點</button>
 <button onclick="delZone()">刪最後一個區域</button>
 <button onclick="exportJson()">匯出 __CAMERA__.json</button>
 <span id="status" style="color:#aaa"></span><span id="xy"></span>
</div>
<div id="wrap"><canvas id="c"></canvas></div>
<textarea id="out" placeholder="匯出後 JSON 內容會顯示在這裡,同時自動下載"></textarea>
<script>
const W=__W__, H=__H__, FRAME=__FRAME__, CAMERA="__CAMERA__", NAMES=__NAMES__;
const COLORS=["#ff5252","#00e5ff","#4caf50","#ff9100","#e040fb","#c6ff00"];
const sel=document.getElementById("name");
NAMES.concat(["(自訂名稱)"]).forEach(n=>{const o=document.createElement("option");o.textContent=n;sel.appendChild(o);});
const cv=document.getElementById("c"), ctx=cv.getContext("2d"), img=new Image();
let scale=1, zones=[], cur=[];
img.onload=()=>{scale=Math.min(1,(Math.min(1280,window.innerWidth-40))/W);
  cv.width=Math.round(W*scale); cv.height=Math.round(H*scale); draw();};
img.src="__SRC__";
function pos(e){const r=cv.getBoundingClientRect();return [Math.round((e.clientX-r.left)/scale),Math.round((e.clientY-r.top)/scale)];}
cv.addEventListener("click",e=>{cur.push(pos(e));draw();});
cv.addEventListener("mousemove",e=>{const p=pos(e);document.getElementById("xy").textContent=`游標 x=${p[0]} y=${p[1]}`;});
function finishZone(){
  if(cur.length<3){alert("至少要 3 個點");return;}
  let n=sel.value; if(n==="(自訂名稱)"){n=prompt("區域名稱:");if(!n)return;}
  zones.push({name:n.trim(),points:cur.slice()}); cur=[]; draw();}
function undoPoint(){cur.pop();draw();}
function delZone(){zones.pop();draw();}
function poly(p,c,close){if(!p.length)return;ctx.strokeStyle=c;ctx.lineWidth=2;ctx.beginPath();
  ctx.moveTo(p[0][0]*scale,p[0][1]*scale);for(let i=1;i<p.length;i++)ctx.lineTo(p[i][0]*scale,p[i][1]*scale);
  if(close){ctx.closePath();ctx.fillStyle=c+"33";ctx.fill();}ctx.stroke();
  p.forEach(q=>{ctx.fillStyle=c;ctx.beginPath();ctx.arc(q[0]*scale,q[1]*scale,3,0,7);ctx.fill();});}
function draw(){ctx.drawImage(img,0,0,cv.width,cv.height);
  zones.forEach((z,i)=>{const c=COLORS[i%COLORS.length];poly(z.points,c,true);
    const x=z.points.reduce((a,p)=>a+p[0],0)/z.points.length*scale, y=z.points.reduce((a,p)=>a+p[1],0)/z.points.length*scale;
    ctx.fillStyle=c;ctx.font="16px sans-serif";ctx.fillText(z.name,x-20,y);});
  poly(cur,"#ffffff",false);
  document.getElementById("status").textContent=`已完成 ${zones.length} 個區域,目前 ${cur.length} 點`;}
function exportJson(){
  if(cur.length>=3&&!confirm("還有未完成的區域,要忽略它直接匯出嗎?"))return;
  const txt=JSON.stringify({source:CAMERA+" 鏡頭的區域",frame:FRAME,width:W,height:H,zones:zones},null,2);
  document.getElementById("out").value=txt;
  const a=document.createElement("a");a.href=URL.createObjectURL(new Blob([txt],{type:"application/json"}));
  a.download=CAMERA+".json";a.click();}
</script></body></html>
"""


def grab(source, frame_idx):
    p = Path(source)
    if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp"):
        return cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_COLOR)
    for fid, _t, fr in iter_frames(str(p)):
        if fid >= frame_idx:
            return fr
    return None


def main():
    ap = argparse.ArgumentParser(description="畫各鏡頭的區域,輸出 zones JSON")
    ap.add_argument("source", help="鏡頭影片或一張畫面")
    ap.add_argument("--frame", type=int, default=0, help="用影片第幾幀(挑區域沒被擋住的畫面)")
    ap.add_argument("--camera", default=None, help="鏡頭名稱,決定匯出檔名;預設用影片檔名")
    ap.add_argument("--out-dir", default="outputs/zones", help="標註頁 / 預覽圖輸出資料夾")
    ap.add_argument("--preview", metavar="ZONES_JSON", help="不產生標註頁,改把這個 JSON 疊到畫面上輸出 PNG")
    args = ap.parse_args()

    frame = grab(args.source, args.frame)
    if frame is None:
        raise SystemExit(f"無法取得畫面:{args.source}")
    cam = args.camera or Path(args.source).stem
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    H, W = frame.shape[:2]

    if args.preview:
        z = load_zones(args.preview)
        if (z.get("width"), z.get("height")) != (W, H):
            print(f"注意:JSON 的解析度 {z.get('width')}x{z.get('height')} 與畫面 {W}x{H} 不同")
        zs = z.get("zones", [])
        # cv2 無法畫中文,區域名稱改用編號標示,對照表印在終端機
        vis = draw_zones(frame, [dict(x, name=str(i + 1)) for i, x in enumerate(zs)])
        out = out_dir / f"{Path(args.preview).stem}_preview.png"
        imwrite_unicode(out, vis)
        print(f"預覽圖:{out}")
        for i, x in enumerate(zs):
            print(f"  {i + 1} = {x['name']}({len(x['points'])} 點)")
        return

    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
    src = "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()
    html = (HTML.replace("__SRC__", src).replace("__W__", str(W)).replace("__H__", str(H))
            .replace("__FRAME__", str(args.frame)).replace("__CAMERA__", cam)
            .replace("__NAMES__", json.dumps(ZONE_NAMES, ensure_ascii=False)))
    out = out_dir / f"{cam}_zones.html"
    out.write_text(html, encoding="utf-8")
    print(f"標註頁:{out}")
    print(f"用瀏覽器打開 → 畫區域 → 匯出 {cam}.json → 放到 configs/zones/,並在 violation_rules.yaml 的 zones 指向它")


if __name__ == "__main__":
    main()
