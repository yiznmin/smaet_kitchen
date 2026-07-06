"""
產生「本機瀏覽器版 bbox 標註器」HTML(不需 opencv GUI)。

- 載入 data/m3_finetune/images 的所有幀 + person 預標(綠框,來自 _draft.json)。
- 你用滑鼠拖拉方框標 knife(紅框);沒刀的幀跳過即可。
- 匯出 COCO(coco.json)→ 直接可微調。

全程本機、不上傳。之後真實資料標註也用同一個工具。
用法:python scripts/make_bbox_labeler_html.py
輸出:results/m2/bbox_labeler.html(瀏覽器打開)
"""
import base64
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

BASE = ROOT / "data" / "m3_finetune"
IMG_DIR = BASE / "images"
DRAFT = BASE / "_draft.json"
OUT = ROOT / "results" / "m2" / "bbox_labeler.html"

TEMPLATE = r"""<!doctype html>
<html lang="zh-Hant"><head><meta charset="utf-8"><title>bbox 標註器</title>
<style>
 body{font-family:system-ui,"Microsoft JhengHei",sans-serif;margin:10px;background:#1e1e1e;color:#eee}
 #bar button,#bar select{padding:5px 10px;margin-right:6px;font-size:14px}
 #wrap{position:relative;display:inline-block;border:1px solid #555}
 canvas{display:block;cursor:crosshair}
 .hint{color:#aaa;font-size:13px;margin:4px 0}
 b{color:#ffd54a}
</style></head><body>
<h3>bbox 標註器 — knife 標註</h3>
<div class="hint">
 <b>拖拉滑鼠</b>=畫框(目前類別) &nbsp;|&nbsp; <b>←/→</b>=上/下一張 &nbsp;|&nbsp;
 <b>d</b>=刪掉這張最後一個框 &nbsp;|&nbsp; 沒刀的幀直接跳過 &nbsp;|&nbsp; 標完按<b>匯出</b>
</div>
<div id="bar">
 <button onclick="prev()">← 上一張</button>
 <button onclick="next()">下一張 →</button>
 類別:<select id="cls"><option value="2">knife(紅)</option><option value="1">person(綠)</option></select>
 <button onclick="delLast()">刪最後一框</button>
 <button onclick="exportCoco()">⬇ 匯出 coco.json</button>
 <span id="status" style="margin-left:10px;color:#aaa"></span>
</div>
<div id="wrap"><canvas id="c"></canvas></div>
<script>
const DATA = __DATA__;
const CLSCOL = {1:"#25e625", 2:"#ff3838"};
const cv=document.getElementById("c"), ctx=cv.getContext("2d");
let idx=0, scale=1, img=new Image(), drawing=false, sx=0, sy=0, cx=0, cy=0;

function load(){
  img=new Image();
  img.onload=()=>{
    const maxW=Math.min(1100, window.innerWidth-30);
    scale=Math.min(1, maxW/DATA.images[idx].w);
    cv.width=Math.round(DATA.images[idx].w*scale);
    cv.height=Math.round(DATA.images[idx].h*scale);
    draw();
  };
  img.src=DATA.images[idx].src;
}
function draw(){
  ctx.clearRect(0,0,cv.width,cv.height);
  ctx.drawImage(img,0,0,cv.width,cv.height);
  (DATA.anns[idx]||[]).forEach(a=>box(a.bbox, CLSCOL[a.cat], a.cat==1?"person":"knife"));
  if(drawing) box([Math.min(sx,cx),Math.min(sy,cy),Math.abs(cx-sx),Math.abs(cy-sy)],"#fff","");
  const nk=(DATA.anns[idx]||[]).filter(a=>a.cat==2).length;
  document.getElementById("status").textContent=
    `第 ${idx+1}/${DATA.images.length} 張  ${DATA.images[idx].file_name}  knife=${nk}`;
}
function box(b,color,label){
  ctx.strokeStyle=color; ctx.lineWidth=2;
  ctx.strokeRect(b[0]*scale,b[1]*scale,b[2]*scale,b[3]*scale);
  if(label){ctx.fillStyle=color; ctx.font="14px sans-serif"; ctx.fillText(label,b[0]*scale,Math.max(12,b[1]*scale-3));}
}
function pos(e){const r=cv.getBoundingClientRect();return [(e.clientX-r.left)/scale,(e.clientY-r.top)/scale];}
cv.addEventListener("mousedown",e=>{[sx,sy]=pos(e);[cx,cy]=[sx,sy];drawing=true;});
cv.addEventListener("mousemove",e=>{if(drawing){[cx,cy]=pos(e);draw();}});
cv.addEventListener("mouseup",e=>{
  if(!drawing)return; drawing=false;
  const x=Math.min(sx,cx),y=Math.min(sy,cy),w=Math.abs(cx-sx),h=Math.abs(cy-sy);
  if(w>4&&h>4){const cat=parseInt(document.getElementById("cls").value);
    (DATA.anns[idx]=DATA.anns[idx]||[]).push({cat, bbox:[Math.round(x),Math.round(y),Math.round(w),Math.round(h)]});}
  draw();
});
function prev(){if(idx>0){idx--;load();}}
function next(){if(idx<DATA.images.length-1){idx++;load();}}
function delLast(){if(DATA.anns[idx]&&DATA.anns[idx].length){DATA.anns[idx].pop();draw();}}
document.addEventListener("keydown",e=>{
  if(e.key==="ArrowLeft")prev(); else if(e.key==="ArrowRight")next();
  else if(e.key==="d")delLast();
});
function exportCoco(){
  const coco={images:[],annotations:[],categories:DATA.categories};
  let aid=1;
  DATA.images.forEach((im,i)=>{
    coco.images.push({id:i+1,file_name:im.file_name,width:im.w,height:im.h});
    (DATA.anns[i]||[]).forEach(a=>{
      const [x,y,w,h]=a.bbox;
      coco.annotations.push({id:aid++,image_id:i+1,category_id:a.cat,bbox:[x,y,w,h],area:w*h,iscrowd:0});
    });
  });
  const txt=JSON.stringify(coco);
  const blob=new Blob([txt],{type:"application/json"});
  const a=document.createElement("a"); a.href=URL.createObjectURL(blob); a.download="coco.json"; a.click();
  const nk=coco.annotations.filter(a=>a.category_id==2).length;
  alert("已匯出 coco.json：person="+coco.annotations.filter(a=>a.category_id==1).length+", knife="+nk);
}
load();
</script></body></html>
"""


def main():
    draft = json.loads(DRAFT.read_text(encoding="utf-8"))
    person_by_name = {}
    idn = {im["id"]: im["file_name"] for im in draft["images"]}
    for a in draft["annotations"]:
        person_by_name.setdefault(idn[a["image_id"]], []).append(a["bbox"])

    imgs = sorted(IMG_DIR.glob("*.jpg"))
    images, anns = [], []
    for p in imgs:
        bgr = cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)
        h, w = bgr.shape[:2]
        ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 62])
        src = "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()
        images.append({"file_name": p.name, "w": w, "h": h, "src": src})
        anns.append([{"cat": 1, "bbox": [round(v, 1) for v in b]} for b in person_by_name.get(p.name, [])])

    data = {"images": images, "anns": anns,
            "categories": [{"id": 1, "name": "person"}, {"id": 2, "name": "knife"}]}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(TEMPLATE.replace("__DATA__", json.dumps(data)), encoding="utf-8")
    mb = OUT.stat().st_size / 1e6
    print(f"已產生:{OUT.relative_to(ROOT).as_posix()}({mb:.1f} MB)")
    print("用瀏覽器打開 → 有刀的幀拖框標 knife(紅)→ 匯出 coco.json")


if __name__ == "__main__":
    main()
