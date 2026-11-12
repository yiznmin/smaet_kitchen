"""產生瀏覽器版的物件標註頁(訓練 M3 用),標完匯出 COCO 格式的 JSON。

類別固定為 M3 的 11 類,順序與 src/m3/classes.py 相同(category_id 1~11)。全程本機,不上傳。
圖片多時用 --batch 分成多個標註頁(每頁一個匯出檔),可分給多人同時標,最後用 build_dataset.py 合併。

用法:
  python scripts/make_labeler.py --img-dir datasets/kitchen/images --out outputs/labeler/labeler.html --batch 200
  #  → 用瀏覽器打開 labeler_1.html、labeler_2.html…,標完各自匯出 coco_1.json、coco_2.json…
操作:拖拉滑鼠畫框;數字鍵 1~9 切類別;←/→ 換張;右鍵點框刪除。
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
from m3.classes import COCO_CATEGORIES   # noqa: E402

TEMPLATE = r"""<!doctype html>
<html lang="zh-Hant"><head><meta charset="utf-8"><title>M3 標註</title>
<style>
 body{font-family:system-ui,"Microsoft JhengHei",sans-serif;margin:10px;background:#1e1e1e;color:#eee}
 #bar button,#bar select{padding:5px 10px;margin-right:6px;font-size:14px}
 #wrap{position:relative;display:inline-block;border:1px solid #555}
 canvas{display:block;cursor:crosshair}
 .hint{color:#aaa;font-size:13px;margin:4px 0}
 b{color:#ffd54a}
</style></head><body>
<h3>M3 物件標註 — 匯出檔:__EXPORT_NAME__</h3>
<div class="hint">
 <b>拖拉滑鼠</b>=畫框(目前類別) &nbsp;|&nbsp; <b>←/→</b>=上/下一張 &nbsp;|&nbsp;
 <b>數字鍵</b>快速切類別 &nbsp;|&nbsp; <b>右鍵點框</b>=刪那個框 &nbsp;|&nbsp; <b>d</b>=刪這張最後一個框 &nbsp;|&nbsp; 畫面中每個看得到的物件都要標 &nbsp;|&nbsp; 標完按<b>匯出</b>(關掉頁面前一定要匯出)
</div>
<div id="bar">
 <button onclick="prev()">← 上一張</button>
 <button onclick="next()">下一張 →</button>
 類別:<select id="cls"></select>
 <button onclick="delLast()">刪最後一框</button>
 <button onclick="exportCoco()">匯出 __EXPORT_NAME__</button>
 <span id="status" style="margin-left:10px;color:#aaa"></span>
</div>
<div id="wrap"><canvas id="c"></canvas></div>
<script>
const DATA = __DATA__;
const PALETTE=["#25e625","#ff3838","#00b0ff","#ffb300","#e040fb","#00e5cc","#ff6e40","#c6ff00","#ff4081","#b0bec5","#8d6e63","#ffff00"];
function catColor(cid){return PALETTE[(cid-1)%PALETTE.length];}
function catName(cid){const c=DATA.categories.find(x=>x.id==cid);return c?c.name:cid;}
const cv=document.getElementById("c"), ctx=cv.getContext("2d");
let idx=0, scale=1, img=new Image(), drawing=false, sx=0, sy=0, cx=0, cy=0;

// 建下拉選單(數字鍵 1..9 快速切)
const sel=document.getElementById("cls");
DATA.categories.forEach((c,i)=>{const o=document.createElement("option");o.value=c.id;
  o.textContent=(i<9?("["+(i+1)+"] "):"")+c.name;sel.appendChild(o);});
sel.value = DATA.categories[0].id;

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
  (DATA.anns[idx]||[]).forEach(a=>box(a.bbox, catColor(a.cat), catName(a.cat), a.draft));
  if(drawing) box([Math.min(sx,cx),Math.min(sy,cy),Math.abs(cx-sx),Math.abs(cy-sy)],"#fff","");
  const cnt={};(DATA.anns[idx]||[]).forEach(a=>cnt[catName(a.cat)]=(cnt[catName(a.cat)]||0)+1);
  const s=Object.entries(cnt).map(([k,v])=>k+"="+v).join(" ");
  document.getElementById("status").textContent=
    `第 ${idx+1}/${DATA.images.length}  ${DATA.images[idx].file_name}  [${s}]`;
}
function box(b,color,label,dashed){
  ctx.strokeStyle=color; ctx.lineWidth=2; ctx.setLineDash(dashed?[6,4]:[]);
  ctx.strokeRect(b[0]*scale,b[1]*scale,b[2]*scale,b[3]*scale); ctx.setLineDash([]);
  if(label){ctx.fillStyle=color; ctx.font="14px sans-serif"; ctx.fillText(label,b[0]*scale,Math.max(12,b[1]*scale-3));}
}
function pos(e){const r=cv.getBoundingClientRect();return [(e.clientX-r.left)/scale,(e.clientY-r.top)/scale];}
cv.addEventListener("mousedown",e=>{if(e.button!==0)return;[sx,sy]=pos(e);[cx,cy]=[sx,sy];drawing=true;});
// 右鍵:刪除包含游標的最小框(草稿或人畫的都可)
cv.addEventListener("contextmenu",e=>{e.preventDefault();const [x,y]=pos(e);const L=DATA.anns[idx]||[];
  let k=-1,best=Infinity;L.forEach((a,i)=>{const [bx,by,bw,bh]=a.bbox;
    if(x>=bx&&x<=bx+bw&&y>=by&&y<=by+bh&&bw*bh<best){best=bw*bh;k=i;}});
  if(k>=0){L.splice(k,1);draw();}});
cv.addEventListener("mousemove",e=>{if(drawing){[cx,cy]=pos(e);draw();}});
cv.addEventListener("mouseup",e=>{
  if(!drawing)return; drawing=false;
  const x=Math.min(sx,cx),y=Math.min(sy,cy),w=Math.abs(cx-sx),h=Math.abs(cy-sy);
  if(w>4&&h>4){const cat=parseInt(sel.value);
    (DATA.anns[idx]=DATA.anns[idx]||[]).push({cat, bbox:[Math.round(x),Math.round(y),Math.round(w),Math.round(h)]});}
  draw();
});
function prev(){if(idx>0){idx--;load();}}
function next(){if(idx<DATA.images.length-1){idx++;load();}}
function delLast(){if(DATA.anns[idx]&&DATA.anns[idx].length){DATA.anns[idx].pop();draw();}}
document.addEventListener("keydown",e=>{
  if(e.key==="ArrowLeft")prev(); else if(e.key==="ArrowRight")next();
  else if(e.key==="d")delLast();
  else if(/^[1-9]$/.test(e.key)){const i=parseInt(e.key)-1; if(i<DATA.categories.length)sel.value=DATA.categories[i].id;}
});
function exportCoco(){
  const coco={images:[],annotations:[],categories:DATA.categories};
  let aid=1;
  DATA.images.forEach((im,i)=>{
    coco.images.push({id:i+1,file_name:im.file_name,width:im.w,height:im.h});
    (DATA.anns[i]||[]).forEach(a=>{const [x,y,w,h]=a.bbox;
      coco.annotations.push({id:aid++,image_id:i+1,category_id:a.cat,bbox:[x,y,w,h],area:w*h,iscrowd:0});});
  });
  const blob=new Blob([JSON.stringify(coco)],{type:"application/json"});
  const a=document.createElement("a"); a.href=URL.createObjectURL(blob); a.download="__EXPORT_NAME__"; a.click();
  const cnt={};coco.annotations.forEach(a=>{const n=catName(a.category_id);cnt[n]=(cnt[n]||0)+1;});
  alert("已匯出 coco.json:\n"+Object.entries(cnt).map(([k,v])=>k+"="+v).join("\n"));
}
load();
</script></body></html>
"""

def main():
    ap = argparse.ArgumentParser(description="產生 M3 物件標註頁")
    ap.add_argument("--img-dir", required=True, help="要標的 jpg 所在資料夾(extract_frames.py 的輸出)")
    ap.add_argument("--out", default="outputs/labeler/labeler.html")
    ap.add_argument("--batch", type=int, default=0, help="每個標註頁放幾張(0 = 全部一頁;圖多建議 200)")
    args = ap.parse_args()

    imgs = sorted(Path(args.img_dir).glob("*.jpg"))
    if not imgs:
        raise SystemExit(f"{args.img_dir} 裡沒有 jpg")
    images = []
    for p in imgs:
        bgr = cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)
        h, w = bgr.shape[:2]
        _, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 70])
        images.append({"file_name": p.name, "w": w, "h": h,
                       "src": "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()})

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    n = args.batch if args.batch > 0 else len(images)
    chunks = [images[k:k + n] for k in range(0, len(images), n)]
    for k, chunk in enumerate(chunks, start=1):
        multi = len(chunks) > 1
        path = out.with_name(f"{out.stem}_{k}{out.suffix}") if multi else out
        export = f"coco_{k}.json" if multi else "coco.json"
        data = {"images": chunk, "anns": [[] for _ in chunk], "categories": COCO_CATEGORIES}
        path.write_text(TEMPLATE.replace("__DATA__", json.dumps(data, ensure_ascii=False))
                        .replace("__EXPORT_NAME__", export), encoding="utf-8")
        print(f"{path}:{len(chunk)} 張 → 匯出 {export}")
    print(f"類別:{[c['name'] for c in COCO_CATEGORIES]}")


if __name__ == "__main__":
    main()
