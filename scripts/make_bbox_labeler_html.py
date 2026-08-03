"""
產生「本機瀏覽器版 bbox 標註器」HTML(不需 opencv GUI)。支援多類別、多視角。

- 載入指定資料夾的所有幀 + person 預標(綠框,來自 draft json)。
- 你用滑鼠拖拉方框標各類物件;選類別後拉框。沒有的跳過。
- 匯出 COCO(coco.json)→ 直接可微調。

全程本機、不上傳。之後真實資料也用同一個工具。
用法:
  python scripts/make_bbox_labeler_html.py                         # 預設 person+knife、data/m3_finetune/images
  python scripts/make_bbox_labeler_html.py \\
     --img_dir data/m3_finetune/mv_images \\
     --draft data/m3_finetune/_mv_draft.json \\
     --out results/m2/bbox_labeler_mv.html \\
     --classes knife,cutting_board,raw_meat,cooked_food,cloth,tongs,container,glove,hand
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
<h3>bbox 標註器 — 匯出檔:__EXPORT_NAME__</h3>
<div class="hint">
 <b>拖拉滑鼠</b>=畫框(目前類別) &nbsp;|&nbsp; <b>←/→</b>=上/下一張 &nbsp;|&nbsp;
 <b>數字鍵</b>快速切類別 &nbsp;|&nbsp; <b>d</b>=刪這張最後一個框 &nbsp;|&nbsp; 沒有的物件跳過 &nbsp;|&nbsp; 標完按<b>匯出</b>
</div>
<div id="bar">
 <button onclick="prev()">← 上一張</button>
 <button onclick="next()">下一張 →</button>
 類別:<select id="cls"></select>
 <button onclick="delLast()">刪最後一框</button>
 <button onclick="exportCoco()">⬇ 匯出 coco.json</button>
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
sel.value = (DATA.categories.find(c=>c.name!=="person")||DATA.categories[0]).id;  // 預設非 person

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
  (DATA.anns[idx]||[]).forEach(a=>box(a.bbox, catColor(a.cat), catName(a.cat)));
  if(drawing) box([Math.min(sx,cx),Math.min(sy,cy),Math.abs(cx-sx),Math.abs(cy-sy)],"#fff","");
  const cnt={};(DATA.anns[idx]||[]).forEach(a=>cnt[catName(a.cat)]=(cnt[catName(a.cat)]||0)+1);
  const s=Object.entries(cnt).map(([k,v])=>k+"="+v).join(" ");
  document.getElementById("status").textContent=
    `第 ${idx+1}/${DATA.images.length}  ${DATA.images[idx].file_name}  [${s}]`;
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--img_dir", default=str(ROOT / "data" / "m3_finetune" / "images"))
    ap.add_argument("--draft", default=str(ROOT / "data" / "m3_finetune" / "_draft.json"))
    ap.add_argument("--out", default=str(ROOT / "results" / "m2" / "bbox_labeler.html"))
    ap.add_argument("--classes", default="person,knife",
                    help="逗號分隔的類別;第 1 個當『已預標的人』(承接 draft 的 person 框)")
    ap.add_argument("--batch", type=int, default=0,
                    help="每個 html 放幾張圖(0=全部放一個檔;圖多建議 200,會分批產生 _1/_2…)")
    args = ap.parse_args()

    IMG_DIR, OUT = Path(args.img_dir), Path(args.out).resolve()
    categories = [{"id": i, "name": name.strip()}
                  for i, name in enumerate(args.classes.split(","), start=1)]

    # person 預標(從 draft 讀,若有)
    person_by_name = {}
    draft_p = Path(args.draft)
    if draft_p.exists():
        draft = json.loads(draft_p.read_text(encoding="utf-8"))
        idn = {im["id"]: im["file_name"] for im in draft["images"]}
        for a in draft["annotations"]:
            if a["category_id"] == 1:
                person_by_name.setdefault(idn[a["image_id"]], []).append(a["bbox"])

    imgs = sorted(IMG_DIR.glob("*.jpg"))
    images, anns = [], []
    for p in imgs:
        bgr = cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)
        h, w = bgr.shape[:2]
        ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 55])
        src = "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()
        images.append({"file_name": p.name, "w": w, "h": h, "src": src})
        anns.append([{"cat": 1, "bbox": [round(v, 1) for v in b]} for b in person_by_name.get(p.name, [])])

    def write_html(path, imgs, anns_sub, export_name):
        data = {"images": imgs, "anns": anns_sub, "categories": categories}
        html = (TEMPLATE.replace("__DATA__", json.dumps(data))
                        .replace("__EXPORT_NAME__", export_name))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(html, encoding="utf-8")
        return path.stat().st_size / 1e6

    if args.batch and args.batch > 0:
        n = args.batch
        nb = (len(images) + n - 1) // n
        print(f"分批:{len(images)} 張 → 每檔 {n} 張,共 {nb} 檔")
        for k in range(nb):
            imgs = images[k * n:(k + 1) * n]
            anns_sub = anns[k * n:(k + 1) * n]
            outk = OUT.with_name(f"{OUT.stem}_{k + 1}{OUT.suffix}")
            mb = write_html(outk, imgs, anns_sub, f"coco_{k + 1}.json")
            print(f"  {outk.name}: {len(imgs)} 張({mb:.1f} MB)→ 匯出 coco_{k + 1}.json")
    else:
        mb = write_html(OUT, images, anns, "coco.json")
        print(f"已產生:{OUT.relative_to(ROOT).as_posix()}({mb:.1f} MB),{len(images)} 張")
    print(f"類別:{[c['name'] for c in categories]}")


if __name__ == "__main__":
    main()
