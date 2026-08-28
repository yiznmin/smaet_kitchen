"""產生「走動測試記錄表」的單檔 HTML —— 業主拿手機/平板就能標真值。

M5 驗收需要 chef_id 的真值,但業主不會逐幀標框。這個工具把它降到
「誰進了哪台鏡頭」按一下的粒度,匯出的 CSV 直接餵給 scripts/acceptance_m5.py。

沿用 repo 既有的瀏覽器工具模式(make_zone_html.py / make_bbox_labeler_html.py):
單一 HTML、無需安裝、離線可用。

用法:
    python scripts/make_walktest_html.py --cameras cam1 cam2 cam3 --chefs 張師傅 李師傅
    → 產生 results/m5_track/walktest.html,用瀏覽器開啟
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

HTML = """<!doctype html>
<html lang="zh-TW"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>M5 走動測試記錄</title><style>
*{box-sizing:border-box}
body{font-family:system-ui,-apple-system,"Noto Sans TC",sans-serif;margin:0;
     padding:16px;background:#111;color:#eee}
h1{font-size:18px;margin:0 0 4px}
.hint{color:#999;font-size:13px;line-height:1.6;margin-bottom:16px}
.clock{font-size:40px;font-variant-numeric:tabular-nums;margin:12px 0;color:#6cf}
button{font-size:15px;padding:12px 14px;margin:4px;border-radius:8px;
       border:1px solid #444;background:#222;color:#eee;cursor:pointer}
button:hover{background:#2c2c2c}
button.on{background:#1a5;border-color:#1a5;color:#fff}
button.big{font-size:17px;padding:14px 22px}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:6px}
.chef{border:1px solid #333;border-radius:10px;padding:12px;margin:10px 0;background:#181818}
.name{font-weight:600;margin-bottom:8px}
table{width:100%;border-collapse:collapse;font-size:13px;margin-top:8px}
th,td{border-bottom:1px solid #2a2a2a;padding:6px 4px;text-align:left}
th{color:#888;font-weight:500}
.bar{position:sticky;bottom:0;background:#111;padding:12px 0;border-top:1px solid #333}
</style></head><body>
<h1>M5 走動測試記錄</h1>
<div class="hint">
按「開始」後計時器歸零,同時開始錄影 —— <b>兩邊的時間必須對齊</b>。<br>
每位人員進入某台鏡頭的畫面時按下該鏡頭;離開畫面時再按一次(按鈕變回灰色)。<br>
不必求精準到毫秒,±1 秒的誤差驗收腳本可以容忍。<br>
測完按「匯出 CSV」,連同影片一起交回。
</div>
<div class="clock" id="clk">00:00.0</div>
<button class="big" id="go">開始</button>
<button class="big" id="csv">匯出 CSV</button>
<div id="chefs"></div>
<div class="bar"><b>已記錄 <span id="n">0</span> 個區間</b></div>
<table id="log"><thead><tr><th>人員</th><th>鏡頭</th><th>進入</th><th>離開</th></tr></thead>
<tbody></tbody></table>
<script>
const CAMS = __CAMS__, CHEFS = __CHEFS__;
let t0 = null, rows = [], open = {};
const pad = n => String(n).padStart(2,'0');
const fmt = s => pad(Math.floor(s/60))+':'+pad(Math.floor(s%60))+'.'+Math.floor(s*10%10);
const now = () => t0 === null ? 0 : (Date.now()-t0)/1000;

function build(){
  document.getElementById('chefs').innerHTML = CHEFS.map((c,i)=>
    `<div class="chef"><div class="name">${c}</div><div class="grid">`
    + CAMS.map(m=>`<button data-c="${i}" data-m="${m}">${m}</button>`).join('')
    + `</div></div>`).join('');
  document.querySelectorAll('[data-m]').forEach(b=>b.onclick=()=>toggle(b));
}
function toggle(b){
  if(t0===null){ alert('請先按「開始」'); return; }
  const key = b.dataset.c+'|'+b.dataset.m, t = now();
  if(open[key]!==undefined){
    rows.push({chef:CHEFS[b.dataset.c], camera:b.dataset.m,
               t_start:open[key], t_end:t});
    delete open[key]; b.classList.remove('on');
  } else { open[key]=t; b.classList.add('on'); }
  render();
}
function render(){
  document.getElementById('n').textContent = rows.length;
  document.querySelector('#log tbody').innerHTML = rows.slice().reverse().map(r=>
    `<tr><td>${r.chef}</td><td>${r.camera}</td><td>${fmt(r.t_start)}</td>`
    + `<td>${fmt(r.t_end)}</td></tr>`).join('');
}
document.getElementById('go').onclick = () => {
  t0 = Date.now(); rows = []; open = {};
  document.querySelectorAll('.on').forEach(b=>b.classList.remove('on'));
  render();
  document.getElementById('go').textContent = '重新開始(會清除紀錄)';
};
document.getElementById('csv').onclick = () => {
  const stillOpen = Object.keys(open).length;
  if(stillOpen && !confirm(stillOpen+' 個區間還沒按「離開」,要用目前時間收尾嗎?')) return;
  const t = now();
  Object.entries(open).forEach(([k,s])=>{
    const [ci,m]=k.split('|');
    rows.push({chef:CHEFS[ci], camera:m, t_start:s, t_end:t});
  });
  open={}; render();
  const csv = 'chef,camera,t_start,t_end\\n' + rows.map(r=>
    [r.chef,r.camera,r.t_start.toFixed(1),r.t_end.toFixed(1)].join(',')).join('\\n');
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([csv],{type:'text/csv;charset=utf-8'}));
  a.download = 'walktest.csv'; a.click();
};
setInterval(()=>{ document.getElementById('clk').textContent = fmt(now()); }, 100);
build(); render();
</script></body></html>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cameras", nargs="+", required=True)
    ap.add_argument("--chefs", nargs="+", required=True)
    ap.add_argument("--out", default=str(ROOT / "results" / "m5_track" / "walktest.html"))
    args = ap.parse_args()

    html = (HTML.replace("__CAMS__", json.dumps(args.cameras, ensure_ascii=False))
                .replace("__CHEFS__", json.dumps(args.chefs, ensure_ascii=False)))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    print(f"已產生 {out}")
    print(f"  鏡頭:{', '.join(args.cameras)}")
    print(f"  人員:{', '.join(args.chefs)}")
    print()
    print("流程:")
    print("  1. 瀏覽器開啟這個檔案(手機/平板都可以,離線可用)")
    print("  2. 按「開始」的**同時**開始錄影 —— 兩邊時間必須對齊")
    print("  3. 每人進入某鏡頭畫面時按該鏡頭,離開時再按一次")
    print("  4. 匯出 CSV,連同影片交回")
    print("  5. python scripts/acceptance_m5.py --truth walktest.csv --events <系統輸出>")


if __name__ == "__main__":
    main()
