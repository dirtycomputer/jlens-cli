"""Self-contained interactive HTML over compact CLI readouts; no CDN dependency."""

from __future__ import annotations

import html
import json


def render(data: dict, *, title: str = "Jacobian Lens") -> str:
    payload = (
        json.dumps(data, ensure_ascii=False, allow_nan=False)
        .replace("<", "\\u003c")
        .replace("&", "\\u0026")
    )
    return TEMPLATE.replace("__TITLE__", html.escape(title)).replace("__DATA__", payload)


TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITLE__</title><style>
:root{font-family:system-ui,sans-serif;color:#19364a;background:#f3f7fa}*{box-sizing:border-box}
body{margin:0;padding:28px}main{max-width:1500px;margin:auto}h1{margin:0 0 8px;font-size:28px}
.muted{color:#526878;font-size:14px}.prompt{padding:16px;background:#fff;border:1px solid #d7e1e9;border-radius:10px;white-space:pre-wrap;overflow-wrap:anywhere}
.grid{overflow:auto;background:white;border:1px solid #d7e1e9;border-radius:10px;max-height:65vh}
table{border-collapse:collapse;font-size:13px}th,td{border:1px solid #e5ebef;padding:6px;min-width:88px}
thead th{position:sticky;top:0;background:#edf3f7;z-index:2}tbody th{position:sticky;left:0;background:#edf3f7;z-index:1;text-align:left;min-width:112px}
td button{border:0;border-radius:5px;padding:8px;width:100%;min-width:72px;cursor:pointer;color:#123d4c;white-space:pre-wrap;font:inherit}
button:hover,button:focus-visible{outline:2px solid #138b9c}small{display:block;color:#5c7182;font-size:10px}
.panel{margin-top:20px;padding:18px;background:white;border:1px solid #d7e1e9;border-radius:10px}.panel table{width:100%;margin-top:10px}
@media(max-width:600px){body{padding:14px}h1{font-size:23px}.grid{max-height:55vh}}
</style></head><body><main><h1>__TITLE__</h1><p id="meta" class="muted"></p>
<p class="muted">Each cell reads an activation at a source token position. Select a cell to inspect its vocabulary readout. These probabilities describe the readout, not causal importance.</p>
<pre id="prompt" class="prompt"></pre><div class="grid"><table id="grid"></table></div>
<section id="details" class="panel" aria-live="polite">Select a cell above.</section>
</main><script type="application/json" id="data">__DATA__</script><script>
"use strict";
const data=JSON.parse(document.getElementById("data").textContent);
const sample=data.results[0], grid=document.getElementById("grid"), details=document.getElementById("details");
document.getElementById("meta").textContent=`${data.model} · ${sample.num_tokens} input tokens · ${data.lens_info.n_prompts} fitting prompts`;
document.getElementById("prompt").textContent=sample.prompt;
function el(tag,text){const n=document.createElement(tag);if(text!==undefined)n.textContent=text;return n;}
function prob(p){return (100*p).toFixed(3)+"%";}
function show(row,cell){
 details.replaceChildren(el("h2",`${row.kind} · layer ${row.layer} · position ${cell.position}`));
 details.append(el("p",`Input token: ${JSON.stringify(cell.input_token)} (ID ${cell.input_token_id})`));
 const t=el("table"), header=el("tr");["Rank","Token","Token ID","Probability","Logit"].forEach(s=>header.append(el("th",s)));t.append(header);
 cell.top_tokens.forEach((v,i)=>{const r=el("tr");[i+1,JSON.stringify(v.token),v.token_id,prob(v.probability),v.logit.toFixed(4)].forEach(s=>r.append(el("td",s)));t.append(r);});details.append(t);
 if(cell.tracked_tokens.length){details.append(el("h3","Tracked vocabulary tokens"));cell.tracked_tokens.forEach(v=>details.append(el("p",`${JSON.stringify(v.token)} (#${v.token_id}): rank ${v.rank}, probability ${prob(v.probability)}`)));}
}
const head=el("thead"), h=el("tr");h.append(el("th","Readout / layer"));
sample.positions.forEach(p=>{const th=el("th",sample.tokens[p].token);th.append(el("small",`position ${p}`));h.append(th);});head.append(h);grid.append(head);
const body=el("tbody");sample.readouts.forEach(row=>{const tr=el("tr");tr.append(el("th",`${row.kind} / ${row.layer}`));row.positions.forEach(cell=>{const td=el("td"), top=cell.top_tokens[0], b=el("button",top.token);b.style.backgroundColor=`hsl(188 55% ${96-30*Math.sqrt(top.probability)}%)`;b.title=`${row.kind}, layer ${row.layer}, position ${cell.position}, ${prob(top.probability)}`;b.append(el("small",prob(top.probability)));b.onclick=()=>show(row,cell);td.append(b);tr.append(td);});body.append(tr);});grid.append(body);
</script></body></html>"""
