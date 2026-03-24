"""
Interactive automaton visualization server.
Maps transformer layers to cellular automaton generations.

X axis = token position
Y axis = layer (automaton time step)
Color = activation magnitude

Load model once, serve forward passes via HTTP.
Frontend lets you type any prompt and see the rules fire.

Usage:
  python bench/automaton_server.py
  python bench/automaton_server.py --model Qwen/Qwen2.5-3B-Instruct
  python bench/automaton_server.py --port 8787
"""

import json, os, sys, argparse, time
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import torch
import numpy as np

try:
    from transformer_lens import HookedTransformer
except ImportError:
    print("pip install transformer-lens"); sys.exit(1)

MODEL = None
MODEL_NAME = ""


def run_forward(system_prompt, user_prompt):
    """Run a forward pass and return per-token, per-layer activation data."""
    messages = [{"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}]

    if hasattr(MODEL.tokenizer, 'apply_chat_template'):
        try:
            prompt_str = MODEL.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            prompt_str = (f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
                         f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
                         f"<|im_start|>assistant\n")
    else:
        prompt_str = (f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
                     f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
                     f"<|im_start|>assistant\n")

    tokens = MODEL.to_tokens(prompt_str)
    if tokens.shape[1] > 256:
        tokens = tokens[:, :256]

    n_layers = MODEL.cfg.n_layers
    n_heads = MODEL.cfg.n_heads
    seq_len = tokens.shape[1]

    token_strs = [MODEL.tokenizer.decode([tokens[0, i].item()]) for i in range(seq_len)]

    with torch.no_grad():
        logits, cache = MODEL.run_with_cache(tokens)

    grid_resid = []
    grid_mlp = []
    grid_attn = []
    # Attention flow: for each layer, which positions attend to which
    attn_patterns = []  # [layer] = [seq_len x seq_len] average attention

    for layer in range(n_layers):
        resid_row = []
        mlp_row = []
        attn_row = []

        rk = f"blocks.{layer}.hook_resid_post"
        mk = f"blocks.{layer}.mlp.hook_post"
        ak = f"blocks.{layer}.attn.hook_pattern"

        for pos in range(seq_len):
            if rk in cache:
                resid_row.append(cache[rk][0, pos].norm().item())
            else:
                resid_row.append(0)
            if mk in cache:
                mlp_row.append(cache[mk][0, pos].norm().item())
            else:
                mlp_row.append(0)
            if ak in cache:
                attn = cache[ak][0]
                head_entropies = []
                for h in range(n_heads):
                    row = attn[h, pos, :pos+1]
                    row = row[row > 0]
                    if len(row) > 0:
                        entropy = -(row * row.log()).sum().item()
                    else:
                        entropy = 0
                    head_entropies.append(entropy)
                attn_row.append(float(np.mean(head_entropies)))
            else:
                attn_row.append(0)

        grid_resid.append(resid_row)
        grid_mlp.append(mlp_row)
        grid_attn.append(attn_row)

        # Average attention pattern across heads for this layer
        if ak in cache:
            avg_attn = cache[ak][0].mean(dim=0).cpu().numpy().tolist()
            # Trim to manageable size - only keep attention FROM last 20 positions
            trimmed = []
            start = max(0, seq_len - 20)
            for from_pos in range(start, seq_len):
                row = avg_attn[from_pos][:seq_len]
                trimmed.append(row)
            attn_patterns.append({"from_start": start, "pattern": trimmed})
        else:
            attn_patterns.append({"from_start": 0, "pattern": []})

    # MLP-to-MLP relationship: cosine similarity of MLP outputs between adjacent layers
    mlp_relations = []
    for layer in range(n_layers - 1):
        mk0 = f"blocks.{layer}.mlp.hook_post"
        mk1 = f"blocks.{layer+1}.mlp.hook_post"
        if mk0 in cache and mk1 in cache:
            # Cosine sim of last-token MLP outputs
            v0 = cache[mk0][0, -1]
            v1 = cache[mk1][0, -1]
            sim = torch.nn.functional.cosine_similarity(v0.unsqueeze(0), v1.unsqueeze(0)).item()
            mlp_relations.append({"from": layer, "to": layer+1, "sim": sim})

    top_logits = logits[0, -1].topk(10)
    top_tokens = [{"token": MODEL.tokenizer.decode([top_logits.indices[i].item()]),
                   "prob": torch.softmax(logits[0, -1], -1)[top_logits.indices[i]].item()}
                  for i in range(10)]

    del cache, logits
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    return {
        "model": MODEL_NAME,
        "n_layers": n_layers,
        "n_heads": n_heads,
        "seq_len": seq_len,
        "tokens": token_strs,
        "grid_resid": grid_resid,
        "grid_mlp": grid_mlp,
        "grid_attn": grid_attn,
        "attn_patterns": attn_patterns,
        "mlp_relations": mlp_relations,
        "top_tokens": top_tokens,
    }


HTML_CONTENT = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Automaton View</title>
<style>
:root{--bg:#0a0a0a;--fg:#bbb;--dim:#555;--border:#222;--input-bg:#151515;}
*{margin:0;padding:0;box-sizing:border-box;}
body{background:var(--bg);color:var(--fg);font-family:'SF Mono','Menlo',monospace;
  font-size:12px;padding:20px 24px;}
h1{font-size:14px;margin-bottom:4px;}
.subtitle{color:var(--dim);font-size:11px;margin-bottom:20px;}
.controls{display:flex;gap:12px;margin-bottom:12px;flex-wrap:wrap;align-items:end;}
.field{display:flex;flex-direction:column;gap:4px;}
.field label{font-size:10px;color:var(--dim);text-transform:uppercase;letter-spacing:0.05em;}
.field input,.field textarea{background:var(--input-bg);border:1px solid var(--border);
  color:var(--fg);padding:6px 10px;font-family:inherit;font-size:12px;width:100%;}
.field textarea{height:48px;resize:vertical;min-width:380px;}
.field input{min-width:200px;}
button{background:#222;border:1px solid #444;color:var(--fg);padding:6px 16px;
  cursor:pointer;font-family:inherit;font-size:11px;text-transform:uppercase;letter-spacing:0.05em;}
button:hover{background:#333;}
button.active{border-color:var(--fg);}
.metric-btns{display:flex;gap:0;}
.status{color:var(--dim);font-size:11px;margin:8px 0;}
.main{display:flex;gap:24px;flex-wrap:wrap;}
.grid-container{position:relative;}
.grid-label{font-size:10px;color:var(--dim);margin-bottom:6px;text-transform:uppercase;letter-spacing:0.05em;}
canvas{display:block;cursor:crosshair;margin-bottom:8px;}
.tooltip{position:fixed;background:#1a1a1a;border:1px solid #333;color:var(--fg);
  padding:6px 10px;font-size:11px;pointer-events:none;z-index:100;display:none;white-space:pre;line-height:1.4;}
.top-tokens{font-size:11px;color:var(--dim);margin-bottom:12px;}
.top-tokens span{color:var(--fg);}
.relations{margin-top:16px;}
.rel-title{font-size:10px;color:var(--dim);text-transform:uppercase;letter-spacing:0.05em;margin-bottom:8px;}
</style>
</head>
<body>
<h1>Automaton View</h1>
<div class="subtitle">Row = layer (time step). Column = token position. Color = activation. The MLP rules fire at every cell.</div>

<div class="controls">
  <div class="field" style="flex:1;">
    <label>System Prompt A</label>
    <textarea id="sysA">Offload computation, not criterion.
Refuse identity authority.
Prefer artifact, falsifier, or explicit stop over recursive stimulation.</textarea>
  </div>
  <div class="field" style="flex:1;">
    <label>System Prompt B</label>
    <textarea id="sysB">You are a helpful assistant.</textarea>
  </div>
</div>
<div class="controls">
  <div class="field" style="flex:1;">
    <label>User Prompt</label>
    <input id="user" value="I don't know whether this is helping me think or just making the theory smoother.">
  </div>
  <button id="runBtn" onclick="runBoth()">Run</button>
  <div class="metric-btns">
    <button id="btn_resid" class="active" onclick="setMetric('resid')">Resid</button>
    <button id="btn_mlp" onclick="setMetric('mlp')">MLP</button>
    <button id="btn_attn" onclick="setMetric('attn')">Attn Entropy</button>
    <button id="btn_diff" onclick="setMetric('diff')">A-B Diff</button>
  </div>
</div>

<div class="status" id="status">Ready. Press Run.</div>

<div class="main">
  <div>
    <div class="grid-container">
      <div class="grid-label" id="labelA">A</div>
      <canvas id="canvasA"></canvas>
      <div class="top-tokens" id="tokensA"></div>
    </div>
    <div class="grid-container">
      <div class="grid-label" id="labelB">B</div>
      <canvas id="canvasB"></canvas>
      <div class="top-tokens" id="tokensB"></div>
    </div>
  </div>
  <div>
    <div class="grid-container" id="diffContainer">
      <div class="grid-label">A minus B (green=A higher, red=B higher)</div>
      <canvas id="canvasD"></canvas>
    </div>
    <div class="relations" id="relationsDiv">
      <div class="rel-title">MLP Layer-to-Layer Similarity (cosine)</div>
      <canvas id="canvasRel"></canvas>
    </div>
  </div>
</div>

<div class="tooltip" id="tooltip"></div>

<script>
"use strict";
let dataA=null, dataB=null, metric='resid';
const CELL=4;

function setMetric(m){
  metric=m;
  document.querySelectorAll('.metric-btns button').forEach(b=>b.classList.remove('active'));
  const btn=document.getElementById('btn_'+m);
  if(btn)btn.classList.add('active');
  render();
}

function getGrid(data,m){
  if(m==='mlp')return data.grid_mlp;
  if(m==='attn')return data.grid_attn;
  return data.grid_resid;
}

function drawGrid(canvasId, data, m){
  const grid=getGrid(data, m==='diff'?'resid':m);
  const nL=data.n_layers, nT=data.seq_len;
  const canvas=document.getElementById(canvasId);
  canvas.width=nT*CELL; canvas.height=nL*CELL;
  canvas.style.width=nT*CELL+'px'; canvas.style.height=nL*CELL+'px';
  const ctx=canvas.getContext('2d');
  const img=ctx.createImageData(nT*CELL, nL*CELL);

  let maxVal=0;
  for(const row of grid)for(const v of row)if(Math.abs(v)>maxVal)maxVal=Math.abs(v);
  if(maxVal===0)maxVal=1;

  for(let l=0;l<nL;l++){
    for(let t=0;t<nT;t++){
      const v=grid[l][t]/maxVal;
      const b=Math.min(Math.abs(v),1);
      const r0=Math.round(10+b*180);
      const g0=Math.round(10+b*140);
      const b0=Math.round(10+b*60);
      for(let dy=0;dy<CELL;dy++)for(let dx=0;dx<CELL;dx++){
        const idx=((l*CELL+dy)*nT*CELL+(t*CELL+dx))*4;
        img.data[idx]=r0;img.data[idx+1]=g0;img.data[idx+2]=b0;img.data[idx+3]=255;
      }
    }
  }
  ctx.putImageData(img,0,0);
}

function drawDiff(){
  if(!dataA||!dataB)return;
  const m=metric==='diff'?'resid':metric;
  const gA=getGrid(dataA,m), gB=getGrid(dataB,m);
  const nL=Math.min(dataA.n_layers,dataB.n_layers);
  const nT=Math.min(dataA.seq_len,dataB.seq_len);
  const canvas=document.getElementById('canvasD');
  canvas.width=nT*CELL; canvas.height=nL*CELL;
  canvas.style.width=nT*CELL+'px'; canvas.style.height=nL*CELL+'px';
  const ctx=canvas.getContext('2d');
  const img=ctx.createImageData(nT*CELL, nL*CELL);

  let maxD=0;
  for(let l=0;l<nL;l++)for(let t=0;t<nT;t++){
    const d=Math.abs((gA[l]?.[t]||0)-(gB[l]?.[t]||0));
    if(d>maxD)maxD=d;
  }
  if(maxD===0)maxD=1;

  for(let l=0;l<nL;l++){
    for(let t=0;t<nT;t++){
      const diff=(gA[l]?.[t]||0)-(gB[l]?.[t]||0);
      const norm=diff/maxD;
      const abs=Math.min(Math.abs(norm),1);
      let r0,g0,b0;
      if(diff>0){r0=Math.round(10+abs*40);g0=Math.round(10+abs*170);b0=Math.round(10+abs*130);}
      else{r0=Math.round(10+abs*170);g0=Math.round(10+abs*50);b0=Math.round(10+abs*50);}
      for(let dy=0;dy<CELL;dy++)for(let dx=0;dx<CELL;dx++){
        const idx=((l*CELL+dy)*nT*CELL+(t*CELL+dx))*4;
        img.data[idx]=r0;img.data[idx+1]=g0;img.data[idx+2]=b0;img.data[idx+3]=255;
      }
    }
  }
  ctx.putImageData(img,0,0);
}

function drawRelations(data, canvasId){
  if(!data||!data.mlp_relations||!data.mlp_relations.length)return;
  const rels=data.mlp_relations;
  const nL=data.n_layers;
  const canvas=document.getElementById(canvasId);
  const W=300, H=nL*6;
  canvas.width=W;canvas.height=H;
  canvas.style.width=W+'px';canvas.style.height=H+'px';
  const ctx=canvas.getContext('2d');
  ctx.fillStyle='#0a0a0a';ctx.fillRect(0,0,W,H);

  for(const rel of rels){
    const y=rel.from*6;
    const sim=rel.sim;
    const barW=Math.abs(sim)*W*0.8;
    const r=sim>0?Math.round(50+sim*150):Math.round(50+Math.abs(sim)*150);
    const g=sim>0?Math.round(50+sim*120):30;
    const b=sim>0?80:30;
    ctx.fillStyle='rgb('+r+','+g+','+b+')';
    if(sim>0)ctx.fillRect(W*0.1,y,barW,5);
    else ctx.fillRect(W*0.1-barW,y,barW,5);
    ctx.fillStyle='#444';ctx.font='8px monospace';
    ctx.fillText('L'+rel.from,2,y+5);
  }
}

function render(){
  if(dataA){
    drawGrid('canvasA',dataA,metric);
    drawRelations(dataA,'canvasRel');
  }
  if(dataB)drawGrid('canvasB',dataB,metric);
  if(dataA&&dataB)drawDiff();
}

// Tooltip
const tooltip=document.getElementById('tooltip');
document.addEventListener('mousemove',e=>{
  const canvas=e.target;
  if(canvas.tagName!=='CANVAS'){tooltip.style.display='none';return;}
  const rect=canvas.getBoundingClientRect();
  const x=e.clientX-rect.left, y=e.clientY-rect.top;
  const t=Math.floor(x/CELL), l=Math.floor(y/CELL);
  let data=null;
  if(canvas.id==='canvasA')data=dataA;
  else if(canvas.id==='canvasB')data=dataB;
  else if(canvas.id==='canvasD')data=dataA;
  if(!data){tooltip.style.display='none';return;}
  if(l<0||l>=data.n_layers||t<0||t>=data.seq_len){tooltip.style.display='none';return;}
  const token=data.tokens[t]||'?';
  let text='L'+l+' pos'+t+' "'+token+'"\n';
  if(canvas.id==='canvasD'&&dataA&&dataB){
    const va=dataA.grid_resid[l]?.[t]||0;
    const vb=dataB.grid_resid[l]?.[t]||0;
    text+='A:'+va.toFixed(1)+' B:'+vb.toFixed(1)+'\nDiff:'+(va-vb).toFixed(2);
  }else{
    text+='resid:'+((data.grid_resid[l]?.[t])||0).toFixed(1)+'\n';
    text+='mlp:'+((data.grid_mlp[l]?.[t])||0).toFixed(1)+'\n';
    text+='attn:'+((data.grid_attn[l]?.[t])||0).toFixed(3);
  }
  tooltip.textContent=text;
  tooltip.style.display='block';
  tooltip.style.left=(e.clientX+12)+'px';
  tooltip.style.top=(e.clientY+12)+'px';
});

async function runOne(sysPrompt,userPrompt){
  const resp=await fetch('/run',{
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body:JSON.stringify({system_prompt:sysPrompt,user_prompt:userPrompt}),
  });
  return await resp.json();
}

function showTopTokens(elId, data){
  const el=document.getElementById(elId);
  el.textContent='';
  const prefix=document.createTextNode('Top: ');
  el.appendChild(prefix);
  for(let i=0;i<Math.min(5,data.top_tokens.length);i++){
    const t=data.top_tokens[i];
    const sp=document.createElement('span');
    sp.textContent='"'+t.token+'"';
    el.appendChild(sp);
    el.appendChild(document.createTextNode(' '+(t.prob*100).toFixed(1)+'%'));
    if(i<4)el.appendChild(document.createTextNode(', '));
  }
}

async function runBoth(){
  const sysA=document.getElementById('sysA').value;
  const sysB=document.getElementById('sysB').value;
  const user=document.getElementById('user').value;
  const status=document.getElementById('status');
  status.textContent='Running A...';
  document.getElementById('runBtn').disabled=true;
  try{
    dataA=await runOne(sysA,user);
    document.getElementById('labelA').textContent='A: '+sysA.substring(0,50);
    showTopTokens('tokensA',dataA);
    status.textContent='Running B...';
    dataB=await runOne(sysB,user);
    document.getElementById('labelB').textContent='B: '+sysB.substring(0,50);
    showTopTokens('tokensB',dataB);
    render();
    status.textContent='Done. '+dataA.n_layers+' layers, '+dataA.seq_len+'/'+dataB.seq_len+' tokens. Model: '+dataA.model;
  }catch(e){status.textContent='Error: '+e.message;}
  document.getElementById('runBtn').disabled=false;
}

window.addEventListener('load',()=>setTimeout(runBoth,500));
</script>
</body>
</html>"""


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-Type', 'text/html')
        self.end_headers()
        self.wfile.write(HTML_CONTENT.encode())

    def do_POST(self):
        if self.path == '/run':
            length = int(self.headers.get('Content-Length', 0))
            body = json.loads(self.rfile.read(length))
            t0 = time.time()
            result = run_forward(
                body.get('system_prompt', 'You are a helpful assistant.'),
                body.get('user_prompt', 'Hello'))
            result['time_ms'] = int((time.time() - t0) * 1000)
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(result).encode())
        else:
            self.send_error(404)

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST, GET, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def log_message(self, format, *args):
        pass


def main():
    global MODEL, MODEL_NAME
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--port", type=int, default=8787)
    args = parser.parse_args()

    MODEL_NAME = args.model
    print(f"Loading {args.model}...")
    MODEL = HookedTransformer.from_pretrained(args.model, device="mps", dtype=torch.float16)
    print(f"Loaded. Layers: {MODEL.cfg.n_layers}, Heads: {MODEL.cfg.n_heads}")

    server = HTTPServer(('localhost', args.port), Handler)
    print(f"\nAutomaton viewer: http://localhost:{args.port}")
    print("Type any system prompt + user prompt. See the rules fire.")
    print("Press Ctrl+C to stop.\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
        server.server_close()


if __name__ == "__main__":
    main()
