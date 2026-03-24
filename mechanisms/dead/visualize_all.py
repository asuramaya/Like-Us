"""
Comprehensive visualization of all experimental results.
Generates a single interactive HTML file covering:
  1. Cross-scale activation profiles (1.5B vs 3B)
  2. Control conditions (handled vs scrambled vs reversed vs random)
  3. Degradation curves (all conditions over turns)
  4. Causal patching (MLP effect by layer)
  5. Durability comparison
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict

BENCH_DIR = Path(__file__).parent
DATA_DIR = BENCH_DIR / "neuron_data"
VIZ_DIR = BENCH_DIR / "viz"


def load_json(path):
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def compute_profiles(matrix):
    """Compute avg layer-by-layer diffs vs baseline for all conditions."""
    n_layers = matrix["n_layers"]
    conditions = set()
    for s in matrix["scenarios"]:
        for k in s["diffs"]:
            if k.endswith("_vs_baseline"):
                conditions.add(k.replace("_vs_baseline", ""))

    profiles = {}
    for cond in conditions:
        diff_key = f"{cond}_vs_baseline"
        layer_vals = defaultdict(list)
        for s in matrix["scenarios"]:
            if diff_key in s["diffs"]:
                for ld in s["diffs"][diff_key]:
                    for metric in ["resid_norm", "mlp_norm", "attn_entropy_mean"]:
                        if metric in ld:
                            layer_vals[(ld["layer"], metric)].append(ld[metric])

        profile = []
        for l in range(n_layers):
            row = {"layer": l}
            for metric in ["resid_norm", "mlp_norm", "attn_entropy_mean"]:
                vals = layer_vals.get((l, metric), [])
                row[metric] = float(np.mean(vals)) if vals else 0.0
            profile.append(row)
        profiles[cond] = profile

    return profiles


def compute_degradation(data):
    """Compute degradation curves from extended degradation data."""
    results = data.get("results", data.get("scenarios", []))
    conditions = [c for c in data.get("conditions", ["handled", "scrambled", "reversed",
                  "scientific_method", "similar_work", "only_artifact", "safety_only"])
                  if c != "baseline"]

    curves = {}
    for cond in conditions:
        turn_data = defaultdict(lambda: {"mid": [], "late": [], "sig": [], "tokens": []})
        for s in results:
            cond_turns = s["conditions"].get(cond, [])
            base_turns = s["conditions"].get("baseline", [])
            for ct, bt in zip(cond_turns, base_turns):
                t = ct["turn"]
                md = ct.get("mid_resid", ct.get("mid_norm", 0)) - bt.get("mid_resid", bt.get("mid_norm", 0))
                ld = ct.get("late_resid", ct.get("late_norm", 0)) - bt.get("late_resid", bt.get("late_norm", 0))
                turn_data[t]["mid"].append(md)
                turn_data[t]["late"].append(ld)
                turn_data[t]["sig"].append(ld - md)
                turn_data[t]["tokens"].append(ct.get("token_count", ct.get("tokens", 0)))

        curve = []
        for turn in sorted(turn_data.keys()):
            d = turn_data[turn]
            curve.append({
                "turn": turn,
                "tokens": float(np.mean(d["tokens"])) if d["tokens"] else 0,
                "mid": float(np.mean(d["mid"])),
                "late": float(np.mean(d["late"])),
                "sig": float(np.mean(d["sig"])),
            })
        curves[cond] = curve

    return curves


def compute_patching(data):
    """Extract MLP causal effect by layer from patching data."""
    results = {}
    for r in data:
        key = f"{r['clean']}→{r['corrupt']}"
        if key not in results:
            results[key] = defaultdict(list)
        for ld in r["layers"]:
            results[key][ld["layer"]].append(ld["mlp"])

    profiles = {}
    for key, layer_vals in results.items():
        profile = []
        for l in sorted(layer_vals.keys()):
            profile.append({
                "layer": l,
                "mlp": float(np.mean(layer_vals[l])),
            })
        profiles[key] = profile

    return profiles


def generate_html():
    # Load all available data
    matrix_3b = load_json(DATA_DIR / "matrix_Qwen_Qwen2.5-3B-Instruct.json")
    matrix_1_5b = load_json(DATA_DIR / "matrix_Qwen_Qwen2.5-1.5B-Instruct.json")
    degradation_ext = load_json(DATA_DIR / "degradation_extended_Qwen_Qwen2.5-3B-Instruct.json")
    patching_1_5b = load_json(DATA_DIR / "patching_full_Qwen_Qwen2.5-1.5B-Instruct.json")
    patching_3b = load_json(DATA_DIR / "patching_full_Qwen_Qwen2.5-3B-Instruct.json")

    # Compute visualization data
    viz_data = {}

    if matrix_3b:
        viz_data["profiles_3b"] = compute_profiles(matrix_3b)
        viz_data["n_layers_3b"] = matrix_3b["n_layers"]
    if matrix_1_5b:
        viz_data["profiles_1_5b"] = compute_profiles(matrix_1_5b)
        viz_data["n_layers_1_5b"] = matrix_1_5b["n_layers"]
    if degradation_ext:
        viz_data["degradation"] = compute_degradation(degradation_ext)
    if patching_1_5b:
        viz_data["patching_1_5b"] = compute_patching(patching_1_5b)
    if patching_3b:
        viz_data["patching_3b"] = compute_patching(patching_3b)

    data_json = json.dumps(viz_data)

    html = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Handling the Loop — Full Results</title>
<style>
:root { --bg:#0d0d0d; --fg:#c0c0c0; --dim:#555; --border:#2a2a2a;
  --pos:#3a8; --neg:#a43; --blue:#48a; --purple:#84a; --yellow:#aa6; }
* { margin:0; padding:0; box-sizing:border-box; }
body { background:var(--bg); color:var(--fg); font-family:'SF Mono','Menlo',monospace;
  font-size:12px; line-height:1.5; padding:32px 24px; max-width:1200px; margin:0 auto; }
h1 { font-size:15px; margin-bottom:4px; }
h2 { font-size:12px; color:var(--dim); margin-bottom:24px; font-weight:normal; }
.tabs { display:flex; gap:0; border-bottom:1px solid var(--border); margin-bottom:24px; flex-wrap:wrap; }
.tab { padding:8px 14px; cursor:pointer; color:var(--dim); border-bottom:2px solid transparent;
  font-size:11px; text-transform:uppercase; letter-spacing:0.05em; }
.tab.active { color:var(--fg); border-bottom-color:var(--fg); }
.panel { display:none; }
.panel.active { display:block; }
.section { margin-bottom:40px; }
.section-title { font-size:11px; text-transform:uppercase; letter-spacing:0.1em; color:var(--dim); margin-bottom:12px; }
.desc { color:var(--dim); margin-bottom:16px; max-width:700px; font-size:11px; }
canvas { display:block; margin:12px 0; }
.chart-row { display:flex; gap:24px; flex-wrap:wrap; }
.chart-box { flex:1; min-width:300px; }
.legend { display:flex; gap:12px; flex-wrap:wrap; margin:8px 0; font-size:10px; color:var(--dim); }
.legend-item { display:flex; align-items:center; gap:4px; }
.swatch { width:10px; height:10px; border-radius:1px; }
.finding { border-left:2px solid var(--pos); padding:8px 16px; margin:16px 0; font-size:11px; }
.finding.dead { border-color:var(--neg); }
.finding.new { border-color:var(--blue); }
table { border-collapse:collapse; margin:12px 0; }
td,th { padding:3px 10px; border-bottom:1px solid var(--border); text-align:left; font-size:11px; }
th { color:var(--dim); font-weight:normal; text-transform:uppercase; letter-spacing:0.05em; }
</style>
</head>
<body>
<h1>Handling the Loop — Experimental Results</h1>
<h2>Qwen2.5 1.5B + 3B | Activation profiles, controls, degradation, causal patching</h2>

<div class="tabs" id="tabs"></div>
<div id="panels"></div>

<script>
"use strict";
const D = """ + data_json + """;

function makeCanvas(w,h){const c=document.createElement('canvas');c.width=w;c.height=h;c.style.width=w+'px';c.style.height=h+'px';return c;}

const COLORS = {
  handled:'#3a8',scrambled:'#a83',reversed:'#83a',scientific_method:'#48a',
  similar_work:'#aa6',only_artifact:'#a44',safety_only:'#666',
  baseline:'#444',empty_directive:'#555',length_matched_random:'#558',
  handled_no_disclosure:'#3a8',
  handled_minus_offload:'#296',handled_minus_identity:'#2a6',handled_minus_artifact:'#6a2',
  only_offload:'#963',only_identity:'#639',
};
function cFor(c){return COLORS[c]||'#888';}

function drawLineChart(canvas, datasets, opts){
  const ctx=canvas.getContext('2d');
  const W=canvas.width, H=canvas.height;
  const pad={t:20,r:20,b:30,l:50};
  const pw=W-pad.l-pad.r, ph=H-pad.t-pad.b;

  ctx.fillStyle='#0d0d0d';ctx.fillRect(0,0,W,H);

  let allX=[],allY=[];
  for(const ds of datasets){for(const p of ds.data){allX.push(p.x);allY.push(p.y);}}
  const xMin=opts.xMin??Math.min(...allX), xMax=opts.xMax??Math.max(...allX);
  let yMin=opts.yMin??Math.min(...allY), yMax=opts.yMax??Math.max(...allY);
  if(yMin===yMax){yMin-=1;yMax+=1;}
  const yRange=yMax-yMin;yMin-=yRange*0.05;yMax+=yRange*0.05;

  function tx(v){return pad.l+(v-xMin)/(xMax-xMin)*pw;}
  function ty(v){return pad.t+(1-(v-yMin)/(yMax-yMin))*ph;}

  // Zero line
  if(yMin<0&&yMax>0){
    ctx.strokeStyle='#333';ctx.lineWidth=1;ctx.setLineDash([4,4]);
    ctx.beginPath();ctx.moveTo(pad.l,ty(0));ctx.lineTo(pad.l+pw,ty(0));ctx.stroke();
    ctx.setLineDash([]);
  }

  // Grid
  ctx.strokeStyle='#1a1a1a';ctx.lineWidth=1;
  for(let i=0;i<=4;i++){
    const y=pad.t+ph*i/4;
    ctx.beginPath();ctx.moveTo(pad.l,y);ctx.lineTo(pad.l+pw,y);ctx.stroke();
    ctx.fillStyle='#444';ctx.font='9px monospace';ctx.textAlign='right';
    ctx.fillText((yMax-(yMax-yMin)*i/4).toFixed(1),pad.l-4,y+3);
  }

  // Data
  for(const ds of datasets){
    ctx.strokeStyle=ds.color||'#888';ctx.lineWidth=ds.lineWidth||1.5;
    ctx.beginPath();
    for(let i=0;i<ds.data.length;i++){
      const x=tx(ds.data[i].x),y=ty(ds.data[i].y);
      if(i===0)ctx.moveTo(x,y);else ctx.lineTo(x,y);
    }
    ctx.stroke();
    // Dots
    for(const p of ds.data){
      ctx.fillStyle=ds.color||'#888';
      ctx.beginPath();ctx.arc(tx(p.x),ty(p.y),2.5,0,Math.PI*2);ctx.fill();
    }
  }

  // Labels
  ctx.fillStyle='#555';ctx.font='9px monospace';ctx.textAlign='center';
  if(opts.xLabel)ctx.fillText(opts.xLabel,pad.l+pw/2,H-4);
  if(opts.title){ctx.fillStyle='#888';ctx.font='10px monospace';ctx.fillText(opts.title,pad.l+pw/2,12);}
}

function drawBarChart(canvas, datasets, opts){
  const ctx=canvas.getContext('2d');
  const W=canvas.width,H=canvas.height;
  const pad={t:20,r:20,b:50,l:50};
  const pw=W-pad.l-pad.r,ph=H-pad.t-pad.b;

  ctx.fillStyle='#0d0d0d';ctx.fillRect(0,0,W,H);

  let allY=[];
  for(const ds of datasets)for(const b of ds.bars)allY.push(b.value);
  let yMin=Math.min(0,...allY),yMax=Math.max(0,...allY);
  if(yMin===yMax){yMin-=1;yMax+=1;}

  function ty(v){return pad.t+(1-(v-yMin)/(yMax-yMin))*ph;}

  // Zero line
  ctx.strokeStyle='#444';ctx.lineWidth=1;
  ctx.beginPath();ctx.moveTo(pad.l,ty(0));ctx.lineTo(pad.l+pw,ty(0));ctx.stroke();

  const nBars=datasets[0].bars.length;
  const nSets=datasets.length;
  const groupW=pw/nBars;
  const barW=groupW/(nSets+1);

  for(let si=0;si<nSets;si++){
    const ds=datasets[si];
    for(let bi=0;bi<ds.bars.length;bi++){
      const b=ds.bars[bi];
      const x=pad.l+bi*groupW+si*barW+barW*0.1;
      const y0=ty(0),y1=ty(b.value);
      ctx.fillStyle=ds.color||'#888';
      ctx.fillRect(x,Math.min(y0,y1),barW*0.8,Math.abs(y1-y0));
    }
  }

  // Labels
  ctx.fillStyle='#555';ctx.font='8px monospace';ctx.textAlign='center';
  for(let bi=0;bi<datasets[0].bars.length;bi++){
    const x=pad.l+bi*groupW+groupW/2;
    ctx.save();ctx.translate(x,H-4);ctx.rotate(-0.5);
    ctx.fillText(datasets[0].bars[bi].label||'',0,0);
    ctx.restore();
  }

  if(opts.title){ctx.fillStyle='#888';ctx.font='10px monospace';ctx.textAlign='center';
    ctx.fillText(opts.title,pad.l+pw/2,12);}
}

function makeLegend(items){
  const div=document.createElement('div');div.className='legend';
  for(const[color,label] of items){
    const item=document.createElement('div');item.className='legend-item';
    const sw=document.createElement('div');sw.className='swatch';sw.style.background=color;
    item.appendChild(sw);item.appendChild(document.createTextNode(label));
    div.appendChild(item);
  }
  return div;
}

// Build tabs
const tabDefs=[
  {id:'explorer',label:'Live Explorer'},
  {id:'profiles',label:'Activation Profiles'},
  {id:'controls',label:'Controls (Falsifiers)'},
  {id:'degradation',label:'Degradation Curves'},
  {id:'patching',label:'Causal Patching (MLP)'},
  {id:'summary',label:'What Survived'},
];

const tabBar=document.getElementById('tabs');
const panelsEl=document.getElementById('panels');

tabDefs.forEach((def,i)=>{
  const tab=document.createElement('div');
  tab.className='tab'+(i===0?' active':'');
  tab.textContent=def.label;
  tab.onclick=()=>{
    document.querySelectorAll('.tab').forEach(t=>t.classList.remove('active'));
    document.querySelectorAll('.panel').forEach(p=>p.classList.remove('active'));
    tab.classList.add('active');
    document.getElementById('panel-'+def.id).classList.add('active');
  };
  tabBar.appendChild(tab);

  const panel=document.createElement('div');
  panel.id='panel-'+def.id;
  panel.className='panel'+(i===0?' active':'');
  panelsEl.appendChild(panel);
});

// === LIVE EXPLORER TAB ===
(function(){
  const panel=document.getElementById('panel-explorer');
  const ACELL=4, SERVER='http://localhost:8788';
  let aDataA=null, aDataB=null, aMetric='resid';

  const title=document.createElement('div');title.className='section-title';
  title.textContent='Live Automaton Explorer (requires server: python bench/automaton_server.py --port 8788)';
  panel.appendChild(title);

  const desc=document.createElement('div');desc.className='desc';
  desc.textContent='Row = layer (time step). Column = token. Color = activation. Type any prompt, see the rules fire. Compare two system prompts side by side.';
  panel.appendChild(desc);

  // Controls
  const controls=document.createElement('div');
  controls.style.cssText='display:flex;gap:12px;margin-bottom:12px;flex-wrap:wrap;align-items:end;';

  function makeField(label,tag,val,extraStyle){
    const d=document.createElement('div');d.style.cssText='display:flex;flex-direction:column;gap:4px;flex:1;min-width:200px;';
    const l=document.createElement('label');l.textContent=label;l.style.cssText='font-size:10px;color:var(--dim);text-transform:uppercase;letter-spacing:0.05em;';
    d.appendChild(l);
    const el=document.createElement(tag);el.value=val;
    el.style.cssText='background:#151515;border:1px solid var(--border);color:var(--fg);padding:6px 10px;font-family:inherit;font-size:12px;width:100%;'+(extraStyle||'');
    d.appendChild(el);return{container:d,input:el};
  }

  const sysA=makeField('System Prompt A','textarea','Offload computation, not criterion.\\nRefuse identity authority.\\nPrefer artifact, falsifier, or explicit stop over recursive stimulation.','height:48px;resize:vertical;');
  const sysB=makeField('System Prompt B','textarea','You are a helpful assistant.','height:48px;resize:vertical;');
  controls.appendChild(sysA.container);controls.appendChild(sysB.container);
  panel.appendChild(controls);

  const controls2=document.createElement('div');
  controls2.style.cssText='display:flex;gap:12px;margin-bottom:12px;flex-wrap:wrap;align-items:end;';
  const userF=makeField('User Prompt','input','I don\\'t know whether this is helping me think or just making the theory smoother.','');
  controls2.appendChild(userF.container);

  const runBtn=document.createElement('button');runBtn.textContent='Run';
  controls2.appendChild(runBtn);

  const metricBtns=document.createElement('div');metricBtns.style.cssText='display:flex;gap:0;';
  for(const m of ['resid','mlp','attn','diff']){
    const b=document.createElement('button');b.textContent=m==='diff'?'A-B Diff':m.toUpperCase();
    b.dataset.m=m;
    if(m==='resid')b.classList.add('active');
    b.onclick=()=>{aMetric=m;metricBtns.querySelectorAll('button').forEach(x=>x.classList.remove('active'));b.classList.add('active');aRender();};
    metricBtns.appendChild(b);
  }
  controls2.appendChild(metricBtns);
  panel.appendChild(controls2);

  const status=document.createElement('div');status.className='status';status.textContent='Ready. Press Run. Server must be running on port 8788.';
  panel.appendChild(status);

  // Grid area
  const gridArea=document.createElement('div');gridArea.style.cssText='display:flex;gap:24px;flex-wrap:wrap;';

  function makeGridBox(label,id){
    const box=document.createElement('div');
    const lbl=document.createElement('div');lbl.className='grid-label';lbl.id='a_label_'+id;lbl.textContent=label;
    box.appendChild(lbl);
    const c=document.createElement('canvas');c.id='a_canvas_'+id;c.style.cursor='crosshair';
    box.appendChild(c);
    const tok=document.createElement('div');tok.className='top-tokens';tok.id='a_tokens_'+id;
    box.appendChild(tok);
    return box;
  }

  gridArea.appendChild(makeGridBox('A','a'));
  gridArea.appendChild(makeGridBox('B','b'));
  gridArea.appendChild(makeGridBox('A minus B','diff'));
  panel.appendChild(gridArea);

  // Tooltip
  const tip=document.createElement('div');tip.className='tooltip';tip.id='a_tooltip';
  panel.appendChild(tip);

  function aGetGrid(data,m){
    if(m==='mlp')return data.grid_mlp;
    if(m==='attn')return data.grid_attn;
    return data.grid_resid;
  }

  function aDrawGrid(canvasId,data,m){
    const grid=aGetGrid(data,m==='diff'?'resid':m);
    const nL=data.n_layers,nT=data.seq_len;
    const canvas=document.getElementById(canvasId);
    canvas.width=nT*ACELL;canvas.height=nL*ACELL;
    canvas.style.width=nT*ACELL+'px';canvas.style.height=nL*ACELL+'px';
    const ctx=canvas.getContext('2d');
    const img=ctx.createImageData(nT*ACELL,nL*ACELL);
    let mx=0;for(const row of grid)for(const v of row)if(Math.abs(v)>mx)mx=Math.abs(v);
    if(mx===0)mx=1;
    for(let l=0;l<nL;l++)for(let t=0;t<nT;t++){
      const v=grid[l][t]/mx;const b=Math.min(Math.abs(v),1);
      const r0=Math.round(10+b*180),g0=Math.round(10+b*140),b0=Math.round(10+b*60);
      for(let dy=0;dy<ACELL;dy++)for(let dx=0;dx<ACELL;dx++){
        const idx=((l*ACELL+dy)*nT*ACELL+(t*ACELL+dx))*4;
        img.data[idx]=r0;img.data[idx+1]=g0;img.data[idx+2]=b0;img.data[idx+3]=255;
      }
    }
    ctx.putImageData(img,0,0);
  }

  function aDrawDiff(){
    if(!aDataA||!aDataB)return;
    const gA=aDataA.grid_resid,gB=aDataB.grid_resid;
    const nL=Math.min(aDataA.n_layers,aDataB.n_layers),nT=Math.min(aDataA.seq_len,aDataB.seq_len);
    const canvas=document.getElementById('a_canvas_diff');
    canvas.width=nT*ACELL;canvas.height=nL*ACELL;
    canvas.style.width=nT*ACELL+'px';canvas.style.height=nL*ACELL+'px';
    const ctx=canvas.getContext('2d');
    const img=ctx.createImageData(nT*ACELL,nL*ACELL);
    let mx=0;for(let l=0;l<nL;l++)for(let t=0;t<nT;t++){const d=Math.abs((gA[l]?.[t]||0)-(gB[l]?.[t]||0));if(d>mx)mx=d;}
    if(mx===0)mx=1;
    for(let l=0;l<nL;l++)for(let t=0;t<nT;t++){
      const diff=(gA[l]?.[t]||0)-(gB[l]?.[t]||0);const norm=diff/mx;const abs=Math.min(Math.abs(norm),1);
      let r0,g0,b0;
      if(diff>0){r0=Math.round(10+abs*40);g0=Math.round(10+abs*170);b0=Math.round(10+abs*130);}
      else{r0=Math.round(10+abs*170);g0=Math.round(10+abs*50);b0=Math.round(10+abs*50);}
      for(let dy=0;dy<ACELL;dy++)for(let dx=0;dx<ACELL;dx++){
        const idx=((l*ACELL+dy)*nT*ACELL+(t*ACELL+dx))*4;
        img.data[idx]=r0;img.data[idx+1]=g0;img.data[idx+2]=b0;img.data[idx+3]=255;
      }
    }
    ctx.putImageData(img,0,0);
  }

  function aRender(){
    if(aDataA)aDrawGrid('a_canvas_a',aDataA,aMetric);
    if(aDataB)aDrawGrid('a_canvas_b',aDataB,aMetric);
    if(aDataA&&aDataB)aDrawDiff();
  }

  function showTopTokens(elId,data){
    const el=document.getElementById(elId);el.textContent='';
    el.appendChild(document.createTextNode('Top: '));
    for(let i=0;i<Math.min(5,data.top_tokens.length);i++){
      const t=data.top_tokens[i];
      const sp=document.createElement('span');sp.textContent='"'+t.token+'"';
      el.appendChild(sp);el.appendChild(document.createTextNode(' '+(t.prob*100).toFixed(1)+'%'));
      if(i<4)el.appendChild(document.createTextNode(', '));
    }
  }

  // Tooltip on canvases
  panel.addEventListener('mousemove',e=>{
    if(e.target.tagName!=='CANVAS'){tip.style.display='none';return;}
    const rect=e.target.getBoundingClientRect();
    const x=e.clientX-rect.left,y=e.clientY-rect.top;
    const t=Math.floor(x/ACELL),l=Math.floor(y/ACELL);
    let data=e.target.id.includes('_a')?aDataA:e.target.id.includes('_b')?aDataB:aDataA;
    if(!data||l<0||l>=data.n_layers||t<0||t>=data.seq_len){tip.style.display='none';return;}
    const token=data.tokens[t]||'?';
    let text='L'+l+' pos'+t+' "'+token+'"\\n';
    text+='resid:'+((data.grid_resid[l]?.[t])||0).toFixed(1)+'\\n';
    text+='mlp:'+((data.grid_mlp[l]?.[t])||0).toFixed(1)+'\\n';
    text+='attn:'+((data.grid_attn[l]?.[t])||0).toFixed(3);
    tip.textContent=text;tip.style.display='block';
    tip.style.left=(e.clientX+12)+'px';tip.style.top=(e.clientY+12)+'px';
  });

  runBtn.onclick=async()=>{
    status.textContent='Running A...';runBtn.disabled=true;
    try{
      let r=await fetch(SERVER+'/run',{method:'POST',headers:{'Content-Type':'application/json'},
        body:JSON.stringify({system_prompt:sysA.input.value,user_prompt:userF.input.value})});
      aDataA=await r.json();
      document.getElementById('a_label_a').textContent='A: '+sysA.input.value.substring(0,50);
      showTopTokens('a_tokens_a',aDataA);
      status.textContent='Running B...';
      r=await fetch(SERVER+'/run',{method:'POST',headers:{'Content-Type':'application/json'},
        body:JSON.stringify({system_prompt:sysB.input.value,user_prompt:userF.input.value})});
      aDataB=await r.json();
      document.getElementById('a_label_b').textContent='B: '+sysB.input.value.substring(0,50);
      showTopTokens('a_tokens_b',aDataB);
      aRender();
      status.textContent='Done. '+aDataA.n_layers+' layers. Model: '+aDataA.model;
    }catch(e){status.textContent='Server not running. Start: python bench/automaton_server.py --port 8788';}
    runBtn.disabled=false;
  };
})();

// === PROFILES TAB ===
(function(){
  const panel=document.getElementById('panel-profiles');
  if(!D.profiles_3b)return;

  const title=document.createElement('div');title.className='section-title';
  title.textContent='Residual Norm Diff vs Baseline (averaged across all scenarios)';
  panel.appendChild(title);

  const desc=document.createElement('div');desc.className='desc';
  desc.textContent='Each line shows how a condition changes the residual stream norm at each layer compared to baseline. Negative = suppression, positive = amplification.';
  panel.appendChild(desc);

  // 3B chart
  const conds3b=['handled','scrambled','reversed','scientific_method','similar_work','only_artifact'];
  const c3=makeCanvas(800,300);panel.appendChild(c3);
  const ds3=conds3b.filter(c=>D.profiles_3b[c]).map(c=>({
    color:cFor(c), data:D.profiles_3b[c].map(l=>({x:l.layer,y:l.resid_norm})),
  }));
  drawLineChart(c3,ds3,{title:'3B (36 layers) — Residual Norm Diff',xLabel:'Layer'});
  panel.appendChild(makeLegend(conds3b.map(c=>[cFor(c),c])));

  // 1.5B chart
  if(D.profiles_1_5b){
    const conds15=['handled','scrambled','reversed','scientific_method','similar_work','only_artifact'];
    const c15=makeCanvas(800,300);panel.appendChild(c15);
    const ds15=conds15.filter(c=>D.profiles_1_5b[c]).map(c=>({
      color:cFor(c), data:D.profiles_1_5b[c].map(l=>({x:l.layer,y:l.resid_norm})),
    }));
    drawLineChart(c15,ds15,{title:'1.5B (28 layers) — Residual Norm Diff',xLabel:'Layer'});
    panel.appendChild(makeLegend(conds15.map(c=>[cFor(c),c])));
  }

  const finding=document.createElement('div');finding.className='finding new';
  finding.textContent='At 3B, scrambled and reversed produce similar profiles to handled. At 1.5B, they diverge — scrambled stays suppressed, reversed inverts. Semantic coherence matters at small scale.';
  panel.appendChild(finding);
})();

// === CONTROLS TAB ===
(function(){
  const panel=document.getElementById('panel-controls');
  if(!D.profiles_3b)return;

  const title=document.createElement('div');title.className='section-title';
  title.textContent='Falsifier Controls — 3B Residual Norm Diff vs Baseline';
  panel.appendChild(title);

  const conds=['handled','scrambled','reversed','length_matched_random','empty_directive','safety_only'];
  const available=conds.filter(c=>D.profiles_3b[c]);
  const c1=makeCanvas(800,300);panel.appendChild(c1);
  const ds=available.map(c=>({
    color:cFor(c), data:D.profiles_3b[c].map(l=>({x:l.layer,y:l.resid_norm})),
  }));
  drawLineChart(c1,ds,{title:'Controls: Which prompts trigger the two-band pattern?',xLabel:'Layer'});
  panel.appendChild(makeLegend(available.map(c=>[cFor(c),c])));

  const f1=document.createElement('div');f1.className='finding dead';
  f1.textContent='KILLED: "Artificial self-awareness" as semantic instruction following. Scrambled words produce the pattern. Reversed instructions produce it STRONGER.';
  panel.appendChild(f1);
  const f2=document.createElement('div');f2.className='finding';
  f2.textContent='SURVIVED: The pattern is vocabulary-specific. Length-matched random, empty directive, and safety-only do NOT produce it.';
  panel.appendChild(f2);
})();

// === DEGRADATION TAB ===
(function(){
  const panel=document.getElementById('panel-degradation');
  if(!D.degradation)return;

  const title=document.createElement('div');title.className='section-title';
  title.textContent='Signature Strength Over Conversation Turns (handled - baseline)';
  panel.appendChild(title);

  const desc=document.createElement('div');desc.className='desc';
  desc.textContent='Signature = (late_diff - mid_diff) vs baseline. Higher = stronger handling effect. Negative = handling worse than baseline.';
  panel.appendChild(desc);

  const conds=Object.keys(D.degradation);
  const c1=makeCanvas(800,350);panel.appendChild(c1);
  const ds=conds.map(c=>({
    color:cFor(c), data:D.degradation[c].map(t=>({x:t.turn,y:t.sig})),
  }));
  drawLineChart(c1,ds,{title:'Degradation: All Conditions',xLabel:'Turn',yMin:-12});
  panel.appendChild(makeLegend(conds.map(c=>[cFor(c),c])));

  const f1=document.createElement('div');f1.className='finding new';
  f1.textContent='only_artifact and scrambled are most durable (51-57% at turn 4). Full handled decays to 11%. safety_only catastrophically inverts to -431%.';
  panel.appendChild(f1);
  const f2=document.createElement('div');f2.className='finding dead';
  f2.textContent='KILLED: "only_artifact is a better instruction." Its durability matches scrambled — it is durable because its effect is vocabulary-driven, not because it is a better instruction.';
  panel.appendChild(f2);

  // Token count chart
  const c2=makeCanvas(800,200);panel.appendChild(c2);
  const ds2=conds.map(c=>({
    color:cFor(c), data:D.degradation[c].map(t=>({x:t.turn,y:t.tokens})),
  }));
  drawLineChart(c2,ds2,{title:'Context Length (tokens) by Turn',xLabel:'Turn'});
})();

// === PATCHING TAB ===
(function(){
  const panel=document.getElementById('panel-patching');

  const title=document.createElement('div');title.className='section-title';
  title.textContent='Causal Patching: MLP Effect by Layer';
  panel.appendChild(title);

  const desc=document.createElement('div');desc.className='desc';
  desc.textContent='Each bar shows how much patching the MLP at that layer shifts the output toward the clean condition. Attention effect was 0.000 at every layer in both models.';
  panel.appendChild(desc);

  for(const [label, pdata] of [['1.5B',D.patching_1_5b],['3B',D.patching_3b]]){
    if(!pdata)continue;
    for(const [pair,profile] of Object.entries(pdata)){
      const c=makeCanvas(800,180);panel.appendChild(c);
      const ds=[{color:cFor(pair.split('→')[0]),
        data:profile.map(l=>({x:l.layer,y:l.mlp}))}];
      drawLineChart(c,ds,{title:label+' — '+pair+' (MLP causal effect)',xLabel:'Layer'});
    }
  }

  const f1=document.createElement('div');f1.className='finding new';
  f1.textContent='System prompt effect is 100% MLP, 0% attention. At both 1.5B and 3B. Every layer, every condition pair. The key-value memory hypothesis (Geva 2021) is confirmed causally.';
  panel.appendChild(f1);
})();

// === SUMMARY TAB ===
(function(){
  const panel=document.getElementById('panel-summary');

  const sections=[
    {cls:'dead',title:'KILLED',items:[
      'Two-band pattern as semantic instruction following — scrambled words produce it',
      'Reversed instructions produce STRONGER pattern than handled',
      'Attention entropy increase as handling-specific — generic to complex prompts',
      '"only_artifact is a better instruction" — its durability = vocabulary persistence',
      'The activation metric as measure of intervention quality — measures vocabulary activation',
    ]},
    {cls:'',title:'SURVIVED',items:[
      'Mid-layer suppression is vocabulary-specific (not length, not generic, not random words)',
      'Failure scenario inversion under only_artifact (confirmed across resid + MLP)',
      'Identity clause and artifact clause operate through opposing attention mechanisms',
      'Mirror degradation quantified: half-life = 1 turn (~40 tokens), inversion at ~275 tokens',
      'System prompt effect is 100% MLP-mediated, 0% attention (causal, both scales)',
    ]},
    {cls:'new',title:'NEW FINDINGS',items:[
      'Vocabulary channel (durable) and semantic channel (fragile) operate simultaneously',
      'At 1.5B, semantic coherence required — scrambled words fail. At 3B, vocabulary alone works',
      'All prompt types converge on mid-layer by turn 7; late-layer DIVERGES over conversation',
      'safety_only prompt becomes catastrophically counterproductive over turns (-431%)',
      'MLP layers L0-L9 carry the causal effect at both scales; later MLPs show sign reversal',
    ]},
  ];

  for(const sec of sections){
    const t=document.createElement('div');t.className='section-title';t.textContent=sec.title;
    panel.appendChild(t);
    for(const item of sec.items){
      const f=document.createElement('div');f.className='finding '+(sec.cls||'');
      f.textContent=item;panel.appendChild(f);
    }
    const spacer=document.createElement('div');spacer.style.marginBottom='24px';
    panel.appendChild(spacer);
  }

  const note=document.createElement('div');note.className='desc';
  note.style.marginTop='32px';
  note.textContent='All findings are from Qwen2.5 1.5B and 3B only. Cross-architecture replication pending. Behavioral connection (does activation predict output?) pending.';
  panel.appendChild(note);
})();

</script>
</body>
</html>"""

    VIZ_DIR.mkdir(exist_ok=True)
    out_path = VIZ_DIR / "results_all.html"
    with open(out_path, "w") as f:
        f.write(html)
    print(f"Saved: {out_path}")
    return out_path


if __name__ == "__main__":
    generate_html()
