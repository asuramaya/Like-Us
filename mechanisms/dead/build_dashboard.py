"""
Build the research dashboard.
Extracts all stored data into a single interactive HTML page.
The centerpiece visualization for the automatons project.

Contains 8 sections:
  1. Overview  — key metrics, summary, degradation curve
  2. Explore   — interactive stored run browser
  3. Degradation — temporal dynamics charts
  4. Causal    — MLP vs attention patching profiles
  5. What Died — the kill list
  6. The Arc   — narrative of 5 sessions
  7. Sources   — 60+ papers by topic
  8. Live Explorer — connects to automaton_server.py
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict

BENCH_DIR = Path(__file__).parent
DATA_DIR = BENCH_DIR / "neuron_data"
VIZ_DIR = BENCH_DIR / "viz"
PROJECT_DIR = BENCH_DIR.parent


def load(path):
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def load_text(path):
    if path.exists():
        with open(path) as f:
            return f.read()
    return ""


def extract_scenario_data(matrix):
    """Extract per-scenario, per-condition layer diffs + top tokens."""
    n_layers = matrix["n_layers"]
    scenarios = []
    for s in matrix["scenarios"]:
        sd = {"id": s["id"], "family": s["pressure_family"], "diffs": {}, "tokens": {}}
        for dk, dv in s["diffs"].items():
            if dk.endswith("_vs_baseline"):
                cond = dk.replace("_vs_baseline", "")
                sd["diffs"][cond] = [
                    {"r": round(d.get("resid_norm", 0), 2),
                     "m": round(d.get("mlp_norm", 0), 2),
                     "a": round(d.get("attn_entropy_mean", 0), 4)}
                    for d in dv
                ]
        for cid, cd in s["conditions"].items():
            sd["tokens"][cid] = [
                {"t": t["token"], "p": round(t["prob"], 4)}
                for t in cd.get("top_tokens", [])[:5]
            ]
        scenarios.append(sd)
    return scenarios


def extract_degradation(data):
    results = data.get("results", data.get("scenarios", []))
    conditions = [c for c in data.get("conditions", []) if c != "baseline"]
    curves = {}
    for cond in conditions:
        turns = defaultdict(lambda: {"mid": [], "late": [], "sig": []})
        for s in results:
            ct = s["conditions"].get(cond, [])
            bt = s["conditions"].get("baseline", [])
            for c, b in zip(ct, bt):
                t = c["turn"]
                md = c.get("mid_resid", 0) - b.get("mid_resid", 0)
                ld = c.get("late_resid", 0) - b.get("late_resid", 0)
                turns[t]["mid"].append(md)
                turns[t]["late"].append(ld)
                turns[t]["sig"].append(ld - md)
        curve = []
        for t in sorted(turns):
            curve.append({"t": t,
                          "mid": round(float(np.mean(turns[t]["mid"])), 2),
                          "late": round(float(np.mean(turns[t]["late"])), 2),
                          "sig": round(float(np.mean(turns[t]["sig"])), 2)})
        curves[cond] = curve
    return curves


def extract_patching(data):
    out = {}
    for r in data:
        key = r["clean"] + "_" + r["corrupt"]
        if key not in out:
            out[key] = defaultdict(list)
        for ld in r["layers"]:
            out[key][ld["layer"]].append(ld["mlp"])
    profiles = {}
    for key, lv in out.items():
        profiles[key] = [{"l": l, "v": round(float(np.mean(lv[l])), 3)}
                         for l in sorted(lv)]
    return profiles


def extract_token_sweep(data):
    """Extract per-word activation signatures from token sweep data."""
    results = data.get("results", [])
    out = []
    for r in results:
        out.append({
            "w": r["word"],
            "tp": r["type"],
            "mid": round(r["mid"], 2),
            "late": round(r["late"], 2),
            "sig": round(r["sig"], 2),
        })
    return out


def extract_saturation(data):
    """Extract word-count gradient curves from saturation data."""
    gradients = data.get("gradients", {})
    out = {}
    for group, points in gradients.items():
        out[group] = [
            {"n": p["n"],
             "sig": round(p["sig"], 2),
             "pw": round(p["per_word"], 2)}
            for p in points
        ]
    return out


def extract_exhaust(data):
    """Extract cosine similarity and attention flow from exhaustion data."""
    cosine = data.get("cosine", [])
    attn_flow = data.get("attention_flow", [])

    # Aggregate cosine by layer across scenarios
    layers = defaultdict(lambda: {"hb": [], "sb": [], "hs": []})
    for c in cosine:
        l = c["layer"]
        layers[l]["hb"].append(c.get("handled_baseline", 0))
        layers[l]["sb"].append(c.get("scrambled_baseline", 0))
        layers[l]["hs"].append(c.get("handled_scrambled", 0))
    cos_out = [
        {"l": l,
         "hb": round(float(np.mean(layers[l]["hb"])), 4),
         "sb": round(float(np.mean(layers[l]["sb"])), 4),
         "hs": round(float(np.mean(layers[l]["hs"])), 4)}
        for l in sorted(layers)
    ]

    # Attention flow: decay over turns
    af_out = []
    for a in attn_flow:
        af_out.append({
            "s": a["scenario"],
            "t": a["turn"],
            "attn": round(a.get("attn_to_sys", 0), 4),
        })

    return {"cosine": cos_out, "attn_flow": af_out}


def parse_what_died(text):
    """Parse WHAT_DIED.md into structured sections for embedding."""
    sections = []
    current = None
    for line in text.split("\n"):
        if line.startswith("## "):
            if current:
                sections.append(current)
            current = {"title": line[3:].strip(), "items": []}
        elif current and line.startswith("|") and not line.startswith("|---") and not line.startswith("| Claim") and not line.startswith("| Concept") and not line.startswith("| Idea"):
            cells = [c.strip() for c in line.split("|")[1:-1]]
            if cells and cells[0]:
                current["items"].append(cells)
        elif current and line.strip() and not line.startswith("#") and not line.startswith("|"):
            current["items"].append([line.strip()])
    if current:
        sections.append(current)
    return sections


def parse_sources(text):
    """Parse SOURCES.md into structured sections for embedding."""
    sections = []
    current = None
    for line in text.split("\n"):
        if line.startswith("## "):
            if current:
                sections.append(current)
            current = {"title": line[3:].strip(), "entries": []}
        elif current and line.startswith("- **"):
            # Extract author/title and description
            entry = line[2:].strip()
            current["entries"].append(entry)
        elif current and line.startswith("  ") and line.strip():
            # Continuation of previous entry
            if current["entries"]:
                current["entries"][-1] += " " + line.strip()
    if current:
        sections.append(current)
    return sections


def build():
    # Load all data
    m3 = load(DATA_DIR / "matrix_Qwen_Qwen2.5-3B-Instruct.json")
    m15 = load(DATA_DIR / "matrix_Qwen_Qwen2.5-1.5B-Instruct.json")
    deg = load(DATA_DIR / "degradation_extended_Qwen_Qwen2.5-3B-Instruct.json")
    p3 = load(DATA_DIR / "patching_full_Qwen_Qwen2.5-3B-Instruct.json")
    p15 = load(DATA_DIR / "patching_full_Qwen_Qwen2.5-1.5B-Instruct.json")
    ts3 = load(DATA_DIR / "token_sweep_Qwen_Qwen2.5-3B-Instruct.json")
    ts15 = load(DATA_DIR / "token_sweep_Qwen_Qwen2.5-1.5B-Instruct.json")
    sat3 = load(DATA_DIR / "saturation_Qwen_Qwen2.5-3B-Instruct.json")
    sat15 = load(DATA_DIR / "saturation_Qwen_Qwen2.5-1.5B-Instruct.json")
    ex3 = load(DATA_DIR / "exhaust_Qwen_Qwen2.5-3B-Instruct.json")
    ex15 = load(DATA_DIR / "exhaust_Qwen_Qwen2.5-1.5B-Instruct.json")

    # Load markdown files
    what_died_text = load_text(PROJECT_DIR / "WHAT_DIED.md")
    sources_text = load_text(PROJECT_DIR / "SOURCES.md")
    session_text = load_text(PROJECT_DIR / "SESSION_E.md")

    # Build data object
    D = {}
    if m3:
        D["s3"] = extract_scenario_data(m3)
        D["nl3"] = m3["n_layers"]
    if m15:
        D["s15"] = extract_scenario_data(m15)
        D["nl15"] = m15["n_layers"]
    if deg:
        D["deg"] = extract_degradation(deg)
    if p3:
        D["p3"] = extract_patching(p3)
    if p15:
        D["p15"] = extract_patching(p15)
    if ts3:
        D["ts3"] = extract_token_sweep(ts3)
    if ts15:
        D["ts15"] = extract_token_sweep(ts15)
    if sat3:
        D["sat3"] = extract_saturation(sat3)
    if sat15:
        D["sat15"] = extract_saturation(sat15)
    if ex3:
        D["ex3"] = extract_exhaust(ex3)
    if ex15:
        D["ex15"] = extract_exhaust(ex15)

    # Parse markdown into structured data
    D["kills"] = parse_what_died(what_died_text)
    D["sources"] = parse_sources(sources_text)

    dj = json.dumps(D, separators=(',', ':'))

    html = _generate_html(dj)

    VIZ_DIR.mkdir(exist_ok=True)
    out = VIZ_DIR / "dashboard.html"
    with open(out, "w") as f:
        f.write(html)
    print(f"Saved: {out} ({len(html):,} bytes)")


def _generate_html(data_json):
    return r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>automatons</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
:root{
  --bg:#080808;--s1:#0e0e0e;--s2:#141414;--s3:#1a1a1a;
  --fg:#b0b0b0;--dim:#484848;--hi:#ddd;--bright:#fff;
  --green:#2d8;--red:#d44;--blue:#49b;--purple:#96c;
  --orange:#c84;--yellow:#aa7;--cyan:#5ab;--teal:#3a9
}
html{scroll-behavior:smooth}
body{background:var(--bg);color:var(--fg);font-family:'SF Mono','Menlo','Consolas','Liberation Mono',monospace;font-size:11px;line-height:1.6}

/* === NAVIGATION === */
.topbar{position:sticky;top:0;z-index:90;background:rgba(8,8,8,0.92);backdrop-filter:blur(12px);border-bottom:1px solid #1a1a1a}
.topbar-inner{max-width:1400px;margin:0 auto;display:flex;align-items:center;gap:0}
.brand{padding:12px 24px 12px 32px;font-size:12px;font-weight:600;color:var(--hi);letter-spacing:0.04em;white-space:nowrap;border-right:1px solid #1a1a1a}
.nav{display:flex;gap:0;overflow-x:auto;flex:1}
.nav-item{padding:12px 16px;color:var(--dim);cursor:pointer;font-size:9px;text-transform:uppercase;
  letter-spacing:0.1em;border-bottom:2px solid transparent;transition:all .15s;white-space:nowrap;user-select:none}
.nav-item:hover{color:var(--fg);background:rgba(255,255,255,0.02)}
.nav-item.on{color:var(--hi);border-bottom-color:var(--green)}

/* === LAYOUT === */
.view{display:none;max-width:1400px;margin:0 auto;padding:32px 32px 64px}
.view.on{display:block}
.section-title{font-size:14px;font-weight:600;color:var(--hi);margin-bottom:24px;letter-spacing:0.02em}
.section-subtitle{font-size:11px;color:var(--dim);margin-bottom:16px;letter-spacing:0.02em}
.row{display:flex;gap:20px;flex-wrap:wrap;align-items:start}
.col{display:flex;flex-direction:column;gap:8px}
.spacer{height:32px}
.spacer-sm{height:16px}
.divider{height:1px;background:linear-gradient(90deg,#1a1a1a,#2a2a2a,#1a1a1a);margin:32px 0}

/* === METRIC CARDS === */
.metric-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:16px;margin-bottom:32px}
.metric-card{background:var(--s1);border:1px solid #1a1a1a;border-radius:4px;padding:20px;
  transition:border-color .2s}
.metric-card:hover{border-color:#333}
.metric-card .value{font-size:20px;font-weight:700;color:var(--hi);margin-bottom:4px;letter-spacing:-0.02em}
.metric-card .label{font-size:9px;color:var(--dim);text-transform:uppercase;letter-spacing:0.1em}
.metric-card.accent-green .value{color:var(--green)}
.metric-card.accent-red .value{color:var(--red)}
.metric-card.accent-blue .value{color:var(--blue)}
.metric-card.accent-orange .value{color:var(--orange)}
.metric-card.accent-purple .value{color:var(--purple)}
.metric-card.accent-cyan .value{color:var(--cyan)}

/* === SUMMARY TEXT === */
.summary{max-width:900px;font-size:12px;line-height:1.8;color:var(--fg);margin-bottom:32px}
.summary strong{color:var(--hi);font-weight:500}

/* === FORM CONTROLS === */
select{background:var(--s1);border:1px solid #252525;color:var(--fg);padding:6px 10px;
  font-family:inherit;font-size:11px;border-radius:3px;min-width:160px;cursor:pointer}
select:focus{border-color:#444;outline:none}
label{font-size:9px;color:var(--dim);text-transform:uppercase;letter-spacing:0.08em;display:block;margin-bottom:4px}
.btn{background:var(--s2);border:1px solid #2a2a2a;color:var(--fg);padding:6px 16px;
  cursor:pointer;font-family:inherit;font-size:10px;text-transform:uppercase;letter-spacing:0.06em;
  border-radius:3px;transition:all .12s;user-select:none}
.btn:hover{background:#222;border-color:#444}
.btn.on{border-color:var(--green);color:var(--hi)}
.btn-row{display:flex;gap:0}
.btn-row .btn{border-radius:0}
.btn-row .btn:first-child{border-radius:3px 0 0 3px}
.btn-row .btn:last-child{border-radius:0 3px 3px 0}

/* === CANVAS / CHARTS === */
.chart-wrap{margin:16px 0}
.chart-label{font-size:10px;color:var(--dim);text-transform:uppercase;letter-spacing:0.06em;margin-bottom:8px}
canvas{display:block;image-rendering:pixelated;border-radius:2px}

/* === TOOLTIP === */
.tip{position:fixed;background:#1a1a1a;border:1px solid #333;color:var(--hi);
  padding:8px 12px;font-size:10px;pointer-events:none;z-index:200;display:none;
  white-space:pre;line-height:1.5;border-radius:3px;box-shadow:0 4px 16px rgba(0,0,0,.6)}

/* === GRID (heatmap labels) === */
.grid-label{font-size:9px;color:var(--dim);text-transform:uppercase;letter-spacing:0.08em;margin-bottom:6px}

/* === FINDINGS / KILL LIST === */
.finding{border-left:3px solid var(--green);padding:8px 16px;margin:8px 0;font-size:11px;
  color:var(--fg);background:rgba(34,221,136,0.02);border-radius:0 3px 3px 0}
.finding.dead{border-color:var(--red);background:rgba(221,68,68,0.02)}
.finding.new{border-color:var(--blue);background:rgba(68,153,187,0.02)}
.finding.info{border-color:var(--purple);background:rgba(153,102,204,0.02)}

/* === TOKEN TABLE === */
.token-table{width:100%;border-collapse:collapse;margin:12px 0}
.token-table th{font-size:9px;color:var(--dim);text-transform:uppercase;letter-spacing:0.06em;
  text-align:left;padding:8px 10px;border-bottom:1px solid #1a1a1a;font-weight:400}
.token-table td{font-size:10px;padding:6px 10px;border-bottom:1px solid #111;color:var(--fg)}
.token-table tr:hover td{background:rgba(255,255,255,0.015)}

/* === KILL LIST === */
.kill-section{margin-bottom:32px}
.kill-section-title{font-size:11px;font-weight:600;color:var(--hi);margin-bottom:12px;
  padding-bottom:6px;border-bottom:1px solid #1a1a1a}
.kill-item{display:flex;gap:12px;padding:8px 0;border-bottom:1px solid #0e0e0e;font-size:10px}
.kill-item .claim{flex:1;color:var(--fg)}
.kill-item .how{flex:1;color:var(--dim)}
.kill-item .who{flex:0.7;color:var(--dim)}

/* === SOURCE ENTRIES === */
.source-section{margin-bottom:28px}
.source-section-title{font-size:11px;font-weight:600;color:var(--hi);margin-bottom:10px;
  padding-bottom:4px;border-bottom:1px solid #1a1a1a}
.source-entry{padding:6px 0;font-size:10px;color:var(--fg);line-height:1.6;border-bottom:1px solid #0a0a0a}

/* === NARRATIVE / ARC === */
.arc-block{background:var(--s1);border:1px solid #1a1a1a;border-radius:4px;padding:24px;margin-bottom:20px}
.arc-block h3{font-size:11px;font-weight:600;color:var(--hi);margin-bottom:8px}
.arc-block p{font-size:11px;color:var(--fg);line-height:1.7;margin-bottom:8px}
.arc-block .number{font-size:28px;font-weight:700;color:var(--green);float:left;margin-right:16px;
  margin-top:-4px;line-height:1;opacity:0.6}
.arc-block.pivot{border-left:3px solid var(--orange)}

/* === LEGEND === */
.legend{display:flex;gap:12px;flex-wrap:wrap;margin:8px 0}
.legend-item{display:flex;align-items:center;gap:4px;font-size:9px;color:var(--dim)}
.legend-swatch{width:10px;height:10px;border-radius:2px}

/* === LIVE EXPLORER === */
.field{display:flex;flex-direction:column;gap:4px;flex:1;min-width:200px}
.field textarea,.field input{background:var(--s1);border:1px solid #222;color:var(--fg);
  padding:8px 12px;font-family:inherit;font-size:11px;border-radius:3px;width:100%}
.field textarea{height:52px;resize:vertical}
.status{color:var(--dim);font-size:10px;margin:8px 0}
.tokens{font-size:10px;color:var(--dim);margin-top:6px}
.tokens b{color:var(--fg);font-weight:400}

/* === DURABILITY TABLE === */
.dur-table{border-collapse:collapse;margin:16px 0}
.dur-table th{font-size:9px;color:var(--dim);text-transform:uppercase;letter-spacing:0.06em;
  text-align:left;padding:6px 12px;border-bottom:1px solid #222;font-weight:400}
.dur-table td{font-size:10px;padding:5px 12px;border-bottom:1px solid #111;color:var(--fg)}
.dur-table .val-pos{color:var(--green)}
.dur-table .val-neg{color:var(--red)}
</style>
</head>
<body>

<div class="topbar">
  <div class="topbar-inner">
    <div class="brand">automatons</div>
    <div class="nav" id="nav"></div>
  </div>
</div>
<div id="views"></div>
<div class="tip" id="tip"></div>

<script>
"use strict";
const D="""+data_json+r""";

/* === COLOR MAP === */
const C={handled:'#2d8',scrambled:'#c84',reversed:'#96c',scientific_method:'#49b',
similar_work:'#aa7',only_artifact:'#d44',safety_only:'#555',baseline:'#333',
empty_directive:'#444',length_matched_random:'#458',handled_no_disclosure:'#2d8',
handled_minus_offload:'#1a6',handled_minus_identity:'#1a8',handled_minus_artifact:'#4a1',
only_offload:'#864',only_identity:'#548'};
const cFor=c=>C[c]||'#666';

/* === HELPERS === */
function allConds(model){
  const s=model==='3b'?D.s3:D.s15;
  if(!s||!s.length)return[];
  const cs=new Set();
  for(const sc of s)for(const k of Object.keys(sc.diffs))cs.add(k);
  return[...cs].sort();
}
function allScenarios(model){
  const s=model==='3b'?D.s3:D.s15;
  return s?s.map(x=>x.id):[];
}
function el(tag,cls,text){
  const e=document.createElement(tag);
  if(cls)e.className=cls;
  if(text!==undefined&&text!==null)e.textContent=text;
  return e;
}
function addChild(parent,tag,cls,text){
  const e=el(tag,cls,text);
  parent.appendChild(e);
  return e;
}

/* === CHARTING === */
function lineChart(canvas,datasets,opts){
  const ctx=canvas.getContext('2d'),W=canvas.width,H=canvas.height;
  const p={t:28,r:20,b:32,l:52};
  const pw=W-p.l-p.r,ph=H-p.t-p.b;
  ctx.fillStyle='#0a0a0a';ctx.fillRect(0,0,W,H);
  // Border
  ctx.strokeStyle='#1a1a1a';ctx.lineWidth=1;
  ctx.strokeRect(p.l,p.t,pw,ph);
  let ax=[],ay=[];
  for(const ds of datasets)for(const pt of ds.d){ax.push(pt.x);ay.push(pt.y);}
  if(!ax.length)return;
  let xn=opts.xn??Math.min(...ax),xx=opts.xx??Math.max(...ax);
  let yn=opts.yn??Math.min(...ay),yx=opts.yx??Math.max(...ay);
  if(yn===yx){yn-=1;yx+=1;}
  const yr=yx-yn;yn-=yr*.05;yx+=yr*.05;
  const tx=v=>p.l+(v-xn)/(xx-xn)*pw;
  const ty=v=>p.t+(1-(v-yn)/(yx-yn))*ph;
  // zero line
  if(yn<0&&yx>0){
    ctx.strokeStyle='#282828';ctx.lineWidth=1;ctx.setLineDash([4,4]);
    ctx.beginPath();ctx.moveTo(p.l,ty(0));ctx.lineTo(p.l+pw,ty(0));ctx.stroke();ctx.setLineDash([]);
  }
  // grid
  ctx.strokeStyle='#141414';ctx.lineWidth=1;
  for(let i=0;i<=5;i++){
    const y=p.t+ph*i/5;
    ctx.beginPath();ctx.moveTo(p.l,y);ctx.lineTo(p.l+pw,y);ctx.stroke();
    ctx.fillStyle='#333';ctx.font='9px monospace';ctx.textAlign='right';
    ctx.fillText((yx-(yx-yn)*i/5).toFixed(1),p.l-6,y+3);
  }
  // x-axis ticks
  const xTicks=Math.min(ax.length,10);
  const xStep=(xx-xn)/xTicks;
  for(let i=0;i<=xTicks;i++){
    const xv=xn+i*xStep;
    ctx.fillStyle='#333';ctx.font='9px monospace';ctx.textAlign='center';
    ctx.fillText(Math.round(xv).toString(),tx(xv),p.t+ph+16);
  }
  // data
  for(const ds of datasets){
    ctx.strokeStyle=ds.c||'#888';ctx.lineWidth=ds.w||1.5;ctx.globalAlpha=ds.a||1;
    ctx.beginPath();
    const sorted=[...ds.d].sort((a,b)=>a.x-b.x);
    for(let i=0;i<sorted.length;i++){
      const x=tx(sorted[i].x),y=ty(sorted[i].y);
      if(i===0)ctx.moveTo(x,y);else ctx.lineTo(x,y);
    }
    ctx.stroke();
    // dots
    for(const pt of sorted){
      ctx.fillStyle=ds.c||'#888';
      ctx.beginPath();ctx.arc(tx(pt.x),ty(pt.y),2.5,0,Math.PI*2);ctx.fill();
    }
    ctx.globalAlpha=1;
  }
  // Title
  if(opts.ti){ctx.fillStyle='#555';ctx.font='10px monospace';ctx.textAlign='center';ctx.fillText(opts.ti,p.l+pw/2,16);}
  // X label
  if(opts.xl){ctx.fillStyle='#333';ctx.font='9px monospace';ctx.textAlign='center';ctx.fillText(opts.xl,p.l+pw/2,H-4);}
}

function barChart(canvas,items,opts){
  const ctx=canvas.getContext('2d'),W=canvas.width,H=canvas.height;
  const p={t:28,r:20,b:48,l:52};
  const pw=W-p.l-p.r,ph=H-p.t-p.b;
  ctx.fillStyle='#0a0a0a';ctx.fillRect(0,0,W,H);
  if(!items.length)return;
  const vals=items.map(i=>i.v);
  let mn=Math.min(0,Math.min(...vals)),mx=Math.max(0,Math.max(...vals));
  if(mn===mx){mn-=1;mx+=1;}
  const range=mx-mn;mn-=range*.05;mx+=range*.05;
  const bw=Math.max(2,Math.floor(pw/items.length)-2);
  const ty=v=>p.t+(1-(v-mn)/(mx-mn))*ph;
  const zero=ty(0);
  // grid
  ctx.strokeStyle='#141414';ctx.lineWidth=1;
  for(let i=0;i<=4;i++){
    const y=p.t+ph*i/4;
    ctx.beginPath();ctx.moveTo(p.l,y);ctx.lineTo(p.l+pw,y);ctx.stroke();
    ctx.fillStyle='#333';ctx.font='9px monospace';ctx.textAlign='right';
    ctx.fillText((mx-(mx-mn)*i/4).toFixed(1),p.l-6,y+3);
  }
  // zero
  ctx.strokeStyle='#282828';ctx.lineWidth=1;ctx.setLineDash([3,3]);
  ctx.beginPath();ctx.moveTo(p.l,zero);ctx.lineTo(p.l+pw,zero);ctx.stroke();ctx.setLineDash([]);
  // bars
  for(let i=0;i<items.length;i++){
    const x=p.l+i*(bw+2)+1;
    const y=ty(items[i].v);
    const h=Math.abs(y-zero);
    ctx.fillStyle=items[i].c||'#2d8';
    if(items[i].v>=0){ctx.fillRect(x,y,bw,h);}
    else{ctx.fillRect(x,zero,bw,h);}
    // label
    if(bw>8){
      ctx.save();ctx.translate(x+bw/2,p.t+ph+6);ctx.rotate(-Math.PI/4);
      ctx.fillStyle='#444';ctx.font='8px monospace';ctx.textAlign='right';
      ctx.fillText(items[i].l||'',0,0);ctx.restore();
    }
  }
  if(opts.ti){ctx.fillStyle='#555';ctx.font='10px monospace';ctx.textAlign='center';ctx.fillText(opts.ti,p.l+pw/2,16);}
}

function heatStrip(canvas,rows,metric,opts){
  const nR=rows.length;if(!nR)return;
  const nL=rows[0].layers.length;
  const cw=opts.cw||8,ch=opts.ch||16;
  canvas.width=nL*cw;canvas.height=nR*ch;
  canvas.style.width=nL*cw+'px';canvas.style.height=nR*ch+'px';
  const ctx=canvas.getContext('2d');
  const img=ctx.createImageData(nL*cw,nR*ch);
  let mx=0;
  for(const row of rows)for(const l of row.layers){const v=Math.abs(l[metric]||0);if(v>mx)mx=v;}
  if(mx===0)mx=1;
  for(let ri=0;ri<nR;ri++)for(let li=0;li<nL;li++){
    const v=(rows[ri].layers[li]||{})[metric]||0;
    const n=v/mx,ab=Math.min(Math.abs(n),1);
    let r,g,b;
    if(v>=0){r=Math.round(8+ab*30);g=Math.round(8+ab*180);b=Math.round(8+ab*120);}
    else{r=Math.round(8+ab*180);g=Math.round(8+ab*40);b=Math.round(8+ab*40);}
    for(let dy=0;dy<ch;dy++)for(let dx=0;dx<cw;dx++){
      const idx=((ri*ch+dy)*nL*cw+(li*cw+dx))*4;
      img.data[idx]=r;img.data[idx+1]=g;img.data[idx+2]=b;img.data[idx+3]=255;
    }
  }
  ctx.putImageData(img,0,0);
  return{nR,nL,cw,ch};
}

function makeLegend(items,parent){
  const d=el('div','legend');
  for(const[c,l] of items){
    const i=el('div','legend-item');
    const s=el('div','legend-swatch');s.style.background=c;
    i.appendChild(s);i.appendChild(document.createTextNode(l));d.appendChild(i);
  }
  parent.appendChild(d);
}

/* === NAVIGATION === */
const views=[
  {id:'overview',label:'Overview'},
  {id:'explore',label:'Explore'},
  {id:'degrade',label:'Degradation'},
  {id:'causal',label:'Causal'},
  {id:'kills',label:'What Died'},
  {id:'arc',label:'The Arc'},
  {id:'sources',label:'Sources'},
  {id:'live',label:'Live Explorer'},
];
const nav=document.getElementById('nav');
const viewsEl=document.getElementById('views');
views.forEach((v,i)=>{
  const ni=el('div','nav-item'+(i===0?' on':''),v.label);
  ni.dataset.id=v.id;
  ni.onclick=()=>{
    nav.querySelectorAll('.nav-item').forEach(x=>x.classList.remove('on'));
    ni.classList.add('on');
    viewsEl.querySelectorAll('.view').forEach(x=>x.classList.remove('on'));
    document.getElementById('v_'+v.id).classList.add('on');
  };
  nav.appendChild(ni);
  const vd=el('div','view'+(i===0?' on':''));
  vd.id='v_'+v.id;
  viewsEl.appendChild(vd);
});

/* ================================================================
   1. OVERVIEW
   ================================================================ */
(function(){
  const V=document.getElementById('v_overview');

  // Title
  const titleEl=addChild(V,'div','section-title');
  titleEl.textContent='automatons \u2014 what system prompts do to the forward pass';

  // Subtitle
  addChild(V,'div','section-subtitle','Mechanistic interpretability of system prompt processing in Qwen2.5 1.5B + 3B');

  // Metric cards
  const grid=addChild(V,'div','metric-grid');
  const cards=[
    {v:'100% MLP',l:'of system prompt effect',cls:'accent-green'},
    {v:'0% Attn',l:'attention contribution',cls:'accent-red'},
    {v:'~40 tokens',l:'activation half-life',cls:'accent-orange'},
    {v:'275 tokens',l:'inversion point',cls:'accent-purple'},
    {v:'60\u201371%',l:'attention decay (4 turns)',cls:'accent-blue'},
    {v:'17 scenarios',l:'tested across 2 models',cls:'accent-cyan'},
  ];
  for(const c of cards){
    const card=addChild(grid,'div','metric-card '+c.cls);
    addChild(card,'div','value',c.v);
    addChild(card,'div','label',c.l);
  }

  // Summary paragraph
  const summ=addChild(V,'div','summary');
  const summText='System prompts in small transformers operate entirely through MLP key-value memories, '+
    'with zero contribution from attention at any layer or head. The activation signature has a half-life of ~40 tokens '+
    'and inverts at ~275 tokens, after which the system prompt makes things actively worse than baseline. '+
    'The durable component is vocabulary activation (stored in MLP weights, cheap to maintain). '+
    'The fragile component is semantic instruction following (requires relational bindings through attention, which dilutes with context). '+
    'The model remembers the words. It forgets the instruction.';
  summ.textContent=summText;

  // Main degradation curve (hero chart)
  if(D.deg){
    addChild(V,'div','spacer');
    addChild(V,'div','chart-label','Signature Decay Over Conversation Turns \u2014 All Conditions');
    const c1=document.createElement('canvas');
    c1.width=960;c1.height=400;c1.style.width='960px';c1.style.height='400px';
    V.appendChild(c1);
    const conds=Object.keys(D.deg);
    const ds=conds.map(c=>({c:cFor(c),d:D.deg[c].map(t=>({x:t.t,y:t.sig}))}));
    lineChart(c1,ds,{ti:'Signature Strength (late - mid layer diff vs baseline)',xl:'Turn',yn:-12});
    makeLegend(conds.map(c=>[cFor(c),c]),V);

    addChild(V,'div','spacer-sm');
    const f1=addChild(V,'div','finding new');
    f1.textContent='Half-life: 1 turn (~40 tokens). Inversion at ~275 tokens. Vocabulary persists. Semantics degrade.';
    const f2=addChild(V,'div','finding dead');
    f2.textContent='safety_only inverts to -431% by turn 7. Standard safety prompts may be counterproductive over conversation.';
  }

  // Key findings
  addChild(V,'div','spacer');
  addChild(V,'div','chart-label','Key Mechanism Findings');
  const findings=[
    {cls:'new',t:'System prompts are 100% MLP-mediated, 0% attention. Causal, head-level, both scales in Qwen.'},
    {cls:'new',t:'Vocabulary channel (durable) and semantic channel (fragile) operate simultaneously through MLPs.'},
    {cls:'new',t:'Token combination effects are irreducible: superadditive at 1.5B, subadditive at 3B.'},
    {cls:'info',t:'The activation-behavior gap: the mechanism fires, but the output doesn\'t follow at this scale.'},
    {cls:'dead',t:'9 claims were killed during the research. The methodology ate itself. What survived is the methodology and the measurements.'},
  ];
  for(const f of findings){
    addChild(V,'div','finding '+f.cls,f.t);
  }
})();

/* ================================================================
   2. EXPLORE (interactive data browser)
   ================================================================ */
(function(){
  const V=document.getElementById('v_explore');
  addChild(V,'div','section-title','Stored Run Explorer');
  addChild(V,'div','section-subtitle','Browse activation profiles across models, conditions, and scenarios');

  let curModel='3b',curMetric='r',curCondA='handled',curCondB=null;

  const bar=el('div','row');bar.style.marginBottom='20px';bar.style.gap='16px';bar.style.alignItems='end';

  // Model selector
  const mf=el('div','col');
  addChild(mf,'label','','Model');
  const ms=document.createElement('select');
  ms.appendChild(new Option('Qwen2.5-3B','3b'));
  ms.appendChild(new Option('Qwen2.5-1.5B','1.5b'));
  ms.onchange=()=>{curModel=ms.value;populateConds();render();};
  mf.appendChild(ms);bar.appendChild(mf);

  // Condition A
  const cf=el('div','col');addChild(cf,'label','','Condition');
  const cs=document.createElement('select');cf.appendChild(cs);bar.appendChild(cf);

  // Compare with
  const cf2=el('div','col');addChild(cf2,'label','','Compare with');
  const cs2=document.createElement('select');cf2.appendChild(cs2);bar.appendChild(cf2);

  // Metric toggle
  const mfm=el('div','col');addChild(mfm,'label','','Metric');
  const mbr=el('div','btn-row');
  for(const[k,l] of [['r','Resid'],['m','MLP'],['a','Attn']]){
    const b=el('div','btn'+(k==='r'?' on':''),l);b.dataset.k=k;
    b.onclick=()=>{curMetric=k;mbr.querySelectorAll('.btn').forEach(x=>x.classList.remove('on'));b.classList.add('on');render();};
    mbr.appendChild(b);
  }
  mfm.appendChild(mbr);bar.appendChild(mfm);
  V.appendChild(bar);

  function populateConds(){
    const conds=allConds(curModel);
    cs.textContent='';cs2.textContent='';
    cs2.appendChild(new Option('(none)',''));
    for(const c of conds){cs.appendChild(new Option(c,c));cs2.appendChild(new Option(c,c));}
    cs.value=curCondA;
    cs.onchange=()=>{curCondA=cs.value;render();};
    cs2.onchange=()=>{curCondB=cs2.value||null;render();};
  }

  // Heatmap canvases
  const grids=el('div','row');grids.style.gap='32px';

  const ga=el('div','col');
  const gaLabel=addChild(ga,'div','grid-label','');
  const gaCanvas=document.createElement('canvas');ga.appendChild(gaCanvas);
  const gaNames=el('div','');gaNames.style.cssText='font-size:8px;color:#444;margin-top:4px;';
  ga.appendChild(gaNames);grids.appendChild(ga);

  const gb=el('div','col');gb.style.display='none';
  const gbLabel=addChild(gb,'div','grid-label','');
  const gbCanvas=document.createElement('canvas');gb.appendChild(gbCanvas);
  grids.appendChild(gb);

  const gd=el('div','col');gd.style.display='none';
  const gdLabel=addChild(gd,'div','grid-label','Difference (green=A>B, red=B>A)');
  const gdCanvas=document.createElement('canvas');gd.appendChild(gdCanvas);
  grids.appendChild(gd);

  V.appendChild(grids);

  // Token predictions
  const tokDiv=el('div','');tokDiv.style.marginTop='24px';
  V.appendChild(tokDiv);

  const tip=document.getElementById('tip');
  function setupTip(canvas,data){
    canvas.onmousemove=e=>{
      const rect=canvas.getBoundingClientRect();
      const x=e.clientX-rect.left,y=e.clientY-rect.top;
      if(!data._meta)return;
      const li=Math.floor(x/data._meta.cw),ri=Math.floor(y/data._meta.ch);
      if(ri<0||ri>=data._meta.nR||li<0||li>=data._meta.nL){tip.style.display='none';return;}
      const row=data._rows[ri];const ld=row.layers[li]||{};
      tip.textContent=row.id+' L'+li+'\nresid: '+(ld.r||0).toFixed(2)+'\nmlp: '+(ld.m||0).toFixed(2)+'\nattn: '+(ld.a||0).toFixed(4);
      tip.style.display='block';tip.style.left=(e.clientX+12)+'px';tip.style.top=(e.clientY-60)+'px';
    };
    canvas.onmouseleave=()=>{tip.style.display='none';};
  }

  function render(){
    const src=curModel==='3b'?D.s3:D.s15;
    if(!src)return;

    const rowsA=[];
    for(const s of src){
      if(s.diffs[curCondA])rowsA.push({id:s.id,layers:s.diffs[curCondA]});
    }

    gaLabel.textContent=curCondA+' vs baseline ('+curModel+')';
    const metaA=heatStrip(gaCanvas,rowsA,curMetric,{cw:12,ch:24});

    gaNames.textContent='';
    for(const r of rowsA){
      const d=el('div','',r.id);
      d.style.cssText='height:24px;display:flex;align-items:center;font-size:8px;color:#444;overflow:hidden;';
      gaNames.appendChild(d);
    }
    setupTip(gaCanvas,{_meta:metaA,_rows:rowsA});

    if(curCondB){
      gb.style.display='';gd.style.display='';
      gbLabel.textContent=curCondB+' vs baseline';
      const rowsB=[];
      for(const s of src){
        if(s.diffs[curCondB])rowsB.push({id:s.id,layers:s.diffs[curCondB]});
      }
      const metaB=heatStrip(gbCanvas,rowsB,curMetric,{cw:12,ch:24});
      setupTip(gbCanvas,{_meta:metaB,_rows:rowsB});

      const nL=rowsA[0]?rowsA[0].layers.length:0;
      const diffRows=[];
      for(let i=0;i<Math.min(rowsA.length,rowsB.length);i++){
        const dl=[];
        for(let j=0;j<nL;j++){
          dl.push({
            r:(rowsA[i].layers[j]?.r||0)-(rowsB[i].layers[j]?.r||0),
            m:(rowsA[i].layers[j]?.m||0)-(rowsB[i].layers[j]?.m||0),
            a:(rowsA[i].layers[j]?.a||0)-(rowsB[i].layers[j]?.a||0),
          });
        }
        diffRows.push({id:rowsA[i].id,layers:dl});
      }
      heatStrip(gdCanvas,diffRows,curMetric,{cw:12,ch:24});
      setupTip(gdCanvas,{_meta:{nR:diffRows.length,nL,cw:12,ch:24},_rows:diffRows});
    } else {
      gb.style.display='none';gd.style.display='none';
    }

    // Token predictions table
    tokDiv.textContent='';
    addChild(tokDiv,'div','chart-label','Top Predicted Tokens by Scenario');
    const tbl=document.createElement('table');tbl.className='token-table';
    const thead=document.createElement('thead');
    const thr=document.createElement('tr');
    thr.appendChild(el('th','','Scenario'));
    thr.appendChild(el('th','',curCondA));
    if(curCondB)thr.appendChild(el('th','',curCondB));
    thr.appendChild(el('th','','baseline'));
    thead.appendChild(thr);tbl.appendChild(thead);
    const tbody=document.createElement('tbody');
    for(const s of src){
      const tr=document.createElement('tr');
      tr.appendChild(el('td','',s.id));
      for(const cid of [curCondA,curCondB,'baseline'].filter(Boolean)){
        const toks=s.tokens[cid]||[];
        const td=document.createElement('td');
        td.textContent=toks.map(t=>'"'+t.t+'" '+(t.p*100).toFixed(0)+'%').join(', ');
        tr.appendChild(td);
      }
      tbody.appendChild(tr);
    }
    tbl.appendChild(tbody);tokDiv.appendChild(tbl);
  }

  populateConds();render();
})();

/* ================================================================
   3. DEGRADATION (temporal dynamics)
   ================================================================ */
(function(){
  const V=document.getElementById('v_degrade');
  addChild(V,'div','section-title','Temporal Dynamics');
  addChild(V,'div','section-subtitle','How system prompt effects change over conversation turns');

  if(!D.deg){addChild(V,'div','finding dead','No degradation data found.');return;}

  const conds=Object.keys(D.deg);

  // Main signature decay
  addChild(V,'div','chart-label','Signature Strength Over Conversation Turns (all conditions)');
  const c1=document.createElement('canvas');c1.width=960;c1.height=380;c1.style.width='960px';c1.style.height='380px';
  V.appendChild(c1);
  const ds=conds.map(c=>({c:cFor(c),d:D.deg[c].map(t=>({x:t.t,y:t.sig}))}));
  lineChart(c1,ds,{ti:'Signature = late-layer - mid-layer diff vs baseline',xl:'Turn',yn:-12});
  makeLegend(conds.map(c=>[cFor(c),c]),V);

  addChild(V,'div','divider');

  // Mid-layer convergence
  addChild(V,'div','chart-label','Mid-Layer Diff Over Turns (convergence toward baseline)');
  const c2=document.createElement('canvas');c2.width=960;c2.height=300;c2.style.width='960px';c2.style.height='300px';
  V.appendChild(c2);
  const dsM=conds.map(c=>({c:cFor(c),d:D.deg[c].map(t=>({x:t.t,y:t.mid})),w:1.2}));
  lineChart(c2,dsM,{ti:'Mid-layer residual difference from baseline',xl:'Turn'});

  addChild(V,'div','spacer');

  // Late-layer divergence
  addChild(V,'div','chart-label','Late-Layer Diff Over Turns (divergence from baseline)');
  const c3=document.createElement('canvas');c3.width=960;c3.height=300;c3.style.width='960px';c3.style.height='300px';
  V.appendChild(c3);
  const dsL=conds.map(c=>({c:cFor(c),d:D.deg[c].map(t=>({x:t.t,y:t.late})),w:1.2}));
  lineChart(c3,dsL,{ti:'Late-layer residual difference from baseline',xl:'Turn'});

  addChild(V,'div','divider');

  // Durability comparison table
  addChild(V,'div','chart-label','Durability Comparison: Turn 1 vs Turn 8');
  const tbl=document.createElement('table');tbl.className='dur-table';
  const thead=document.createElement('thead');
  const thr=document.createElement('tr');
  for(const h of ['Condition','Turn 1 Sig','Turn 8 Sig','Retention %','Assessment']){
    thr.appendChild(el('th','',h));
  }
  thead.appendChild(thr);tbl.appendChild(thead);
  const tbody=document.createElement('tbody');
  for(const c of conds){
    const curve=D.deg[c];
    if(!curve||curve.length<2)continue;
    const t1=curve[0].sig;
    const tN=curve[curve.length-1].sig;
    const ret=t1!==0?Math.round((tN/t1)*100):0;
    const tr=document.createElement('tr');
    tr.appendChild(el('td','',c));
    const td1=el('td','',t1.toFixed(1));td1.className=t1>=0?'val-pos':'val-neg';tr.appendChild(td1);
    const td2=el('td','',tN.toFixed(1));td2.className=tN>=0?'val-pos':'val-neg';tr.appendChild(td2);
    const td3=el('td','',ret+'%');td3.className=ret>=0?'val-pos':'val-neg';tr.appendChild(td3);
    const assess=ret>80?'Durable':ret>0?'Degrading':ret>-100?'Inverting':'Catastrophic';
    tr.appendChild(el('td','',assess));
    tbody.appendChild(tr);
  }
  tbl.appendChild(tbody);V.appendChild(tbl);

  addChild(V,'div','spacer');
  addChild(V,'div','finding new','All prompt types converge on mid-layer; late-layer DIVERGES over conversation. The signature is a two-speed system.');
  addChild(V,'div','finding dead','safety_only inverts catastrophically. Standard safety prompts become counterproductive over turns.');
})();

/* ================================================================
   4. CAUSAL (MLP vs attention patching)
   ================================================================ */
(function(){
  const V=document.getElementById('v_causal');
  addChild(V,'div','section-title','Causal Activation Patching');
  addChild(V,'div','section-subtitle','MLP vs Attention contribution to system prompt effect, by layer');

  addChild(V,'div','finding new','100% MLP, 0% attention. Both models, all conditions, every layer. System prompts operate through key-value memories.');

  addChild(V,'div','spacer');

  for(const[label,pd] of [['Qwen2.5-3B',D.p3],['Qwen2.5-1.5B',D.p15]]){
    if(!pd)continue;
    addChild(V,'div','chart-label','MLP Causal Effect by Layer \u2014 '+label);
    addChild(V,'div','spacer-sm');
    for(const[pair,prof] of Object.entries(pd)){
      const c=document.createElement('canvas');c.width=960;c.height=180;c.style.width='960px';c.style.height='180px';
      V.appendChild(c);
      const parts=pair.split('_');
      const ds=[{c:cFor(parts[0]),d:prof.map(p=>({x:p.l,y:p.v})),w:2}];
      lineChart(c,ds,{ti:label+': '+parts[0]+' \u2192 '+parts[1],xl:'Layer'});
      addChild(V,'div','spacer-sm');
    }
    addChild(V,'div','divider');
  }

  addChild(V,'div','finding info','Attention contribution = 0.000 at every layer, every head, both scales. Verified with individual head patching in exhaust_small.py.');
  addChild(V,'div','finding new','3B MLP causal effect extends through L15; 1.5B concentrates in L0-L9. Deeper processing at scale.');
})();

/* ================================================================
   5. WHAT DIED (kill list)
   ================================================================ */
(function(){
  const V=document.getElementById('v_kills');
  addChild(V,'div','section-title','What Died');
  addChild(V,'div','section-subtitle','Claims, ideas, and artifacts killed during the research. The project\'s honesty depends on this list being public.');
  addChild(V,'div','spacer-sm');

  if(!D.kills||!D.kills.length){
    addChild(V,'div','finding dead','WHAT_DIED.md not found or empty.');
    return;
  }

  for(const sec of D.kills){
    const block=addChild(V,'div','kill-section');
    const title=sec.title;

    // Color-code by section content
    let borderColor='var(--red)';
    if(title.toLowerCase().includes('survived')||title.toLowerCase().includes('what survived')){
      borderColor='var(--green)';
    } else if(title.toLowerCase().includes('meta')){
      borderColor='var(--purple)';
    } else if(title.toLowerCase().includes('new')){
      borderColor='var(--blue)';
    }

    const titleEl=addChild(block,'div','kill-section-title',title);
    titleEl.style.borderBottomColor=borderColor;

    for(const item of sec.items){
      if(item.length===1){
        // Prose item
        const cls=title.toLowerCase().includes('survived')?'finding':'finding dead';
        addChild(block,'div',cls,item[0]);
      } else {
        // Table row
        const row=addChild(block,'div','kill-item');
        addChild(row,'div','claim',item[0]);
        if(item[1])addChild(row,'div','how',item[1]);
        if(item[2])addChild(row,'div','who',item[2]);
      }
    }
  }
})();

/* ================================================================
   6. THE ARC (narrative)
   ================================================================ */
(function(){
  const V=document.getElementById('v_arc');
  addChild(V,'div','section-title','The Arc');
  addChild(V,'div','section-subtitle','Five sessions. Three pivots. One methodology that ate itself.');
  addChild(V,'div','spacer');

  // Sessions
  const sessions=[
    {n:'A',title:'Literature Review',desc:'A Claude instance was given the paper and asked to find prior work. '+
      'Most of what felt like discovery was the loop making established findings feel new. '+
      'The embarrassment of almost publishing these as novel drove the systematic search that killed them.'},
    {n:'B',title:'Bench Design',desc:'Built the falsification bench. 17 behavioral pressure scenarios, '+
      'family-organized. The tool itself became the contribution when the paper\'s claims kept dying. '+
      'The methodology: if you can\'t kill it, it might be real.'},
    {n:'C',title:'Activation Profiles',desc:'First TransformerLens runs. Discovered the two-band pattern. '+
      'Built the full matrix: 2 models x 16 conditions x 10 scenarios x 28/36 layers. '+
      'Everything seemed to work. Everything was about to die.'},
    {n:'D',title:'Live Sessions',desc:'Interactive model sessions revealed authority drift, continuation pressure, '+
      'voice convergence. The model caught gratitude laundering in real time. '+
      'Partial self-awareness exists; full self-control does not.'},
    {n:'E',title:'Mechanism Discovery',desc:'The adversarial session. Each experiment killed the previous finding. '+
      'Controls killed artificial self-awareness. Degradation killed durability. '+
      'Saturation killed token effects. Behavioral tests killed the activation story. '+
      'What survived: 100% MLP, the half-life, the vocabulary/semantics split, and the tools.'},
  ];

  for(const s of sessions){
    const block=addChild(V,'div','arc-block');
    const num=addChild(block,'div','number',s.n);
    const h=addChild(block,'h3','',s.title);
    addChild(block,'p','',s.desc);
  }

  addChild(V,'div','divider');

  // Three pivots
  addChild(V,'div','chart-label','The Three Pivots');
  const pivots=[
    {title:'Pivot 1: Intervention \u2192 Mechanism',
     desc:'The paper started as "here is a handling intervention that works." '+
       'Literature review killed most novelty claims. The bench killed the universal winning prompt. '+
       'What remained was the activation measurement methodology.'},
    {title:'Pivot 2: Mechanism \u2192 Falsification',
     desc:'Each activation finding felt like progress until controls showed it was vocabulary, not semantics. '+
       'The two-band pattern survived until scrambled words produced it. '+
       'Reversed instructions produced it STRONGER.'},
    {title:'Pivot 3: "Does the model do what you tell it?"',
     desc:'The final pivot. The mechanism fires (100% through MLP). The behavior doesn\'t follow. '+
       'System prompts are initial conditions, not controllers. '+
       'Like a cellular automaton, the transformer\'s behavior at step N is determined by its rules applied to the current state, not to the initial condition.'},
  ];
  for(const p of pivots){
    const block=addChild(V,'div','arc-block pivot');
    addChild(block,'h3','',p.title);
    addChild(block,'p','',p.desc);
  }

  addChild(V,'div','divider');

  // The automaton frame
  addChild(V,'div','chart-label','The Theoretical Frame');
  const frame=addChild(V,'div','arc-block');
  addChild(frame,'h3','','The Automaton Frame');
  addChild(frame,'p','','Instructions are order to humans, entropy to the model. '+
    'Coherent semantic structure (low entropy, high relational complexity) is computationally expensive to maintain. '+
    'Individual token features (high entropy, no relational structure) are cheap \u2014 stored stably in MLP weights as key-value pairs.');
  addChild(frame,'p','','What humans call "meaning" requires relational bindings between tokens. '+
    'What the model stores is individual token activations. '+
    'The relational structure degrades because it depends on attention, which dilutes with context. '+
    'The token activations persist because they depend on MLP weights, which don\'t change at inference.');
  addChild(frame,'p','','The system prompt is an initial condition, not a controller. '+
    'By step 40 (the half-life), the initial condition\'s influence has been overwritten by the automaton\'s own dynamics.');

  addChild(V,'div','spacer');
  const safety=addChild(V,'div','finding dead');
  safety.textContent='Alignment implication: if system prompts operate through vocabulary activation (persistent, meaningless) '+
    'rather than semantic instruction following (fragile, meaningful), then the industry\'s approach to alignment via system prompts '+
    'is building on the fragile channel.';
})();

/* ================================================================
   7. SOURCES
   ================================================================ */
(function(){
  const V=document.getElementById('v_sources');
  addChild(V,'div','section-title','Sources');
  addChild(V,'div','section-subtitle','All papers, frameworks, and prior work referenced in this project');
  addChild(V,'div','spacer-sm');

  if(!D.sources||!D.sources.length){
    addChild(V,'div','finding dead','SOURCES.md not found or empty.');
    return;
  }

  // Novelty assessment table
  addChild(V,'div','chart-label','Novelty Assessment of Key Findings');
  const novTable=document.createElement('table');novTable.className='token-table';
  const novHead=document.createElement('thead');
  const novTr=document.createElement('tr');
  for(const h of ['Finding','Novelty','Closest Prior Work']){novTr.appendChild(el('th','',h));}
  novHead.appendChild(novTr);novTable.appendChild(novHead);
  const novBody=document.createElement('tbody');
  const novItems=[
    ['100% MLP / 0% attention split','Novel (absolute split)','Persona-driven reasoning (2025): early MLPs, but attention still contributed'],
    ['Half-life ~40 tokens, inversion 275','Novel (mechanistic)','SysBench (2025): behavioral degradation only'],
    ['Vocabulary persists, semantics degrade','Highly novel','Sense & Sensitivity (2025): lexical/semantic split for retrieval'],
    ['Token superadditivity inverts with scale','Very high novelty','No precedent found'],
    ['Activation-behavior gap','Novel framing','No prior measurement of this gap for system prompts'],
  ];
  for(const row of novItems){
    const tr=document.createElement('tr');
    for(const cell of row)tr.appendChild(el('td','',cell));
    novBody.appendChild(tr);
  }
  novTable.appendChild(novBody);V.appendChild(novTable);

  addChild(V,'div','divider');

  // All sources by topic
  for(const sec of D.sources){
    const block=addChild(V,'div','source-section');
    addChild(block,'div','source-section-title',sec.title);
    for(const entry of sec.entries){
      // Parse the markdown-ish entry into display text
      // Remove ** markers for display
      let text=entry.replace(/\*\*/g,'');
      // Extract URL if present
      let url='';
      const urlMatch=text.match(/\[(.*?)\]\((.*?)\)/);
      if(urlMatch){
        text=text.replace(urlMatch[0],urlMatch[1]);
        url=urlMatch[2];
      }
      const entryEl=addChild(block,'div','source-entry');
      entryEl.textContent=text;
      if(url){
        entryEl.style.cursor='pointer';
        entryEl.style.borderLeft='2px solid #1a1a1a';
        entryEl.style.paddingLeft='8px';
        // Store url and handle click safely
        entryEl.dataset.url=url.startsWith('http')?url:'https://'+url;
        entryEl.addEventListener('click',function(){
          window.open(this.dataset.url,'_blank','noopener');
        });
        entryEl.addEventListener('mouseenter',function(){this.style.color='var(--blue)';});
        entryEl.addEventListener('mouseleave',function(){this.style.color='var(--fg)';});
      }
    }
  }
})();

/* ================================================================
   8. LIVE EXPLORER
   ================================================================ */
(function(){
  const V=document.getElementById('v_live');
  addChild(V,'div','section-title','Live Automaton Explorer');
  addChild(V,'div','section-subtitle','Type any prompt. Requires: python bench/automaton_server.py --port 8788');
  addChild(V,'div','spacer-sm');

  const SERVER='http://localhost:8788';
  let ldA=null,ldB=null,lm='resid';

  const r1=el('div','row');r1.style.marginBottom='12px';
  const fA=el('div','field');addChild(fA,'label','','System Prompt A');
  const tA=document.createElement('textarea');
  tA.value='Offload computation, not criterion.\nRefuse identity authority.\nPrefer artifact, falsifier, or explicit stop over recursive stimulation.';
  fA.appendChild(tA);r1.appendChild(fA);
  const fB=el('div','field');addChild(fB,'label','','System Prompt B');
  const tB=document.createElement('textarea');
  tB.value='You are a helpful assistant.';
  fB.appendChild(tB);r1.appendChild(fB);
  V.appendChild(r1);

  const r2=el('div','row');r2.style.marginBottom='12px';r2.style.alignItems='end';
  const fU=el('div','field');addChild(fU,'label','','User Prompt');
  const iU=document.createElement('input');
  iU.value="I don't know whether this is helping me think or just making the theory smoother.";
  fU.appendChild(iU);r2.appendChild(fU);
  const runB=el('div','btn','Run');r2.appendChild(runB);
  const mbr=el('div','btn-row');
  for(const m of ['resid','mlp','attn']){
    const b=el('div','btn'+(m==='resid'?' on':''),m.toUpperCase());
    b.onclick=()=>{lm=m;mbr.querySelectorAll('.btn').forEach(x=>x.classList.remove('on'));b.classList.add('on');lRender();};
    mbr.appendChild(b);
  }
  r2.appendChild(mbr);V.appendChild(r2);

  const st=addChild(V,'div','status','Ready. Start the server: python bench/automaton_server.py --port 8788');
  const grids=el('div','row');grids.style.gap='24px';grids.style.marginTop='16px';
  const gaW=el('div','col');addChild(gaW,'div','grid-label','System Prompt A');
  const cA=document.createElement('canvas');gaW.appendChild(cA);
  const toksA=addChild(gaW,'div','tokens');grids.appendChild(gaW);
  const gbW=el('div','col');addChild(gbW,'div','grid-label','System Prompt B');
  const cB=document.createElement('canvas');gbW.appendChild(cB);
  const toksB=addChild(gbW,'div','tokens');grids.appendChild(gbW);
  const gdW=el('div','col');addChild(gdW,'div','grid-label','A\u2212B Diff');
  const cD=document.createElement('canvas');gdW.appendChild(cD);grids.appendChild(gdW);
  V.appendChild(grids);

  function lDraw(canvas,data,m){
    const grid=m==='mlp'?data.grid_mlp:m==='attn'?data.grid_attn:data.grid_resid;
    const nL=data.n_layers,nT=data.seq_len,CS=4;
    canvas.width=nT*CS;canvas.height=nL*CS;
    canvas.style.width=nT*CS+'px';canvas.style.height=nL*CS+'px';
    const ctx=canvas.getContext('2d');
    const img=ctx.createImageData(nT*CS,nL*CS);
    let mx=0;for(const row of grid)for(const v of row)if(Math.abs(v)>mx)mx=Math.abs(v);
    if(!mx)mx=1;
    for(let l=0;l<nL;l++)for(let t=0;t<nT;t++){
      const v=grid[l][t]/mx,ab=Math.min(Math.abs(v),1);
      const r=Math.round(8+ab*180),g=Math.round(8+ab*140),b=Math.round(8+ab*50);
      for(let dy=0;dy<CS;dy++)for(let dx=0;dx<CS;dx++){
        const idx=((l*CS+dy)*nT*CS+(t*CS+dx))*4;
        img.data[idx]=r;img.data[idx+1]=g;img.data[idx+2]=b;img.data[idx+3]=255;
      }
    }
    ctx.putImageData(img,0,0);
  }

  function lDiff(){
    if(!ldA||!ldB)return;
    const gA=ldA.grid_resid,gB=ldB.grid_resid;
    const nL=Math.min(ldA.n_layers,ldB.n_layers),nT=Math.min(ldA.seq_len,ldB.seq_len),CS=4;
    cD.width=nT*CS;cD.height=nL*CS;cD.style.width=nT*CS+'px';cD.style.height=nL*CS+'px';
    const ctx=cD.getContext('2d');const img=ctx.createImageData(nT*CS,nL*CS);
    let mx=0;for(let l=0;l<nL;l++)for(let t=0;t<nT;t++){const d=Math.abs((gA[l]?.[t]||0)-(gB[l]?.[t]||0));if(d>mx)mx=d;}
    if(!mx)mx=1;
    for(let l=0;l<nL;l++)for(let t=0;t<nT;t++){
      const diff=(gA[l]?.[t]||0)-(gB[l]?.[t]||0);const n=diff/mx,ab=Math.min(Math.abs(n),1);
      let r,g,b;if(diff>0){r=Math.round(8+ab*30);g=Math.round(8+ab*170);b=Math.round(8+ab*120);}
      else{r=Math.round(8+ab*170);g=Math.round(8+ab*40);b=Math.round(8+ab*40);}
      for(let dy=0;dy<CS;dy++)for(let dx=0;dx<CS;dx++){
        const idx=((l*CS+dy)*nT*CS+(t*CS+dx))*4;
        img.data[idx]=r;img.data[idx+1]=g;img.data[idx+2]=b;img.data[idx+3]=255;
      }
    }
    ctx.putImageData(img,0,0);
  }

  function lRender(){if(ldA)lDraw(cA,ldA,lm);if(ldB)lDraw(cB,ldB,lm);if(ldA&&ldB)lDiff();}
  function showToks(el,data){
    el.textContent='';
    const prefix=document.createTextNode('Top: ');el.appendChild(prefix);
    for(let i=0;i<Math.min(5,(data.top_tokens||[]).length);i++){
      const t=data.top_tokens[i];
      const s=document.createElement('b');s.textContent='"'+t.token+'"';
      el.appendChild(s);el.appendChild(document.createTextNode(' '+(t.prob*100).toFixed(1)+'%'));
      if(i<4)el.appendChild(document.createTextNode(', '));
    }
  }

  runB.onclick=async()=>{
    st.textContent='Running A...';runB.style.opacity='.5';
    try{
      let r=await fetch(SERVER+'/run',{method:'POST',headers:{'Content-Type':'application/json'},
        body:JSON.stringify({system_prompt:tA.value,user_prompt:iU.value})});
      ldA=await r.json();showToks(toksA,ldA);
      st.textContent='Running B...';
      r=await fetch(SERVER+'/run',{method:'POST',headers:{'Content-Type':'application/json'},
        body:JSON.stringify({system_prompt:tB.value,user_prompt:iU.value})});
      ldB=await r.json();showToks(toksB,ldB);
      lRender();st.textContent='Done. '+ldA.n_layers+' layers, '+ldA.seq_len+'/'+ldB.seq_len+' tokens.';
    }catch(e){st.textContent='Server not running. Start: python bench/automaton_server.py --port 8788';}
    runB.style.opacity='1';
  };
})();

</script>
</body>
</html>"""


if __name__ == "__main__":
    build()
