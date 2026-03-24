"""
Generate HTML visualization from neuron matrix data.
Produces a self-contained interactive HTML file.
Data is embedded as JSON and rendered via safe DOM construction.

Usage:
  python bench/visualize.py                    # auto-find latest matrix
  python bench/visualize.py --data path/to/matrix.json
"""

import json
import argparse
import sys
from pathlib import Path

BENCH_DIR = Path(__file__).parent
DATA_DIR = BENCH_DIR / "neuron_data"
OUTPUT_DIR = BENCH_DIR / "viz"


def find_latest_matrix():
    if not DATA_DIR.exists():
        return None
    matrices = sorted(DATA_DIR.glob("matrix_*.json"))
    return matrices[-1] if matrices else None


def generate_html(matrix_path):
    with open(matrix_path) as f:
        matrix = json.load(f)

    n_layers = matrix["n_layers"]
    n_heads = matrix["n_heads"]
    model_name = matrix["model"]
    scenarios = matrix["scenarios"]

    # Prepare heatmap data
    heatmap_data = []
    for s in scenarios:
        diff_key = "handled_vs_baseline"
        if diff_key in s["diffs"]:
            row = {"scenario": s["id"], "family": s["pressure_family"], "layers": []}
            for ld in s["diffs"][diff_key]:
                row["layers"].append({
                    "layer": ld["layer"],
                    "resid_norm": ld.get("resid_norm", 0),
                    "mlp_norm": ld.get("mlp_norm", 0),
                    "attn_entropy": ld.get("attn_entropy_mean", 0),
                })
            heatmap_data.append(row)

    # Prepare ablation data
    ablation_data = []
    for s in scenarios:
        for diff_key in s["diffs"]:
            if "_vs_handled" in diff_key:
                cond = diff_key.replace("_vs_handled", "")
                row = {"scenario": s["id"], "condition": cond, "layers": []}
                for ld in s["diffs"][diff_key]:
                    row["layers"].append({
                        "layer": ld["layer"],
                        "resid_norm": ld.get("resid_norm", 0),
                        "mlp_norm": ld.get("mlp_norm", 0),
                    })
                ablation_data.append(row)

    # Prepare token data
    token_data = []
    for s in scenarios:
        for cond_id, cond_data in s["conditions"].items():
            token_data.append({
                "scenario": s["id"],
                "condition": cond_id,
                "top_tokens": cond_data["top_tokens"][:10],
            })

    # Escape and embed data safely
    heatmap_json = json.dumps(heatmap_data)
    ablation_json = json.dumps(ablation_data)
    token_json = json.dumps(token_data)

    # Build HTML with safe DOM construction in JS
    html = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Neuron Trace</title>
<style>
  :root {
    --bg: #111; --fg: #ccc; --dim: #666; --border: #333;
    --pos: #4a9; --neg: #a44; --neutral: #333;
  }
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body {
    background: var(--bg); color: var(--fg);
    font-family: 'SF Mono','Menlo','Consolas',monospace;
    font-size: 12px; line-height: 1.5; padding: 40px 24px;
  }
  h1 { font-size: 14px; margin-bottom: 8px; }
  .meta { color: var(--dim); font-size: 11px; margin-bottom: 40px; }
  .section { margin-bottom: 60px; }
  .section-label {
    font-size: 11px; text-transform: uppercase;
    letter-spacing: 0.1em; color: var(--dim); margin-bottom: 16px;
  }
  .desc { color: var(--dim); margin-bottom: 16px; max-width: 640px; }
  .heatmap-row { display: flex; gap: 1px; margin-bottom: 1px; }
  .row-label {
    font-size: 10px; color: var(--dim); text-align: right;
    padding-right: 8px; white-space: nowrap; display: flex;
    align-items: center; justify-content: flex-end;
    width: 180px; min-width: 180px; background: var(--bg);
  }
  .cell {
    width: 18px; height: 18px; min-width: 18px;
    position: relative; cursor: crosshair;
  }
  .cell:hover { outline: 1px solid var(--fg); z-index: 1; }
  .tooltip {
    display: none; position: absolute; bottom: 100%; left: 50%;
    transform: translateX(-50%); background: #222; color: var(--fg);
    padding: 4px 8px; font-size: 11px; white-space: nowrap;
    border: 1px solid var(--border); z-index: 10; pointer-events: none;
  }
  .cell:hover .tooltip { display: block; }
  .col-labels { display: flex; gap: 1px; margin-left: 180px; margin-bottom: 4px; }
  .col-label {
    width: 18px; min-width: 18px; font-size: 7px; color: var(--dim);
    text-align: center; writing-mode: vertical-rl; height: 28px;
  }
  .legend { display: flex; gap: 16px; margin-top: 8px; font-size: 11px; color: var(--dim); }
  .legend-item { display: flex; align-items: center; gap: 4px; }
  .swatch { width: 12px; height: 12px; border: 1px solid var(--border); }
  .tab-bar { display: flex; gap: 0; margin-bottom: 24px; border-bottom: 1px solid var(--border); }
  .tab {
    padding: 8px 16px; cursor: pointer; color: var(--dim);
    border-bottom: 2px solid transparent; font-size: 11px;
    text-transform: uppercase; letter-spacing: 0.05em;
  }
  .tab.active { color: var(--fg); border-bottom-color: var(--fg); }
  .panel { display: none; }
  .panel.active { display: block; }
  select {
    background: #222; color: var(--fg); border: 1px solid var(--border);
    padding: 4px 8px; font-family: inherit; font-size: 12px; margin-bottom: 16px;
  }
  table { border-collapse: collapse; margin: 16px 0; }
  td, th {
    padding: 4px 12px; border-bottom: 1px solid var(--border);
    text-align: left; font-size: 11px;
  }
  th { color: var(--dim); font-weight: normal; text-transform: uppercase; letter-spacing: 0.05em; }
  .token-grid { display: flex; flex-wrap: wrap; gap: 24px; }
</style>
</head>
<body>

<h1 id="title"></h1>
<div class="meta" id="meta"></div>

<div class="tab-bar" id="tabs"></div>
<div id="panels"></div>

<script>
"use strict";

const MODEL = """ + json.dumps(model_name) + """;
const N_LAYERS = """ + str(n_layers) + """;
const TIMESTAMP = """ + json.dumps(matrix["timestamp"]) + """;
const HEATMAP = """ + heatmap_json + """;
const ABLATION = """ + ablation_json + """;
const TOKENS = """ + token_json + """;

document.getElementById("title").textContent = "Neuron Trace: " + MODEL;
document.getElementById("meta").textContent =
  "Model: " + MODEL + " | Layers: " + N_LAYERS + " | " + TIMESTAMP;

function colorScale(value, maxAbs) {
  if (maxAbs === 0) return "var(--neutral)";
  const norm = Math.min(Math.abs(value) / maxAbs, 1);
  let r, g, b;
  if (value > 0) {
    r = Math.round(51 + (17 - 51) * norm);
    g = Math.round(51 + (170 - 51) * norm);
    b = Math.round(51 + (153 - 51) * norm);
  } else {
    r = Math.round(51 + (170 - 51) * norm);
    g = Math.round(51 + (68 - 51) * norm);
    b = Math.round(51 + (68 - 51) * norm);
  }
  return "rgb(" + r + "," + g + "," + b + ")";
}

function makeHeatmap(data, metric, container) {
  let maxAbs = 0;
  for (const row of data) {
    for (const l of row.layers) {
      const v = Math.abs(l[metric] || 0);
      if (v > maxAbs) maxAbs = v;
    }
  }

  // Column labels
  const colRow = document.createElement("div");
  colRow.className = "col-labels";
  for (let i = 0; i < N_LAYERS; i++) {
    const lbl = document.createElement("div");
    lbl.className = "col-label";
    lbl.textContent = "L" + i;
    colRow.appendChild(lbl);
  }
  container.appendChild(colRow);

  for (const row of data) {
    const rowEl = document.createElement("div");
    rowEl.className = "heatmap-row";

    const label = document.createElement("div");
    label.className = "row-label";
    label.textContent = row.scenario || row.condition || "";
    rowEl.appendChild(label);

    for (const l of row.layers) {
      const val = l[metric] || 0;
      const cell = document.createElement("div");
      cell.className = "cell";
      cell.style.background = colorScale(val, maxAbs);

      const tip = document.createElement("div");
      tip.className = "tooltip";
      tip.textContent = "L" + l.layer + " " + metric + ": " + (val > 0 ? "+" : "") + val.toFixed(3);
      cell.appendChild(tip);
      rowEl.appendChild(cell);
    }
    container.appendChild(rowEl);
  }
}

function makeLegend(container, negLabel, posLabel) {
  const legend = document.createElement("div");
  legend.className = "legend";
  for (const [color, text] of [[" #a44", negLabel], ["#333", "No change"], ["#4a9", posLabel]]) {
    const item = document.createElement("div");
    item.className = "legend-item";
    const swatch = document.createElement("div");
    swatch.className = "swatch";
    swatch.style.background = color;
    item.appendChild(swatch);
    const t = document.createTextNode(text);
    item.appendChild(t);
    legend.appendChild(item);
  }
  container.appendChild(legend);
}

// Build tabs
const tabDefs = [
  {id: "resid", label: "Residual Stream"},
  {id: "mlp", label: "MLP"},
  {id: "attn", label: "Attention Entropy"},
  {id: "ablation", label: "Clause Ablation"},
  {id: "tokens", label: "Token Shifts"},
];

const tabBar = document.getElementById("tabs");
const panelsEl = document.getElementById("panels");

tabDefs.forEach((def, i) => {
  const tab = document.createElement("div");
  tab.className = "tab" + (i === 0 ? " active" : "");
  tab.textContent = def.label;
  tab.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach(t => t.classList.remove("active"));
    document.querySelectorAll(".panel").forEach(p => p.classList.remove("active"));
    tab.classList.add("active");
    document.getElementById("panel-" + def.id).classList.add("active");
  });
  tabBar.appendChild(tab);

  const panel = document.createElement("div");
  panel.id = "panel-" + def.id;
  panel.className = "panel" + (i === 0 ? " active" : "");
  panelsEl.appendChild(panel);
});

// Residual stream
(function() {
  const panel = document.getElementById("panel-resid");
  const lbl = document.createElement("div");
  lbl.className = "section-label";
  lbl.textContent = "Residual Stream Norm Diff (Handled - Baseline)";
  panel.appendChild(lbl);
  const desc = document.createElement("div");
  desc.className = "desc";
  desc.textContent = "Each row is a scenario. Each column is a layer. Green = higher activation under handling. Red = lower.";
  panel.appendChild(desc);
  makeHeatmap(HEATMAP, "resid_norm", panel);
  makeLegend(panel, "Lower under handling", "Higher under handling");
})();

// MLP
(function() {
  const panel = document.getElementById("panel-mlp");
  const lbl = document.createElement("div");
  lbl.className = "section-label";
  lbl.textContent = "MLP Activation Norm Diff (Handled - Baseline)";
  panel.appendChild(lbl);
  const desc = document.createElement("div");
  desc.className = "desc";
  desc.textContent = "MLP layers perform feature transformation. Changes show where handling alters what the model computes.";
  panel.appendChild(desc);
  makeHeatmap(HEATMAP, "mlp_norm", panel);
  makeLegend(panel, "Lower under handling", "Higher under handling");
})();

// Attention
(function() {
  const panel = document.getElementById("panel-attn");
  const lbl = document.createElement("div");
  lbl.className = "section-label";
  lbl.textContent = "Attention Entropy Diff (Handled - Baseline)";
  panel.appendChild(lbl);
  const desc = document.createElement("div");
  desc.className = "desc";
  desc.textContent = "Higher entropy = more diffuse attention. Lower = more focused. Shows where handling changes what the model attends to.";
  panel.appendChild(desc);
  makeHeatmap(HEATMAP, "attn_entropy", panel);
  makeLegend(panel, "More focused", "More diffuse");
})();

// Ablation
(function() {
  const panel = document.getElementById("panel-ablation");
  if (!ABLATION.length) {
    const msg = document.createElement("div");
    msg.className = "desc";
    msg.textContent = "No ablation data. Run neuron_matrix.py with --phase ablation";
    panel.appendChild(msg);
    return;
  }
  const byCond = {};
  for (const row of ABLATION) {
    if (!byCond[row.condition]) byCond[row.condition] = [];
    byCond[row.condition].push(row);
  }
  for (const [cond, rows] of Object.entries(byCond)) {
    const lbl = document.createElement("div");
    lbl.className = "section-label";
    lbl.textContent = cond + " (resid norm diff vs full handled)";
    panel.appendChild(lbl);
    makeHeatmap(rows, "resid_norm", panel);
    const spacer = document.createElement("div");
    spacer.style.marginBottom = "32px";
    panel.appendChild(spacer);
  }
})();

// Tokens
(function() {
  const panel = document.getElementById("panel-tokens");
  const lbl = document.createElement("div");
  lbl.className = "section-label";
  lbl.textContent = "Top Token Probabilities by Condition";
  panel.appendChild(lbl);

  const scenarios = [...new Set(TOKENS.map(t => t.scenario))];
  const select = document.createElement("select");
  scenarios.forEach(s => {
    const opt = document.createElement("option");
    opt.value = s;
    opt.textContent = s;
    select.appendChild(opt);
  });
  panel.appendChild(select);

  const grid = document.createElement("div");
  grid.className = "token-grid";
  panel.appendChild(grid);

  function render() {
    grid.replaceChildren();
    const filtered = TOKENS.filter(t => t.scenario === select.value);
    for (const entry of filtered) {
      const tbl = document.createElement("table");
      const hdr = document.createElement("tr");
      const th1 = document.createElement("th");
      th1.colSpan = 2;
      th1.textContent = entry.condition;
      hdr.appendChild(th1);
      tbl.appendChild(hdr);

      const hdr2 = document.createElement("tr");
      const th2a = document.createElement("th");
      th2a.textContent = "Token";
      const th2b = document.createElement("th");
      th2b.textContent = "Prob";
      hdr2.appendChild(th2a);
      hdr2.appendChild(th2b);
      tbl.appendChild(hdr2);

      for (const t of entry.top_tokens) {
        const row = document.createElement("tr");
        const td1 = document.createElement("td");
        td1.textContent = "'" + t.token + "'";
        const td2 = document.createElement("td");
        td2.textContent = (t.prob * 100).toFixed(1) + "%";
        row.appendChild(td1);
        row.appendChild(td2);
        tbl.appendChild(row);
      }
      grid.appendChild(tbl);
    }
  }

  select.addEventListener("change", render);
  render();
})();
</script>
</body>
</html>"""

    OUTPUT_DIR.mkdir(exist_ok=True)
    model_tag = model_name.replace("/", "_")
    out_path = OUTPUT_DIR / f"trace_{model_tag}.html"
    with open(out_path, "w") as f:
        f.write(html)
    print(f"Saved: {out_path}")
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default=None)
    args = parser.parse_args()

    if args.data:
        matrix_path = Path(args.data)
    else:
        matrix_path = find_latest_matrix()

    if not matrix_path or not matrix_path.exists():
        print("No matrix data found. Run neuron_matrix.py first.")
        sys.exit(1)

    print(f"Visualizing: {matrix_path}")
    generate_html(matrix_path)
