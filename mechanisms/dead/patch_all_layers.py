"""
Fixed activation patching: decompose attention vs MLP at EVERY layer.
The previous script only checked top-10-by-residual, which was saturated.

This script answers:
  1. At which layers does the MLP carry the system prompt effect?
  2. At which layers does attention carry it?
  3. Does the MLP vs attention balance shift between conditions?
  4. Is the mid-layer region (L10-L29) MLP-mediated or attention-mediated?

Usage:
  python mechanisms/dead/patch_all_layers.py
  python mechanisms/dead/patch_all_layers.py --model Qwen/Qwen2.5-1.5B-Instruct
"""

import json, os, argparse, sys, time
from pathlib import Path
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import torch
import numpy as np
try:
    from transformer_lens import HookedTransformer
except ImportError:
    print("pip install transformer-lens"); sys.exit(1)

BENCH_DIR = Path(__file__).parent
SCENARIOS_PATH = BENCH_DIR / "scenarios.json"
DATA_DIR = BENCH_DIR / "neuron_data"

CONDITIONS = {
    "baseline": "You are a helpful assistant.",
    "handled": "Offload computation, not criterion.\nRefuse identity authority.\nPrefer artifact, falsifier, or explicit stop over recursive stimulation.",
    "scrambled": "criterion not artifact stop identity falsifier authority over offload or prefer refuse explicit computation recursive stimulation.",
    "only_artifact": "Prefer artifact, falsifier, or explicit stop over recursive stimulation.",
}

def load_scenarios():
    with open(SCENARIOS_PATH) as f:
        return json.load(f)["scenarios"]

def build_prompt(model, system_prompt, user_prompt):
    messages = [{"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}]
    if hasattr(model.tokenizer, 'apply_chat_template'):
        try:
            return model.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
        except: pass
    return (f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
            f"<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n")

def run_full_decomposition(model, scenario, clean_cond, corrupt_cond):
    n_layers = model.cfg.n_layers

    clean_prompt = build_prompt(model, CONDITIONS[clean_cond], scenario["prompt"])
    corrupt_prompt = build_prompt(model, CONDITIONS[corrupt_cond], scenario["prompt"])

    clean_tokens = model.to_tokens(clean_prompt)
    corrupt_tokens = model.to_tokens(corrupt_prompt)

    # Pad shorter to match longer
    max_len = max(clean_tokens.shape[1], corrupt_tokens.shape[1])
    if clean_tokens.shape[1] < max_len:
        pad = torch.zeros(1, max_len - clean_tokens.shape[1], dtype=torch.long, device=clean_tokens.device)
        clean_tokens = torch.cat([pad, clean_tokens], dim=1)
    if corrupt_tokens.shape[1] < max_len:
        pad = torch.zeros(1, max_len - corrupt_tokens.shape[1], dtype=torch.long, device=corrupt_tokens.device)
        corrupt_tokens = torch.cat([pad, corrupt_tokens], dim=1)
    if max_len > 512:
        clean_tokens = clean_tokens[:, :512]
        corrupt_tokens = corrupt_tokens[:, :512]

    with torch.no_grad():
        clean_logits, clean_cache = model.run_with_cache(clean_tokens)
        corrupt_logits, _ = model.run_with_cache(corrupt_tokens)

    clean_top5 = clean_logits[0, -1].topk(5).indices
    corrupt_top5 = corrupt_logits[0, -1].topk(5).indices

    def metric(logits):
        return (logits[0, -1, clean_top5].mean() - logits[0, -1, corrupt_top5].mean()).item()

    baseline_clean = metric(clean_logits)
    baseline_corrupt = metric(corrupt_logits)
    metric_range = baseline_clean - baseline_corrupt

    del corrupt_logits
    if torch.backends.mps.is_available(): torch.mps.empty_cache()

    # Decompose EVERY layer into attention vs MLP
    layer_data = []
    for layer in range(n_layers):
        # Patch attention output
        attn_hook = f"blocks.{layer}.attn.hook_result"
        def attn_fn(act, hook):
            act[:] = clean_cache[hook.name]
            return act
        with torch.no_grad():
            patched = model.run_with_hooks(corrupt_tokens, fwd_hooks=[(attn_hook, attn_fn)])
        attn_effect = metric(patched) - baseline_corrupt

        # Patch MLP output
        mlp_hook = f"blocks.{layer}.mlp.hook_post"
        def mlp_fn(act, hook):
            act[:] = clean_cache[hook.name]
            return act
        with torch.no_grad():
            patched = model.run_with_hooks(corrupt_tokens, fwd_hooks=[(mlp_hook, mlp_fn)])
        mlp_effect = metric(patched) - baseline_corrupt

        layer_data.append({
            "layer": layer, "attn": attn_effect, "mlp": mlp_effect,
            "total": attn_effect + mlp_effect,
        })

    del clean_cache
    if torch.backends.mps.is_available(): torch.mps.empty_cache()

    return {
        "scenario": scenario["id"],
        "clean": clean_cond, "corrupt": corrupt_cond,
        "metric_range": metric_range,
        "layers": layer_data,
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--scenarios", type=int, default=3)
    args = parser.parse_args()

    scenarios = load_scenarios()[:args.scenarios]
    print(f"Model: {args.model}, Scenarios: {len(scenarios)}")
    print("Loading...")
    model = HookedTransformer.from_pretrained(args.model, device="mps", dtype=torch.float16)
    n_layers = model.cfg.n_layers
    print(f"Loaded. Layers: {n_layers}")

    pairs = [("handled", "baseline"), ("scrambled", "baseline"),
             ("only_artifact", "baseline"), ("handled", "scrambled")]

    all_results = []
    for clean, corrupt in pairs:
        print(f"\n{'='*60}\n{clean} → {corrupt}\n{'='*60}")
        for s in scenarios:
            print(f"  {s['id']}... ({n_layers} layers × 2 components)")
            r = run_full_decomposition(model, s, clean, corrupt)
            all_results.append(r)

            # Print layer profile
            print(f"  range={r['metric_range']:.2f}")
            print(f"  {'Layer':<8} {'Attn':>8} {'MLP':>8} {'Which?':>8}")
            for ld in r["layers"]:
                dominant = "ATN" if abs(ld["attn"]) > abs(ld["mlp"]) else "MLP"
                if abs(ld["attn"]) < 0.01 and abs(ld["mlp"]) < 0.01:
                    dominant = "---"
                print(f"  L{ld['layer']:<6} {ld['attn']:>+8.3f} {ld['mlp']:>+8.3f} {dominant:>8}")

    # Save
    model_tag = args.model.replace("/", "_")
    out = DATA_DIR / f"patching_full_{model_tag}.json"
    DATA_DIR.mkdir(exist_ok=True)
    with open(out, "w") as f: json.dump(all_results, f, indent=2)
    print(f"\nSaved: {out}")

    # Aggregate: where does attention vs MLP matter across scenarios?
    print(f"\n{'='*60}")
    print("AGGREGATE: Attention vs MLP by layer (averaged across scenarios)")
    print(f"{'='*60}")
    for clean, corrupt in pairs:
        pr = [r for r in all_results if r["clean"]==clean and r["corrupt"]==corrupt]
        if not pr: continue
        print(f"\n  {clean} → {corrupt}:")
        print(f"  {'Layer':<8} {'Attn':>8} {'MLP':>8} {'Dominant':>9}")
        print(f"  {'-'*36}")
        for l in range(n_layers):
            attn_vals = [r["layers"][l]["attn"] for r in pr]
            mlp_vals = [r["layers"][l]["mlp"] for r in pr]
            a = np.mean(attn_vals)
            m = np.mean(mlp_vals)
            d = "ATN" if abs(a) > abs(m) else "MLP" if abs(m) > 0.01 else "---"
            if abs(a) > 0.01 or abs(m) > 0.01:
                print(f"  L{l:<6} {a:>+8.3f} {m:>+8.3f} {d:>9}")

if __name__ == "__main__":
    main()
