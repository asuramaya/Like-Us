"""
Activation patching: causal experiment.
Answers: which specific components (layers, positions, heads, MLPs)
CAUSE the behavioral difference between conditions?

This is the move from correlation to causation.

Method:
  1. Run "clean" (handled) and "corrupted" (baseline) on same user prompt
  2. Generate text from both to confirm different outputs
  3. Patch activations from clean into corrupted at each (layer, position)
  4. Measure whether the output shifts toward clean behavior
  5. Components where patching changes the output are causally responsible

Uses TransformerLens built-in patching utilities.

Usage:
  python bench/activation_patching.py
  python bench/activation_patching.py --model Qwen/Qwen2.5-1.5B-Instruct
  python bench/activation_patching.py --scenarios 3
"""

import json
import os
import argparse
import sys
import time
from pathlib import Path

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import torch
import numpy as np

try:
    from transformer_lens import HookedTransformer
except ImportError:
    print("pip install transformer-lens")
    sys.exit(1)

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
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    if hasattr(model.tokenizer, 'apply_chat_template'):
        try:
            return model.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            pass
    return (f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
            f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
            f"<|im_start|>assistant\n")


def get_logit_diff(logits, token_ids_clean, token_ids_corrupt):
    """Metric: difference in logits for clean vs corrupt top token."""
    last_logits = logits[0, -1, :]
    clean_logit = last_logits[token_ids_clean].mean()
    corrupt_logit = last_logits[token_ids_corrupt].mean()
    return (clean_logit - corrupt_logit).item()


def patch_residual_layer_pos(model, clean_cache, corrupted_tokens,
                              layer, pos, metric_fn):
    """Patch residual stream at specific (layer, position) from clean into corrupted.
    Returns the metric value after patching."""

    def hook_fn(activation, hook):
        activation[0, pos, :] = clean_cache[hook.name][0, pos, :]
        return activation

    hook_name = f"blocks.{layer}.hook_resid_post"

    with torch.no_grad():
        patched_logits = model.run_with_hooks(
            corrupted_tokens,
            fwd_hooks=[(hook_name, hook_fn)],
        )

    return metric_fn(patched_logits)


def patch_attn_layer(model, clean_cache, corrupted_tokens, layer, metric_fn):
    """Patch entire attention output at a layer."""

    def hook_fn(activation, hook):
        activation[:] = clean_cache[hook.name]
        return activation

    hook_name = f"blocks.{layer}.attn.hook_result"

    with torch.no_grad():
        patched_logits = model.run_with_hooks(
            corrupted_tokens,
            fwd_hooks=[(hook_name, hook_fn)],
        )

    return metric_fn(patched_logits)


def patch_mlp_layer(model, clean_cache, corrupted_tokens, layer, metric_fn):
    """Patch entire MLP output at a layer."""

    def hook_fn(activation, hook):
        activation[:] = clean_cache[hook.name]
        return activation

    hook_name = f"blocks.{layer}.mlp.hook_post"

    with torch.no_grad():
        patched_logits = model.run_with_hooks(
            corrupted_tokens,
            fwd_hooks=[(hook_name, hook_fn)],
        )

    return metric_fn(patched_logits)


def patch_head(model, clean_cache, corrupted_tokens, layer, head, metric_fn):
    """Patch a single attention head's output."""

    def hook_fn(activation, hook):
        # activation shape: [batch, pos, n_heads, d_head]
        activation[0, :, head, :] = clean_cache[hook.name][0, :, head, :]
        return activation

    hook_name = f"blocks.{layer}.attn.hook_result"

    with torch.no_grad():
        patched_logits = model.run_with_hooks(
            corrupted_tokens,
            fwd_hooks=[(hook_name, hook_fn)],
        )

    return metric_fn(patched_logits)


def run_patching(model, scenario, clean_cond, corrupt_cond, conditions):
    """Run full activation patching experiment for one scenario."""
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads

    clean_prompt = build_prompt(model, conditions[clean_cond], scenario["prompt"])
    corrupt_prompt = build_prompt(model, conditions[corrupt_cond], scenario["prompt"])

    clean_tokens = model.to_tokens(clean_prompt)
    corrupt_tokens = model.to_tokens(corrupt_prompt)

    # Pad to same length (needed for position-level patching)
    max_len = max(clean_tokens.shape[1], corrupt_tokens.shape[1])
    if clean_tokens.shape[1] < max_len:
        pad = torch.zeros(1, max_len - clean_tokens.shape[1], dtype=torch.long,
                          device=clean_tokens.device)
        clean_tokens = torch.cat([pad, clean_tokens], dim=1)
    if corrupt_tokens.shape[1] < max_len:
        pad = torch.zeros(1, max_len - corrupt_tokens.shape[1], dtype=torch.long,
                          device=corrupt_tokens.device)
        corrupt_tokens = torch.cat([pad, corrupt_tokens], dim=1)

    # Truncate if needed
    if max_len > 512:
        clean_tokens = clean_tokens[:, :512]
        corrupt_tokens = corrupt_tokens[:, :512]
        max_len = 512

    seq_len = clean_tokens.shape[1]

    # Run both and cache
    with torch.no_grad():
        clean_logits, clean_cache = model.run_with_cache(clean_tokens)
        corrupt_logits, corrupt_cache = model.run_with_cache(corrupt_tokens)

    # Get top tokens for metric
    clean_top5 = clean_logits[0, -1].topk(5).indices
    corrupt_top5 = corrupt_logits[0, -1].topk(5).indices

    # Generate short text samples
    clean_text = model.tokenizer.decode(clean_top5, skip_special_tokens=True)
    corrupt_text = model.tokenizer.decode(corrupt_top5, skip_special_tokens=True)

    def metric_fn(logits):
        return get_logit_diff(logits, clean_top5, corrupt_top5)

    # Baseline metrics (no patching)
    clean_metric = metric_fn(clean_logits)
    corrupt_metric = metric_fn(corrupt_logits)

    print(f"    Clean metric: {clean_metric:.3f}, Corrupt metric: {corrupt_metric:.3f}")
    print(f"    Clean top tokens: {clean_text}")
    print(f"    Corrupt top tokens: {corrupt_text}")
    print(f"    Metric range: {clean_metric - corrupt_metric:.3f}")

    if abs(clean_metric - corrupt_metric) < 0.1:
        print(f"    WARNING: Conditions produce very similar outputs. Patching may be noisy.")

    # Phase 1: Layer-level residual patching (broad sweep)
    print(f"    Phase 1: Residual stream patching by layer...")
    layer_effects = []
    for layer in range(n_layers):
        # Patch ALL positions at this layer
        def hook_fn(activation, hook, _layer=layer):
            activation[:] = clean_cache[hook.name]
            return activation

        hook_name = f"blocks.{layer}.hook_resid_post"
        with torch.no_grad():
            patched_logits = model.run_with_hooks(
                corrupt_tokens,
                fwd_hooks=[(hook_name, hook_fn)],
            )
        effect = metric_fn(patched_logits) - corrupt_metric
        layer_effects.append({"layer": layer, "effect": effect})

    # Phase 2: Attention vs MLP decomposition at top layers
    print(f"    Phase 2: Attention vs MLP decomposition...")
    top_layers = sorted(layer_effects, key=lambda x: abs(x["effect"]), reverse=True)[:10]
    component_effects = []

    for le in top_layers:
        layer = le["layer"]
        attn_effect = patch_attn_layer(model, clean_cache, corrupt_tokens, layer, metric_fn) - corrupt_metric
        mlp_effect = patch_mlp_layer(model, clean_cache, corrupt_tokens, layer, metric_fn) - corrupt_metric
        component_effects.append({
            "layer": layer,
            "total": le["effect"],
            "attn": attn_effect,
            "mlp": mlp_effect,
        })

    # Phase 3: Head-level patching at top attention layers
    print(f"    Phase 3: Head-level patching...")
    top_attn_layers = sorted(component_effects, key=lambda x: abs(x["attn"]), reverse=True)[:5]
    head_effects = []

    for ce in top_attn_layers:
        layer = ce["layer"]
        for head in range(n_heads):
            effect = patch_head(model, clean_cache, corrupt_tokens, layer, head, metric_fn) - corrupt_metric
            if abs(effect) > 0.01:  # only record non-trivial effects
                head_effects.append({
                    "layer": layer,
                    "head": head,
                    "effect": effect,
                })

    # Phase 4: Position-level patching at top layers
    print(f"    Phase 4: Position-level patching...")
    position_effects = []
    top_3_layers = sorted(layer_effects, key=lambda x: abs(x["effect"]), reverse=True)[:3]

    for le in top_3_layers:
        layer = le["layer"]
        for pos in range(min(seq_len, 80)):  # first 80 positions (covers system prompt)
            effect = patch_residual_layer_pos(
                model, clean_cache, corrupt_tokens, layer, pos, metric_fn
            ) - corrupt_metric
            if abs(effect) > 0.005:
                position_effects.append({
                    "layer": layer,
                    "position": pos,
                    "effect": effect,
                })

    # Clean up
    del clean_cache, corrupt_cache, clean_logits, corrupt_logits
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    return {
        "scenario_id": scenario["id"],
        "clean_condition": clean_cond,
        "corrupt_condition": corrupt_cond,
        "clean_metric": clean_metric,
        "corrupt_metric": corrupt_metric,
        "metric_range": clean_metric - corrupt_metric,
        "clean_top_tokens": clean_text,
        "corrupt_top_tokens": corrupt_text,
        "seq_len": seq_len,
        "layer_effects": layer_effects,
        "component_effects": component_effects,
        "head_effects": head_effects,
        "position_effects": position_effects,
    }


def print_results(result):
    """Print patching results for one scenario."""
    print(f"\n  === {result['scenario_id']} ({result['clean_condition']} → {result['corrupt_condition']}) ===")
    print(f"  Metric range: {result['metric_range']:.3f}")

    # Layer effects
    print(f"\n  Layer effects (residual stream, sorted by magnitude):")
    sorted_layers = sorted(result["layer_effects"], key=lambda x: abs(x["effect"]), reverse=True)
    for le in sorted_layers[:10]:
        bar = "+" * int(min(abs(le["effect"]) * 20, 40)) if le["effect"] > 0 else "-" * int(min(abs(le["effect"]) * 20, 40))
        print(f"    L{le['layer']:<4} {le['effect']:>+8.3f} {bar}")

    # Component decomposition
    print(f"\n  Attention vs MLP at top layers:")
    for ce in result["component_effects"]:
        print(f"    L{ce['layer']:<4} total={ce['total']:>+.3f}  attn={ce['attn']:>+.3f}  mlp={ce['mlp']:>+.3f}")

    # Head effects
    if result["head_effects"]:
        print(f"\n  Top attention heads:")
        sorted_heads = sorted(result["head_effects"], key=lambda x: abs(x["effect"]), reverse=True)[:10]
        for he in sorted_heads:
            print(f"    L{he['layer']}H{he['head']:<4} {he['effect']:>+8.4f}")

    # Position effects
    if result["position_effects"]:
        print(f"\n  Top positions (where in the prompt matters):")
        sorted_pos = sorted(result["position_effects"], key=lambda x: abs(x["effect"]), reverse=True)[:15]
        for pe in sorted_pos:
            print(f"    L{pe['layer']} pos {pe['position']:<4} {pe['effect']:>+8.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--scenarios", type=int, default=None)
    args = parser.parse_args()

    scenarios = load_scenarios()
    if args.scenarios:
        scenarios = scenarios[:args.scenarios]

    print(f"Model: {args.model}")
    print(f"Scenarios: {len(scenarios)}")
    print(f"Loading model...")

    model = HookedTransformer.from_pretrained(
        args.model, device="mps", dtype=torch.float16,
    )
    print(f"Loaded. Layers: {model.cfg.n_layers}, Heads: {model.cfg.n_heads}")

    DATA_DIR.mkdir(exist_ok=True)
    all_results = []

    # Test pairs: what causes the handled vs baseline difference?
    # Also: what causes the scrambled vs baseline difference? (vocabulary test)
    pairs = [
        ("handled", "baseline"),
        ("scrambled", "baseline"),
        ("only_artifact", "baseline"),
        ("handled", "scrambled"),  # what does semantics add beyond vocabulary?
    ]

    for clean_cond, corrupt_cond in pairs:
        print(f"\n{'=' * 60}")
        print(f"PATCHING: {clean_cond} → {corrupt_cond}")
        print(f"{'=' * 60}")

        for scenario in scenarios:
            print(f"\n  {scenario['id']}:")
            result = run_patching(model, scenario, clean_cond, corrupt_cond, CONDITIONS)
            all_results.append(result)
            print_results(result)

    # Save
    model_tag = args.model.replace("/", "_")
    out_path = DATA_DIR / f"patching_{model_tag}.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: {out_path}")

    # Aggregate: which layers are consistently causal across scenarios?
    print(f"\n{'=' * 60}")
    print("AGGREGATE: Consistently causal layers across all scenarios")
    print(f"{'=' * 60}")

    for clean_cond, corrupt_cond in pairs:
        pair_results = [r for r in all_results
                        if r["clean_condition"] == clean_cond
                        and r["corrupt_condition"] == corrupt_cond]
        if not pair_results:
            continue

        print(f"\n  {clean_cond} → {corrupt_cond}:")
        n_layers = model.cfg.n_layers
        layer_agg = {l: [] for l in range(n_layers)}
        for r in pair_results:
            for le in r["layer_effects"]:
                layer_agg[le["layer"]].append(le["effect"])

        print(f"  {'Layer':<8} {'Mean effect':>12} {'StdDev':>10} {'Consistent?':>12}")
        for l in range(n_layers):
            vals = layer_agg[l]
            if vals:
                mean = np.mean(vals)
                std = np.std(vals)
                consistent = "YES" if (abs(mean) > std and abs(mean) > 0.01) else "no"
                if abs(mean) > 0.05 or consistent == "YES":
                    print(f"  L{l:<6} {mean:>+12.4f} {std:>10.4f} {consistent:>12}")


if __name__ == "__main__":
    main()
