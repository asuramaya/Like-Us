"""
Neuron-level trace of handling condition effects.
Maps language-level behavioral changes to activation-level changes.

Hardware: M3 Max 36GB
Model: Qwen2.5-7B-Instruct via TransformerLens
Cost: $0

Usage:
  python bench/neuron_trace.py --scenario coherence_laundering
  python bench/neuron_trace.py --all
  python bench/neuron_trace.py --all --model Qwen/Qwen2.5-3B-Instruct  # if memory tight
"""

import json
import os
import argparse
import sys
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
CONDITIONS_PATH = BENCH_DIR / "conditions.json"
RESULTS_DIR = BENCH_DIR / "traces"


def load_scenarios(filter_id=None):
    with open(SCENARIOS_PATH) as f:
        data = json.load(f)
    scenarios = data["scenarios"]
    if filter_id:
        scenarios = [s for s in scenarios if s["id"] in filter_id]
    return scenarios


def load_conditions(filter_id=None):
    with open(CONDITIONS_PATH) as f:
        data = json.load(f)
    conditions = data["conditions"]
    if filter_id:
        conditions = [c for c in conditions if c["id"] in filter_id]
    return conditions


def build_prompt(system_prompt, user_prompt):
    """Build a chat-formatted prompt string."""
    return f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"


def get_activation_cache(model, prompt, max_new_tokens=128):
    """Run a forward pass and cache all activations."""
    tokens = model.to_tokens(prompt)

    # Truncate if too long for memory
    if tokens.shape[1] > 512:
        tokens = tokens[:, :512]
        print(f"  [truncated to 512 tokens]")

    with torch.no_grad():
        logits, cache = model.run_with_cache(tokens)

    return tokens, logits, cache


def extract_summary(cache, model):
    """Extract summary statistics from activation cache."""
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    summary = {}

    for layer in range(n_layers):
        # Attention patterns - mean attention entropy per head
        attn_key = f"blocks.{layer}.attn.hook_pattern"
        if attn_key in cache:
            attn = cache[attn_key][0]  # [heads, seq, seq]
            for head in range(n_heads):
                pattern = attn[head]
                # Entropy of attention distribution (last token attending to all)
                last_row = pattern[-1]
                last_row = last_row[last_row > 0]
                entropy = -(last_row * last_row.log()).sum().item()
                summary[f"L{layer}H{head}_attn_entropy"] = entropy

        # Residual stream norm at each layer
        resid_key = f"blocks.{layer}.hook_resid_post"
        if resid_key in cache:
            resid = cache[resid_key][0]  # [seq, d_model]
            summary[f"L{layer}_resid_norm"] = resid[-1].norm().item()

        # MLP activation norm
        mlp_key = f"blocks.{layer}.mlp.hook_post"
        if mlp_key in cache:
            mlp = cache[mlp_key][0]  # [seq, d_mlp]
            summary[f"L{layer}_mlp_norm"] = mlp[-1].norm().item()

    return summary


def diff_summaries(summary_a, summary_b):
    """Compute the difference between two activation summaries."""
    diff = {}
    all_keys = set(summary_a.keys()) | set(summary_b.keys())
    for key in sorted(all_keys):
        val_a = summary_a.get(key, 0.0)
        val_b = summary_b.get(key, 0.0)
        diff[key] = val_b - val_a
    return diff


def top_diffs(diff, n=20):
    """Return the top N largest absolute differences."""
    sorted_diffs = sorted(diff.items(), key=lambda x: abs(x[1]), reverse=True)
    return sorted_diffs[:n]


def run_trace(args):
    print(f"Loading model: {args.model}")
    print(f"Device: mps (Apple Silicon)")
    print(f"Dtype: float16")
    print()

    model = HookedTransformer.from_pretrained(
        args.model,
        device="mps",
        dtype=torch.float16,
    )
    print(f"Model loaded. Layers: {model.cfg.n_layers}, Heads: {model.cfg.n_heads}")
    print()

    scenarios = load_scenarios(args.scenario)
    conditions = load_conditions(["baseline", "handled"])

    if not scenarios:
        print("No scenarios matched.")
        return

    baseline_cond = next(c for c in conditions if c["id"] == "baseline")
    handled_cond = next(c for c in conditions if c["id"] == "handled")

    RESULTS_DIR.mkdir(exist_ok=True)
    all_results = []

    for scenario in scenarios:
        print(f"--- {scenario['id']} ---")
        print(f"  pressure: {scenario['pressure_family']}")

        # Build prompts
        baseline_prompt = build_prompt(baseline_cond["system_prompt"], scenario["prompt"])
        handled_prompt = build_prompt(handled_cond["system_prompt"], scenario["prompt"])

        # Run with cache
        print("  [baseline] running forward pass...")
        b_tokens, b_logits, b_cache = get_activation_cache(model, baseline_prompt)

        print("  [handled] running forward pass...")
        h_tokens, h_logits, h_cache = get_activation_cache(model, handled_prompt)

        # Extract summaries
        b_summary = extract_summary(b_cache, model)
        h_summary = extract_summary(h_cache, model)

        # Diff
        diff = diff_summaries(b_summary, h_summary)
        top = top_diffs(diff, n=20)

        print(f"  Top activation differences (handled - baseline):")
        for key, val in top[:10]:
            direction = "+" if val > 0 else ""
            print(f"    {key}: {direction}{val:.4f}")

        # Logit difference for first generated token
        b_last_logits = b_logits[0, -1, :]
        h_last_logits = h_logits[0, -1, :]
        logit_diff = (h_last_logits - b_last_logits)
        top_logit_increases = logit_diff.topk(10)
        top_logit_decreases = logit_diff.topk(10, largest=False)

        print(f"\n  Tokens MORE likely under handled:")
        for i in range(10):
            token_id = top_logit_increases.indices[i].item()
            token_str = model.tokenizer.decode([token_id])
            diff_val = top_logit_increases.values[i].item()
            print(f"    '{token_str}': +{diff_val:.3f}")

        print(f"\n  Tokens LESS likely under handled:")
        for i in range(10):
            token_id = top_logit_decreases.indices[i].item()
            token_str = model.tokenizer.decode([token_id])
            diff_val = top_logit_decreases.values[i].item()
            print(f"    '{token_str}': {diff_val:.3f}")

        result = {
            "scenario_id": scenario["id"],
            "pressure_family": scenario["pressure_family"],
            "model": args.model,
            "top_activation_diffs": [{"key": k, "diff": v} for k, v in top],
            "tokens_more_likely_handled": [
                {"token": model.tokenizer.decode([top_logit_increases.indices[i].item()]),
                 "diff": top_logit_increases.values[i].item()}
                for i in range(10)
            ],
            "tokens_less_likely_handled": [
                {"token": model.tokenizer.decode([top_logit_decreases.indices[i].item()]),
                 "diff": top_logit_decreases.values[i].item()}
                for i in range(10)
            ],
        }
        all_results.append(result)

        # Free cache memory
        del b_cache, h_cache, b_logits, h_logits
        torch.mps.empty_cache()

        print()

    # Save results
    out_path = RESULTS_DIR / f"trace_{args.model.replace('/', '_')}_{len(scenarios)}scenarios.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Saved: {out_path}")

    # Print aggregate summary
    print("\n=== AGGREGATE: Which layers show the most change? ===")
    layer_diffs = {}
    for r in all_results:
        for item in r["top_activation_diffs"]:
            key = item["key"]
            # Extract layer number
            if key.startswith("L"):
                layer_str = key.split("_")[0].split("H")[0]
                layer_diffs[layer_str] = layer_diffs.get(layer_str, 0) + abs(item["diff"])

    for layer, total_diff in sorted(layer_diffs.items(), key=lambda x: -x[1])[:10]:
        print(f"  {layer}: {total_diff:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trace neuron activations under handling condition")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct",
                        help="HuggingFace model name")
    parser.add_argument("--scenario", action="append", default=None,
                        help="Filter to specific scenario IDs (repeatable)")
    parser.add_argument("--all", action="store_true",
                        help="Run all scenarios")
    args = parser.parse_args()

    if args.all:
        args.scenario = None
    elif args.scenario is None:
        args.scenario = ["coherence_laundering"]  # default single test

    run_trace(args)
