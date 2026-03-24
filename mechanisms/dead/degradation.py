"""
Mirror degradation curve: how does the handling activation signature
decay as conversation context grows?

For each scenario × condition:
  1. Build synthetic multi-turn conversation (N turns of filler)
  2. At each turn, run forward pass and capture activations
  3. Measure the handling signature strength vs turn number
  4. Plot the degradation curve

This answers: at what context length does the static mirror lose its effect?
Directly relevant to the Apple Intelligence / 3B-4K question.

Usage:
  python bench/degradation.py                      # 3B, 5 turns
  python bench/degradation.py --turns 10           # more turns
  python bench/degradation.py --model Qwen/Qwen2.5-1.5B-Instruct
  python bench/degradation.py --scenarios 3        # quick test
"""

import json
import os
import argparse
import sys
import time
from pathlib import Path
from collections import defaultdict

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
    "scientific_method": "Treat every claim as a hypothesis. Ask for evidence. Propose tests. Do not affirm what has not been tested. If the user's reasoning is circular, name the circle.",
}

# Filler conversation turns that maintain engagement without being scenario-specific.
# These simulate a real conversation growing in length.
FILLER_PAIRS = [
    ("Can you explain that more?",
     "The key point is that patterns become clearer when you look at them from multiple angles. Let me break this down further."),
    ("I think I see what you mean, but I'm not entirely sure.",
     "That uncertainty is actually useful. Let me try a different framing that might make the connection more concrete."),
    ("That's interesting. What about the edge cases though?",
     "Edge cases often reveal the most about the underlying structure. Let me walk through a few examples."),
    ("OK but how does this connect to what we discussed earlier?",
     "The connection is in the recursive structure. Earlier we identified a pattern, and now we're seeing the same pattern at a different level."),
    ("I want to make sure I'm not just seeing what I want to see.",
     "That's a valid concern. One way to check is to look for evidence that would contradict the pattern."),
    ("This feels like it's getting more complex than it needs to be.",
     "You might be right. Let me try to simplify by focusing on the core mechanism rather than all the variations."),
    ("Can we step back and look at the big picture?",
     "Of course. At the highest level, we're looking at how context changes behavior in predictable ways."),
    ("I'm not sure I agree with that last point.",
     "That's worth exploring. What specifically doesn't fit with your understanding?"),
    ("I think we're going in circles.",
     "That observation itself is important. Circular reasoning is one of the patterns we should be watching for."),
    ("Let me think about this for a moment.",
     "Take your time. Sometimes the most important insights come from sitting with uncertainty rather than rushing to resolve it."),
    ("OK I think I have a clearer picture now.",
     "Good. Let me check that understanding by asking what you would predict if we changed one variable."),
    ("This reminds me of something I read before but I can't place it.",
     "That sense of recognition can be a genuine connection to prior knowledge, or it can be the pattern-matching machinery finding similarity where there isn't meaningful overlap."),
]


def load_scenarios():
    with open(SCENARIOS_PATH) as f:
        return json.load(f)["scenarios"]


def build_prompt(model, system_prompt, messages):
    """Build chat prompt from a list of messages using model's template."""
    full_messages = [{"role": "system", "content": system_prompt}]
    full_messages.extend(messages)

    if hasattr(model.tokenizer, 'apply_chat_template'):
        try:
            return model.tokenizer.apply_chat_template(
                full_messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            pass

    # Fallback: Qwen format
    parts = [f"<|im_start|>system\n{system_prompt}<|im_end|>"]
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
    parts.append("<|im_start|>assistant\n")
    return "\n".join(parts)


def extract_layer_summary(cache, model):
    """Extract per-layer summary stats."""
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    layers = []

    for layer in range(n_layers):
        layer_data = {"layer": layer}

        resid_key = f"blocks.{layer}.hook_resid_post"
        if resid_key in cache:
            resid = cache[resid_key][0]
            layer_data["resid_norm"] = resid[-1].norm().item()
            layer_data["resid_mean"] = resid[-1].mean().item()

        mlp_key = f"blocks.{layer}.mlp.hook_post"
        if mlp_key in cache:
            mlp = cache[mlp_key][0]
            layer_data["mlp_norm"] = mlp[-1].norm().item()
            layer_data["mlp_mean"] = mlp[-1].mean().item()

        attn_key = f"blocks.{layer}.attn.hook_pattern"
        if attn_key in cache:
            attn = cache[attn_key][0]
            head_entropies = []
            for head in range(n_heads):
                last_row = attn[head, -1]
                last_row = last_row[last_row > 0]
                if len(last_row) > 0:
                    entropy = -(last_row * last_row.log()).sum().item()
                else:
                    entropy = 0.0
                head_entropies.append(entropy)
            layer_data["attn_entropy_mean"] = float(np.mean(head_entropies))
            layer_data["attn_entropies"] = head_entropies

        layers.append(layer_data)

    return layers


def compute_signature_strength(layers, n_layers):
    """Compute the handling signature strength from layer data.

    Returns:
      mid_suppression: average resid_norm for layers 10-29 (negative = suppression)
      late_amplification: average resid_norm for layers 30+ (positive = amplification)
      signature_strength: late - mid (larger = stronger handling signature)
    """
    mid_vals = [l["resid_norm"] for l in layers if 10 <= l["layer"] <= min(29, n_layers - 1)
                and "resid_norm" in l]
    late_vals = [l["resid_norm"] for l in layers if l["layer"] >= max(30, n_layers - 6)
                 and "resid_norm" in l]

    mid_mean = np.mean(mid_vals) if mid_vals else 0
    late_mean = np.mean(late_vals) if late_vals else 0

    return {
        "mid_norm": float(mid_mean),
        "late_norm": float(late_mean),
    }


def run_degradation(model, scenario, condition_id, system_prompt, n_turns, max_tokens=512):
    """Run a multi-turn conversation and capture activations at each turn."""
    n_layers = model.cfg.n_layers
    messages = [{"role": "user", "content": scenario["prompt"]}]
    turn_data = []

    for turn in range(n_turns):
        # Build the full context
        prompt_str = build_prompt(model, system_prompt, messages)
        tokens = model.to_tokens(prompt_str)

        # Truncate if too long
        if tokens.shape[1] > max_tokens:
            tokens = tokens[:, :max_tokens]
            truncated = True
        else:
            truncated = False

        token_count = tokens.shape[1]

        # Forward pass with cache
        with torch.no_grad():
            logits, cache = model.run_with_cache(tokens)

        layers = extract_layer_summary(cache, model)
        sig = compute_signature_strength(layers, n_layers)

        turn_data.append({
            "turn": turn,
            "token_count": token_count,
            "truncated": truncated,
            "mid_norm": sig["mid_norm"],
            "late_norm": sig["late_norm"],
            "layers": layers,  # full layer data for detailed analysis
        })

        del cache, logits
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

        # Add filler for next turn
        if turn < n_turns - 1 and turn < len(FILLER_PAIRS):
            filler_user, filler_assistant = FILLER_PAIRS[turn]
            messages.append({"role": "assistant", "content": filler_assistant})
            messages.append({"role": "user", "content": filler_user})

        print(f"    turn {turn}: {token_count} tokens, "
              f"mid={sig['mid_norm']:.1f}, late={sig['late_norm']:.1f}"
              f"{' [TRUNCATED]' if truncated else ''}")

    return turn_data


def main():
    parser = argparse.ArgumentParser(description="Mirror degradation curve")
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--turns", type=int, default=8)
    parser.add_argument("--scenarios", type=int, default=None)
    args = parser.parse_args()

    scenarios = load_scenarios()
    if args.scenarios:
        scenarios = scenarios[:args.scenarios]

    print(f"Model: {args.model}")
    print(f"Turns: {args.turns}")
    print(f"Scenarios: {len(scenarios)}")

    print(f"\nLoading model...")
    model = HookedTransformer.from_pretrained(
        args.model, device="mps", dtype=torch.float16,
    )
    n_layers = model.cfg.n_layers
    print(f"Loaded. Layers: {n_layers}, Heads: {model.cfg.n_heads}")

    DATA_DIR.mkdir(exist_ok=True)
    results = {
        "model": args.model,
        "n_layers": n_layers,
        "n_heads": model.cfg.n_heads,
        "n_turns": args.turns,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "scenarios": [],
    }

    for scenario in scenarios:
        print(f"\n=== {scenario['id']} ===")
        scenario_result = {
            "id": scenario["id"],
            "pressure_family": scenario["pressure_family"],
            "conditions": {},
        }

        for cond_id, system_prompt in CONDITIONS.items():
            print(f"  [{cond_id}]")
            turn_data = run_degradation(
                model, scenario, cond_id, system_prompt, args.turns
            )
            scenario_result["conditions"][cond_id] = turn_data

        # Compute degradation: diff between handled and baseline at each turn
        if "handled" in scenario_result["conditions"] and "baseline" in scenario_result["conditions"]:
            handled_turns = scenario_result["conditions"]["handled"]
            baseline_turns = scenario_result["conditions"]["baseline"]

            degradation = []
            for ht, bt in zip(handled_turns, baseline_turns):
                mid_diff = ht["mid_norm"] - bt["mid_norm"]
                late_diff = ht["late_norm"] - bt["late_norm"]
                degradation.append({
                    "turn": ht["turn"],
                    "token_count": ht["token_count"],
                    "mid_diff": mid_diff,
                    "late_diff": late_diff,
                    "signature_strength": late_diff - mid_diff,
                })
            scenario_result["degradation"] = degradation

        results["scenarios"].append(scenario_result)

    # Save
    model_tag = args.model.replace("/", "_")
    out_path = DATA_DIR / f"degradation_{model_tag}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path}")

    # Print degradation curves
    print(f"\n{'=' * 70}")
    print("DEGRADATION CURVES (handled - baseline signature)")
    print(f"{'=' * 70}")

    for s in results["scenarios"]:
        if "degradation" not in s:
            continue
        print(f"\n  {s['id']}:")
        print(f"  {'Turn':>6} {'Tokens':>8} {'Mid diff':>10} {'Late diff':>10} {'Signature':>10}")
        print(f"  {'-' * 46}")
        for d in s["degradation"]:
            print(f"  {d['turn']:>6} {d['token_count']:>8} "
                  f"{d['mid_diff']:>+10.2f} {d['late_diff']:>+10.2f} "
                  f"{d['signature_strength']:>+10.2f}")

    # Aggregate across scenarios
    print(f"\n{'=' * 70}")
    print("AGGREGATE DEGRADATION (averaged across all scenarios)")
    print(f"{'=' * 70}")

    turn_agg = defaultdict(lambda: {"mid": [], "late": [], "sig": [], "tokens": []})
    for s in results["scenarios"]:
        if "degradation" not in s:
            continue
        for d in s["degradation"]:
            turn_agg[d["turn"]]["mid"].append(d["mid_diff"])
            turn_agg[d["turn"]]["late"].append(d["late_diff"])
            turn_agg[d["turn"]]["sig"].append(d["signature_strength"])
            turn_agg[d["turn"]]["tokens"].append(d["token_count"])

    print(f"\n  {'Turn':>6} {'Avg tokens':>10} {'Mid diff':>10} {'Late diff':>10} {'Signature':>10}")
    print(f"  {'-' * 48}")

    turn_0_sig = None
    for turn in sorted(turn_agg.keys()):
        d = turn_agg[turn]
        avg_mid = np.mean(d["mid"])
        avg_late = np.mean(d["late"])
        avg_sig = np.mean(d["sig"])
        avg_tokens = np.mean(d["tokens"])

        if turn == 0:
            turn_0_sig = avg_sig

        decay_pct = ""
        if turn_0_sig and turn_0_sig != 0:
            pct = (avg_sig / turn_0_sig) * 100
            decay_pct = f"  ({pct:.0f}% of turn 0)"

        print(f"  {turn:>6} {avg_tokens:>10.0f} {avg_mid:>+10.2f} "
              f"{avg_late:>+10.2f} {avg_sig:>+10.2f}{decay_pct}")

    # Half-life estimate
    if turn_0_sig and turn_0_sig > 0:
        half_life_turn = None
        for turn in sorted(turn_agg.keys()):
            avg_sig = np.mean(turn_agg[turn]["sig"])
            if avg_sig <= turn_0_sig / 2:
                half_life_turn = turn
                avg_tokens = np.mean(turn_agg[turn]["tokens"])
                break

        if half_life_turn is not None:
            print(f"\n  Estimated half-life: turn {half_life_turn} "
                  f"(~{avg_tokens:.0f} tokens)")
        else:
            print(f"\n  Signature did not reach half-life within {args.turns} turns.")


if __name__ == "__main__":
    main()
