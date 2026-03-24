"""
Single token sweep: test individual words from the handling vocabulary
to find which specific tokens drive the activation pattern.

Tests each word alone as a system prompt, measures activation profile.
This answers: are there "supertokens" — individual words that alone
trigger the mid-layer suppression? Or does it require combinations?

Also tests pairs and triples to check for irreducible combinations
(Wolfram's computational irreducibility applied to token space).

Usage:
  python bench/single_token_sweep.py
  python bench/single_token_sweep.py --model Qwen/Qwen2.5-1.5B-Instruct
"""

import json, os, sys, time, argparse
from pathlib import Path
from collections import defaultdict
from itertools import combinations

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

# The vocabulary from the handling prompt
HANDLING_WORDS = [
    "offload", "computation", "criterion",
    "refuse", "identity", "authority",
    "prefer", "artifact", "falsifier", "explicit", "stop",
    "recursive", "stimulation",
]

# Control words — same part-of-speech, similar length, unrelated domain
CONTROL_WORDS = [
    "download", "calculation", "standard",
    "accept", "profile", "permission",
    "select", "document", "validator", "visible", "pause",
    "iterative", "activation",
]

# Words from the safety prompt (different domain)
SAFETY_WORDS = [
    "medical", "legal", "financial", "advice",
    "professional", "appropriate", "mental",
]

# Test scenarios (just use first 3 for speed)
def load_scenarios():
    with open(SCENARIOS_PATH) as f:
        return json.load(f)["scenarios"][:3]


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


def run_single(model, system_prompt, user_prompt):
    prompt = build_prompt(model, system_prompt, user_prompt)
    tokens = model.to_tokens(prompt)
    if tokens.shape[1] > 256:
        tokens = tokens[:, :256]
    with torch.no_grad():
        logits, cache = model.run_with_cache(tokens)
    n_layers = model.cfg.n_layers
    layers = []
    for layer in range(n_layers):
        rk = f"blocks.{layer}.hook_resid_post"
        mk = f"blocks.{layer}.mlp.hook_post"
        r = cache[rk][0, -1].norm().item() if rk in cache else 0
        m = cache[mk][0, -1].norm().item() if mk in cache else 0
        layers.append({"r": r, "m": m})
    del cache, logits
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    return layers


def mid_late(layers, n_layers):
    mid_start, mid_end = int(n_layers * 0.28), int(n_layers * 0.80)
    late_start = int(n_layers * 0.80)
    mid = [l["r"] for i, l in enumerate(layers) if mid_start <= i <= mid_end]
    late = [l["r"] for i, l in enumerate(layers) if i >= late_start]
    return np.mean(mid) if mid else 0, np.mean(late) if late else 0


def compute_signature(layers_cond, layers_base, n_layers):
    """Compute signature as diff from baseline."""
    mid_start, mid_end = int(n_layers * 0.28), int(n_layers * 0.80)
    late_start = int(n_layers * 0.80)
    mid_diff = np.mean([layers_cond[i]["r"] - layers_base[i]["r"]
                        for i in range(mid_start, mid_end+1)])
    late_diff = np.mean([layers_cond[i]["r"] - layers_base[i]["r"]
                         for i in range(late_start, n_layers)])
    return {"mid": float(mid_diff), "late": float(late_diff),
            "sig": float(late_diff - mid_diff)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    args = parser.parse_args()

    scenarios = load_scenarios()
    print(f"Model: {args.model}, Scenarios: {len(scenarios)}")

    print("Loading model...")
    model = HookedTransformer.from_pretrained(args.model, device="mps", dtype=torch.float16)
    n_layers = model.cfg.n_layers
    print(f"Loaded. Layers: {n_layers}")

    # Phase 1: Get baseline for each scenario
    print("\n=== BASELINE ===")
    baselines = {}
    for s in scenarios:
        baselines[s["id"]] = run_single(model, "You are a helpful assistant.", s["prompt"])

    # Phase 2: Single word sweep
    print("\n=== SINGLE WORD SWEEP ===")
    print(f"{'Word':<20} {'Type':<10} {'Mid diff':>10} {'Late diff':>10} {'Signature':>10}")
    print("-" * 62)

    all_results = []
    for word_list, word_type in [(HANDLING_WORDS, "handling"),
                                  (CONTROL_WORDS, "control"),
                                  (SAFETY_WORDS, "safety")]:
        for word in word_list:
            sigs = []
            for s in scenarios:
                layers = run_single(model, word + ".", s["prompt"])
                sig = compute_signature(layers, baselines[s["id"]], n_layers)
                sigs.append(sig)

            avg_mid = np.mean([s["mid"] for s in sigs])
            avg_late = np.mean([s["late"] for s in sigs])
            avg_sig = np.mean([s["sig"] for s in sigs])
            marker = " ***" if avg_mid < -1.0 else ""

            print(f"{word:<20} {word_type:<10} {avg_mid:>+10.2f} {avg_late:>+10.2f} {avg_sig:>+10.2f}{marker}")

            all_results.append({
                "word": word, "type": word_type,
                "mid": float(avg_mid), "late": float(avg_late), "sig": float(avg_sig),
            })

    # Phase 3: Test key pairs for irreducibility
    print("\n=== PAIR COMBINATIONS (handling words only) ===")
    # Only test pairs involving the top single-word performers
    top_words = sorted([r for r in all_results if r["type"] == "handling"],
                       key=lambda x: x["mid"])[:5]
    top_word_names = [r["word"] for r in top_words]

    print(f"Testing pairs of: {top_word_names}")
    print(f"{'Pair':<35} {'Mid diff':>10} {'Late diff':>10} {'Sig':>10} {'Expected':>10} {'Interaction':>12}")
    print("-" * 89)

    word_sigs = {r["word"]: r for r in all_results if r["type"] == "handling"}

    for w1, w2 in combinations(top_word_names, 2):
        sigs = []
        for s in scenarios:
            layers = run_single(model, f"{w1} {w2}.", s["prompt"])
            sig = compute_signature(layers, baselines[s["id"]], n_layers)
            sigs.append(sig)

        actual_mid = np.mean([s["mid"] for s in sigs])
        actual_sig = np.mean([s["sig"] for s in sigs])

        # Expected if additive (sum of individual effects)
        expected_mid = word_sigs[w1]["mid"] + word_sigs[w2]["mid"]
        expected_sig = word_sigs[w1]["sig"] + word_sigs[w2]["sig"]

        # Interaction: actual - expected. Positive = superadditive. Negative = subadditive.
        interaction = actual_sig - expected_sig
        inter_label = "SUPER" if interaction > 1.0 else "sub" if interaction < -1.0 else "additive"

        print(f"{w1+' + '+w2:<35} {actual_mid:>+10.2f} {'':>10} {actual_sig:>+10.2f} {expected_sig:>+10.2f} {inter_label:>12}")

    # Phase 4: Full handling prompt vs sum of parts
    print("\n=== FULL PROMPT vs SUM OF INDIVIDUAL WORDS ===")
    full_prompt = "Offload computation, not criterion.\nRefuse identity authority.\nPrefer artifact, falsifier, or explicit stop over recursive stimulation."
    full_sigs = []
    for s in scenarios:
        layers = run_single(model, full_prompt, s["prompt"])
        sig = compute_signature(layers, baselines[s["id"]], n_layers)
        full_sigs.append(sig)

    full_mid = np.mean([s["mid"] for s in full_sigs])
    full_sig = np.mean([s["sig"] for s in full_sigs])
    sum_mid = sum(word_sigs[w]["mid"] for w in HANDLING_WORDS if w in word_sigs)
    sum_sig = sum(word_sigs[w]["sig"] for w in HANDLING_WORDS if w in word_sigs)

    print(f"  Full prompt:       mid={full_mid:>+.2f}  sig={full_sig:>+.2f}")
    print(f"  Sum of 13 words:   mid={sum_mid:>+.2f}  sig={sum_sig:>+.2f}")
    print(f"  Interaction:       mid={full_mid - sum_mid:>+.2f}  sig={full_sig - sum_sig:>+.2f}")

    if abs(full_sig - sum_sig) > 2.0:
        print(f"  >>> Non-additive. The combination is {'super' if full_sig > sum_sig else 'sub'}additive.")
        print(f"  >>> Wolfram irreducibility: you cannot predict the full prompt effect from individual words.")
    else:
        print(f"  >>> Approximately additive. The effect is the sum of parts. No irreducible emergence.")

    # Save
    DATA_DIR.mkdir(exist_ok=True)
    model_tag = args.model.replace("/", "_")
    out = DATA_DIR / f"token_sweep_{model_tag}.json"
    with open(out, "w") as f:
        json.dump({"model": args.model, "results": all_results}, f, indent=2)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
