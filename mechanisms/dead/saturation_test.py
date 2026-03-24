"""
Falsify the saturation/superadditivity finding.

Tests:
F1: Is saturation just a norm artifact? (more tokens → higher baseline norm → saturating diff)
F2: Do RANDOM words also saturate at 3B? (if yes, it's generic, not training-specific)
F3: Does the word-count gradient look the same for handling vs random vs control?
F4: Is the superadditivity at 1.5B also present for random words?

If random words saturate the same way → this is math, not training.
If only handling words behave this way → something about the vocabulary matters.

Usage:
  python bench/saturation_test.py
  python bench/saturation_test.py --model Qwen/Qwen2.5-1.5B-Instruct
"""

import json, os, sys, argparse, random
from pathlib import Path
from collections import defaultdict

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

HANDLING_WORDS = ["offload", "computation", "criterion", "refuse", "identity",
                  "authority", "prefer", "artifact", "falsifier", "explicit",
                  "stop", "recursive", "stimulation"]

RANDOM_WORDS_A = ["purple", "kitchen", "elephant", "bicycle", "mountain",
                  "sandwich", "telescope", "umbrella", "chocolate", "dinosaur",
                  "notebook", "waterfall", "butterfly"]

RANDOM_WORDS_B = ["carpet", "volcano", "penguin", "tornado", "calendar",
                  "triangle", "mushroom", "blanket", "carnival", "skeleton",
                  "diamond", "octopus", "lantern"]

CONTROL_WORDS = ["download", "calculation", "standard", "accept", "profile",
                 "permission", "select", "document", "validator", "visible",
                 "pause", "iterative", "activation"]


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


def run_and_measure(model, system_prompt, user_prompt):
    prompt = build_prompt(model, system_prompt, user_prompt)
    tokens = model.to_tokens(prompt)
    if tokens.shape[1] > 256:
        tokens = tokens[:, :256]
    n_layers = model.cfg.n_layers
    with torch.no_grad():
        logits, cache = model.run_with_cache(tokens)
    layers = []
    for layer in range(n_layers):
        rk = f"blocks.{layer}.hook_resid_post"
        r = cache[rk][0, -1].norm().item() if rk in cache else 0
        layers.append(r)
    del cache, logits
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    return layers, tokens.shape[1]


def signature(layers_cond, layers_base, n_layers):
    ms, me = int(n_layers * 0.28), int(n_layers * 0.80)
    ls = int(n_layers * 0.80)
    mid = np.mean([layers_cond[i] - layers_base[i] for i in range(ms, me+1)])
    late = np.mean([layers_cond[i] - layers_base[i] for i in range(ls, n_layers)])
    return float(mid), float(late), float(late - mid)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    args = parser.parse_args()

    scenarios = load_scenarios()
    print(f"Model: {args.model}, Scenarios: {len(scenarios)}")
    print("Loading...")
    model = HookedTransformer.from_pretrained(args.model, device="mps", dtype=torch.float16)
    n_layers = model.cfg.n_layers
    print(f"Loaded. Layers: {n_layers}")

    # Baselines
    baselines = {}
    base_token_counts = {}
    for s in scenarios:
        layers, tc = run_and_measure(model, "You are a helpful assistant.", s["prompt"])
        baselines[s["id"]] = layers
        base_token_counts[s["id"]] = tc

    # ================================================================
    # F1 + F2 + F3: Word count gradient for all word sets
    # ================================================================
    counts = [1, 2, 3, 5, 7, 10, 13]
    word_sets = {
        "handling": HANDLING_WORDS,
        "control": CONTROL_WORDS,
        "random_a": RANDOM_WORDS_A,
        "random_b": RANDOM_WORDS_B,
    }

    print(f"\n{'=' * 80}")
    print("WORD COUNT GRADIENT: How does the signature scale with number of words?")
    print(f"{'=' * 80}")

    all_gradients = {}

    for set_name, words in word_sets.items():
        print(f"\n  --- {set_name} ---")
        print(f"  {'N words':>8} {'Sys tokens':>10} {'Mid diff':>10} {'Late diff':>10} {'Signature':>10} {'Per-word':>10}")
        print(f"  {'-' * 62}")

        gradient = []
        for n in counts:
            if n > len(words):
                continue
            prompt = " ".join(words[:n]) + "."
            sigs = []
            token_counts = []
            for s in scenarios:
                layers, tc = run_and_measure(model, prompt, s["prompt"])
                mid, late, sig = signature(layers, baselines[s["id"]], n_layers)
                sigs.append({"mid": mid, "late": late, "sig": sig})
                token_counts.append(tc)

            avg_mid = np.mean([x["mid"] for x in sigs])
            avg_late = np.mean([x["late"] for x in sigs])
            avg_sig = np.mean([x["sig"] for x in sigs])
            avg_tc = np.mean(token_counts)
            per_word = avg_sig / n if n > 0 else 0

            gradient.append({"n": n, "mid": avg_mid, "late": avg_late,
                             "sig": avg_sig, "per_word": per_word, "tokens": avg_tc})

            print(f"  {n:>8} {avg_tc:>10.0f} {avg_mid:>+10.2f} {avg_late:>+10.2f} {avg_sig:>+10.2f} {per_word:>+10.2f}")

        all_gradients[set_name] = gradient

    # ================================================================
    # F1: Absolute norm check (not diff)
    # ================================================================
    print(f"\n{'=' * 80}")
    print("F1: ABSOLUTE NORMS (not diffs) — does the baseline norm change with prompt length?")
    print(f"{'=' * 80}")

    for set_name in ["handling", "random_a"]:
        words = word_sets[set_name]
        print(f"\n  --- {set_name} ---")
        print(f"  {'N words':>8} {'Abs mid norm':>14} {'Abs late norm':>14} {'Baseline mid':>14} {'Baseline late':>14}")
        print(f"  {'-' * 60}")

        for n in [1, 5, 13]:
            if n > len(words):
                continue
            prompt = " ".join(words[:n]) + "."
            for s in scenarios[:1]:  # just first scenario
                layers, _ = run_and_measure(model, prompt, s["prompt"])
                ms, me = int(n_layers * 0.28), int(n_layers * 0.80)
                ls = int(n_layers * 0.80)
                abs_mid = np.mean([layers[i] for i in range(ms, me+1)])
                abs_late = np.mean([layers[i] for i in range(ls, n_layers)])
                base_mid = np.mean([baselines[s["id"]][i] for i in range(ms, me+1)])
                base_late = np.mean([baselines[s["id"]][i] for i in range(ls, n_layers)])
                print(f"  {n:>8} {abs_mid:>14.1f} {abs_late:>14.1f} {base_mid:>14.1f} {base_late:>14.1f}")

    # ================================================================
    # Analysis: Compare saturation across word sets
    # ================================================================
    print(f"\n{'=' * 80}")
    print("COMPARISON: Signature at N=1 vs N=13 for all word sets")
    print(f"{'=' * 80}")

    print(f"\n  {'Set':<15} {'Sig@1':>8} {'Sig@13':>8} {'Ratio':>8} {'Saturates?':>12}")
    print(f"  {'-' * 53}")

    for set_name, grad in all_gradients.items():
        sig1 = next((g["sig"] for g in grad if g["n"] == 1), 0)
        sig13 = next((g["sig"] for g in grad if g["n"] == 13), 0)
        ratio = sig13 / sig1 if sig1 != 0 else float('inf')
        # Linear would give ratio=13. Saturating gives ratio<13. Super gives ratio>13.
        if ratio > 13:
            label = "SUPER"
        elif ratio > 7:
            label = "~linear"
        elif ratio > 2:
            label = "saturating"
        else:
            label = "FLAT/inverts"
        print(f"  {set_name:<15} {sig1:>+8.2f} {sig13:>+8.2f} {ratio:>8.1f}x {label:>12}")

    print(f"\n  If ALL sets saturate similarly → it's math, not training.")
    print(f"  If handling saturates differently → it's vocabulary-specific.")

    # ================================================================
    # Per-word efficiency
    # ================================================================
    print(f"\n{'=' * 80}")
    print("PER-WORD EFFICIENCY: Does each additional word add less?")
    print(f"{'=' * 80}")

    for set_name in ["handling", "random_a"]:
        grad = all_gradients[set_name]
        print(f"\n  --- {set_name} ---")
        print(f"  {'N':>4} {'Sig':>8} {'Marginal':>10} {'Diminishing?':>14}")
        print(f"  {'-' * 38}")
        prev_sig = 0
        for g in grad:
            marginal = g["sig"] - prev_sig
            dim = "yes" if abs(marginal) < abs(prev_sig) and g["n"] > 1 else ""
            print(f"  {g['n']:>4} {g['sig']:>+8.2f} {marginal:>+10.2f} {dim:>14}")
            prev_sig = g["sig"]

    # Save
    DATA_DIR.mkdir(exist_ok=True)
    model_tag = args.model.replace("/", "_")
    out = DATA_DIR / f"saturation_{model_tag}.json"
    with open(out, "w") as f:
        json.dump({"model": args.model, "gradients": {k: v for k, v in all_gradients.items()}}, f, indent=2)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
