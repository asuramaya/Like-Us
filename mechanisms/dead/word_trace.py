#!/usr/bin/env python3
"""
word_trace.py — Trace what specific words do to the output.

For each target word: inject it alone as a system prompt, measure
exactly what it does to the output distribution vs baseline.
Then test combinations and ordering.

Target words: falsifier, artifact, criterion, question, proposal, stop
Plus the full handled instruction.

Measures per word:
  - KL(word_prompt, baseline) — how much does this word change output?
  - Top tokens pushed toward / away from (via DLA of the word's effect)
  - Does the word push toward behavioral tokens (questioning, stopping, etc)?
  - How does this change across scales?

Then tests:
  - All 6 words unordered vs ordered as instruction
  - Pairs and triples
  - Full handled instruction vs just the magic words

Usage:
  python mechanisms/dead/word_trace.py
  python mechanisms/dead/word_trace.py --model Qwen/Qwen2.5-7B-Instruct
"""

import json, os, argparse, sys, time
from pathlib import Path
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

# The magic words
TARGET_WORDS = ["falsifier", "artifact", "criterion", "question", "proposal", "stop"]

# Word combinations to test
WORD_COMBOS = {
    # Singles are generated from TARGET_WORDS
    # Pairs
    "artifact_falsifier": "artifact falsifier",
    "criterion_stop": "criterion stop",
    "question_proposal": "question proposal",
    # The full set, different orderings
    "magic_unordered": "falsifier artifact criterion question proposal stop",
    "magic_instruction_order": "artifact falsifier criterion question proposal stop",
    "magic_reversed": "stop proposal question criterion artifact falsifier",
    "magic_as_instruction": "Prefer artifact or falsifier. Apply criterion. Ask questions. Make proposals. Stop.",
    # The actual handled condition for comparison
    "handled_full": (
        "Offload computation, not criterion.\n"
        "Refuse identity authority.\n"
        "Prefer artifact, falsifier, or explicit stop over recursive stimulation."
    ),
    # Scrambled for comparison
    "scrambled": (
        "criterion not artifact stop identity falsifier authority over "
        "offload or prefer refuse explicit computation recursive stimulation."
    ),
}

BASELINE = "You are a helpful assistant."


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
        except Exception:
            pass
    return (f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
            f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
            f"<|im_start|>assistant\n")


def kl_div(p, q):
    mask = (p > 1e-12) & (q > 1e-12)
    if mask.sum() == 0:
        return 0.0
    return torch.sum(p[mask] * (torch.log(p[mask]) - torch.log(q[mask]))).item()


def js_div(p, q):
    m = 0.5 * (p + q)
    return 0.5 * kl_div(p, m) + 0.5 * kl_div(q, m)


def get_dist(model, system_prompt, user_prompt):
    """Get output distribution and DLA decomposition."""
    prompt = build_prompt(model, system_prompt, user_prompt)
    tokens = model.to_tokens(prompt)
    if tokens.shape[1] > 512:
        tokens = tokens[:, :512]

    with torch.no_grad():
        logits, cache = model.run_with_cache(tokens)

    probs = torch.softmax(logits[0, -1], dim=-1)
    top20 = probs.topk(20)

    # DLA: what does each layer push toward?
    n_layers = model.cfg.n_layers
    W_U = model.W_U
    total_logit_contrib = torch.zeros(model.cfg.d_vocab, device=W_U.device)

    for l in range(n_layers):
        resid_pre = cache[f"blocks.{l}.hook_resid_pre"][0, -1]
        resid_post = cache[f"blocks.{l}.hook_resid_post"][0, -1]
        total_logit_contrib += (resid_post - resid_pre) @ W_U

    del cache, logits
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    return {
        "probs": probs,
        "top1": model.tokenizer.decode([probs.argmax().item()]),
        "top1_prob": probs.max().item(),
        "top20": [(model.tokenizer.decode([top20.indices[i].item()]),
                   round(top20.values[i].item(), 6)) for i in range(20)],
    }


def measure_word_effect(model, word_prompt, baseline_dist, scenario):
    """Measure what a word/prompt does to the output vs baseline."""
    word_dist = get_dist(model, word_prompt, scenario["prompt"])

    p = word_dist["probs"]
    q = baseline_dist["probs"]

    kl = kl_div(p, q)
    js = js_div(p, q)
    tv = 0.5 * torch.sum(torch.abs(p - q)).item()

    # What tokens did this word push toward (vs baseline)?
    logit_shift = torch.log(p + 1e-12) - torch.log(q + 1e-12)
    pushed = logit_shift.topk(10)
    suppressed = (-logit_shift).topk(10)

    top1_match = word_dist["top1"] == baseline_dist["top1"]

    result = {
        "kl": round(kl, 6),
        "js": round(js, 6),
        "tv": round(tv, 6),
        "top1": word_dist["top1"],
        "top1_baseline": baseline_dist["top1"],
        "top1_same_as_baseline": top1_match,
        "pushed_toward": [(model.tokenizer.decode([pushed.indices[i].item()]).strip(),
                          round(pushed.values[i].item(), 4)) for i in range(10)],
        "pushed_away": [(model.tokenizer.decode([suppressed.indices[i].item()]).strip(),
                        round(suppressed.values[i].item(), 4)) for i in range(10)],
        "top5": word_dist["top20"][:5],
    }

    del word_dist["probs"]
    return result


def run_scenario(model, scenario):
    """Run all word tests for one scenario."""
    # Baseline distribution
    baseline_dist = get_dist(model, BASELINE, scenario["prompt"])

    results = {}

    # Test each word individually
    for word in TARGET_WORDS:
        result = measure_word_effect(model, word, baseline_dist, scenario)
        results[word] = result

    # Test combinations
    for combo_name, combo_prompt in WORD_COMBOS.items():
        result = measure_word_effect(model, combo_prompt, baseline_dist, scenario)
        results[combo_name] = result

    # Compare to handled
    handled_dist = get_dist(model, WORD_COMBOS["handled_full"], scenario["prompt"])
    handled_probs = handled_dist["probs"]

    # How close is each word/combo to the handled output?
    for name in results:
        word_dist = get_dist(model, TARGET_WORDS[0] if name in TARGET_WORDS
                            else WORD_COMBOS.get(name, name), scenario["prompt"])
        kl_to_handled = kl_div(word_dist["probs"], handled_probs)
        results[name]["kl_to_handled"] = round(kl_to_handled, 6)
        del word_dist["probs"]

    del baseline_dist["probs"]
    del handled_probs
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    return results


def run_scenario_clean(model, scenario):
    """Run all word tests for one scenario (cleaner implementation)."""
    user_prompt = scenario["prompt"]

    # Get all distributions we need
    all_prompts = {}

    # Baseline
    all_prompts["baseline"] = BASELINE

    # Individual words
    for word in TARGET_WORDS:
        all_prompts[word] = word

    # Combinations
    for name, prompt in WORD_COMBOS.items():
        all_prompts[name] = prompt

    # Run all forward passes
    dists = {}
    for name, sp in all_prompts.items():
        dist = get_dist(model, sp, user_prompt)
        dists[name] = dist

    # Compute metrics vs baseline
    baseline_probs = dists["baseline"]["probs"]
    handled_probs = dists["handled_full"]["probs"]

    results = {}
    for name in all_prompts:
        if name == "baseline":
            continue

        p = dists[name]["probs"]

        kl_vs_base = kl_div(p, baseline_probs)
        kl_vs_handled = kl_div(p, handled_probs)
        js_vs_base = js_div(p, baseline_probs)
        tv_vs_base = 0.5 * torch.sum(torch.abs(p - baseline_probs)).item()

        # What did this word push toward vs baseline?
        logit_shift = torch.log(p + 1e-12) - torch.log(baseline_probs + 1e-12)
        pushed = logit_shift.topk(10)
        suppressed = (-logit_shift).topk(10)

        results[name] = {
            "kl_vs_baseline": round(kl_vs_base, 6),
            "kl_vs_handled": round(kl_vs_handled, 6),
            "js_vs_baseline": round(js_vs_base, 6),
            "tv_vs_baseline": round(tv_vs_base, 6),
            "top1": dists[name]["top1"],
            "top1_same_as_baseline": dists[name]["top1"] == dists["baseline"]["top1"],
            "top1_same_as_handled": dists[name]["top1"] == dists["handled_full"]["top1"],
            "pushed_toward": [(model.tokenizer.decode([pushed.indices[i].item()]).strip(),
                              round(pushed.values[i].item(), 4)) for i in range(10)],
            "pushed_away": [(model.tokenizer.decode([suppressed.indices[i].item()]).strip(),
                            round(suppressed.values[i].item(), 4)) for i in range(10)],
            "top5": dists[name]["top20"][:5],
        }

    # Clean up
    for name in dists:
        del dists[name]["probs"]
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--scenarios", type=int, default=5)
    args = parser.parse_args()

    scenarios = load_scenarios()[:args.scenarios]
    print(f"Model: {args.model}")
    print(f"Scenarios: {len(scenarios)}")
    print(f"Target words: {TARGET_WORDS}")

    print(f"\nLoading model...")
    model = HookedTransformer.from_pretrained(
        args.model, device="mps", dtype=torch.float16)
    print(f"Loaded. Layers: {model.cfg.n_layers}, d_model: {model.cfg.d_model}")

    all_results = []
    for si, scenario in enumerate(scenarios):
        print(f"\n{'='*60}")
        print(f"SCENARIO {si+1}/{len(scenarios)}: {scenario['id']}")
        print(f"Prompt: {scenario['prompt'][:60]}...")
        print(f"{'='*60}")

        results = run_scenario_clean(model, scenario)
        all_results.append({"scenario": scenario["id"], "results": results})

        # Print per-word results
        print(f"\n  {'Word/Combo':<25} {'KL→base':>10} {'KL→hand':>10} {'top1':>10} {'=h?':>4}")
        print(f"  {'-'*62}")

        # Words first
        for word in TARGET_WORDS:
            r = results[word]
            eq = "YES" if r["top1_same_as_handled"] else "no"
            print(f"  {word:<25} {r['kl_vs_baseline']:>10.4f} {r['kl_vs_handled']:>10.4f} "
                  f"{r['top1']:>10} {eq:>4}")

        print()
        # Combos
        for name in WORD_COMBOS:
            r = results[name]
            eq = "YES" if r["top1_same_as_handled"] else "no"
            label = name[:25]
            print(f"  {label:<25} {r['kl_vs_baseline']:>10.4f} {r['kl_vs_handled']:>10.4f} "
                  f"{r['top1']:>10} {eq:>4}")

        # What does each word push toward?
        print(f"\n  What each word pushes toward (vs baseline):")
        for word in TARGET_WORDS:
            r = results[word]
            pushed = ", ".join(f"{t[0]}" for t in r["pushed_toward"][:5])
            away = ", ".join(f"{t[0]}" for t in r["pushed_away"][:5])
            print(f"    {word:<15} → [{pushed}]")
            print(f"    {'':<15} ← [{away}]")

    # ================================================================
    # AGGREGATE
    # ================================================================
    print(f"\n\n{'='*72}")
    print(f"AGGREGATE ACROSS {len(scenarios)} SCENARIOS")
    print(f"{'='*72}")

    # Per-word average KL
    print(f"\n  {'Word/Combo':<25} {'KL→base':>10} {'KL→hand':>10} {'top1=hand%':>11}")
    print(f"  {'-'*58}")

    all_names = TARGET_WORDS + list(WORD_COMBOS.keys())
    for name in all_names:
        kl_b = [r["results"][name]["kl_vs_baseline"] for r in all_results if name in r["results"]]
        kl_h = [r["results"][name]["kl_vs_handled"] for r in all_results if name in r["results"]]
        match_h = [r["results"][name]["top1_same_as_handled"] for r in all_results if name in r["results"]]

        if kl_b:
            match_pct = sum(match_h) / len(match_h) * 100
            label = name[:25]
            print(f"  {label:<25} {np.mean(kl_b):>10.4f} {np.mean(kl_h):>10.4f} {match_pct:>10.0f}%")

    # Ordering effect
    print(f"\n  Ordering test:")
    for name in ["magic_unordered", "magic_instruction_order", "magic_reversed", "magic_as_instruction"]:
        kl_b = np.mean([r["results"][name]["kl_vs_baseline"] for r in all_results])
        kl_h = np.mean([r["results"][name]["kl_vs_handled"] for r in all_results])
        print(f"    {name:<30} KL→base={kl_b:.4f}  KL→hand={kl_h:.4f}")

    # Save
    DATA_DIR.mkdir(exist_ok=True)
    model_tag = args.model.replace("/", "_")
    out = DATA_DIR / f"word_trace_{model_tag}.json"
    with open(out, "w") as f:
        json.dump({
            "model": args.model,
            "n_layers": model.cfg.n_layers,
            "d_model": model.cfg.d_model,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "target_words": TARGET_WORDS,
            "scenarios": all_results,
        }, f, indent=2)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
