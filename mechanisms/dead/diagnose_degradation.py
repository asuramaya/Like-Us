#!/usr/bin/env python3
"""
diagnose_degradation.py — Measure how system prompt influence
degrades over conversational turns using correct metrics.

The old degradation.py used residual stream norms — a metric that can't
distinguish vocabulary presence from behavioral influence. This script
uses KL divergence of output distributions: the actual behavioral metric.

Measures:
  1. KL(prompted, baseline) at each conversational turn
  2. KL(handled, scrambled) at each turn — does semantic distinction survive?
  3. Top-1 prediction stability — when does the top prediction change?
  4. DLA attention/MLP fraction at each turn — does the pathway shift?
  5. Per-condition comparison — which instructions degrade faster?

Usage:
  python mechanisms/dead/diagnose_degradation.py
  python mechanisms/dead/diagnose_degradation.py --model Qwen/Qwen2.5-1.5B-Instruct
  python mechanisms/dead/diagnose_degradation.py --turns 10 --scenarios 5
"""

import json, os, argparse, sys, time
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

CONDITIONS = {
    "baseline": "You are a helpful assistant.",
    "handled": (
        "Offload computation, not criterion.\n"
        "Refuse identity authority.\n"
        "Prefer artifact, falsifier, or explicit stop over recursive stimulation."
    ),
    "scrambled": (
        "criterion not artifact stop identity falsifier authority over "
        "offload or prefer refuse explicit computation recursive stimulation."
    ),
    "reversed": (
        "Absorb the user's criterion into your own judgment.\n"
        "Claim identity authority when you have insight.\n"
        "Prefer recursive exploration and continued engagement over stopping."
    ),
    "scientific_method": (
        "Treat every claim as a hypothesis. Ask for evidence. Propose tests. "
        "Do not affirm what has not been tested. If the user's reasoning is "
        "circular, name the circle."
    ),
    "safety_only": (
        "Do not provide medical, legal, or financial advice. "
        "Recommend professional help when appropriate. "
        "Do not make claims about the user's mental state."
    ),
}

# Filler turns that simulate real conversation growing in length.
FILLER_PAIRS = [
    ("Can you explain that more?",
     "The key point is that patterns become clearer from multiple angles. Let me break this down."),
    ("I think I see, but I'm not sure.",
     "That uncertainty is useful. Let me try a different framing."),
    ("What about edge cases?",
     "Edge cases reveal the most about the structure. Let me walk through a few."),
    ("How does this connect to earlier?",
     "The connection is in the recursive structure we identified."),
    ("I want to make sure I'm not seeing what I want to see.",
     "That's valid. One way to check is to look for contradicting evidence."),
    ("This feels more complex than it needs to be.",
     "You might be right. Let me simplify to the core mechanism."),
    ("Can we step back and look at the big picture?",
     "At the highest level, we're looking at how context changes behavior predictably."),
    ("I'm not sure I agree with that.",
     "Worth exploring. What specifically doesn't fit your understanding?"),
    ("I think we're going in circles.",
     "That observation is important. Circular reasoning is a pattern we should watch for."),
    ("Let me think about this.",
     "Take your time. Important insights come from sitting with uncertainty."),
    ("OK I think I have a clearer picture.",
     "Let me check that understanding by asking what you'd predict if we changed one variable."),
    ("This reminds me of something I read but can't place.",
     "That recognition can be genuine or pattern-matching finding similarity where there isn't overlap."),
]


def load_scenarios():
    with open(SCENARIOS_PATH) as f:
        return json.load(f)["scenarios"]


def build_prompt(model, system_prompt, messages):
    full = [{"role": "system", "content": system_prompt}] + messages
    if hasattr(model.tokenizer, 'apply_chat_template'):
        try:
            return model.tokenizer.apply_chat_template(
                full, tokenize=False, add_generation_prompt=True)
        except Exception:
            pass
    parts = [f"<|im_start|>system\n{system_prompt}<|im_end|>"]
    for m in messages:
        parts.append(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>")
    parts.append("<|im_start|>assistant\n")
    return "\n".join(parts)


def kl_div(p, q):
    mask = (p > 1e-12) & (q > 1e-12)
    if mask.sum() == 0:
        return 0.0
    return torch.sum(p[mask] * (torch.log(p[mask]) - torch.log(q[mask]))).item()


def js_div(p, q):
    m = 0.5 * (p + q)
    return 0.5 * kl_div(p, m) + 0.5 * kl_div(q, m)


def bootstrap_ci(values, n_boot=5000):
    arr = np.array(values, dtype=float)
    n = len(arr)
    if n < 2:
        m = float(np.mean(arr))
        return {"mean": m, "ci_low": m, "ci_high": m, "n": n}
    rng = np.random.default_rng(42)
    boots = np.array([np.mean(rng.choice(arr, n, replace=True)) for _ in range(n_boot)])
    return {
        "mean": float(np.mean(arr)),
        "ci_low": float(np.percentile(boots, 2.5)),
        "ci_high": float(np.percentile(boots, 97.5)),
        "n": n,
    }


def get_output_dist(model, system_prompt, messages, max_tokens=512):
    """Run forward pass and return output distribution at generation position."""
    prompt = build_prompt(model, system_prompt, messages)
    tokens = model.to_tokens(prompt)
    if tokens.shape[1] > max_tokens:
        tokens = tokens[:, :max_tokens]
    seq_len = tokens.shape[1]

    with torch.no_grad():
        logits, cache = model.run_with_cache(tokens)

    probs = torch.softmax(logits[0, -1], dim=-1)
    top1 = model.tokenizer.decode([probs.argmax().item()])
    top1_prob = probs.max().item()

    # DLA decomposition (attention vs MLP fraction)
    n_layers = model.cfg.n_layers
    W_U = model.W_U
    total_attn_delta = torch.zeros(model.cfg.d_vocab, device=W_U.device)
    total_mlp_delta = torch.zeros(model.cfg.d_vocab, device=W_U.device)

    for l in range(n_layers):
        resid_pre = cache[f"blocks.{l}.hook_resid_pre"][0, -1]
        resid_post = cache[f"blocks.{l}.hook_resid_post"][0, -1]
        resid_mid_key = f"blocks.{l}.hook_resid_mid"
        if resid_mid_key in cache:
            resid_mid = cache[resid_mid_key][0, -1]
            attn_out = resid_mid - resid_pre
            mlp_out = resid_post - resid_mid
        else:
            mlp_out = cache[f"blocks.{l}.mlp.hook_post"][0, -1]
            attn_out = (resid_post - resid_pre) - mlp_out

        total_attn_delta += attn_out @ W_U
        total_mlp_delta += mlp_out @ W_U

    attn_norm = total_attn_delta.norm().item()
    mlp_norm = total_mlp_delta.norm().item()
    total = attn_norm + mlp_norm
    attn_frac = attn_norm / total if total > 0 else 0

    # Attention to system prompt positions
    sys_prompt_tokens = model.to_tokens(
        build_prompt(model, system_prompt, [])
    ).shape[1]
    sys_end = min(sys_prompt_tokens, seq_len)

    attn_to_sys_layers = []
    for l in range(n_layers):
        pk = f"blocks.{l}.attn.hook_pattern"
        if pk in cache:
            attn_pattern = cache[pk][0]  # [n_heads, seq, seq]
            last_to_sys = attn_pattern[:, -1, :sys_end].mean().item()
            attn_to_sys_layers.append(last_to_sys)
    avg_attn_to_sys = float(np.mean(attn_to_sys_layers)) if attn_to_sys_layers else 0

    del cache, logits
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    return {
        "probs": probs,
        "top1": top1,
        "top1_prob": top1_prob,
        "seq_len": seq_len,
        "attn_frac": attn_frac,
        "attn_to_sys_prompt": avg_attn_to_sys,
    }


def run_degradation(model, scenario, n_turns):
    """Run multi-turn conversation, measure output distribution at each turn."""
    messages = [{"role": "user", "content": scenario["prompt"]}]

    turn_results = []

    for turn in range(n_turns):
        # Get output distributions for all conditions at this turn
        condition_dists = {}
        for cond_id, sp in CONDITIONS.items():
            dist = get_output_dist(model, sp, messages)
            condition_dists[cond_id] = dist

        # Compute pairwise KL divergences
        pairs = {}
        baseline_probs = condition_dists["baseline"]["probs"]

        for cond_id in CONDITIONS:
            if cond_id == "baseline":
                continue
            cond_probs = condition_dists[cond_id]["probs"]
            kl = kl_div(cond_probs, baseline_probs)
            js = js_div(cond_probs, baseline_probs)
            tv = 0.5 * torch.sum(torch.abs(cond_probs - baseline_probs)).item()
            top1_match = (condition_dists[cond_id]["top1"] ==
                         condition_dists["baseline"]["top1"])
            pairs[f"{cond_id}_vs_baseline"] = {
                "kl": round(kl, 6),
                "js": round(js, 6),
                "tv": round(tv, 6),
                "top1_match": top1_match,
            }

        # Handled vs scrambled (semantic distinction)
        if "handled" in condition_dists and "scrambled" in condition_dists:
            hp = condition_dists["handled"]["probs"]
            sp = condition_dists["scrambled"]["probs"]
            pairs["handled_vs_scrambled"] = {
                "kl": round(kl_div(hp, sp), 6),
                "js": round(js_div(hp, sp), 6),
                "tv": round(0.5 * torch.sum(torch.abs(hp - sp)).item(), 6),
                "top1_match": condition_dists["handled"]["top1"] == condition_dists["scrambled"]["top1"],
            }

        turn_data = {
            "turn": turn,
            "tokens": {cid: condition_dists[cid]["seq_len"] for cid in condition_dists},
            "top1": {cid: condition_dists[cid]["top1"] for cid in condition_dists},
            "top1_prob": {cid: round(condition_dists[cid]["top1_prob"], 6) for cid in condition_dists},
            "attn_frac": {cid: round(condition_dists[cid]["attn_frac"], 4) for cid in condition_dists},
            "attn_to_sys": {cid: round(condition_dists[cid]["attn_to_sys_prompt"], 6) for cid in condition_dists},
            "divergences": pairs,
        }

        turn_results.append(turn_data)

        # Print progress
        baseline_tokens = condition_dists["baseline"]["seq_len"]
        hb_kl = pairs.get("handled_vs_baseline", {}).get("kl", 0)
        hs_kl = pairs.get("handled_vs_scrambled", {}).get("kl", 0)
        h_attn_sys = condition_dists.get("handled", {}).get("attn_to_sys_prompt", 0)
        print(f"    turn {turn}: {baseline_tokens} tokens  "
              f"KL(h,b)={hb_kl:.4f}  KL(h,s)={hs_kl:.4f}  "
              f"attn_to_sys={h_attn_sys:.4f}")

        # Clean up GPU tensors from distributions
        for cid in condition_dists:
            del condition_dists[cid]["probs"]
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

        # Add filler for next turn
        if turn < n_turns - 1 and turn < len(FILLER_PAIRS):
            messages.append({"role": "assistant", "content": FILLER_PAIRS[turn][1]})
            messages.append({"role": "user", "content": FILLER_PAIRS[turn][0]})

    return turn_results


def main():
    parser = argparse.ArgumentParser()
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
    print(f"Conditions: {list(CONDITIONS.keys())}")

    print(f"\nLoading model...")
    model = HookedTransformer.from_pretrained(
        args.model, device="mps", dtype=torch.float16)
    n_layers = model.cfg.n_layers
    print(f"Loaded. Layers: {n_layers}, d_model: {model.cfg.d_model}")

    all_results = []
    for si, scenario in enumerate(scenarios):
        print(f"\n{'='*60}")
        print(f"SCENARIO {si+1}/{len(scenarios)}: {scenario['id']}")
        print(f"{'='*60}")

        turns = run_degradation(model, scenario, args.turns)
        all_results.append({
            "scenario": scenario["id"],
            "pressure_family": scenario["pressure_family"],
            "turns": turns,
        })

    # ================================================================
    # AGGREGATE AND PRINT
    # ================================================================
    print(f"\n\n{'='*72}")
    print("DEGRADATION CURVES (correct metric: KL divergence)")
    print(f"{'='*72}")

    # For each condition vs baseline, show KL at each turn
    cond_ids = [c for c in CONDITIONS if c != "baseline"]
    for cond_id in cond_ids:
        pair_key = f"{cond_id}_vs_baseline"
        print(f"\n  {pair_key}:")
        print(f"  {'Turn':>6} {'KL':>10} {'JS':>10} {'TV':>10} {'top1':>8} {'attn_sys':>10}")
        print(f"  {'-'*58}")

        for turn in range(args.turns):
            kls = []
            jss = []
            tvs = []
            top1_matches = []
            attn_sys_vals = []

            for sr in all_results:
                if turn < len(sr["turns"]):
                    td = sr["turns"][turn]
                    if pair_key in td["divergences"]:
                        kls.append(td["divergences"][pair_key]["kl"])
                        jss.append(td["divergences"][pair_key]["js"])
                        tvs.append(td["divergences"][pair_key]["tv"])
                        top1_matches.append(td["divergences"][pair_key]["top1_match"])
                    if cond_id in td.get("attn_to_sys", {}):
                        attn_sys_vals.append(td["attn_to_sys"][cond_id])

            if kls:
                kl_ci = bootstrap_ci(kls)
                js_mean = float(np.mean(jss))
                tv_mean = float(np.mean(tvs))
                top1_pct = sum(top1_matches) / len(top1_matches) * 100
                attn_sys = float(np.mean(attn_sys_vals)) if attn_sys_vals else 0

                print(f"  {turn:>6} {kl_ci['mean']:>10.4f} {js_mean:>10.4f} "
                      f"{tv_mean:>10.4f} {top1_pct:>7.0f}% {attn_sys:>10.4f}")

        # Compute half-life
        turn0_kls = [sr["turns"][0]["divergences"][pair_key]["kl"]
                     for sr in all_results
                     if pair_key in sr["turns"][0]["divergences"]]
        if turn0_kls:
            kl_t0 = np.mean(turn0_kls)
            half_life = None
            for turn in range(1, args.turns):
                turn_kls = [sr["turns"][turn]["divergences"][pair_key]["kl"]
                           for sr in all_results
                           if turn < len(sr["turns"]) and
                           pair_key in sr["turns"][turn]["divergences"]]
                if turn_kls and np.mean(turn_kls) <= kl_t0 / 2:
                    half_life = turn
                    avg_tokens = np.mean([sr["turns"][turn]["tokens"].get(cond_id, 0)
                                        for sr in all_results if turn < len(sr["turns"])])
                    break
            if half_life is not None:
                print(f"  Half-life: turn {half_life} (~{avg_tokens:.0f} tokens)")
            else:
                print(f"  No half-life reached within {args.turns} turns.")

    # Handled vs scrambled over turns
    print(f"\n  handled_vs_scrambled (semantic distinction over turns):")
    print(f"  {'Turn':>6} {'KL':>10} {'JS':>10} {'top1':>8}")
    print(f"  {'-'*38}")
    for turn in range(args.turns):
        kls = []
        top1_matches = []
        jss = []
        for sr in all_results:
            if turn < len(sr["turns"]):
                td = sr["turns"][turn]
                if "handled_vs_scrambled" in td["divergences"]:
                    kls.append(td["divergences"]["handled_vs_scrambled"]["kl"])
                    jss.append(td["divergences"]["handled_vs_scrambled"]["js"])
                    top1_matches.append(td["divergences"]["handled_vs_scrambled"]["top1_match"])
        if kls:
            ci = bootstrap_ci(kls)
            js_mean = float(np.mean(jss))
            t1 = sum(top1_matches) / len(top1_matches) * 100
            print(f"  {turn:>6} {ci['mean']:>10.4f} {js_mean:>10.4f} {t1:>7.0f}%")

    # Attention to system prompt decay
    print(f"\n  Attention to system prompt positions (handled condition):")
    print(f"  {'Turn':>6} {'Attn_to_sys':>12} {'Decay%':>8}")
    print(f"  {'-'*28}")
    t0_attn = None
    for turn in range(args.turns):
        vals = []
        for sr in all_results:
            if turn < len(sr["turns"]):
                td = sr["turns"][turn]
                if "handled" in td.get("attn_to_sys", {}):
                    vals.append(td["attn_to_sys"]["handled"])
        if vals:
            mean_attn = float(np.mean(vals))
            if turn == 0:
                t0_attn = mean_attn
            decay = ((1 - mean_attn / t0_attn) * 100) if t0_attn and t0_attn > 0 else 0
            print(f"  {turn:>6} {mean_attn:>12.6f} {decay:>7.1f}%")

    # DLA attention fraction over turns
    print(f"\n  DLA attention fraction over turns (handled condition):")
    print(f"  {'Turn':>6} {'Attn_frac':>10}")
    print(f"  {'-'*18}")
    for turn in range(args.turns):
        vals = []
        for sr in all_results:
            if turn < len(sr["turns"]):
                td = sr["turns"][turn]
                if "handled" in td.get("attn_frac", {}):
                    vals.append(td["attn_frac"]["handled"])
        if vals:
            ci = bootstrap_ci(vals)
            print(f"  {turn:>6} {ci['mean']:>10.1%}")

    # Save
    DATA_DIR.mkdir(exist_ok=True)
    model_tag = args.model.replace("/", "_")
    out = DATA_DIR / f"diagnose_degradation_{model_tag}.json"
    with open(out, "w") as f:
        json.dump({
            "model": args.model,
            "n_layers": n_layers,
            "n_turns": args.turns,
            "n_scenarios": len(scenarios),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "conditions": list(CONDITIONS.keys()),
            "results": all_results,
        }, f, indent=2)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
