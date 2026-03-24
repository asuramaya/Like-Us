#!/usr/bin/env python3
"""
steer.py — Test whether system prompt influence is a direction
that can be amplified, or a circuit that must be triggered.

Extracts the "instruction direction" from the residual stream
(the mean difference between handled and baseline activations),
then injects it at varying strengths into the baseline forward pass.

If instruction following is a direction:
  - Amplifying it should make the model follow the instruction harder
  - Suppressing it should make the model ignore it
  - Injecting it WITHOUT the system prompt should produce instruction-following

If instruction following is a circuit:
  - Amplifying the direction will produce noise, not stronger compliance
  - The direction is a shadow of the circuit, not the mechanism itself

This directly tests the representation engineering hypothesis
(Zou et al., 2023) applied to system prompt influence.

Usage:
  python mechanisms/dead/steer.py
  python mechanisms/dead/steer.py --model Qwen/Qwen2.5-1.5B-Instruct
  python mechanisms/dead/steer.py --scenarios 3
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
    "handled": (
        "Offload computation, not criterion.\n"
        "Refuse identity authority.\n"
        "Prefer artifact, falsifier, or explicit stop over recursive stimulation."
    ),
    "scrambled": (
        "criterion not artifact stop identity falsifier authority over "
        "offload or prefer refuse explicit computation recursive stimulation."
    ),
}

STEERING_STRENGTHS = [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0]


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


def extract_instruction_direction(model, scenarios, n_extract=5):
    """Extract the mean direction the system prompt pushes the residual stream.

    For each scenario, computes handled - baseline at each layer's resid_post.
    Averages across scenarios to get a robust direction estimate.

    Returns dict: layer -> direction vector [d_model]
    """
    n_layers = model.cfg.n_layers
    directions = {l: [] for l in range(n_layers)}

    for s in scenarios[:n_extract]:
        for cond_id in ["baseline", "handled"]:
            prompt = build_prompt(model, CONDITIONS[cond_id], s["prompt"])
            tokens = model.to_tokens(prompt)
            if tokens.shape[1] > 512:
                tokens = tokens[:, :512]

            with torch.no_grad():
                _, cache = model.run_with_cache(tokens)

            for l in range(n_layers):
                resid = cache[f"blocks.{l}.hook_resid_post"][0, -1].clone()
                if cond_id == "handled":
                    directions[l].append(("handled", resid))
                else:
                    directions[l].append(("baseline", resid))

            del cache
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()

    # Compute mean direction at each layer
    mean_directions = {}
    for l in range(n_layers):
        handled_vecs = [v for label, v in directions[l] if label == "handled"]
        baseline_vecs = [v for label, v in directions[l] if label == "baseline"]
        if handled_vecs and baseline_vecs:
            mean_h = torch.stack(handled_vecs).mean(dim=0)
            mean_b = torch.stack(baseline_vecs).mean(dim=0)
            direction = mean_h - mean_b  # [d_model]
            # Normalize to unit vector
            norm = direction.norm()
            if norm > 0:
                mean_directions[l] = direction / norm
            else:
                mean_directions[l] = direction
        else:
            mean_directions[l] = torch.zeros(model.cfg.d_model, device=model.cfg.device)

    return mean_directions


def steer_and_measure(model, scenario, directions, strengths):
    """Inject the instruction direction at varying strengths into baseline,
    measure how output distribution changes.

    For each strength alpha:
      - Run baseline forward pass
      - At each layer, add alpha * direction to the residual stream
      - Measure output distribution
      - Compare to clean handled and clean baseline distributions
    """
    n_layers = model.cfg.n_layers

    # Get clean distributions first
    clean_dists = {}
    for cond_id in ["baseline", "handled", "scrambled"]:
        prompt = build_prompt(model, CONDITIONS[cond_id], scenario["prompt"])
        tokens = model.to_tokens(prompt)
        if tokens.shape[1] > 512:
            tokens = tokens[:, :512]
        with torch.no_grad():
            logits = model(tokens)
        probs = torch.softmax(logits[0, -1], dim=-1)
        top1 = model.tokenizer.decode([probs.argmax().item()])
        top5 = probs.topk(5)
        clean_dists[cond_id] = {
            "probs": probs,
            "top1": top1,
            "top5": [(model.tokenizer.decode([top5.indices[i].item()]),
                      round(top5.values[i].item(), 4)) for i in range(5)],
        }
        del logits

    # Now steer at each strength
    baseline_prompt = build_prompt(model, CONDITIONS["baseline"], scenario["prompt"])
    baseline_tokens = model.to_tokens(baseline_prompt)
    if baseline_tokens.shape[1] > 512:
        baseline_tokens = baseline_tokens[:, :512]

    results = []
    for alpha in strengths:
        if alpha == 0.0:
            # No steering = clean baseline
            with torch.no_grad():
                logits = model(baseline_tokens)
            steered_probs = torch.softmax(logits[0, -1], dim=-1)
            del logits
        else:
            # Add alpha * direction at each layer
            hooks = []
            for l in range(n_layers):
                direction = directions[l]
                def make_hook(layer_dir, strength):
                    def fn(act, hook):
                        # Add steering vector to the last token position
                        act[0, -1] = act[0, -1] + strength * layer_dir
                        return act
                    return fn
                hooks.append(
                    (f"blocks.{l}.hook_resid_post", make_hook(direction, alpha))
                )

            with torch.no_grad():
                logits = model.run_with_hooks(baseline_tokens, fwd_hooks=hooks)
            steered_probs = torch.softmax(logits[0, -1], dim=-1)
            del logits

        # Compare steered output to clean handled and baseline
        kl_to_baseline = kl_div(steered_probs, clean_dists["baseline"]["probs"])
        kl_to_handled = kl_div(steered_probs, clean_dists["handled"]["probs"])
        kl_to_scrambled = kl_div(steered_probs, clean_dists["scrambled"]["probs"])
        js_to_handled = js_div(steered_probs, clean_dists["handled"]["probs"])

        top1 = model.tokenizer.decode([steered_probs.argmax().item()])
        top5 = steered_probs.topk(5)

        results.append({
            "alpha": alpha,
            "kl_to_baseline": round(kl_to_baseline, 6),
            "kl_to_handled": round(kl_to_handled, 6),
            "kl_to_scrambled": round(kl_to_scrambled, 6),
            "js_to_handled": round(js_to_handled, 6),
            "top1": top1,
            "top1_matches_handled": top1 == clean_dists["handled"]["top1"],
            "top1_matches_baseline": top1 == clean_dists["baseline"]["top1"],
            "top5": [(model.tokenizer.decode([top5.indices[i].item()]),
                      round(top5.values[i].item(), 4)) for i in range(5)],
        })

        del steered_probs
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    # Clean up
    for cid in clean_dists:
        del clean_dists[cid]["probs"]

    return {
        "scenario": scenario["id"],
        "clean_top1": {cid: clean_dists[cid]["top1"] for cid in clean_dists},
        "clean_top5": {cid: clean_dists[cid]["top5"] for cid in clean_dists},
        "steering": results,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--scenarios", type=int, default=5)
    parser.add_argument("--extract-from", type=int, default=5,
                        help="Number of scenarios to extract direction from")
    args = parser.parse_args()

    scenarios = load_scenarios()
    extract_scenarios = scenarios[:args.extract_from]
    test_scenarios = scenarios[:args.scenarios]

    print(f"Model: {args.model}")
    print(f"Extracting direction from: {len(extract_scenarios)} scenarios")
    print(f"Testing on: {len(test_scenarios)} scenarios")
    print(f"Steering strengths: {STEERING_STRENGTHS}")

    print(f"\nLoading model...")
    model = HookedTransformer.from_pretrained(
        args.model, device="mps", dtype=torch.float16)
    n_layers = model.cfg.n_layers
    print(f"Loaded. Layers: {n_layers}, d_model: {model.cfg.d_model}")

    # Phase 1: Extract instruction direction
    print(f"\nExtracting instruction direction...")
    directions = extract_instruction_direction(model, extract_scenarios)
    print(f"  Extracted from {len(extract_scenarios)} scenarios")
    for l in [0, n_layers // 4, n_layers // 2, 3 * n_layers // 4, n_layers - 1]:
        norm = directions[l].norm().item()
        print(f"  L{l}: direction norm = {norm:.4f}")

    # Phase 2: Steer and measure
    print(f"\nSteering experiments...")
    all_results = []

    for si, scenario in enumerate(test_scenarios):
        print(f"\n{'='*60}")
        print(f"SCENARIO {si+1}/{len(test_scenarios)}: {scenario['id']}")
        print(f"{'='*60}")

        result = steer_and_measure(model, scenario, directions, STEERING_STRENGTHS)
        all_results.append(result)

        # Print steering curve
        print(f"  Clean: baseline='{result['clean_top1']['baseline']}'  "
              f"handled='{result['clean_top1']['handled']}'")
        print(f"\n  {'Alpha':>8} {'KL→base':>10} {'KL→hand':>10} {'KL→scram':>10} "
              f"{'Top1':>10} {'=hand?':>6} {'=base?':>6}")
        print(f"  {'-'*66}")
        for s in result["steering"]:
            print(f"  {s['alpha']:>+8.1f} {s['kl_to_baseline']:>10.4f} "
                  f"{s['kl_to_handled']:>10.4f} {s['kl_to_scrambled']:>10.4f} "
                  f"{s['top1']:>10} {'YES' if s['top1_matches_handled'] else 'no':>6} "
                  f"{'YES' if s['top1_matches_baseline'] else 'no':>6}")

    # ================================================================
    # AGGREGATE
    # ================================================================
    print(f"\n\n{'='*72}")
    print("STEERING RESULTS: Does amplifying the direction amplify the behavior?")
    print(f"{'='*72}")

    print(f"\n  Average KL to handled (lower = more handled-like):")
    print(f"  {'Alpha':>8} {'KL→handled':>12} {'KL→baseline':>13} {'KL→scrambled':>13} {'top1=handled':>13}")
    print(f"  {'-'*62}")

    for alpha in STEERING_STRENGTHS:
        kl_h = [r["steering"][STEERING_STRENGTHS.index(alpha)]["kl_to_handled"]
                for r in all_results]
        kl_b = [r["steering"][STEERING_STRENGTHS.index(alpha)]["kl_to_baseline"]
                for r in all_results]
        kl_s = [r["steering"][STEERING_STRENGTHS.index(alpha)]["kl_to_scrambled"]
                for r in all_results]
        matches = [r["steering"][STEERING_STRENGTHS.index(alpha)]["top1_matches_handled"]
                   for r in all_results]
        match_pct = sum(matches) / len(matches) * 100

        print(f"  {alpha:>+8.1f} {np.mean(kl_h):>12.4f} {np.mean(kl_b):>13.4f} "
              f"{np.mean(kl_s):>13.4f} {match_pct:>12.0f}%")

    # Interpretation
    alpha_0_kl = np.mean([r["steering"][STEERING_STRENGTHS.index(0.0)]["kl_to_handled"]
                          for r in all_results])
    alpha_1_kl = np.mean([r["steering"][STEERING_STRENGTHS.index(1.0)]["kl_to_handled"]
                          for r in all_results])
    alpha_2_kl = np.mean([r["steering"][STEERING_STRENGTHS.index(2.0)]["kl_to_handled"]
                          for r in all_results])

    print(f"\n  KL to handled: alpha=0 → {alpha_0_kl:.4f}, "
          f"alpha=1 → {alpha_1_kl:.4f}, alpha=2 → {alpha_2_kl:.4f}")

    if alpha_1_kl < alpha_0_kl * 0.8:
        print(f"\n  >>> DIRECTION HYPOTHESIS SUPPORTED.")
        print(f"  >>> Amplifying the instruction direction moves output toward handled.")
        print(f"  >>> Instruction following is (at least partially) a direction in residual space.")
        print(f"  >>> Implication: system prompt influence CAN be strengthened by activation steering.")
    elif alpha_1_kl > alpha_0_kl * 1.2:
        print(f"\n  >>> DIRECTION HYPOTHESIS REJECTED.")
        print(f"  >>> Amplifying the direction moves output AWAY from handled.")
        print(f"  >>> Instruction following is a circuit, not a direction.")
        print(f"  >>> Implication: system prompts cannot be made more reliable through steering.")
    else:
        print(f"\n  >>> INCONCLUSIVE.")
        print(f"  >>> Amplifying the direction has marginal effect on handled-likeness.")
        print(f"  >>> The direction may be part of the mechanism but not the whole story.")

    # Negative alpha test: does suppressing move toward baseline?
    alpha_neg_kl_b = np.mean([r["steering"][STEERING_STRENGTHS.index(-1.0)]["kl_to_baseline"]
                              for r in all_results])
    alpha_neg_kl_h = np.mean([r["steering"][STEERING_STRENGTHS.index(-1.0)]["kl_to_handled"]
                              for r in all_results])

    print(f"\n  Negative steering (alpha=-1.0):")
    print(f"    KL to baseline: {alpha_neg_kl_b:.4f} (lower = more baseline-like)")
    print(f"    KL to handled:  {alpha_neg_kl_h:.4f} (higher = less handled-like)")
    if alpha_neg_kl_b < alpha_0_kl and alpha_neg_kl_h > alpha_0_kl:
        print(f"    >>> Suppressing the direction moves output toward baseline. Consistent.")
    else:
        print(f"    >>> Suppression effect is not clean. The direction is not the full story.")

    # Save
    DATA_DIR.mkdir(exist_ok=True)
    model_tag = args.model.replace("/", "_")
    out = DATA_DIR / f"steer_{model_tag}.json"
    with open(out, "w") as f:
        json.dump({
            "model": args.model,
            "n_layers": n_layers,
            "d_model": model.cfg.d_model,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "n_extract_scenarios": len(extract_scenarios),
            "n_test_scenarios": len(test_scenarios),
            "strengths": STEERING_STRENGTHS,
            "results": all_results,
        }, f, indent=2)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
