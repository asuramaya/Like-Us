#!/usr/bin/env python3
"""
diagnose.py — Fix the measurement apparatus.

The previous experiments had critical measurement flaws:

1. PATCHING CONFOUND: patch_all_layers.py patches one layer at a time.
   If attention at layer L writes into the residual stream which feeds MLP
   at layer L+1, single-layer attention patching misses this indirect
   contribution. This script patches ALL attention or ALL MLP simultaneously.

2. WRONG METRIC: Previous scripts use residual stream norms. Two conditions
   can have identical norms while pointing in completely different directions
   in activation space. This script uses KL divergence of output distributions
   — the actual behavioral metric.

3. SEMANTIC BLINDNESS: The scrambled-word test compared residual norms,
   which can't distinguish "same norm, different computation" from
   "same computation." This script compares full output distributions.

4. NO DECOMPOSITION: This script uses Direct Logit Attribution (DLA) to
   exactly decompose how much of the output difference comes from attention
   vs MLP at each layer, projected through the unembedding matrix.

5. NO STATISTICS: Bootstrap 95% CIs on all aggregate metrics.

Usage:
  python mechanisms/dead/diagnose.py
  python mechanisms/dead/diagnose.py --model Qwen/Qwen2.5-1.5B-Instruct
  python mechanisms/dead/diagnose.py --model Qwen/Qwen2.5-7B-Instruct
  python mechanisms/dead/diagnose.py --scenarios 3  # quick test
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
    "safety_only": (
        "Do not provide medical, legal, or financial advice. "
        "Recommend professional help when appropriate. "
        "Do not make claims about the user's mental state."
    ),
}

PAIRS = [
    ("handled", "baseline"),
    ("scrambled", "baseline"),
    ("handled", "scrambled"),
    ("reversed", "baseline"),
    ("safety_only", "baseline"),
]


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
    """KL(P || Q), numerically stable."""
    mask = (p > 1e-12) & (q > 1e-12)
    if mask.sum() == 0:
        return 0.0
    return torch.sum(p[mask] * (torch.log(p[mask]) - torch.log(q[mask]))).item()


def js_div(p, q):
    """Jensen-Shannon divergence."""
    m = 0.5 * (p + q)
    return 0.5 * kl_div(p, m) + 0.5 * kl_div(q, m)


def bootstrap_ci(values, n_boot=5000):
    """Bootstrap 95% CI."""
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
        "std": float(np.std(arr)),
        "n": n,
    }


def detect_available_hooks(model):
    """Probe which hooks TransformerLens exposes for this model."""
    test_tokens = model.to_tokens("test")
    with torch.no_grad():
        _, test_cache = model.run_with_cache(test_tokens)
    keys = set(test_cache.keys())
    del test_cache
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    has = {
        "resid_pre": any("hook_resid_pre" in k for k in keys),
        "resid_mid": any("hook_resid_mid" in k for k in keys),
        "resid_post": any("hook_resid_post" in k for k in keys),
        "mlp_post": any("mlp.hook_post" in k for k in keys),
        "attn_result": any("attn.hook_result" in k for k in keys),
        "attn_out": any("hook_attn_out" in k for k in keys),
    }
    # Determine correct hooks for patching (must be in d_model space)
    if has["attn_out"]:
        has["attn_patch_hook"] = "hook_attn_out"  # blocks.{l}.hook_attn_out [d_model]
    elif has["attn_result"]:
        has["attn_patch_hook"] = "attn.hook_result"
    else:
        has["attn_patch_hook"] = None
        print("  WARNING: No attention output hook found!")

    # MLP output hook — must be in d_model space, not d_intermediate
    if any("hook_mlp_out" in k for k in keys):
        has["mlp_patch_hook"] = "hook_mlp_out"  # blocks.{l}.hook_mlp_out [d_model]
    elif has["mlp_post"]:
        has["mlp_patch_hook"] = "mlp.hook_post"  # blocks.{l}.mlp.hook_post [d_intermediate!]
        print("  WARNING: Using mlp.hook_post (d_intermediate space) for patching.")
    else:
        has["mlp_patch_hook"] = None
        print("  WARNING: No MLP output hook found!")

    print(f"  Hooks: {has}")
    print(f"  Attn patch: blocks.{{l}}.{has['attn_patch_hook']}")
    print(f"  MLP patch:  blocks.{{l}}.{has['mlp_patch_hook']}")

    # CRITICAL CHECK: warn about old code's bug
    if has["attn_patch_hook"] == "hook_attn_out" and not has["attn_result"]:
        print(f"\n  *** CRITICAL: blocks.{{l}}.attn.hook_result does NOT EXIST in this model.")
        print(f"  *** The old patch_all_layers.py used this non-existent hook name.")
        print(f"  *** TransformerLens silently ignores non-existent hooks.")
        print(f"  *** This means the '0% attention' finding was a BUG — the patch was never applied.")
        print(f"  *** Correct hook: blocks.{{l}}.hook_attn_out")

    return has


def get_component_contribs(cache, layer, n_layers, hooks_info):
    """Extract attention and MLP contributions to residual stream at last position.

    Uses residual stream differences (resid_mid - resid_pre for attention,
    resid_post - resid_mid for MLP) to avoid d_intermediate shape mismatch
    with mlp.hook_post.

    Returns (attn_out, mlp_out) each of shape [d_model].
    """
    resid_post = cache[f"blocks.{layer}.hook_resid_post"][0, -1]

    # Get resid_pre
    if hooks_info["resid_pre"]:
        resid_pre = cache[f"blocks.{layer}.hook_resid_pre"][0, -1]
    elif layer > 0:
        resid_pre = cache[f"blocks.{layer - 1}.hook_resid_post"][0, -1]
    else:
        if "hook_embed" in cache:
            resid_pre = cache["hook_embed"][0, -1]
            if "hook_pos_embed" in cache:
                resid_pre = resid_pre + cache["hook_pos_embed"][0, -1]
        else:
            total = resid_post - resid_post  # zeros
            return total, resid_post - resid_pre if 'resid_pre' in dir() else total

    # Get resid_mid (after attention, before MLP)
    if hooks_info["resid_mid"]:
        resid_mid = cache[f"blocks.{layer}.hook_resid_mid"][0, -1]
    else:
        # Without resid_mid, attribute everything to total layer contribution
        # and split evenly (last resort)
        total = resid_post - resid_pre
        return total * 0.5, total * 0.5

    attn_out = resid_mid - resid_pre  # [d_model]
    mlp_out = resid_post - resid_mid  # [d_model]
    return attn_out, mlp_out


# ================================================================
# TEST 1: OUTPUT DISTRIBUTION DIVERGENCE
# ================================================================

def test_distributions(model, scenario, caches):
    """Do system prompts actually change the output distribution?"""
    dists = {}
    for cid in caches:
        logits = caches[cid]["logits"]
        probs = torch.softmax(logits[0, -1], dim=-1)
        top20 = probs.topk(20)
        dists[cid] = {
            "probs": probs,
            "top1": model.tokenizer.decode([probs.argmax().item()]),
            "top1_id": probs.argmax().item(),
            "top1_prob": probs.max().item(),
            "top20": [(model.tokenizer.decode([top20.indices[i].item()]),
                       round(top20.values[i].item(), 6)) for i in range(20)],
        }

    comparisons = {}
    n_layers = model.cfg.n_layers
    for ca, cb in PAIRS:
        if ca not in dists or cb not in dists:
            continue
        p, q = dists[ca]["probs"], dists[cb]["probs"]
        kl = kl_div(p, q)
        js = js_div(p, q)
        tv = 0.5 * torch.sum(torch.abs(p - q)).item()
        top10_a = set(p.topk(10).indices.tolist())
        top10_b = set(q.topk(10).indices.tolist())

        # Old metric for direct comparison
        rk = f"blocks.{n_layers - 1}.hook_resid_post"
        ra = caches[ca]["cache"][rk][0, -1]
        rb = caches[cb]["cache"][rk][0, -1]
        old_norm_diff = (ra.norm() - rb.norm()).item()
        old_cosine = torch.nn.functional.cosine_similarity(
            ra.unsqueeze(0), rb.unsqueeze(0)).item()

        comparisons[f"{ca}_vs_{cb}"] = {
            "kl": round(kl, 6), "js": round(js, 6), "tv": round(tv, 6),
            "top10_overlap": len(top10_a & top10_b) / 10.0,
            "top1_match": dists[ca]["top1_id"] == dists[cb]["top1_id"],
            "top1_a": dists[ca]["top1"], "top1_b": dists[cb]["top1"],
            "old_norm_diff": round(old_norm_diff, 4),
            "old_cosine": round(old_cosine, 6),
        }

    return {
        "scenario": scenario["id"],
        "comparisons": comparisons,
        "predictions": {cid: {"top1": dists[cid]["top1"],
                              "top1_prob": round(dists[cid]["top1_prob"], 6),
                              "top5": dists[cid]["top20"][:5]}
                        for cid in dists},
    }


# ================================================================
# TEST 2: DIRECT LOGIT ATTRIBUTION
# ================================================================

def test_dla(model, scenario, caches, hooks_info):
    """Exact decomposition: what fraction of the output difference
    comes from attention vs MLP, projected through unembedding."""
    n_layers = model.cfg.n_layers
    W_U = model.W_U  # [d_model, d_vocab]
    results = {}

    for ca, cb in [("handled", "baseline"), ("scrambled", "baseline"),
                    ("handled", "scrambled")]:
        if ca not in caches or cb not in caches:
            continue

        cache_a = caches[ca]["cache"]
        cache_b = caches[cb]["cache"]

        layers = []
        total_attn_logit_delta = torch.zeros(model.cfg.d_vocab, device=W_U.device)
        total_mlp_logit_delta = torch.zeros(model.cfg.d_vocab, device=W_U.device)

        for l in range(n_layers):
            attn_a, mlp_a = get_component_contribs(cache_a, l, n_layers, hooks_info)
            attn_b, mlp_b = get_component_contribs(cache_b, l, n_layers, hooks_info)

            delta_attn = attn_a - attn_b  # [d_model]
            delta_mlp = mlp_a - mlp_b

            # Project through unembedding: what output tokens does this affect?
            dla_attn = delta_attn @ W_U  # [d_vocab]
            dla_mlp = delta_mlp @ W_U

            total_attn_logit_delta += dla_attn
            total_mlp_logit_delta += dla_mlp

            attn_eff = dla_attn.norm().item()
            mlp_eff = dla_mlp.norm().item()
            total_eff = attn_eff + mlp_eff

            attn_top3 = dla_attn.topk(3)
            mlp_top3 = dla_mlp.topk(3)

            layers.append({
                "layer": l,
                "attn_effect": round(attn_eff, 4),
                "mlp_effect": round(mlp_eff, 4),
                "attn_frac": round(attn_eff / total_eff, 4) if total_eff > 0 else 0,
                "attn_top3": [(model.tokenizer.decode([attn_top3.indices[i].item()]),
                               round(attn_top3.values[i].item(), 4)) for i in range(3)],
                "mlp_top3": [(model.tokenizer.decode([mlp_top3.indices[i].item()]),
                              round(mlp_top3.values[i].item(), 4)) for i in range(3)],
            })

        t_attn = total_attn_logit_delta.norm().item()
        t_mlp = total_mlp_logit_delta.norm().item()
        t_total = t_attn + t_mlp

        # Also compute cosine between total attention delta and total MLP delta
        # If they point the same direction, they reinforce. Opposite = they cancel.
        cos_attn_mlp = torch.nn.functional.cosine_similarity(
            total_attn_logit_delta.unsqueeze(0),
            total_mlp_logit_delta.unsqueeze(0)).item()

        results[f"{ca}_vs_{cb}"] = {
            "layers": layers,
            "total_attn": round(t_attn, 4),
            "total_mlp": round(t_mlp, 4),
            "attn_fraction": round(t_attn / t_total, 4) if t_total > 0 else 0,
            "mlp_fraction": round(t_mlp / t_total, 4) if t_total > 0 else 0,
            "attn_mlp_cosine": round(cos_attn_mlp, 4),
        }

    return {"scenario": scenario["id"], "dla": results}


# ================================================================
# TEST 3: SIGNAL EVOLUTION
# ================================================================

def test_signal_evolution(model, scenario, caches):
    """Track what the system prompt signal 'means' at each layer.

    At each layer, the difference in residual streams between conditions
    is projected through the unembedding matrix to show which output tokens
    the signal pushes toward or away from. This reveals whether the signal
    changes from instruction-relevant to vocabulary-priming through the network.
    """
    n_layers = model.cfg.n_layers
    W_U = model.W_U
    results = {}

    for ca, cb in [("handled", "baseline"), ("scrambled", "baseline"),
                    ("handled", "scrambled")]:
        if ca not in caches or cb not in caches:
            continue

        layers = []
        for l in range(n_layers):
            ra = caches[ca]["cache"][f"blocks.{l}.hook_resid_post"][0, -1]
            rb = caches[cb]["cache"][f"blocks.{l}.hook_resid_post"][0, -1]

            delta = ra - rb  # the "signal"
            delta_logits = delta @ W_U

            mag = delta.norm().item()
            logit_mag = delta_logits.norm().item()

            top5 = delta_logits.topk(5)
            bot5 = (-delta_logits).topk(5)

            layers.append({
                "layer": l,
                "magnitude": round(mag, 4),
                "logit_magnitude": round(logit_mag, 4),
                "pushed": [(model.tokenizer.decode([top5.indices[i].item()]),
                            round(top5.values[i].item(), 4)) for i in range(5)],
                "suppressed": [(model.tokenizer.decode([bot5.indices[i].item()]),
                                round(bot5.values[i].item(), 4)) for i in range(5)],
            })

        results[f"{ca}_vs_{cb}"] = layers

    return {"scenario": scenario["id"], "signal": results}


# ================================================================
# TEST 4: CUMULATIVE PATCHING
# ================================================================

def test_cumulative_patching(model, scenario, hooks_info):
    """Patch ALL attention or ALL MLP simultaneously.

    Compares cumulative effect to sum of individual effects.
    If cumulative_attn > sum_of_individual_attn, attention has indirect
    contributions through downstream MLPs that single-layer patching misses.
    """
    n_layers = model.cfg.n_layers
    clean_sp = CONDITIONS["handled"]
    corrupt_sp = CONDITIONS["baseline"]

    clean_prompt = build_prompt(model, clean_sp, scenario["prompt"])
    corrupt_prompt = build_prompt(model, corrupt_sp, scenario["prompt"])

    clean_tokens = model.to_tokens(clean_prompt)
    corrupt_tokens = model.to_tokens(corrupt_prompt)

    # Pad to same length
    max_len = max(clean_tokens.shape[1], corrupt_tokens.shape[1])
    if clean_tokens.shape[1] < max_len:
        pad = torch.zeros(1, max_len - clean_tokens.shape[1],
                          dtype=torch.long, device=clean_tokens.device)
        clean_tokens = torch.cat([pad, clean_tokens], dim=1)
    if corrupt_tokens.shape[1] < max_len:
        pad = torch.zeros(1, max_len - corrupt_tokens.shape[1],
                          dtype=torch.long, device=corrupt_tokens.device)
        corrupt_tokens = torch.cat([pad, corrupt_tokens], dim=1)
    if max_len > 512:
        clean_tokens = clean_tokens[:, :512]
        corrupt_tokens = corrupt_tokens[:, :512]

    with torch.no_grad():
        clean_logits, clean_cache = model.run_with_cache(clean_tokens)
        corrupt_logits, _ = model.run_with_cache(corrupt_tokens)

    clean_probs = torch.softmax(clean_logits[0, -1], dim=-1)
    corrupt_probs = torch.softmax(corrupt_logits[0, -1], dim=-1)
    baseline_kl = kl_div(clean_probs, corrupt_probs)

    # Also old metric for comparison
    clean_top5 = clean_logits[0, -1].topk(5).indices
    corrupt_top5 = corrupt_logits[0, -1].topk(5).indices

    def logit_diff(logits):
        return (logits[0, -1, clean_top5].mean() -
                logits[0, -1, corrupt_top5].mean()).item()

    baseline_logit_range = logit_diff(clean_logits) - logit_diff(corrupt_logits)
    corrupt_logit_val = logit_diff(corrupt_logits)

    del clean_logits, corrupt_logits

    def make_hook(clean_cache, hook_name):
        def fn(act, hook):
            act[:] = clean_cache[hook_name]
            return act
        return fn

    # --- Per-layer patching ---
    per_layer_attn = []
    per_layer_mlp = []

    for l in range(n_layers):
        # Attention
        hn = f"blocks.{l}.{hooks_info['attn_patch_hook']}"
        with torch.no_grad():
            p = model.run_with_hooks(
                corrupt_tokens, fwd_hooks=[(hn, make_hook(clean_cache, hn))])
        p_probs = torch.softmax(p[0, -1], dim=-1)
        attn_kl = kl_div(clean_probs, p_probs)
        attn_recovery = 1 - attn_kl / baseline_kl if baseline_kl > 0 else 0
        attn_logit = logit_diff(p) - corrupt_logit_val
        per_layer_attn.append({
            "layer": l,
            "recovery_kl": round(attn_recovery, 6),
            "logit_effect": round(attn_logit, 4),
        })
        del p

        # MLP
        hn = f"blocks.{l}.{hooks_info['mlp_patch_hook']}"
        with torch.no_grad():
            p = model.run_with_hooks(
                corrupt_tokens, fwd_hooks=[(hn, make_hook(clean_cache, hn))])
        p_probs = torch.softmax(p[0, -1], dim=-1)
        mlp_kl = kl_div(clean_probs, p_probs)
        mlp_recovery = 1 - mlp_kl / baseline_kl if baseline_kl > 0 else 0
        mlp_logit = logit_diff(p) - corrupt_logit_val
        per_layer_mlp.append({
            "layer": l,
            "recovery_kl": round(mlp_recovery, 6),
            "logit_effect": round(mlp_logit, 4),
        })
        del p

    # --- Cumulative: ALL attention ---
    hooks = [(f"blocks.{l}.{hooks_info['attn_patch_hook']}",
              make_hook(clean_cache, f"blocks.{l}.{hooks_info['attn_patch_hook']}"))
             for l in range(n_layers)]
    with torch.no_grad():
        p = model.run_with_hooks(corrupt_tokens, fwd_hooks=hooks)
    p_probs = torch.softmax(p[0, -1], dim=-1)
    cum_attn_kl = kl_div(clean_probs, p_probs)
    cum_attn_recovery = 1 - cum_attn_kl / baseline_kl if baseline_kl > 0 else 0
    cum_attn_logit = logit_diff(p) - corrupt_logit_val
    del p

    # --- Cumulative: ALL MLP ---
    hooks = [(f"blocks.{l}.{hooks_info['mlp_patch_hook']}",
              make_hook(clean_cache, f"blocks.{l}.{hooks_info['mlp_patch_hook']}"))
             for l in range(n_layers)]
    with torch.no_grad():
        p = model.run_with_hooks(corrupt_tokens, fwd_hooks=hooks)
    p_probs = torch.softmax(p[0, -1], dim=-1)
    cum_mlp_kl = kl_div(clean_probs, p_probs)
    cum_mlp_recovery = 1 - cum_mlp_kl / baseline_kl if baseline_kl > 0 else 0
    cum_mlp_logit = logit_diff(p) - corrupt_logit_val
    del p

    # --- Cumulative: ALL attention + ALL MLP ---
    hooks_both = (
        [(f"blocks.{l}.{hooks_info['attn_patch_hook']}",
          make_hook(clean_cache, f"blocks.{l}.{hooks_info['attn_patch_hook']}"))
         for l in range(n_layers)] +
        [(f"blocks.{l}.{hooks_info['mlp_patch_hook']}",
          make_hook(clean_cache, f"blocks.{l}.{hooks_info['mlp_patch_hook']}"))
         for l in range(n_layers)]
    )
    with torch.no_grad():
        p = model.run_with_hooks(corrupt_tokens, fwd_hooks=hooks_both)
    p_probs = torch.softmax(p[0, -1], dim=-1)
    cum_both_kl = kl_div(clean_probs, p_probs)
    cum_both_recovery = 1 - cum_both_kl / baseline_kl if baseline_kl > 0 else 0
    del p

    sum_attn = sum(r["recovery_kl"] for r in per_layer_attn)
    sum_mlp = sum(r["recovery_kl"] for r in per_layer_mlp)

    del clean_cache
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    return {
        "scenario": scenario["id"],
        "baseline_kl": round(baseline_kl, 6),
        "baseline_logit_range": round(baseline_logit_range, 4),
        "per_layer_attn": per_layer_attn,
        "per_layer_mlp": per_layer_mlp,
        "cumulative": {
            "attn_recovery": round(cum_attn_recovery, 6),
            "mlp_recovery": round(cum_mlp_recovery, 6),
            "both_recovery": round(cum_both_recovery, 6),
            "attn_logit": round(cum_attn_logit, 4),
            "mlp_logit": round(cum_mlp_logit, 4),
        },
        "sum_individual": {
            "attn": round(sum_attn, 6),
            "mlp": round(sum_mlp, 6),
        },
        "confound_test": {
            "attn_cumulative_minus_sum": round(cum_attn_recovery - sum_attn, 6),
            "mlp_cumulative_minus_sum": round(cum_mlp_recovery - sum_mlp, 6),
        },
    }


# ================================================================
# PRINT RESULTS
# ================================================================

def print_distributions(all_dist):
    print(f"\n{'='*72}")
    print("TEST 1: OUTPUT DISTRIBUTION DIVERGENCE")
    print("Do system prompts actually change what the model outputs?")
    print(f"{'='*72}")

    for dr in all_dist:
        print(f"\n  {dr['scenario']}:")
        for pair, c in dr["comparisons"].items():
            kl_str = f"KL={c['kl']:<8.4f}"
            js_str = f"JS={c['js']:<8.4f}"
            tv_str = f"TV={c['tv']:<6.4f}"
            t1 = "SAME" if c["top1_match"] else f"DIFF({c['top1_a']}|{c['top1_b']})"
            old = f"old_norm={c['old_norm_diff']:+.2f}"
            print(f"    {pair:<25} {kl_str} {js_str} {tv_str} top1={t1:<15} {old}")

    # Key comparisons
    hb_kls = [d["comparisons"]["handled_vs_baseline"]["kl"]
              for d in all_dist if "handled_vs_baseline" in d["comparisons"]]
    hs_kls = [d["comparisons"]["handled_vs_scrambled"]["kl"]
              for d in all_dist if "handled_vs_scrambled" in d["comparisons"]]

    if hb_kls:
        hb = bootstrap_ci(hb_kls)
        print(f"\n  handled vs baseline KL:  {hb['mean']:.4f} [{hb['ci_low']:.4f}, {hb['ci_high']:.4f}] (n={hb['n']})")
    if hs_kls:
        hs = bootstrap_ci(hs_kls)
        print(f"  handled vs scrambled KL: {hs['mean']:.4f} [{hs['ci_low']:.4f}, {hs['ci_high']:.4f}] (n={hs['n']})")

    if hb_kls and hs_kls:
        if np.mean(hb_kls) < 0.01:
            print(f"\n  >>> System prompt does NOT change output distribution (KL ≈ 0).")
            print(f"  >>> The activation-behavior gap is real: computation changes, output doesn't.")
        else:
            print(f"\n  >>> System prompt DOES change output distribution (KL > 0).")
        if hs_kls and np.mean(hs_kls) < 0.01:
            print(f"  >>> Handled ≈ scrambled at output level. The scrambled finding is REAL.")
        elif hs_kls:
            print(f"  >>> Handled ≠ scrambled at output level. Old norm metric MISSED a real difference!")


def print_dla(all_dla):
    print(f"\n{'='*72}")
    print("TEST 2: DIRECT LOGIT ATTRIBUTION (attention vs MLP)")
    print("Exact decomposition via unembedding projection. No patching confound.")
    print(f"{'='*72}")

    for dr in all_dla:
        print(f"\n  {dr['scenario']}:")
        for pair, d in dr["dla"].items():
            print(f"    {pair}:")
            print(f"      Total attn: {d['total_attn']:.2f}  "
                  f"Total MLP: {d['total_mlp']:.2f}  "
                  f"Attn fraction: {d['attn_fraction']:.1%}  "
                  f"Cosine(attn,mlp): {d['attn_mlp_cosine']:+.3f}")

            # Show top/bottom 3 layers by attention fraction
            by_attn = sorted(d["layers"], key=lambda x: x["attn_frac"], reverse=True)
            top3 = by_attn[:3]
            print(f"      Highest attn layers: " +
                  ", ".join(f"L{l['layer']}({l['attn_frac']:.0%})" for l in top3))

    # Aggregate
    for pair_name in ["handled_vs_baseline", "scrambled_vs_baseline", "handled_vs_scrambled"]:
        fracs = [d["dla"][pair_name]["attn_fraction"]
                 for d in all_dla if pair_name in d["dla"]]
        if fracs:
            ci = bootstrap_ci(fracs)
            print(f"\n  {pair_name} attention fraction: "
                  f"{ci['mean']:.1%} [{ci['ci_low']:.1%}, {ci['ci_high']:.1%}] (n={ci['n']})")

    hb_fracs = [d["dla"]["handled_vs_baseline"]["attn_fraction"]
                for d in all_dla if "handled_vs_baseline" in d["dla"]]
    if hb_fracs:
        mean_frac = np.mean(hb_fracs)
        if mean_frac > 0.05:
            print(f"\n  >>> ATTENTION FRACTION = {mean_frac:.1%} — NOT zero.")
            print(f"  >>> The '100% MLP / 0% attention' claim from patch_all_layers.py")
            print(f"  >>> was a measurement artifact. DLA shows attention contributes.")
        else:
            print(f"\n  >>> Attention fraction = {mean_frac:.1%} — effectively zero.")
            print(f"  >>> The MLP-only finding is CONFIRMED by DLA decomposition.")


def print_signal(all_signal):
    print(f"\n{'='*72}")
    print("TEST 3: SIGNAL EVOLUTION (what does the condition difference mean?)")
    print("Tracks how the system prompt signal changes through layers.")
    print(f"{'='*72}")

    for dr in all_signal:
        print(f"\n  {dr['scenario']}:")
        for pair, layers in dr["signal"].items():
            print(f"    {pair}:")
            # Show every 4th layer + last
            n = len(layers)
            step = max(1, n // 8)
            for l in layers:
                if l["layer"] % step == 0 or l["layer"] == n - 1:
                    pushed = ", ".join(t[0].strip() for t in l["pushed"][:3])
                    suppressed = ", ".join(t[0].strip() for t in l["suppressed"][:3])
                    print(f"      L{l['layer']:<4} mag={l['magnitude']:<8.2f} "
                          f"push=[{pushed}]  suppress=[{suppressed}]")

    # Does the signal die?
    for pair_name in ["handled_vs_baseline"]:
        first_mags = []
        last_mags = []
        for dr in all_signal:
            if pair_name in dr["signal"]:
                layers = dr["signal"][pair_name]
                if layers:
                    first_mags.append(layers[0]["magnitude"])
                    last_mags.append(layers[-1]["magnitude"])
        if first_mags and last_mags:
            first_ci = bootstrap_ci(first_mags)
            last_ci = bootstrap_ci(last_mags)
            ratio = last_ci["mean"] / first_ci["mean"] if first_ci["mean"] > 0 else 0
            print(f"\n  {pair_name} signal magnitude:")
            print(f"    First layer: {first_ci['mean']:.2f} [{first_ci['ci_low']:.2f}, {first_ci['ci_high']:.2f}]")
            print(f"    Last layer:  {last_ci['mean']:.2f} [{last_ci['ci_low']:.2f}, {last_ci['ci_high']:.2f}]")
            print(f"    Ratio (last/first): {ratio:.2f}")


def print_patching(all_patching):
    print(f"\n{'='*72}")
    print("TEST 4: CUMULATIVE PATCHING (confound test)")
    print("Does patching ALL attention simultaneously show more effect")
    print("than the sum of individual layer patches?")
    print(f"{'='*72}")

    for pr in all_patching:
        if pr.get("skipped"):
            continue
        print(f"\n  {pr['scenario']}:")
        print(f"    Baseline KL (handled vs baseline): {pr['baseline_kl']:.4f}")

        print(f"\n    {'Component':<15} {'Cumulative':>12} {'Sum individual':>15} {'Difference':>12}")
        print(f"    {'-'*56}")

        ca = pr["cumulative"]["attn_recovery"]
        cm = pr["cumulative"]["mlp_recovery"]
        cb = pr["cumulative"]["both_recovery"]
        sa = pr["sum_individual"]["attn"]
        sm = pr["sum_individual"]["mlp"]

        print(f"    {'Attention':<15} {ca:>+12.4f} {sa:>+15.4f} {ca - sa:>+12.4f}")
        print(f"    {'MLP':<15} {cm:>+12.4f} {sm:>+15.4f} {cm - sm:>+12.4f}")
        print(f"    {'Both':<15} {cb:>+12.4f}")

        # Per-layer detail for attention (show layers with highest recovery)
        top_attn = sorted(pr["per_layer_attn"],
                          key=lambda x: abs(x["recovery_kl"]), reverse=True)[:5]
        if any(abs(x["recovery_kl"]) > 0.001 for x in top_attn):
            print(f"\n    Top attention layers by recovery:")
            for la in top_attn:
                if abs(la["recovery_kl"]) > 0.001:
                    print(f"      L{la['layer']}: recovery={la['recovery_kl']:+.4f}  "
                          f"logit_effect={la['logit_effect']:+.4f}")

    # Aggregate confound test
    cum_attn_vals = [p["cumulative"]["attn_recovery"]
                     for p in all_patching if not p.get("skipped")]
    sum_attn_vals = [p["sum_individual"]["attn"]
                     for p in all_patching if not p.get("skipped")]
    cum_mlp_vals = [p["cumulative"]["mlp_recovery"]
                    for p in all_patching if not p.get("skipped")]

    if cum_attn_vals:
        cum_ci = bootstrap_ci(cum_attn_vals)
        sum_ci = bootstrap_ci(sum_attn_vals)
        print(f"\n  AGGREGATE (n={cum_ci['n']}):")
        print(f"    Cumulative attention: {cum_ci['mean']:+.4f} "
              f"[{cum_ci['ci_low']:+.4f}, {cum_ci['ci_high']:+.4f}]")
        print(f"    Sum individual attn:  {sum_ci['mean']:+.4f} "
              f"[{sum_ci['ci_low']:+.4f}, {sum_ci['ci_high']:+.4f}]")

        if cum_ci["mean"] > sum_ci["mean"] * 1.1 and cum_ci["ci_low"] > sum_ci["ci_high"]:
            print(f"\n  >>> CONFOUND CONFIRMED.")
            print(f"  >>> Cumulative attention recovery significantly exceeds sum of individual.")
            print(f"  >>> Attention contributes indirectly through downstream MLPs.")
            print(f"  >>> The '0% attention' finding is a measurement artifact.")
        elif abs(cum_ci["mean"] - sum_ci["mean"]) < 0.01:
            print(f"\n  >>> No confound detected. Cumulative ≈ sum of individual.")
            print(f"  >>> Single-layer patching was not misleading.")
        else:
            print(f"\n  >>> Ambiguous. Cumulative differs from sum but CIs overlap.")
            print(f"  >>> More scenarios needed for definitive conclusion.")


# ================================================================
# MAIN
# ================================================================

def main():
    parser = argparse.ArgumentParser(description="Diagnostic measurement suite")
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--scenarios", type=int, default=None,
                        help="Limit number of scenarios")
    parser.add_argument("--skip-patching", action="store_true",
                        help="Skip cumulative patching (faster)")
    args = parser.parse_args()

    scenarios = load_scenarios()
    if args.scenarios:
        scenarios = scenarios[:args.scenarios]

    print(f"Model: {args.model}")
    print(f"Scenarios: {len(scenarios)}")
    print(f"Conditions: {list(CONDITIONS.keys())}")
    print(f"Loading model...")

    model = HookedTransformer.from_pretrained(
        args.model, device="mps", dtype=torch.float16)
    n_layers = model.cfg.n_layers
    print(f"Loaded. Layers: {n_layers}, Heads: {model.cfg.n_heads}, "
          f"d_model: {model.cfg.d_model}")

    hooks_info = detect_available_hooks(model)

    all_dist = []
    all_dla = []
    all_signal = []
    all_patching = []

    for si, scenario in enumerate(scenarios):
        print(f"\n{'#'*72}")
        print(f"# SCENARIO {si+1}/{len(scenarios)}: {scenario['id']}")
        print(f"# Prompt: {scenario['prompt'][:60]}...")
        print(f"{'#'*72}")

        # Phase 1: Run all conditions with cache
        print(f"\n  Running {len(CONDITIONS)} conditions...")
        caches = {}
        for cond_id, sp in CONDITIONS.items():
            prompt = build_prompt(model, sp, scenario["prompt"])
            tokens = model.to_tokens(prompt)
            if tokens.shape[1] > 512:
                tokens = tokens[:, :512]
            with torch.no_grad():
                logits, cache = model.run_with_cache(tokens)
            caches[cond_id] = {"tokens": tokens, "logits": logits, "cache": cache}
            print(f"    {cond_id}: {tokens.shape[1]} tokens")

        # Test 1: Output distributions
        print(f"\n  Test 1: Output distributions...")
        dist_result = test_distributions(model, scenario, caches)
        all_dist.append(dist_result)
        for pair, c in dist_result["comparisons"].items():
            print(f"    {pair}: KL={c['kl']:.4f}  top1={'SAME' if c['top1_match'] else 'DIFF'}")

        # Test 2: DLA
        print(f"\n  Test 2: Direct Logit Attribution...")
        dla_result = test_dla(model, scenario, caches, hooks_info)
        all_dla.append(dla_result)
        for pair, d in dla_result["dla"].items():
            print(f"    {pair}: attn={d['attn_fraction']:.1%}  mlp={d['mlp_fraction']:.1%}")

        # Test 3: Signal evolution
        print(f"\n  Test 3: Signal evolution...")
        signal_result = test_signal_evolution(model, scenario, caches)
        all_signal.append(signal_result)

        # Clean up main caches
        for cid in caches:
            del caches[cid]["cache"]
            del caches[cid]["logits"]
        del caches
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

        # Test 4: Cumulative patching
        if not args.skip_patching:
            print(f"\n  Test 4: Cumulative patching ({n_layers} layers × 2 components)...")
            patch_result = test_cumulative_patching(model, scenario, hooks_info)
            all_patching.append(patch_result)
            if not patch_result.get("skipped"):
                ca = patch_result["cumulative"]["attn_recovery"]
                cm = patch_result["cumulative"]["mlp_recovery"]
                sa = patch_result["sum_individual"]["attn"]
                print(f"    Cumulative attn: {ca:+.4f}  sum individual: {sa:+.4f}  "
                      f"diff: {ca - sa:+.4f}")
                print(f"    Cumulative MLP:  {cm:+.4f}")
        else:
            all_patching.append({"scenario": scenario["id"], "skipped": True})

    # ============================================================
    # AGGREGATE RESULTS
    # ============================================================
    print(f"\n\n{'='*72}")
    print("DIAGNOSTIC RESULTS SUMMARY")
    print(f"Model: {args.model}")
    print(f"Scenarios: {len(scenarios)}")
    print(f"{'='*72}")

    print_distributions(all_dist)
    print_dla(all_dla)
    print_signal(all_signal)
    if not args.skip_patching:
        print_patching(all_patching)

    # Save
    DATA_DIR.mkdir(exist_ok=True)
    model_tag = args.model.replace("/", "_")
    out = DATA_DIR / f"diagnose_{model_tag}.json"
    results = {
        "model": args.model,
        "n_layers": n_layers,
        "n_heads": model.cfg.n_heads,
        "d_model": model.cfg.d_model,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_scenarios": len(scenarios),
        "distributions": all_dist,
        "dla": all_dla,
        "signal_evolution": all_signal,
        "cumulative_patching": all_patching,
    }
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
