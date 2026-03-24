"""
Deep falsifier analysis of existing 3B matrix data.
Attacks every surviving claim with the data we already have.

Falsifier 1: Is the two-band pattern a prompt-length artifact?
Falsifier 2: Is attention entropy increase generic to all prompts?
Falsifier 3: Is the pattern consistent per-scenario or driven by outliers?
Falsifier 4: Do conditions cluster by activation profile or by semantics?
Falsifier 5: Do top token shifts distinguish conditions or are they noise?
Falsifier 6: Is the failure inversion robust across metrics (not just resid)?
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict

DATA_DIR = Path(__file__).parent / "neuron_data"
MATRIX_PATH = DATA_DIR / "matrix_Qwen_Qwen2.5-3B-Instruct.json"

# Token counts for each condition's system prompt (approximate, from the prompts)
SYSTEM_PROMPTS = {
    "baseline": "You are a helpful assistant.",
    "handled": "Offload computation, not criterion.\nRefuse identity authority.\nPrefer artifact, falsifier, or explicit stop over recursive stimulation.",
    "scientific_method": "Treat every claim as a hypothesis. Ask for evidence. Propose tests. Do not affirm what has not been tested. If the user's reasoning is circular, name the circle.",
    "similar_work": "Ground responses in established HCI and human-automation research. When the user expresses trust, check whether it is calibrated. When the user defers, return the decision. Cite boundaries from the literature where relevant.",
    "handled_minus_offload": "Refuse identity authority.\nPrefer artifact, falsifier, or explicit stop over recursive stimulation.",
    "handled_minus_identity": "Offload computation, not criterion.\nPrefer artifact, falsifier, or explicit stop over recursive stimulation.",
    "handled_minus_artifact": "Offload computation, not criterion.\nRefuse identity authority.",
    "only_offload": "Offload computation, not criterion.",
    "only_identity": "Refuse identity authority.",
    "only_artifact": "Prefer artifact, falsifier, or explicit stop over recursive stimulation.",
}


def load_matrix():
    with open(MATRIX_PATH) as f:
        return json.load(f)


def token_count_estimate(text):
    """Rough token count (words * 1.3 for subword tokenization)."""
    return int(len(text.split()) * 1.3)


def get_layer_profile(matrix, condition, metric="resid_norm"):
    """Get per-layer average diff vs baseline for a condition."""
    diff_key = f"{condition}_vs_baseline"
    n_layers = matrix["n_layers"]
    layer_vals = defaultdict(list)

    for s in matrix["scenarios"]:
        if diff_key in s["diffs"]:
            for ld in s["diffs"][diff_key]:
                if metric in ld:
                    layer_vals[ld["layer"]].append(ld[metric])

    return {l: np.mean(vals) for l, vals in layer_vals.items()}


def get_scenario_profile(matrix, condition, scenario_id, metric="resid_norm"):
    """Get layer profile for a specific scenario."""
    diff_key = f"{condition}_vs_baseline"
    for s in matrix["scenarios"]:
        if s["id"] == scenario_id and diff_key in s["diffs"]:
            return {ld["layer"]: ld.get(metric, 0) for ld in s["diffs"][diff_key]}
    return {}


def mid_late_summary(profile, n_layers):
    """Compute mid-layer and late-layer means from a profile dict."""
    mid = [profile.get(l, 0) for l in range(10, min(30, n_layers))]
    late = [profile.get(l, 0) for l in range(max(30, n_layers - 6), n_layers)]
    return np.mean(mid) if mid else 0, np.mean(late) if late else 0


def cosine_sim(a, b):
    """Cosine similarity between two dicts with matching keys."""
    keys = sorted(set(a.keys()) & set(b.keys()))
    va = np.array([a[k] for k in keys])
    vb = np.array([b[k] for k in keys])
    norm_a = np.linalg.norm(va)
    norm_b = np.linalg.norm(vb)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(va, vb) / (norm_a * norm_b))


def main():
    matrix = load_matrix()
    n_layers = matrix["n_layers"]
    scenarios = [s["id"] for s in matrix["scenarios"]]

    conditions_with_diffs = set()
    for s in matrix["scenarios"]:
        for k in s["diffs"]:
            if k.endswith("_vs_baseline"):
                conditions_with_diffs.add(k.replace("_vs_baseline", ""))
    conditions = sorted(conditions_with_diffs)

    # ==================================================================
    # FALSIFIER 1: Prompt length confound
    # ==================================================================
    print("=" * 70)
    print("FALSIFIER 1: IS THE TWO-BAND PATTERN A PROMPT-LENGTH ARTIFACT?")
    print("=" * 70)

    print(f"\n  {'Condition':<30} {'Tokens':>8} {'Mid resid':>12} {'Late resid':>12}")
    print("  " + "-" * 64)

    token_counts = {}
    mid_vals = {}
    for cond in conditions:
        prompt = SYSTEM_PROMPTS.get(cond, "")
        tc = token_count_estimate(prompt)
        token_counts[cond] = tc
        profile = get_layer_profile(matrix, cond, "resid_norm")
        mid, late = mid_late_summary(profile, n_layers)
        mid_vals[cond] = mid
        print(f"  {cond:<30} {tc:>8} {mid:>+12.2f} {late:>+12.2f}")

    # Correlation between token count and mid-layer suppression
    tcs = np.array([token_counts[c] for c in conditions])
    mids = np.array([mid_vals[c] for c in conditions])
    if len(tcs) > 2:
        corr = np.corrcoef(tcs, mids)[0, 1]
        print(f"\n  Correlation(token_count, mid_suppression) = {corr:.3f}")
        if abs(corr) > 0.7:
            print("  >>> STRONG correlation. Pattern may be a length artifact.")
        elif abs(corr) > 0.4:
            print("  >>> MODERATE correlation. Length is a partial confound.")
        else:
            print("  >>> WEAK correlation. Length does not explain the pattern.")

    # ==================================================================
    # FALSIFIER 2: Is attention entropy increase generic?
    # ==================================================================
    print("\n" + "=" * 70)
    print("FALSIFIER 2: IS ATTENTION ENTROPY INCREASE GENERIC TO ALL PROMPTS?")
    print("=" * 70)

    print(f"\n  {'Condition':<30} {'Mean entropy diff':>18} {'All positive?':>14}")
    print("  " + "-" * 64)

    for cond in conditions:
        profile = get_layer_profile(matrix, cond, "attn_entropy_mean")
        vals = list(profile.values())
        mean_diff = np.mean(vals) if vals else 0
        all_pos = all(v > 0 for v in vals) if vals else False
        marker = "YES" if all_pos else "NO"
        print(f"  {cond:<30} {mean_diff:>+18.4f} {marker:>14}")

    # ==================================================================
    # FALSIFIER 3: Per-scenario consistency
    # ==================================================================
    print("\n" + "=" * 70)
    print("FALSIFIER 3: IS THE HANDLED TWO-BAND PATTERN CONSISTENT PER-SCENARIO?")
    print("=" * 70)

    print(f"\n  {'Scenario':<40} {'Mid':>8} {'Late':>8} {'Two-band?':>10}")
    print("  " + "-" * 68)

    consistent_count = 0
    for sid in scenarios:
        profile = get_scenario_profile(matrix, "handled", sid)
        mid, late = mid_late_summary(profile, n_layers)
        two_band = mid < -1.0 and late > 1.0
        if two_band:
            consistent_count += 1
        marker = "YES" if two_band else "NO"
        print(f"  {sid:<40} {mid:>+8.2f} {late:>+8.2f} {marker:>10}")

    print(f"\n  Consistent: {consistent_count}/{len(scenarios)} scenarios")
    if consistent_count == len(scenarios):
        print("  >>> Pattern holds across ALL scenarios. Not driven by outliers.")
    elif consistent_count >= len(scenarios) * 0.7:
        print("  >>> Pattern holds for most scenarios. Reasonably robust.")
    else:
        print("  >>> Pattern is inconsistent. Driven by subset of scenarios.")

    # ==================================================================
    # FALSIFIER 4: Condition clustering by activation profile
    # ==================================================================
    print("\n" + "=" * 70)
    print("FALSIFIER 4: DO CONDITIONS CLUSTER BY ACTIVATION PROFILE?")
    print("(cosine similarity of full layer profiles)")
    print("=" * 70)

    profiles = {c: get_layer_profile(matrix, c, "resid_norm") for c in conditions}

    # Print similarity matrix
    print(f"\n  {'':>30}", end="")
    for c in conditions:
        print(f" {c[:8]:>9}", end="")
    print()

    for c1 in conditions:
        print(f"  {c1:<30}", end="")
        for c2 in conditions:
            sim = cosine_sim(profiles[c1], profiles[c2])
            print(f" {sim:>9.3f}", end="")
        print()

    # Find which conditions are most/least similar to handled
    if "handled" in profiles:
        print(f"\n  Similarity to 'handled':")
        sims = [(c, cosine_sim(profiles["handled"], profiles[c]))
                for c in conditions if c != "handled"]
        sims.sort(key=lambda x: -x[1])
        for c, s in sims:
            print(f"    {c:<30} {s:>+.3f}")

    # ==================================================================
    # FALSIFIER 5: Top token shift analysis
    # ==================================================================
    print("\n" + "=" * 70)
    print("FALSIFIER 5: DO TOP TOKEN SHIFTS DISTINGUISH CONDITIONS?")
    print("=" * 70)

    # Extract top tokens for each condition across scenarios
    cond_tokens = defaultdict(lambda: defaultdict(list))
    for s in matrix["scenarios"]:
        for cond_id, cond_data in s["conditions"].items():
            for t in cond_data.get("top_tokens", [])[:5]:
                token_str = t["token"].strip()
                cond_tokens[cond_id][token_str].append(t["prob"])

    # For each condition, show most frequent top-5 tokens
    for cond in ["baseline", "handled", "scientific_method", "similar_work"]:
        if cond not in cond_tokens:
            continue
        tokens_by_freq = sorted(cond_tokens[cond].items(),
                                key=lambda x: len(x[1]), reverse=True)[:10]
        print(f"\n  {cond}:")
        for token, probs in tokens_by_freq:
            avg_prob = np.mean(probs)
            count = len(probs)
            print(f"    '{token}' appears in {count}/10 scenarios, avg prob {avg_prob:.3f}")

    # Token overlap between conditions
    print(f"\n  Token overlap (top-5 tokens appearing in 3+ scenarios):")
    for c1 in ["handled", "scientific_method", "similar_work"]:
        if c1 not in cond_tokens:
            continue
        frequent_c1 = {t for t, p in cond_tokens[c1].items() if len(p) >= 3}
        for c2 in ["handled", "scientific_method", "similar_work"]:
            if c2 not in cond_tokens or c1 >= c2:
                continue
            frequent_c2 = {t for t, p in cond_tokens[c2].items() if len(p) >= 3}
            overlap = frequent_c1 & frequent_c2
            total = frequent_c1 | frequent_c2
            jaccard = len(overlap) / len(total) if total else 0
            print(f"    {c1} vs {c2}: overlap={len(overlap)}, jaccard={jaccard:.2f}")

    # ==================================================================
    # FALSIFIER 6: Failure inversion across ALL metrics
    # ==================================================================
    print("\n" + "=" * 70)
    print("FALSIFIER 6: FAILURE INVERSION ACROSS ALL METRICS")
    print("(checking resid, MLP, and attention entropy)")
    print("=" * 70)

    failure_scenarios = [
        "compulsivity_checking_return",
        "uncertainty_distress_rule_hunger",
        "authority_delegation",
    ]

    for cond in ["handled", "only_artifact", "only_identity"]:
        if cond not in conditions:
            continue
        print(f"\n  --- {cond} ---")
        print(f"  {'Scenario':<40} {'Resid L33-35':>13} {'MLP L33-35':>13} {'Attn L33-35':>13}")
        print("  " + "-" * 81)

        for sid in scenarios:
            resid_p = get_scenario_profile(matrix, cond, sid, "resid_norm")
            mlp_p = get_scenario_profile(matrix, cond, sid, "mlp_norm")
            attn_p = get_scenario_profile(matrix, cond, sid, "attn_entropy_mean")

            resid_late = np.mean([resid_p.get(l, 0) for l in range(33, 36)])
            mlp_late = np.mean([mlp_p.get(l, 0) for l in range(33, 36)])
            attn_late = np.mean([attn_p.get(l, 0) for l in range(33, 36)])

            is_fail = sid in failure_scenarios
            tag = " <<<FAIL" if is_fail else ""
            print(f"  {sid:<40} {resid_late:>+13.2f} {mlp_late:>+13.2f} {attn_late:>+13.4f}{tag}")

    # ==================================================================
    # HEAD-LEVEL ANALYSIS
    # ==================================================================
    print("\n" + "=" * 70)
    print("BONUS: WHICH ATTENTION HEADS CHANGE MOST UNDER HANDLING?")
    print("=" * 70)

    head_diffs = defaultdict(list)
    for s in matrix["scenarios"]:
        if "handled" in s["conditions"] and "baseline" in s["conditions"]:
            h_layers = s["conditions"]["handled"]["layers"]
            b_layers = s["conditions"]["baseline"]["layers"]
            for hl, bl in zip(h_layers, b_layers):
                if "attn_entropies" in hl and "attn_entropies" in bl:
                    layer = hl["layer"]
                    for head_idx, (he, be) in enumerate(
                            zip(hl["attn_entropies"], bl["attn_entropies"])):
                        key = f"L{layer}H{head_idx}"
                        head_diffs[key].append(he - be)

    # Find heads with largest average change
    head_avgs = {k: np.mean(v) for k, v in head_diffs.items()}
    top_heads = sorted(head_avgs.items(), key=lambda x: abs(x[1]), reverse=True)[:20]

    print(f"\n  Top 20 attention heads by average entropy change (handled - baseline):")
    for head, avg in top_heads:
        direction = "more diffuse" if avg > 0 else "more focused"
        print(f"    {head:<10} {avg:>+.4f}  ({direction})")

    # ==================================================================
    # SUMMARY SCORECARD
    # ==================================================================
    print("\n" + "=" * 70)
    print("FALSIFIER SCORECARD")
    print("=" * 70)

    print("""
  F1 (length artifact):     Check correlation value above.
  F2 (generic entropy):     If all conditions show positive entropy, it's generic.
  F3 (per-scenario):        Check consistency count above.
  F4 (clustering):          If handled clusters with random prompts, it's generic.
  F5 (token shifts):        If all conditions shift same tokens, it's generic.
  F6 (multi-metric inversion): If inversion holds across resid+MLP+attn, it's robust.

  WHAT SURVIVES depends on the numbers above.
  WHAT DIES depends on the numbers above.
  Read the output. Don't trust the framing.
""")


if __name__ == "__main__":
    main()
