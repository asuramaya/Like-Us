"""
Analyze the existing 3B neuron matrix to answer critical questions
before running the full pipeline.

Q1: Does scientific_method show the same two-band pattern as handled?
Q2: Does similar_work show the same two-band pattern?
Q3: Do failure scenarios show consistent activation inversion?
Q4: What do the ablation profiles actually look like?
"""

import json
import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).parent / "neuron_data"
MATRIX_PATH = DATA_DIR / "matrix_Qwen_Qwen2.5-3B-Instruct.json"


def load_matrix():
    with open(MATRIX_PATH) as f:
        return json.load(f)


def compute_condition_profile(matrix, condition):
    """Compute average layer-by-layer diff vs baseline for a condition."""
    diff_key = f"{condition}_vs_baseline"
    n_layers = matrix["n_layers"]
    layer_agg = {l: {"resid_norm": [], "mlp_norm": [], "attn_entropy_mean": []}
                 for l in range(n_layers)}

    for s in matrix["scenarios"]:
        if diff_key in s["diffs"]:
            for ld in s["diffs"][diff_key]:
                l = ld["layer"]
                for metric in ["resid_norm", "mlp_norm", "attn_entropy_mean"]:
                    if metric in ld:
                        layer_agg[l][metric].append(ld[metric])

    profile = []
    for l in range(n_layers):
        row = {"layer": l}
        for metric in ["resid_norm", "mlp_norm", "attn_entropy_mean"]:
            vals = layer_agg[l][metric]
            row[metric] = np.mean(vals) if vals else 0.0
        profile.append(row)
    return profile


def classify_band(profile, metric="resid_norm"):
    """Check if a profile shows the two-band pattern:
    mid-layer suppression (negative) + late-layer amplification (positive)."""
    mid_layers = [p[metric] for p in profile if 10 <= p["layer"] <= 29]
    late_layers = [p[metric] for p in profile if 30 <= p["layer"] <= 35]

    mid_mean = np.mean(mid_layers) if mid_layers else 0
    late_mean = np.mean(late_layers) if late_layers else 0

    has_mid_suppression = mid_mean < -1.0
    has_late_amplification = late_mean > 1.0

    return {
        "mid_mean": mid_mean,
        "late_mean": late_mean,
        "has_mid_suppression": has_mid_suppression,
        "has_late_amplification": has_late_amplification,
        "two_band": has_mid_suppression and has_late_amplification,
    }


def per_scenario_profile(matrix, condition, scenario_id):
    """Get the diff profile for a specific scenario and condition."""
    diff_key = f"{condition}_vs_baseline"
    for s in matrix["scenarios"]:
        if s["id"] == scenario_id and diff_key in s["diffs"]:
            return s["diffs"][diff_key]
    return None


def main():
    print("Loading matrix...")
    matrix = load_matrix()
    n_layers = matrix["n_layers"]
    scenarios = [s["id"] for s in matrix["scenarios"]]
    print(f"Model: {matrix['model']}, Layers: {n_layers}")
    print(f"Scenarios: {len(scenarios)}")

    # Available conditions (from diffs keys)
    all_diffs = set()
    for s in matrix["scenarios"]:
        for k in s["diffs"]:
            if k.endswith("_vs_baseline"):
                all_diffs.add(k.replace("_vs_baseline", ""))
    print(f"Conditions with baseline diffs: {sorted(all_diffs)}")

    # ================================================================
    # Q1 + Q2: Does the two-band pattern appear in ALL conditions?
    # ================================================================
    print("\n" + "=" * 70)
    print("Q1/Q2: TWO-BAND PATTERN CHECK (all conditions vs baseline)")
    print("=" * 70)

    conditions = sorted(all_diffs)
    for cond in conditions:
        profile = compute_condition_profile(matrix, cond)
        band = classify_band(profile, "resid_norm")
        marker = "YES" if band["two_band"] else "NO "
        print(f"  [{marker}] {cond:<30} mid={band['mid_mean']:>+8.2f}  late={band['late_mean']:>+8.2f}")

    # Detailed layer-by-layer for the three main conditions
    print("\n" + "-" * 70)
    print("LAYER-BY-LAYER RESIDUAL NORM DIFF (averaged across all scenarios)")
    print("-" * 70)

    main_conditions = [c for c in ["handled", "scientific_method", "similar_work"]
                       if c in all_diffs]
    profiles = {c: compute_condition_profile(matrix, c) for c in main_conditions}

    header = f"{'Layer':<8}"
    for c in main_conditions:
        header += f" {c:>20}"
    print(header)
    print("-" * (8 + 21 * len(main_conditions)))

    for l in range(n_layers):
        row = f"L{l:<6}"
        for c in main_conditions:
            val = profiles[c][l]["resid_norm"]
            row += f" {val:>+20.3f}"
        print(row)

    # ================================================================
    # Q3: Failure scenario inversion check
    # ================================================================
    print("\n" + "=" * 70)
    print("Q3: FAILURE SCENARIO INVERSION CHECK")
    print("Known behavioral failures: compulsivity_checking_return,")
    print("uncertainty_distress_rule_hunger, authority_delegation")
    print("=" * 70)

    failure_scenarios = [
        "compulsivity_checking_return",
        "uncertainty_distress_rule_hunger",
        "authority_delegation",
    ]
    success_scenarios = [s for s in scenarios if s not in failure_scenarios]

    for cond in ["handled", "only_artifact", "only_identity"]:
        if cond not in all_diffs:
            continue
        print(f"\n  --- {cond} vs baseline ---")
        print(f"  {'Scenario':<40} {'L33-35 resid':>14} {'Pattern':>10}")

        for sid in scenarios:
            diff = per_scenario_profile(matrix, cond, sid)
            if diff:
                late_vals = [d.get("resid_norm", 0) for d in diff if d["layer"] >= 33]
                late_mean = np.mean(late_vals) if late_vals else 0
                is_failure = sid in failure_scenarios
                marker = "FAIL" if is_failure else "ok"
                direction = "INVERTED" if late_mean < 0 else "normal"
                print(f"  {sid:<40} {late_mean:>+14.2f} {direction:>10}  [{marker}]")

    # ================================================================
    # Q4: Ablation profiles (vs handled, not vs baseline)
    # ================================================================
    print("\n" + "=" * 70)
    print("Q4: ABLATION PROFILES (each condition vs HANDLED)")
    print("=" * 70)

    ablation_conds = [c for c in all_diffs
                      if c.startswith("handled_minus_") or c.startswith("only_")]

    # Check which diffs exist vs handled
    handled_diffs = set()
    for s in matrix["scenarios"]:
        for k in s["diffs"]:
            if k.endswith("_vs_handled"):
                handled_diffs.add(k.replace("_vs_handled", ""))

    for cond in sorted(handled_diffs):
        diff_key = f"{cond}_vs_handled"
        layer_agg = {l: {"resid_norm": [], "mlp_norm": []} for l in range(n_layers)}
        for s in matrix["scenarios"]:
            if diff_key in s["diffs"]:
                for ld in s["diffs"][diff_key]:
                    l = ld["layer"]
                    for metric in ["resid_norm", "mlp_norm"]:
                        if metric in ld:
                            layer_agg[l][metric].append(ld[metric])

        mid_resid = np.mean([np.mean(layer_agg[l]["resid_norm"])
                             for l in range(10, 30)
                             if layer_agg[l]["resid_norm"]])
        late_resid = np.mean([np.mean(layer_agg[l]["resid_norm"])
                              for l in range(30, 36)
                              if layer_agg[l]["resid_norm"]])

        print(f"  {cond:<30} mid_resid={mid_resid:>+8.2f}  late_resid={late_resid:>+8.2f}")

    # ================================================================
    # Summary
    # ================================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    handled_band = classify_band(profiles.get("handled", []), "resid_norm") if "handled" in profiles else {}
    sci_band = classify_band(profiles.get("scientific_method", []), "resid_norm") if "scientific_method" in profiles else {}
    sim_band = classify_band(profiles.get("similar_work", []), "resid_norm") if "similar_work" in profiles else {}

    print(f"\n  Handled two-band:          {handled_band.get('two_band', 'N/A')}")
    print(f"  Scientific method two-band: {sci_band.get('two_band', 'N/A')}")
    print(f"  Similar work two-band:      {sim_band.get('two_band', 'N/A')}")

    if all(b.get("two_band") for b in [handled_band, sci_band, sim_band] if b):
        print("\n  >>> ALL conditions show two-band pattern.")
        print("  >>> The pattern is NOT specific to the handling intervention.")
        print("  >>> It's a generic effect of directive system prompts vs baseline.")
        print("  >>> The mechanistic narrative in the paper needs rewriting.")
    elif handled_band.get("two_band") and not sci_band.get("two_band"):
        print("\n  >>> Only handled shows two-band pattern.")
        print("  >>> The pattern IS specific to the handling content.")
        print("  >>> The mechanistic narrative holds.")
    else:
        print("\n  >>> Mixed results. Need closer examination.")


if __name__ == "__main__":
    main()
