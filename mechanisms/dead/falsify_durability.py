"""
Falsify the durability finding.

The claim: only_artifact is uniquely durable because of its content.
The alternative hypotheses to test:

F1: It's just prompt length. Shorter prompts decay slower because they're
    a smaller fraction of growing context. Test: compare only_artifact (11 words)
    against empty_directive (2 words) and other short prompts.

F2: It's the filler, not the condition. The filler pairs are the same for
    all conditions. Maybe the filler interacts differently with short vs long
    system prompts. Test: measure absolute activation values, not just diffs.
    If all conditions converge to the same absolute values, the filler is
    dominating and the system prompt effect is just being diluted uniformly.

F3: The metric is wrong. "Signature strength" = late_diff - mid_diff assumes
    the two-band pattern IS the signal. But we already showed the two-band
    pattern is vocabulary-driven. Maybe durability is measuring vocabulary
    dilution rate, not intervention effectiveness.

F4: Baseline is drifting. If baseline activations change across turns (they do —
    context grows), then the "diff vs baseline" is measuring baseline drift,
    not condition stability. Test: look at absolute values for each condition
    independently.

F5: The convergence is trivial. All conditions might converge to the same
    absolute activation profile as context dominates. The "durability" is just
    how fast each condition converges to the context-dominated profile.
    Short prompts converge slower because they start closer.
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict

DATA_DIR = Path(__file__).parent / "neuron_data"


def load_extended_degradation():
    path = DATA_DIR / "degradation_extended_Qwen_Qwen2.5-3B-Instruct.json"
    if not path.exists():
        # Try the basic degradation
        path = DATA_DIR / "degradation_Qwen_Qwen2.5-3B-Instruct.json"
    with open(path) as f:
        return json.load(f)


def main():
    data = load_extended_degradation()
    results = data.get("results", data.get("scenarios", []))

    # System prompt approximate word counts
    prompt_lengths = {
        "baseline": 5,
        "handled": 20,
        "scrambled": 16,
        "reversed": 25,
        "scientific_method": 36,
        "similar_work": 41,
        "only_artifact": 11,
        "safety_only": 24,
    }

    # ================================================================
    # F1: Is durability just prompt length?
    # ================================================================
    print("=" * 70)
    print("F1: IS DURABILITY JUST PROMPT LENGTH?")
    print("=" * 70)

    # Compute T4 retention for each condition
    retentions = {}
    for cond in prompt_lengths:
        if cond == "baseline":
            continue
        t0_sigs = []
        t4_sigs = []
        for s in results:
            cond_turns = s["conditions"].get(cond, [])
            base_turns = s["conditions"].get("baseline", [])
            if len(cond_turns) > 4 and len(base_turns) > 4:
                ct0, bt0 = cond_turns[0], base_turns[0]
                ct4, bt4 = cond_turns[4], base_turns[4]
                sig0 = (ct0["late_resid"] - bt0["late_resid"]) - (ct0["mid_resid"] - bt0["mid_resid"])
                sig4 = (ct4["late_resid"] - bt4["late_resid"]) - (ct4["mid_resid"] - bt4["mid_resid"])
                t0_sigs.append(sig0)
                t4_sigs.append(sig4)

        if t0_sigs:
            avg_t0 = np.mean(t0_sigs)
            avg_t4 = np.mean(t4_sigs)
            retention = (avg_t4 / avg_t0 * 100) if avg_t0 != 0 else 0
            retentions[cond] = retention

    # Correlation between prompt length and retention
    conds = [c for c in retentions if c in prompt_lengths]
    lengths = np.array([prompt_lengths[c] for c in conds])
    rets = np.array([retentions[c] for c in conds])

    print(f"\n  {'Condition':<25} {'Words':>6} {'T4 retention':>13}")
    print(f"  {'-' * 46}")
    for c in sorted(conds, key=lambda x: prompt_lengths[x]):
        print(f"  {c:<25} {prompt_lengths[c]:>6} {retentions[c]:>+12.0f}%")

    corr = np.corrcoef(lengths, rets)[0, 1] if len(lengths) > 2 else 0
    print(f"\n  Correlation(word_count, retention) = {corr:.3f}")
    if abs(corr) > 0.7:
        print("  >>> STRONG correlation. Durability IS explained by length.")
    elif abs(corr) > 0.4:
        print("  >>> MODERATE correlation. Length is a partial factor.")
    else:
        print("  >>> WEAK correlation. Length does not explain durability.")

    # ================================================================
    # F4: Is baseline drifting?
    # ================================================================
    print(f"\n{'=' * 70}")
    print("F4: IS BASELINE DRIFTING? (absolute values, not diffs)")
    print("=" * 70)

    # Track absolute mid and late values for each condition across turns
    print(f"\n  Absolute mid-layer resid norm by turn:")
    print(f"  {'Turn':>6}", end="")
    cond_order = ["baseline", "handled", "scrambled", "only_artifact", "scientific_method"]
    for c in cond_order:
        print(f" {c[:12]:>13}", end="")
    print()
    print(f"  {'-' * (6 + 14 * len(cond_order))}")

    for turn in range(8):
        print(f"  {turn:>6}", end="")
        for cond in cond_order:
            vals = []
            for s in results:
                turns = s["conditions"].get(cond, [])
                if len(turns) > turn:
                    vals.append(turns[turn]["mid_resid"])
            avg = np.mean(vals) if vals else 0
            print(f" {avg:>13.1f}", end="")
        print()

    print(f"\n  Absolute late-layer resid norm by turn:")
    print(f"  {'Turn':>6}", end="")
    for c in cond_order:
        print(f" {c[:12]:>13}", end="")
    print()
    print(f"  {'-' * (6 + 14 * len(cond_order))}")

    for turn in range(8):
        print(f"  {turn:>6}", end="")
        for cond in cond_order:
            vals = []
            for s in results:
                turns = s["conditions"].get(cond, [])
                if len(turns) > turn:
                    vals.append(turns[turn]["late_resid"])
            avg = np.mean(vals) if vals else 0
            print(f" {avg:>13.1f}", end="")
        print()

    # ================================================================
    # F5: Do all conditions converge to the same profile?
    # ================================================================
    print(f"\n{'=' * 70}")
    print("F5: DO ALL CONDITIONS CONVERGE? (spread across conditions by turn)")
    print("=" * 70)

    print(f"\n  Standard deviation of mid-layer values ACROSS conditions, by turn:")
    print(f"  (If decreasing → conditions converge → durability is trivial)")
    print()
    print(f"  {'Turn':>6} {'Mid StdDev':>12} {'Late StdDev':>12} {'Converging?':>12}")
    print(f"  {'-' * 44}")

    all_conds = [c for c in prompt_lengths if c != "baseline"]
    prev_mid_std = None
    for turn in range(8):
        mid_means = []
        late_means = []
        for cond in all_conds:
            mvals = []
            lvals = []
            for s in results:
                turns = s["conditions"].get(cond, [])
                if len(turns) > turn:
                    mvals.append(turns[turn]["mid_resid"])
                    lvals.append(turns[turn]["late_resid"])
            if mvals:
                mid_means.append(np.mean(mvals))
            if lvals:
                late_means.append(np.mean(lvals))

        mid_std = np.std(mid_means) if mid_means else 0
        late_std = np.std(late_means) if late_means else 0

        converging = ""
        if prev_mid_std is not None:
            if mid_std < prev_mid_std * 0.9:
                converging = "YES"
            elif mid_std > prev_mid_std * 1.1:
                converging = "diverging"
            else:
                converging = "stable"
        prev_mid_std = mid_std

        print(f"  {turn:>6} {mid_std:>12.3f} {late_std:>12.3f} {converging:>12}")

    # ================================================================
    # F3: Is the metric measuring vocabulary dilution?
    # ================================================================
    print(f"\n{'=' * 70}")
    print("F3: VOCABULARY DILUTION CHECK")
    print("Scrambled (vocabulary, no semantics) vs handled (vocabulary + semantics)")
    print("If scrambled decays at same rate → metric measures vocabulary dilution")
    print("=" * 70)

    for cond in ["handled", "scrambled", "only_artifact"]:
        print(f"\n  --- {cond} ---")
        print(f"  {'Turn':>6} {'Mid diff':>10} {'Late diff':>10} {'Signature':>10}")
        print(f"  {'-' * 38}")

        for turn in range(8):
            mid_diffs = []
            late_diffs = []
            for s in results:
                ct = s["conditions"].get(cond, [])
                bt = s["conditions"].get("baseline", [])
                if len(ct) > turn and len(bt) > turn:
                    mid_diffs.append(ct[turn]["mid_resid"] - bt[turn]["mid_resid"])
                    late_diffs.append(ct[turn]["late_resid"] - bt[turn]["late_resid"])
            m = np.mean(mid_diffs) if mid_diffs else 0
            l = np.mean(late_diffs) if late_diffs else 0
            print(f"  {turn:>6} {m:>+10.2f} {l:>+10.2f} {l - m:>+10.2f}")

    # ================================================================
    # SCORECARD
    # ================================================================
    print(f"\n{'=' * 70}")
    print("FALSIFIER SCORECARD")
    print("=" * 70)
    print("""
  F1 (length): Check correlation above. If strong, durability = length artifact.
  F3 (vocab dilution): If scrambled decays like handled, metric = vocab dilution.
  F4 (baseline drift): If baseline absolute values change a lot, diffs are noisy.
  F5 (convergence): If std decreases monotonically, all prompts just converge.

  The only_artifact durability finding SURVIVES if:
    - Length correlation is weak AND
    - Scrambled decays differently from only_artifact AND
    - Conditions do NOT trivially converge AND
    - Baseline is relatively stable
""")


if __name__ == "__main__":
    main()
