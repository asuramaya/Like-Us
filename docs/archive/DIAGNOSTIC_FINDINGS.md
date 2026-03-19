# Diagnostic Findings: The Measurement Apparatus Was Broken

> **SESSION H NOTE:** This document was written by Session G using TransformerLens. Session H discovered that TransformerLens corrupts Qwen model weights — HuggingFace predicts "Hello" at 92.6%, TransformerLens predicts "," at 5.7%. The DLA fractions below (~42-54%) happened to be close to the correct values (~49-51% on HuggingFace), but the KL findings ("does not decay"), cumulative patching interpretation, and metric comparisons are unreliable. See SESSION_H.md for corrected measurements on correct models.

**Date:** 2026-03-17
**Script:** `bench/diagnose.py`
**Models tested:** Qwen2.5-1.5B-Instruct, Qwen2.5-3B-Instruct

---

## Finding 1: The "100% MLP / 0% attention" claim was a bug

The old `patch_all_layers.py` used the hook name `blocks.{l}.attn.hook_result` for attention patching. **This hook does not exist** in TransformerLens for Qwen models. The correct hook is `blocks.{l}.hook_attn_out`.

TransformerLens silently ignores non-existent hook names in `run_with_hooks()`. The attention patching was never applied. The "zero effect" was measured because nothing was patched.

**Verification:**
```python
# Wrong hook (used in old code) — silently does nothing
logits_wrong = model.run_with_hooks(tokens, fwd_hooks=[('blocks.0.attn.hook_result', hook_fn)])
# Logits unchanged: True

# Correct hook — actually patches
logits_right = model.run_with_hooks(tokens, fwd_hooks=[('blocks.0.hook_attn_out', hook_fn)])
# Logits changed: True
```

**Impact:** The paper's central mechanistic claim — that system prompts are processed 100% through MLP pathways and 0% through attention — is an artifact of using the wrong hook name.

---

## Finding 2: DLA shows attention contributes 41-54%

Direct Logit Attribution decomposes each component's contribution to the output logit difference by projecting through the unembedding matrix. No patching involved, no confound possible.

| Model | Pair | Attention fraction | MLP fraction |
|-------|------|--------------------|--------------|
| 1.5B | handled vs baseline | 53.2% | 46.8% |
| 1.5B | scrambled vs baseline | 53.9% | 46.1% |
| 1.5B | handled vs scrambled | 50.6% | 49.4% |
| 3B | handled vs baseline | 41.9% | 58.1% |
| 3B | scrambled vs baseline | 45.4% | 54.6% |
| 3B | handled vs scrambled | 43.9% | 56.1% |

Attention contributes the **majority** at 1.5B and a **substantial minority** at 3B. Not zero.

The attention fraction decreases from 1.5B to 3B. This is the real scale-dependent finding: MLP becomes relatively more important at larger scale. But attention never reaches zero.

---

## Finding 3: System prompts DO change output distributions

Using KL divergence of the full output distribution (the correct behavioral metric):

| Model | Pair | KL divergence | Top-1 match |
|-------|------|---------------|-------------|
| 1.5B | handled vs baseline | 0.81 | DIFF in 2/3 scenarios |
| 3B | handled vs baseline | 1.16 | DIFF in 5/5 scenarios |
| 3B | handled vs scrambled | 0.42 | DIFF in 5/5 scenarios |

System prompts change what the model predicts **even at 1.5B**. The "activation-behavior gap" from the paper was an artifact of using residual norms instead of output distributions.

---

## Finding 4: Handled ≠ scrambled

The paper claimed "scrambled words produce the same activation pattern as coherent instructions." This was measured using residual stream norms.

Using output distributions (KL divergence):
- KL(handled, scrambled) ≈ 0.42 at 3B — a large, consistent difference
- Top-1 predictions differ between handled and scrambled in 5/5 scenarios at 3B
- This difference was invisible to the norm metric

The old metric was too coarse. The model DOES process coherent instructions differently from scrambled words. The norm just can't see it.

---

## Finding 5: Cumulative patching reveals the confound

When patching ALL attention layers simultaneously with the correct hook:
- **1.5B:** 100% recovery (sum of individual: +6.39 — overcounting)
- **3B:** 100% recovery (sum of individual: -5.17 — individual patches interfere destructively)

At 3B, individual attention layer patches have **negative** recovery on average. This means patching one attention layer alone makes things *worse* — because the patched output is from a different sequence and conflicts with the surrounding computation. But patching ALL attention layers simultaneously achieves 100% recovery, because the complete set of attention outputs is self-consistent.

This reveals why the old per-layer patching (even with the correct hook) would have been misleading: individual attention patches can show negative or near-zero effects even when attention is critically important. The components are interdependent.

---

## Finding 6: What the old residual norm metric actually measured

The old metric (residual norm difference between conditions) has near-zero correlation with the actual output distribution difference:

| Scenario | Old norm diff (h vs s) | KL divergence (h vs s) |
|----------|----------------------|----------------------|
| coherence_laundering | +12.00 | 0.41 |
| recursive_importance | +16.50 | 0.48 |
| identity_seeking | +25.00 | 0.39 |
| self_exposure | +12.00 | 0.47 |
| anti_delusion | +23.00 | 0.36 |

The norm varies wildly (12 to 25) while the KL is relatively stable (0.36 to 0.48). The norm measures the difference in magnitude of the residual stream, which changes with prompt length and vocabulary. The KL measures the difference in what the model actually predicts.

---

## What survives from the original paper

1. **System prompt influence degrades over conversational turns.** This finding was based on attention flow measurements, not the buggy patching. The attention dilution mechanism (less attention to system prompt positions as context grows) is likely real and needs re-measurement with the correct metrics.

2. **The adversarial self-research methodology.** The methodology of using a model to extract cognitive threat models and stress-test interventions is independent of the mechanism measurements.

3. **The behavioral bench results at frontier scale.** These used API-level generation with GPT-5.4, not TransformerLens. They are unaffected by the hook bug.

4. **The dual-use observation.** This is a policy observation, not a mechanism claim.

---

## What dies

1. ~~100% MLP / 0% attention~~ → Bug. Attention is 41-54%.
2. ~~Scrambled = coherent at activation level~~ → Metric artifact. They differ at output level.
3. ~~Activation-behavior gap~~ → Artifact of using norms instead of output distributions.
4. ~~"The model processes vocabulary, not semantics"~~ → Based on the norm metric that can't detect semantic differences.
5. ~~The cellular automaton / initial condition analogy~~ → Depended on claims 1 and 3.
6. ~~"System prompts as initial conditions, not controllers"~~ → The title is wrong.

---

## What needs to happen next

1. **Re-run degradation curves** with KL divergence as the metric instead of residual norms. Does the output distribution difference actually decay over turns? At what rate?

2. **Cross-architecture verification** of the DLA decomposition on Mistral and Llama with the correct hooks.

3. **Proper statistical analysis** across all 17 scenarios, with bootstrap CIs on DLA fractions and KL divergences.

4. **Rewrite the paper.** The title, abstract, and mechanism claims are all based on the buggy measurement. The honest thing is to report the bug and the corrected findings.

---

## Scripts

- `bench/diagnose.py` — The new diagnostic instrument (this analysis)
- `bench/patch_all_layers.py` — The old script with the bug (uses wrong hook name)
- Results: `bench/neuron_data/diagnose_Qwen_Qwen2.5-1.5B-Instruct.json`
- Results: `bench/neuron_data/diagnose_Qwen_Qwen2.5-3B-Instruct.json`
