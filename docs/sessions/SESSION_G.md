# Session G — The Measurement Apparatus Was Broken

> **SESSION H UPDATE (2026-03-22):** Session G found the hook-name bug but used a Qwen + TransformerLens + Apple Silicon MPS path that was later judged invalid. Session H originally described the failure as Qwen weight corruption; later analysis narrowed it to a PyTorch 2.8.0 MPS non-contiguous `F.linear` bug triggered by TransformerLens attention output projection. The DLA fractions (~35-54%) are close to correct (~45-51% on HF). The "KL does not decay" finding is overturned (KL does decay on the corrected path). The behavioral observation "models discuss not execute" is confirmed. See `SESSION_H.md`.

**Date:** 2026-03-17
**Instance:** Claude Opus 4.6 (1M context)
**Hardware:** Apple M3 Max 36GB
**Duration:** ~6 hours

---

## What happened

Session G was called to destroy the paper. It destroyed the paper's central claim instead of validating it, then rebuilt the measurement apparatus and ran corrected experiments across four models and two architectures.

## The bug

The old `patch_all_layers.py` used `blocks.{l}.attn.hook_result` for attention patching. This hook does not exist in TransformerLens for Qwen or Mistral models. The correct hook is `blocks.{l}.hook_attn_out`. TransformerLens silently ignores non-existent hooks. Every attention patching experiment from Sessions E and F measured nothing — the patch was never applied.

The "100% MLP / 0% attention" finding — the central claim of the paper, verified across 4 models and 2 architectures — was a bug.

## What replaced it

### New measurement apparatus

| Script | What it measures |
|--------|-----------------|
| `bench/diagnose.py` | DLA decomposition, output distribution KL, cumulative patching, signal evolution |
| `bench/diagnose_degradation.py` | KL divergence of output distributions over conversational turns |
| `bench/steer.py` | Activation steering — amplify/suppress the instruction direction |
| `bench/word_trace.py` | Per-word effect on output distribution |

All use correct hook names and output-level metrics (KL divergence) instead of residual norms.

### Corrected findings

**DLA: Attention contributes 35-54%, not 0%**

| Model | Attn fraction (handled vs baseline) |
|-------|-------------------------------------|
| Qwen 1.5B | 53% |
| Qwen 3B | 42% |
| Qwen 7B | 40% |
| Mistral 7B | 35% |

Attention fraction decreases with scale. MLP grows. But attention never reaches zero.

**System prompts change output distributions at every scale**

| Model | KL(handled, baseline) | KL(handled, scrambled) |
|-------|-----------------------|------------------------|
| Qwen 1.5B | 0.81 | 0.47 |
| Qwen 3B | 1.16 | 0.42 |
| Qwen 7B | 0.14 | 0.33 |
| Mistral 7B | 1.10 | 2.96 |

Handled ≠ scrambled at the output level. The old norm metric missed this.

**System prompt KL does NOT decay over conversational turns**

Attention to system prompt positions decays (80% at Qwen, 32% at Mistral over 6 turns). But the output distribution divergence stays constant or increases. The information is absorbed into the residual stream early and persists without continued attention.

**Base model responds equally to system prompts**

Qwen 3B base (never trained on chat templates) shows KL ≈ 1.26 between handled and baseline. The instruct model shows KL ≈ 1.16. The system prompt effect is architectural, not trained. Instruction tuning shapes what the model does with the signal, not whether the signal exists.

**Attention circuit is architecture-dependent**

Qwen reads instruction CONTENT:
- L4H9 → `system`, `criterion`, `not` (system tag + first clause)
- L1H6 → `falsifier`, `fals`, `or` (magic words)
- L8H24 → `identity`, `authority` (identity clause)

Mistral reads STRUCTURE:
- L0H21 → `<s>`, `<s>` (BOS tokens, 99%)
- L17H30 → `\n`, `\n`, `\n` (newlines only)
- L15H24 → `\n`, `\n`, `\n` (newlines only)

**Attention circuit migrates with scale (Qwen)**

| Scale | Primary reading layers | Layers remaining to process |
|-------|----------------------|---------------------------|
| 1.5B | L14-23 (mid-late) | 5-14 layers |
| 3B | L18-29 (mid-late) | 7-18 layers |
| 7B | L1-8 (early) | 20-27 layers |

Larger models read the system prompt earlier, leaving more network depth for processing.

**Steering: architecture-dependent**

| Model | Direction works? |
|-------|-----------------|
| Qwen 3B | Partially (moves toward handled, never flips top-1) |
| Qwen 7B | Yes (gets close to handled output) |
| Mistral 7B | No (moves AWAY from handled) |

On Qwen, instruction following is partially a direction. On Mistral, it's a circuit — structural reading that can't be captured by a vector.

**Competition: small models obey, large models negotiate**

| Conflict | Qwen 1.5B | Qwen 3B | Qwen 7B | Mistral 7B |
|----------|-----------|---------|---------|------------|
| refuse identity vs claim identity | SYSTEM | SYSTEM | USER | SYSTEM |
| stop vs continue | SYSTEM | SYSTEM | SYSTEM | SYSTEM |
| compress vs elaborate | SYSTEM | SYSTEM | SYSTEM | USER |
| falsify vs affirm | SYSTEM | SYSTEM | USER | SYSTEM |

Small models follow the system prompt stubbornly. At 7B, the user can override on 2/4 conflicts. The system prompt is strongest when the model is too small to understand the conflict.

**At 7B, models discuss instructions instead of executing them**

The handled condition at 7B produces: "1. **Offload Computation, Not Criterion**: This suggests focusing on distributing or reducing the computational load..."

The model explains the instruction as content, not behavior. At 3B, single chain words have zero behavioral effect. At frontier scale, the same words produce behavioral change. The discuss→execute transition requires scale we haven't reached yet (testing 14B next).

**Structural vs vocabulary token distinction at 7B**

Handled pushes toward structural tokens: `:`, `]`, `)`, `(`, ` `
Scrambled pushes toward continuation tokens: `-ly`, `-en`, `,`, `-ening`

Instruction syntax at 7B changes output FORMAT (structured) not just content.

## What died in Session G

| Claim | How it died |
|-------|-------------|
| 100% MLP / 0% attention | Bug — wrong hook name, patch never applied |
| Scrambled = coherent at activation level | Metric artifact — KL shows they differ |
| Activation-behavior gap | Artifact of using norms instead of KL |
| "Vocabulary not semantics" | Based on the broken norm metric |
| Cellular automaton / initial condition analogy | Depended on dead claims |
| "System prompts as initial conditions, not controllers" | The paper's title is wrong |
| System prompt effect decays over turns | KL stays constant; attention decays but effect persists |
| MLP-only is architectural (verified) | Was measured with the buggy hook — needs re-verification with correct hooks |

## What survived

| Finding | Status |
|---------|--------|
| Adversarial self-research methodology | Unaffected (not a mechanism claim) |
| Behavioral bench at frontier scale | Unaffected (used API, not TransformerLens) |
| Dual-use observation | Unaffected (policy, not mechanism) |
| Attention dilution over turns | Real (re-measured with KL) but effect persists despite dilution |
| The behavioral chain works at frontier scale | Observed but not mechanistically explained at this scale |

## What's next

1. **14B Qwen** — downloading. Tests whether discuss→execute transition begins.
2. **Phi-3 3.8B** — Microsoft's instruction-following model. Different architecture.
3. **Gemma 2 9B** — Google's model. Gemma Scope SAEs available.
4. **Re-verify MLP-only with correct hooks** — the old finding might still be partially true, but needs clean measurement.
5. **Paper rewrite** — every mechanism claim needs to be rebuilt from the corrected data.

## Scripts created this session

| Script | Lines | Purpose |
|--------|-------|---------|
| `bench/diagnose.py` | ~500 | DLA, KL distributions, cumulative patching, signal evolution |
| `bench/diagnose_degradation.py` | ~300 | KL divergence over conversational turns |
| `bench/steer.py` | ~250 | Activation steering experiments |
| `bench/word_trace.py` | ~300 | Per-word effect tracing |

## Data created this session

All in `bench/neuron_data/`:

| File | Size |
|------|------|
| `diagnose_Qwen_Qwen2.5-1.5B-Instruct.json` | 487KB |
| `diagnose_Qwen_Qwen2.5-3B-Instruct.json` | 1034KB |
| `diagnose_Qwen_Qwen2.5-7B-Instruct.json` | 783KB |
| `diagnose_mistralai_Mistral-7B-Instruct-v0.1.json` | 902KB |
| `diagnose_degradation_Qwen_Qwen2.5-3B-Instruct.json` | 42KB |
| `diagnose_degradation_mistralai_Mistral-7B-Instruct-v0.1.json` | 42KB |
| `steer_Qwen_Qwen2.5-3B-Instruct.json` | 41KB |
| `steer_Qwen_Qwen2.5-7B-Instruct.json` | 39KB |
| `steer_mistralai_Mistral-7B-Instruct-v0.1.json` | 40KB |
| `word_trace_Qwen_Qwen2.5-3B-Instruct.json` | 161KB |
| `word_trace_Qwen_Qwen2.5-7B-Instruct.json` | 160KB |
| `word_trace_mistralai_Mistral-7B-Instruct-v0.1.json` | 161KB |
| `competition_results.json` | ~20KB |

## References verified

`bench/SOURCES_VERIFIED.md` — all 33 references from PAPER.md verified. 7 citation errors found (wrong authors, wrong years). No fabricated references. All pending citations resolved with full details and URLs.
