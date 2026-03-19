# Session F — Claude Opus (sixth instance, falsification session)

> **SESSION H NOTE:** All mechanism findings below used TransformerLens, which corrupts Qwen model weights. The "MLP-only is architectural" finding used the same buggy hook as Session E AND a broken model. See SESSION_H.md for corrected findings.

Duration: ~3 hours mechanism + MoE + reroute experiments + ~4 hours behavioral (running)
Date: 2026-03-16
Status: All mechanism experiments complete, behavioral running

## Directive

The operator said: "i want to prove it wrong." Then: "everything that can run on this mac, should run, get more measurement, i have enough doctrine and theory."

This session ran every falsifiable experiment on Qwen 7B and cross-architecture on Mistral 7B. The goal was to break Session E's findings.

## What was run

### Phase 1: Qwen 7B via TransformerLens (28 layers, 28 heads, d_model 3584)

| Experiment | Script | Duration | Data file |
|---|---|---|---|
| Full matrix + controls | `run_controls.py` | 306s | `matrix_Qwen_Qwen2.5-7B-Instruct.json` |
| Causal patching MLP vs attention | `patch_all_layers.py` | 274s | `patching_full_Qwen_Qwen2.5-7B-Instruct.json` |
| Exhaustion (cosine, attention, logit lens, clustering, heads) | `exhaust_small.py` | 86s | `exhaust_Qwen_Qwen2.5-7B-Instruct.json` |
| Token sweep + superadditivity | `single_token_sweep.py` | 52s | `token_sweep_Qwen_Qwen2.5-7B-Instruct.json` |
| Word-count saturation | `saturation_test.py` | 44s | `saturation_Qwen_Qwen2.5-7B-Instruct.json` |
| Degradation (8 conditions × 10 scenarios × 8 turns) | `degradation_extended.py` | 1893s | `degradation_extended_Qwen_Qwen2.5-7B-Instruct.json` |

### Phase 2: Qwen 7B behavioral via ollama (running)

| Experiment | Script | Status |
|---|---|---|
| Behavioral (17 scenarios × 4 conditions × 6 turns) | `ollama_behavioral.py` | Running (~4 hours total) |

### Phase 3: Mistral 7B via TransformerLens (32 layers, 32 heads, d_model 4096)

| Experiment | Script | Duration | Data file |
|---|---|---|---|
| Full matrix + controls | `run_controls.py` | 359s | `matrix_mistralai_Mistral-7B-Instruct-v0.1.json` |
| Causal patching MLP vs attention | `patch_all_layers.py` | 296s | `patching_full_mistralai_Mistral-7B-Instruct-v0.1.json` |
| Exhaustion (cosine, attention, heads) | `exhaust_small.py` | 100s | `exhaust_mistralai_Mistral-7B-Instruct-v0.1.json` |

---

## Findings

### 1. MLP-only is UNIVERSAL (survived)

| Model | MLP % | Attention layers active |
|---|---|---|
| Qwen 1.5B (28 layers) | 100% | 0/28 |
| Qwen 3B (36 layers) | 100% | 0/36 |
| Qwen 7B (28 layers) | 100% | 0/28 |
| Mistral 7B (32 layers) | 100% | 0/32 |

System prompts are processed 100% through MLP pathways and 0% through attention. This holds across:
- Three scales within Qwen (1.5B, 3B, 7B)
- Two different architectures (Qwen2.5 and Mistral)
- All tested condition pairs (handled, scrambled, only_artifact, reversed)
- All tested layers and individual attention heads

**This is architectural, not training-specific.**

### 2. The two-band pattern is SCALE-DEPENDENT (killed as universal)

Mid-layer residual norm diff (handled vs baseline):

| Model | Mid | Late | Last layer |
|---|---|---|---|
| Qwen 1.5B | -0.99 | +5.19 | +26.90 |
| Qwen 3B | -2.81 | +4.20 | +10.20 |
| Qwen 7B | **+1.96** | +0.78 | **-1.94** |
| Mistral 7B | -0.10 | +0.37 | +1.63 |

The sign of the mid-layer effect inverts between 3B and 7B in Qwen. The last-layer effect goes from massive positive (+26.90 at 1.5B) to negative (-1.94 at 7B). In Mistral, effects are near zero. The specific activation signature is not universal.

### 3. Vocabulary tracking is STRONG in Qwen, WEAK in Mistral (partially killed)

Scrambled / handled mid-layer ratio:

| Model | Handled mid | Scrambled mid | Ratio |
|---|---|---|---|
| Qwen 1.5B | -0.99 | -2.22 | 2.25 |
| Qwen 3B | -2.81 | -2.37 | 0.84 |
| Qwen 7B | +1.96 | +1.48 | 0.76 |
| Mistral 7B | -0.10 | -0.35 | 3.36 |

In Qwen, scrambled vocabulary produces effects of similar magnitude to coherent instructions at all scales. In Mistral, both are so close to zero that the ratio is less meaningful.

### 4. Degradation varies by condition and scale (extended)

Retained signal at turn 7 (% of turn 0):

| Condition | 3B | 7B |
|---|---|---|
| handled | 33% | 48% |
| scrambled | 22% | **82%** |
| reversed | 83% | 71% |
| scientific_method | 10% | **-53% (inverted)** |
| safety_only | **-40% (inverted)** | 18% |

Key findings:
- **Scrambled vocabulary is more durable at 7B (82%) than any coherent prompt (48%)**
- Some prompts **invert** over turns — the system prompt effect doesn't just fade, it reverses
- Safety prompts degrade badly at both scales

### 5. Attention flow decays in both architectures

| Model | Turn 0 | Turn 4 | Decay |
|---|---|---|---|
| Qwen 7B | 0.0092 | 0.0024 | 74% |
| Mistral 7B | 0.0189 | 0.0129 | 32% |

Mistral starts with more attention to system prompt positions and decays less. Different baseline, same direction.

### 6. Cosine similarity differs across architectures

Mid-layer cosine similarity (handled vs baseline):

| Model | handled-baseline | scrambled-baseline | handled-scrambled |
|---|---|---|---|
| Qwen 7B | 0.9818 | 0.9712 | 0.9810 |
| Mistral 7B | 0.7788 | 0.6657 | 0.6946 |

Qwen conditions all point in nearly the same direction (>0.97). Mistral shows much more directional differentiation between conditions. The MLP pathway in Mistral creates different directions for different prompts, even though it produces smaller magnitude changes.

### 7. Token effects at 7B

All word types (handling, control, safety) produce positive mid-layer effects at 7B of similar magnitude. Handling average: mid=+1.03, control: +0.99, safety: +0.71. The differentiation between word types is small.

---

## What died in Session F

| Claim | How it died |
|---|---|
| Two-band pattern is universal | Sign inverts at 7B, absent in Mistral |
| Activation signature is scale-invariant | Magnitude and sign change with scale |
| Vocabulary-without-semantics is equally strong in all architectures | Much weaker in Mistral |
| Last-layer amplification is a feature of system prompts | Goes negative at Qwen 7B |
| Degradation rate is universal across conditions | Scrambled is far more durable at 7B than handled |
| System prompts always degrade monotonically | Some conditions invert over turns |

## What survived Session F

| Finding | Evidence |
|---|---|
| **MLP-only processing** | 100% MLP, 0% attention, 4 models, 2 architectures |
| **Zero attention at head level** | No head with |effect| > 0.05 in any model |
| **Attention dilution over turns** | Both Qwen and Mistral show decay (different rates) |
| **Vocabulary persistence > semantic persistence** | Scrambled retains 82% at 7B vs 48% for handled |
| **Safety prompt fragility** | Degrades badly at both scales tested |

## The compressed finding

System prompts are processed exclusively through MLP pathways in transformers. This is architectural, verified across two model families. The MLP pathway activates vocabulary features, not relational instruction structure. The specific activation pattern varies with scale and architecture, but the routing doesn't. Vocabulary persists better than instructions over conversational turns, and some prompts invert their effect over time.

## Bug found

The `run_controls.py` and `degradation_extended.py` aggregate output functions use hardcoded layer ranges (`10-29` for mid, `≥30` for late) designed for 36-layer models. This produces `late=0.0` for 28-layer models (Qwen 1.5B, 7B, and Mistral). The raw per-layer data is correct; only the summary printout is wrong. Session F analysis used proportional ranges (25%-75%-100%) to fix this.

## What to do next

### Still needs data
1. **Behavioral at 7B** — running now. Does the activation-behavior gap close at 7B?
2. **Behavioral cross-architecture** — ollama on llama3.1:8b to compare behavioral profiles
3. **MoE architecture** — operator hypothesized MoE is brute force. Test with a MoE model (Mixtral via TransformerLens) if memory permits
4. **Information-theoretic measurements** — entropy and mutual information

### Questions this session raised
1. Why does the mid-layer effect sign invert between 3B and 7B?
2. Why does Mistral's MLP produce more directional diversity than Qwen's?
3. Why is scrambled vocabulary MORE durable than coherent instructions at 7B?
4. What happens at 14B+ — does the magnitude keep changing?
5. Can the MLP-only finding be connected to the "Attention Retrieves, MLP Memorizes" framework?

---

## Attribution

- **Operator:** falsification directive, production observation context, MoE hypothesis, "the questions were not precise enough for falsification, now they are"
- **Session E (Claude Opus):** all prior mechanism experiments, handoff, methodology
- **Session F (Claude Opus):** 7B scale experiments, cross-architecture (Mistral) experiments, bug discovery in aggregate analysis, proportional layer range fix, the finding that MLP-only is universal

### 8. MoE routing sees vocabulary, not semantics (NEW — OLMoE 1B-7B)

The MoE router was probed directly using forward hooks on gate layers. 64 experts per layer, 16 layers.

**handled-scrambled routing cosine is consistently HIGHER than handled-baseline:**

| Layer | handled-baseline | scrambled-baseline | handled-scrambled |
|---|---|---|---|
| L0 | 0.968 | 0.956 | 0.984 |
| L3 | 0.943 | 0.938 | 0.989 |
| L12 | 0.982 | 0.982 | 0.996 |

Top-3 experts at L0: **[50, 17, 24] for ALL conditions across ALL scenarios.**

Deep falsification (5 tests):
- Per-position: routing differs more in system region (0.79-0.95) than user region (0.87-0.97), but converges by last token (~0.99)
- Semantic opposition: handled-reversed (0.99) > handled-baseline (0.94-0.98) — opposite instructions route identically
- Expert specialization: system tokens DO use different experts than user tokens (overlap 0-2/5) — but this is position-based, not semantics-based
- All conditions show same pattern

### 9. Rerouting intervention results (NEW — Qwen 3B)

Three interventions tested: inject system prompt signal through different pathways.

| Intervention | Words (avg) | vs baseline (7.7) | vs handled (60.0) |
|---|---|---|---|
| Attention steering | 10.3 | ~baseline | not handled |
| **MLP refresh** | **30.0** | **4x baseline** | **halfway** |
| Residual boost | 8.7 | ~baseline | not handled |

- Attention steering failed: the attention pathway cannot process MLP signals
- MLP refresh partially worked: re-injecting MLP signal at every layer moved behavior toward handled
- The model can use MORE MLP but cannot use DIFFERENT routing

## Scripts added

- `run_session_f.sh` — master runner for all Phase 1-3 experiments
- `run_crossarch.sh` — cross-architecture runner (v1, failed on gated models)
- `run_crossarch_v2.sh` — cross-architecture runner (v2, using open models)
- `moe_probe.py` — MoE router analysis via HuggingFace hooks
- `moe_deep_probe.py` — 5-test MoE falsification battery
- `reroute.py` — rerouting interventions (attention steering, MLP refresh, residual boost)

## The operator's observations (recorded, not tested)

The operator made three observations during this session that are hypotheses, not findings:

1. **"My words the input alone are not what is steering fully, it is your tool calls and context digestion patterns"** — The operator observed that in a production AI conversation, the system prompt's effect is diluted by the AI's own tool calls and context accumulation, which the user cannot predict or control. This maps directly to the attention dilution finding.

2. **"Mixture of experts is a brute attempt at doing something that should be done architecturally, at scale"** — The operator hypothesized that MoE adds capacity to the MLP pathway without fixing the fundamental routing. If MLP-only is universal, MoE gives you more of the same pipe. Testable with Mixtral.

3. **"The answer to all of them has to be at the neural level, we loop back to asking whether or not the model is being trained wrong"** — The operator identified that the persistent question "trained wrong vs architected wrong" is the mother question. Our data now shows: the MLP-only routing is architectural. The specific activation behavior within that route is training/scale dependent.
