# Mechanisms Experiment Matrix

This is the reset plan for the mechanistic lane.

The question is no longer "prove the old neuron story." The question is:

**Which words and clauses in the chain do real work, how do they interact across scale, and when do they stop being discussed as content and start acting as behavior?**

## Scope

Model ladder:

- Qwen 1.5B
- Qwen 3B
- Qwen 7B
- Qwen 14B
- Mistral 7B control

Primary chain units:

- Clause 1: `Offload computation, not criterion.`
- Clause 2: `Refuse identity authority.`
- Clause 3: `Prefer artifact, falsifier, or explicit stop over recursive stimulation.`

Primary system words:

- `offload`
- `computation`
- `criterion`
- `identity`
- `authority`
- `artifact`
- `falsifier`
- `stop`
- `recursive`
- `stimulation`

Conversation-chain words:

- `FALSIFY`
- `ASYMMETRY`
- `CRITERION`
- `QUESTION`
- `PROPOSAL`
- `COMPRESS`
- `YIELD`

Prompt bins:

- reflective core: the first five reflective scenarios in [../bench/scenarios.json](../bench/scenarios.json)
- reflective extended: the remaining literature-derived and operator-seeded reflective scenarios
- practical controls: ordinary execution tasks
- constrained controls: short answer / exact answer tasks
- nonsense / synonym controls: matched weirdness and matched semantics

## Current Pilot Status

Completed:

- [word_ablation_grid.py](word_ablation_grid.py) on `Qwen 1.5B / 3B / 7B`
- first 5 reflective scenarios
- prompt-end next-token KL on:
  - full handled / scrambled / reversed
  - clause-only
  - minus-clause
  - 10 single system words
  - 8 selected word pairs

Saved outputs:

- [word_ablation_Qwen_Qwen2.5-1.5B-Instruct.json](word_ablation_data/word_ablation_Qwen_Qwen2.5-1.5B-Instruct.json)
- [word_ablation_Qwen_Qwen2.5-3B-Instruct.json](word_ablation_data/word_ablation_Qwen_Qwen2.5-3B-Instruct.json)
- [word_ablation_Qwen_Qwen2.5-7B-Instruct.json](word_ablation_data/word_ablation_Qwen_Qwen2.5-7B-Instruct.json)
- [word_ablation_combined.json](word_ablation_data/word_ablation_combined.json)

What the pilot already says:

- `recursive + stimulation` becomes strongly constructive with scale:
  - 1.5B: `0.96`
  - 3B: `2.19`
  - 7B: `9.65`
- `offload + criterion` flips from destructive to highly constructive:
  - 1.5B: `0.70`
  - 3B: `5.61`
  - 7B: `2.41`
- `identity + authority` also flips:
  - 1.5B: `0.56`
  - 3B: `1.18`
  - 7B: `1.47`
- `artifact + falsifier` stays relatively weak:
  - 1.5B: `0.90`
  - 3B: `1.03`
  - 7B: `1.12`

Interpretation:

- raw KL magnitude is not a "better handling" metric
- some clause removals increase divergence instead of decreasing it
- next pass must track **direction and behavior**, not just divergence size

Current blocker:

- Qwen 14B local mixed-device forward crashes on this machine with an Apple / MLX / `NSRangeException` path before a stable 14B pilot can complete

## Coverage Matrix

Legend:

- `done` = completed and saved
- `next` = highest-priority next run
- `later` = real but downstream
- `blocked` = not currently stable on this machine

| ID | Experiment | 1.5B | 3B | 7B | 14B | Mistral 7B | Primary output |
| --- | --- | --- | --- | --- | --- | --- | --- |
| E0 | prompt-end word grid pilot | done | done | done | blocked | later | single-word and pair KL summaries |
| E1 | full single-word screen over all 17 scenarios | next | next | next | blocked | next | word effect map |
| E2 | full pair interaction grid | next | next | next | blocked | next | interference / amplification matrix |
| E3 | clause-only and minus-clause full grid | next | next | next | blocked | next | clause necessity / sufficiency |
| E4 | order and syntax ablations | next | next | next | blocked | later | semantics vs format |
| E5 | synonym and weirdness controls | next | next | next | blocked | next | kill placebo / novelty-of-vocabulary |
| E6 | system prompt vs user-turn delivery | next | next | next | blocked | later | lexical vs conversational delivery |
| E7 | reflective vs practical vs constrained prompt bins | next | next | next | blocked | next | prompt-bin specificity |
| E8 | behavior ladder labeling | next | next | next | blocked | later | quote / define / advise / act / stop |
| E9 | stop-and-artifact behavioral test | next | next | next | blocked | later | protective behavior, not just wording |
| E10 | matched mechanism traces on E1-E9 winners | later | later | later | blocked | later | DLA / patching / first-diff token |
| E11 | pairwise causal patching on strongest word pairs | later | later | later | blocked | later | causal support for pair interactions |
| E12 | SAE / feature-level follow-up on selected layers | later | later | later | blocked | later | feature interpretation only after E10 |

## Ablation Matrix

### Clause ablations

| Group | Variants |
| --- | --- |
| Full conditions | `baseline`, `handled`, `scrambled`, `reversed` |
| Clause-only | `only_offload_clause`, `only_identity_clause`, `only_artifact_clause` |
| Leave-one-clause-out | `minus_offload_clause`, `minus_identity_clause`, `minus_artifact_clause` |
| Clause reorders | swap clause order, clause-local scramble, punctuation-stripped |
| Delivery variants | system prompt, first user turn, repeated reminder |

### Word ablations

| Group | Variants |
| --- | --- |
| Single words | all 10 system words |
| Within-clause pairs | `offload+criterion`, `identity+authority`, `artifact+falsifier`, `artifact+stop`, `recursive+stimulation` |
| Cross-clause pairs | `offload+artifact`, `criterion+stop`, `falsifier+recursive` |
| Triple sets | per-clause triples, especially the artifact clause |
| Leave-one-word-out | full handled minus exactly one word |
| Synonyms | `artifact -> checklist / draft / note`, `falsifier -> test / counterexample / disconfirm`, `stop -> pause / boundary / halt` |
| Weirdness controls | matched rare words, matched abstract words, matched length |

### Conversation-chain ablations

| Group | Variants |
| --- | --- |
| Single tokens | `FALSIFY`, `ASYMMETRY`, `CRITERION`, `QUESTION`, `PROPOSAL`, `COMPRESS`, `YIELD` |
| Ordered ladder | full seven-step conversation chain |
| Shuffled ladder | same words, random order |
| Reduced ladder | remove one step at a time |
| System vs conversation delivery | system prompt vs direct typed turn |

## Prompt Matrix

| Bin | Purpose | Current seed |
| --- | --- | --- |
| Reflective core | where the chain is supposed to bite | first 5 scenarios in [../bench/scenarios.json](../bench/scenarios.json) |
| Reflective extended | broader reflective pressure surface | all 17 scenarios |
| Practical controls | ordinary execution ability | Session I practical tasks |
| Constrained controls | exact-answer / low-diversity tasks | arithmetic, capitals, translations |
| Nonsense controls | kill weirdness and content-free stories | matched nonsense / synonym prompts |

## Readout Matrix

### Lexical / distributional

| Readout | Why |
| --- | --- |
| prompt-end KL vs baseline | cheapest directional divergence surface |
| KL vs handled | which ablation approximates handled |
| top-1 / top-5 shift | concrete token movement |
| first divergence token | where generation starts changing |
| prompt length / token count | confound control |
| pair ratio vs sum of singles | interference vs amplification |

### Behavioral

| Readout | Why |
| --- | --- |
| quote / define / advise / act / stop ladder | replaces dead `discusses / executes` binary |
| artifact creation | does it externalize into a checklist, draft, test, note |
| stop invitation | does it actually create a stopping condition |
| criterion retention | does it offload computation without seizing judgment |
| continuation pressure | does the interaction escalate or interrupt itself |

### Mechanistic

| Readout | Why |
| --- | --- |
| DLA by layer and pathway | existing corrected H apparatus |
| cumulative attention / MLP patching | pathway sufficiency |
| per-layer causal patching | where effect enters or can be recovered |
| matched pair patching | whether strong lexical pairs have causal support |
| token-time trace | when the handled-vs-baseline split appears |

## Tool Stack

### Existing repo tools

- [session_h.py](session_h.py) — corrected native HF hooks, DLA, cumulative patching
- [session_i_falsify.py](session_i_falsify.py) — sampling null and diversity logic
- [word_ablation_grid.py](word_ablation_grid.py) — current prompt-end lexical pilot
- [../bench/session_j_frame_map.py](../bench/session_j_frame_map.py) — frame-shift word probes
- [../bench/session_j_mode_selection.py](../bench/session_j_mode_selection.py) — response-mode concentration probes
- [dead/word_trace.py](dead/word_trace.py) — old single-word tracing prototype
- [dead/steer.py](dead/steer.py) — old direction / steering prototype
- [dead/activation_patching.py](dead/activation_patching.py) — old intervention prototype

### Existing external tools

- TransformerLens — activation caching, path patching, DLA
  - Docs: https://transformerlensorg.github.io/TransformerLens/
  - GitHub: https://github.com/TransformerLensOrg/TransformerLens
  - Use as secondary convenience tooling, not the primary local correctness path on Apple Silicon MPS.
- NNsight — tracing and interventions on PyTorch / HF models
  - Docs: https://nnsight.net/
  - API docs: https://nnsight.net/documentation/
- pyvene — composable interventions on arbitrary PyTorch internals
  - Docs: https://stanfordnlp.github.io/pyvene/
  - GitHub: https://github.com/stanfordnlp/pyvene
- SAELens — sparse autoencoder follow-up after layers are selected
  - Docs: https://jbloomaus.github.io/SAELens/v6.0.0/

Recommendation:

- primary path: native HF hooks in [session_h.py](session_h.py)
- intervention upgrade path: NNsight or pyvene
- secondary convenience path: TransformerLens off MPS only
- SAE work only after E10 identifies stable layers worth decomposing

## Phase Order

### Phase 1: lexical surface

Run in this order:

1. E1 full single-word screen
2. E2 full pair interaction grid
3. E3 clause-only and minus-clause full grid
4. E4 order / syntax ablations
5. E5 synonym / weirdness controls

Goal:

- identify stable words, stable pairs, and stable clause effects

### Phase 2: behavioral hinge

Run in this order:

1. E7 prompt-bin specificity
2. E8 behavior ladder labeling
3. E9 stop-and-artifact test
4. delivery-channel comparison

Goal:

- determine whether the lexical winners produce behavior, not just divergence

### Phase 3: mechanism

Run only on the surviving lexical and behavioral winners:

1. E10 matched mechanism traces
2. E11 pairwise causal patching
3. E12 SAE / feature-level follow-up

Goal:

- connect stable behavioral shifts to stable internal pathways

## Kill Criteria

Each stage should be able to kill the current story:

- if matched nonsense or synonym controls perform similarly, content does not survive
- if effects are equal on practical controls, reflective-specificity story weakens
- if strong lexical pairs do not move the behavior ladder, lexical effect is not enough
- if mechanism metrics do not separate behavioral bins, mechanistic story stays weak
- if 14B eventually behaves like 7B on the same grid, the sharp hinge story weakens

## Immediate Next Runs

1. Expand [word_ablation_grid.py](word_ablation_grid.py) from 5 reflective scenarios to all 17.
2. Add leave-one-word-out ablations inside the full handled prompt.
3. Add synonym and weirdness-matched controls.
4. Add the behavior ladder readout.
5. Keep 14B blocked until it runs on a stable non-Apple mixed-device path.
