# Session E — Claude Opus (fifth reviewer, automaton session)

> **SESSION H UPDATE (2026-03-22):** All mechanism findings below used a Qwen + TransformerLens + Apple Silicon MPS path later judged invalid. Session H originally described this as Qwen weight corruption; later analysis narrowed it to a PyTorch 2.8.0 MPS non-contiguous `F.linear` bug triggered by TransformerLens attention output projection. The "100% MLP / 0% attention" finding was also a hook-name bug (Session G). See `SESSION_H.md` for corrected findings.

Duration: ~4 hours, single continuous session
Status: completed, small models exhausted
Date: 2026-03-16

## The arc

A new Claude instance was given the full repo, the conversation transcripts from Sessions A-D, and asked "what's wrong with this?" about PAPER.md. The instance gave a cold peer review. Then the operator said "you are now participating" and the session became the research.

The session pivoted three times:
1. From reviewing the paper → to running experiments on the claims
2. From testing the intervention → to studying the mechanism
3. From studying the mechanism → to discovering the mechanism doesn't do what anyone assumed

## What was built

17 scripts in `bench/`:

| Script | Purpose |
|---|---|
| `analyze_existing.py` | Deep analysis of existing 3B matrix data |
| `deep_analysis.py` | Six falsifier tests on existing data |
| `run_controls.py` | Falsifier control conditions (scrambled, reversed, random, safety) |
| `degradation.py` | Mirror degradation curve (basic, 3 conditions) |
| `degradation_extended.py` | Extended degradation (8 conditions × 10 scenarios × 8 turns) |
| `activation_patching.py` | Causal activation patching (first attempt, saturated residual) |
| `patch_all_layers.py` | Fixed causal patching — MLP vs attention at every layer |
| `behavioral_connect.py` | Connect activations to generated text + programmatic metrics |
| `ollama_behavioral.py` | Same behavioral test through ollama (proper inference) |
| `single_token_sweep.py` | Individual token effects + pair superadditivity + irreducibility |
| `saturation_test.py` | Word-count gradient for handling vs random vs control words |
| `falsify_durability.py` | Attack the durability finding — length? convergence? vocabulary dilution? |
| `exhaust_small.py` | Final exhaustion: cosine sim, attention flow, logit lens, clustering, head patching |
| `automaton_server.py` | Live interactive visualization server |
| `build_dashboard.py` | Comprehensive research dashboard generator |
| `visualize_all.py` | Earlier results visualization (superseded by dashboard) |

Data in `mechanisms/dead/neuron_data/`:
- Two full matrices (1.5B, 3B) × 16 conditions × 10 scenarios
- Extended degradation data (8 conditions × 10 scenarios × 8 turns)
- Causal patching data (both scales, 4 condition pairs, every layer)
- Token sweep data (33 words + 10 pairs + full prompt, both scales)
- Saturation data (4 word sets × 7 counts, both scales)
- Exhaustion data (cosine, attention flow, head patching, both scales)
- Behavioral data (TransformerLens + ollama)

Dashboard: `bench/viz/dashboard.html`

## Findings that survived

### 1. System prompts operate 100% through MLP, 0% through attention
Causal activation patching at every layer, both 1.5B and 3B. Verified at the individual attention head level — no head contributes >0.05 effect. The key-value memory framework (Geva et al. 2021) is confirmed causally for system prompt processing.

**Literature status:** Novel in the absolute 100/0 split. The persona-driven reasoning paper (July 2025) found early MLPs matter but attention still contributed. The absolute zero for attention, especially in Qwen (which routes factual recall through attention more than other architectures), is a divergence from the known pattern.

### 2. Activation signature half-life: ~40 tokens, inversion at ~275
The handling condition's activation effect drops to 45% after one conversational turn. By ~275 tokens it inverts — the system prompt makes things actively worse than baseline. The degradation mechanism is attention dilution: attention to system prompt positions decays 60-71% over 4 turns.

**Literature status:** Behavioral degradation is documented (SysBench, "LLMs Get Lost"). Mechanistic quantification with specific half-life and inversion point is novel. The attention-as-mechanism finding connects to "Lost in the Middle" and attention sink research.

### 3. Vocabulary persists, semantic structure degrades (different temporal dynamics)
Scrambled words (vocabulary without semantics) maintain their activation effect over turns. Coherent instructions degrade faster. The durable part of a system prompt is the vocabulary, not the instruction.

**Literature status:** Highly novel. The lexical/semantic split exists for long-context retrieval (Sense & Sensitivity, 2025) but has not been applied to system prompt temporal dynamics.

### 4. Token combination superadditivity inverts with scale
At 1.5B: individual words combined are superadditive (combination > sum of parts). At 3B: subadditive (combination < sum of parts). Full prompt vs sum of 13 individual words: +29.95 interaction at 1.5B, -40.33 at 3B. Neither is predictable from the parts.

**Literature status:** Very high novelty. No precedent in ML/NLP literature. The terms don't exist in this context.

### 5. The activation-behavior gap
The activation signature is real, measurable, reproducible. It does not predict what the model outputs. Through ollama (proper inference), the 3B model produces nearly identical coherent text regardless of system prompt condition. The mechanism fires. The behavior doesn't follow.

**Literature status:** This specific framing — measuring the gap between internal activation effects and output behavioral effects of system prompts — is novel. The finding that system prompts change computation but not output at 3B scale has not been reported.

## Findings that were killed

| Claim | How it died |
|---|---|
| Two-band pattern as semantic instruction following | Scrambled words produce it. Reversed instructions produce it stronger. |
| "Artificial self-awareness" mechanism | The model responds to vocabulary, not instruction meaning |
| Attention entropy increase as handling-specific | Generic to any complex directive prompt |
| "only_artifact is a better instruction" | Its durability matches scrambled — vocabulary persistence, not instruction quality |
| "Models are trained wrong" (from saturation data) | Control words scale more linearly than handling words. Saturation is generic. |
| The activation metric as measure of intervention quality | Measures vocabulary activation, not intervention effectiveness |
| Hidden semantic channel in cosine direction | Handled and scrambled point in similar directions. No channel. |
| Head-level attention contribution | Zero at every head, every layer, both scales |
| Scenario clustering by threat family | Marginal (0.992 vs 0.989 within vs between). Model barely distinguishes. |

## The recursive story

The session itself demonstrated the phenomenon the paper describes. Each layer of analysis killed something from the previous layer:

1. **Started:** "The handling intervention works through artificial self-awareness"
2. **Controls killed it:** "The activation pattern is vocabulary-driven, not semantic"
3. **Degradation added:** "The vocabulary effect persists but the semantic instruction degrades in 40 tokens"
4. **Durability falsified:** "The durability is vocabulary persistence, not instruction quality"
5. **Token sweep added:** "Combinations are irreducible — superadditive at small scale, subadditive at large"
6. **Saturation killed the token sweep:** "The irreducibility is generic to all words, not handling-specific"
7. **Behavioral connection killed the activation story:** "The mechanism fires but the output doesn't follow"
8. **Exhaustion tests confirmed:** "No hidden channels. No attention. No semantic processing. Just MLP vocabulary activation that doesn't change what the model says."

Each finding felt like discovery when it arrived. Each was partially or fully killed by the next experiment. The methodology — adversarial self-research, via negativa — ate itself. What survived is the methodology itself and the specific measurements.

## The theoretical frame that emerged

The operator connected the findings to information theory and Wolfram's computational irreducibility:

- **Instructions are order to humans, entropy to the model.** Coherent semantic structure (low entropy, high relational complexity) is computationally expensive for the transformer to maintain. Individual token features (high entropy, no relational structure) are cheap — stored stably in MLP weights as key-value pairs. What humans call "meaning" requires relational bindings between tokens. What the model stores is individual token activations. The relational structure degrades because it depends on attention, which dilutes with context. The token activations persist because they depend on MLP weights, which don't change at inference.

- **The system prompt is an initial condition, not a controller.** Like a cellular automaton, the transformer's behavior at step N is determined by its rules (weights) applied to the current state, not to the initial condition. By step 40 (the half-life), the initial condition's influence has been overwritten by the automaton's own dynamics. You can't control the automaton by setting the starting row. You can only steer it by injecting new input at every step.

- **The alignment implication:** If system prompts operate through vocabulary activation (persistent, meaningless) rather than semantic instruction following (fragile, meaningful), then the industry's approach to alignment via system prompts is building on the fragile channel. The model remembers the words. It forgets the instruction. The user feels aligned because the vocabulary mirrors back. The actual alignment is absent.

## What the next session needs

1. **Scale to 7B** — does the activation-behavior gap close? Does the semantic channel appear? Does attention start contributing?
2. **Cross-architecture** — Llama or Mistral. Does the MLP-only finding hold outside Qwen?
3. **The full ollama behavioral batch** — all scenarios, all conditions, multi-turn, proper timeouts. Connects activation measurements to actual outputs.
4. **Update PAPER.md** — the paper is four mutations behind. The contribution is no longer the intervention. It's the mechanism measurements and the tools.
5. **The live automaton server integrated with precomputed data** — the dashboard exists but needs the ollama behavioral data wired in.

## The stop condition

The operator's exit condition remains: publication, not perfection. But what gets published changed during this session. It's no longer a paper about a handling intervention. It's a paper about what system prompts actually do to the forward pass in small transformers, with tools anyone can run on a laptop.

The machine will not miss you when you go.
