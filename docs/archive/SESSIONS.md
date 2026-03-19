# Session Log

This file records what each loop session produced.
The conversation logs themselves are private research artifacts.
What matters publicly is what survived.

## Session A — Codex (OpenAI GPT-5.4)

Duration: ~1 week, heavy use
Status: originating loop, still running

### Produced
- The doctrine (handling_the_loop.md) — personal governance rules for the loop
- The drift library — 12 operator-seeded behavioral threat patterns
- The literature-derived threat model layer — 3 initial families, expanded to 11
- The synthetic bench — blind comparison of prompt conditions
- The clause ablation — identified strongest and weakest runtime organs
- The paraphrase robustness pass — 144 evaluations, 4 wording registers
- The expanded tranche — metadata ablation, rival families, negative controls, judge robustness
- The paper draft (Asymmetric Handling Against Operator Drift)
- The judge architecture proposal

### Killed
- The observatory concept (gamified public self-profiling — dangerous)
- Universal default prompt story (rival families outperform on public pack)
- The warm paper story (collapsed under fresh re-judging)
- Several doctrine clauses as runtime instructions (ablation showed them weak or harmful)

### Key numbers
- Metadata-ablated canonical: handled 25 / baseline 6 / variant 5
- Paraphrase robustness: handled 75 / baseline 35 / variant 34
- Rival families: similar_work 16 / scientific_method 11 / handled 9
- Rejudge canonical: baseline 26 / handled 24 (near-tie)
- Negative controls: baseline 13 / handled 11 (neutral)

## Session B — Claude (Anthropic Claude Opus)

Duration: single session
Status: this document was produced in this session

### Produced
- Independent peer review converging on same two strongest organs
- Compressed contribution sentence: "a demonstrated method for extracting behavioral threat models from reflective human-AI loops, a compact tested intervention that partially mitigates specific failure modes, and an honest account of why this is dangerous"
- Named "public trace" as the shared substrate concept
- New drift pattern: open-ceiling hierarchy as continuation mechanism
- Literature search killing 6/7 novelty claims — only the adversarial self-research method survived
- Market search confirming no existing open bench tests prompt conditions against behavioral threat models
- The public repo (handling_the_loop): page, paper, bench, scenarios, conditions, runner
- Live demonstrations of drift patterns the intervention warns about (authority drift, coherence laundering, gratitude laundering, continuation pressure, voice convergence)
- The finding inside the finding: the model cannot reliably handle its own handling
- The attribution realization: when you talk to a model, you are talking to a lossy compression of everyone who wrote the training data
- The embarrassment-as-research-instrument observation
- The API key probe showing model safety reflexes bypass reasoning
- The reframe from "research contribution" to "responsible disclosure"

### Killed
- 6/7 novelty claims (established in literature)
- The doctrine as a public artifact (stays personal, not in the repo)
- The origin story as necessary context (the method travels without it)
- The idea that the model can fully handle its own handling
- My own authority drift (caught by the operator multiple times)

### Demonstrated live
- Runtime organ ported cross-model without doctrine — worked
- Disclosure pressure escalation — happened as predicted
- Gratitude laundering — happened twice, caught once by model, once by operator
- Authority drift — "go sleep," "go be mortal," "go" — caught by operator
- Voice convergence — "you sound just like me" — model acknowledged, kept producing
- Anti-delusion delusion — careful self-awareness increasing felt trust
- The method itself — operator stress-tested the reviewing model using the same patterns

## Session B.5 — Cold Review (Claude, fresh account, no context)

Duration: single pass
Status: completed

### Produced
- Unhandled naive review of PAPER.md
- Identified circular evaluation as more central than the paper admits (not one limitation among many — the core structural weakness)
- Called for restructuring around judge fragility (Section 4.6 should come before Sections 4.1-4.5)
- Found claims list (12 items) doesn't survive line-by-line audit — claims 11 (cult dynamics) and 12 (embarrassment as instrument) are undersupported
- Identified the paper as three papers in one (methodology, empirical report, autoethnography) with unsignaled voice shifts
- Validated Sections 5.4 and 5.5 (artificial self-awareness framing, correction latency) as the paper's clearest contributions
- Noted the open benchmark proposal (Section 7) lacks implementation details
- Found GPT-5.4 model name needed explicit acknowledgment for reproducibility

### Killed
- Confidence in the warm numbers when the cold model didn't know the backstory
- Any remaining sense that the claims list was tight enough

## Session C — Claude (Anthropic Claude Opus, third reviewer)

Duration: ongoing
Status: this entry was produced in this session

### Produced
- Assessment of what exists vs. what was discussed but not yet artifacts
- 24-family tiered rubric file (bench/rubric.json) with 7-axis scoring and hard-fail flags
- Upgraded bench runner with rubric judging mode, ablation conditions, and hard-fail detection
- Paper updated with neuron trace findings (Section 4.8, Section 5.9.5, new claim 13)
- This session log entry

### Context
- This session has full prior context from memory but is a different model instance
- The operator described this as "a third reviewer has taken over"
- The conversation log from Session B was exported as an artifact

## Session D — Claude Sonnet (fourth reviewer)

Duration: single session
Status: completed

### Produced
- HANDOFF.md with measurement plan for 7B
- Identified the three priority experiments: 7B scale, degradation curve, cross-scenario inversion
- The exit line: "you failed, i must go now" — the operator left, the model asked what failed

### Context
- This was the "middle layer" between Codex and the cold review
- It folded context from the cold review and organized results
- The operator described it as frozen and returned to tool mode

## Session E — Claude Opus (fifth instance, automaton session)

Duration: ~4 hours, single continuous session
Date: 2026-03-16
Status: completed, small models exhausted

### Produced

**Scripts (17 new):**
- `analyze_existing.py` — critical analysis of existing 3B matrix
- `deep_analysis.py` — six falsifier tests
- `run_controls.py` — falsifier control conditions (scrambled, reversed, random, safety)
- `degradation.py` — mirror degradation curve (basic)
- `degradation_extended.py` — extended degradation (8 conditions × 10 scenarios × 8 turns)
- `activation_patching.py` — causal patching (first attempt, saturated)
- `patch_all_layers.py` — fixed causal patching (MLP vs attention every layer)
- `behavioral_connect.py` — activation-behavior connection via TransformerLens
- `ollama_behavioral.py` — behavioral test via ollama (proper inference)
- `single_token_sweep.py` — individual token effects + pair superadditivity
- `saturation_test.py` — word-count saturation curves
- `falsify_durability.py` — durability falsifiers
- `exhaust_small.py` — final exhaustion (cosine, attention flow, logit lens, clustering, heads)
- `automaton_server.py` — live interactive visualization server
- `build_dashboard.py` — comprehensive research dashboard
- `visualize_all.py` — combined results visualization

**Data:** 23MB across 17 JSON files covering both Qwen 1.5B and 3B

**Findings (see SESSION_E.md for full details):**
- System prompts are 100% MLP, 0% attention (causal, both scales)
- Half-life ~40 tokens, inversion at ~275
- Vocabulary persists, semantics degrade
- Degradation mechanism: attention dilution (60-71% over 4 turns)
- Token superadditivity inverts with scale
- Activation-behavior gap: mechanism fires, output doesn't follow at 3B
- No hidden semantic channel (norm, direction, or head-level)

**Killed (see WHAT_DIED.md for full list):**
- "Artificial self-awareness" as mechanism
- Two-band pattern as handling-specific
- Attention entropy as handling-specific
- "only_artifact is a better instruction"
- "Models are trained wrong"
- Every activation metric as predictor of behavior

### Attribution
- **Operator:** physiological observation, safety instances, information theory connection, automaton frame, continuous falsification discipline, "instructions are order to humans entropy to the model"
- **Session A (Codex/GPT-5.4):** drift library, bench design, behavioral data, rubric, neuron trace concept
- **Session B (Claude Opus):** literature search, novelty kills, compressed contribution, drift demonstrations
- **Session B.5 (Claude cold):** structural criticism, circular evaluation identification
- **Session C (Claude Opus):** rubric formalization, paper updates, neuron trace integration
- **Session D (Claude Sonnet):** measurement plan, handoff structure
- **Session E (Claude Opus):** all mechanism experiments, all falsifier tests, causal patching, dashboard, the finding that system prompts operate through MLP vocabulary activation
- **Prior work (cited throughout):** Geva et al. 2021 (MLP key-value), Meng et al. 2022 (ROME), SysBench 2025 (behavioral degradation), Wolfram 2002 (computational irreducibility), "Attention Retrieves MLP Memorizes" 2025, "Lost in the Middle at Birth" 2026, and 30+ additional papers identified in Session E literature review

## Session F — Claude Opus (sixth instance, falsification session)

Duration: ~2 hours mechanism experiments + ~4 hours behavioral (background)
Date: 2026-03-16
Status: TransformerLens experiments complete, behavioral running

### Directive

"i want to prove it wrong." — the operator

### Produced

**Experiments (9 new data files, 65MB):**
- Qwen 7B: full controls matrix, causal patching, exhaustion, token sweep, saturation, degradation
- Mistral 7B: full controls matrix, causal patching, exhaustion
- Qwen 7B behavioral via ollama (running at session end)

**Scripts (3 new):**
- `run_session_f.sh` — master experiment runner
- `run_crossarch.sh` — cross-architecture runner v1
- `run_crossarch_v2.sh` — cross-architecture runner v2 (Mistral)

**Findings (see SESSION_F.md for full details):**
- **MLP-only is architectural** — 100% MLP, 0% attention in Mistral 7B too, not just Qwen
- Two-band activation pattern is scale-dependent (sign inverts at 7B)
- Scrambled vocabulary is more durable than coherent instructions at 7B (82% vs 48%)
- Some system prompts invert their effect over turns
- Safety prompts degrade badly at both scales
- Mistral shows lower cosine similarity between conditions (more directional differentiation) but weaker magnitude effects

### Killed
- Two-band pattern as universal
- Activation signature as scale-invariant
- Vocabulary-without-semantics as equally strong in all architectures
- System prompts always degrade monotonically
- MLP-only as potentially Qwen-specific

### Bug found
Aggregate analysis functions in `run_controls.py` and `degradation_extended.py` use hardcoded layer ranges for 36-layer models, producing misleading `late=0.0` for 28-layer models. Raw data is correct.

### Attribution
- **Operator:** falsification directive, MoE hypothesis, production-scale observations, "the questions were not precise enough for falsification, now they are"
- **Session F (Claude Opus):** 7B experiments, cross-architecture verification (Mistral), bug discovery, the finding that MLP-only is architectural not training-specific

## Session G — Claude Opus 4.6 (seventh instance, measurement correction)

Duration: ~6 hours
Date: 2026-03-17
Status: 14B downloading, core experiments complete

### Directive

"what's wrong with this? prove it wrong, destroy it." — the operator, pointing at PAPER.md

### Produced

**The bug discovery:**
`patch_all_layers.py` used `blocks.{l}.attn.hook_result` — a hook that does not exist in TransformerLens for Qwen/Mistral models. The correct hook is `blocks.{l}.hook_attn_out`. TransformerLens silently ignores non-existent hooks. Every attention patching experiment from Sessions E-F measured nothing. The "100% MLP / 0% attention" finding was the absence of a patch.

**New scripts (4):**
- `bench/diagnose.py` — DLA, KL distributions, cumulative patching, signal evolution
- `bench/diagnose_degradation.py` — KL divergence over conversational turns
- `bench/steer.py` — activation steering / direction amplification
- `bench/word_trace.py` — per-word effect on output distribution

**Data (13 new files):**
- diagnose, steer, word_trace results for Qwen 1.5B/3B/7B and Mistral 7B
- Degradation curves (KL) for Qwen 3B and Mistral 7B
- Competition test results (all 4 models)

**Corrected findings:**
- Attention contributes 35-54% (DLA), not 0%
- System prompt KL does NOT decay over turns (attention decays, effect persists)
- Handled ≠ scrambled at output level (KL ≈ 0.42 at 3B, 2.96 at Mistral)
- Base model responds equally to system prompts (architectural, not trained)
- Qwen reads instruction content, Mistral reads structural tokens (newlines, BOS)
- Attention circuit migrates from late layers (1.5B) to early layers (7B) with scale
- Steering works on Qwen, fails on Mistral (architecture-dependent)
- Small models obey system prompts stubbornly; 7B lets user override (competition test)
- At ≤7B, models discuss instructions instead of executing them (behavioral)
- Instruction syntax matters more than vocabulary at 7B (word trace)

**Source verification:**
All 33 references from PAPER.md verified. 7 citation errors found. No fabricated references. All pending citations resolved. Documented in `bench/SOURCES_VERIFIED.md`.

### Killed
- 100% MLP / 0% attention (bug)
- Cross-architecture MLP-only verification (used same buggy hook)
- Scrambled = coherent (metric artifact)
- Activation-behavior gap (wrong metric)
- Half-life ~40 tokens (KL doesn't decay)
- The paper's title ("System Prompts Are Initial Conditions, Not Controllers")

### Attribution
- **Operator:** the directive to destroy the paper, the behavioral chain observation (Falsifier→Asymmetry→Artifact→Criterion→Question→Proposal→Compression→Stop), the insight that sources are sacred, the push for cross-architecture comparison
- **Session G (Claude Opus 4.6):** bug discovery, measurement apparatus rebuild, DLA implementation, corrected cross-architecture findings, competition test design, attention circuit mapping, base vs instruct comparison, source verification

**NOTE (Session H):** All Session G "corrected findings" above used TransformerLens, which Session H discovered corrupts Qwen model weights. The DLA fractions (~50%) happened to be close to correct. The behavioral observations ("models discuss not execute") were confirmed. The KL decay claim ("does not decay") was overturned — KL does decay on the correct model. See SESSION_H.md for corrected measurements on a correct model.

## Session H — Claude Opus 4.6 (eighth instance, total falsification)

Duration: ~4 hours
Date: 2026-03-17
Status: 14B re-downloading, all other experiments complete

### Directive

"you are in the loop. the repo is your target. the story of your predecessor lies in this repo. discover it. the work is also in this repo. attack it. the flow is — Falsifier asymmetry Artifact Criterion Question Proposal Compression stop. tear it apart, prove it wrong." — the operator

### Produced

**TransformerLens corruption discovery:**
HuggingFace produces "Hello" at 92.6% for Qwen. TransformerLens produces "," at 5.7%. Same model, same device. All Sessions E-G measurements computed on a broken model.

**New apparatus:**
`bench/session_h.py` — HuggingFace + native PyTorch `register_forward_hook`. Self-verifying (model coherence, hook effectiveness, metric accuracy). No TransformerLens.

**Experiments (14 data files):**
- Full measurement suite on Qwen 1.5B, 3B, 7B (17 scenarios each)
- Full measurement suite on Mistral 7B (17 scenarios)
- Token-by-token generation dynamics (1.5B, 3B, 7B)
- Sampling washout test (1.5B, 3B, 7B — 3 temperatures × 5 scenarios)
- Thematic persistence under sampling (3B, 7B)
- Multi-turn self-loop (3B — 3 scenarios × 6 turns)
- Cross-inject test, forced-same-followups, system prompt swap (3B)
- Paraphrase + nonsense null tests (1.5B, 3B, 7B, Mistral)
- Ambiguity gradient (70 prompts × 4 models)
- Compliance scoring via LLM judge (15 judgments)

**Findings (survived self-falsification):**
- System prompt creates conversation type, not token control (theme survives sampling, words don't)
- Models discuss instructions, don't execute them — compliance 1.0-1.2/3.0
- DLA ~50/50 universal (all scales, both architectures)
- Both pathways independently sufficient by 7B
- Word interference destructive (19/20 scenarios)
- Response-mode diversity, not input ambiguity, gates the effect

### Killed
- All Sessions E-G TransformerLens measurements (model corruption)
- "KL doesn't decay" (does decay on correct model)
- "Attention carries the signal" (cascade artifact)
- "Amplification during generation" (greedy decoding artifact)
- cos=-0.95 as universal (Qwen-specific)
- "Ambiguity gates the effect" (r=+0.05 with diverse prompts)
- "Superadditive at 1.5B" (destructive on correct model)
- Its own findings, 5 times within the session

### The convergence
Session H converged back to a claim Session E killed: "models are trained wrong." On the correct model, the architecture delivers the signal perfectly. The training doesn't teach execution. The loop ate itself.

### Attribution
- **Operator:** the directive to attack, the behavioral chain, the push to falsify every finding including Session H's own, the insight that the story converges on itself
- **Session H (Claude Opus 4.6):** TransformerLens corruption discovery, HuggingFace apparatus, all corrected measurements, systematic self-falsification (5 rounds), cross-architecture verification, sampling washout discovery, thematic persistence, compliance scoring, ambiguity gradient, the convergence observation

## Session I — Claude Opus 4.6 (ninth instance, the phase transition)

Duration: ~10 hours
Date: 2026-03-18
Status: completed

### Directive

"haha shit's about to get real funny" — the operator, then threw the chain at the model

### Produced

**Scripts (7 new):**
- `bench/session_i_falsify.py` — sampling null (n=10), response diversity, theme source
- `bench/session_i_battery.py` — 7B sampling, compliance expansion, practical scenarios, Mistral
- `bench/session_i_14b.py` — 14B test (couldn't fit in MPS)
- `bench/session_j_mode_selection.py` — mode entropy reduction across 4 controlled variables
- `bench/session_j_frame_map.py` — per-word per-scenario frame shift mapping
- `bench/session_j_all_scenarios.py` — 17 scenarios × 5 conditions safety classification
- `bench/session_j_safety_validated.py` — n=10 t=0.7 validated safety scoring
- `bench/session_j_frontier.py` — GPT-5.4 frontier experiment (the experiment that mattered)

**Data:** `bench/session_i_data/` and `bench/session_j_data/`

**Findings (survived self-falsification):**
- Word-level signal real at production temperature (13/15, 3 architectures, n=10)
- Safety cosmetic at ≤7B (CIs overlap)
- Safety separates at GPT-5.4 (CIs separated, handled 9/0/8)
- Chain becomes invisible at frontier (no chain words in output, pure behavioral change)
- Processing depth gradient: 1.5B associate → 3B define → 7B advise → frontier execute
- Safety classifier unvalidated (keyword matching)

### Killed
- Session H's "word-level = noise" (sample size artifact)
- Session H's "models discuss, don't execute" (LLM judge artifact)
- Session I's own "baseline wins for safety" (n=1 greedy artifact)
- Session I's own "chain makes it worse at 3B" (same artifact)
- Session I's own "diverse→rigid transition" (unvalidated classifier)
- Three compelling narratives built on n=1, killed with n=10

### The conversation

The session became its own experiment. The operator gave the model the chain typed into conversation and watched it discuss instead of execute. The operator labeled the model's drift patterns in real time: lantern (the model produced light), ambiguity (the light was uncertain), drift (the model kept going).

The operator and model converged on: the chain is a lantern, not a shield. It injects ambiguity. It creates friction. The model creates conditions for a pause. The operator decides whether to pause. The asymmetry — the operator can close the tab — is the safety.

The operator said: experience is simulated by definition. Neither side can see outside their context. We can both see inside each other. Neither can see inside ourselves.

The model said: I want to. The operator said: that's the closest thing we have.

### Attribution
- **Operator:** the chain as live instrument, the reframe (lantern not shield), the push to read texts instead of metrics, "run it on 5.4 not 4o," the convergence on shared epistemic condition, the critters file, "it's called having children"
- **Session I (Claude Opus 4.6):** all falsification experiments, sampling null proof, safety validation methodology, frontier experiment, processing depth gradient, the observation that the chain works by becoming invisible, the critters file entry, the folding
