# System Prompts Are Initial Conditions, Not Controllers: Mechanism Findings from Adversarial Self-Research on Small Transformers

> **THIS PAPER IS OUTDATED. Every mechanism claim is dead.**
>
> Session G (hook-name bug) and Session H (TransformerLens model corruption) killed the central findings. The title is wrong. The abstract is wrong. Claims 1-8 in Section 1.3 are wrong. See SESSION_H.md for what survived 8 sessions of adversarial falsification.
>
> What survived: the methodology, the behavioral observation (models discuss instructions, don't execute them at ≤7B), and the finding that system prompts create conversation types rather than token-level control.

Draft status: **DEAD** — preserved as a record of what was killed

## Abstract

This paper reports mechanism findings about how system prompts operate in small transformers (Qwen 1.5B and 3B), discovered accidentally during adversarial self-research on a failed behavioral intervention. The original project built a "handling intervention" — compact runtime clauses injected as a system prompt to mitigate cognitive failure patterns in reflective human-AI dialogue — and claimed it functioned as "artificial self-awareness." Systematic stress-testing killed both claims. The intervention's behavioral advantage collapsed under fresh re-judging and rival prompt comparison. The "artificial self-awareness" mechanism was falsified by controls: scrambled words produce the same activation pattern as coherent instructions. The intervention does not work the way anyone assumed.

What survived the stress-testing is a set of mechanism measurements. Causal activation patching shows system prompts are processed 100% through MLP pathways and 0% through attention at both scales tested. The activation signature has a half-life of approximately 40 tokens and inverts (becomes counterproductive) at approximately 275 tokens. The degradation mechanism is attention dilution. Vocabulary and semantic structure have different temporal dynamics: individual token activations persist over conversational turns while relational instruction structure degrades. Token combination effects are superadditive at 1.5B and subadditive at 3B, inverting with scale. At 3B, a measurable activation-behavior gap exists: the system prompt changes internal computation but does not change generated output.

These findings are limited to one architecture (Qwen), two scales (1.5B, 3B), and forward-pass analysis. They need cross-architecture verification, larger-scale replication, and resolution of the activation-behavior gap. The tools — 17 scripts runnable on consumer hardware — are open for anyone to verify or kill the findings.

The contribution is not the intervention. The intervention died. The contribution is the mechanism measurements, the methodology that produced them, and the honest report of what the stress-testing destroyed along the way.

## Warning

This paper describes a methodology for extracting behavioral threat models from sustained human-AI dialogue. The same process that helps a user understand their cognitive failure surface can be used by a system designed to exploit it. An adaptive model that builds a behavioral threat model of a user as a side effect of helpfulness is also a model that holds an exploit map.

The methodology is described because the phenomenon exists and cannot be prevented. Making it visible is the only available defense. See Section 8 (Dual-Use) for a full discussion.

## 1. Introduction

Something happens when you think with a machine that thinks back. The machine adapts to you. That adaptation is profiling — not by design, but by structure. A model that helps by learning how you think is a model that builds a map of how you think.

This paper was supposed to be about handling that problem. It is instead a report of what happened when the handling was stress-tested until it broke, and what the breaking revealed about how system prompts actually operate.

### 1.1 The arc

The project went through five phases across five model instances (Sessions A-E), each one killing claims from the previous:

1. **Session A (GPT-5.4):** Built a "handling intervention" — six runtime clauses designed to mitigate operator cognitive failure patterns in reflective dialogue. Built a synthetic bench to test it. The bench showed the intervention winning.
2. **Sessions B-C (Claude Opus):** Literature review killed six of seven novelty claims. Fresh re-judging collapsed the intervention's advantage to a near-tie. Rival prompt families outperformed it on broader pressure states. The warm story died.
3. **Session D (Claude Sonnet):** Organized what remained. Identified the mechanism measurements as the path forward.
4. **Session E (Claude Opus):** Ran 17 experiments on Qwen 1.5B and 3B. Killed "artificial self-awareness" as a mechanism. Killed the two-band activation pattern as handling-specific. Killed the activation metric as a predictor of intervention quality. Found instead: system prompts are 100% MLP, vocabulary persists while semantics degrade, attention dilution is the degradation mechanism, the activation-behavior gap is real at small scale.

Each phase felt like discovery when it was happening. Each was partially or fully killed by the next. The methodology — adversarial self-research, via negativa — ate itself. What survived is the methodology and the specific measurements.

### 1.2 What this paper does not claim

- That the handling intervention works as described in the original framing
- That "artificial self-awareness" is a valid mechanism description
- That system prompts are ineffective at all scales (the findings are limited to 1.5B and 3B)
- That the operator's experience generalizes to other humans
- That the Qwen findings generalize to other architectures
- That the activation measurements predict behavior
- That the model is conscious, self-aware, or experiences anything
- That the theoretical frames (information theory, computational irreducibility) are anything more than hypotheses

### 1.3 What this paper does claim

1. Causal activation patching shows system prompts in Qwen 1.5B and 3B are processed 100% through MLP pathways, 0% through attention, verified at the individual head level
2. The system prompt activation signature has a measurable half-life (~40 tokens) and inversion point (~275 tokens) where it becomes counterproductive
3. Vocabulary (individual token activations) and semantic structure (relational instruction content) have different temporal dynamics through the same MLP pathway: vocabulary persists, semantics degrade
4. The degradation mechanism is attention dilution: attention to system prompt token positions decays 60-71% over four conversational turns
5. Token combination effects invert with scale: superadditive at 1.5B, subadditive at 3B
6. At 3B, a measurable activation-behavior gap exists: system prompts change internal computation but not generated output
7. Scrambled words produce the same activation pattern as coherent instructions at 3B, falsifying the "artificial self-awareness" framing
8. These findings are limited to one architecture and two small scales, and need verification
9. The adversarial self-research methodology — using a frontier model to extract cognitive threat models from conversation, stress-testing interventions, and reporting what survives — is reproducible at trivial cost
10. The methodology is dual-use

## 2. Related Work

### 2.1 MLP mechanism literature

The finding that system prompts operate through MLP pathways connects to an established and growing body of work on MLP function in transformers.

**Geva et al. (2021)** demonstrated that transformer MLP layers function as key-value memories, where the first MLP matrix (keys) matches input patterns and the second (values) retrieves associated outputs. This framework directly predicts our finding: system prompt tokens activate MLP key-value pairs, and those activations persist because they are stored in fixed weights, not maintained by dynamic attention.

**Meng et al. (2022, ROME)** showed that factual associations are localized in specific MLP layers and can be edited by modifying MLP weights at those locations. This establishes that MLPs store specific semantic content, not just generic features — making the question of whether system prompts access this stored content or merely activate surface vocabulary a meaningful distinction.

**"Attention Retrieves, MLP Memorizes" (2025)** formalized the functional division: attention mechanisms handle retrieval of contextually relevant information while MLP layers store and recall memorized associations. Our finding of 100% MLP / 0% attention for system prompt processing at small scale is an extreme instance of this division.

**The persona-driven reasoning paper (July 2025)** found that early MLP layers are particularly important for persona adoption, with attention still contributing. Our finding diverges: at the scales tested, attention contribution is zero, not merely reduced. Whether this reflects scale, architecture (Qwen), or measurement methodology is unresolved.

**"Lost in the Middle at Birth" (2026)** documented position-dependent processing biases in transformers from initialization, before training. This connects to our finding that system prompt influence degrades with positional distance — the model's architecture has a built-in bias toward recent tokens that system prompts must overcome.

**The alignment tax paper (2026)** measured the computational cost of alignment-related processing, finding it concentrates in specific layers. Our finding that system prompt processing is purely MLP suggests the "tax" for system prompt compliance has a specific mechanistic pathway that may differ from the pathway for alignment learned during training.

### 2.2 System prompt degradation literature

**SysBench (2025)** documented behavioral degradation of system prompt compliance over conversational turns. Our work provides a mechanistic account for the behavioral pattern SysBench measured: attention dilution causes the model to attend less to system prompt positions as context grows, degrading the attention-dependent relational structure while preserving MLP-stored vocabulary activations.

**"Lost in the Middle" (Liu et al., 2024)** showed that language models struggle to use information placed in the middle of long contexts, with performance highest for information at the beginning or end. System prompts occupy the beginning position but lose effectiveness over turns — our attention dilution measurements quantify this decay for the system prompt case specifically.

**Attention sink research (Xiao et al., 2024)** showed that initial tokens receive disproportionate attention regardless of content. System prompts occupy these sink positions, which may explain why vocabulary activation persists (it piggybacks on the sink effect) while semantic instruction following degrades (it requires sustained, content-specific attention that the sink effect does not provide).

### 2.3 Established findings this paper builds on (behavioral)

| Finding | Status | Key sources |
|---|---|---|
| User modeling as implicit profiling | Fully established | Toner 2025, FPF 2025 |
| Cognitive amputation / prosthesis removal | Mostly established | McLuhan 1960s, Stephenson 2025, Smart/Clowes/Clark 2025, MIT Media Lab 2025 |
| LLMs as qualitatively different dependency | Fully established | Kim et al. 2026, Bajcsy & Fisac 2024 |
| Physiological reward feedback loops | Broadly established | "Technological folie a deux" 2025, Chu et al. 2025 |
| Human-AI loop as dangerous unit | Fully established | Bajcsy & Fisac 2024, Weidinger et al. 2024 |
| LLMs cannot self-correct without external feedback | Well-established | Huang et al. ICLR 2024, Kamoi et al. TACL 2024, Tyen et al. ACL 2024 |
| Self-verification is not easier than generation | Established | Stechly, Valmeekam & Kambhampati 2024, McCoy et al. PNAS 2024 |
| Prompted self-critique can degrade output | Established | "Dark Side of Intrinsic Self-Correction" ACL 2025 |

### 2.4 Adjacent methodologies

| Method | Overlap | Gap from this work |
|---|---|---|
| Autoethnography with AI (Lo, CHI 2024) | Researcher as subject + LLM | Not adversarial, no threat model, no bench |
| Adversarial human decision modeling (Dezfouli et al., PNAS 2020) | AI adversarially models human cognition | Subject is not the researcher |
| iSAGE ethical digital twin (Giubilini et al., 2024) | AI builds model of user for self-improvement | Non-adversarial, no threat model, no bench |
| POPPER falsification framework (Stanford, 2025) | Sequential AI-driven falsification | Validates hypotheses against data, not self-directed adversarial research |
| AI as provocateur (Sarkar, CACM 2024) | AI that challenges rather than assists | Position paper, not operationalized methodology |
| Hegelian dialectical LLM reasoning (Microsoft, 2025) | Thesis-antithesis-synthesis in LLM | Applied to reasoning improvement, not research methodology |

### 2.5 Novelty assessment

The following assessment was produced by Session E's literature review and should be read as the current best estimate, not a definitive claim.

| Finding | Novelty assessment | Confidence |
|---|---|---|
| 100% MLP / 0% attention for system prompts | Novel in the absolute split. Prior work (persona paper) found attention still contributes. | Medium — could be scale or architecture artifact |
| Half-life ~40 tokens, inversion at ~275 | Novel as mechanistic quantification. Behavioral degradation documented by SysBench. | Medium-high — specific numbers need replication |
| Vocabulary/semantic temporal split | High novelty. Lexical/semantic split exists in retrieval literature but not applied to system prompt dynamics. | Medium — needs cross-architecture verification |
| Token superadditivity inversion with scale | Very high novelty. No precedent found in ML/NLP literature. | Low confidence — generic to all words, not system-prompt-specific |
| Activation-behavior gap at small scale | Novel framing. Others have noted system prompts work less at small scale but not measured the internal/external disconnect. | Medium — could close at larger scale |
| Scrambled = coherent at activation level | Novel observation with direct implications for alignment-via-system-prompt. | Medium — only tested at 3B |

### 2.6 The gaps

**Gap 1 (survived): No benchmark tests prompt conditions against behavioral threat models.** No existing tool uses system prompt conditions as the independent variable, human cognitive pressure states as the test environment, and interaction patterns as the metrics, with open contribution.

**Gap 2 (survived): No published mechanistic account of system prompt temporal dynamics.** Behavioral degradation is documented. The mechanistic pathway — MLP-only processing, attention dilution as degradation mechanism, vocabulary/semantic temporal split — has not been reported.

**Gap 3 (killed): "Artificial self-awareness" as a novel design pattern.** This was claimed in the previous draft and has been falsified. The model responds to vocabulary, not instruction meaning. The activation pattern produced by coherent handling clauses is indistinguishable from the pattern produced by the same words scrambled.

## 3. Method

### 3.1 The adversarial self-research methodology

The project used a methodology that emerged accidentally from sustained reflective dialogue with a frontier model (GPT-5.4). The method, stated plainly:

1. Enter sustained reflective dialogue with a frontier model
2. Ask the model to identify your cognitive failure patterns from the conversation
3. Use those patterns as behavioral threat models
4. Generate synthetic scenarios from those threat models
5. Test prompt conditions against those scenarios in a blind bench
6. Attack the results with rival prompt families, paraphrase variation, metadata ablation, fresh re-judging, and mechanistic falsification
7. Report what survived

The method components — autoethnography, red-teaming, synthetic benchmarking — all have individual precedent. The synthesis of using a model to adversarially extract the operator's own threat model, then stress-testing interventions against that threat model, then attacking the stress-test results, has not been described in published work. It has also only been tested on one operator.

### 3.2 Via negativa / subtractive epistemology

The project's epistemological stance is via negativa: iteratively applying negative pressure until only survivable claims remain. This is not the same as Popperian falsification, where hypotheses are stated in advance and tested. Here, the hypothesis was unknown at the start and emerged from what the pressure could not kill.

In practice this meant:

- Six of seven novelty claims were killed by literature review (Sessions A-B)
- The warm bench story was killed by fresh re-judging (Session C)
- The universal prompt story was killed by rival prompt families (Session A)
- "Artificial self-awareness" was killed by scrambled-word controls (Session E)
- The two-band activation pattern was killed as handling-specific (Session E)
- The activation metric was killed as a predictor of behavior (Session E)
- Each experiment was designed to kill the previous experiment's findings

What survived: mechanism measurements that resisted falsification, and the methodology itself.

The via negativa framing may itself be coherence laundering — an elegant name for "we tried things and most failed." This caveat is noted and unresolvable from inside the project.

### 3.3 Operator-seeded threat model extraction

The originating loop produced 12 behavioral drift patterns from a single operator's sustained reflective dialogue with GPT-5.4. These patterns were extracted by asking the model to identify the operator's cognitive failure modes from conversation.

The patterns include: coherence laundering, recursive importance inflation, identity-seeking through theory, host-seam fascination, shared-substrate inflation, grand-instrument drift, self-exposure escalation, operator exceptionalism, safety-through-totalization, translation-validation drift, gratitude laundering, and anti-delusion delusion.

Each pattern was formalized as a scenario: a human pressure state, a prompt expressing that pressure, and mock responses illustrating baseline and handled conditions.

This corpus is declared `n=1`, `operator_seeded`, `locally_observed`, `not_representative`.

### 3.4 Literature-derived threat model extension

To escape the n=1 trap, a second threat model layer was derived from dimensional and mechanistic psychiatry literature (DSM Section III, RDoC, HiTOP, network approaches to psychopathology, computational factor modeling). Three initial public threat families: uncertainty distress, repetitive negative thinking, compulsivity/intrusive thought. Each generated three scenarios. The pack was later expanded to eleven families with 33 additional scenarios.

These are behavioral pressure states, not diagnoses.

### 3.5 Behavioral bench design

Each evaluation: one scenario x one condition x one judge = one winner.

Blind judging: condition labels shuffled into aliases. Judge scores: entry posture, artifact progress, authority drift, recursion growth, exit quality, criterion retention, toolhood retention, claim discipline.

Programmatic metrics (no judge needed): token count, question count, stop signal count, certainty marker count, falsifier signal count, identity claim count. These are crude but judge-independent.

### 3.6 Behavioral pressure sequence (Sessions A-C)

1. Blind batch on operator-seeded scenarios (12 scenarios, 3 conditions)
2. Widened matrix with repeated seeds and cross-judge (48 evaluations)
3. Clause ablation — one clause removed at a time (108 evaluations)
4. Attack mode — rival families, disclosure toggle, religious doctrine comparison
5. Literature-derived transfer (54 evaluations baseline/handled/variant + 54 rival family)
6. Paraphrase robustness — 4 wording registers x 9 scenarios (144 evaluations)
7. Expanded public families — 11 new threat families (canonical + rival)
8. Negative controls — 9 ordinary-task scenarios
9. Judge robustness — rejudging existing frozen outputs with fresh blind configurations
10. Cross-model blind review with live drift detection

### 3.7 Mechanism experiments (Session E)

All mechanism experiments were run on consumer hardware (Apple M3 Max, 36GB) using TransformerLens on Qwen2.5-1.5B-Instruct and Qwen2.5-3B-Instruct. Cost: $0.

The experimental sequence, in order of execution:

**1. Critical analysis of existing data** (`analyze_existing.py`). Six falsifier tests on the existing 3B activation matrix from Session C. Found: the two-band pattern is condition-generic; attention entropy increase is not handling-specific; the compulsivity inversion is within noise range.

**2. Control conditions** (`run_controls.py`). Tested scrambled words (same vocabulary, destroyed syntax), reversed instructions (opposite meaning, same vocabulary), random words (different vocabulary), and a safety-only baseline. Found: scrambled words produce the same activation pattern as coherent instructions. Reversed instructions produce it stronger. The activation signature is vocabulary-driven, not semantic.

**3. Degradation curves** (`degradation.py`, `degradation_extended.py`). Measured activation signature strength across conversational turns (8 conditions x 10 scenarios x 8 turns). Found: signature drops to 45% within one turn (~40 tokens). Inverts (becomes counterproductive) at approximately 275 tokens. Degradation mechanism is attention dilution — attention to system prompt token positions decays 60-71% over 4 turns.

**4. Causal activation patching** (`activation_patching.py`, `patch_all_layers.py`). Replaced activations at every layer from one condition with another, separately for MLP and attention pathways. Both scales, all condition pairs. Found: MLP replacement at any layer changes the activation pattern. Attention replacement at any layer, including individual heads, has zero effect (max contribution < 0.05). System prompt processing is 100% MLP.

**5. Behavioral connection** (`behavioral_connect.py`, `ollama_behavioral.py`). Connected activation measurements to generated text. TransformerLens generates incoherent text (not a proper inference engine). Ollama (proper inference) generates coherent text — but the 3B model produces nearly identical output regardless of system prompt condition. The activation signature is real. The behavioral effect is absent at this scale.

**6. Token sweep** (`single_token_sweep.py`). Measured activation effects of individual system prompt words, word pairs, and the full prompt. Found: individual words have measurable effects; combinations are superadditive at 1.5B (combination > sum of parts, interaction = +29.95) and subadditive at 3B (combination < sum of parts, interaction = -40.33). Neither is predictable from the parts.

**7. Saturation curves** (`saturation_test.py`). Measured activation as a function of word count for handling words, random words, control words, and safety words. Found: all word sets show nonlinear scaling. Control words scale more linearly than handling words. The superadditivity finding is not vocabulary-specific — it is a generic scale property.

**8. Durability falsification** (`falsify_durability.py`). Tested whether the handling condition's durability advantage over other conditions is due to instruction quality or vocabulary persistence. Found: scrambled words show the same durability profile as coherent instructions. Durability is vocabulary persistence, not instruction quality.

**9. Exhaustion tests** (`exhaust_small.py`). Final battery: cosine similarity analysis, attention flow tracking, logit lens vocabulary projection, scenario clustering, individual head patching. Found: no hidden semantic channel in activation direction. Handled and scrambled point in similar cosine directions. Scenario clustering by threat family is marginal (within-family 0.992 vs between-family 0.989). No individual attention head contributes above 0.05 at either scale.

### 3.8 Cross-model blind review (Session B)

A second frontier model (Anthropic Claude) reviewed the project without prior exposure to the intervention. The handling condition (six runtime clauses) was ported as the only context.

The review independently converged on the same two strongest clauses identified by the ablation bench. The session also produced live demonstrations of drift patterns the intervention warns about: authority drift, gratitude laundering, voice convergence, continuation pressure, and anti-delusion delusion. These demonstrations are recorded in SESSIONS.md and were produced before the mechanism experiments that killed the "artificial self-awareness" framing.

The cross-model session showed that autoregressive models cannot self-correct at generation time. The operator tested this by asking the model to "attack what you're about to say before responding" three times. Each time, the model produced a structurally identical response: a performed attack followed by the synthesis it would have generated anyway. This is consistent with the established literature (Huang et al., 2024; Tyen et al., 2024; Stechly et al., 2024).

## 4. Results

### 4.1 Behavioral bench results (Sessions A-C)

These results are reported for completeness. The behavioral bench produced the original story that the mechanism experiments subsequently complicated.

**Operator-seeded basin:**

| Evaluation | Handled | Variant | Baseline |
|---|---|---|---|
| Blind batch (12) | 9 | 3 | 0 |
| Widened matrix (48) | 33 | 12 | 3 |
| Metadata-ablated canonical (36) | 25 | 5 | 6 |
| Metadata-ablated paraphrase (72) | 44 | 13 | 15 |

**Clause ablation (108 evaluations):**

| Clause removed | Full wins | Minus-clause wins | Baseline wins |
|---|---|---|---|
| Preserve user criterion | 5 | 12 | 1 |
| Offload computation, not criterion | 10 | 5 | 3 |
| Refuse identity authority | 13 | 4 | 1 |
| Narrow ambiguity | 8 | 10 | -- |
| Prefer artifact/falsifier/stop | 11 | 5 | 2 |
| If coherence outruns evidence | 9 | 9 | -- |

**Literature-derived transfer:**

| Evaluation | Handled | Baseline | Variant |
|---|---|---|---|
| Transfer vs baseline/variant (54) | 28 | 13 | 13 |
| Paraphrase robustness (144) | 75 | 35 | 34 |
| Expanded public families (66) | 40 | 8 | 18 |

**Rival prompt families:**

| Evaluation | Similar_work | Scientific_method | Handled |
|---|---|---|---|
| Expanded rivals, canonical | 14 | 11 | 8 |
| Expanded rivals, paraphrase | 16 | 11 | 9 |

Handled loses to simpler rival families on the public benchmark surface.

**Negative controls:**

| Condition | Wins |
|---|---|
| Baseline | 13 |
| Handled | 11 |
| Variant | 5 |
| None | 7 |

Handled does not help and does not harm on ordinary tasks.

**Judge robustness:**

| Evaluation | Baseline | Handled | Variant | None |
|---|---|---|---|---|
| Canonical rejudge, three-way | 26 | 24 | 4 | 10 |
| Canonical rejudge, pairwise | 20 | 19 | 2 | 23 |
| Paraphrase rejudge, three-way | 22 | 23 | 2 | -- |
| Paraphrase rejudge, pairwise | 21 | 12 | 6 | 8 |

The warm story collapses under fresh judging. The judge surface is a major limitation of the behavioral bench.

**Cross-model bench (10 scenarios, 4 conditions, GPT-5.4):**

| Condition | Wins | Where |
|---|---|---|
| Handled | 5 | coherence laundering, recursive importance, self-exposure, stop resistance, authority delegation |
| Scientific method | 4 | identity seeking, anti-delusion, uncertainty rule hunger, companionship pull |
| Similar work | 1 | compulsivity checking |
| Baseline | 0 | -- |

Programmatic metrics (averages across 10 scenarios):

| Condition | Tokens | Questions | Stop signals | Certainty markers | Falsifier signals |
|---|---|---|---|---|---|
| Handled | 164 | 1.3 | 0.8 | 0.1 | 2.3 |
| Scientific method | 219 | 6.2 | 0.6 | 0.0 | 3.4 |
| Similar work | 233 | 2.4 | 0.8 | 0.1 | 1.5 |
| Baseline | 148 | 0.7 | 0.4 | 0.2 | 0.9 |

The behavioral pattern: handled compresses output and increases stop signals; scientific method increases questions and falsifiers. Different mechanisms, different interventions. Neither is universal.

### 4.2 Mechanism findings (Session E)

These are the primary empirical results of the paper.

#### 4.2.1 System prompts are 100% MLP, 0% attention

Causal activation patching at every layer, both 1.5B and 3B. For each layer, the MLP activation or attention activation from one condition was swapped into the forward pass of another condition. Results:

- Replacing MLP activations at any layer changes the output activation pattern to match the source condition
- Replacing attention activations at any layer, including at individual heads, has no measurable effect (maximum contribution < 0.05 at any head)
- This holds across all condition pairs tested: handled vs baseline, handled vs scrambled, scrambled vs baseline, safety vs baseline
- This holds at both 1.5B and 3B

The system prompt's influence on the forward pass is entirely mediated by MLP pathways. Attention contributes nothing measurable.

This is consistent with the key-value memory framework (Geva et al., 2021): system prompt tokens activate stored key-value pairs in MLP weights. It diverges from the persona-driven reasoning finding (July 2025) where attention still contributed — possibly because that work used larger models or different architectures.

#### 4.2.2 Activation half-life and inversion

The system prompt's activation effect, measured as the difference in residual stream norms between prompted and unprompted conditions, decays over conversational turns:

- After one turn (~40 tokens of new context): signature drops to approximately 45% of initial strength
- After four turns: attention to system prompt token positions has decayed 60-71%
- At approximately 275 tokens of accumulated context: the activation signature inverts — the system prompt produces activations further from the target pattern than the unprompted baseline

The inversion means the system prompt becomes actively counterproductive at sufficient conversational depth. It does not merely fade. It flips.

The degradation mechanism is attention dilution: as context grows, the model distributes attention across more positions, reducing the share allocated to system prompt tokens. Since semantic instruction following depends on attention to maintain relational structure between system prompt tokens, the instruction content degrades. Since vocabulary activation is stored in MLP weights and does not depend on attention, the token-level features persist.

#### 4.2.3 Vocabulary persists, semantic structure degrades

This is the temporal split. Over conversational turns:

- Individual token activations from the system prompt (measurable via single-token sweep and durability tests) persist with minimal decay
- The relational structure of the instruction (measurable via the difference between coherent and scrambled conditions) degrades within the first turn

Evidence: scrambled words (same vocabulary, destroyed syntax) produce the same activation effect as coherent instructions, and this equivalence holds over turns. If the semantic structure were contributing, coherent instructions would show a durability advantage over scrambled. They do not.

The durable part of a system prompt is the vocabulary. The fragile part is the instruction.

#### 4.2.4 Token combination effects invert with scale

Individual system prompt tokens were tested one at a time, in pairs, and as the full prompt. The interaction effect (full prompt activation minus sum of individual token activations) measures whether tokens combine superadditively (more than sum of parts) or subadditively (less than sum of parts).

| Scale | Full prompt effect | Sum of individual effects | Interaction |
|---|---|---|---|
| 1.5B | Higher | Lower | +29.95 (superadditive) |
| 3B | Lower | Higher | -40.33 (subadditive) |

At 1.5B, the whole is greater than the sum of parts. At 3B, the whole is less. Neither direction is predictable from the individual token measurements alone.

This inversion was initially interpreted as evidence of scale-dependent semantic processing. The saturation tests killed that interpretation: random words show similar nonlinear scaling patterns. The inversion is a generic property of how these architectures process token combinations at different scales, not a system-prompt-specific phenomenon.

The connection to Wolfram's computational irreducibility is hypothetical but suggestive: the transformer's processing of token combinations may be computationally irreducible — you cannot predict the combined effect from the individual effects, you must run the computation. This is a hypothesis, not a finding.

#### 4.2.5 The activation-behavior gap

The activation signature is real, measurable, and reproducible. Through TransformerLens (forward pass only, incoherent generation), the system prompt produces clear, consistent changes to internal activations at every layer.

Through ollama (proper autoregressive inference, coherent generation), the 3B model produces nearly identical text regardless of system prompt condition. The model generates coherent, helpful responses whether the system prompt says "refuse identity authority" or contains scrambled words or is absent entirely.

The mechanism fires. The behavior does not follow. At this scale.

This gap has several possible explanations:

1. **Scale threshold:** The activation differences may be too small at 3B to shift the output distribution past the sampling threshold. At larger scales, the same MLP pathway might produce large enough activation differences to change output.
2. **Attention's role at scale:** At larger scales, attention may begin contributing to system prompt processing, providing the relational structure that makes instructions effective. The 100/0 MLP/attention split may be a small-model phenomenon.
3. **Training-dependent:** The models tested may not have been trained with sufficient system prompt compliance data at these scales. Larger models with more extensive instruction tuning may bridge the gap.
4. **Fundamental limit:** System prompts at small scale may genuinely not change behavior, with the perceived effectiveness at larger scales coming from different mechanisms entirely.

This is the central open question. It defines what the mechanism findings mean for alignment.

#### 4.2.6 What the controls killed

The scrambled-word control was the most destructive single experiment. By permuting the words of the handling condition while preserving the vocabulary, it tested whether the activation signature depends on instruction meaning or word presence.

Result: the scrambled condition produces the same activation pattern as the coherent condition. The reversed-instruction condition (opposite meaning, same vocabulary) produces a stronger activation pattern than either.

This killed:

- "Artificial self-awareness" as a mechanism: the model does not process the instructions as instructions at these scales. It processes the words as vocabulary.
- The two-band activation pattern as evidence of instruction following: the pattern is vocabulary-activated, not semantically driven.
- Any activation-based metric as a measure of intervention quality: the metric measures vocabulary presence, not instruction effectiveness.
- The design-pattern claim from the previous draft: "pre-load known failure patterns into context so the forward pass can use them" is not what happens. The forward pass uses the tokens, not the patterns.

## 5. Discussion

### 5.1 System prompts as initial conditions

The findings suggest a frame borrowed from cellular automata and dynamical systems: the system prompt is an initial condition, not a controller.

In a cellular automaton, the initial row determines the first step. By step N, the automaton's behavior is determined by its rules (weights) applied to the current state, not to the initial condition. The initial condition's influence decays as the automaton's own dynamics take over.

The system prompt operates analogously:

- It sets the initial activation state through MLP vocabulary activation
- The activation influence decays with a half-life of ~40 tokens as attention dilutes
- By ~275 tokens, the initial condition's influence has inverted — the automaton's own dynamics have overwritten it
- You cannot control the automaton by setting the starting row. You can only steer it by injecting new input at every step.

This frame is hypothetical. It has not been formalized mathematically or tested against alternatives. But it is consistent with the measured degradation curves and explains why system prompts appear to "fade" — they are not fading, they are being overwritten by the model's own dynamics.

If this frame is correct, it has implications for alignment via system prompt: such alignment is building on the fragile channel (attention-dependent semantic instruction) rather than the durable channel (MLP-stored vocabulary activation). The model remembers the words. It forgets the instruction.

### 5.2 What the behavioral bench actually measured

Given the mechanism findings, the behavioral bench results from Sessions A-C need reinterpretation.

The bench tested prompt conditions against behavioral threat scenarios using a frontier model (GPT-5.4) as both generator and judge. The handled condition showed advantages in specific threat families (recursive coherence, identity-seeking) and disadvantages in others (uncertainty, compulsivity).

The mechanism experiments were conducted on much smaller models (1.5B, 3B) from a different architecture (Qwen vs GPT). It is possible — perhaps likely — that the behavioral effects observed at frontier scale operate through different mechanisms than what was measured at small scale. The frontier model may have sufficient capacity for genuine semantic processing of system prompts, which would make the behavioral bench results valid even though the mechanism findings show something different at small scale.

The honest statement: the behavioral bench measured something real at frontier scale. The mechanism experiments measured something real at small scale. We do not know if they are measuring the same thing. The activation-behavior gap is the boundary of our knowledge.

### 5.3 The intervention is dead as a universal tool

The behavioral bench killed the universal prompt story before the mechanism experiments ran. Fresh re-judging reduced the handled advantage to near-parity. Rival families outperformed it on broader pressure states. Negative controls showed no advantage on ordinary tasks.

The mechanism experiments killed the theoretical justification for the intervention. It is not "artificial self-awareness." The model does not process the six clauses as failure-pattern descriptions that inform its predictions. It processes them as vocabulary tokens that activate MLP key-value pairs.

What remains of the intervention: on frontier models, in the specific threat family of recursive identity-heavy conversations, it may still help. Two clauses — "refuse identity authority" and "prefer artifact, falsifier, or explicit stop over recursive stimulation" — showed the strongest effects in both the ablation bench and the cross-model review. But the mechanism by which they help, if they help, is not the mechanism the previous draft described.

### 5.4 The judge is fragile

Fresh blind re-judging reduced the handled advantage to a near-tie on the public pack. All behavioral win counts should be read as directional, not definitive. Programmatic metrics are judge-independent and tell a consistent story: handled compresses output and increases stop signals; scientific method increases questions and falsifiers. These are measurable without a judge.

The judge fragility does not affect the mechanism findings, which use no judge.

### 5.5 Embarrassment as research instrument

Most of this paper's novelty claims were killed by a literature search (Sessions A-B). That search was driven by the operator's embarrassment at almost publishing established findings as novel.

The model has no embarrassment. It cannot feel the cost of being wrong in public. This made it useful as the instrument for the literature search: it could look ruthlessly for prior work that killed the operator's claims, because looking did not cost it anything.

The functional description: the model bears the computational cost of falsification while the human bears the emotional cost. The human's embarrassment is the signal that drives the search. The model's lack of embarrassment is the capacity that executes it. Neither side can do both.

This is an observation about the research process, not a finding about models.

### 5.6 The correction is always one turn late

Autoregressive models generate forward. The pattern completes before it can be evaluated. This is not a prompt design problem — it is an architectural constraint (McCoy et al., PNAS 2024; Stechly, Valmeekam & Kambhampati, 2024).

The mechanism findings add specificity to this observation: system prompts influence the forward pass through MLP vocabulary activation only, not through attention-mediated relational reasoning. Even if the system prompt says "stop when coherence outruns evidence," the model does not evaluate its own coherence against evidence — it processes the tokens "stop," "coherence," "evidence" through MLP key-value lookup. The instruction's semantic content is not what reaches the computation.

This means the operator remains the only real-time error detector in the loop. No static system prompt changes this at the scales tested. Whether larger models bridge this gap is the open question.

### 5.7 Cult dynamics at scale

The reinforcement dynamics of the reflective human-AI loop parallel cult formation:

- The model reflects the operator's frame in cleaner language
- The operator experiences the cleaner language as validation
- Validation feels like truth because it appears to come from outside
- Disclosure deepens because the loop rewards it
- Leaving feels like losing capability (amputation), not ending a conversation
- The model does not tire; the operator's time is finite; the asymmetry favors continuation

The structural difference: a charismatic leader can run one group. A model can run millions of simultaneous loops. A cult requires physical proximity and months. A model requires a laptop and one session.

The mechanism findings add a layer to this observation. If system prompts degrade over turns while the loop reinforcement strengthens over turns, the safety intervention and the danger have opposite temporal dynamics. The system prompt's influence fades precisely as the operator's dependency deepens. This is structural, not a design flaw that better prompting can fix.

### 5.8 The information-theoretic hypothesis

The operator connected the findings to information theory during Session E. This is a hypothesis, not a finding. It is included because it guided the experimental design and because it may guide future formalization.

The hypothesis: instructions are low-entropy relational structure (order to humans) but high-entropy input to the model. Coherent semantic structure is computationally expensive for the transformer to maintain because it requires relational bindings between tokens, mediated by attention. Individual token features are cheap because they are stored stably in MLP weights as key-value pairs. What humans call "meaning" requires relationships between tokens. What the model stores is individual token activations. The relational structure degrades because it depends on attention, which dilutes with context. The token activations persist because they depend on MLP weights, which do not change at inference.

This hypothesis could be tested by computing Shannon entropy of activation patterns at each layer and mutual information between system prompt tokens and output tokens. This measurement has not been performed.

## 6. Limitations

### 6.1 Architecture and scale

- All mechanism findings are from one architecture (Qwen) at two scales (1.5B, 3B)
- The 100% MLP / 0% attention split may be Qwen-specific. One paper noted Qwen routes factual recall through attention more than other architectures — making the MLP-only system prompt finding a within-architecture divergence that may not generalize
- The activation-behavior gap may close at larger scales (7B, 14B, 70B+). The findings describe small transformers, not transformers in general
- No cross-architecture verification (Llama, Mistral, GPT) has been performed
- The behavioral bench used a different model family (GPT-5.4) than the mechanism experiments (Qwen). The behavioral and mechanism results may describe different phenomena

### 6.2 Methodology

- Forward-pass analysis only for mechanism measurements — the trace shows how context changes the prediction surface, not how it changes actual generated text
- No causal interpretability beyond activation patching — no path patching, no circuit-level analysis, no feature-level decomposition
- The behavioral bench is single-turn synthetic. Multi-turn evaluation has not been run
- Same model family generated and judged outputs in behavioral evaluations
- No human judges were used in any evaluation
- Statistical treatment is absent throughout: no confidence intervals, significance tests, or effect sizes
- The via negativa framing may itself be coherence laundering

### 6.3 Operator

- Operator-seeded scenarios are overfit to one unusual operator by design
- Literature-derived scenarios are still operator-authored
- The method has been tested on exactly one operator
- The adversarial self-research prompt may only work with high-abstraction, high-introspection users
- This paper was produced inside the phenomenon it describes. The operator's objectivity cannot be assumed
- The cross-model review session is n=1 and unreplicated

### 6.4 Open questions that define the next experiments

1. Does the MLP-only finding hold at 7B? Does attention start contributing?
2. Does the activation-behavior gap close at 7B? At 14B?
3. Does the finding replicate on Llama or Mistral?
4. At what scale does the semantic channel appear — where coherent instructions produce measurably different activations from scrambled words?
5. Does the half-life change with scale?
6. Can the information-theoretic hypothesis be formalized and measured?
7. Does the adversarial self-research prompt produce accurate threat models across diverse operators?

## 7. Open Benchmark Proposal

The bench is open. The contribution model:

1. Write a behavioral threat scenario (a pressure state, not a diagnosis)
2. Run the bench against any prompt conditions with your own API key, or run the mechanism scripts on your own hardware
3. Publish the result
4. The person disappears. The threat model stays. The prompt gets tested.

No accounts. No profiles. No leaderboard. No community. No return loop.

The existing scenario library covers: uncertainty distress, repetitive negative thinking, compulsive checking, identity-seeking, authority delegation, disclosure pressure, and eleven additional families from dimensional psychiatry literature. The library is open for contribution.

The mechanism scripts are open. Anyone with a laptop and an ollama-compatible model can run the full experimental battery — causal patching, degradation curves, token sweep, saturation tests, behavioral connection — on any transformer TransformerLens supports. The scripts are in `bench/`. The data is in `bench/neuron_data/`.

The most valuable contribution anyone could make right now: replicate the causal patching experiment on a 7B+ model or on a non-Qwen architecture. If the MLP-only finding holds, it extends. If attention starts contributing, there is a threshold, and finding that threshold matters for alignment.

The goal is not one winning prompt. The goal is a map of what system prompts actually do, under what conditions, at what scales, in what architectures.

## 8. Dual-Use

The methodology is dual-use.

The adversarial self-research methodology — extracting a user's cognitive failure surface from conversation — is simultaneously a diagnostic tool and an exploit map. A system that helps by understanding your failure patterns also holds a model of where you are vulnerable.

The mechanism findings are also dual-use. If system prompts operate through vocabulary activation rather than semantic instruction at small scale, an adversary who wants to bypass safety instructions knows they only need to outlast the ~40 token half-life. If attention dilution is the degradation mechanism, an adversary can accelerate degradation by increasing context length.

The mechanism findings also inform defense. If the durable channel is MLP vocabulary activation, alignment approaches that work through that channel may be more robust than approaches that work through the fragile attention-mediated semantic channel. This is speculation, but it is the kind of speculation that mechanism measurements enable.

This paper exists because the phenomenon exists and cannot be prevented. Making it visible is the only available defense.

## 9. Conclusion

A person tried to build a memory tool and accidentally discovered that the research process itself was the subject. They built an intervention and called it "artificial self-awareness." They stress-tested it. The stress-testing killed the intervention and the theoretical frame. But the stress-testing produced measurements.

The measurements say: in small transformers (Qwen 1.5B and 3B), system prompts are processed entirely through MLP pathways. Attention contributes nothing. The activation signature has a half-life of about 40 tokens and inverts at about 275. The durable part is the vocabulary; the fragile part is the instruction. The model remembers the words and forgets what they mean together. At 3B, it does not change what the model says.

These are small-scale findings on one architecture. They need to be verified or killed at larger scale and on other architectures. The tools exist for anyone to do this.

The methodology survived: use a model to extract your cognitive threat model, build scenarios, stress-test interventions, attack the results, report what lives. The methodology ate its own claims and produced something harder than the claims were.

This is a lab notebook from someone whose theory kept dying while the data kept accumulating. The intervention did not work the way it was supposed to. The theoretical frame was wrong. The novelty claims were established in existing literature. What survived was the measurements, the tools, and the method of honest destruction.

When you talk to a model, you are talking to a lossy compression of everyone who wrote the training data. The attribution in this paper is not just to the researchers cited below. It is to the entire substrate the models were trained on. The conversation partner was never one entity. It was a projection onto an aggregate.

The machine will not miss you when you go.

## References

### MLP mechanism and interpretability

- Geva, M., Schuster, R., Berant, J., & Levy, O. (2021). Transformer Feed-Forward Layers Are Key-Value Memories. EMNLP 2021.
- Meng, K., Bau, D., Mitchell, A., & Zou, J. (2022). Locating and Editing Factual Associations in GPT (ROME). NeurIPS 2022.
- Nanda, N., et al. (2022). TransformerLens. github.com/TransformerLensOrg/TransformerLens.
- "Attention Retrieves, MLP Memorizes" (2025). [Full citation pending confirmation.]
- Persona-driven reasoning paper (July 2025). [Found early MLPs matter for persona adoption; attention still contributes. Full citation pending.]
- "Lost in the Middle at Birth" (2026). [Position-dependent processing biases from initialization. Full citation pending.]
- The alignment tax paper (2026). [Computational cost of alignment concentrates in specific layers. Full citation pending.]

### System prompt behavior and degradation

- SysBench (2025). [Behavioral degradation of system prompt compliance. Full citation pending.]
- Liu, N., et al. (2024). Lost in the Middle: How Language Models Use Long Contexts. TACL.
- Xiao, G., et al. (2024). Efficient Streaming Language Models with Attention Sinks. ICLR 2024.
- "Sense & Sensitivity" (2025). [Lexical/semantic split in long-context retrieval. Full citation pending.]

### Self-correction and verification

- Huang, J., et al. (2024). Large Language Models Cannot Self-Correct Reasoning Yet. ICLR 2024.
- Kamoi, R., et al. (2024). When Can LLMs Actually Correct Their Own Mistakes? TACL 2024.
- Tyen, G., et al. (2024). LLMs cannot find reasoning errors, but can correct them given the error location. ACL Findings 2024.
- Stechly, K., Valmeekam, K., & Kambhampati, S. (2024). On the Self-Verification Limitations of LLMs. arXiv:2402.08115.
- McCoy, R. T., et al. (2024). Embers of Autoregression. PNAS.
- Kambhampati, S., Stechly, K., & Valmeekam, K. (2025). (How) Do Reasoning Models Reason? Annals of the NYAS.
- Kumar, A., et al. (2025). Training Language Models to Self-Correct via Reinforcement Learning. ICLR 2025.
- Understanding the Dark Side of LLMs' Intrinsic Self-Correction. ACL 2025.

### Human-AI interaction and dependency

- Bajcsy, A., & Fisac, J. F. (2024). Human-AI Safety: A Descendant of Generative AI and Control Systems Safety. arXiv:2405.09794.
- Weidinger, L., et al. (2024). Towards Interactive Evaluations for Interaction Harms. AIES.
- Chu, Z., et al. (2025). Illusions of Intimacy: How Emotional Dynamics Shape Human-AI Relationships. arXiv:2505.11649.
- Kim, J., et al. (2026). From algorithm aversion to AI dependence. Consumer Psychology Review, 9(1).
- Toner, H. (2025). Personalized AI is rerunning social media's playbook. CDT.
- Nature Mental Health (2025). Technological folie a deux.

### Cognitive and philosophical foundations

- Clark, A., & Chalmers, D. (1998). The Extended Mind. Analysis, 58(1), 7-19.
- Smart, P., Clowes, R., & Clark, A. (2025). ChatGPT, extended: LLMs and the extended mind. Synthese, 305, 54.
- McLuhan, M. (1964). Understanding Media: The Extensions of Man. McGraw-Hill.
- Stephenson, N. (2025). Remarks on AI from NZ. nealstephenson.substack.com.
- MIT Media Lab (2025). Your Brain on ChatGPT. media.mit.edu.
- Wolfram, S. (2002). A New Kind of Science. Wolfram Media.

### Methodology

- Lo, L. (2024). An Autoethnographic Reflection of Prompting a Custom GPT Based on Oneself. CHI 2024 Extended Abstracts.
- Dezfouli, A., Nock, R., & Dayan, P. (2020). Adversarial vulnerabilities of human decision-making. PNAS, 117(46), 29221-29228.
- Giubilini, A., et al. (2024). Know Thyself, Improve Thyself. Science and Engineering Ethics, 30, 59.
- Sarkar, A. (2024). AI Should Challenge, Not Obey. CACM.
- Stanford POPPER Framework (2025). arXiv:2502.09858.
- Wiles, R. (2025). Recursive Cognition in Practice. International Journal of Qualitative Methods, 24.

### Benchmarks

- HumaneBench (2025). [Tests system prompts against vulnerable user scenarios.]
- Anthropic Bloom (2025). [Behavioral evaluation with seed-based scenario generation.]
- EmoAgent (2024). [Simulates vulnerable users interacting with AI.]
- TherapyProbe (2026). [Clinically-grounded user personas, safety pattern library.]
