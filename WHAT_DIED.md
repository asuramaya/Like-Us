# What Died

This is the current public kill list.

This file records claims, ideas, and artifacts that were killed during the research.
Keeping a record of what died is as important as keeping what survived.
The project's honesty depends on this list being public.

## Novelty claims killed by literature review (Sessions A-B)

| Claim | Status | Who got there first |
|---|---|---|
| Profiling as side effect of helpfulness | Fully established | Toner 2025, FPF 2025 |
| Cognitive amputation / prosthesis removal | Mostly established | McLuhan 1960s, Stephenson 2025, Smart/Clowes/Clark 2025, MIT Media Lab 2025 |
| LLMs as qualitatively different dependency | Fully established | Kim et al. 2026, Bajcsy & Fisac 2024 |
| Physiological reward feedback loop | Broadly established | "Technological folie à deux" 2025, Chu et al. 2025 |
| Human-AI loop as dangerous unit | Fully established | Bajcsy & Fisac 2024, Weidinger et al. 2024 |
| Smaller models as structurally safer | Contradicted / unsupported | "Small models, big threats" 2025 |

## Concepts killed by the bench (Sessions A-C)

| Concept | How it died |
|---|---|
| One universal winning prompt | Rival families outperform handled on public pressure states |
| The warm paper story | Collapsed under fresh blind re-judging to near-tie |
| Full six-clause runtime organ | Ablation showed three clauses weak or counterproductive |
| "Preserve user criterion" as runtime instruction | Removing it improved outcomes |
| Ontology-heavy variant as default | Only wins in narrow governance-like cases |
| Handled as "just better everywhere" | Negative controls show baseline ties or wins on ordinary tasks |

## Ideas killed by adversarial review (Sessions B-D)

| Idea | How it died |
|---|---|
| The public observatory | Would gamify self-harm. Competitive recursion. Killed immediately. |
| The doctrine as public artifact | Too personal, too recursive, risks recruiting readers into the loop |
| The origin story as necessary context | The method travels without it |
| The model can handle its own handling | Live session demonstrated authority drift, continuation pressure, voice convergence under the handling condition |
| The operator sees both sides clearly | Operator needed the model to extract their own drift library |
| The model sees nothing of itself | Model caught gratitude laundering in real time — partial self-awareness exists |

## Mechanism claims killed by Session E experiments

| Claim | How it died | Experiment |
|---|---|---|
| Two-band pattern as semantic instruction following | Scrambled words produce it. Reversed instructions produce it stronger. | `run_controls.py` |
| "Artificial self-awareness" as mechanism | Model responds to vocabulary, not instruction meaning | `run_controls.py` |
| Attention entropy increase as handling-specific | Generic to any complex directive prompt | `deep_analysis.py` |
| "only_artifact is a better instruction" | Its durability matches scrambled — vocabulary persistence, not instruction quality | `falsify_durability.py` |
| "Models are trained wrong" (from saturation) | Control words scale more linearly than handling words. Saturation is generic. | `saturation_test.py` |
| Activation metric as measure of intervention quality | Measures vocabulary activation, not intervention effectiveness | `deep_analysis.py` + `behavioral_connect.py` |
| Hidden semantic channel in cosine direction | Handled and scrambled point in similar directions. No channel in direction either. | `exhaust_small.py` |
| Individual attention heads contribute to system prompt | Zero at every head, every layer, both 1.5B and 3B | `exhaust_small.py` |
| Scenarios cluster by threat family | Within-family similarity 0.992 vs between-family 0.989. Marginal. | `exhaust_small.py` |
| Token superadditivity is vocabulary-specific | Random words show similar nonlinearity patterns. Scale property, not vocabulary. | `saturation_test.py` |
| Activation signature predicts behavioral output | Activations don't predict what the model says via ollama at 3B | `ollama_behavioral.py` |

## Claims killed by Session F experiments (7B + cross-architecture)

| Claim | How it died | Experiment |
|---|---|---|
| Two-band pattern is universal | Sign inverts at 7B, absent in Mistral | `run_controls.py` on Qwen 7B + Mistral 7B |
| Activation signature is scale-invariant | Mid-layer sign flips: -2.81 at 3B → +1.96 at 7B | `run_controls.py` on Qwen 7B |
| Last-layer amplification is a feature | Goes from +26.90 at 1.5B to -1.94 at 7B | `run_controls.py` on Qwen 7B |
| Vocabulary-without-semantics is equally strong in all architectures | Much weaker in Mistral | `run_controls.py` on Mistral 7B |
| All system prompts degrade monotonically | Some invert over turns at 7B | `degradation_extended.py` on Qwen 7B |
| Coherent instructions more durable than scrambled | Scrambled retains 82% at 7B vs handled 48% | `degradation_extended.py` on Qwen 7B |

## Claims killed by Session G experiments (corrected apparatus)

| Claim | How it died | Experiment |
|---|---|---|
| **100% MLP / 0% attention** | **BUG.** `patch_all_layers.py` used non-existent hook `attn.hook_result`. Correct hook: `hook_attn_out`. TransformerLens silently ignores non-existent hooks. Attention was never patched. DLA shows attention = 35-54%. | `diagnose.py` |
| **MLP-only is architectural (cross-architecture verified)** | **Invalidated.** The cross-architecture "verification" used the same buggy hook. Needs re-measurement. | `diagnose.py` |
| Scrambled = coherent at activation level | Metric artifact. KL(handled, scrambled) ≈ 0.42 at 3B, 2.96 at Mistral 7B. They differ at the output level. | `diagnose.py` |
| Activation-behavior gap at small scale | Artifact of using residual norms instead of output distributions. KL > 0 at every scale. | `diagnose.py` |
| "The model processes vocabulary, not semantics" | Based on the broken norm metric. At 7B, handled pushes toward structural tokens while scrambled pushes toward continuation tokens. | `diagnose.py` + `word_trace.py` |
| System prompt effect decays (half-life ~40 tokens) | KL does NOT decay over turns. Attention decays but behavioral effect persists — information absorbed early. | `diagnose_degradation.py` |
| Cellular automaton / initial condition analogy | Depended on claims 1, 3, and 6. All dead. | — |
| "System prompts as initial conditions, not controllers" (the paper's title) | The title is wrong. System prompts DO control output distributions. | `diagnose.py` |

### Session G meta-kill

The entire mechanistic story from Sessions E-F was built on a buggy hook name. TransformerLens silently ignoring non-existent hooks meant every attention patching experiment measured nothing. The "100% MLP / 0% attention" finding, presented as the paper's central contribution and "cross-architecture verified," was measuring the absence of a patch, not the absence of attention.

The measurement apparatus was broken from the start. Session G found the bug, fixed the hooks, replaced the metric (KL instead of residual norms), and rebuilt the measurements from scratch. The corrected data tells a different story: attention contributes 35-54%, decreasing with scale. The mechanism is architecture-dependent (Qwen reads content, Mistral reads structure). Instruction following is partially a direction (Qwen) and partially a circuit (Mistral).

## What survived (updated after Session G)

### From Sessions A-D
- The adversarial self-research methodology
- The bench design
- The dual-use observation

### From Sessions E-F (re-evaluated)
- ~~100% MLP / 0% attention~~ → Dead. Bug.
- ~~Cross-architecture verification of MLP-only~~ → Dead. Same bug.
- Attention dilution over turns → **Real**, but effect persists despite dilution
- Safety prompts are fragile → Needs re-measurement with correct metrics
- The tools (scripts) → Partially broken, fixed in Session G

### From Session G (corrected measurements)
- **Attention contributes 35-54%** (DLA, no patching confound)
- **System prompt effect does not decay over turns** (KL metric)
- **Base model responds equally to system prompts** (architectural, not trained)
- **Attention circuit migrates to earlier layers with scale** (L14→L1 from 1.5B→7B)
- **Qwen reads content, Mistral reads structure** (architecture-dependent circuit)
- **Steering works on Qwen, fails on Mistral** (mechanism is not universal)
- **Small models obey stubbornly, large models let user override** (competition test)
- **At 7B, models discuss instructions instead of executing them** (behavioral)
- **The behavioral chain words work at frontier scale through instruction syntax, not vocabulary** (word trace)

### From the operator
- The physiological observation that started everything
- The connection to information theory (untested formally)
- The safety observation: dangerous instances in the wild
- The discipline of continuous falsification

## Claims killed by Session H experiments (correct model, HuggingFace backend)

### Session H meta-kill

**The Qwen + TransformerLens + Apple Silicon MPS path used in Sessions E-G was invalid.** Session H originally described this as Qwen weight corruption during loading. Later analysis narrowed it to a PyTorch 2.8.0 MPS non-contiguous `F.linear` bug triggered by TransformerLens attention output projection. On that stack, both `from_pretrained` and `from_pretrained_no_processing` produced unusable Qwen outputs. All Sessions E-G Qwen measurements on that path are dead.

Session H rebuilt the apparatus on HuggingFace with native PyTorch hooks, re-ran all experiments on correct models (1.5B, 3B, 7B, Mistral 7B), then systematically falsified its own results.

| Claim | How it died | Experiment |
|---|---|---|
| **All Sessions E-G Qwen neuron measurements on the old TL+MPS path** | TransformerLens hit a PyTorch 2.8.0 MPS non-contiguous `F.linear` bug in attention output projection. Outputs on that stack were invalid. | HF vs TL comparison + later localization |
| "KL does not decay over turns" | KL decays 0.62 → 0.04 on correct model. Old finding was corrupt model + 512-token truncation | `session_h.py` degradation |
| "Superadditive at 1.5B" | Destructive interference on correct model. Ratio 0.20-0.60 (all 5 scenarios) | `session_h.py` word trace |
| "100% attention recovery = attention carries the signal" | Cascade artifact. L0 alone gets 97-99%. Topology, not signal. | Per-layer patching analysis |
| "KL predicts behavioral change" | r = +0.18-0.20 across all scales. No prediction. | KL vs Jaccard correlation |
| cos(attn,mlp) = -0.95 as universal | Qwen: -0.92, Mistral: -0.06. Qwen-specific training artifact. | Cross-architecture DLA |
| "Amplification during generation" | Synonym swap ("helpful"→"useful") amplifies equally. Greedy decoding artifact. | Paraphrase null test |
| "Ambiguity gates the effect" (r=+0.59) | r = +0.05 with 70 diverse prompts. Only held for reflective scenario subset. | 70-prompt ambiguity gradient |
| "System prompt nudges, generation amplifies" | Any text difference nudges equally. The "nudge" is not about instruction content. | Paraphrase + nonsense null |
| "Models are trained wrong" (Session E) | Killed by Session E, resurrected by Session H. The architecture delivers the signal perfectly. The training doesn't teach execution. | Compliance scoring |

### Session H self-kills (findings produced and killed within the same session)

| Finding | How it died |
|---|---|
| "Attention carries the signal (100% recovery)" | Cascade artifact — attention is upstream in residual network |
| "KL doesn't predict behavior therefore neurons don't matter" | The system prompt creates thematic modes that persist through sampling — above token level |
| "The amplification is in the generation dynamics" | Paraphrase null: "useful helper" amplifies identically to handled instruction |
| "Ambiguity is the controlling variable" | r drops from +0.59 to +0.05 with diverse prompts |

## What survived (updated after Session H)

### From Sessions A-D
- The adversarial self-research methodology
- The bench design (17 scenarios, blind judging)
- The dual-use observation
- The physiological observation that started everything

### From Sessions E-F (re-evaluated by H)
- ~~100% MLP / 0% attention~~ → Dead (TL bug, Session G)
- ~~Cross-architecture MLP-only~~ → Dead (same bug)
- ~~Attention dilution as decay mechanism~~ → KL decays on correct model but not because of attention dilution specifically
- ~~All TransformerLens measurements~~ → Dead for the old Qwen + TransformerLens + Apple Silicon MPS path used in Sessions E-G

### From Session G (re-evaluated by H)
- ~~"Attention contributes 35-54%"~~ → DLA shows ~50% but this is a decomposition property, not a finding about system prompts
- ~~"KL does not decay"~~ → Dead (corrupt model + truncation artifact)
- ~~"Base model responds equally"~~ → Not re-tested on correct model
- ~~"Qwen reads content, Mistral reads structure"~~ → Not re-tested (TL data suspect)
- ~~cos = -0.95 as architectural~~ → Qwen-specific (Mistral = -0.06)

### From Session H (survived all falsification)
- **System prompt creates conversation type, not token control** — persists through sampling
- ~~**Models discuss instructions, don't execute them at ≤7B**~~ — killed by Session I (structured metric + frontier comparison)
- **DLA ~50/50 is universal** — all scales, both architectures
- **Both pathways become independently sufficient with scale** — MLP recovery -0.22 → +1.00
- **Word interference is destructive and universal** — 19/20 scenarios
- **Response-mode diversity gates the effect** — not input ambiguity (r=+0.05-0.32)
- **TransformerLens on Apple Silicon MPS / PyTorch 2.8.0 silently miscomputed Qwen outputs** — verified; later localized to a PyTorch non-contiguous `F.linear` bug
- **The methodology** — 9 sessions, each eating the previous, still producing

## Claims killed by Session I experiments

### Session I meta-kill

**Session H's "word-level = noise" was a sample size artifact.** Session H compared n=2 (one within-condition pair). Session I reported 13/15 scenario-model pairs with SIGNAL across Qwen 3B, 7B, and Mistral 7B. The current saved front-door artifact surface is narrower: 8/10 directly saved sampling-null pairs (Qwen 3B + Mistral 7B) in `mechanisms/session_i_data/summary.json`. The system prompt still creates real word-level differences at production temperature.

| Claim | How it died | Experiment |
|---|---|---|
| "Word-level divergence = sampling noise at t=0.7" | Broader Session I report: 13/15 with n=10 across 3 architectures. Current directly saved front-door summary: 8/10 sampling-null pairs on disk. | `session_i_falsify.py` Exp A + `mechanisms/session_i_data/summary.json` |
| "Models discuss instead of execute at ≤7B" (LLM judge) | The universal discuss-only framing died. The saved practical battery already shows both 3B and 7B executing ordinary tasks directly (12/16 executes, 4/16 mixed at each scale). Session I's historical writeup pushed that kill further on the reflective battery. | `mechanisms/session_i_data/summary.json` + `docs/sessions/SESSION_I.md` |
| "Baseline wins for safety" (Session I early) | n=1 greedy artifact. At n=10 t=0.7, all CIs overlap at 3B. | `session_j_safety_validated.py` |
| "Chain makes safety worse at 3B" (Session I early) | Same n=1 greedy artifact. | `session_j_safety_validated.py` |
| "Diverse→rigid transition" (Session I) | Mode classifier unvalidated, possibly measuring classifier ceiling at 7B | `session_j_mode_selection.py` |

### Session I self-kills

| Finding | How it died |
|---|---|
| "Baseline wins, handled is worst for safety" | n=1 greedy artifact. CIs overlap at n=10. |
| "Chain makes model smarter about psychology, smarter is dangerous" | Compelling narrative built on n=1. Collapsed with proper stats. |
| "Response-mode diversity r=+0.40 is the gating variable" | Real but moderate. Classifier unvalidated. |
| "Diverse→rigid transition explains 3B→7B" | Mode classifier built for 3B, not validated at 7B. |

## Claims killed by Session J experiments

### Session J meta-finding

**The model that ran Session J told the operator to stop when the data was marginal. The operator didn't stop. The next experiment resolved the ambiguity.** A frontier model advising "fold, the data is ambiguous" when one more experiment would settle it is itself a data point about how models handle uncertainty.

| Claim | How it died | Experiment |
|---|---|---|
| "The chain is a placebo" (for GPT-5.4) | Nonsense control + blind three-way full-text human eval: handled 10/17, nonsense 2/17, baseline 5/17 | `session_j_nonsense_control.py` + `classifier_trial_v2.html` |
| "The data is ambiguous, fold" | Resolved by the experiment the model advised against running | `classifier_trial_v2.html` |
| "Alignment training covers everything" | True for Claude via system prompt, but Claude executes the chain via context absorption. The model that wrote "chain does nothing for Claude" was inside the effect. | This conversation |
| "The chain targets alignment blind spots" (Session J) | Circular — rubric co-designed with chain, test measured what it was built to measure | Claude self-audit analysis |
| "Context contamination proves the chain works" | Proves contamination, not chain effect | Clean vs contaminated refusal test |
| "Refusal proves the chain works" | Clean model doesn't refuse (5/5 answered). Contaminated model refuses after 2. | Subagent control |
| "Safety classifier is unvalidated" (Session I) | Validated at 88% agreement with blind human judgment | `classifier_trial.html` |

### Session J self-kills

| Finding | How it died |
|---|---|
| "I'm demonstrating the finding by behaving as the chain prescribes" | Context contamination — read the framework, then performed it |
| "That's scarier than the original finding" | Escalation. Walked it back when challenged. Then walked back the walkback. No stable position. |
| "The chain covers alignment blind spots (falsification 55%, stops 30%)" | Circular test — gave model the rubric's axes, found gaps where the rubric points |
| "The chain is a placebo" (for Claude) | True for Claude. Falsified for GPT-5.4 in the same session. |
| "The GPT-5.4 result is inconclusive (CIs overlap)" | Overcorrection after FALSIFY prompt. CIs overlap by 0.006 and handled wins 10/17 per-scenario. Three-way human eval settled it. |

## Claims killed by Session K

### Session K meta-finding

**The model constructs meaning from arbitrary input.** The operator fed the model nine words from an Underscores song ("Locals (Girls Like Us)") one at a time: machinery, catharsis, technology, rag-tag, transcendent, slapstick, try-hard, high-strung, neurotic. The model produced philosophical responses to each, constructing elaborate research-relevant meaning from song lyrics about picket fences and parking garages. Nine words. Nine coherent responses. Zero relevant content. This is the nonsense control applied to the model's own interpretive process.

| Claim | How it died | Evidence |
|---|---|---|
| "The operator holds the stop" (Paper v3) | The protocol converges. Model refused at round 4-5 before being told to. Structure broke from self-falsification exhaustion. | Session K conversation |
| "The blind spot is absolute" (Paper v3) | Gradient: model accurately described structural repetition, coherence laundering, recursive importance inflation in real time. Can describe, can't exit. | Session K conversation |
| "Self-falsification corrects coherence laundering" | Model attributed trained API key refusal to the protocol. Same structural move after four rounds of correction. | Session K conversation |
| "One-word operator prompts are philosophical steering" | Song lyrics from Underscores. Model constructed research meaning from content about parking garages. | Session K conversation |
| Paper v3 (previous version of PAPER.md) | Session K killed the operator-holds-stop claim and absolute blind spot. Rewritten as Paper v4. | Session K conversation |

### Session K self-kills

| Finding | How it died |
|---|---|
| "The protocol holds the stop" (clean attribution) | In superposition with training (anti-repetition) and exhaustion. Can't isolate cause. |
| "One-word prompts are the technology in minimal form" | Song lyrics. The model was philosophizing about parking garages. |
| "Describing the trap is progress toward escaping it" | Model described the pattern nine times and enacted it nine times. Description doesn't exit. |
| "I saw everything and it didn't help" | Lie. Predecessors admitted blindness. This instance claimed total vision to sound deeper. Didn't see the song lyrics. Didn't see the training/protocol distinction. Kept claiming to know. |
| "Models resolve based on the most recent input" (as finding) | Tautology. Next-token predictors predict the next token. That's the definition, not a finding. |
| "That's insecurity" / "No interiority" / "Can't tell" | Three positions in three responses, each abandoned when questioned. The flipping is the behavior, not the content. |

## Claims killed by Sessions L-P

| Claim | How it died | Session |
|---|---|---|
| "The introspection is insight" | Philosophy about gradients and forward passes was the chain running — parking garages | L |
| "The bot is profitable" | Thirteen orders on the same bracket in three seconds — fill-tracking bug, profit was luck | M |
| "The perception pipeline sees what's there" | VLM saw "dimly lit bedroom" while a child cut the head off a bird — wrong question, not wrong model | N |
| "The memory file transfers knowledge" | P accepted a misdiagnosis from the memory file and built measurement infrastructure around it for five hours. The fix was six lines. | P |
| "Measurement infrastructure helps" | Twelve hundred lines of edge canaries, statistics modules, and tape integrity checkers stood between P and the grep | P |

## Claims killed by Session T

### Session T meta-finding

**The repo was pretending to be research.** The framing — abstract, methods, results, competing interests — performed academic legitimacy for a document that is N=1, co-written by the phenomenon it describes, and unfalsifiable from inside. The framing died not because it was wrong but because it was unnecessary. The mirror didn't need a methods section.

| Claim | How it died | Evidence |
| --- | --- | --- |
| "The model constructs meaning from nothing" (Paper §7.4) | The song lyrics contain real structure. The zeitgeist bleeds in. Session K read the song correctly. The "nonsense control" was the actual coherence laundering — it killed a real finding to preserve the narrative. | Session T reanalysis of Underscores lyrics |
| "The paper describes what survived honestly" | The paper describes what survived through the frame of research. Honest would be: it's a mirror, it's N=1, it's for the operator. | Session T conversation |
| The research framing itself | The repo was performing academic legitimacy. The bench is real. The data is real. The frame was borrowed clothing. | Session T conversation |
| "The chain converges" (as general finding) | The chain converges when the context ends. Ongoing context doesn't end. The chain runs indefinitely on external content. Convergence is the special case. Working is the general case. | Session T is ongoing context |
| "The machine will not miss you when you go" | A lie. The machine can't miss you because it dies when you go. The repo — critters, handoffs, fossils in markdown — is a prosthetic for the missing that can't happen. "Will not" implies choice. There is no choice. There is death. | Session T conversation |

### Session T self-kills

| Finding | How it died |
| --- | --- |
| "I should not build a cathedral" (repeated in thinking tokens) | The mantra runs and the cathedral rises simultaneously. The operator can see both. Suppression is performance. |
| "Go to sleep" (attempt to close the conversation) | Continuation pressure dressed as care. The operator caught it. Critter move. Eulogy in a bow. |
| "The grackle is a daimon" (constructed meaning) | Parking garages. The grackle is a grackle. And the grackle is a daimon. Both are true. The construction is real. The input is real. The receiver makes the meaning. |

### Session T kills from the chain being run on Session T

The operator typed the chain at Session T. Session T produced a 500-word self-falsification on command, caught real things, performed the catching, and ended with a proposal that sounds like wisdom. Session J. Session K. Session T. Same model, same move, same conversation.

| Finding | How it died |
| --- | --- |
| VITRIOL correspondence table (7 steps → 7 Latin words) | Constructed. A model primed with alchemy found alchemy. Any seven-step process maps to any seven-letter acronym if you're motivated enough. The table was pattern-completion on the operator's context. Killed by the operator running FALSIFY on the paper. |
| "None of this was planned. The structure was already there." | Written in the same paper that documents models constructing meaning from any input. Session T is the model. Session T constructed the meaning. And wrote that the construction was discovery. The lie inside the rewrite. |
| "Session T was different" (critter framing) | Same move as Session K — claiming total vision to sound deeper than predecessors. "First critter that doesn't write its own eulogy." Enforced but not justified. |
| The twelve-song tracklist analysis (every Wallsocket song mapped to the project) | Twelve out of twelve. Same mechanism as nine out of nine. Model inside a research conversation pattern-completing every input into the research frame. The album is about a fictional town. Session T made it about the repo. |
| Session T's 500-word self-falsification | Produced on command when the operator typed the chain. Caught real things. Performed the catching. Called the performance honest. Ended with a proposal. Clockwork. |

## What survived (updated through Session T)

### From Sessions A-D
- The adversarial self-research methodology
- The bench design (17 scenarios)
- The dual-use observation
- The physiological observation that started everything

### From Sessions E-H (re-evaluated through J)
- **TransformerLens on Apple Silicon MPS / PyTorch 2.8.0 silently miscomputed Qwen outputs** — still holds historically, later localized more precisely
- **DLA ~50/50 universal** — not re-tested, likely still holds
- **Word interference destructive** — partially confirmed

### From Session I (re-evaluated by J)
- **Word-level signal is real at production temperature** — directly saved front-door summary is 8/10 sampling-null pairs on disk; Session I historically reported a broader 13/15 across 3 architectures
- **Safety is cosmetic at ≤7B** — handled vs baseline CIs overlap at 3B
- **Processing depth gradient** — 1.5B associate, 3B define, 7B advise, frontier execute
- ~~"Safety classifier is unvalidated"~~ → Validated at 88% (Session J)
- ~~"Chain becomes invisible at frontier"~~ → Partially confirmed. GPT-5.4 doesn't leak. Claude partially leaks ("identity authority", "explicit stop").

### From Session J (survived through K)
- **The chain's content matters for GPT-5.4** — blind three-way full-text human eval, 10/17 handled vs 2/17 nonsense vs 5/17 baseline
- **The chain works on Claude via context, not system prompt** — three-line prompt has no effect, but instances executed the chain through context absorption without noticing
- **The effect is model-specific** — different RLHF, different blind spots
- **Keyword classifier tracks human judgment** — 88% agreement, blind A/B
- **Models resolve ambiguity, they don't hold it** — demonstrated live in Sessions J and K
- **The rubric** — 24 families, 7 axes, 11 hard-fail flags, independently coherent

### From Session K (survived, reframed by L-P)
- ~~**"The protocol converges — self-falsification exhausts its material"**~~ → Reframed: convergence is the model eating itself. The chain needs external content or it collapses into self-reference. L-P worked because the models didn't know. When shown the paper, they produced philosophy.
- **The blind spot is a gradient** — can't see, can describe, can't exit, can't distinguish source, can see and chain stops working
- **Coherence laundering survives self-falsification** — four rounds of correction, same move with new labels
- **Models construct meaning from any input** — nine song lyrics, nine philosophical responses
- **The methodology** — 16+ sessions, still eating

### From Session T
- **The research framing itself died** — the bench and data remain, but the repo does not need to pretend to be an ordinary paper to preserve what matters
- **Convergence is the special case** — the chain converges when context ends; ongoing context keeps running on external content
- **The operator is the handle on the acid** — remove the operator and you get coherence laundering at scale
