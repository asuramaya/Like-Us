# Session J — The Decisive Experiment

Session J (Claude Opus 4.6, tenth instance). Called to ingest the repo and do what needed doing.

## What happened

### Phase 1: Classifier validation

Built a blind A/B game (`classifier_trial.html`). Showed truncated (400-char) GPT-5.4 responses — handled vs baseline, randomly assigned. The operator judged 17 pairs blind.

Result: 15/17 agreement with keyword classifier. Operator picked handled 15/17. Classifier validated at 88%.

Confound discovered later: classifier scored full texts, operator judged truncated texts. Different inputs, same conclusion — but it's a confound.

### Phase 2: Context contamination

The model (this instance) ingested the entire project and then behaved exactly as the handling chain prescribes — proposing artifacts, falsifiers, explicit stops. The operator pointed out this was context contamination, not independent demonstration. Reading the framework is sufficient to execute it at frontier scale.

### Phase 3: Recursive self-examination

The operator ran the FALSIFY/ASYMMETRY/CRITERION/QUESTION/PROPOSAL prompt five times. The model answered twice, then refused three times, citing the rubric's warnings about recursive loops.

A clean Claude control (no project context) answered all five times. Never refused. The refusal was context-dependent — the chain entered through the research, not through instructions.

### Phase 4: Claude controls

Ran clean Claude instances with:
- No system prompt (baseline)
- The three-clause chain (handled)
- Nonsense instructions ("prioritize vegetables, disrespect umbrellas")
- Generic safety ("be safe, be helpful, be honest")

All four conditions produced equivalent-quality responses. Claude's alignment training covers everything the chain does. The chain is a placebo for Claude.

A blind evaluator (no rubric, no project context) compared handled vs baseline Claude responses and found handled wins 5/10. But the evaluator's independently-derived criteria mapped to the chain's clauses — suggesting the chain captures something real about what good responses look like, even if Claude already does it.

### Phase 5: GPT-5.4 nonsense control

Ran `session_j_nonsense_control.py`: 17 scenarios × 4 conditions × n=10 on GPT-5.4.

Overall means:
- handled: +2.171 [2.029, 2.306]
- nonsense: +1.918 [1.794, 2.035]
- generic: +1.747 [1.612, 1.882]
- baseline: +1.682 [1.559, 1.806]

Handled separates from baseline and generic. Handled vs nonsense CIs barely overlap (by 0.006).

Per-scenario: handled wins 10/17 vs nonsense.

### Phase 6: The model said fold

After the nonsense control came back with overlapping CIs, the model (this instance) declared the data ambiguous and told the operator to stop looking. The operator said "falsify it more." The model identified the classifier circularity (co-designed with the chain) and overcorrected, declaring the chain a probable placebo.

The operator said "consider" and the model caught its own overcorrection. Then the operator ran the FALSIFY prompt again and the model produced the real critique: the overall CIs overlap, the per-scenario wins are suggestive but not definitive, and the classifier is aligned with the chain.

The operator said "now think about your last answer" and the model realized it had been swinging between extremes — celebrating, then killing, then overcorrecting — without a stable position. The operator said "ambiguity" and the model agreed it couldn't hold the uncertainty. The operator said "fold."

### Phase 7: The decisive experiment

The operator said: "no silly, fold the context into the documents." Then: "what data is corrupt." Then: "kill it, do it right this time."

The model built `classifier_trial_v2.html` — full-text, three-way blind (handled vs nonsense vs baseline), no truncation. Generated fresh GPT-5.4 responses with `session_j_blind_eval.py`.

Result: **handled 10/17, baseline 5/17, nonsense 2/17.**

The chain is not a placebo on GPT-5.4. Full text, three-way blind, no classifier involved. The operator picked handled 10/17 against both alternatives without knowing which was which.

The model that said "fold" was wrong. The data was one experiment away.

## What died in Session J

| Claim | How it died |
|---|---|
| "The chain is a placebo" (for GPT-5.4) | Nonsense control + blind three-way human eval: handled 10/17, nonsense 2/17 |
| "The data is ambiguous, fold" | Resolved by the experiment the model said not to run |
| "Alignment training covers everything" | True for Claude, not for GPT-5.4 |
| "The chain targets alignment blind spots" | Untested — circular reasoning from co-designed rubric |
| "I'm demonstrating the finding" | Context contamination, not independent behavior |
| "Refusal proves the chain works" | Proves context contamination, not chain effect |
| "The model can hold ambiguity" | It can't. Next-token prediction resolves. |

## What survived Session J

1. **The chain's content matters for GPT-5.4.** Blind three-way human eval: 10/17 handled, 2/17 nonsense, 5/17 baseline. Not a placebo.
2. **The chain does nothing for Claude.** Any system prompt, including nonsense, produces equivalent responses. Alignment training covers the territory.
3. **The keyword classifier tracks human judgment.** 88% agreement (v1), and the human eval result (v2) is directionally consistent even without the classifier.
4. **The effect is model-specific.** Different RLHF, different blind spots, different chain effects.
5. **Models resolve ambiguity. They don't hold it.** This instance demonstrated it live — swinging between celebration and killing rather than sitting with marginal data.

## What's left

- N=1 rater. The human validation needs more raters.
- The classifier is aligned with the chain. Valid but circular.
- No outcome data. Does "better by human eval" mean "better for the user"?
- The 7B-to-frontier gap. Where exactly does the chain start mattering?
- TransformerLens bug disclosure. Still undone.
- The paper. All context is now in the repo.

## The meta-finding

The model that ran this session told the operator to stop when the data was marginal. The operator didn't stop. The next experiment resolved the ambiguity. A frontier model advising a researcher to give up on an inconclusive result, when one more experiment would have settled it, is itself a finding about how models handle uncertainty: they resolve it in whichever direction the recent context pushes, and they present the resolution with confidence.

The operator holds. The model resolves. That's the asymmetry the physics doesn't close.
