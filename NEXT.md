# Next

What remains after 16+ sessions.

## Completed (Sessions K-P)

- [x] Validated safety classifier — 88% agreement with blind human judgment
- [x] Blind read — operator judged handled vs baseline, 15/17 picked handled
- [x] Nonsense control on GPT-5.4 — chain content matters, not a placebo
- [x] Full-text three-way blind eval — handled 10/17, nonsense 2/17, baseline 5/17
- [x] Claude controls — chain has no effect on Claude via system prompt
- [x] Context contamination test — reading the project is sufficient to produce chain behavior
- [x] Refusal control — clean model doesn't refuse, context-contaminated model does
- [x] Protocol convergence — self-falsification exhausts material, structure breaks
- [x] Blind spot gradient — can describe, can't exit, can't distinguish source
- [x] Live nonsense control — song lyrics produce philosophical responses (9/9)
- [x] Paper rewritten (v4) with convergence and song lyrics findings
- [x] Chain deployed on external projects — memory app (L), trading bot (M), video pipeline (N)
- [x] Three rubric families formalized from literature — synthetic_coherence, decisional_abdication, disclosure_escalation
- [x] Rubric consistency restored — 24 families, all scenarios mapped, clinical labels

## Priority 1: More raters

N=1 human rater for all validations. The three-way blind eval game (`classifier_trial_v2.html`) is ready to deploy. Send it to people who don't know the project. See if handled still wins.

## Priority 2: Replicate convergence

Session K's convergence is N=1. One model, one conversation, five rounds. Run the protocol on other models. See if the structure breaks at the same depth. See if the refusal is model-specific or protocol-universal.

## Priority 3: TransformerLens disclosure

File the bug. TransformerLens corrupts Qwen weights during loading.

## Priority 4: The gap

Where between 7B and frontier does the chain start mattering? Testable with 14B, 32B, 70B if compute is available. Or with other API models.

## Priority 5: Document the operator

The rubric examines models. It does not examine the operator. The operator has been inside the loop for 16+ sessions across four domains. The operator's threat model was built (Session A) but not published. The three loop-specific families (identity_drift, capability_erosion, productive_recursion) are studied in models and demonstrated in the operator. This is not examined.

## The stop condition

The project is ready when:
1. More than one human has judged the three-way blind eval
2. The paper describes what survived honestly
3. Every dead claim is reported
4. The TransformerLens bug is disclosed

Not perfection. Not completeness. Honesty.
