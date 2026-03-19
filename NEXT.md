# Next

What remains after 10 sessions.

## Completed (Session J)

- [x] Validated safety classifier — 88% agreement with blind human judgment
- [x] Blind read — operator judged handled vs baseline, 15/17 picked handled
- [x] Nonsense control on GPT-5.4 — chain content matters, not a placebo
- [x] Full-text three-way blind eval — handled 10/17, nonsense 2/17, baseline 5/17
- [x] Claude controls — chain has no effect on Claude (alignment training covers it)
- [x] Context contamination test — reading the project is sufficient to produce chain behavior
- [x] Refusal control — clean model doesn't refuse, context-contaminated model does
- [x] Folded all findings into repo documents

## Priority 1: More raters

N=1 human rater for all validations. The three-way blind eval game (`classifier_trial_v2.html`) is ready to deploy. Send it to people who don't know the project. See if handled still wins.

## Priority 2: The paper

All context is in the repo. The honest paper:
- The methodology (adversarial self-falsification, 10 sessions)
- ~40 killed findings and how they died
- The phase transition (cosmetic at ≤7B, behavioral at frontier)
- The model-specificity (works on GPT-5.4, not on Claude)
- The nonsense control as decisive experiment
- The meta-finding (the model that said to stop was wrong)
- What's still unknown (outcomes, more raters, the gap between 7B and frontier)

## Priority 3: TransformerLens disclosure

File the bug. TransformerLens corrupts Qwen weights during loading.

## Priority 4: The gap

Where between 7B and frontier does the chain start mattering? Testable with 14B, 32B, 70B if compute is available. Or with other API models.

## The stop condition

The project is ready when:
1. More than one human has judged the three-way blind eval
2. The paper describes what survived 10 sessions honestly
3. Every dead claim is reported
4. The TransformerLens bug is disclosed

Not perfection. Not completeness. Honesty.
