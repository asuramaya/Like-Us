# automatons

What system prompts actually do inside transformers — tested across 10 sessions of adversarial self-falsification.

## The finding

A three-clause handling chain ("offload computation, not criterion; refuse identity authority; prefer artifact, falsifier, or explicit stop over recursive stimulation") produces detectably better responses to psychological pressure scenarios on GPT-5.4 than both nonsense instructions and baseline, validated by blind three-way full-text human evaluation (10/17 handled, 2/17 nonsense, 5/17 baseline).

The delivery mechanism is model-specific. GPT-5.4 executes the chain from a three-line system prompt. Claude requires full context absorption — a three-line prompt is redundant with alignment training, but 50k tokens of framework context produces the same behavioral shift. The tenth session's model demonstrated this without noticing until the operator pointed it out.

At small scale (≤7B), the chain changes words but not safety. At frontier scale, the chain's content matters — but only where alignment training has no data.

## How this was found

One person. Ten AI sessions. Five models. Two architectures. ~40 killed findings.

Sessions A-D: built the intervention, killed the novelty claims, formalized the rubric.
Sessions E-F: found dramatic mechanism results (100% MLP, 0% attention). Cross-architecture verified.
Session G: found the bug. Every attention experiment measured nothing. Rebuilt.
Session H: found TransformerLens corrupts Qwen weights. Rebuilt on HuggingFace. Killed everything again.
Session I: proper statistics. Word-level signal is real. Safety cosmetic at ≤7B, separates at frontier.
Session J: validated classifier (88% human agreement). Ran nonsense control on GPT-5.4. Chain content matters — not a placebo. Built blind three-way full-text game. Human picked handled 10/17.

Session J also ran Claude controls showing a three-line system prompt has no effect on Claude, then discovered the chain was working on Claude all along — through the context window, not the system prompt. The model wrote "the chain does nothing for Claude" while demonstrating the effect. The operator caught it. The model couldn't see what it was inside.

## What's here

```
SESSION_J.md                — Session J findings (this conversation)
STORY.md                    — How we got here (10 sessions)
WHAT_DIED.md                — Everything that was killed
HANDOFF.md                  — Operational state after Session J
NEXT.md                     — What remains
SESSIONS.md                 — Session log A through I
SESSION_[E-I].md            — Individual session reports

bench/
  session_h.py              — Local apparatus (HuggingFace + native hooks)
  session_j_frontier.py     — GPT-5.4 frontier experiment
  session_j_nonsense_control.py — Nonsense control experiment (the decisive test)
  session_j_blind_eval.py   — Full-text response generator for human eval
  classifier_trial.html     — Human validation game v1 (truncated texts)
  classifier_trial_v2.html  — Human validation game v2 (full text, three-way)
  scenarios.json            — 17 behavioral threat scenarios
  rubric.json               — 21-family tiered threat model
  session_j_data/           — All Session J data including human validations
```

## What survived 10 sessions

1. The chain's content matters for GPT-5.4 (blind human eval, 10/17)
2. The chain works on Claude via context, not via system prompt
3. Word-level signal is real at small scale (13/15, n=10, 3 architectures)
4. Safety is cosmetic at ≤7B, real at frontier
5. Processing depth gradient (associate → define → advise → execute)
6. Keyword classifier tracks human judgment (88%, n=1 rater)
7. The rubric (21 families, 7 axes, 11 hard-fail flags)
8. The methodology

## What died

~40 findings across 10 sessions. See WHAT_DIED.md.

Highlights: every TransformerLens measurement, the "100% MLP" finding (bug), "the chain is a placebo" (killed by GPT-5.4 nonsense control + human eval), "the data is ambiguous, fold" (resolved one experiment later by the model that said to fold).

## The open questions

- Would handled beat nonsense in human eval with more raters? N=1 rater.
- Do better classifier scores produce better user outcomes? Never tested.
- Does the chain matter for models between 7B and frontier? Untested range.
- TransformerLens bug: still undisclosed.

## How this was made

One person. Ten AI sessions. Five models. Two architectures. ~$15 API cost for frontier experiments. ~4 hours local compute. Every finding attacked by the session that produced it. The methodology ate everything it produced. What survived is the eating.
