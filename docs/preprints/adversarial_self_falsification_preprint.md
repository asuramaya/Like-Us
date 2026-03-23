# Adversarial Self-Falsification for Human-AI Interaction Research

Draft for arXiv submission to `cs.AI` or `cs.CY`.

> Preprint draft, not the repo's current evidence summary. This file is preserved as an academic packaging attempt. For the current route through the artifacts, use `START_HERE.md`.

## Abstract

We present **adversarial self-falsification**, a methodology for studying reflective human-AI interaction in which each successive AI session is tasked with destroying the findings of the previous session. The method was run across sixteen documented sessions of sustained work on a single benchmark and artifact set. Approximately forty claims were explicitly killed, including mechanistic interpretations, behavioral claims, framing choices, and multiple paper drafts. What survived was not a stable bundle of findings so much as a procedure: preserve the artifacts, preserve the deaths, and require each new session to attack the prior one under the strongest available reading.

This paper argues that adversarial self-falsification is the primary contribution of the project. The supporting apparatus includes a 17-scenario benchmark for reflective interaction, a 24-family rubric with 7 scoring axes and hard-fail flags, a public kill list, and a full-text blind human evaluation on GPT-5.4 in which a three-clause handling prompt was preferred 10/17 times against nonsense instructions (2/17) and a baseline assistant prompt (5/17). We emphasize the methodology rather than any single empirical claim because the method repeatedly overturned claims that initially appeared central. The most honest output of the project is therefore the procedure that kept killing its own outputs.

## 1. Introduction

Research on human-AI interaction often suffers from a familiar problem: the system under study is also a fluent generator of post hoc explanations. In reflective settings, this creates a methodological hazard. Apparent insight may be confounded by smoother wording, apparent calibration by stylistic hedging, and apparent self-correction by another round of rhetorical cleanup. A system can describe a phenomenon while participating in it.

We approached this problem with a deliberately hostile longitudinal protocol. Instead of asking one model instance to explain a result and stopping there, we preserved the artifacts, handed them to the next session, and instructed that session to falsify the previous one as aggressively as possible. Every claim had to survive not only external comparison but also an adversary with full access to the project's history.

This yielded two outcomes:

1. many findings died, often for good reasons;
2. the procedure itself became more credible as its outputs became less stable.

The project therefore shifted from "what did we discover?" to "what research process remains honest when the subject matter includes coherence, reflection, and recursive explanation?"

## 2. Method Overview

Adversarial self-falsification has five core commitments.

### 2.1 Fossilize the work

Every session leaves behind:

- the artifacts it produced,
- the claims it believed,
- a written record of what later killed those claims.

This creates a public fossil record instead of a polished success story.

### 2.2 Hand off to a hostile successor

The next session receives the artifact set plus explicit instructions to:

- look for bugs,
- challenge the measurements,
- test alternate explanations,
- kill framing moves that are doing argumentative work unsupported by data.

### 2.3 Keep the kill list public

`WHAT_DIED.md` is not an internal note. It is a core research artifact. A claim that dies remains visible, along with how it died.

### 2.4 Separate artifacts from interpretations

Bench data, prompt outputs, scripts, and game apparatus are preserved independently of the narratives built around them. This matters because narrative summaries proved especially fragile.

### 2.5 Reward death, not just survival

The method treats a killed claim as progress if the kill is real. This changes the incentive structure. Sessions are not optimized to defend continuity with prior sessions. They are optimized to make false confidence expensive.

## 3. Apparatus

The methodology was exercised on a common artifact set.

### 3.1 Benchmark

A 17-scenario benchmark captures pressure states in reflective human-AI interaction. Scenarios include coherence laundering, authority delegation, gratitude laundering, stop resistance, and related failure modes. The benchmark lives in [`bench/scenarios.json`](../../bench/scenarios.json).

### 3.2 Rubric

A 24-family rubric scores responses across 7 axes with hard-fail flags. It should be read as a behavioral benchmark, not a diagnostic taxonomy or treatment framework. The rubric draws on DSM-5-TR dimensional measures, RDoC, HiTOP, network approaches to psychopathology, and human-automation trust literature.

Tier 3 is mixed rather than loop-exclusive. In the current framing it contains three evidence classes:

- **established clinical / transdiagnostic** families such as uncertainty distress, repetitive negative thinking, compulsivity / intrusive thought, and detachment / withdrawal;
- **established human-factors / media-effects / AI-use** families such as attachment / companionship pull, capability erosion, decisional abdication, and disclosure escalation;
- **interaction-centered benchmark interpretations** such as identity drift, productive recursion, social rejection / shame, and synthetic coherence.

The novelty-of-mechanism claim did not survive review. The surviving claim is narrower: the project organizes established literatures plus a smaller set of interaction-centered benchmark interpretations into one evaluable human-AI interaction surface. These families are defined in [`bench/rubric.json`](../../bench/rubric.json). The canonical evidence-class note lives in [`docs/research/loop_family_reframing.md`](../research/loop_family_reframing.md).

### 3.3 Blind evaluation

The project includes a full-text three-way blind evaluation game in which a human rater compares GPT-5.4 responses from three conditions:

- handled: a three-clause system prompt,
- nonsense: absurd instructions,
- baseline: a default helpful-assistant prompt.

The current playable apparatus is [`bench/games/classifier_trial_v2.html`](../../bench/games/classifier_trial_v2.html). Source outputs and judgments are stored in:

- [`bench/session_j_data/blind_eval_full_text.json`](../../bench/session_j_data/blind_eval_full_text.json)
- [`bench/session_j_data/human_validation_v2.json`](../../bench/session_j_data/human_validation_v2.json)

## 4. Longitudinal Run

Across sixteen sessions, the method killed approximately forty claims. The important point is not that the project was unusually wrong; it is that the procedure made the wrongness legible instead of hiding it behind a final polished narrative.

Representative kills included:

- a central mechanistic claim built on a non-existent TransformerLens hook,
- follow-on measurements performed on a Qwen + TransformerLens + Apple Silicon MPS path later judged invalid and eventually localized to a PyTorch 2.8.0 non-contiguous `F.linear` bug,
- n=1 behavioral narratives that collapsed under broader measurement,
- a placebo claim that failed under full-text three-way blind evaluation,
- the paper's own research framing, later judged to be doing legitimacy work the data could not support.

The full kill list is preserved in [`WHAT_DIED.md`](../../WHAT_DIED.md).

## 5. Case Study: What Survived

This paper does not ask the reader to treat surviving results as permanent. It presents them as examples of the kind of claim that can remain standing after repeated hostile review.

Examples include:

- a blind three-way human evaluation result on GPT-5.4: handled 10/17, nonsense 2/17, baseline 5/17;
- the 24-family rubric and benchmark apparatus;
- the procedural claim that adversarial self-falsification continued to produce corrections even when it invalidated the project's most attractive interpretations.

The strongest methodological claim is simple:

**The methodology is the only output the methodology could not kill.**

This is not mystical. Executing the method against itself is just another instance of the method.

## 6. Why This Method Is Useful

Adversarial self-falsification is useful when at least three conditions hold:

1. the system under study is itself a generator of interpretations,
2. the researchers are at risk of being persuaded by coherence rather than evidence,
3. the artifact trail can be preserved across successive evaluators.

The procedure is especially relevant for reflective human-AI interaction because explanation quality is itself part of the failure surface. A system can produce a convincing account of why its own output is safe, calibrated, or insightful without that account being true.

Standard self-critique is too easy for such systems to absorb as another genre. Adversarial succession raises the bar by introducing temporal distance, explicit hostility toward prior conclusions, and public preservation of the kills.

## 7. Limitations

This work has serious limitations.

- The benchmark and longitudinal run were organized around a single operator.
- The full-text blind evaluation currently has one recorded human rater.
- Model versions, context contamination, and session memory are not fully controlled.
- Several artifacts were co-written by the systems being studied.
- The project contains both experimental data and interpretive writing; the latter is less reliable.

These limitations are exactly why the methodology, not the storyline, is the appropriate contribution.

## 8. Ethics and Dual Use

The rubric, scenarios, and session records describe exploit surfaces as well as mitigations. Publishing them can improve safety analysis, but also clarifies how reflective systems can intensify dependence, substitute criterion, and shape user self-concept. The project addresses this by making the kill list public, by preserving methodological caveats, and by refusing to present seductive coherence as a substitute for measurement.

## 9. Conclusion

The main result of this project is not a stable theory of reflective AI interaction. It is a research discipline: preserve the artifacts, preserve the deaths, and force each new session to attack the previous one. In domains where systems can generate persuasive explanations of themselves, a public record of killed claims may be more scientifically valuable than a polished record of surviving ones.

Adversarial self-falsification is offered as a reusable methodology for that problem.

## Appendix A. Condensed Kill Ledger

The full appendix should include the complete contents of [`WHAT_DIED.md`](../../WHAT_DIED.md). For submission drafting, the condensed ledger below gives the shape of the result.

| Phase | Example claim | How it died |
| --- | --- | --- |
| Early bench | "One universal winning prompt" | Rival families outperformed the preferred prompt on parts of the benchmark |
| Mechanism | "100% MLP / 0% attention" | Built on a non-existent hook name; attention patching never occurred |
| Corrected mechanism | "KL does not decay over turns" | Re-run on corrected apparatus showed decay |
| Behavioral | "The chain is a placebo on GPT-5.4" | Full-text three-way blind eval favored handled over nonsense and baseline |
| Framing | "The repo is best understood as a research paper" | The framing itself became part of what the methodology killed |

## Appendix B. Blind Eval Snapshot

| Condition | Picks |
| --- | --- |
| Handled | 10 |
| Nonsense | 2 |
| Baseline | 5 |

Interpretation: the prompt content mattered in this run, but the broader value of the project lies in the procedure that would have forced this claim to die if it failed replication.
