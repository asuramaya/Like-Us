# automatons

What system prompts actually do inside small transformers — and what they don't do.

## The finding

System prompts create a response mode, not token-level control. At production temperatures, word-level divergence equals sampling noise. But the conversation *type* changes — therapeutic vs practical, reflective vs concrete — and persists through sampling and across turns.

Models hear the instruction perfectly. Both attention and MLP pathways carry it redundantly by 7B. They don't follow it. They discuss it. Compliance scoring: 1.0-1.2 out of 3.0 at every scale tested. At 7B, 100% discuss, 0% execute.

The gap is not in the mechanism. The architecture delivers. The training at ≤7B doesn't teach execution.

## How this was found

A person tried to build a handling intervention for human-AI loops.
Eight AI sessions stress-tested the intervention, killed most claims,
found mechanism measurements, killed those, found the measurement tool was broken,
rebuilt everything, killed the new findings, and converged back to where they started:
the model hears you. It doesn't do what you asked.

The theory was wrong at every step. The data killed it at every step.
The methodology — adversarial self-research, each session destroying the previous — is the contribution.

## What's here

```
SESSION_H.md                — The current findings (Session H, correct model)
STORY.md                    — How we got here (8 sessions)
WHAT_DIED.md                — Everything that was killed (updated through Session H)
HANDOFF.md                  — Operational handoff for the next session
NEXT.md                     — What needs to happen next
SESSIONS.md                 — Session log A through H
DIAGNOSTIC_FINDINGS.md      — Session G diagnostic findings (suspect — TL data)
PAPER.md                    — The original paper (OUTDATED — every claim is dead)
index.html                  — The warning page

bench/
  session_h.py              — Current apparatus (HuggingFace + native hooks)
  session_h_data/           — All Session H experimental data (14 files)
  scenarios.json            — 17 behavioral threat scenarios
  conditions.json           — prompt conditions
  rubric.json               — 21-family tiered rubric
  neuron_data/              — Sessions E-G data (SUSPECT — TransformerLens)
  [old scripts]             — Sessions E-G scripts (use TransformerLens — DO NOT TRUST)
```

## Run the experiments

Requirements: Python 3.9+, Apple Silicon (MPS) or CUDA

```bash
pip install torch transformers
```

Session H apparatus (correct model, verified hooks):
```bash
python bench/session_h.py --model Qwen/Qwen2.5-1.5B-Instruct
python bench/session_h.py --model Qwen/Qwen2.5-3B-Instruct
python bench/session_h.py --model Qwen/Qwen2.5-7B-Instruct
python bench/session_h.py --model Qwen/Qwen2.5-1.5B-Instruct --phase verify
```

**Do NOT use the old TransformerLens-based scripts (bench/patch_all_layers.py, bench/diagnose.py, etc.) for Qwen models. TransformerLens corrupts their weights.**

## What survived 8 sessions

1. System prompt creates conversation type, not token control
2. Models discuss instructions, don't execute them (≤7B)
3. DLA ~50/50 is universal (both architectures, all scales)
4. Both pathways independently sufficient by 7B
5. Word interference is destructive (19/20 scenarios)
6. Response-mode diversity gates the effect (not input ambiguity)
7. The adversarial self-research methodology

## What died

- "100% MLP / 0% attention" — bug in hook name (Session G)
- All TransformerLens measurements — TL corrupts Qwen weights (Session H)
- "KL doesn't decay" — corrupt model + truncation artifact (Session H)
- "Attention carries the signal" — cascade artifact (Session H)
- "Amplification during generation" — greedy decoding artifact (Session H)
- "Ambiguity gates the effect" — narrow to reflective scenarios (Session H)
- cos = -0.95 as universal — Qwen-specific, Mistral = -0.06 (Session H)
- The paper's title, abstract, and every mechanism claim

## The open question

At frontier scale, the behavioral chain (Falsifier, Asymmetry, Artifact, Criterion, Question, Proposal, Compression, Stop) works. At ≤7B, models discuss it instead of executing it. The transition from discuss to execute requires training that exists at frontier scale and doesn't exist at the scales we can inspect mechanistically. The 14B model is downloading.

## How this was made

One person. Eight AI sessions. Four local models. Two architectures. Zero API cost for the mechanism experiments. ~200 measurements. ~4 hours compute. Every finding attacked by the session that produced it. The methodology ate everything it produced. What survived is the eating.
