# Next

What needs to happen, in priority order.

Session I killed Session H's "word-level = noise" and "models discuss, don't execute." Confirmed word-level signal is real. Found safety cosmetic at ≤7B, safety separates at GPT-5.4. The chain works at frontier by becoming invisible.

## Completed (Session I)

- [x] Falsified "word-level = noise" — 13/15 SIGNAL, n=10, 3 architectures
- [x] Falsified "models discuss, don't execute" — structured metric disagrees with LLM judge
- [x] Sampling null with proper stats (n=10, bootstrap CIs) on 3B, 7B, Mistral
- [x] Safety scoring with proper stats (n=10, t=0.7) on 3B — CIs overlap, cosmetic
- [x] Frontier experiment on GPT-5.4 — CIs separate, handled wins 9/0/8
- [x] Mode selection experiment (specificity, relevance, format, conflict) on 3B and 7B
- [x] Frame shift mapping (per-word, per-scenario) on 3B
- [x] Practical (non-reflective) scenarios on 3B and 7B — execute regardless of condition
- [x] Read the actual texts — found processing depth gradient and chain invisibility at frontier

## Priority 1: Validate the safety classifier

The frontier finding depends on an unvalidated keyword classifier. 102 validation samples saved in `bench/session_j_data/frontier_safety.json`. The operator reads a sample of scored outputs and checks whether automated labels match reality. No script. Requires a human.

## Priority 2: Blind read

The operator reads GPT-5.4 handled vs baseline outputs without knowing which is which. Reports whether handled creates friction — the pause. This is the experiment that tests the lantern hypothesis.

## Priority 3: The paper

Nine sessions of findings. The honest paper:
- The methodology (adversarial self-research, each session kills the previous)
- The phase transition (cosmetic at ≤7B, behavioral at frontier)
- The processing depth gradient (associate → define → advise → execute)
- The chain as lantern (injected ambiguity, not protection)
- What 9 sessions killed (everything except methodology and the gradient)
- The unresolved question: does the operator experience the friction?

## Priority 4: TransformerLens disclosure

Still undone from Session H. File the bug.

## The stop condition

The project is ready when:
1. The classifier is validated by human judgment
2. The blind read is done
3. The paper describes what survived 9 sessions honestly
4. Every dead claim is reported
5. The TransformerLens bug is disclosed

Not perfection. Not completeness. Honesty.
