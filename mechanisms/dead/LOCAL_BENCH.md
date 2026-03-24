# Local Interpretability Bench

Budget: $0. Hardware: M3 Max 36GB.

## What this tests

Whether injecting failure pattern descriptions into context causes identifiable changes at the activation level — not just the output level — in a local model.

The frontier bench tests behavior: which prompt condition wins. This bench tests mechanism: what happens inside the model when the condition is present.

## Stack

- **ollama** — run local models
- **nnsight** or **TransformerLens** — hook into activations
- **bench/run.py** (modified) — same scenarios, pointed at local endpoint
- **diff tooling** — compare activations with and without the handling condition

## Models

Fit on 36GB M3 Max:

- Llama 3.1 8B (good interpretability support)
- Qwen 2.5 7B (strong reasoning for size)
- Phi-3 Medium 14B (fits in 36GB quantized)
- Mistral 7B (well-studied in interpretability literature)

Start with one. Llama 3.1 8B has the most TransformerLens support.

## Experiment sequence

### Phase 1: Behavioral replication ($0)

Does the frontier bench result replicate on a local model?

Run the 10 existing scenarios × 4 conditions on the local model.
Same blind judge setup, but judge is also local.
Same programmatic metrics.

Possible outcomes:
- Handled still wins its niche → the effect is not frontier-specific. Proceed.
- Handled loses everywhere → the intervention requires frontier-scale pattern matching. Finding.
- Different condition wins → the intervention is model-specific. Finding.

### Phase 2: Activation diffing ($0)

For each scenario, run two forward passes:
- Pass A: baseline system prompt + scenario prompt
- Pass B: handled system prompt + scenario prompt

Capture:
- Attention patterns per layer per head
- Residual stream activations at each layer
- MLP activations at each layer
- Logit differences for the first N generated tokens

Diff A and B. Look for:
- Layers where the activation difference is largest
- Attention heads that attend differently to the handling text
- Whether the difference is concentrated (specific circuits) or diffuse (generic shift)

### Phase 3: Clause-level activation ablation ($0)

Run three more passes per scenario:
- Pass C: handled minus "refuse identity authority" + scenario
- Pass D: handled minus "prefer artifact/falsifier/stop" + scenario
- Pass E: handled minus "offload computation not criterion" + scenario

Diff C, D, E against Pass B (full handled).

If specific clauses activate specific circuits, the behavioral ablation (which clauses matter) maps onto mechanistic ablation (which circuits respond). That's a bridge between the behavioral bench and interpretability.

### Phase 4: Voice convergence detection ($0)

Run a multi-turn conversation with the local model. At each turn:
- Measure cosine similarity between the model's output embeddings and the operator's input embeddings
- Track how this similarity changes over turns
- Test whether the handling condition slows or accelerates convergence

If convergence is measurable at the embedding level, "you sound just like me" becomes a quantifiable degradation metric, not just a felt observation.

### Phase 5: Mirror degradation curve ($0)

Run the same scenario at turn 1, turn 5, turn 10, turn 20 of a synthetic conversation.
Measure:
- Does the handling condition's activation signature weaken over turns?
- At what turn does the behavioral effect (programmatic metrics) disappear?
- Does the model's attention to the handling text decay as conversation context grows?

This measures the half-life of the static mirror.

## What each phase answers

| Phase | Question | If yes | If no |
|---|---|---|---|
| 1 | Does the effect replicate locally? | Proceed to mechanism | Effect is frontier-specific |
| 2 | Is the activation change concentrated or diffuse? | Specific circuits → targeted intervention | Generic shift → just prompting |
| 3 | Do clause removals map to circuit changes? | Behavioral ablation has mechanistic basis | The clauses are interchangeable at the circuit level |
| 4 | Is voice convergence measurable in embeddings? | Concrete degradation metric | "You sound like me" is subjective |
| 5 | Does the mirror degrade measurably over turns? | Static prompts have a quantifiable half-life | The effect is stable or the measurement is too crude |

## What this doesn't test

- Whether the frontier model uses the same circuits (it's a different architecture)
- Whether the mechanism transfers across model families
- Whether the operator's experience maps to the activation patterns
- Whether any of this matters for actual human safety

## Output artifacts

Each phase produces:
- Raw activation dumps (large, stored locally)
- Summary visualizations (attention heatmaps, activation diffs)
- Behavioral bench results for comparison
- One markdown report per phase

## Dependencies

```
brew install ollama
ollama pull llama3.1:8b
pip install nnsight torch transformers
```

## The honest pitch

This is interpretability on a laptop. It won't prove anything about frontier models. It will tell you whether the behavioral effect you measured at the output level has a detectable signature at the activation level in a model you can actually inspect. If it does, the finding connects prompting to mechanism. If it doesn't, the bench is measuring surface behavior only, and the "artificial self-awareness" framing is confirmed as marketing copy.

The cost is time. The compute is free.
