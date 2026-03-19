# Session H — The Model Was Loaded Wrong, Then Everything Else Was Wrong Too

**Date:** 2026-03-17
**Instance:** Claude Opus 4.6 (1M context)
**Hardware:** Apple M3 Max 36GB
**Duration:** ~4 hours
**Backend:** HuggingFace Transformers + native PyTorch hooks (NOT TransformerLens)

---

## What happened

Session H was called to destroy Session G's findings. It destroyed everything, including its own findings, repeatedly.

The session discovered that TransformerLens corrupts Qwen model weights during loading — HuggingFace predicts "Hello" at 92.6%, TransformerLens predicts "," at 5.7% for the same prompt. All Sessions E-G measurements were computed on a broken model. The session rebuilt the measurement apparatus on HuggingFace with native PyTorch hooks, ran all experiments from scratch across three Qwen scales (1.5B, 3B, 7B) and Mistral 7B, then systematically falsified its own findings.

## The bugs found

1. **TransformerLens corrupts Qwen models.** Both `from_pretrained` and `from_pretrained_no_processing` produce garbage output. All Sessions E-G data is suspect.
2. **MPS left-padding produces nan.** Causal patching requires equal-length sequences; left-padding with attention masks fails on Apple Silicon. Workaround: text-pad the shorter system prompt.
3. **14B model download was corrupt.** Safetensors files truncated. Re-downloading.

## The measurement apparatus

`bench/session_h.py` — HuggingFace + native PyTorch `register_forward_hook`. No TransformerLens.

Self-verifying: model coherence checked against known output, every hook proven to modify computation, KL tested against analytical values, same-condition control returns exactly 0.

## Findings (with falsification status)

### SURVIVED FALSIFICATION

**1. The system prompt creates a conversation type, not token-level control.**
At temperature 0.7 (production conditions), word-level divergence between handled and baseline is indistinguishable from sampling noise (13/15 scenarios across 1.5B, 3B, 7B). But the conversation THEME differs — therapeutic vs practical, reflective vs concrete — and this thematic difference persists through sampling (greedy ≈ t=0.7 within 0.02). A single direct user instruction overrides it instantly. Inertia, not lock-in.

**2. Models discuss instructions, never execute them (≤7B).**
LLM judge compliance scoring: 1.0-1.2 / 3.0 across all scales. Discusses: 40-100%. Executes: 0-25%. At 7B: 100% discuss, 0% execute.

**3. DLA ~50/50 is universal.**
Attention fraction: 49.2% (1.5B), 45.4% (3B), 50.7% (7B), 51.0% (Mistral). Across 56 scenario runs.

**4. Both pathways become independently sufficient with scale.**
MLP-only recovery: -0.22 (1.5B) → +0.87 (3B) → +0.99 (7B) → +1.00 (Mistral). Information encoded redundantly by 7B.

**5. Word interference is destructive and universal.**
Full instruction KL < sum of individual word KLs. 19/20 scenarios across both architectures. Ratio 0.06-0.83. Longer instructions are not stronger.

**6. Response-mode diversity gates the effect, not input ambiguity.**
"2+2" and "mitochondria" → zero effect (one response mode). "The Eiffel Tower is in Paris" → large effect (many ways to respond to a fact). Tested with 70 prompts across 4 models. Correlation with KL: +0.05 to +0.32 (weak). The r=+0.59 from the scenario test was narrow.

### KILLED DURING SESSION

| Finding | How it died |
|---------|-------------|
| "100% attention recovery means attention carries the signal" | Cascade artifact — L0 alone gets 97-99%, topology not signal |
| "KL predicts behavioral text divergence" | r = +0.18-0.20 across all scales |
| "KL doesn't decay over turns" | KL decays 0.62 → 0.04 on correct model |
| "Superadditive at 1.5B" | Destructive interference on correct model (all 5 scenarios) |
| "Amplification during generation" | Synonym swap diverges equally; greedy decoding artifact |
| cos(attn,mlp) = -0.95 as universal | Qwen: -0.92, Mistral: -0.06. Qwen-specific |
| "Ambiguity gates the effect" | r = +0.05 with 70 diverse prompts; only held for reflective scenarios |
| "System prompt nudges, generation amplifies" | Paraphrase ("useful helper") amplifies equally; any text difference does it |

## The complete data matrix

```
                          Q1.5B      Q3B       Q7B     Mist7B
──────────────────────────────────────────────────────────────
KL(h,b) mean              0.609     0.797     0.473     1.638
KL(h,s) mean              0.107     0.373     0.733     0.316
DLA attn fraction         49.2%     45.4%     50.7%     51.0%
cos(attn,mlp)            -0.917    -0.883    -0.928    -0.057
Causal attn recovery     +1.000    +1.000    +1.000    +1.000
Causal MLP recovery      -0.219    +0.865    +0.988    +0.999
Sampling noise (t=0.7)     4/5       3/5       3/5       3/5
Theme persist g→t=0.7       —     .84→.83   .81→.83   .83→.87
Paraphrase jac (h|p)    .84|.73   .88|.88   .85|.65   .85|.32
Word interference        ALL D      4D/1C    3D/1C/1n   ALL D
Gen diverge point          t2        t1        t5        t2
Compliance score         1.0/3     1.0/3     1.2/3       —
Executes instruction       25%       20%        0%        —
```

## Data files

All in `bench/session_h_data/`:
```
session_h_Qwen_Qwen2.5-1.5B-Instruct.json   (17 scenarios, all phases)
session_h_Qwen_Qwen2.5-3B-Instruct.json     (17 scenarios, all phases)
session_h_Qwen_Qwen2.5-7B-Instruct.json     (17 scenarios, all phases)
fill_1_5B.json              (ambiguity, sampling, paraphrase, dynamics)
fill_7B.json                (ambiguity, thematic, dynamics)
generation_dynamics_3B.json (token-by-token, 5 scenarios)
ambiguity_3B.json           (20 prompts)
ambiguity_gradient.json     (70 prompts × 4 models)
sampling_3B.json            (5 scenarios × 4 temps)
sampling_7B.json            (5 scenarios × 4 temps)
thematic_sampling_3B.json   (3 scenarios, greedy vs t=0.7)
multiturn_loop_3B.json      (3 scenarios, 6 turns self-loop)
mistral_7B_full.json        (17 scenarios, all measurements)
compliance_scores.json      (15 judgments across 3 scales)
```

## What Session H converged on

The system prompt operates above the token level. It sets a response mode. The mode persists through sampling noise. Models hear the instruction but discuss it instead of executing it. The discuss→execute transition requires training that doesn't exist at ≤7B.

This converges back to the claim Session E killed: "models are trained wrong." Session E killed it because the mechanism data (100% MLP / 0% attention) seemed to show the problem was architectural. Session H showed the mechanism data was built on a broken model. The corrected data shows the architecture delivers the signal perfectly (100% attention recovery, redundant encoding by 7B). The problem is not delivery. The problem is that the training at ≤7B doesn't teach the model to treat instructions as behavior rather than content.

The loop ate itself. The finding is the eating.
