# TransformerLens MPS Evidence Bundle

This page is the citation-safe entry point for the TransformerLens/Qwen/MPS bug evidence in this repo.

The repo also contains narrative writing, outreach drafts, and preprint drafts. Those are not part of the evidence bundle for this bug report. For upstream discussion, use only the files listed below.

## Current narrow claim

On Apple Silicon MPS with:

- `macOS 26.3`
- `torch==2.8.0`
- `transformer_lens==2.17.0`
- `transformers==4.57.6`

TransformerLens produces incorrect next-token output for `Qwen/Qwen2.5-1.5B-Instruct` on a simple chat-formatted greeting prompt, while:

- Hugging Face on MPS produces `"Hello"` at `0.926177`
- TransformerLens default on MPS produces `"osoph"` at `0.094720`
- TransformerLens with `set_use_attn_result(True)` produces `"Hello"` at `0.926176`

In the same 1.5B run, the first large CPU-vs-MPS cache divergence is `blocks.0.hook_attn_out`.

Separately, a standalone synthetic MPS reproduction with non-contiguous `F.linear` weights also fails, which is consistent with `pytorch/pytorch#161640`.

## Primary evidence

1. Qwen 1.5B reproduction script:
   - `bench/transformerlens_qwen_mps_repro.py`
2. Saved Qwen 1.5B output from:
   - `python3 bench/transformerlens_qwen_mps_repro.py --model Qwen/Qwen2.5-1.5B-Instruct --compare-layer0`
   - artifact: `bench/transformerlens_qwen_mps_repro_1p5b.txt`
3. Synthetic standalone reproduction script:
   - `bench/mps_noncontiguous_linear_repro.py`
4. Saved synthetic output:
   - `bench/mps_noncontiguous_linear_repro.txt`

## Expected output highlights

From `bench/transformerlens_qwen_mps_repro_1p5b.txt`:

- `HF          [('Hello', 0.9261773824691772), ...]`
- `TL default  [('osoph', 0.09471998363733292), ...]`
- `TL explicit [('Hello', 0.9261762499809265), ...]`
- `mps_cache          max=4.9491339 mean=0.30307725`
- `mps_linear_contig  max=0.0004199 mean=0.00000459`

From `bench/mps_noncontiguous_linear_repro.txt`:

- `w.is_contiguous() = False`
- `linear_vs_contiguous max=210.4180603 mean=44.25615692`
- `linear_vs_matmul     max=210.4180603 mean=44.25615692`

## Supporting context

These files explain how the bug was surfaced and why older Qwen/TransformerLens/MPS measurements were invalidated, but they are context, not the primary evidence bundle:

- `docs/sessions/SESSION_H.md`
- `WHAT_DIED.md`
- `mechanisms/session_h.py`

## Posting guidance

If this repo is linked in an upstream thread, link this file or the exact artifacts above rather than the repo root.
