# TransformerLens Publication Gate

This note records the narrower, publication-safe posture for the TransformerLens/MPS evidence in this repo.

## What is safe to say publicly from saved artifacts

- TransformerLens is user-visibly wrong on Apple Silicon MPS for `Qwen/Qwen2.5-1.5B-Instruct` on the saved reproduction in `bench/transformerlens_qwen_mps_repro_1p5b.txt`.
- In that same saved reproduction, the first large CPU-vs-MPS divergence is `blocks.0.hook_attn_out`.
- A standalone synthetic MPS reproduction with non-contiguous `F.linear` weights also fails, as saved in `bench/mps_noncontiguous_linear_repro.txt`.
- Those results are consistent with `pytorch/pytorch#161640`.

## What is not in the current publication-safe bundle

- unsaved or unstable reruns
- 3B follow-up numbers without a linked saved artifact
- arguments for a new TransformerLens PR
- arguments for a new PyTorch issue

## Current public move

- Comment on TransformerLens issue `#1178`.
- Link `docs/issues/transformerlens_mps_evidence_bundle.md`.
- Keep the claim narrow and empirical.

## Retired public moves

- opening a new TransformerLens PR
- opening a new PyTorch issue
- using the repo root as the provenance link
