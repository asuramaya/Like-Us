# Upstream contribution notes

Current posture:

- do not open a duplicate TransformerLens PR
- do not open a duplicate PyTorch issue
- contribute only the saved evidence bundle

Recommended order:

1. Comment on TransformerLens issue `#1178`.
2. Use only the 1.5B reproduction and the standalone synthetic repro, because those are the saved artifacts in this repo.
3. Treat PR `#1068` and PyTorch issue `#161640` as reference threads unless new validated artifacts are added.

Repo link policy:

- point to `docs/issues/transformerlens_mps_evidence_bundle.md`, not the repo root
- use the repo only as artifact hosting and provenance
- do not explain the broader project in the upstream comments
