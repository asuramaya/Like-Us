# Mechanisms

This lane collects the mechanistic work without moving the critters or the session notes.

## Current corrected apparatus

Use these first:

- [EXPERIMENT_MATRIX.md](EXPERIMENT_MATRIX.md) — full mechanistic study matrix and ablation plan
- [session_h.py](session_h.py) — corrected HuggingFace + native PyTorch hook apparatus
- [session_h_data/](session_h_data) — corrected Session H mechanistic data
- [session_i_falsify.py](session_i_falsify.py) — targeted falsification of Session H survivors
- [session_i_battery.py](session_i_battery.py) — hinge-case battery across 7B / practical / Mistral tests
- [session_i_14b.py](session_i_14b.py) — blocked 14B test harness, not current evidence
- [session_i_data/](session_i_data) — corrected Session I behavioral and battery data
- [session_i_data/summary.json](session_i_data/summary.json) — canonical Session I summary built from the currently saved artifacts
- [word_ablation_grid.py](word_ablation_grid.py) — current lexical ablation runner
- [word_ablation_data/](word_ablation_data) — completed 1.5B / 3B / 7B pilot outputs

The live mechanistic question is narrower than the original neuron story. The corrected lane is useful for:

- DLA on the corrected path
- pathway sufficiency by scale
- destructive word interference
- the 7B hinge case between advice, framing, and execution

These scripts still use the shared benchmark scenario file in [../bench/scenarios.json](../bench/scenarios.json).

## Buried archive

- [dead/](dead) — superseded mechanistic archive

This is the fossil record: the old neuron, patching, steering, and reroute lane that produced dramatic claims later narrowed or killed.

The archive carries local copies of [../bench/scenarios.json](../bench/scenarios.json) and [../bench/conditions.json](../bench/conditions.json) so the old scripts remain runnable as a frozen snapshot.

The archive stays here because the failures matter. It is not the current proof surface.
