# Mechanisms

This directory is now best read as a **lineage lane**.

It preserves the mechanistic work that grew inside `Like-Us` before parts of that work split into repos that could own them more cleanly.

If you are here because you want the living descendants, start there instead:

- [`decepticons`](https://github.com/asuramaya/decepticons) — shared predictive primitives, reusable O(n) mechanisms, substrates, readouts, and descendant-friendly boundaries
- [`chronohorn`](https://github.com/asuramaya/chronohorn) — experiment runtime, manifests, fleet control, learning curves, and MCP operations
- [`heinrich`](https://github.com/asuramaya/heinrich) — model MRI, geometry, activations, and later safety / jailbreak evidence work

This directory matters because the split has history. It should not be confused with the cleanest present-day home of every mechanistic claim.

## What This Lane Still Holds

Use these first if you want the corrected mechanistic record that still belongs in the ancestor repo:

- [EXPERIMENT_MATRIX.md](EXPERIMENT_MATRIX.md) — full mechanistic study matrix and ablation plan
- [session_h.py](session_h.py) — corrected HuggingFace + native PyTorch hook apparatus
- [session_h_data/](session_h_data/) — corrected Session H mechanistic data
- [session_i_falsify.py](session_i_falsify.py) — targeted falsification of Session H survivors
- [session_i_battery.py](session_i_battery.py) — hinge-case battery across 7B / practical / Mistral tests
- [session_i_14b.py](session_i_14b.py) — blocked 14B test harness, not current evidence
- [session_i_data/](session_i_data/) — corrected Session I behavioral and battery data
- [session_i_data/summary.json](session_i_data/summary.json) — canonical Session I summary built from the currently saved artifacts
- [word_ablation_grid.py](word_ablation_grid.py) — lexical ablation runner
- [word_ablation_data/](word_ablation_data/) — completed 1.5B / 3B / 7B pilot outputs

The live question that remains legible in this ancestor repo is narrower than the old neuron story:

- DLA on the corrected path
- pathway sufficiency by scale
- destructive word interference
- the hinge between advice, framing, and execution at small-to-mid scales

These scripts still use the shared scenario file in [../bench/scenarios.json](../bench/scenarios.json), because the benchmark and the mechanistic lane were born in the same repo even if they no longer end there.

## Where The Claims Moved

The easiest way to misread this directory is to treat it as if every surviving mechanistic thread should still mature here.

What moved out:

- reusable predictive mechanisms moved toward `decepticons`
- experiment runtime and operational surfaces moved toward `chronohorn`
- model-internal forensics and later safety / jailbreak evidence moved toward `heinrich`

What remains here:

- the origin measurements
- the corrected-vs-killed lineage
- the bridge between the original loop work and the later descendant repos

## Buried Archive

- [dead/](dead/) — superseded mechanistic archive

This is the fossil record: the old neuron, patching, steering, reroute, and early dashboard lane that produced dramatic claims later narrowed or died.

The archive keeps local copies of [../bench/scenarios.json](../bench/scenarios.json) and [../bench/conditions.json](../bench/conditions.json) so the older scripts remain runnable as a frozen snapshot.

The archive stays here because the failures matter.

It is not the current proof surface.
