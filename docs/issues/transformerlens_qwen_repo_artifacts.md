# TransformerLens MPS repo artifacts

For upstream bug discussion, start with:

- `docs/issues/transformerlens_mps_evidence_bundle.md`

That file is the citation-safe entry point for the bug evidence in this repo.

## Primary evidence

- `bench/transformerlens_qwen_mps_repro.py`
- `bench/transformerlens_qwen_mps_repro_1p5b.txt`
- `bench/mps_noncontiguous_linear_repro.py`
- `bench/mps_noncontiguous_linear_repro.txt`

## Supporting context

These files explain how the invalid Qwen/TransformerLens/MPS path was discovered and replaced, but they are context rather than the primary evidence bundle:

- `docs/sessions/SESSION_H.md`
- `WHAT_DIED.md`
- `bench/session_h.py`
- `bench/session_h_data/session_h_Qwen_Qwen2.5-1.5B-Instruct.json`
- `bench/session_h_data/session_h_Qwen_Qwen2.5-3B-Instruct.json`
- `bench/session_h_data/session_h_Qwen_Qwen2.5-7B-Instruct.json`
