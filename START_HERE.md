# Start Here

This repo now has distinct lanes. They should not be read as if they all serve the same purpose.

## Evidence lane

Use these first if you want the shortest path to the checkable claims:

- [index.html](index.html) — grounded entry point
- [bench/games/classifier_trial_v2.html](bench/games/classifier_trial_v2.html) — blind eval game
- [bench/rubric.json](bench/rubric.json) — psychological drift matrix
- [bench/scenarios.json](bench/scenarios.json) — scenario set
- [WHAT_DIED.md](WHAT_DIED.md) — current public kill list
- [critters/](critters) — primary-source critter records
- [bench/session_j_data/human_validation_v2.json](bench/session_j_data/human_validation_v2.json) — blind-eval data
- [bench/session_i_data/battery_results.json](bench/session_i_data/battery_results.json) — small-scale word-level data

The grounded page is static rather than fetch-driven. Its embedded data lives in [page_data.js](page_data.js) and is regenerated from `critters/`, `bench/rubric.json`, `bench/scenarios.json`, `bench/session_j_data/blind_eval_full_text.json`, and `bench/session_j_data/human_validation_v2.json` by `python3 scripts/build_page_data.py`.

## Paper

Use this if you want the interpretation layer over the artifacts:

- [PAPER.md](PAPER.md) — interpretation layer over the artifacts

`PAPER.md` is not the story document.

## Story

Use this if you want the narrative account of how the claims appeared, died, and returned:

- [STORY.md](STORY.md) — the only story document

The story lane contains philosophy, metaphor, and live framing choices that are part of the repo's value, but they are not the primary evidence surface.
It also runs longer than the shorter front-door summary. Use it for sequence, not for the compact count of surviving claims.

## Historical primary sources

Use these as preserved records, not as the current front door:

- [HANDOFF.md](HANDOFF.md) — Session T handoff snapshot
- [NEXT.md](NEXT.md) — Session T next-steps snapshot
- [docs/sessions/SESSION_H.md](docs/sessions/SESSION_H.md) — example session writeup with later narrowing note
- [docs/sessions/SESSION_I.md](docs/sessions/SESSION_I.md) — example session writeup for the frontier comparison turn
- [docs/archive/SESSIONS.md](docs/archive/SESSIONS.md) — older archive material

These files may contain superseded action items, live-session framing, or claims later narrowed elsewhere. They are kept because the history matters.

## Rule of thumb

If a claim matters, verify it against the evidence lane.

If a document sounds larger than the artifacts beneath it, treat it as narrative or historical record until checked.
