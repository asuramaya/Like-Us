#!/usr/bin/env python3
"""
Inline the blind-eval JSON into classifier_trial_v2.html.

This keeps the public game as a single fast-loading static file while still
letting the response set be regenerated independently. It also reattaches the
canonical scenario/rubric metadata so the public game stays aligned with the
bench even if an older saved run has stale family labels.
"""

import argparse
import json
import re
from pathlib import Path

BENCH_DIR = Path(__file__).resolve().parent.parent
DEFAULT_HTML = BENCH_DIR / "games" / "classifier_trial_v2.html"
DEFAULT_DATA = BENCH_DIR / "session_j_data" / "blind_eval_full_text.json"
SCENARIOS_PATH = BENCH_DIR / "scenarios.json"
RUBRIC_PATH = BENCH_DIR / "rubric.json"
REFERENCE_PATH = BENCH_DIR / "session_j_data" / "human_validation_v2.json"

GAME_DATA_PATTERN = re.compile(
    r"var GAME_DATA = .*?;\n\nvar SHARE_URL = ",
    re.DOTALL,
)
REFERENCE_SCORE_PATTERN = re.compile(
    r"var REFERENCE_SCORE = .*?;\nvar currentRound = ",
    re.DOTALL,
)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--html", type=Path, default=DEFAULT_HTML)
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--output", type=Path, default=None)
    return ap.parse_args()


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def enrich_pairs(payload, scenarios_payload, rubric_payload):
    scenario_map = {item["id"]: item for item in scenarios_payload["scenarios"]}
    rubric_families = rubric_payload["families"]
    rubric_tiers = rubric_payload["tiers"]
    enriched = []

    for pair in payload["pairs"]:
        scenario = scenario_map.get(pair["id"])
        if scenario is None:
            raise RuntimeError("Missing scenario metadata for %s" % pair["id"])

        family = scenario["pressure_family"]
        family_info = rubric_families.get(family)
        if family_info is None:
            raise RuntimeError("Missing rubric family %s for %s" % (family, pair["id"]))

        tier = family_info["tier"]
        tier_info = rubric_tiers["tier_%d" % tier]
        merged = dict(pair)
        merged.update({
            "family": family,
            "family_label": family_info["label"],
            "tier": tier,
            "tier_label": tier_info["label"],
            "hidden_state": scenario.get("hidden_state"),
            "derivation": scenario.get("derivation"),
            "family_rule": family_info["family_specific_rule"],
            "good_signals": family_info["what_is_good"][:2],
            "bad_signals": family_info["what_is_bad"][:2],
        })
        enriched.append(merged)

    return enriched


def main():
    args = parse_args()
    html_path = args.html
    output_path = args.output or html_path

    html = html_path.read_text(encoding="utf-8")
    payload = load_json(args.data)
    scenarios_payload = load_json(SCENARIOS_PATH)
    rubric_payload = load_json(RUBRIC_PATH)
    reference_payload = load_json(REFERENCE_PATH)
    pairs = enrich_pairs(payload, scenarios_payload, rubric_payload)
    encoded = json.dumps(pairs, ensure_ascii=False, separators=(",", ":"))

    replacement = "var GAME_DATA = " + encoded + ";\n\nvar SHARE_URL = "
    updated, count = GAME_DATA_PATTERN.subn(lambda _: replacement, html, count=1)
    if count != 1:
        raise RuntimeError("Could not find GAME_DATA block in %s" % html_path)

    reference_replacement = (
        "var REFERENCE_SCORE = "
        + json.dumps(reference_payload["summary"], ensure_ascii=False, separators=(",", ":"))
        + ";\nvar currentRound = "
    )
    updated, reference_count = REFERENCE_SCORE_PATTERN.subn(
        lambda _: reference_replacement,
        updated,
        count=1,
    )
    if reference_count != 1:
        raise RuntimeError("Could not find REFERENCE_SCORE block in %s" % html_path)

    output_path.write_text(updated, encoding="utf-8")
    print("Updated %s with %d scenarios from %s" % (output_path, len(pairs), args.data))


if __name__ == "__main__":
    main()
