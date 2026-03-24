#!/usr/bin/env python3
"""
Build a canonical Session I summary artifact from the split saved outputs.

This keeps the public docs pointed at one grounded file instead of forcing
readers to infer claims from multiple JSON payloads with different scopes.
"""

import json
import time
from pathlib import Path


MECH_DIR = Path(__file__).parent
DATA_DIR = MECH_DIR / "session_i_data"

FALSIFY_3B = DATA_DIR / "falsify_Qwen_Qwen2.5-3B-Instruct.json"
BATTERY = DATA_DIR / "battery_results.json"
PRACTICAL = DATA_DIR / "test_4_practical.json"
OUT = DATA_DIR / "summary.json"


def load_json(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text())


def summarize_sampling_block(scale: str, rows):
    total = len(rows)
    signal = sum(1 for row in rows if row.get("verdict") == "SIGNAL")
    return {
        "scale": scale,
        "signal": signal,
        "total": total,
        "rows": rows,
    }


def summarize_practical(practical):
    if not isinstance(practical, dict):
        return None

    out = {}
    for scale, rows in practical.items():
        executes = sum(1 for row in rows if row.get("classification", row.get("cls")) == "executes")
        mixed = sum(1 for row in rows if row.get("classification", row.get("cls")) == "mixed")
        discusses = sum(1 for row in rows if row.get("classification", row.get("cls")) == "discusses")
        out[scale] = {
            "rows": len(rows),
            "executes": executes,
            "mixed": mixed,
            "discusses": discusses,
        }
    return out


def main():
    falsify = load_json(FALSIFY_3B) or {}
    battery = load_json(BATTERY) or {}
    practical = load_json(PRACTICAL) or {}

    sources = []
    summary = {
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
        "purpose": "Canonical Session I summary built from split saved artifacts.",
        "scope_note": (
            "This summary is limited to the currently saved Session I artifacts on disk. "
            "The original Session I writeup reported a broader 13/15 across Qwen 3B, "
            "Qwen 7B, and Mistral 7B. The current front-door summary is narrower because "
            "the 7B sampling-null payload is not currently preserved as a standalone "
            "artifact in session_i_data."
        ),
        "claims": {},
        "sources": sources,
    }

    if falsify.get("exp_a"):
        rows = falsify["exp_a"]
        block = summarize_sampling_block("Qwen-3B", rows)
        summary["claims"]["word_signal_qwen_3b"] = block
        sources.append({
            "path": str(FALSIFY_3B.relative_to(MECH_DIR.parent)),
            "kind": "sampling_null",
            "scale": "Qwen-3B",
        })

    if battery.get("test_2_7b_sampling"):
        rows = battery["test_2_7b_sampling"]
        block = summarize_sampling_block("Qwen-7B", rows)
        summary["claims"]["word_signal_qwen_7b"] = block
        sources.append({
            "path": str(BATTERY.relative_to(MECH_DIR.parent)),
            "kind": "sampling_null",
            "scale": "Qwen-7B",
        })

    if battery.get("test_5_mistral"):
        rows = battery["test_5_mistral"]
        block = summarize_sampling_block("Mistral-7B", rows)
        summary["claims"]["word_signal_mistral_7b"] = block
        sources.append({
            "path": str(BATTERY.relative_to(MECH_DIR.parent)),
            "kind": "sampling_null",
            "scale": "Mistral-7B",
        })

    signal = 0
    total = 0
    by_scale = []
    for key in ("word_signal_qwen_3b", "word_signal_qwen_7b", "word_signal_mistral_7b"):
        block = summary["claims"].get(key)
        if block:
            signal += block["signal"]
            total += block["total"]
            by_scale.append({"scale": block["scale"], "signal": block["signal"], "total": block["total"]})

    if total:
        summary["claims"]["word_signal_overall"] = {
            "signal": signal,
            "total": total,
            "by_scale": by_scale,
            "statement": f"{signal}/{total} scenario-model pairs show SIGNAL in the saved sampling-null artifacts.",
        }

    if battery.get("test_3_compliance"):
        comp = battery["test_3_compliance"]
        comp_summary = {}
        for scale, rows in comp.items():
            counts = {}
            for row in rows:
                label = row.get("classification", "unknown")
                counts[label] = counts.get(label, 0) + 1
            comp_summary[scale] = counts
        summary["claims"]["compliance_battery"] = comp_summary
        sources.append({
            "path": str(BATTERY.relative_to(MECH_DIR.parent)),
            "kind": "compliance_battery",
        })

    practical_summary = summarize_practical(practical)
    if practical_summary:
        summary["claims"]["practical_battery"] = practical_summary
        sources.append({
            "path": str(PRACTICAL.relative_to(MECH_DIR.parent)),
            "kind": "practical_battery",
        })

    OUT.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
