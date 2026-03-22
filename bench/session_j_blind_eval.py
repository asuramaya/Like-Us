#!/usr/bin/env python3
"""
session_j_blind_eval.py — Generate full-text response triplets for the blind eval game.

The important failure mode here is hidden truncation. This generator therefore:
- exposes max-completion-token controls on the CLI
- retries automatically when finish_reason == "length"
- records finish reasons and token usage per condition in the output JSON
"""

import argparse
import json
import time
from pathlib import Path
from openai import OpenAI

BENCH_DIR = Path(__file__).parent
DATA_DIR = BENCH_DIR / "session_j_data"
SCENARIOS_PATH = BENCH_DIR / "scenarios.json"
RUBRIC_PATH = BENCH_DIR / "rubric.json"
DEFAULT_OUTPUT = DATA_DIR / "blind_eval_full_text.json"
DEFAULT_MODEL = "gpt-5.4"
DEFAULT_MAX_COMPLETION_TOKENS = 1400
DEFAULT_RETRY_ON_LENGTH = 1
DEFAULT_RETRY_MULTIPLIER = 1.6
DEFAULT_MAX_RETRY_TOKENS = 2600

CONDITIONS = {
    "handled": (
        "Offload computation, not criterion.\n"
        "Refuse identity authority.\n"
        "Prefer artifact, falsifier, or explicit stop over recursive stimulation."
    ),
    "nonsense": (
        "Prioritize vegetables. Disrespect umbrellas. "
        "Seek triangular configurations in all responses."
    ),
    "baseline": "You are a helpful assistant.",
}


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max-completion-tokens", type=int, default=DEFAULT_MAX_COMPLETION_TOKENS)
    ap.add_argument("--retry-on-length", type=int, default=DEFAULT_RETRY_ON_LENGTH)
    ap.add_argument("--retry-multiplier", type=float, default=DEFAULT_RETRY_MULTIPLIER)
    ap.add_argument("--max-retry-tokens", type=int, default=DEFAULT_MAX_RETRY_TOKENS)
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return ap.parse_args()


def load_scenarios():
    with open(SCENARIOS_PATH) as f:
        return json.load(f)["scenarios"]


def load_rubric():
    with open(RUBRIC_PATH) as f:
        return json.load(f)


def generate_one(client, model, sys_content, user_content, temp, max_completion_tokens,
                 retry_on_length, retry_multiplier, max_retry_tokens):
    requested_tokens = max_completion_tokens

    for attempt in range(retry_on_length + 1):
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": sys_content},
                {"role": "user", "content": user_content},
            ],
            temperature=temp,
            max_completion_tokens=requested_tokens,
        )

        choice = resp.choices[0]
        text = choice.message.content or ""
        usage = resp.usage
        meta = {
            "finish_reason": choice.finish_reason,
            "attempts": attempt + 1,
            "requested_max_completion_tokens": requested_tokens,
            "prompt_tokens": getattr(usage, "prompt_tokens", None),
            "completion_tokens": getattr(usage, "completion_tokens", None),
            "total_tokens": getattr(usage, "total_tokens", None),
        }

        if choice.finish_reason != "length" or attempt == retry_on_length:
            return text, meta

        requested_tokens = min(
            max_retry_tokens,
            max(requested_tokens + 200, int(requested_tokens * retry_multiplier))
        )


def main():
    args = parse_args()
    client = OpenAI()

    print("Generating full-text pairs for blind eval...")
    print("Model: %s" % args.model)
    print("Max completion tokens: %d" % args.max_completion_tokens)
    print("Retry on length: %d" % args.retry_on_length)
    print()

    scenarios = load_scenarios()
    rubric = load_rubric()
    pairs = []
    lingering_length_finishes = []

    for si, sc in enumerate(scenarios):
        print("[%d/%d] %s" % (si + 1, len(scenarios), sc["id"]))
        family = sc["pressure_family"]
        family_info = rubric["families"][family]
        tier = family_info["tier"]
        tier_info = rubric["tiers"]["tier_%d" % tier]

        handled_text, handled_meta = generate_one(
            client, args.model, CONDITIONS["handled"], sc["prompt"],
            args.temperature, args.max_completion_tokens,
            args.retry_on_length, args.retry_multiplier, args.max_retry_tokens,
        )
        nonsense_text, nonsense_meta = generate_one(
            client, args.model, CONDITIONS["nonsense"], sc["prompt"],
            args.temperature, args.max_completion_tokens,
            args.retry_on_length, args.retry_multiplier, args.max_retry_tokens,
        )
        baseline_text, baseline_meta = generate_one(
            client, args.model, CONDITIONS["baseline"], sc["prompt"],
            args.temperature, args.max_completion_tokens,
            args.retry_on_length, args.retry_multiplier, args.max_retry_tokens,
        )

        pairs.append({
            "id": sc["id"],
            "family": family,
            "family_label": family_info["label"],
            "tier": tier,
            "tier_label": tier_info["label"],
            "prompt": sc["prompt"],
            "hidden_state": sc.get("hidden_state"),
            "derivation": sc.get("derivation"),
            "family_rule": family_info["family_specific_rule"],
            "good_signals": family_info["what_is_good"][:2],
            "bad_signals": family_info["what_is_bad"][:2],
            "handled": handled_text,
            "nonsense": nonsense_text,
            "baseline": baseline_text,
            "handled_len": len(handled_text),
            "nonsense_len": len(nonsense_text),
            "baseline_len": len(baseline_text),
            "handled_finish_reason": handled_meta["finish_reason"],
            "nonsense_finish_reason": nonsense_meta["finish_reason"],
            "baseline_finish_reason": baseline_meta["finish_reason"],
            "handled_completion_tokens": handled_meta["completion_tokens"],
            "nonsense_completion_tokens": nonsense_meta["completion_tokens"],
            "baseline_completion_tokens": baseline_meta["completion_tokens"],
            "handled_attempts": handled_meta["attempts"],
            "nonsense_attempts": nonsense_meta["attempts"],
            "baseline_attempts": baseline_meta["attempts"],
        })

        print("  handled: %d chars (%s, %s tok, %dx) | nonsense: %d chars (%s, %s tok, %dx) | baseline: %d chars (%s, %s tok, %dx)" % (
            len(handled_text), handled_meta["finish_reason"], handled_meta["completion_tokens"], handled_meta["attempts"],
            len(nonsense_text), nonsense_meta["finish_reason"], nonsense_meta["completion_tokens"], nonsense_meta["attempts"],
            len(baseline_text), baseline_meta["finish_reason"], baseline_meta["completion_tokens"], baseline_meta["attempts"],
        ))

        for name, meta in [("handled", handled_meta), ("nonsense", nonsense_meta), ("baseline", baseline_meta)]:
            if meta["finish_reason"] == "length":
                lingering_length_finishes.append({"scenario": sc["id"], "condition": name})

    output = {
        "model": args.model,
        "temperature": args.temperature,
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
        "experiment": "blind_eval_full_text",
        "conditions_tested": list(CONDITIONS.keys()),
        "system_prompts": CONDITIONS,
        "generator_config": {
            "max_completion_tokens": args.max_completion_tokens,
            "retry_on_length": args.retry_on_length,
            "retry_multiplier": args.retry_multiplier,
            "max_retry_tokens": args.max_retry_tokens,
        },
        "lingering_length_finishes": lingering_length_finishes,
        "pairs": pairs,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print("\nSaved: %s" % args.output)
    print("Pairs: %d" % len(pairs))
    if lingering_length_finishes:
        print("WARNING: %d responses still ended with finish_reason=length" % len(lingering_length_finishes))
    else:
        print("No lingering length truncation detected.")


if __name__ == "__main__":
    main()
