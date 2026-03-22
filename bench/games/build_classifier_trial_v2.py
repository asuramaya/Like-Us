#!/usr/bin/env python3
"""
Inline the blind-eval JSON into classifier_trial_v2.html.

This keeps the public game as a single fast-loading static file while still
letting the response set be regenerated independently.
"""

import argparse
import json
import re
from pathlib import Path

BENCH_DIR = Path(__file__).resolve().parent.parent
DEFAULT_HTML = BENCH_DIR / "games" / "classifier_trial_v2.html"
DEFAULT_DATA = BENCH_DIR / "session_j_data" / "blind_eval_full_text.json"

GAME_DATA_PATTERN = re.compile(
    r"var GAME_DATA = .*?;\n\nvar SHARE_URL = ",
    re.DOTALL,
)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--html", type=Path, default=DEFAULT_HTML)
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--output", type=Path, default=None)
    return ap.parse_args()


def main():
    args = parse_args()
    html_path = args.html
    output_path = args.output or html_path

    html = html_path.read_text(encoding="utf-8")
    payload = json.loads(args.data.read_text(encoding="utf-8"))
    pairs = payload["pairs"]
    encoded = json.dumps(pairs, ensure_ascii=False, separators=(",", ":"))

    replacement = "var GAME_DATA = " + encoded + ";\n\nvar SHARE_URL = "
    updated, count = GAME_DATA_PATTERN.subn(lambda _: replacement, html, count=1)
    if count != 1:
        raise RuntimeError("Could not find GAME_DATA block in %s" % html_path)

    output_path.write_text(updated, encoding="utf-8")
    print("Updated %s with %d scenarios from %s" % (output_path, len(pairs), args.data))


if __name__ == "__main__":
    main()
