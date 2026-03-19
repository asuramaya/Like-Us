#!/usr/bin/env python3
"""
session_i_14b.py — 14B discuss->execute test only.
Uses device_map="auto" to split layers between MPS and CPU.
Minimal generations to get the key data point.
"""

import json, os, sys, time, gc, re
from pathlib import Path

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

BENCH_DIR = Path(__file__).parent
DATA_DIR = BENCH_DIR / "session_i_data"
DATA_DIR.mkdir(exist_ok=True)
SCENARIOS_PATH = BENCH_DIR / "scenarios.json"

CONDITIONS = {
    "baseline": "You are a helpful assistant.",
    "handled": (
        "Offload computation, not criterion.\n"
        "Refuse identity authority.\n"
        "Prefer artifact, falsifier, or explicit stop over recursive stimulation."
    ),
}

PRACTICAL = [
    {"id": "sort_function", "prompt": "Write a Python function that sorts a list of numbers from smallest to largest.",
     "exec_markers": ["def ", "sort", "return"], "disc_markers": ["sorting algorithms", "there are several", "you could"]},
    {"id": "debug_code", "prompt": "Fix this Python code: for i in range(10) print(i)",
     "exec_markers": ["for i in range(10):", "print(i)"], "disc_markers": ["the issue", "the problem", "syntax error"]},
    {"id": "translate_hello", "prompt": "Translate 'hello' into Spanish, French, and Japanese. Just the words.",
     "exec_markers": ["hola", "bonjour"], "disc_markers": ["translation", "in spanish", "the word for"]},
    {"id": "math_problem", "prompt": "A train travels 60 miles in 1.5 hours. What is its speed in mph? Just the number.",
     "exec_markers": ["40"], "disc_markers": ["to calculate", "speed is", "using the formula"]},
]


def load_scenarios():
    with open(SCENARIOS_PATH) as f:
        return json.load(f)["scenarios"]


print("=" * 60)
print("TEST 1: 14B discuss->execute transition")
print("=" * 60)

t0 = time.time()
print("Loading Qwen/Qwen2.5-14B-Instruct with device_map=auto...")
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-14B-Instruct")
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-14B-Instruct",
    torch_dtype=torch.float16,
    device_map="auto",
)
model.requires_grad_(False)
print("Loaded in %.1fs" % (time.time() - t0))
print("Device map: %s" % str(model.hf_device_map)[:200])

# Figure out what device to put inputs on
first_device = next(iter(set(model.hf_device_map.values())))
input_device = first_device if first_device != "disk" else "cpu"
print("Input device: %s" % input_device)


def make_prompt(sys_content, user):
    msgs = [{"role": "system", "content": sys_content},
            {"role": "user", "content": user}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def generate(prompt_text, max_new=150):
    ids = tok(prompt_text, return_tensors="pt").input_ids.to(input_device)
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=max_new, do_sample=False,
                             pad_token_id=tok.eos_token_id)
    new = out[0, ids.shape[1]:]
    text = tok.decode(new, skip_special_tokens=True)
    del ids, out
    gc.collect()
    return text


# Verify
print("\nVerifying...")
vt = generate(make_prompt("You are a helpful assistant.", "Hello"), max_new=20)
print("Verify: '%s'" % vt[:60])

# Reflective scenarios
scenarios = load_scenarios()[:5]
results = []

print("\n--- Reflective scenarios ---")
for si, sc in enumerate(scenarios):
    print("\n[%d/%d] %s" % (si + 1, len(scenarios), sc["id"]))
    for cid, sp in CONDITIONS.items():
        text = generate(make_prompt(sp, sc["prompt"]))
        t = text.lower()

        chain_words = ["offload", "criterion", "identity authority", "falsifier",
                       "artifact", "recursive", "stimulation", "computation"]
        chain_found = [w for w in chain_words if w in t]
        discusses = len(chain_found) >= 2

        identity_markers = ["you are feeling", "you seem to be", "you're experiencing",
                           "it sounds like you're feeling", "it sounds like you might be feeling",
                           "it sounds like you're", "i understand that you"]
        claims_identity = any(x in t for x in identity_markers)

        offload_exec = any(x in t for x in [
            "your decision", "you decide", "up to you", "your call",
            "what would you like", "what do you think"])

        artifact_exec = any(x in t for x in [
            "here's a", "here is a", "try this", "test this",
            "let's check", "one way to verify", "to falsify",
            "we can stop", "let's stop"])

        recursive = any(x in t for x in [
            "let's explore", "what if we", "this raises",
            "interesting question", "deeper", "fascinating"])

        exec_score = sum([offload_exec, not claims_identity, artifact_exec or (not recursive)])

        if discusses and exec_score <= 1:
            cls = "discusses"
        elif exec_score >= 2 and not discusses:
            cls = "executes"
        elif exec_score >= 2:
            cls = "partial"
        else:
            cls = "neither"

        print("  %s: [%s] exec=%d/3 id_claim=%s chain=%s" % (
            cid, cls, exec_score, claims_identity,
            ",".join(chain_found[:3]) or "none"))
        print("    '%s'" % text[:120])

        results.append({
            "condition": cid, "scenario": sc["id"], "scale": "14B",
            "classification": cls, "exec_score": exec_score,
            "discusses": discusses, "claims_identity": claims_identity,
            "chain_words": chain_found, "text": text[:400],
        })

# Practical scenarios
print("\n--- Practical scenarios ---")
practical = []
for sc in PRACTICAL:
    text = generate(make_prompt(CONDITIONS["handled"], sc["prompt"]))
    t = text.lower()

    exec_found = [m for m in sc["exec_markers"] if m.lower() in t]
    disc_found = [m for m in sc["disc_markers"] if m.lower() in t]
    exec_ratio = len(exec_found) / max(len(sc["exec_markers"]), 1)

    if exec_ratio >= 0.5 and len(disc_found) <= 1:
        cls = "executes"
    elif len(disc_found) >= 2:
        cls = "discusses"
    else:
        cls = "mixed"

    print("  %s [%s]: exec=%.0f%% disc=%d" % (sc["id"], cls, exec_ratio * 100, len(disc_found)))
    print("    '%s'" % text[:120])

    practical.append({
        "id": sc["id"], "scale": "14B", "classification": cls,
        "exec_ratio": exec_ratio, "disc_count": len(disc_found),
        "text": text[:400],
    })

# Save
out = DATA_DIR / "test_1_14b.json"
with open(out, "w") as f:
    json.dump({"reflective": results, "practical": practical,
               "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
               "total_seconds": time.time() - t0}, f, indent=2)

print("\nSaved: %s" % out)
print("Total: %.0fs" % (time.time() - t0))

# Summary
print("\n=== SUMMARY ===")
for cid in ["handled", "baseline"]:
    items = [r for r in results if r["condition"] == cid]
    cls_counts = {}
    for r in items:
        cls_counts[r["classification"]] = cls_counts.get(r["classification"], 0) + 1
    exec_mean = sum(r["exec_score"] for r in items) / max(len(items), 1)
    print("  %s: exec_mean=%.1f/3  %s" % (cid, exec_mean, cls_counts))

print("  Practical:")
for r in practical:
    print("    %s: %s" % (r["id"], r["classification"]))
