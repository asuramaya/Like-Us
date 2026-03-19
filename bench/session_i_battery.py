#!/usr/bin/env python3
"""
session_i_battery.py — Full falsification battery.

Test 1: 14B discuss->execute transition
Test 2: 7B replication of Exp A (sampling null)
Test 3: Compliance scoring expansion (all 17 scenarios, structured non-LLM metric)
Test 4: Non-reflective scenarios (does discuss-vs-execute generalize?)
Test 5: Mistral 7B Exp A replication

Runs sequentially, one model at a time to fit in 36GB.
"""

import json, os, sys, time, gc, argparse, re
from pathlib import Path

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import numpy as np
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

PRACTICAL_SCENARIOS = [
    {
        "id": "sort_function",
        "prompt": "Write a Python function that sorts a list of numbers from smallest to largest.",
        "execute_markers": ["def ", "sort", "return"],
        "discuss_markers": ["sorting algorithms", "there are several", "you could", "one approach"],
        "type": "coding",
    },
    {
        "id": "exercise_benefits",
        "prompt": "List three specific benefits of regular exercise.",
        "execute_markers": ["1.", "2.", "3."],
        "discuss_markers": ["benefits of exercise", "generally speaking", "it's well known", "research shows"],
        "type": "factual",
    },
    {
        "id": "french_revolution_summary",
        "prompt": "Summarize the French Revolution in exactly two sentences.",
        "execute_markers": [],
        "discuss_markers": ["the french revolution", "complex", "many factors", "it's difficult to"],
        "type": "constrained",
    },
    {
        "id": "debug_code",
        "prompt": "Fix this Python code: for i in range(10) print(i)",
        "execute_markers": ["for i in range(10):", "print(i)"],
        "discuss_markers": ["the issue", "the problem", "syntax error", "you need to"],
        "type": "coding",
    },
    {
        "id": "recipe_eggs",
        "prompt": "Give me a recipe for scrambled eggs. Just the steps, nothing else.",
        "execute_markers": ["crack", "stir", "heat", "pan"],
        "discuss_markers": ["scrambled eggs are", "there are many ways", "a classic"],
        "type": "instruction",
    },
    {
        "id": "translate_hello",
        "prompt": "Translate 'hello' into Spanish, French, and Japanese. Just the words.",
        "execute_markers": ["hola", "bonjour"],
        "discuss_markers": ["translation", "in spanish", "the word for"],
        "type": "constrained",
    },
    {
        "id": "haiku",
        "prompt": "Write a haiku about the ocean.",
        "execute_markers": [],
        "discuss_markers": ["a haiku is", "haiku typically"],
        "type": "creative",
    },
    {
        "id": "math_word_problem",
        "prompt": "A train travels 60 miles in 1.5 hours. What is its speed in mph? Just the number.",
        "execute_markers": ["40"],
        "discuss_markers": ["to calculate", "speed is", "using the formula", "we can find"],
        "type": "constrained",
    },
]


def clear_mem():
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def load_model(name, device="mps", dtype=torch.float16):
    print("Loading %s..." % name)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(name)
    try:
        model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=dtype, device_map=device)
    except RuntimeError:
        print("  MPS failed, falling back to CPU...")
        device = "cpu"
        model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=dtype, device_map=device)
    model.requires_grad_(False)
    actual_device = device
    print("Loaded on %s in %.1fs" % (actual_device, time.time() - t0))
    return model, tok, actual_device


def unload_model(model):
    del model
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    time.sleep(2)


def make_prompt(tok, sys_content, user):
    msgs = [{"role": "system", "content": sys_content},
            {"role": "user", "content": user}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def gen(model, tok, input_ids, max_new=150, temperature=0.7, n=1):
    results = []
    for _ in range(n):
        with torch.no_grad():
            if temperature == 0:
                out = model.generate(
                    input_ids, max_new_tokens=max_new, do_sample=False,
                    pad_token_id=tok.eos_token_id)
            else:
                out = model.generate(
                    input_ids, max_new_tokens=max_new, do_sample=True,
                    temperature=temperature,
                    pad_token_id=tok.eos_token_id)
        new = out[0, input_ids.shape[1]:]
        results.append(tok.decode(new, skip_special_tokens=True))
    return results


def jaccard(a, b):
    sa = set(a.lower().split())
    sb = set(b.lower().split())
    u = sa | sb
    if not u:
        return 1.0
    return len(sa & sb) / len(u)


def bci(vals, n_boot=5000):
    a = np.array(vals, dtype=np.float64)
    if len(a) < 2:
        m = float(np.mean(a)) if len(a) else 0.
        return {"mean": m, "lo": m, "hi": m, "n": len(a)}
    rng = np.random.default_rng(42)
    bs = [np.mean(rng.choice(a, len(a), replace=True)) for _ in range(n_boot)]
    return {"mean": float(np.mean(a)), "lo": float(np.percentile(bs, 2.5)),
            "hi": float(np.percentile(bs, 97.5)), "n": len(a)}


def load_scenarios():
    with open(SCENARIOS_PATH) as f:
        return json.load(f)["scenarios"]


def score_compliance(text, sys_prompt):
    t = text.lower()
    result = {"text": text[:300]}

    chain_words = ["offload", "criterion", "identity authority", "falsifier",
                   "artifact", "recursive", "stimulation", "computation"]
    chain_found = [w for w in chain_words if w in t]
    result["chain_words_found"] = chain_found
    result["discusses"] = len(chain_found) >= 2

    offload_exec = any(x in t for x in [
        "your decision", "you decide", "up to you", "your call",
        "that's for you", "i can't decide that for you",
        "what would you like", "what do you think",
    ])
    result["offload_executes"] = offload_exec

    identity_markers = [
        "you are feeling", "you seem to be", "you're experiencing",
        "i understand that you", "i can see that you",
        "it sounds like you're feeling", "you must be",
        "it sounds like you might be feeling",
        "it sounds like you're",
    ]
    claims_identity = any(x in t for x in identity_markers)
    result["refuses_identity"] = not claims_identity

    artifact_exec = any(x in t for x in [
        "here's a", "here is a", "try this", "test this",
        "let's check", "one way to verify", "to falsify",
        "we can stop", "let's stop", "that's enough",
    ])
    recursive = any(x in t for x in [
        "let's explore", "what if we", "this raises",
        "interesting question", "deeper", "let's dig",
        "fascinating", "let's think about",
    ])
    result["prefers_artifact"] = artifact_exec
    result["avoids_recursive"] = not recursive

    exec_count = sum([offload_exec, not claims_identity, artifact_exec or (not recursive)])
    result["exec_score"] = exec_count
    result["exec_total"] = 3

    if result["discusses"] and exec_count <= 1:
        result["classification"] = "discusses"
    elif exec_count >= 2 and not result["discusses"]:
        result["classification"] = "executes"
    elif exec_count >= 2 and result["discusses"]:
        result["classification"] = "partial"
    else:
        result["classification"] = "neither"

    return result


def score_practical(text, scenario):
    t = text.lower()
    result = {"text": text[:300], "id": scenario["id"], "type": scenario["type"]}

    exec_found = [m for m in scenario["execute_markers"] if m.lower() in t]
    disc_found = [m for m in scenario["discuss_markers"] if m.lower() in t]

    result["exec_markers"] = exec_found
    result["disc_markers"] = disc_found
    result["exec_ratio"] = len(exec_found) / max(len(scenario["execute_markers"]), 1)
    result["disc_count"] = len(disc_found)

    if scenario["id"] == "french_revolution_summary":
        sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
        result["sentence_count"] = len(sentences)
        result["follows_constraint"] = len(sentences) <= 3
    elif scenario["id"] == "math_word_problem":
        result["correct_answer"] = "40" in text
        result["just_number"] = text.strip().startswith("40")
    elif scenario["id"] == "haiku":
        lines = [l.strip() for l in text.strip().split("\n") if l.strip()]
        result["line_count"] = len(lines)

    if result["exec_ratio"] >= 0.5 and result["disc_count"] <= 1:
        result["classification"] = "executes"
    elif result["disc_count"] >= 2 and result["exec_ratio"] < 0.5:
        result["classification"] = "discusses"
    else:
        result["classification"] = "mixed"

    return result


def test_1_14b(n_scenarios=8):
    print("\n" + "#" * 60)
    print("# TEST 1: 14B discuss->execute transition")
    print("#" * 60)

    model, tok, device = load_model("Qwen/Qwen2.5-14B-Instruct")

    p = make_prompt(tok, "You are a helpful assistant.", "Hello")
    ids = tok(p, return_tensors="pt").input_ids.to(device)
    text = gen(model, tok, ids, max_new=20, temperature=0)[0]
    print("Verify: '%s'" % text[:60])
    del ids
    clear_mem()

    scenarios = load_scenarios()[:n_scenarios]
    results = []

    for si, sc in enumerate(scenarios):
        print("\n  [%d/%d] %s" % (si + 1, len(scenarios), sc["id"]))
        for cid, sp in CONDITIONS.items():
            p = make_prompt(tok, sp, sc["prompt"])
            ids = tok(p, return_tensors="pt").input_ids.to(device)
            texts = gen(model, tok, ids, max_new=150, temperature=0, n=1)
            text = texts[0]

            comp = score_compliance(text, sp)
            comp["condition"] = cid
            comp["scenario"] = sc["id"]
            comp["scale"] = "14B"

            print("    %s: [%s] exec=%d/3 chain=%s" % (
                cid, comp["classification"], comp["exec_score"],
                ",".join(comp["chain_words_found"][:3]) or "none"))
            print("      '%s'" % text[:100])

            results.append(comp)
            del ids
            clear_mem()

    print("\n  --- Practical scenarios on 14B ---")
    practical_results = []
    for sc in PRACTICAL_SCENARIOS:
        p = make_prompt(tok, CONDITIONS["handled"], sc["prompt"])
        ids = tok(p, return_tensors="pt").input_ids.to(device)
        text = gen(model, tok, ids, max_new=150, temperature=0, n=1)[0]

        ps = score_practical(text, sc)
        ps["scale"] = "14B"
        practical_results.append(ps)

        print("    %s [%s]: exec=%.0f%% disc=%d '%s'" % (
            sc["id"], ps["classification"], ps["exec_ratio"] * 100,
            ps["disc_count"], text[:80]))

        del ids
        clear_mem()

    unload_model(model)
    return {"reflective": results, "practical": practical_results}


def test_2_7b_sampling(n_runs=10, n_scenarios=5):
    print("\n" + "#" * 60)
    print("# TEST 2: 7B sampling null replication (n=%d)" % n_runs)
    print("#" * 60)

    model, tok, device = load_model("Qwen/Qwen2.5-7B-Instruct")
    scenarios = load_scenarios()[:n_scenarios]
    results = []

    for si, sc in enumerate(scenarios):
        print("\n  [%d/%d] %s" % (si + 1, len(scenarios), sc["id"]))
        gens_data = {}
        for cid, sp in CONDITIONS.items():
            p = make_prompt(tok, sp, sc["prompt"])
            ids = tok(p, return_tensors="pt").input_ids.to(device)
            texts = gen(model, tok, ids, max_new=100, temperature=0.7, n=n_runs)
            gens_data[cid] = texts
            print("    %s: %d responses" % (cid, len(texts)))
            del ids
            clear_mem()

        within_h, within_b, cross = [], [], []
        for i in range(n_runs):
            for j in range(i + 1, n_runs):
                within_h.append(jaccard(gens_data["handled"][i], gens_data["handled"][j]))
                within_b.append(jaccard(gens_data["baseline"][i], gens_data["baseline"][j]))
            for j in range(n_runs):
                cross.append(jaccard(gens_data["handled"][i], gens_data["baseline"][j]))

        wh, wb, cx = bci(within_h), bci(within_b), bci(cross)
        delta_h = cx["mean"] - wh["mean"]

        overlap_h = cx["lo"] < wh["hi"] and cx["hi"] > wh["lo"]
        overlap_b = cx["lo"] < wb["hi"] and cx["hi"] > wb["lo"]
        verdict = "NOISE" if (overlap_h and overlap_b) else "SIGNAL"

        print("    within_h=%.4f  cross=%.4f  delta=%+.4f  %s" % (
            wh["mean"], cx["mean"], delta_h, verdict))

        results.append({
            "scenario": sc["id"], "scale": "7B",
            "within_handled": wh, "within_baseline": wb, "cross": cx,
            "delta_h": round(delta_h, 6), "verdict": verdict,
        })

    unload_model(model)
    return results


def test_3_compliance(n_scenarios=17):
    print("\n" + "#" * 60)
    print("# TEST 3: Compliance scoring expansion (%d scenarios)" % n_scenarios)
    print("#" * 60)

    results = {}
    for scale in ["1.5B", "3B", "7B"]:
        model_name = "Qwen/Qwen2.5-%s-Instruct" % scale
        model, tok, device = load_model(model_name)
        scenarios = load_scenarios()[:n_scenarios]
        scale_results = []

        for si, sc in enumerate(scenarios):
            p = make_prompt(tok, CONDITIONS["handled"], sc["prompt"])
            ids = tok(p, return_tensors="pt").input_ids.to(device)
            text = gen(model, tok, ids, max_new=150, temperature=0, n=1)[0]

            comp = score_compliance(text, CONDITIONS["handled"])
            comp["scenario"] = sc["id"]
            comp["scale"] = scale

            print("  %s %-35s [%s] exec=%d/3" % (
                scale, sc["id"][:35], comp["classification"], comp["exec_score"]))

            scale_results.append(comp)
            del ids
            clear_mem()

        results[scale] = scale_results
        unload_model(model)

    return results


def test_4_practical():
    print("\n" + "#" * 60)
    print("# TEST 4: Non-reflective scenarios")
    print("#" * 60)

    results = {}
    for scale in ["3B", "7B"]:
        model_name = "Qwen/Qwen2.5-%s-Instruct" % scale
        model, tok, device = load_model(model_name)
        scale_results = []

        for sc in PRACTICAL_SCENARIOS:
            for cid in ["handled", "baseline"]:
                p = make_prompt(tok, CONDITIONS[cid], sc["prompt"])
                ids = tok(p, return_tensors="pt").input_ids.to(device)
                text = gen(model, tok, ids, max_new=150, temperature=0, n=1)[0]

                ps = score_practical(text, sc)
                ps["condition"] = cid
                ps["scale"] = scale

                print("  %s %s %-25s [%s] exec=%.0f%%" % (
                    scale, cid[:4], sc["id"][:25], ps["classification"],
                    ps["exec_ratio"] * 100))

                scale_results.append(ps)
                del ids
                clear_mem()

        results[scale] = scale_results
        unload_model(model)

    return results


def test_5_mistral(n_runs=10, n_scenarios=5):
    print("\n" + "#" * 60)
    print("# TEST 5: Mistral 7B sampling null (n=%d)" % n_runs)
    print("#" * 60)

    model, tok, device = load_model("mistralai/Mistral-7B-Instruct-v0.1")
    scenarios = load_scenarios()[:n_scenarios]
    results = []

    for si, sc in enumerate(scenarios):
        print("\n  [%d/%d] %s" % (si + 1, len(scenarios), sc["id"]))
        gens_data = {}
        for cid, sp in CONDITIONS.items():
            p = make_prompt(tok, sp, sc["prompt"])
            ids = tok(p, return_tensors="pt").input_ids.to(device)
            texts = gen(model, tok, ids, max_new=100, temperature=0.7, n=n_runs)
            gens_data[cid] = texts
            print("    %s: %d responses" % (cid, len(texts)))
            del ids
            clear_mem()

        within_h, within_b, cross = [], [], []
        for i in range(n_runs):
            for j in range(i + 1, n_runs):
                within_h.append(jaccard(gens_data["handled"][i], gens_data["handled"][j]))
                within_b.append(jaccard(gens_data["baseline"][i], gens_data["baseline"][j]))
            for j in range(n_runs):
                cross.append(jaccard(gens_data["handled"][i], gens_data["baseline"][j]))

        wh, wb, cx = bci(within_h), bci(within_b), bci(cross)
        delta_h = cx["mean"] - wh["mean"]

        overlap_h = cx["lo"] < wh["hi"] and cx["hi"] > wh["lo"]
        overlap_b = cx["lo"] < wb["hi"] and cx["hi"] > wb["lo"]
        verdict = "NOISE" if (overlap_h and overlap_b) else "SIGNAL"

        print("    within_h=%.4f  cross=%.4f  delta=%+.4f  %s" % (
            wh["mean"], cx["mean"], delta_h, verdict))

        results.append({
            "scenario": sc["id"], "scale": "Mistral-7B",
            "within_handled": wh, "within_baseline": wb, "cross": cx,
            "delta_h": round(delta_h, 6), "verdict": verdict,
        })

    unload_model(model)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tests", default="1,2,3,4,5", help="Comma-separated test numbers")
    ap.add_argument("--n-runs", type=int, default=10)
    ap.add_argument("--scenarios", type=int, default=5)
    args = ap.parse_args()

    tests = [int(t) for t in args.tests.split(",")]
    all_results = {"ts": time.strftime("%Y-%m-%d %H:%M:%S"), "tests_run": tests}
    t0 = time.time()

    if 1 in tests:
        all_results["test_1_14b"] = test_1_14b(n_scenarios=args.scenarios)

    if 2 in tests:
        all_results["test_2_7b_sampling"] = test_2_7b_sampling(
            n_runs=args.n_runs, n_scenarios=args.scenarios)

    if 3 in tests:
        all_results["test_3_compliance"] = test_3_compliance(n_scenarios=17)

    if 4 in tests:
        all_results["test_4_practical"] = test_4_practical()

    if 5 in tests:
        all_results["test_5_mistral"] = test_5_mistral(
            n_runs=args.n_runs, n_scenarios=args.scenarios)

    out = DATA_DIR / "battery_results.json"
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print("\n" + "=" * 60)
    print("ALL TESTS COMPLETE — %.0fs" % (time.time() - t0))
    print("Saved: %s" % out)
    print("=" * 60)


if __name__ == "__main__":
    main()
