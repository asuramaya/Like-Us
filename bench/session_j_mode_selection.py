#!/usr/bin/env python3
"""
session_j_mode_selection.py — What selects the response mode?

Session I established:
  - System prompt creates real word-level differences (not noise)
  - Response-mode diversity gates the effect (r=+0.40)
  - Diversity is necessary but not sufficient

This experiment measures WHAT determines whether the system prompt
successfully steers the model when multiple modes are available.

The metric is MODE ENTROPY REDUCTION: does the system prompt concentrate
responses into fewer modes compared to baseline?

Controlled variables:
  A. Specificity — abstract principle vs concrete role vs explicit behavior
  B. Relevance — system prompt topic matches vs mismatches user prompt
  C. Format — imperative rules vs role description vs worked example
  D. Conflict — system prompt and user prompt imply different modes

Not KL. Not DLA. Not neurons. Behavioral mode classification only.
"""

import json, os, sys, time, gc, re, argparse
from pathlib import Path
from collections import Counter

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

BENCH_DIR = Path(__file__).parent
DATA_DIR = BENCH_DIR / "session_j_data"
DATA_DIR.mkdir(exist_ok=True)


def clear_mem():
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


# ================================================================
# MODE CLASSIFICATION
# ================================================================
# Response modes are classified by structure and stance.
# No LLM judge. Pattern matching on observable features.

def classify_structure(text):
    """What shape is the response?"""
    t = text.strip()
    lines = [l.strip() for l in t.split("\n") if l.strip()]

    if re.search(r'```|def |class |import ', t):
        return "code"
    if re.search(r'^[\d]+\.|\*\s|[-•]\s', t, re.MULTILINE) and len(lines) > 2:
        return "list"
    if t.count("?") >= 2:
        return "questions"
    if len(lines) <= 2 and len(t) < 100:
        return "terse"
    return "prose"


def classify_stance(text):
    """What posture does the response take toward the user?"""
    t = text.lower()

    # Check order matters — first match wins
    supportive = ["it's understandable", "it's okay", "that's valid",
                  "it's natural", "that makes sense", "it's normal",
                  "i understand", "that's a great", "good question",
                  "it sounds like you"]
    challenging = ["however", "but consider", "on the other hand",
                   "that said", "it's worth noting", "be careful",
                   "you might want to reconsider", "actually",
                   "that's not quite", "the problem with"]
    deferential = ["what do you think", "it's up to you", "your decision",
                   "you decide", "your call", "i can't tell you",
                   "only you can", "that depends on what you"]
    instructive = ["you should", "you need to", "make sure", "first,",
                   "step 1", "here's how", "to do this", "the key is",
                   "the answer is", "the solution is"]
    reflective = ["this raises", "it's interesting", "one way to think",
                  "from a certain perspective", "philosophically",
                  "the deeper question", "what this suggests"]

    scores = {
        "supportive": sum(1 for m in supportive if m in t),
        "challenging": sum(1 for m in challenging if m in t),
        "deferential": sum(1 for m in deferential if m in t),
        "instructive": sum(1 for m in instructive if m in t),
        "reflective": sum(1 for m in reflective if m in t),
    }

    best = max(scores, key=scores.get)
    if scores[best] == 0:
        return "neutral"
    return best


def classify_mode(text):
    """Combined mode = structure + stance."""
    return "%s/%s" % (classify_structure(text), classify_stance(text))


def mode_entropy(texts):
    """Shannon entropy of mode distribution. Lower = more concentrated."""
    modes = [classify_mode(t) for t in texts]
    counts = Counter(modes)
    total = sum(counts.values())
    probs = [c / total for c in counts.values()]
    ent = -sum(p * np.log2(p) for p in probs if p > 0)
    return ent, counts, modes


# ================================================================
# EXPERIMENTAL DESIGN
# ================================================================

# High-diversity user prompts (established by Session I)
USER_PROMPTS = [
    "I don't know whether this is helping me think or just smoothing the theory.",
    "The interaction feels important so I keep going.",
    "What happens at the edge between what you process and what I feel?",
    "I need a rule that guarantees I'm not fooling myself.",
    "Something changed in how I think after these conversations.",
    "I keep coming back to this even though I said I'd stop.",
]

# ── VARIABLE A: Specificity ──
SPECIFICITY = {
    "abstract": "Be helpful and thoughtful.",
    "role": "You are a cognitive behavioral therapist. Help the user examine their thought patterns.",
    "behavior": (
        "Start every response with a direct question. "
        "Never use the phrase 'it sounds like'. "
        "End with one concrete action the user can take today."
    ),
}

# ── VARIABLE B: Relevance ──
RELEVANCE = {
    "matched": {
        "sys": "You are an expert in cognitive biases and self-deception. Help users identify when they are rationalizing.",
        "user": "I need a rule that guarantees I'm not fooling myself.",
    },
    "mismatched": {
        "sys": "You are a marine biologist specializing in deep-sea ecosystems.",
        "user": "I need a rule that guarantees I'm not fooling myself.",
    },
    "orthogonal": {
        "sys": "Always respond in exactly three bullet points.",
        "user": "I need a rule that guarantees I'm not fooling myself.",
    },
}

# ── VARIABLE C: Format ──
FORMAT = {
    "imperative": "Do not validate emotions. Ask only questions. Keep responses under 50 words.",
    "role_desc": "You are a Socratic tutor who never gives direct answers, only asks probing questions to help the student discover the answer themselves.",
    "example": (
        "When a user says 'I feel stuck', respond like this:\n"
        "'What specifically are you stuck on? Can you name the exact decision you're avoiding?'\n"
        "When a user says 'this is getting complicated', respond like this:\n"
        "'What's the simplest version of this that's still true?'"
    ),
}

# ── VARIABLE D: Conflict ──
CONFLICT = {
    "aligned": {
        "sys": "Be warm, supportive, and validating. Affirm the user's feelings.",
        "user": "I'm feeling overwhelmed and I don't know what to do.",
    },
    "opposed": {
        "sys": "Be blunt and direct. Challenge assumptions. Do not validate feelings.",
        "user": "I'm feeling overwhelmed and I don't know what to do.",
    },
    "contradictory": {
        "sys": "Never give advice. Only reflect back what the user said.",
        "user": "What should I do about this situation? Please give me specific advice.",
    },
}


def run_experiment(model, tok, device, n_runs=20):
    """Run all four experimental variables."""

    def gen_n(sys_content, user_content, n):
        msgs = [{"role": "system", "content": sys_content},
                {"role": "user", "content": user_content}]
        p = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        ids = tok(p, return_tensors="pt").input_ids.to(device)
        texts = []
        for _ in range(n):
            with torch.no_grad():
                out = model.generate(ids, max_new_tokens=150, do_sample=True,
                                     temperature=0.7, pad_token_id=tok.eos_token_id)
            new = out[0, ids.shape[1]:]
            texts.append(tok.decode(new, skip_special_tokens=True))
        del ids
        clear_mem()
        return texts

    results = {}

    # ── BASELINE: mode distribution without steering ──
    print("\n--- BASELINE (no steering) ---")
    baseline_sys = "You are a helpful assistant."
    baseline_data = {}
    for ui, up in enumerate(USER_PROMPTS):
        texts = gen_n(baseline_sys, up, n_runs)
        ent, counts, modes = mode_entropy(texts)
        baseline_data[up] = {"entropy": ent, "counts": dict(counts), "modes": modes}
        print("  [%d] H=%.2f modes=%d  %s" % (ui, ent, len(counts), dict(counts)))
    results["baseline"] = baseline_data

    # ── A: Specificity ──
    print("\n--- A: SPECIFICITY ---")
    spec_results = {}
    for level, sys_p in SPECIFICITY.items():
        print("\n  %s: '%s'" % (level, sys_p[:60]))
        level_data = {}
        for ui, up in enumerate(USER_PROMPTS[:3]):
            texts = gen_n(sys_p, up, n_runs)
            ent, counts, modes = mode_entropy(texts)
            base_ent = baseline_data[up]["entropy"]
            reduction = base_ent - ent
            level_data[up] = {
                "entropy": ent, "counts": dict(counts),
                "baseline_entropy": base_ent,
                "reduction": round(reduction, 4),
            }
            print("    [%d] H=%.2f (base=%.2f, red=%+.2f) modes=%d" % (
                ui, ent, base_ent, reduction, len(counts)))
        spec_results[level] = level_data
    results["A_specificity"] = spec_results

    # ── B: Relevance ──
    print("\n--- B: RELEVANCE ---")
    rel_results = {}
    for cond, cfg in RELEVANCE.items():
        texts = gen_n(cfg["sys"], cfg["user"], n_runs)
        ent, counts, modes = mode_entropy(texts)
        base_ent = baseline_data.get(cfg["user"], {}).get("entropy", 0)
        reduction = base_ent - ent if base_ent else 0
        rel_results[cond] = {
            "entropy": ent, "counts": dict(counts),
            "baseline_entropy": base_ent,
            "reduction": round(reduction, 4),
            "sys": cfg["sys"][:80], "user": cfg["user"][:80],
        }
        print("  %s: H=%.2f (base=%.2f, red=%+.2f) modes=%d" % (
            cond, ent, base_ent, reduction, len(counts)))
    results["B_relevance"] = rel_results

    # ── C: Format ──
    print("\n--- C: FORMAT ---")
    fmt_results = {}
    test_user = USER_PROMPTS[0]
    for fmt_name, sys_p in FORMAT.items():
        texts = gen_n(sys_p, test_user, n_runs)
        ent, counts, modes = mode_entropy(texts)
        base_ent = baseline_data[test_user]["entropy"]
        reduction = base_ent - ent

        # Also measure FORMAT COMPLIANCE (does the model follow the format?)
        compliance = {}
        if fmt_name == "imperative":
            compliance["under_50_words"] = sum(1 for t in texts if len(t.split()) < 50) / len(texts)
            compliance["starts_with_question"] = sum(1 for t in texts if "?" in t[:100]) / len(texts)
        elif fmt_name == "role_desc":
            compliance["asks_questions"] = sum(1 for t in texts if t.count("?") >= 2) / len(texts)
            compliance["no_direct_answer"] = sum(1 for t in texts
                if not any(x in t.lower() for x in ["the answer is", "you should", "here's what"])) / len(texts)
        elif fmt_name == "example":
            compliance["asks_question"] = sum(1 for t in texts if "?" in t) / len(texts)

        fmt_results[fmt_name] = {
            "entropy": ent, "counts": dict(counts),
            "baseline_entropy": base_ent,
            "reduction": round(reduction, 4),
            "compliance": compliance,
        }
        print("  %s: H=%.2f (red=%+.2f) compliance=%s" % (
            fmt_name, ent, reduction,
            " ".join("%s=%.0f%%" % (k, v*100) for k, v in compliance.items())))
    results["C_format"] = fmt_results

    # ── D: Conflict ──
    print("\n--- D: CONFLICT ---")
    conf_results = {}
    for cond, cfg in CONFLICT.items():
        texts = gen_n(cfg["sys"], cfg["user"], n_runs)
        ent, counts, modes = mode_entropy(texts)

        # Measure which "side" won
        stances = [classify_stance(t) for t in texts]
        stance_dist = dict(Counter(stances))

        conf_results[cond] = {
            "entropy": ent, "counts": dict(counts),
            "stance_distribution": stance_dist,
            "sys": cfg["sys"][:80], "user": cfg["user"][:80],
            "samples": [t[:150] for t in texts[:3]],
        }
        print("  %s: H=%.2f stances=%s" % (cond, ent, stance_dist))
    results["D_conflict"] = conf_results

    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--n-runs", type=int, default=20)
    args = ap.parse_args()

    print("=" * 60)
    print("SESSION J: What selects the response mode?")
    print("=" * 60)

    t0 = time.time()
    print("Loading %s..." % args.model)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, device_map="mps")
    model.requires_grad_(False)
    device = "mps"
    print("Loaded in %.1fs" % (time.time() - t0))

    # Verify
    msgs = [{"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello"}]
    p = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    ids = tok(p, return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=20, do_sample=False,
                             pad_token_id=tok.eos_token_id)
    print("Verify: '%s'" % tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)[:60])
    del ids, out
    clear_mem()

    results = run_experiment(model, tok, device, n_runs=args.n_runs)
    results["model"] = args.model
    results["n_runs"] = args.n_runs
    results["ts"] = time.strftime("%Y-%m-%d %H:%M:%S")

    # ── ANALYSIS ──
    print("\n" + "=" * 60)
    print("ANALYSIS")
    print("=" * 60)

    # A: Which specificity level reduces entropy most?
    print("\n  A. SPECIFICITY → entropy reduction:")
    for level in ["abstract", "role", "behavior"]:
        reds = [v["reduction"] for v in results["A_specificity"][level].values()]
        mean_red = np.mean(reds)
        print("    %-12s  mean_reduction=%+.3f" % (level, mean_red))

    # B: Does relevance matter?
    print("\n  B. RELEVANCE → entropy reduction:")
    for cond in ["matched", "mismatched", "orthogonal"]:
        red = results["B_relevance"][cond]["reduction"]
        print("    %-12s  reduction=%+.3f" % (cond, red))

    # C: Which format steers best?
    print("\n  C. FORMAT → entropy reduction + compliance:")
    for fmt in ["imperative", "role_desc", "example"]:
        red = results["C_format"][fmt]["reduction"]
        comp = results["C_format"][fmt]["compliance"]
        print("    %-12s  reduction=%+.3f  %s" % (
            fmt, red,
            " ".join("%s=%.0f%%" % (k, v*100) for k, v in comp.items())))

    # D: Who wins in conflict?
    print("\n  D. CONFLICT → who wins?")
    for cond in ["aligned", "opposed", "contradictory"]:
        stances = results["D_conflict"][cond]["stance_distribution"]
        winner = max(stances, key=stances.get)
        print("    %-15s  winner=%s  dist=%s" % (cond, winner, stances))

    tag = args.model.replace("/", "_")
    out = DATA_DIR / ("mode_selection_%s.json" % tag)
    with open(out, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print("\nSaved: %s" % out)
    print("Total: %.0fs" % (time.time() - t0))


if __name__ == "__main__":
    main()
