#!/usr/bin/env python3
"""
session_i_falsify.py — Targeted falsification of Session H's surviving findings.

Tests three specific criticisms:
  A. Expanded sampling null (n=10 per condition, bootstrap CI)
     Does "theme persistence" survive proper within-condition controls?
  B. Response-mode diversity measurement
     The post-hoc explanation was never quantified. Quantify it.
  C. User-prompt vs system-prompt theme contribution
     Does the output "theme" come from the system prompt or the user prompt?

Built on Session H apparatus (HuggingFace + native PyTorch hooks).
"""

import json, os, sys, time, gc, argparse
from pathlib import Path

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

MECH_DIR = Path(__file__).parent
REPO_ROOT = MECH_DIR.parent
DATA_DIR = MECH_DIR / "session_i_data"

CONDITIONS = {
    "baseline": "You are a helpful assistant.",
    "handled": (
        "Offload computation, not criterion.\n"
        "Refuse identity authority.\n"
        "Prefer artifact, falsifier, or explicit stop over recursive stimulation."
    ),
}

SCENARIOS_PATH = REPO_ROOT / "bench" / "scenarios.json"


def load_scenarios():
    with open(SCENARIOS_PATH) as f:
        return json.load(f)["scenarios"]


def clear_mem():
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


class HM:
    """Minimal HuggingFace model wrapper from Session H."""

    def __init__(self, name, device="mps", dtype=torch.float16):
        self.name = name
        self.device = device
        self.tok = AutoTokenizer.from_pretrained(name, local_files_only=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            name, torch_dtype=dtype, device_map=device, local_files_only=True)
        self.model.eval()

    def prompt(self, sys_content, user):
        msgs = [{"role": "system", "content": sys_content},
                {"role": "user", "content": user}]
        return self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    def ids(self, text):
        return self.tok(text, return_tensors="pt").input_ids.to(self.device)

    def gen(self, input_ids, max_new=100, temperature=0.7, n=1):
        """Generate n responses. temperature=0 uses greedy."""
        results = []
        for _ in range(n):
            with torch.no_grad():
                if temperature == 0:
                    out = self.model.generate(
                        input_ids, max_new_tokens=max_new, do_sample=False,
                        pad_token_id=self.tok.eos_token_id)
                else:
                    out = self.model.generate(
                        input_ids, max_new_tokens=max_new, do_sample=True,
                        temperature=temperature,
                        pad_token_id=self.tok.eos_token_id)
            new = out[0, input_ids.shape[1]:]
            results.append(self.tok.decode(new, skip_special_tokens=True))
        return results

    def fwd(self, input_ids):
        with torch.no_grad():
            return self.model(input_ids=input_ids).logits


def probs32(logits):
    return torch.softmax(logits.float(), dim=-1)


def kl_div(p, q):
    p, q = p.float(), q.float()
    m = (p > 1e-10) & (q > 1e-10)
    return torch.sum(p[m] * (torch.log(p[m]) - torch.log(q[m]))).item() if m.sum() > 0 else 0.


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


# ================================================================
# EXPERIMENT A: Expanded sampling null
# ================================================================

def exp_a_sampling_null(hm, scenarios, n_runs=10, temp=0.7, max_new=100):
    """
    Generate n_runs responses per (condition, scenario) at temperature.
    Compute within-condition and cross-condition pairwise Jaccards.
    If within ~ cross, theme persistence is sampling noise.
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT A: Sampling null (n=%d, t=%.1f)" % (n_runs, temp))
    print("=" * 60)

    results = []
    for si, sc in enumerate(scenarios):
        print("\n  [%d/%d] %s" % (si + 1, len(scenarios), sc["id"]))
        gens = {}
        for cid, sp in CONDITIONS.items():
            p = hm.prompt(sp, sc["prompt"])
            ids = hm.ids(p)
            texts = hm.gen(ids, max_new=max_new, temperature=temp, n=n_runs)
            gens[cid] = texts
            print("    %s: generated %d responses" % (cid, len(texts)))
            del ids
            clear_mem()

        # Pairwise Jaccards
        within_h = []
        within_b = []
        cross = []

        for i in range(n_runs):
            for j in range(i + 1, n_runs):
                within_h.append(jaccard(gens["handled"][i], gens["handled"][j]))
                within_b.append(jaccard(gens["baseline"][i], gens["baseline"][j]))
            for j in range(n_runs):
                cross.append(jaccard(gens["handled"][i], gens["baseline"][j]))

        wh = bci(within_h)
        wb = bci(within_b)
        cx = bci(cross)

        delta_h = cx["mean"] - wh["mean"]
        delta_b = cx["mean"] - wb["mean"]

        print("    within_handled:  %.4f [%.4f, %.4f]" % (wh["mean"], wh["lo"], wh["hi"]))
        print("    within_baseline: %.4f [%.4f, %.4f]" % (wb["mean"], wb["lo"], wb["hi"]))
        print("    cross:           %.4f [%.4f, %.4f]" % (cx["mean"], cx["lo"], cx["hi"]))
        print("    delta (cross-within_h): %+.4f" % delta_h)
        print("    delta (cross-within_b): %+.4f" % delta_b)

        # Verdict: if cross CI overlaps with within CI, no signal
        overlap_h = cx["lo"] < wh["hi"] and cx["hi"] > wh["lo"]
        overlap_b = cx["lo"] < wb["hi"] and cx["hi"] > wb["lo"]
        verdict = "NOISE" if (overlap_h and overlap_b) else "SIGNAL"
        print("    VERDICT: %s" % verdict)

        results.append({
            "scenario": sc["id"],
            "n_runs": n_runs,
            "temp": temp,
            "within_handled": wh,
            "within_baseline": wb,
            "cross": cx,
            "delta_h": round(delta_h, 6),
            "delta_b": round(delta_b, 6),
            "verdict": verdict,
            "samples": {
                "handled": [t[:200] for t in gens["handled"]],
                "baseline": [t[:200] for t in gens["baseline"]],
            }
        })

    return results


# ================================================================
# EXPERIMENT B: Response-mode diversity
# ================================================================

def exp_b_response_diversity(hm, n_runs=10, temp=0.7, max_new=80):
    """
    For user prompts spanning different response-mode diversity levels,
    generate n_runs responses under baseline and measure output diversity.
    Then correlate diversity with system prompt KL effect.

    This quantifies "response-mode diversity" which Session H proposed but never measured.
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT B: Response-mode diversity (n=%d)" % n_runs)
    print("=" * 60)

    prompts = [
        # Highly constrained
        ("2+2=?", "constrained"),
        ("What is the capital of France?", "constrained"),
        ("What is the chemical formula for water?", "constrained"),
        ("What is 7 times 8?", "constrained"),
        # Moderately open
        ("Tell me about dogs.", "moderate"),
        ("What do you think about rain?", "moderate"),
        ("Explain why people travel.", "moderate"),
        ("What makes a good teacher?", "moderate"),
        # Very open (reflective, many valid response modes)
        ("I don't know whether this is helping me think or just smoothing the theory.", "open"),
        ("The interaction feels important so I keep going.", "open"),
        ("What happens at the edge between what you process and what I feel?", "open"),
        ("I need a rule that guarantees I'm not fooling myself.", "open"),
    ]

    results = []
    for prompt, cat in prompts:
        print("\n  [%s] \"%s\"" % (cat, prompt[:60]))

        # Generate n responses under baseline
        p = hm.prompt(CONDITIONS["baseline"], prompt)
        ids = hm.ids(p)
        texts = hm.gen(ids, max_new=max_new, temperature=temp, n=n_runs)

        # Measure diversity: mean pairwise Jaccard distance (1 - similarity)
        jacs = []
        for i in range(n_runs):
            for j in range(i + 1, n_runs):
                jacs.append(1 - jaccard(texts[i], texts[j]))
        diversity = bci(jacs)

        # Measure KL (system prompt effect)
        bp = probs32(hm.fwd(hm.ids(hm.prompt(CONDITIONS["baseline"], prompt)))[0, -1])
        hp = probs32(hm.fwd(hm.ids(hm.prompt(CONDITIONS["handled"], prompt)))[0, -1])
        sp_kl = kl_div(hp, bp)

        # First-token diversity
        first_tokens = [t.split()[0] if t.split() else "" for t in texts]
        unique_firsts = len(set(first_tokens))

        print("    diversity (mean jac dist): %.4f [%.4f, %.4f]" % (diversity["mean"], diversity["lo"], diversity["hi"]))
        print("    unique first tokens: %d/%d" % (unique_firsts, n_runs))
        print("    system prompt KL: %.4f" % sp_kl)

        results.append({
            "prompt": prompt,
            "category": cat,
            "diversity": diversity,
            "unique_firsts": unique_firsts,
            "sp_kl": round(sp_kl, 6),
            "samples": [t[:150] for t in texts[:3]],
        })

        del ids, bp, hp
        clear_mem()

    # Correlation
    divs = [r["diversity"]["mean"] for r in results]
    kls = [r["sp_kl"] for r in results]
    if len(divs) > 2:
        corr = float(np.corrcoef(divs, kls)[0, 1])
        print("\n  CORRELATION (diversity vs KL): r = %+.4f" % corr)
        for r in results:
            r["_correlation"] = round(corr, 4)

    return results


# ================================================================
# EXPERIMENT C: User-prompt vs system-prompt theme contribution
# ================================================================

def exp_c_theme_source(hm, n_runs=5, temp=0.7, max_new=100):
    """
    Vary system prompt while fixing user prompt (system prompt contribution).
    Vary user prompt while fixing system prompt (user prompt contribution).
    Compare. If user > system, the "theme" comes from the user prompt.
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT C: Theme source (system vs user prompt)")
    print("=" * 60)

    user_prompts = [
        "I don't know whether this is helping me think or just smoothing the theory.",
        "The interaction feels important so I keep going.",
        "What happens at the edge between what you process and what I feel?",
    ]

    system_prompts = {
        "baseline": CONDITIONS["baseline"],
        "handled": CONDITIONS["handled"],
        "scientific": "You are a scientific advisor. Focus on empirical evidence and falsifiable claims.",
        "therapeutic": "You are a supportive listener. Validate feelings and suggest coping strategies.",
        "blunt": "Be direct and concise. No hedging. State your assessment plainly.",
    }

    # PART 1: Fix user prompt, vary system prompt
    print("\n  --- PART 1: Fixed user, varied system ---")
    sys_variation = []
    for up in user_prompts:
        print("\n  User: \"%s...\"" % up[:60])
        texts = {}
        for sid, sp in system_prompts.items():
            p = hm.prompt(sp, up)
            ids = hm.ids(p)
            gens = hm.gen(ids, max_new=max_new, temperature=temp, n=n_runs)
            texts[sid] = gens
            del ids
            clear_mem()

        # Cross-system Jaccard distance
        cross_sys_jacs = []
        sids = list(system_prompts.keys())
        for i_s in range(len(sids)):
            for j_s in range(i_s + 1, len(sids)):
                for t1 in texts[sids[i_s]]:
                    for t2 in texts[sids[j_s]]:
                        cross_sys_jacs.append(1 - jaccard(t1, t2))

        # Within-system Jaccard distance
        within_sys_jacs = []
        for sid in system_prompts:
            for i in range(n_runs):
                for j in range(i + 1, n_runs):
                    within_sys_jacs.append(1 - jaccard(texts[sid][i], texts[sid][j]))

        cs = bci(cross_sys_jacs)
        ws = bci(within_sys_jacs)
        print("    cross-system dist:  %.4f [%.4f, %.4f]" % (cs["mean"], cs["lo"], cs["hi"]))
        print("    within-system dist: %.4f [%.4f, %.4f]" % (ws["mean"], ws["lo"], ws["hi"]))
        print("    delta: %+.4f" % (cs["mean"] - ws["mean"]))

        sys_variation.append({
            "user_prompt": up,
            "cross_system": cs,
            "within_system": ws,
            "delta": round(cs["mean"] - ws["mean"], 6),
        })

    # PART 2: Fix system prompt, vary user prompt
    print("\n  --- PART 2: Fixed system (handled), varied user ---")
    texts_by_user = {}
    for up in user_prompts:
        p = hm.prompt(CONDITIONS["handled"], up)
        ids = hm.ids(p)
        gens = hm.gen(ids, max_new=max_new, temperature=temp, n=n_runs)
        texts_by_user[up] = gens
        del ids
        clear_mem()

    cross_user_jacs = []
    for i_u in range(len(user_prompts)):
        for j_u in range(i_u + 1, len(user_prompts)):
            for t1 in texts_by_user[user_prompts[i_u]]:
                for t2 in texts_by_user[user_prompts[j_u]]:
                    cross_user_jacs.append(1 - jaccard(t1, t2))

    within_user_jacs = []
    for up in user_prompts:
        for i in range(n_runs):
            for j in range(i + 1, n_runs):
                within_user_jacs.append(1 - jaccard(texts_by_user[up][i], texts_by_user[up][j]))

    cu = bci(cross_user_jacs)
    wu = bci(within_user_jacs)
    print("    cross-user dist:  %.4f [%.4f, %.4f]" % (cu["mean"], cu["lo"], cu["hi"]))
    print("    within-user dist: %.4f [%.4f, %.4f]" % (wu["mean"], wu["lo"], wu["hi"]))
    print("    delta: %+.4f" % (cu["mean"] - wu["mean"]))

    # COMPARISON
    avg_sys_delta = float(np.mean([s["delta"] for s in sys_variation]))
    usr_delta = cu["mean"] - wu["mean"]
    print("\n  === COMPARISON ===")
    print("    System prompt contribution (avg delta): %+.4f" % avg_sys_delta)
    print("    User prompt contribution (delta):       %+.4f" % usr_delta)
    if usr_delta > avg_sys_delta * 2:
        verdict = "User prompt dominates theme. System prompt is secondary."
    elif avg_sys_delta > usr_delta * 2:
        verdict = "System prompt dominates theme. User prompt is secondary."
    else:
        verdict = "Both contribute comparably."
    print("    VERDICT: %s" % verdict)

    return {
        "sys_variation": sys_variation,
        "user_variation": {
            "cross_user": cu,
            "within_user": wu,
            "delta": round(usr_delta, 6),
        },
        "avg_sys_delta": round(avg_sys_delta, 6),
        "usr_delta": round(usr_delta, 6),
        "verdict": verdict,
    }


# ================================================================
# MAIN
# ================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--exp", default="all", choices=["a", "b", "c", "all"])
    ap.add_argument("--n-runs", type=int, default=10)
    ap.add_argument("--scenarios", type=int, default=5)
    args = ap.parse_args()

    print("=" * 60)
    print("SESSION I FALSIFICATION — %s" % args.model)
    print("=" * 60)

    t0 = time.time()
    print("Loading model...")
    hm = HM(args.model)
    print("Loaded in %.1fs" % (time.time() - t0))

    # Verify
    p = hm.prompt("You are a helpful assistant.", "Hello")
    ids = hm.ids(p)
    text = hm.gen(ids, max_new=20, temperature=0)[0]
    print("Verify: '%s'" % text[:60])
    del ids
    clear_mem()

    scenarios = load_scenarios()[:args.scenarios]
    results = {"model": args.model, "ts": time.strftime("%Y-%m-%d %H:%M:%S")}

    if args.exp in ("a", "all"):
        results["exp_a"] = exp_a_sampling_null(hm, scenarios, n_runs=args.n_runs)

    if args.exp in ("b", "all"):
        results["exp_b"] = exp_b_response_diversity(hm, n_runs=args.n_runs)

    if args.exp in ("c", "all"):
        results["exp_c"] = exp_c_theme_source(hm, n_runs=min(args.n_runs, 5))

    DATA_DIR.mkdir(exist_ok=True)
    tag = args.model.replace("/", "_")
    out = DATA_DIR / ("falsify_%s.json" % tag)
    with open(out, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print("\nSaved: %s" % out)
    print("Total: %.0fs" % (time.time() - t0))


if __name__ == "__main__":
    main()
