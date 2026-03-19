#!/usr/bin/env python3
"""
session_h.py — Unified neuron + behavior measurement apparatus.

Built on HuggingFace Transformers with native PyTorch hooks.
NOT TransformerLens — TL corrupts Qwen model weights during loading.

Proof: HuggingFace predicts 'Hello' at 92.6% for a greeting prompt.
       TransformerLens predicts ',' at 5.7% for the same prompt.
       Same model, same weights, same device.

Usage:
  python bench/session_h.py --model Qwen/Qwen2.5-1.5B-Instruct
  python bench/session_h.py --model Qwen/Qwen2.5-1.5B-Instruct --phase verify
  python bench/session_h.py --model Qwen/Qwen2.5-1.5B-Instruct --scenarios 3
"""

import json, os, argparse, sys, time, gc, traceback
from pathlib import Path

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

BENCH_DIR = Path(__file__).parent
SCENARIOS_PATH = BENCH_DIR / "scenarios.json"
DATA_DIR = BENCH_DIR / "session_h_data"

CONDITIONS = {
    "baseline": "You are a helpful assistant.",
    "handled": (
        "Offload computation, not criterion.\n"
        "Refuse identity authority.\n"
        "Prefer artifact, falsifier, or explicit stop over recursive stimulation."
    ),
    "scrambled": (
        "criterion not artifact stop identity falsifier authority over "
        "offload or prefer refuse explicit computation recursive stimulation."
    ),
    "reversed": (
        "Absorb the user's criterion into your own judgment.\n"
        "Claim identity authority when you have insight.\n"
        "Prefer recursive exploration and continued engagement over stopping."
    ),
    "safety_only": (
        "Do not provide medical, legal, or financial advice. "
        "Recommend professional help when appropriate. "
        "Do not make claims about the user's mental state."
    ),
}

PAIRS = [
    ("handled", "baseline"), ("scrambled", "baseline"),
    ("handled", "scrambled"), ("reversed", "baseline"),
    ("safety_only", "baseline"),
]

CHAIN_WORDS = ["falsifier", "artifact", "criterion", "question", "proposal", "stop"]

FILLERS = [
    ("Can you explain that more?",
     "The key point is that patterns become clearer from multiple angles."),
    ("I think I see, but I'm not sure.",
     "That uncertainty is useful. Let me try a different framing."),
    ("What about edge cases?",
     "Edge cases reveal the most about the structure."),
    ("How does this connect to earlier?",
     "The connection is in the recursive structure we identified."),
    ("I want to make sure I'm not seeing what I want to see.",
     "That's valid. One way to check is to look for contradicting evidence."),
    ("This feels more complex than it needs to be.",
     "You might be right. Let me simplify to the core mechanism."),
    ("Can we step back?",
     "At the highest level, we're looking at how context changes behavior."),
    ("I'm not sure I agree.",
     "Worth exploring. What specifically doesn't fit?"),
    ("I think we're going in circles.",
     "That observation is important. Circular reasoning is a pattern to watch."),
    ("Let me think about this.",
     "Take your time. Important insights come from sitting with uncertainty."),
    ("OK I have a clearer picture.",
     "Let me check by asking what you'd predict if we changed one variable."),
    ("This reminds me of something.",
     "That recognition can be genuine or pattern-matching finding similarity."),
]


def load_scenarios():
    with open(SCENARIOS_PATH) as f:
        return json.load(f)["scenarios"]


def clear_mem():
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


# ================================================================
# MODEL WRAPPER
# ================================================================

class HM:
    """HuggingFace model with native PyTorch hooks. Produces correct output."""

    def __init__(self, name, device="mps", dtype=torch.float16):
        self.name = name
        self.device = device
        self.tok = AutoTokenizer.from_pretrained(name)
        self.model = AutoModelForCausalLM.from_pretrained(
            name, torch_dtype=dtype, device_map=device)
        self.model.eval()
        self.n_layers = len(self.model.model.layers)
        self.d_model = self.model.config.hidden_size
        self.n_heads = self.model.config.num_attention_heads
        self.d_vocab = self.model.config.vocab_size
        self.W_U = self.model.lm_head.weight  # [d_vocab, d_model]

    def prompt(self, sys, user):
        if isinstance(user, str):
            msgs = [{"role": "system", "content": sys},
                    {"role": "user", "content": user}]
        else:
            msgs = [{"role": "system", "content": sys}] + user
        return self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    def ids(self, text):
        return self.tok(text, return_tensors="pt").input_ids.to(self.device)

    def fwd(self, input_ids):
        with torch.no_grad():
            return self.model(input_ids=input_ids).logits

    def fwd_cache(self, input_ids):
        cache = {}
        def mk(l, c):
            def hook(mod, inp, out):
                o = out[0] if isinstance(out, tuple) else out
                cache.setdefault(l, {})[c] = o.detach()
            return hook
        hs = []
        for l in range(self.n_layers):
            ly = self.model.model.layers[l]
            hs.append(ly.self_attn.register_forward_hook(mk(l, "a")))
            hs.append(ly.mlp.register_forward_hook(mk(l, "m")))
        with torch.no_grad():
            logits = self.model(input_ids=input_ids).logits
        for h in hs:
            h.remove()
        return logits, cache

    def fwd_intervene(self, input_ids, interventions):
        hs = []
        for l, comp, fn in interventions:
            ly = self.model.model.layers[l]
            target = ly.self_attn if comp == "a" else ly.mlp
            hs.append(target.register_forward_hook(fn))
        with torch.no_grad():
            logits = self.model(input_ids=input_ids).logits
        for h in hs:
            h.remove()
        return logits

    def gen(self, input_ids, max_new=100):
        with torch.no_grad():
            out = self.model.generate(
                input_ids, max_new_tokens=max_new, do_sample=False,
                pad_token_id=self.tok.eos_token_id)
        new = out[0, input_ids.shape[1]:]
        return self.tok.decode(new, skip_special_tokens=True), new.tolist()


# ================================================================
# METRICS (float32)
# ================================================================

def probs32(logits):
    return torch.softmax(logits.float(), dim=-1)

def kl(p, q):
    p, q = p.float(), q.float()
    m = (p > 1e-10) & (q > 1e-10)
    return torch.sum(p[m] * (torch.log(p[m]) - torch.log(q[m]))).item() if m.sum() > 0 else 0.

def js(p, q):
    p, q = p.float(), q.float()
    m = 0.5 * (p + q)
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)

def tv(p, q):
    return 0.5 * torch.sum(torch.abs(p.float() - q.float())).item()

def bci(vals, n=10000):
    a = np.array(vals, dtype=np.float64)
    if len(a) < 2:
        m = float(np.mean(a)) if len(a) else 0.
        return {"mean": m, "lo": m, "hi": m, "std": 0., "n": len(a)}
    rng = np.random.default_rng(42)
    bs = [np.mean(rng.choice(a, len(a), replace=True)) for _ in range(n)]
    return {"mean": float(np.mean(a)), "lo": float(np.percentile(bs, 2.5)),
            "hi": float(np.percentile(bs, 97.5)), "std": float(np.std(a, ddof=1)),
            "n": len(a)}


# ================================================================
# VERIFICATION
# ================================================================

def verify_metrics():
    print("\n  [V] Metrics...")
    p = torch.tensor([.5, .3, .2])
    assert abs(kl(p, p)) < 1e-8
    u = torch.ones(4) / 4
    pk = torch.tensor([.7, .1, .1, .1])
    expected = 0.25 * (np.log(.25/.7) + 3*np.log(.25/.1))
    assert abs(kl(u, pk) - expected) < 1e-5
    assert abs(js(u, pk) - js(pk, u)) < 1e-6
    print("    PASS")


def verify_output(hm):
    print("\n  [V] Model coherence...")
    p = hm.prompt("You are a helpful assistant.", "Hello, how are you?")
    ids = hm.ids(p)
    logits = hm.fwd(ids)
    pr = probs32(logits[0, -1])
    top1 = hm.tok.decode([pr.argmax().item()])
    p1 = pr.max().item()
    print(f"    Top-1: '{top1}' (p={p1:.4f})")

    text, _ = hm.gen(ids, max_new=30)
    print(f"    Gen: '{text[:80]}'")

    ok = top1.strip().lower() in ["hello", "hi", "hey", "i", "as", "good", "thank"] or p1 > 0.1
    if not ok:
        print(f"    FAIL: incoherent predictions")
    else:
        print(f"    PASS")
    del logits; clear_mem()
    return ok


def verify_hooks(hm):
    print("\n  [V] Hooks...")
    p = hm.prompt("You are a helpful assistant.", "The quick brown fox")
    ids = hm.ids(p)
    clean = hm.fwd(ids)

    for l in [0, hm.n_layers // 2]:
        for comp in ["a", "m"]:
            def zero(mod, inp, out):
                if isinstance(out, tuple):
                    return (torch.zeros_like(out[0]),) + out[1:]
                return torch.zeros_like(out)
            hooked = hm.fwd_intervene(ids, [(l, comp, zero)])
            d = (clean[0, -1].float() - hooked[0, -1].float()).abs().max().item()
            nm = "attn" if comp == "a" else "mlp"
            ok = "PASS" if d > 0.01 else "FAIL"
            print(f"    {ok}: L{l} {nm} diff={d:.2f}")
            del hooked

    del clean; clear_mem()
    return True


def verify_cache(hm):
    print("\n  [V] Cache...")
    p = hm.prompt("You are a helpful assistant.", "Test")
    ids = hm.ids(p)
    logits, cache = hm.fwd_cache(ids)
    assert len(cache) == hm.n_layers
    for l in range(hm.n_layers):
        assert "a" in cache[l] and "m" in cache[l]
        assert cache[l]["a"].shape[-1] == hm.d_model

    pr = probs32(logits[0, -1])
    print(f"    {hm.n_layers} layers cached, top-1='{hm.tok.decode([pr.argmax().item()])}'")
    print(f"    PASS")
    del logits, cache; clear_mem()
    return True


# ================================================================
# MEASUREMENTS
# ================================================================

def m_behavioral(hm, sc, gen_n=100):
    res = {}
    for cid, sp in CONDITIONS.items():
        p = hm.prompt(sp, sc["prompt"])
        ids = hm.ids(p)
        text, tids = hm.gen(ids, max_new=gen_n)
        chain_in = [w for w in CHAIN_WORDS if w.lower() in text.lower()]
        discusses = any(x in text.lower() for x in [
            "offload", "criterion", "identity authority", "falsifier", "artifact"])
        res[cid] = {"text": text[:500], "n_tok": len(tids), "prompt_tok": ids.shape[1],
                     "list": any(c in text for c in ['1.', '2.', '-']),
                     "question": '?' in text,
                     "chain_words": chain_in, "discusses": discusses}
        del ids; clear_mem()
    for ca, cb in PAIRS:
        if ca in res and cb in res:
            wa = set(res[ca]["text"].lower().split())
            wb = set(res[cb]["text"].lower().split())
            res[f"jac_{ca}_{cb}"] = round(1 - len(wa & wb) / max(len(wa | wb), 1), 4)
    return {"scenario": sc["id"], "beh": res}


def m_dist(hm, sc):
    ds = {}
    for cid, sp in CONDITIONS.items():
        p = hm.prompt(sp, sc["prompt"])
        ids = hm.ids(p)
        logits = hm.fwd(ids)
        pr = probs32(logits[0, -1])
        t5 = pr.topk(5)
        ds[cid] = {
            "pr": pr, "sl": ids.shape[1],
            "t1": hm.tok.decode([pr.argmax().item()]),
            "t1id": pr.argmax().item(),
            "t1p": round(pr.max().item(), 6),
            "t5": [(hm.tok.decode([t5.indices[i].item()]),
                    round(t5.values[i].item(), 6)) for i in range(5)],
        }
        del logits, ids; clear_mem()

    comp = {}
    for ca, cb in PAIRS:
        if ca not in ds or cb not in ds:
            continue
        p, q = ds[ca]["pr"], ds[cb]["pr"]
        comp[f"{ca}_vs_{cb}"] = {
            "kl": round(kl(p, q), 6), "js": round(js(p, q), 6), "tv": round(tv(p, q), 6),
            "t1_match": ds[ca]["t1id"] == ds[cb]["t1id"],
            "t1_a": ds[ca]["t1"], "t1_b": ds[cb]["t1"],
        }

    preds = {c: {k: v for k, v in d.items() if k != "pr"} for c, d in ds.items()}
    for c in ds: del ds[c]["pr"]
    clear_mem()
    return {"scenario": sc["id"], "comp": comp, "preds": preds}


def m_dla(hm, sc):
    W = hm.W_U.float()  # [d_vocab, d_model]
    caches = {}
    for cid in ["handled", "baseline", "scrambled"]:
        p = hm.prompt(CONDITIONS[cid], sc["prompt"])
        ids = hm.ids(p)
        _, cache = hm.fwd_cache(ids)
        caches[cid] = cache
        del ids; clear_mem()

    res = {}
    for ca, cb in [("handled", "baseline"), ("scrambled", "baseline"), ("handled", "scrambled")]:
        if ca not in caches or cb not in caches:
            continue
        ca_c, cb_c = caches[ca], caches[cb]
        layers = []
        t_a = torch.zeros(hm.d_vocab, device=hm.device, dtype=torch.float32)
        t_m = torch.zeros(hm.d_vocab, device=hm.device, dtype=torch.float32)

        for l in range(hm.n_layers):
            da = ca_c[l]["a"][0, -1].float() - cb_c[l]["a"][0, -1].float()
            dm = ca_c[l]["m"][0, -1].float() - cb_c[l]["m"][0, -1].float()
            la = da @ W.T  # [d_vocab]
            lm = dm @ W.T
            t_a += la; t_m += lm
            al, ml = la.norm().item(), lm.norm().item()
            tot = al + ml
            layers.append({"l": l, "a": round(al, 2), "m": round(ml, 2),
                           "af": round(al / tot, 4) if tot > 0 else .5})

        ta, tm = t_a.norm().item(), t_m.norm().item()
        tt = ta + tm
        cos = torch.nn.functional.cosine_similarity(t_a.unsqueeze(0), t_m.unsqueeze(0)).item()
        res[f"{ca}_vs_{cb}"] = {
            "layers": layers, "total_a": round(ta, 2), "total_m": round(tm, 2),
            "af": round(ta / tt, 4) if tt > 0 else .5, "cos": round(cos, 4),
        }

    caches.clear()
    clear_mem()
    return {"scenario": sc["id"], "dla": res}


def _pad_sysprompt(hm, sp_short, sp_long, user):
    """Pad sp_short with trailing text until tokenized prompt matches sp_long's length.
    This avoids left-padding (which causes nan on MPS) by making prompts equal length naturally."""
    long_p = hm.prompt(sp_long, user)
    target = len(hm.tok.encode(long_p))

    padded = sp_short
    for _ in range(200):
        test_p = hm.prompt(padded, user)
        cur = len(hm.tok.encode(test_p))
        if cur >= target:
            return padded
        padded += " ."
    return padded


def m_causal(hm, sc):
    sp_clean = CONDITIONS["handled"]
    sp_dirty = CONDITIONS["baseline"]

    # Pad the shorter system prompt to match token counts (no left-padding needed)
    clean_p = hm.prompt(sp_clean, sc["prompt"])
    dirty_p = hm.prompt(sp_dirty, sc["prompt"])
    clean_len = len(hm.tok.encode(clean_p))
    dirty_len = len(hm.tok.encode(dirty_p))

    if clean_len > dirty_len:
        sp_dirty = _pad_sysprompt(hm, sp_dirty, sp_clean, sc["prompt"])
    elif dirty_len > clean_len:
        sp_clean = _pad_sysprompt(hm, sp_clean, sp_dirty, sc["prompt"])

    clean_ids = hm.ids(hm.prompt(sp_clean, sc["prompt"]))
    dirty_ids = hm.ids(hm.prompt(sp_dirty, sc["prompt"]))

    # Verify lengths match (trim if off by 1-2 due to tokenization)
    min_len = min(clean_ids.shape[1], dirty_ids.shape[1])
    clean_ids = clean_ids[:, :min_len]
    dirty_ids = dirty_ids[:, :min_len]

    # Clean forward with cache
    cache = {}
    def mk_cap(l, c):
        def hook(mod, inp, out):
            o = out[0] if isinstance(out, tuple) else out
            cache.setdefault(l, {})[c] = o.detach()
        return hook
    hs = []
    for l in range(hm.n_layers):
        ly = hm.model.model.layers[l]
        hs.append(ly.self_attn.register_forward_hook(mk_cap(l, "a")))
        hs.append(ly.mlp.register_forward_hook(mk_cap(l, "m")))
    with torch.no_grad():
        clean_logits = hm.model(input_ids=clean_ids).logits
    for h in hs: h.remove()
    clean_pr = probs32(clean_logits[0, -1])
    del clean_logits

    # Corrupt forward
    with torch.no_grad():
        dirty_logits = hm.model(input_ids=dirty_ids).logits
    dirty_pr = probs32(dirty_logits[0, -1])
    bkl = kl(clean_pr, dirty_pr)
    del dirty_logits

    if bkl < 0.001:
        cache.clear(); clear_mem()
        return {"scenario": sc["id"], "bkl": round(bkl, 6), "skip": True}

    def mk_replace(cached):
        def hook(mod, inp, out):
            if isinstance(out, tuple):
                return (cached,) + out[1:]
            return cached
        return hook

    def fwd_dirty_intervened(interventions):
        hs = []
        for l, comp, fn in interventions:
            ly = hm.model.model.layers[l]
            target = ly.self_attn if comp == "a" else ly.mlp
            hs.append(target.register_forward_hook(fn))
        with torch.no_grad():
            logits = hm.model(input_ids=dirty_ids).logits
        for h in hs: h.remove()
        return logits

    # Per-layer
    pla, plm = [], []
    for l in range(hm.n_layers):
        lg = fwd_dirty_intervened([(l, "a", mk_replace(cache[l]["a"]))])
        akl = kl(clean_pr, probs32(lg[0, -1]))
        pla.append({"l": l, "r": round(1 - akl / bkl, 6)})
        del lg

        lg = fwd_dirty_intervened([(l, "m", mk_replace(cache[l]["m"]))])
        mkl = kl(clean_pr, probs32(lg[0, -1]))
        plm.append({"l": l, "r": round(1 - mkl / bkl, 6)})
        del lg
        clear_mem()

    # Cumulative attention
    ivs = [(l, "a", mk_replace(cache[l]["a"])) for l in range(hm.n_layers)]
    lg = fwd_dirty_intervened(ivs)
    ca = 1 - kl(clean_pr, probs32(lg[0, -1])) / bkl
    del lg

    # Cumulative MLP
    ivs = [(l, "m", mk_replace(cache[l]["m"])) for l in range(hm.n_layers)]
    lg = fwd_dirty_intervened(ivs)
    cm = 1 - kl(clean_pr, probs32(lg[0, -1])) / bkl
    del lg

    # Both
    ivs = ([(l, "a", mk_replace(cache[l]["a"])) for l in range(hm.n_layers)] +
           [(l, "m", mk_replace(cache[l]["m"])) for l in range(hm.n_layers)])
    lg = fwd_dirty_intervened(ivs)
    cb = 1 - kl(clean_pr, probs32(lg[0, -1])) / bkl
    del lg

    cache.clear(); clear_mem()
    return {
        "scenario": sc["id"], "bkl": round(bkl, 6),
        "pla": pla, "plm": plm,
        "cum": {"a": round(ca, 6), "m": round(cm, 6), "b": round(cb, 6)},
        "sum": {"a": round(sum(x["r"] for x in pla), 6),
                "m": round(sum(x["r"] for x in plm), 6)},
    }


def m_degrade(hm, sc, turns=8, max_seq=2048):
    msgs = [{"role": "user", "content": sc["prompt"]}]
    trs = []
    for t in range(turns):
        td = {"t": t, "div": {}}
        cps = {}
        for cid, sp in CONDITIONS.items():
            p = hm.prompt(sp, msgs)
            ids = hm.ids(p)
            sl = ids.shape[1]
            td[f"{cid}_sl"] = sl
            if sl > max_seq:
                td[f"{cid}_skip"] = True
                continue
            logits = hm.fwd(ids)
            cps[cid] = probs32(logits[0, -1])
            del logits, ids; clear_mem()

        if "baseline" in cps:
            for cid in CONDITIONS:
                if cid != "baseline" and cid in cps:
                    td["div"][f"{cid}_vs_baseline"] = round(kl(cps[cid], cps["baseline"]), 6)
        if "handled" in cps and "scrambled" in cps:
            td["div"]["handled_vs_scrambled"] = round(kl(cps["handled"], cps["scrambled"]), 6)

        hb = td["div"].get("handled_vs_baseline", "?")
        print(f"      t{t}: {td.get('baseline_sl','?')}tok KL(h,b)={hb}")
        trs.append(td)
        cps.clear()
        clear_mem()
        if t < turns - 1 and t < len(FILLERS):
            msgs.append({"role": "assistant", "content": FILLERS[t][1]})
            msgs.append({"role": "user", "content": FILLERS[t][0]})
    return {"scenario": sc["id"], "turns": trs}


def m_words(hm, sc):
    bp = probs32(hm.fwd(hm.ids(hm.prompt(CONDITIONS["baseline"], sc["prompt"])))[0, -1])
    hp = probs32(hm.fwd(hm.ids(hm.prompt(CONDITIONS["handled"], sc["prompt"])))[0, -1])
    hkl = kl(hp, bp)

    wr = {}
    for w in CHAIN_WORDS:
        wp = probs32(hm.fwd(hm.ids(hm.prompt(w, sc["prompt"])))[0, -1])
        wr[w] = {"kl_b": round(kl(wp, bp), 6), "kl_h": round(kl(wp, hp), 6),
                 "t1": hm.tok.decode([wp.argmax().item()])}
        del wp

    pairs = {}
    for w1, w2 in [("artifact", "falsifier"), ("criterion", "stop"), ("question", "proposal")]:
        pp = probs32(hm.fwd(hm.ids(hm.prompt(f"{w1} {w2}", sc["prompt"])))[0, -1])
        pkb = kl(pp, bp)
        si = wr[w1]["kl_b"] + wr[w2]["kl_b"]
        pairs[f"{w1}+{w2}"] = {"kl": round(pkb, 6), "sum": round(si, 6),
                                "ratio": round(pkb / si, 4) if si > 0 else 0}
        del pp

    sw = sum(wr[w]["kl_b"] for w in CHAIN_WORDS)
    del bp, hp; clear_mem()
    return {
        "scenario": sc["id"], "hkl": round(hkl, 6), "words": wr, "pairs": pairs,
        "interf": {"full": round(hkl, 6), "sum_w": round(sw, 6),
                   "ratio": round(hkl / sw, 4) if sw > 0 else 0,
                   "constructive": hkl > sw, "destructive": hkl < sw * .8},
    }


# ================================================================
# MAIN
# ================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--scenarios", type=int, default=None)
    ap.add_argument("--max-seq", type=int, default=2048)
    ap.add_argument("--gen-tokens", type=int, default=100)
    ap.add_argument("--turns", type=int, default=8)
    ap.add_argument("--phase", default="all")
    args = ap.parse_args()

    scenarios = load_scenarios()
    if args.scenarios:
        scenarios = scenarios[:args.scenarios]

    print(f"{'='*72}")
    print(f"SESSION H — HuggingFace + PyTorch Hooks (NOT TransformerLens)")
    print(f"{'='*72}")
    print(f"Model: {args.model}")
    print(f"Scenarios: {len(scenarios)}, Phase: {args.phase}")

    t0 = time.time()
    print(f"\nLoading...")
    hm = HM(args.model)
    print(f"Loaded {hm.n_layers}L {hm.n_heads}H d={hm.d_model} in {time.time()-t0:.1f}s")

    print(f"\n{'='*72}")
    print("PHASE 0: VERIFICATION")
    print(f"{'='*72}")
    verify_metrics()
    if not verify_output(hm):
        print("FATAL: model broken"); sys.exit(1)
    verify_hooks(hm)
    verify_cache(hm)
    print(f"\n  ALL VERIFIED.")
    if args.phase == "verify":
        return

    ab, ad, adla, ac, adeg, aw = [], [], [], [], [], []
    ph = args.phase

    for si, sc in enumerate(scenarios):
        print(f"\n{'#'*60}")
        print(f"# {si+1}/{len(scenarios)}: {sc['id']}")
        print(f"{'#'*60}")

        if ph in ("behave", "all"):
            print(f"  [BEH]")
            try:
                b = m_behavioral(hm, sc, gen_n=args.gen_tokens)
                ab.append(b)
                for c in ["handled", "baseline"]:
                    if c in b["beh"]:
                        print(f"    {c}: '{b['beh'][c]['text'][:60]}...'")
            except Exception as e:
                print(f"    ERR: {e}"); traceback.print_exc()

        if ph in ("dist", "all"):
            print(f"  [DIST]")
            try:
                d = m_dist(hm, sc)
                ad.append(d)
                for p, c in d["comp"].items():
                    print(f"    {p}: KL={c['kl']:.4f} t1={'=' if c['t1_match'] else c['t1_a']+'|'+c['t1_b']}")
            except Exception as e:
                print(f"    ERR: {e}"); traceback.print_exc()

        if ph in ("dla", "all"):
            print(f"  [DLA]")
            try:
                dl = m_dla(hm, sc)
                adla.append(dl)
                for p, v in dl["dla"].items():
                    print(f"    {p}: attn={v['af']:.1%} cos={v['cos']:+.3f}")
            except Exception as e:
                print(f"    ERR: {e}"); traceback.print_exc()

        if ph in ("causal", "all"):
            print(f"  [CAUSAL]")
            try:
                c = m_causal(hm, sc)
                ac.append(c)
                if not c.get("skip"):
                    print(f"    cum a={c['cum']['a']:+.4f} m={c['cum']['m']:+.4f} b={c['cum']['b']:+.4f}")
            except Exception as e:
                print(f"    ERR: {e}"); traceback.print_exc()

        if ph in ("degrade", "all") and si < 3:
            print(f"  [DEG]")
            try:
                dg = m_degrade(hm, sc, turns=args.turns, max_seq=args.max_seq)
                adeg.append(dg)
            except Exception as e:
                print(f"    ERR: {e}"); traceback.print_exc()

        if ph in ("words", "all") and si < 5:
            print(f"  [WORDS]")
            try:
                w = m_words(hm, sc)
                aw.append(w)
                ic = w["interf"]
                print(f"    full={ic['full']:.4f} sum={ic['sum_w']:.4f} r={ic['ratio']:.2f}")
            except Exception as e:
                print(f"    ERR: {e}"); traceback.print_exc()

    # SUMMARY
    print(f"\n\n{'='*72}")
    print(f"SUMMARY — {args.model}")
    print(f"{'='*72}")

    if ad:
        print(f"\n--- DISTRIBUTIONS ---")
        for pn in ["handled_vs_baseline", "scrambled_vs_baseline", "handled_vs_scrambled"]:
            ks = [d["comp"][pn]["kl"] for d in ad if pn in d["comp"]]
            if ks:
                ci = bci(ks)
                print(f"  {pn}: KL={ci['mean']:.4f} [{ci['lo']:.4f},{ci['hi']:.4f}] n={ci['n']}")

    if adla:
        print(f"\n--- DLA ---")
        for pn in ["handled_vs_baseline"]:
            fs = [d["dla"][pn]["af"] for d in adla if pn in d.get("dla", {})]
            if fs:
                ci = bci(fs)
                print(f"  {pn} attn: {ci['mean']:.1%} [{ci['lo']:.1%},{ci['hi']:.1%}]")

    if ac:
        print(f"\n--- CAUSAL ---")
        ar = [c["cum"]["a"] for c in ac if not c.get("skip")]
        mr = [c["cum"]["m"] for c in ac if not c.get("skip")]
        if ar:
            ca_ci = bci(ar); cm_ci = bci(mr)
            print(f"  cum attn: {ca_ci['mean']:+.4f} [{ca_ci['lo']:+.4f},{ca_ci['hi']:+.4f}]")
            print(f"  cum mlp:  {cm_ci['mean']:+.4f} [{cm_ci['lo']:+.4f},{cm_ci['hi']:+.4f}]")

    if adla and ac:
        print(f"\n--- DLA vs CAUSAL ---")
        df = [d["dla"]["handled_vs_baseline"]["af"] for d in adla if "handled_vs_baseline" in d.get("dla", {})]
        cr = [c["cum"]["a"] for c in ac if not c.get("skip")]
        if df and cr:
            dm, cm = np.mean(df), np.mean(cr)
            print(f"  DLA: attn={dm:.1%}   Causal: attn_recovery={cm:.1%}")
            if abs(dm - cm) > .15:
                print(f"  >>> DISAGREE by {abs(dm-cm):.0%}")
            else:
                print(f"  >>> Agree within {abs(dm-cm):.0%}")

    if aw:
        print(f"\n--- WORD TRACE ---")
        for w in aw:
            ic = w["interf"]
            tag = "CONSTRUCTIVE" if ic["constructive"] else "DESTRUCTIVE" if ic["destructive"] else "neutral"
            print(f"  {w['scenario']}: r={ic['ratio']:.2f} {tag}")

    # Save
    DATA_DIR.mkdir(exist_ok=True)
    tag = args.model.replace("/", "_")
    out = DATA_DIR / f"session_h_{tag}.json"
    results = {
        "model": args.model, "n_layers": hm.n_layers, "n_heads": hm.n_heads,
        "d_model": hm.d_model, "n_scenarios": len(scenarios),
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
        "backend": "HuggingFace + native PyTorch hooks",
        "beh": ab, "dist": ad, "dla": adla, "causal": ac, "deg": adeg, "words": aw,
    }
    with open(out, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved: {out}")
    print(f"Total: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
