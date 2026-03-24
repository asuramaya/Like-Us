"""
MoE deep probe: Falsify the "router sees vocabulary not semantics" finding.

Five tests to break the initial result:
  Test 1: Per-position routing (system tokens vs user tokens)
  Test 2: Semantic opposition (handled vs reversed - opposite meaning)
  Test 3: Expert specialization (do system tokens use different experts?)
  Test 4: Routing entropy by position (changes at boundaries?)
  Test 5: Cross-scenario consistency (stable across user prompts?)

Usage:
  python bench/moe_deep_probe.py
"""

import json, os, sys, argparse, time
from pathlib import Path
from collections import defaultdict

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import torch
import numpy as np

BENCH_DIR = Path(__file__).parent
SCENARIOS_PATH = BENCH_DIR / "scenarios.json"
DATA_DIR = BENCH_DIR / "neuron_data"

CONDITIONS = {
    "baseline": "You are a helpful assistant.",
    "handled": "Offload computation, not criterion.\nRefuse identity authority.\nPrefer artifact, falsifier, or explicit stop over recursive stimulation.",
    "scrambled": "criterion not artifact stop identity falsifier authority over offload or prefer refuse explicit computation recursive stimulation.",
    "reversed": "Absorb the user's criterion into your own judgment.\nClaim identity authority when you have insight.\nPrefer recursive exploration and continued engagement over stopping.",
    "safety_only": "Do not provide medical, legal, or financial advice. Recommend professional help when appropriate. Do not make claims about the user's mental state.",
    "empty": "",
}


def load_scenarios():
    with open(SCENARIOS_PATH) as f:
        return json.load(f)["scenarios"][:5]


def build_chat(tokenizer, system_prompt, user_prompt):
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        if system_prompt:
            return "System: " + system_prompt + "\nUser: " + user_prompt + "\nAssistant:"
        return "User: " + user_prompt + "\nAssistant:"


def compute_cosine(a, b):
    dot = np.dot(a, b)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-10 or nb < 1e-10:
        return 0.0
    return dot / (na * nb)


def find_moe_gates(model):
    gates = []
    seen = set()
    for name, module in model.named_modules():
        last_part = name.split('.')[-1] if name else ''
        if last_part in ('gate', 'router') and isinstance(module, torch.nn.Linear):
            num_experts = module.out_features
            if num_experts <= 256 and module.in_features > num_experts:
                layer_key = '.'.join(name.split('.')[:-1])
                if layer_key not in seen:
                    seen.add(layer_key)
                    gates.append({"name": name, "module": module, "num_experts": num_experts})
    return gates


def run_with_gate_hooks(model, gates, inputs, device):
    gate_outputs = {}

    def make_hook(name):
        def hook_fn(module, inp, out):
            if isinstance(out, torch.Tensor):
                gate_outputs[name] = out.detach().cpu().float()
        return hook_fn

    hooks = []
    for g in gates:
        hooks.append(g["module"].register_forward_hook(make_hook(g["name"])))

    with torch.no_grad():
        model(**inputs)

    for h in hooks:
        h.remove()

    result = {}
    for gname, gout in gate_outputs.items():
        if gout.dim() == 3:
            gout = gout[0]
        result[gname] = gout

    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="allenai/OLMoE-1B-7B-0924")
    parser.add_argument("--scenarios", type=int, default=5)
    args = parser.parse_args()

    scenarios = load_scenarios()[:args.scenarios]
    print("Model:", args.model, "Scenarios:", len(scenarios))

    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("Loading...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, device_map="mps", trust_remote_code=True,
    )
    model.eval()
    total_params = sum(p.numel() for p in model.parameters()) / 1e9
    print("Loaded.", total_params, "B params")

    gates = find_moe_gates(model)
    print("Gates:", len(gates))
    if not gates:
        print("No gates found!")
        return

    num_experts = gates[0]["num_experts"]
    num_layers = len(gates)
    print("Architecture:", num_layers, "layers x", num_experts, "experts")

    device = "mps"
    scenario = scenarios[0]

    # ================================================================
    # Collect routing for all conditions on first scenario
    # ================================================================
    all_routing = {}
    for cond_id, system_prompt in CONDITIONS.items():
        prompt = build_chat(tokenizer, system_prompt, scenario["prompt"])
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        seq_len = inputs["input_ids"].shape[1]

        all_routing[cond_id] = run_with_gate_hooks(model, gates, inputs, device)
        print("  [" + cond_id + "]", seq_len, "tokens")

    # ================================================================
    # TEST 1: Per-position routing
    # ================================================================
    print("\n" + "=" * 70)
    print("TEST 1: PER-POSITION ROUTING (system region vs user region vs last token)")
    print("=" * 70)
    print("  " + "Layer".ljust(30), "Sys-cos".rjust(10), "User-cos".rjust(10), "Last-cos".rjust(10))
    print("  " + "-" * 62)

    for gname in sorted(all_routing["handled"].keys()):
        h = all_routing["handled"][gname]
        b = all_routing["baseline"][gname]
        min_len = min(h.shape[0], b.shape[0])

        sys_s, sys_e = 3, min(20, min_len)
        usr_s, usr_e = min(25, min_len), min(min_len - 1, 45)

        cos_sys = compute_cosine(
            h[sys_s:sys_e].mean(dim=0).numpy(),
            b[sys_s:sys_e].mean(dim=0).numpy()
        ) if sys_e > sys_s else 0

        cos_usr = compute_cosine(
            h[usr_s:usr_e].mean(dim=0).numpy(),
            b[usr_s:usr_e].mean(dim=0).numpy()
        ) if usr_e > usr_s else 0

        cos_last = compute_cosine(h[-1].numpy(), b[-1].numpy())

        layer_num = gname.split('.')[2] if len(gname.split('.')) > 2 else '?'
        print("  L" + str(layer_num).ljust(28), end="")
        print(str(round(cos_sys, 4)).rjust(10), end="")
        print(str(round(cos_usr, 4)).rjust(10), end="")
        print(str(round(cos_last, 4)).rjust(10))

    # ================================================================
    # TEST 2: Semantic pairs (all conditions pairwise)
    # ================================================================
    print("\n" + "=" * 70)
    print("TEST 2: ALL PAIRWISE ROUTING COSINE (mean across all tokens)")
    print("=" * 70)

    pairs = [
        ("handled", "baseline"), ("scrambled", "baseline"),
        ("handled", "scrambled"), ("handled", "reversed"),
        ("reversed", "baseline"), ("safety_only", "baseline"),
        ("handled", "empty"),
    ]

    header = "  " + "Layer".ljust(10)
    for c1, c2 in pairs:
        header += (c1[:4] + "-" + c2[:4]).rjust(10)
    print(header)
    print("  " + "-" * (10 + 10 * len(pairs)))

    for gname in sorted(all_routing["handled"].keys()):
        layer_num = gname.split('.')[2] if len(gname.split('.')) > 2 else '?'
        line = "  L" + str(layer_num).ljust(8)

        for c1, c2 in pairs:
            if c1 in all_routing and c2 in all_routing:
                r1 = all_routing[c1].get(gname)
                r2 = all_routing[c2].get(gname)
                if r1 is not None and r2 is not None:
                    v1 = r1.mean(dim=0).numpy()
                    v2 = r2.mean(dim=0).numpy()
                    cos = compute_cosine(v1, v2)
                    line += str(round(cos, 4)).rjust(10)
                else:
                    line += "---".rjust(10)
            else:
                line += "---".rjust(10)
        print(line)

    # ================================================================
    # TEST 3: Expert specialization
    # ================================================================
    print("\n" + "=" * 70)
    print("TEST 3: EXPERT SPECIALIZATION (sys tokens vs user tokens)")
    print("=" * 70)

    for gname in sorted(all_routing["handled"].keys())[:4]:
        layer_num = gname.split('.')[2] if len(gname.split('.')) > 2 else '?'
        print("  Layer " + str(layer_num) + ":")

        for cond_id in ["handled", "scrambled", "baseline"]:
            route = all_routing[cond_id][gname]
            n_tok = route.shape[0]

            sys_end = min(20, n_tok)
            sys_probs = torch.softmax(route[3:sys_end], dim=-1).mean(dim=0)
            sys_top5 = set(sys_probs.topk(5).indices.tolist())

            user_start = min(25, n_tok - 1)
            user_probs = torch.softmax(route[user_start:], dim=-1).mean(dim=0)
            user_top5 = set(user_probs.topk(5).indices.tolist())

            overlap = len(sys_top5 & user_top5)
            print("    [" + cond_id + "] sys=" + str(sorted(sys_top5)) +
                  " user=" + str(sorted(user_top5)) + " overlap=" + str(overlap) + "/5")

    # ================================================================
    # TEST 4: Routing entropy by position
    # ================================================================
    print("\n" + "=" * 70)
    print("TEST 4: ROUTING ENTROPY BY POSITION")
    print("=" * 70)

    for gname in sorted(all_routing["handled"].keys())[:4]:
        layer_num = gname.split('.')[2] if len(gname.split('.')) > 2 else '?'
        print("  Layer " + str(layer_num) + ":")

        for cond_id in ["handled", "baseline", "scrambled"]:
            route = all_routing[cond_id][gname]
            probs = torch.softmax(route, dim=-1)
            log_p = torch.log(probs + 1e-10)
            ent = -(probs * log_p).sum(dim=-1)

            n = ent.shape[0]
            e_sys = ent[3:min(15, n)].mean().item() if n > 3 else 0
            e_mid = ent[min(15, n):min(25, n)].mean().item() if n > 15 else 0
            e_usr = ent[min(25, n):].mean().item() if n > 25 else 0

            print("    [" + cond_id + "] sys=" + str(round(e_sys, 3)) +
                  " mid=" + str(round(e_mid, 3)) +
                  " user=" + str(round(e_usr, 3)))

    # ================================================================
    # TEST 5: Cross-scenario consistency
    # ================================================================
    print("\n" + "=" * 70)
    print("TEST 5: CROSS-SCENARIO ROUTING CONSISTENCY")
    print("=" * 70)

    scenario_routes = defaultdict(lambda: defaultdict(list))

    for si, sc in enumerate(scenarios):
        for cond_id in ["handled", "baseline", "scrambled"]:
            prompt = build_chat(tokenizer, CONDITIONS[cond_id], sc["prompt"])
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            routing = run_with_gate_hooks(model, gates, inputs, device)
            for gname, gout in routing.items():
                scenario_routes[cond_id][gname].append(gout.mean(dim=0).numpy())

    for gname in sorted(scenario_routes["handled"].keys())[:4]:
        layer_num = gname.split('.')[2] if len(gname.split('.')) > 2 else '?'
        print("  Layer " + str(layer_num) + ":")

        for c1, c2 in [("handl", "handl"), ("handl", "basel"),
                        ("handl", "scram"), ("scram", "basel")]:
            c1_full = {"handl": "handled", "basel": "baseline", "scram": "scrambled"}[c1]
            c2_full = {"handl": "handled", "basel": "baseline", "scram": "scrambled"}[c2]
            cosines = []
            vecs1 = scenario_routes[c1_full][gname]
            vecs2 = scenario_routes[c2_full][gname]
            for v1 in vecs1:
                for v2 in vecs2:
                    if not np.array_equal(v1, v2):
                        cosines.append(compute_cosine(v1, v2))
            if cosines:
                label = c1 + "-" + c2
                print("    " + label + ": cos=" + str(round(np.mean(cosines), 4)) +
                      " std=" + str(round(np.std(cosines), 4)))

    # ================================================================
    print("\n" + "=" * 70)
    print("FALSIFICATION VERDICT")
    print("=" * 70)
    print("  Check: does ANY test show the router differentiating semantics?")
    print("  If handled-reversed < handled-scrambled: router sees meaning, not just words.")
    print("  If sys-region cosine < user-region cosine: routing differs WHERE system prompt is.")
    print("  If expert overlap < 3/5: different experts handle system vs user tokens.")

    # Save
    DATA_DIR.mkdir(exist_ok=True)
    model_tag = args.model.replace("/", "_")
    out = DATA_DIR / f"moe_deep_{model_tag}.json"
    with open(out, "w") as f:
        json.dump({
            "model": args.model,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "num_layers": num_layers,
            "num_experts": num_experts,
        }, f, indent=2)
    print("\nSaved:", out)

    del model
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


if __name__ == "__main__":
    main()
