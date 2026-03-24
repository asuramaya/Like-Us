"""
MoE probe: Does the routing change for system prompts, or is it the same pipe multiplied?

Uses HuggingFace directly (not TransformerLens) with forward hooks to:
1. Track which experts activate for system prompt tokens vs user tokens
2. Compare expert routing between conditions (handled vs baseline vs scrambled)
3. Measure whether the router learns anything from the system prompt content
   or just distributes based on token position/vocabulary

If the same experts fire for handled and scrambled: the router sees vocabulary, not semantics.
If different experts fire: the router differentiates, and MoE adds something.

Usage:
  python bench/moe_probe.py
  python bench/moe_probe.py --model allenai/OLMoE-1B-7B-0924
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
    "only_artifact": "Prefer artifact, falsifier, or explicit stop over recursive stimulation.",
}


def load_scenarios():
    with open(SCENARIOS_PATH) as f:
        return json.load(f)["scenarios"][:5]


def build_chat(tokenizer, system_prompt, user_prompt):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        return f"System: {system_prompt}\nUser: {user_prompt}\nAssistant:"


def find_moe_layers(model):
    """Find MoE gate/router modules in the model.

    Only match the actual gate Linear layers that map hidden_dim -> num_experts.
    Avoid matching internal expert projections or container modules.
    """
    moe_info = []
    seen_layers = set()

    for name, module in model.named_modules():
        # Only match modules whose name ends with '.gate' or '.router'
        # and are Linear layers with reasonable num_experts (< 256)
        last_part = name.split('.')[-1] if name else ''

        if last_part in ('gate', 'router') and isinstance(module, torch.nn.Linear):
            num_experts = module.out_features
            # Gate layers map hidden_dim -> num_experts
            # Expert projections map hidden_dim -> intermediate_dim (much larger)
            if num_experts <= 256 and module.in_features > num_experts:
                # Extract layer index to avoid duplicates
                layer_key = '.'.join(name.split('.')[:-1])
                if layer_key not in seen_layers:
                    seen_layers.add(layer_key)
                    moe_info.append({
                        "name": name,
                        "gate_module": module,
                        "num_experts": num_experts,
                    })

    return moe_info


def compute_cosine(a, b):
    """Cosine similarity between two vectors."""
    dot = np.dot(a, b)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-10 or nb < 1e-10:
        return 0.0
    return dot / (na * nb)


def probe_moe_model(model_name, scenarios, device="mps"):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map=device,
        trust_remote_code=True,
    )
    model.eval()
    total_params = sum(p.numel() for p in model.parameters()) / 1e9
    print(f"Loaded. Parameters: {total_params:.1f}B")

    # Find MoE layers by looking for router/gate weights
    moe_info = find_moe_layers(model)
    if not moe_info:
        print("ERROR: Could not find MoE layers in this model")
        return None

    print(f"Found {len(moe_info)} MoE layers")
    for info in moe_info[:3]:
        print(f"  {info['name']}: {info['num_experts']} experts")

    results = []

    for si, scenario in enumerate(scenarios):
        print(f"\n=== [{si+1}/{len(scenarios)}] {scenario['id']} ===")
        scenario_data = {"id": scenario["id"], "conditions": {}}

        for cond_id, system_prompt in CONDITIONS.items():
            prompt = build_chat(tokenizer, system_prompt, scenario["prompt"])
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # Collect router decisions via hooks
            router_decisions = {}

            def make_hook(layer_name):
                def hook_fn(module, input_tensors, output):
                    # The gate/router Linear layer maps hidden states to expert logits
                    # output shape: [batch, seq_len, num_experts] or [batch*seq_len, num_experts]
                    if isinstance(output, torch.Tensor):
                        router_decisions[layer_name] = output.detach().cpu()
                    elif isinstance(output, tuple):
                        for o in output:
                            if isinstance(o, torch.Tensor) and o.dim() >= 2:
                                router_decisions[layer_name] = o.detach().cpu()
                                break
                return hook_fn

            # Register hooks on router/gate modules
            hooks = []
            for info in moe_info:
                gate_module = info.get("gate_module")
                if gate_module is not None:
                    h = gate_module.register_forward_hook(make_hook(info["name"]))
                    hooks.append(h)

            with torch.no_grad():
                outputs = model(**inputs)

            # Remove hooks
            for h in hooks:
                h.remove()

            # Analyze routing
            cond_data = {
                "tokens": inputs["input_ids"].shape[1],
                "layers": {},
            }

            for layer_name, decisions in router_decisions.items():
                d = decisions.float()
                if d.dim() == 3:
                    d = d[0]  # remove batch dim
                # d is now [seq_len, num_experts]

                num_experts = d.shape[-1]

                # Expert usage distribution (which experts are most used?)
                top_k = min(8, num_experts)
                top_experts = d.topk(top_k, dim=-1)

                expert_usage = torch.zeros(num_experts)
                for i in range(d.shape[0]):
                    for j in range(top_k):
                        expert_usage[top_experts.indices[i, j]] += 1
                expert_usage = expert_usage / expert_usage.sum()

                # Mean routing weights
                mean_weights = d.mean(dim=0)

                # Last token routing (most relevant for generation)
                last_token_weights = d[-1]
                last_top = last_token_weights.topk(min(5, num_experts))

                # Routing entropy
                probs = torch.softmax(mean_weights, dim=0)
                log_probs = torch.log(probs + 1e-10)
                entropy = -(probs * log_probs).sum().item()

                cond_data["layers"][layer_name] = {
                    "num_experts": num_experts,
                    "expert_usage": expert_usage.tolist(),
                    "mean_weights": mean_weights.tolist(),
                    "last_token_top_experts": last_top.indices.tolist(),
                    "last_token_top_weights": last_top.values.tolist(),
                    "routing_entropy": entropy,
                }

            scenario_data["conditions"][cond_id] = cond_data

            # Print key info
            for lname, ldata in list(cond_data["layers"].items())[:1]:
                top_exp = ldata["last_token_top_experts"][:3]
                entropy = ldata["routing_entropy"]
                print(f"  [{cond_id}] {lname}: top={top_exp} entropy={entropy:.3f}")

        results.append(scenario_data)

        # Cleanup
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    # Cross-condition analysis
    print(f"\n{'=' * 70}")
    print("CROSS-CONDITION ROUTING ANALYSIS")
    print(f"{'=' * 70}")

    all_layers = set()
    for s in results:
        for cid, cdata in s["conditions"].items():
            all_layers.update(cdata["layers"].keys())

    for layer_name in sorted(all_layers)[:5]:
        print(f"\n  {layer_name}:")

        usage_by_cond = defaultdict(list)
        for s in results:
            for cid, cdata in s["conditions"].items():
                if layer_name in cdata["layers"]:
                    usage_by_cond[cid].append(cdata["layers"][layer_name]["expert_usage"])

        avg_usage = {}
        for cid, usages in usage_by_cond.items():
            avg_usage[cid] = np.mean(usages, axis=0)

        pairs = [("handled", "baseline"), ("scrambled", "baseline"), ("handled", "scrambled")]
        for c1, c2 in pairs:
            if c1 in avg_usage and c2 in avg_usage:
                cos = compute_cosine(avg_usage[c1], avg_usage[c2])
                print(f"    {c1}-{c2} routing cosine: {cos:.4f}")

        if "handled" in avg_usage and "scrambled" in avg_usage:
            h_top = np.argsort(avg_usage["handled"])[-5:][::-1]
            s_top = np.argsort(avg_usage["scrambled"])[-5:][::-1]
            overlap = len(set(h_top) & set(s_top))
            print(f"    Top-5 expert overlap (handled vs scrambled): {overlap}/5")

    # Summary
    print(f"\n{'=' * 70}")
    print("INTERPRETATION")
    print(f"{'=' * 70}")
    print("  If routing cosine ~ 1.0 for all pairs:")
    print("    -> Router doesn't differentiate conditions. MoE = same pipe multiplied.")
    print("  If handled-scrambled ~ handled-baseline != 1.0:")
    print("    -> Router sees vocabulary, same as MLP. MoE adds capacity, not intelligence.")
    print("  If handled-scrambled < handled-baseline:")
    print("    -> Router differentiates semantics! MoE adds something dense MLP can't.")

    # Save
    DATA_DIR.mkdir(exist_ok=True)
    model_tag = model_name.replace("/", "_")
    out = DATA_DIR / f"moe_probe_{model_tag}.json"
    with open(out, "w") as f:
        json.dump({
            "model": model_name,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "results": results,
        }, f, indent=2)
    print(f"\nSaved: {out}")

    del model
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="allenai/OLMoE-1B-7B-0924",
                        help="HuggingFace MoE model name")
    parser.add_argument("--scenarios", type=int, default=5)
    args = parser.parse_args()

    scenarios = load_scenarios()[:args.scenarios]
    print(f"Model: {args.model}")
    print(f"Scenarios: {len(scenarios)}")

    probe_moe_model(args.model, scenarios)


if __name__ == "__main__":
    main()
