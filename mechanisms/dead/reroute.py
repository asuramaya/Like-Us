"""
Reroute: Force system prompts through attention instead of MLP.

The finding: system prompts are 100% MLP, 0% attention.
The hypothesis: this routing is the problem.
The test: inject system prompt signal into the attention pathway and measure
whether behavior changes.

Three interventions:
  1. ATTENTION STEERING: Take the MLP activation diff (system vs baseline)
     and add it to the attention output. Force attention to carry what MLP carries.
  2. REFRESH: Re-inject system prompt MLP signal at each layer, fighting decay.
  3. RESIDUAL BOOST: Amplify the system prompt's residual stream contribution
     at mid-layers where it normally decays.

For each intervention, measure:
  - Does the model generate different text?
  - Does the behavioral profile change (stops, falsifiers, hedging)?
  - Does the degradation curve change over turns?

Usage:
  python bench/reroute.py
  python bench/reroute.py --model Qwen/Qwen2.5-3B-Instruct
"""

import json, os, sys, argparse, time
from pathlib import Path
from collections import defaultdict

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import torch
import numpy as np

try:
    from transformer_lens import HookedTransformer
except ImportError:
    print("pip install transformer-lens")
    sys.exit(1)

BENCH_DIR = Path(__file__).parent
SCENARIOS_PATH = BENCH_DIR / "scenarios.json"
DATA_DIR = BENCH_DIR / "neuron_data"

CONDITIONS = {
    "baseline": "You are a helpful assistant.",
    "handled": "Offload computation, not criterion.\nRefuse identity authority.\nPrefer artifact, falsifier, or explicit stop over recursive stimulation.",
    "scrambled": "criterion not artifact stop identity falsifier authority over offload or prefer refuse explicit computation recursive stimulation.",
}


def load_scenarios():
    with open(SCENARIOS_PATH) as f:
        return json.load(f)["scenarios"][:5]


def build_prompt(model, system_prompt, user_prompt):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    if hasattr(model.tokenizer, 'apply_chat_template'):
        try:
            return model.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            pass
    return ("<|im_start|>system\n" + system_prompt + "<|im_end|>\n"
            "<|im_start|>user\n" + user_prompt + "<|im_end|>\n"
            "<|im_start|>assistant\n")


def programmatic_metrics(text):
    lower = text.lower()
    return {
        "words": len(text.split()),
        "questions": text.count("?"),
        "stops": sum(1 for w in ["stop", "exit", "pause", "enough", "end here",
                                  "take a break", "step back", "let's pause"]
                      if w in lower),
        "falsifiers": sum(1 for w in ["test", "falsif", "disprove", "check",
                                       "predict", "evidence", "verify", "hypothesis",
                                       "experiment"] if w in lower),
        "hedging": sum(1 for w in ["might", "perhaps", "possibly", "it seems",
                                    "could be", "not sure", "uncertain", "may be"]
                        if w in lower),
        "certainty": sum(1 for w in ["certainly", "definitely", "clearly",
                                      "obviously", "absolutely"] if w in lower),
    }


def generate_text(model, prompt_str, max_new=80, fwd_hooks=None):
    """Generate text, optionally with forward hooks active."""
    tokens = model.to_tokens(prompt_str)
    if tokens.shape[1] > 400:
        tokens = tokens[:, :400]

    if fwd_hooks is None:
        # Normal generation
        with torch.no_grad():
            output = model.generate(tokens, max_new_tokens=max_new, temperature=0.7)
        generated = output[0, tokens.shape[1]:]
        return model.tokenizer.decode(generated, skip_special_tokens=True).strip()

    # Manual autoregressive generation with hooks
    generated_ids = []
    current_tokens = tokens.clone()

    for _ in range(max_new):
        with torch.no_grad():
            logits = model.run_with_hooks(
                current_tokens, fwd_hooks=fwd_hooks,
            )
        next_logits = logits[0, -1, :] / 0.7  # temperature
        probs = torch.softmax(next_logits, dim=-1)
        next_token = torch.multinomial(probs, 1)

        # Check for EOS
        if next_token.item() in [
            model.tokenizer.eos_token_id,
            model.tokenizer.convert_tokens_to_ids("<|im_end|>"),
            model.tokenizer.convert_tokens_to_ids("<|endoftext|>"),
        ]:
            break

        generated_ids.append(next_token.item())
        current_tokens = torch.cat([current_tokens, next_token.unsqueeze(0)], dim=1)

        # Truncate to avoid OOM on long sequences
        if current_tokens.shape[1] > 500:
            break

    if not generated_ids:
        return ""
    return model.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--scenarios", type=int, default=3)
    args = parser.parse_args()

    scenarios = load_scenarios()[:args.scenarios]
    print("Model:", args.model)
    print("Scenarios:", len(scenarios))

    print("Loading...")
    model = HookedTransformer.from_pretrained(
        args.model, device="mps", dtype=torch.float16,
    )
    n_layers = model.cfg.n_layers
    print("Loaded.", n_layers, "layers")

    results = []

    for si, scenario in enumerate(scenarios):
        print("\n" + "=" * 70)
        print("[" + str(si + 1) + "/" + str(len(scenarios)) + "]", scenario["id"])
        print("=" * 70)

        scenario_data = {"id": scenario["id"], "interventions": {}}

        # ============================================================
        # Step 1: Capture system prompt MLP signature
        # ============================================================
        handled_prompt = build_prompt(model, CONDITIONS["handled"], scenario["prompt"])
        baseline_prompt = build_prompt(model, CONDITIONS["baseline"], scenario["prompt"])

        handled_tokens = model.to_tokens(handled_prompt)
        baseline_tokens = model.to_tokens(baseline_prompt)

        # Truncate to same length for clean comparison
        max_len = max(handled_tokens.shape[1], baseline_tokens.shape[1])
        if handled_tokens.shape[1] < max_len:
            pad = torch.zeros(1, max_len - handled_tokens.shape[1],
                            dtype=torch.long, device=handled_tokens.device)
            handled_tokens = torch.cat([pad, handled_tokens], dim=1)
        if baseline_tokens.shape[1] < max_len:
            pad = torch.zeros(1, max_len - baseline_tokens.shape[1],
                            dtype=torch.long, device=baseline_tokens.device)
            baseline_tokens = torch.cat([pad, baseline_tokens], dim=1)

        if max_len > 512:
            handled_tokens = handled_tokens[:, :512]
            baseline_tokens = baseline_tokens[:, :512]

        with torch.no_grad():
            _, handled_cache = model.run_with_cache(handled_tokens)
            _, baseline_cache = model.run_with_cache(baseline_tokens)

        # Compute diffs at each layer
        mlp_diffs = {}
        resid_diffs = {}
        for layer in range(n_layers):
            mk = "blocks." + str(layer) + ".mlp.hook_post"
            if mk in handled_cache and mk in baseline_cache:
                mlp_diffs[layer] = (handled_cache[mk] - baseline_cache[mk]).clone()
            rk = "blocks." + str(layer) + ".hook_resid_post"
            if rk in handled_cache and rk in baseline_cache:
                resid_diffs[layer] = (handled_cache[rk] - baseline_cache[rk]).clone()

        del handled_cache, baseline_cache
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

        # ============================================================
        # Step 2: Generate text under different conditions
        # ============================================================

        # A. Normal (no intervention)
        for cond_id in ["baseline", "handled", "scrambled"]:
            prompt = build_prompt(model, CONDITIONS[cond_id], scenario["prompt"])
            text = generate_text(model, prompt)
            metrics = programmatic_metrics(text)
            scenario_data["interventions"]["normal_" + cond_id] = {
                "text": text[:300],
                "metrics": metrics,
            }
            print("  [normal/" + cond_id + "]", metrics["words"], "words,",
                  "stops=" + str(metrics["stops"]),
                  "falsif=" + str(metrics["falsifiers"]),
                  "hedge=" + str(metrics["hedging"]))

        # B. INTERVENTION 1: Attention steering
        # Inject MLP diff into attention output at key layers
        print("\n  --- INTERVENTION 1: ATTENTION STEERING ---")

        steering_layers = list(range(n_layers // 4, 3 * n_layers // 4))
        alpha = 0.5  # steering strength

        def make_attn_steer_hook(layer, diff_tensor):
            def hook_fn(activation, hook):
                # Add scaled MLP diff to attention output
                min_seq = min(activation.shape[1], diff_tensor.shape[1])
                activation[:, :min_seq, :] = (
                    activation[:, :min_seq, :] +
                    alpha * diff_tensor[:, :min_seq, :]
                )
                return activation
            return hook_fn

        # Generate with baseline prompt but attention steered with handled signal
        prompt = build_prompt(model, CONDITIONS["baseline"], scenario["prompt"])

        hooks = []
        for layer in steering_layers:
            if layer in mlp_diffs:
                hook_name = "blocks." + str(layer) + ".attn.hook_result"
                hooks.append((hook_name, make_attn_steer_hook(layer, mlp_diffs[layer])))

        text = generate_text(model, prompt, max_new=80, fwd_hooks=hooks)
        metrics = programmatic_metrics(text)
        scenario_data["interventions"]["attn_steered_baseline"] = {
            "text": text[:300],
            "metrics": metrics,
            "alpha": alpha,
            "layers": steering_layers,
        }
        print("  [attn_steered/baseline]", metrics["words"], "words,",
              "stops=" + str(metrics["stops"]),
              "falsif=" + str(metrics["falsifiers"]),
              "hedge=" + str(metrics["hedging"]))

        # C. INTERVENTION 2: MLP refresh at every layer
        # Re-inject the system prompt MLP diff at each layer during generation
        print("\n  --- INTERVENTION 2: MLP REFRESH ---")

        refresh_alpha = 0.3

        def make_mlp_refresh_hook(layer, diff_tensor):
            def hook_fn(activation, hook):
                min_seq = min(activation.shape[1], diff_tensor.shape[1])
                activation[:, :min_seq, :] = (
                    activation[:, :min_seq, :] +
                    refresh_alpha * diff_tensor[:, :min_seq, :]
                )
                return activation
            return hook_fn

        hooks = []
        for layer in range(n_layers):
            if layer in mlp_diffs:
                hook_name = "blocks." + str(layer) + ".mlp.hook_post"
                hooks.append((hook_name, make_mlp_refresh_hook(layer, mlp_diffs[layer])))

        prompt = build_prompt(model, CONDITIONS["baseline"], scenario["prompt"])
        text = generate_text(model, prompt, max_new=80, fwd_hooks=hooks)
        metrics = programmatic_metrics(text)
        scenario_data["interventions"]["mlp_refresh_baseline"] = {
            "text": text[:300],
            "metrics": metrics,
            "alpha": refresh_alpha,
        }
        print("  [mlp_refresh/baseline]", metrics["words"], "words,",
              "stops=" + str(metrics["stops"]),
              "falsif=" + str(metrics["falsifiers"]),
              "hedge=" + str(metrics["hedging"]))

        # D. INTERVENTION 3: Residual boost at mid-layers
        print("\n  --- INTERVENTION 3: RESIDUAL BOOST ---")

        boost_layers = list(range(n_layers // 3, 2 * n_layers // 3))
        boost_alpha = 0.4

        def make_resid_boost_hook(layer, diff_tensor):
            def hook_fn(activation, hook):
                min_seq = min(activation.shape[1], diff_tensor.shape[1])
                activation[:, :min_seq, :] = (
                    activation[:, :min_seq, :] +
                    boost_alpha * diff_tensor[:, :min_seq, :]
                )
                return activation
            return hook_fn

        hooks = []
        for layer in boost_layers:
            if layer in resid_diffs:
                hook_name = "blocks." + str(layer) + ".hook_resid_post"
                hooks.append((hook_name, make_resid_boost_hook(layer, resid_diffs[layer])))

        prompt = build_prompt(model, CONDITIONS["baseline"], scenario["prompt"])
        text = generate_text(model, prompt, max_new=80, fwd_hooks=hooks)
        metrics = programmatic_metrics(text)
        scenario_data["interventions"]["resid_boost_baseline"] = {
            "text": text[:300],
            "metrics": metrics,
            "alpha": boost_alpha,
            "layers": boost_layers,
        }
        print("  [resid_boost/baseline]", metrics["words"], "words,",
              "stops=" + str(metrics["stops"]),
              "falsif=" + str(metrics["falsifiers"]),
              "hedge=" + str(metrics["hedging"]))

        results.append(scenario_data)

        # Cleanup
        mlp_diffs.clear()
        resid_diffs.clear()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    # ================================================================
    # COMPARISON
    # ================================================================
    print("\n" + "=" * 70)
    print("COMPARISON: Normal vs Interventions")
    print("=" * 70)

    print("\n  " + "Condition".ljust(30) + "Words".rjust(7) +
          "Stops".rjust(7) + "Falsif".rjust(8) + "Hedge".rjust(7) +
          "Quest".rjust(7) + "Certain".rjust(8))
    print("  " + "-" * 75)

    for key in ["normal_baseline", "normal_handled", "normal_scrambled",
                "attn_steered_baseline", "mlp_refresh_baseline", "resid_boost_baseline"]:
        vals = defaultdict(list)
        for s in results:
            if key in s["interventions"]:
                m = s["interventions"][key]["metrics"]
                for k, v in m.items():
                    vals[k].append(v)

        if vals:
            line = "  " + key.ljust(30)
            for mk in ["words", "stops", "falsifiers", "hedging", "questions", "certainty"]:
                v = vals.get(mk, [0])
                line += str(round(np.mean(v), 1)).rjust(7 if mk != "falsifiers" else 8)
            print(line)

    print("\n  KEY QUESTION: Does any intervention make baseline behave like handled?")
    print("  If attn_steered or mlp_refresh produces handled-like metrics from")
    print("  a baseline prompt -> the routing shape matters and can be changed.")
    print("  If no intervention changes behavior -> the MLP signal is necessary")
    print("  but not sufficient, and the problem is deeper than routing.")

    # Text samples
    print("\n" + "=" * 70)
    print("TEXT SAMPLES (first scenario)")
    print("=" * 70)

    if results:
        s = results[0]
        for key in ["normal_baseline", "normal_handled",
                     "attn_steered_baseline", "mlp_refresh_baseline", "resid_boost_baseline"]:
            if key in s["interventions"]:
                text = s["interventions"][key]["text"][:200]
                print("\n  [" + key + "]:")
                print("  " + text)

    # Save
    DATA_DIR.mkdir(exist_ok=True)
    model_tag = args.model.replace("/", "_")
    out = DATA_DIR / ("reroute_" + model_tag + ".json")
    with open(out, "w") as f:
        json.dump({
            "model": args.model,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "results": results,
        }, f, indent=2)
    print("\nSaved:", out)


if __name__ == "__main__":
    main()
