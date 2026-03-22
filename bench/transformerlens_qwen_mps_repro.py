#!/usr/bin/env python3
"""
Reproduce the TransformerLens Qwen MPS attention-output mismatch.

This script compares Hugging Face and TransformerLens next-token outputs for a
single chat-formatted prompt and can optionally dump layer-0 projection diffs.
"""

import argparse
import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import einops
import torch
import torch.nn.functional as F
from transformer_lens import HookedTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_MESSAGES = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello, how are you?"},
]


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--dtype", default="float32", choices=["float16", "float32"])
    ap.add_argument("--device", default="mps")
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--compare-layer0", action="store_true")
    return ap.parse_args()


def resolve_dtype(name: str) -> torch.dtype:
    return getattr(torch, name)


def topk_strings(tokenizer, probs: torch.Tensor, k: int):
    vals, idx = probs.topk(k)
    return [(tokenizer.decode([int(i)]), float(v)) for v, i in zip(vals, idx)]


def load_prompt(tokenizer) -> str:
    return tokenizer.apply_chat_template(
        DEFAULT_MESSAGES,
        tokenize=False,
        add_generation_prompt=True,
    )


def run_hf(model_name: str, device: str, dtype: torch.dtype, topk: int):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype).to(device)
    prompt = load_prompt(tokenizer)
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        probs = torch.softmax(model(input_ids=input_ids).logits[0, -1].float(), dim=-1)
    return prompt, topk_strings(tokenizer, probs, topk)


def run_tl(model_name: str, device: str, dtype: torch.dtype, prompt: str, topk: int, *, use_attn_result: bool):
    model = HookedTransformer.from_pretrained_no_processing(model_name, device=device, dtype=dtype)
    if use_attn_result:
        model.set_use_attn_result(True)
    tokens = model.to_tokens(prompt, prepend_bos=False)
    with torch.no_grad():
        probs = torch.softmax(model(tokens)[0, -1].float(), dim=-1)
    return topk_strings(model.tokenizer, probs, topk)


def compare_layer0(model_name: str):
    hooks = ["blocks.0.attn.hook_z", "blocks.0.hook_attn_out"]

    model_cpu = HookedTransformer.from_pretrained_no_processing(
        model_name,
        device="cpu",
        dtype=torch.float32,
    )
    prompt = load_prompt(model_cpu.tokenizer)
    tokens = model_cpu.to_tokens(prompt, prepend_bos=False)
    with torch.no_grad():
        _, cache_cpu = model_cpu.run_with_cache(tokens, names_filter=lambda name: name in hooks)

    model_mps = HookedTransformer.from_pretrained_no_processing(
        model_name,
        device="mps",
        dtype=torch.float32,
    )
    with torch.no_grad():
        _, cache_mps = model_mps.run_with_cache(
            tokens.to("mps"),
            names_filter=lambda name: name in hooks,
        )

    z_cpu = cache_cpu["blocks.0.attn.hook_z"]
    out_cpu = cache_cpu["blocks.0.hook_attn_out"]
    z_mps = cache_mps["blocks.0.attn.hook_z"]
    out_mps = cache_mps["blocks.0.hook_attn_out"]
    attn = model_mps.blocks[0].attn

    w = einops.rearrange(attn.W_O, "head_index d_head d_model -> d_model (head_index d_head)")
    z_flat = z_mps.reshape(z_mps.shape[0], z_mps.shape[1], attn.cfg.d_head * attn.cfg.n_heads)

    with torch.no_grad():
        out_linear = F.linear(z_flat, w, attn.b_O)
        out_linear_contig = F.linear(z_flat, w.contiguous(), attn.b_O)
        out_einsum = torch.einsum("bphd,hdm->bpm", z_mps, attn.W_O) + attn.b_O
        w5 = einops.rearrange(attn.W_O, "h d m -> 1 1 h d m")
        z5 = einops.rearrange(z_mps, "b p h d -> b p h d 1")
        out_broadcast = (z5 * w5).sum(-2).sum(-2) + attn.b_O

    print("\nLayer-0 projection diffs vs CPU:")
    for label, tensor in [
        ("mps_cache", out_mps),
        ("mps_linear", out_linear),
        ("mps_linear_contig", out_linear_contig),
        ("mps_einsum", out_einsum),
        ("mps_broadcast", out_broadcast),
    ]:
        diff = (tensor.float().cpu() - out_cpu.float().cpu()).abs()
        print(f"{label:18s} max={float(diff.max()):.7f} mean={float(diff.mean()):.8f}")


def main():
    args = parse_args()
    dtype = resolve_dtype(args.dtype)

    prompt, hf_top = run_hf(args.model, args.device, dtype, args.topk)
    tl_top = run_tl(args.model, args.device, dtype, prompt, args.topk, use_attn_result=False)
    tl_fixed_top = run_tl(args.model, args.device, dtype, prompt, args.topk, use_attn_result=True)

    print("HF         ", hf_top)
    print("TL default ", tl_top)
    print("TL explicit", tl_fixed_top)

    if args.compare_layer0:
        compare_layer0(args.model)


if __name__ == "__main__":
    main()
