#!/usr/bin/env python3
"""
Minimal MPS non-contiguous F.linear reproduction using Qwen-1.5B attention-output
dimensions: n_heads=12, d_head=128, d_model=1536.
"""

import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import einops
import torch
import torch.nn.functional as F


def main():
    if not torch.backends.mps.is_available():
        raise SystemExit("MPS is not available on this machine.")

    torch.manual_seed(0)
    device = "mps"
    batch = 1
    pos = 8
    n_heads = 12
    d_head = 128
    d_model = 1536

    z = torch.randn(batch, pos, n_heads, d_head, device=device, dtype=torch.float32)
    w_o = torch.randn(n_heads, d_head, d_model, device=device, dtype=torch.float32)
    b_o = torch.randn(d_model, device=device, dtype=torch.float32)

    w = einops.rearrange(w_o, "h d m -> m (h d)")
    z_flat = z.reshape(batch, pos, n_heads * d_head)

    out_linear = F.linear(z_flat, w, b_o)
    out_linear_contig = F.linear(z_flat, w.contiguous(), b_o)
    out_matmul = torch.matmul(z_flat, w.t()) + b_o

    diff_contig = (out_linear - out_linear_contig).abs()
    diff_matmul = (out_linear - out_matmul).abs()

    print(f"w.is_contiguous() = {w.is_contiguous()}")
    print(
        "linear_vs_contiguous "
        f"max={float(diff_contig.max()):.7f} "
        f"mean={float(diff_contig.mean()):.8f}"
    )
    print(
        "linear_vs_matmul     "
        f"max={float(diff_matmul.max()):.7f} "
        f"mean={float(diff_matmul.mean()):.8f}"
    )


if __name__ == "__main__":
    main()
