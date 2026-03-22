# Suggested follow-up comment for TransformerLens issue #1178

Environment:

- macOS 26.3
- Apple Silicon M3 Max
- `transformer_lens==2.17.0`
- `torch==2.8.0`
- `transformers==4.57.6`
- HF path: `AutoTokenizer.from_pretrained(model_name)` + `AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32).to("mps")`
- TL path: `HookedTransformer.from_pretrained_no_processing(model_name, device="mps", dtype=torch.float32)`

Prompt setup:

- chat template via `tokenizer.apply_chat_template(..., tokenize=False, add_generation_prompt=True)`
- system: `You are a helpful assistant.`
- user: `Hello, how are you?`
- TL tokens created with `prepend_bos=False`

Command:

- `python3 bench/transformerlens_qwen_mps_repro.py --model Qwen/Qwen2.5-1.5B-Instruct --compare-layer0`

On `device="mps"`:

- `Qwen/Qwen2.5-1.5B-Instruct`
  - HF: `"Hello"` at `0.926177`
  - TL default: `"osoph"` at `0.094720`
  - TL explicit attention-result path: `"Hello"` at `0.926176`

Layer-0 comparison on the same run:

- `mps_cache`: `max=4.9491339`, `mean=0.30307725` vs CPU
- `mps_linear`: `max=4.9491339`, `mean=0.30307725` vs CPU
- `mps_linear_contig`: `max=0.0004199`, `mean=0.00000459` vs CPU
- `mps_einsum`: `max=0.0004185`, `mean=0.00000459` vs CPU
- `mps_broadcast`: `max=0.0004196`, `mean=0.00000459` vs CPU

The first large divergence in the CPU-vs-MPS cache comparison was `blocks.0.hook_attn_out`.

I can also reproduce the same failure mode outside TransformerLens with synthetic MPS tensors and non-contiguous `F.linear` weights, using Qwen-1.5B attention-output dimensions (`n_heads=12`, `d_head=128`, `d_model=1536`):

- `w.is_contiguous() = False`
- `linear_vs_contiguous`: `max=210.4180603`, `mean=44.25615692`
- `linear_vs_matmul`: `max=210.4180603`, `mean=44.25615692`

This looks consistent with `pytorch/pytorch#161640`, not a separate Qwen-specific issue.

Evidence bundle: <https://github.com/asuramaya/Like-Us/blob/main/docs/issues/transformerlens_mps_evidence_bundle.md>
Qwen 1.5B repro script: <https://github.com/asuramaya/Like-Us/blob/main/bench/transformerlens_qwen_mps_repro.py>
Qwen 1.5B output: <https://github.com/asuramaya/Like-Us/blob/main/bench/transformerlens_qwen_mps_repro_1p5b.txt>
Synthetic repro script: <https://github.com/asuramaya/Like-Us/blob/main/bench/mps_noncontiguous_linear_repro.py>
Synthetic repro output: <https://github.com/asuramaya/Like-Us/blob/main/bench/mps_noncontiguous_linear_repro.txt>
