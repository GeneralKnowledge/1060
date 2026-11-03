# W0 feasibility spike: weight-space programming (partial, 2026-11-03)

Measurement only. llama.cpp b12021 (`7bd18083`). The locked CUDA build was used read-only. llama-quantize and
llama-export-lora come from the same source, built CPU-only into `~/1060/.wsp/build-cpu`. Nothing under winners/
or config/benchmark.lock was touched.

## 1. Machine state at start (17:32Z)
AC online, battery 100%, GPU power limit 70 W (default/max 70 W). Disk 169 GB free, RAM 6.5 GB available of 7.6 GB,
VRAM 6.06 GB free of 6 GB.

## 2–3. Checkpoints and GGUFs
| model | HF safetensors | F16 GGUF (convert peak RSS, time) | Q8_0 GGUF (quantize peak RSS, time) |
|---|---|---|---|
| Qwen3.5-0.8B | 1.75 GB | 1.48 GB (2.37 GB, 1:02) | 785 MiB (1.82 GB, 0:09) |
| Qwen3.5-4B | 9.32 GB | 8.65 GB (see logs) | 4387 MiB (4.45 GB, 6:30, nice 10) |

The HF downloads ran at about 2 MB/s, unauthenticated. An 8 GB box handles both models one at a time.

## 4. Rank-1 LoRA adapters on qwen35 (0.8B, layer 12)
Delta = -d dᵀW for a random unit d in W's output space, saved as a PEFT safetensors adapter (r=1, alpha=1;
`scripts/wsp/make_rank1_lora.py`) and converted with `convert_lora_to_gguf.py`. KLD is over wikitext-2 test,
40×512 tokens, with an F16 base on the GPU.

| HF tensor → GGUF | base+adapter vs base: mean / max KLD | merged (export-lora) vs base+adapter: mean / max KLD | same top-1 |
|---|---|---|---|
| mlp.down_proj → blk.12.ffn_down | 6.0e-5 / 2.0e-3 | 0.000000 / 5.4e-5 | 99.98% |
| linear_attn.out_proj → blk.12.ssm_out (DeltaNet) | 3.1e-5 / 1.1e-3 | 0.000000 / 5.9e-5 | 99.99% |
| linear_attn.in_proj_qkv → blk.12.attn_qkv (DeltaNet) | 5.5e-5 / 2.8e-3 | 0.000000 / 6.5e-5 | 100.00% |

`--lora` also loads on the Q8_0 base (PPL 16.627 vs F16 base 16.607).

Caveat: 0.8B has 16 k-heads and 16 v-heads, so the converter's V-head reorder doesn't apply. 4B has 16 k-heads and 32
v-heads, and its converter has a `NotImplementedError` path for LoRA on some reordered dims. LoRA on 4B DeltaNet
tensors is still untested.

## 6 (partial). PyTorch on sm_61
torch 2.14.1+cu126 from the pip cu126 index. Its arch list is sm_50/60/70/... (no sm_61), but it runs on the 1060:
2048² fp32 matmul max abs err vs CPU 4.6e-4, about 2.9 TFLOPS fp32. The transformers hook pass hasn't been run yet.

## Not yet run
llama-server throughput (`-np 1,2,4`), the transformers layer-15 hook pass and the GSM8K loop audit are deferred.
Scripts for them are in `scripts/wsp/` (`run_w0.sh` chains them unattended). The Timestral benchmark is blocked:
installing `git+https://github.com/mistralai/timestral` needs the owner's permission on this machine.
