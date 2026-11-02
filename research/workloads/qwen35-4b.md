# Workload brief: Qwen3.5-4B (`qwen35-4b`)

This is the research brief for the Qwen3.5-4B workload. Read it before proposing anything for this
workload. Facts are labelled **source** (llama.cpp at the pinned commit `7bd1808`, b12021, or the
model's `config.json`), **measured here** (an experiment record for this workload) or **hypothesis**.
The compatibility assessment is in `docs/qwen35.md`.

## Architecture (source: Qwen3.5-4B `config.json`; llama.cpp `src/models/qwen35.cpp`)

- Hybrid, 32 layers in the pattern L, L, L, F: **24 Gated DeltaNet linear-attention layers** and
  **8 full-attention layers**.
- Full attention: 16 query heads, 4 KV heads (GQA 4:1), **head_dim 256**, multi-section RoPE
  (`ggml_rope_multi`), Q/K RMS norm, sigmoid output gate.
- Linear attention (DeltaNet): 16 key heads and 32 value heads, key/value head dim 128, causal conv1d
  over q/k/v with kernel 4 (`SSM_CONV`), L2-normalised q/k, softplus/sigmoid gates, recurrent state of
  32 x 128 x 128 f32 per layer.
- hidden 2560, FFN 9216 (SwiGLU), **vocab 248,320**, tied input/output embeddings, max context 262k.
- The GGUF also carries a vision projector in a separate mmproj file; the lab does not use it.

## How llama.cpp runs it on this card (source)

- On CUDA the **fused `GGML_OP_GATED_DELTA_NET`** path is used for both prompt batches and decode:
  `llama-context.cpp` enables `fused_gdn_ar` / `fused_gdn_ch` when the backend supports the op, and
  `ggml-cuda` supports it. The unfused chunked path (`CUMSUM`, `TRI`, `SOLVE_TRI`, `EXP`, `PAD`, ...)
  is only a fallback.
- The CUDA DeltaNet kernel (`ggml-cuda/gated_delta_net.cu`) is templated on the value head size
  (16, 32, 64, 128) and splits 4 columns per warp; it has no architecture-specific tuning.
- `SSM_CONV` has CUDA kernels for conv widths 3-5 (and a fused conv+SiLU path), requiring
  `d_inner % 128 == 0`.
- Full-attention layers with flash attention on: the tile kernel for prompt batches and the vec
  kernel for decode, both instantiated for head size 256. No tensor-core path on Pascal.
- Quantised matmuls: MMQ for prompt batches and MMVQ for decode, both using DP4A on SM 6.1 (same as
  for Llama; see `research/priors.md`).

## What carries over from Llama-3.2-3B (priors, not measurements)

The starting config (`winners/qwen35-4b/best_config.json`) uses the settings that won on Llama:
flash attention on, full offload, f16 KV, batch 2048, ubatch 256 and no mmap (the last two promoted by the
known-knobs sweep on Llama: +2-3% prompt speed, -6% VRAM, host RAM 2.2 GB to 0.66 GB). They are a starting point.
Also settled on Llama and not worth repeating without a reason: CUDA graph env knobs (no effect), 2 CPU threads (no
effect), cuBLAS instead of MMQ (prompt about 50% slower).
Re-test one only with a stated reason that is specific to this architecture.

- **KV-cache quantisation is less interesting here (hypothesis).** Only the 8 full-attention layers keep
  a KV cache: about 64 MiB at ctx 4096 in f16. q8_0 KV saved 4% VRAM on Llama and cost 2% speed.
  Here the saving is smaller still, unless the context is very long.
- **Flash attention at head_dim 256 is not the same experiment as at 128 (hypothesis).** The FA win on
  Llama (+6% decode) does not prove one here: the tile and vec kernels run a different head size.

## Measured here (baseline 2026-11-01T213540)

Validated baseline on this laptop (b12021, CUDA, ngl 99, FA on, f16 K/V, batch 2048, ubatch 256,
no mmap), median of 10 samples:

| Metric | Value |
|---|---|
| pp512 | 572.6 tok/s (noise 0.18%) |
| pp2048 | 551.9 tok/s (noise 0.12%) |
| tg128 | 29.27 tok/s (noise 0.12%) |
| TTFT | 0.93 s |
| Mean KLD vs Q8_0 reference | 0.0242 (Q8_0 PPL 10.145, this config 10.240) |
| Decode-path PPL (ubatch 1, 2 chunks) | 8.571 |
| Peak VRAM | 3057 MiB |
| Peak GPU temperature | 69 °C |

Notes:
- Tokens per joule was 0.44 in one round and 0.61 in the other (23% spread). Speed metrics varied
  by under 1%. Treat efficiency gains on this workload as unproven unless they are large.
- The GATED_DELTA_NET / SSM_CONV / L2_NORM kernel suite reports 10 cases that CUDA does not support
  (124 passed, 0 failed). Identify which cases these are and whether the model's own shapes hit them,
  since an unsupported op falls back to the CPU.

## Bandwidth arithmetic (estimate, not measured)

- Decode reads about 2.7 GB of weights per token. That includes the tied 248k x 2560 output matrix,
  which alone is roughly 0.5 GB at Q6_K. At the 80-85 GB/s the card reached on Llama, that would
  give around 30 tok/s.
- DeltaNet state traffic is about 2 MB per layer per token (read and write), around 50 MB in total.
  That is small next to the weights, but it is f32 and goes through a kernel with no Pascal tuning.

## Open questions (suggestions, not a limit: find your own)

1. **Output head.** With a 248k vocabulary, the output matmul is a large share of decode bytes. How
   much of decode time does it take? Does its quantisation type matter for speed per unit of quality?
2. **Gated DeltaNet on Pascal.** Is the fused kernel's launch shape (4 warps, 4 columns per warp)
   good for 10 SMs, 32 value heads and S_v 128? How does decode (one token) compare with chunked
   prompt processing? Fusion with the surrounding gates and L2 norm.
3. **Flash attention at head_dim 256 on Pascal.** Tile and vec kernel occupancy and shared-memory
   use at D=256, compared with FA off.
4. **MMVQ on Pascal** (GENERIC parameters, no Pascal row), as for Llama, but with Qwen's matrix shapes
   (2560 x 9216 FFN, large vocab).
5. **Small-op fusion at batch 1.** The hybrid graph has more small ops per layer than Llama (gates,
   conv, norms). Count kernel launches per token, and try CUDA graphs.
6. **Context length.** Decode cost should grow slowly with context, since only 8 layers attend.
   Measure it before relying on it.
7. **Backends and engines.** CUDA vs Vulkan vs CPU for this architecture, and whether another
   inference engine has a better Pascal path for DeltaNet. A different engine is not a config change:
   write it up as a proposal with evidence.

## Correctness coverage and limits

The kernel check uses this family's own suites (`config/benchmark.toml`,
`[families."qwen3.5-4b"]`), covering:
- `GATED_DELTA_NET`, `SSM_CONV` and `L2_NORM`
- `FLASH_ATTN_EXT` at head size 256
- quantised `MUL_MAT`
- the gating and normalisation ops

test-backend-ops tests its own fixed shapes, which are not all of the model's shapes. A patch that
breaks only one shape could pass the kernel check. The decode-path perplexity check (ubatch 1, which
runs the recurrent decode path end to end) and the batched KL-divergence check are the end-to-end
safety net.
