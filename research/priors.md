# Prior knowledge

Read this before proposing anything. Each entry is either **measured here** (an experiment record on this
laptop) or **known** (hardware specification or llama.cpp source at the pinned commit `7bd1808`, b12021).
Known items are applied as defaults or settled by `python3 -m lab sweep`; they are not research. Spend
experiments on the open questions at the end.

## Measured on this laptop

| Result | Record |
|---|---|
| Baseline Llama-3.2-3B Q4_K_M, full offload, f16 KV: 41.2 tok/s tg128, 773 tok/s pp512, 2.5 GB VRAM | `baseline-2026-11-01T175223+0000` |
| Flash attention on: promoted, +6.2% tg128, about +3% balanced score | `20261101-180123-d476` |
| q8_0 K and V cache (with FA): discarded, balanced score 0.988 at ctx 4096 | `20261101-180922-4bd0` |
| ubatch 1024 (from 512): discarded, balanced score 0.988 | `20261101-181719-662f` |

## Known (do not re-test without new evidence)

- **GPU:** GTX 1060 Max-Q, GP106, SM 6.1, 10 SMs, 6 GB GDDR5 on a 192-bit bus. No tensor cores.
- **FP16 is slow on GP106:** FP16 arithmetic runs at 1/64 of FP32. Builds or flags that move math to FP16
  (e.g. `GGML_CUDA_F16`) are not worth testing.
- **DP4A is available** (SM 6.1). llama.cpp already uses `__dp4a` in both MMQ (prompt) and MMVQ (decode)
  on this card (`ggml-cuda/common.cuh`, `GGML_CUDA_CC_DP4A 610`). "Use DP4A" is not a new idea here;
  better use of it is.
- **MMQ is already the default for every quantised matmul on Pascal** (`ggml_cuda_should_use_mmq`:
  NVIDIA without FP16 tensor cores always takes MMQ). `-DGGML_CUDA_FORCE_MMQ=ON` changes nothing;
  `-DGGML_CUDA_FORCE_CUBLAS=ON` is the only alternative. Both are compile options, not environment
  variables.
- **Decode (MMVQ) on Pascal uses the GENERIC launch parameters** (`ggml-cuda/mmvq.cu`,
  `get_device_table_id`): Turing, RDNA, GCN and GB10 have tuned rows, Pascal has none.
- **Flash attention on Pascal** uses the tile kernel for prompt batches and the vec kernel for decode
  (no tensor-core path).
- **Decode is memory-bound** (expected; estimate, not measured): about 1.9-2.0 GB of weights are read
  per token, so 41-44 tok/s is about 80-85 GB/s against a nominal 182-192 GB/s.
- **Full offload is best when the model fits.** Partial offload only matters for models that do not
  fit in 6 GB.
- **Runtime environment knobs that exist at this commit:** `GGML_CUDA_GRAPH_OPT=1`,
  `GGML_CUDA_GRAPH_CHUNK_SIZE` (default 512 nodes, 0 = whole graph), `GGML_CUDA_DISABLE_FUSION=1`.
  Settled by the `known-knobs` sweep.

## Settled cheaply by sweep (no research agent needed)

`experiments/configurations/sweeps/known-knobs.json`: CUDA graph options, ubatch 256, CPU threads,
load mode, cuBLAS instead of MMQ. Vulkan is not in it yet (needs the Vulkan SDK on the laptop). Run with `python3 -m lab sweep <file>`; each step is
a normal locked A/B experiment against the current best.

## Open questions (where the research budget goes)

1. A Pascal row for the MMVQ parameter table (warps and rows per block for 10 SMs): decode speed.
2. Q4_K / Q6_K data loading in MMVQ: achieved bandwidth vs the 182-192 GB/s ceiling.
3. Fusing small per-token ops at batch 1 (norm, rope, residual adds) to cut launches and memory passes.
4. Which quantisation gives the best speed per unit of quality on this card (needs more model files
   declared in `config/benchmark.toml` by a human).
5. Tensor-level CPU/GPU placement for 7-8B models that do not fit in 6 GB.

Items 1-3 need source changes. Until the agent can read and compile llama.cpp itself (M5 step 2),
propose them as small, self-contained patches.
