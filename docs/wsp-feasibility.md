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
Scripts for them are in `scripts/wsp/` (`run_w0.sh` chains them unattended). The owner chose to run Timestral first (below).

## Timestral (run first, 2026-11-03)
`timestral==1.0.0` (git a534bdb) is installed in `.venv-wsp` on the existing torch 2.14.1+cu126, which it didn't
replace. GPU: GTX 1060 Max-Q, 70 W limit, AC power. Script: `scripts/wsp/timestral_bench.py`. Raw output is in
`.wsp/results/timestral.json` on the laptop. The whole run took 3:39 wall-clock with 3.6 GB peak host RSS.

| model | params | load (cached) | VRAM weights / eval peak | latency ctx 2048, B=1, H64 / H256 | B=16, H64 / H256 | full eval below |
|---|---|---|---|---|---|---|
| Timestral-28M | 27.8 M | 0.41 s | 107 / 144 MiB | 29.8 / 32.7 ms | 84.5 / 91.8 ms | 3.7 s |
| Timestral-209M | 209.3 M | 1.65 s | 814 / 868 MiB | 77.4 / 80.5 ms | 437 / 456 ms | 15.8 s |

Eval setup: each series' last H steps are held out, the context is up to 2048 preceding steps, NaN in the context is
passed as missing, and batches are 16. Up to 100 series per dataset, sampled with seed 0, from
`autogluon/chronos_datasets`: `m4_hourly`, `electricity_15min`, `exchange_rate`, `nn5`, `monash_traffic`. Metrics:
MASE uses the median and the in-context seasonal-naive MAE as scale, averaged over series. WQL covers levels
0.1..0.9 and is aggregated as Σ quantile loss / Σ|y|.

| dataset (season) | H | n | MASE snaive / 28M / 209M | WQL snaive / 28M / 209M |
|---|---|---|---|---|
| m4_hourly (24) | 64 | 100 | 1.381 / 0.835 / 0.761 | 0.040 / 0.027 / 0.024 |
| m4_hourly (24) | 256 | 100 | 2.573 / 1.334 / 1.420 | 0.063 / 0.032 / 0.033 |
| electricity_15min (96) | 64 | 100 | 1.474 / 1.277 / 1.302 | 0.153 / 0.119 / 0.118 |
| electricity_15min (96) | 256 | 100 | 1.137 / 0.848 / 0.841 | 0.126 / 0.072 / 0.068 |
| exchange_rate (1) | 64 | 8 | 8.151 / 7.930 / 7.908 | 0.020 / 0.016 / 0.016 |
| exchange_rate (1) | 256 | 8 | 13.294 / 13.774 / 13.442 | 0.046 / 0.039 / 0.037 |
| nn5 (7) | 64 | 90 | 1.034* / 0.770 / 0.762 | 0.232* / 0.140 / 0.139 |
| nn5 (7) | 256 | 0 | - / - / - | - / - / - |
| monash_traffic (24) | 64 | 100 | 0.897 / 0.671 / 0.655 | 0.302 / 0.184 / 0.179 |
| monash_traffic (24) | 256 | 100 | 1.757 / 1.209 / 1.210 | 0.654 / 0.375 / 0.376 |

\* nn5 contexts all contain NaNs, so the seasonal-naive baseline is computed on a forward-filled context.
nn5 H=256 has n=0 because its series (791 steps) are too short. exchange_rate has 8 series with season 1, so its
MASE scale is the one-step naive error, which is why the values are large.

**Where 28M is weak:**
- Nowhere is it more than 10% worse than 209M on MASE. The biggest gap is m4_hourly H=64 (0.835 vs 0.761, +10%
  MASE, +12% WQL).
- Its only loss to seasonal-naive is exchange_rate H=256 on MASE (13.77 vs 13.29). Its WQL there is still better
  (0.039 vs 0.046), and it's 8 series.
- On m4_hourly H=256 and electricity_15min H=64, 28M is slightly better than 209M.

Overall, 28M matches 209M closely on this set, at 2.6× lower latency for B=1, 4.8× lower for B=16, and about 1/7 of
the VRAM.
