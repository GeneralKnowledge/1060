# Extreme low-bit inference on the GTX 1060: Phase 1 audit and research

Status: Phase 1 (repository and literature audit), 2026-11-20. **Nothing in this document has been run
on the laptop yet.** It decides what to run first.

Every claim carries a label:

- **[measured]**: an experiment record on this laptop (record id given).
- **[source]**: read in source code. llama.cpp means the pinned commit `7bd1808` (tag b12021) that the lab
  already builds. LittleBit means `SamsungLabs/LittleBit` main as of 2026-11-20.
- **[published]**: a paper's own numbers. **[vendor]**: a model publisher's own numbers.
  **[independent]**: a third party's measurement on other hardware.
- **[estimate]**: my arithmetic. It must be measured before anyone relies on it.

## 1. Summary

1. **The pinned llama.cpp (b12021) already contains 1-bit and 2-bit weight formats with Pascal CUDA kernels.**
   `Q1_0` (1 sign bit per weight, one FP16 scale per 128, 1.125 bpw) and `Q2_0` (2 bits per weight, one
   scale per 64, 2.25 bpw, holds ternary weights) have MMVQ (decode) and MMQ (prompt) kernels built on
   `dp4a` and `__byte_perm`, both available on SM 6.1, with a dedicated Pascal-dp4a MMQ config
   [source: `ggml-cuda/vecdotq.cuh`, `mmvq.cu`, `mmq-config-pascal-dp4a.cuh`]. No new build, driver or
   toolkit is needed to try them. Whether they are correct and fast on this card is untested.
2. **The ternary model that matters most is derived from the lab's own model.** PrismML's
   Ternary-Bonsai-2 family is ternary {-1, 0, +1} throughout (including embeddings and LM head) and
   comes as 2B, 4B, 9B and 27B, derived from Qwen3.5-2B/4B/9B and Qwen3.8-27B. Its upstream GGUF format
   is `Q2_0` and needs llama.cpp b11555 or newer; b12021 qualifies, and b12021 also reads the
   `prism.hadamard.*` metadata these files carry [vendor; source]. The 4B file is 1.21 GB against the
   lab's 2.74 GB Qwen3.5-4B Q4_K_M, and the architecture (`qwen35`) is the one the lab already
   benchmarks. That gives an exact like-for-like comparison.
3. **LittleBit and LittleBit-2 cannot be used directly.** No checkpoints are published, training is
   quantisation-aware training with distillation (multi-GPU DeepSpeed in the repo), the code is
   CC BY-NC 4.0, and its inference path is plain PyTorch matmuls with dense ±1 tensors, so the memory
   saving exists only on disk [source: `quantization/modules/littlebit.py`]. Its representation can,
   however, be expressed with existing ggml ops and `Q1_0` tensors (section 4.3). That is a later,
   research-grade experiment, not a first step.
4. **BitNet b1.58 is CPU territory on this card.** Microsoft's GPU kernel targets the 2B4T model and is
   only tested on A100. In llama.cpp the BitNet ternary types `TQ1_0` / `TQ2_0` have no CUDA kernels at
   b12021 [source: no `TQ` cases in `ggml-cuda`], so they would run on the CPU. The bitnet.cpp CPU path
   (AVX2) is relevant, but the largest native BitNet models are small (2B to 3B).
5. **Realistic targets** [estimate, to be measured]: a ternary 9B entirely in VRAM (2.56 GB file); a
   ternary 27B (7.63 GB `Q2_0`) with about 2.3 GB of it on the CPU, at a guessed 4 to 7 tok/s. 13B and
   30B in conventional 4-bit formats need 3 to 13 GB of weights in RAM against 7.6 GB total, so they are
   limited by RAM bandwidth and capacity, not VRAM.
6. **First experiment (section 8):** check the 1-bit and 2-bit CUDA kernels on this card with
   `test-backend-ops` (no download, minutes), then run Ternary-Bonsai-2-4B `Q2_0` (1.21 GB) through the
   existing `smoke` and benchmark harness against the Qwen3.5-4B Q4_K_M baseline.

## 2. What exists in the repository

| Area | State | Relevance here |
|---|---|---|
| Hardware | Dell G3 3779, GTX 1060 Max-Q 6 GB (GP106, SM 6.1, 10 SMs, 192-bit GDDR5, no tensor cores), i7-8750H 6c/12t AVX2 (no AVX-512), 7.6 GiB RAM + 4 GiB swap, PCIe x8. Power-capped (55 to 70 W), memory at 3000 to 3800 MHz under load [measured: M0 audit, `docs/assessment.md`] | DIMM layout (single or dual channel) is **unknown**, and it decides CPU decode speed |
| Toolchain | Driver 550 (CUDA ≤12.4), nvcc 12.0 + g++-12, llama.cpp b12021 with CUDA build `d2db58ba90808f32` | CUDA 13 cannot target SM 6.1; PyTorch drops Pascal wheels from 2.15 (`research/references.md`) |
| Benchmark | `python3 -m lab`: locked llama-bench A/B, KLD vs Q8_0 reference logits, decode-path PPL check, `test-backend-ops` kernel gate, telemetry, safety limits. Lock changes need the user (`approve-benchmark`) | Reused as is. New models enter as new workloads in `config/benchmark.toml` |
| Workloads | `small` Llama-3.2-3B Q4_K_M (41.2 tok/s), `qwen35-4b` Qwen3.5-4B Q4_K_M (tg128 29.3, pp512 573, KLD 0.024, 3.0 GB VRAM) [measured], `capacity` Llama-3.1-8B Q4_K_M declared but never pinned or run | `qwen35-4b` is the baseline for the first experiment; `capacity` is the natural slot for 8B partial-offload work |
| Kernel workbench (M5) | decode-path and per-op gates, `research/priors.md`, scripted sweeps | Gates any custom kernel work later |
| Mind | continuous process in tmux `mind`; its deep tier loads Qwen3.5-4B on the GPU when free | Benchmarks must not run while the mind holds the GPU; RAM it uses reduces offload headroom |

Known weak spots that matter for this mission:

- **No task-quality evaluation.** Quality today is KLD/PPL against the same family's Q8_0. That works
  for "same model, different quant" but not for comparing different models (e.g. Bonsai vs Qwen, 9B vs
  4B). Phase 2 needs a fixed prompt set (section 7).
- **No CPU-offload measurements.** `research/priors.md` lists tensor placement for 7-8B as an open
  question. RAM bandwidth has never been measured.
- `test-backend-ops` suites only cover the types the current models use; `q1_0`, `q2_0` are not in them.

## 3. Hardware facts that decide the design

| Fact | Consequence |
|---|---|
| No tensor cores, FP16 at 1/64 of FP32 on GP106 [known, `research/priors.md`] | Anything that computes in FP16 or BF16 (PyTorch fp16 inference, LittleBit's reference forward, most Triton/Marlin/ExLlama-style kernels) is slow or unsupported. Integer `dp4a` paths are the fast path |
| `dp4a`, `__byte_perm`, `__popc` available on SM 6.1 | Binary and ternary dot products can be done with integer instructions: llama.cpp's `Q1_0`/`Q2_0` already do. XNOR-popcount kernels are possible but only pay off when activations are also binary, which no useful LLM uses |
| No `mma`/`ldmatrix`/`cp.async` (SM 7.0 to 8.0 features) | Rules out kernels written for Ampere+ (Marlin, QTIP/QuIP# CUDA paths, most AQLM kernels, EXL3, Microsoft's BitNet GPU kernel is untested below A100) |
| Decode reads every weight once per token. 4B Q4_K_M: about 2.7 GB per token at 29.3 tok/s, about 80 GB/s achieved [estimate from measured] | Speed scales with bytes per token. A 1.2 GB model could at most roughly double decode speed, minus fixed costs |
| PCIe x8 (Gen3 at best, about 7.9 GB/s theoretical) | Streaming weights to the GPU every token is hopeless for decode: 2 GB of streamed weights costs about 0.25 s per token. Offloaded layers must be **computed on the CPU** during decode (llama.cpp's default). Streaming only pays for big prompt batches |
| RAM: 7.6 GiB, DDR4-2666 class, 21 GB/s single channel or 43 GB/s dual channel theoretical [spec; layout unknown] | CPU-side weights cost roughly 1/20 to 1/40 s per GB per token. The OS, the mind and the KV cache share the same 7.6 GiB |

## 4. Approaches compared

### 4.1 Extreme low-bit methods

| Method | Real bits/weight (what counts) | Quality (published unless marked) | Checkpoints | Conversion of existing models | Inference maturity | CPU | Pascal GPU | Licence |
|---|---|---|---|---|---|---|---|---|
| **LittleBit** (NeurIPS 2025) | Target 1.0 to 0.1 bpw for the factorised linear layers only, counting packed sign bits plus FP16 scales [source: `_compute_eff_bits`]. Embeddings and LM head stay FP16: Llama-3-8B at 0.1 bpw is 0.1 GB body but **2.2 GB total** [published, LittleBit-2 Table 1] | Llama-2-7B WikiText-2 PPL 5.47 (FP16) → 9.08 at 1.0 bpw, 10.47 at 0.55, 15.92 at 0.1. Llama-3-8B: 6.10 → 16.30 at 1.0, 26.11 at 0.1. Zero-shot avg 68.9 → 52.8 (1.0) → 41.3 (0.1) on Llama-2-7B | **None published** (README example id is a placeholder) | Yes, by QAT + distillation from the FP model. Repo example: 5 epochs on C4+Wiki, DeepSpeed ZeRO-3 on 4 GPUs | Research code. Forward is `((((x*g) @ Vsign.T) * l) @ Usign.T) * h` in PyTorch with dense ±1 tensors; bit packing is only used for saving [source] | Through PyTorch only | Runs in principle in FP32 PyTorch, no memory saving at runtime, PyTorch Pascal wheels end at 2.14 | CC BY-NC 4.0 (non-commercial) |
| **LittleBit-2** (ICML 2026) | Same format and accounting as LittleBit; only the initialisation changes (latent rotation + joint ITQ), no inference overhead | Llama-2-7B PPL 8.27 (1.0), 9.65 (0.55), 14.70 (0.1); Llama-3-8B 11.53 (1.0), 23.74 (0.1). Better than LittleBit and OneBit at 1.0 bpw, still far from FP16 | None | Same as LittleBit | Same code base | Same | Same | Same |
| **BitNet b1.58** (trained ternary) | 1.58 information bits; stored 2 bits (bitnet.cpp I2_S, llama.cpp TQ2_0 2.06) or 1.69 (TQ1_0). Embeddings usually higher precision | BitNet b1.58 2B4T is competitive with 2B FP models [published by Microsoft]. Post-hoc conversion is not BitNet | 2B4T (Microsoft), Falcon-E 1B/3B, TriLM/Spectra 99M to 3.9B, community 8B (HF1BitLLM, 100B-token fine-tune) | No: needs training or long fine-tuning. `llama-quantize` to TQ1_0 on a normal model gives garbage (llama.cpp issue #15193) | bitnet.cpp CPU kernels mature for its models; llama.cpp TQ types CPU only | Yes (bitnet.cpp, llama.cpp TQ) | **No llama.cpp CUDA kernels for TQ1_0/TQ2_0** [source]. Microsoft's W2A8 GPU kernel uses dp4a but is tested only on A100 | MIT (models vary) |
| **Bonsai 1-bit** (PrismML, gen 1) | 1.125 bpw everywhere incl. embeddings and LM head (`Q1_0`, 1 sign bit + FP16 scale per 128) [vendor] | Bonsai-8B (from Qwen3-8B): average 70.5 vs 79.3 for Qwen3-8B; MMLU-R 65.7 vs 83, GSM8K 88 vs 93 [vendor] | 1.7B, 4B, 8B, 27B; 8B GGUF 1.16 GB | Method not published | Upstream llama.cpp type `Q1_0`, CUDA MMVQ/MMQ present in b12021 [source]. Vendor docs still point at their fork for Q1_0 | Yes | **Kernels exist with a Pascal config; untested** | Apache 2.0 |
| **Ternary-Bonsai-2** (PrismML, gen 2) | Ternary throughout incl. embeddings and LM head; stored as `Q2_0` 2.28 to 2.33 bpw (upstream), `PTQ1_0` 1.75 to 1.83 or `PQ2_0` 2.13 to 2.20 (fork only). Norms and the DeltaNet state path stay high precision (about 0.1%) [independent] | 27B: 98.2% of Qwen3.8-27B FP16 on a 14-benchmark average [vendor]. Independent: 130/140 vs 131/140 for an abliterated Q8_0 baseline on GSM8K/MATH/MuSR/HumanEval+ (2080 Ti); but 12/47 vs 19/47 on a multi-turn intent task where it lost conversation-dependent fields ("scope broke", 3090) | 2B, 4B, 9B, 27B; Q2_0 sizes 0.55 / 1.21 / 2.56 / 7.63 GB | Method not published | Upstream since b11555; 27B decode 39 tok/s (PQ2_0) on a 2080 Ti, 78 tok/s on a 3090 [independent] | Yes | **Kernels exist with a Pascal config; untested**. Fork types PTQ1_0/PQ2_0 untested and need the fork | Apache 2.0 (base model licences apply) |
| **Binary NNs** (OneBit, BiLLM, ARB-LLM, STBLLM, XNOR-style) | 1.0 to 1.1 bpw nominal (OneBit, BiLLM, ARB); 0.55 to 0.7 (STBLLM, structured sparsity); real storage higher because of scales, salient-weight masks and FP16 embeddings | Llama-3-8B PPL: OneBit 13.09 (1.0), BiLLM 59.37 (1.1), ARB-LLM 27.63, STBLLM 241.96 (0.55) [published, LittleBit-2 Table 1]. PTQ-only binarisation is not usable for chat | Few, mostly research | BiLLM/ARB/STBLLM are PTQ (calibration only); OneBit needs QAT | Research code, PyTorch, dense dequantisation | PyTorch | No fast kernels | Varies |
| **Extreme PTQ** (llama.cpp IQ1_S/IQ1_M/IQ2_XXS/IQ2_S, Q2_K, QuIP#, AQLM, QTIP) | IQ1_S 1.56, IQ1_M 1.75, IQ2_XXS 2.06, IQ2_S 2.5, Q2_K about 2.6 to 3.0 effective [llama.cpp quantize table] | Works on any model with an importance matrix. Quality cliff below about 2 bpw on 4 to 8B models (published community KLD/PPL data; the lab has not measured it). Qwen3.8-27B IQ2_XXS: 84.1% of FP16 on the vendor's 14-benchmark set vs 98.2% for Bonsai 2 [vendor] | Any GGUF repo; can also be made locally with `llama-quantize --imatrix` | Yes, minutes to hours on CPU | IQ types: mature in llama.cpp, CUDA MMVQ via dp4a. QuIP#/AQLM/QTIP CUDA kernels target newer GPUs | Yes | IQ types: yes in source (used by many Pascal users), unmeasured here | MIT (llama.cpp) |
| **QAT** in general (EfficientQAT, PV-Tuning, BitDistiller, Bonsai, LittleBit) | Whatever the format | The only route that has produced usable models below 2 bpw | Model-specific | Needs training compute. Not feasible for 4B+ on this card (FP16 is 1/64 rate, 6 GB). Possible later on user-owned servers | n/a | n/a | n/a | Varies |

Reading of the table:

- **Nominal bits per parameter are misleading in three ways.** LittleBit's 0.1 bpw Llama-3-8B is 2.2 GB
  in total because embeddings and LM head are FP16. Ternary "1.58-bit" models are stored at 2 to 2.3
  bits in the formats that have GPU kernels. Runtime memory adds KV cache, compute buffers and the CUDA
  context (about 0.3 to 0.5 GB here).
- **Quality evidence is strongest for ternary models trained or distilled from a strong base** (Bonsai 2),
  weaker for 1-bit (Bonsai gen 1: about 89% average, with big drops on MMLU-R and tool calling), and
  poor for anything below 1 bpw (LittleBit at 0.1 bpw: PPL 4 times FP16 on Llama-3-8B, zero-shot near the
  level of small models).
- **Every sub-2-bit route that works needs training.** The lab cannot train 4B+ models on this card.
  So the practical choice is between models someone else trained (Bonsai, BitNet) and PTQ (IQ types).

### 4.2 Alternative techniques

| Technique | Verdict for this machine |
|---|---|
| Low-rank / SVD compression (SVD-LLM, ASVD) | Alone, rank cuts hurt quickly ("Tiny-rank FP" at 0.1 bpw: Gemma-3-27B PPL 37.5 vs 4.58 [published]). Useful only as part of LittleBit-style binarised factors |
| Sparsity (2:4, unstructured, Wanda/SparseGPT) | 2:4 hardware sparsity needs Ampere. Unstructured sparse kernels lose to dense quantised GEMV at batch 1. Not worth it here |
| Activation quantisation | Already done: llama.cpp quantises activations to Q8_1 for dp4a. W2A8 is the natural mode on Pascal |
| Mixed precision / layer-wise allocation | Supported today: `llama-quantize --tensor-type` and imatrix per-tensor types. Combining ternary bulk with higher-precision sensitive tensors is cheap to test |
| Weight sharing / codebooks (AQLM, VPTQ) | Lookup-heavy decode; no Pascal kernels. Skip |
| Distillation | The method behind Bonsai and LittleBit. Requires training compute: out of scope on this card, possible on user servers later |
| Recurrent / hybrid architectures | Already in use: Qwen3.5 has 24 of 32 layers as Gated DeltaNet with fixed-size state, so KV cache grows only in 8 layers. Bonsai 2 27B: 48 linear + 16 full-attention layers. This matters more than KV quantisation for long context |
| KV-cache compression | q8_0 KV tested on Llama: -4% VRAM, -2% speed, discarded [measured: `20261101-180922-4bd0`]. Relevant again only when context is long or VRAM is the binding limit (27B offload) |
| Speculative decoding / MTP | Bonsai 2 9B ships an MTP head; independent 2080 Ti test saw +15 to +45% decode with MTP on a 27B graft. Worth a later experiment if the fork or upstream supports it |

Combinations that fit together without new kernels: ternary or 1-bit bulk weights + per-tensor mixed
precision + Q8_1 activations + hybrid linear attention + partial CPU offload. All are in llama.cpp.

### 4.3 Inference engines

| Engine | Verdict |
|---|---|
| **llama.cpp b12021 (already built)** | The base for everything. Has `Q1_0`, `Q2_0`, IQ1/IQ2 types with dp4a CUDA kernels and Pascal MMQ configs; `qwen35` and `bitnet` architectures; partial offload (`-ngl`, `--override-tensor`/`-ot`), mmap, KV types. TQ1_0/TQ2_0: CPU only |
| PrismML llama.cpp fork | Needed only for `PTQ1_0`/`PQ2_0` (the 1.75 bpw packing of Bonsai 2, 27B file 5.95 GB) and MTP/drafters. Based on an older upstream (b10709). Separate build directory if needed; not first |
| bitnet.cpp (Microsoft) | CPU kernels (AVX2 TL2/I2_S) for BitNet models; GPU kernel tested only on A100. Useful for a CPU-vs-GPU comparison on BitNet 2B4T later |
| PyTorch / Transformers | Last Pascal wheels are PyTorch ≤2.14 (CUDA 12.6); FP16 is slow on GP106. Fine for small offline work (LittleBit init, analysis), not for serving |
| Official LittleBit code | Training and evaluation only. Its inference reconstructs nothing but also saves nothing: the ±1 factors live as FP tensors at runtime |
| Custom CUDA kernels | Justified only after profiling shows a gap. Candidates: tuned MMVQ rows for Pascal (open question already in `research/priors.md`), and a fused LittleBit layer if that route gets that far |

**LittleBit on this hardware, as a design note.** A LittleBit layer is two thin matmuls and three
elementwise scalings (times two for the residual path). In ggml that is `mul`, `mul_mat`, `mul`,
`mul_mat`, `mul`, with the ±1 factors stored as `Q1_0` tensors (the sign bit is exactly a ±1 value and the
block scale can be 1). All of these ops have Pascal CUDA kernels in b12021, so a LittleBit model could
run in llama.cpp **without a new kernel**, only a GGUF converter and a model-graph change. Two
consequences:

- Ranks are small: Llama-2-7B attention projections get rank 1000 at 1.0 bpw and rank 80 at 0.1 bpw
  [estimate, from the repo's formula]. Thin matmuls at rank 80 are launch-bound, not bandwidth-bound,
  so speed will not scale with the bit count.
- "Reconstruct the dense block, then multiply" (strategy C below) is the wrong way to run it:
  reconstruction costs d_out × d_in × r operations per layer, far more than running the factorised form
  directly. Factorised execution is both smaller and faster.

This only becomes worth building once a trained LittleBit checkpoint exists, which needs QAT on
non-laptop hardware.

## 5. Execution strategies

| Strategy | Assessment |
|---|---|
| **A, GPU-first + offload** | llama.cpp's default partial offload. During decode the offloaded layers run on the CPU (weights are not copied per token); for prompt batches llama.cpp can copy them to the GPU. Best fit for "almost fits" models such as a 27B ternary at 7.63 GB |
| **B, CPU-first** | Wins only when the model does not fit in VRAM at all and RAM bandwidth is decent. Ternary formats help the CPU more than the GPU because the CPU is the more bandwidth-starved side. Needs the RAM bandwidth measurement |
| **C, hybrid reconstruction / streaming** | Streaming over PCIe x8 costs about 0.13 s per GB per token [estimate], worse than computing on the CPU from RAM (about 0.025 to 0.05 s per GB). Reconstruction from a more compressed form (e.g. LittleBit factors → dense) costs more compute than it saves. Expected to lose for decode; worth one measurement for prompt processing only |

## 6. Answers to the ten questions (current evidence)

1. **Can LittleBit/LittleBit-2 run on a GTX 1060?** The reference code can, in FP32 PyTorch, with no
   runtime memory saving, and there is no checkpoint to run. A real deployment would need a trained
   checkpoint plus a GGUF mapping (section 4.3). Untested.
2. **Without modern GPU instructions?** Yes in principle: its forward pass is ordinary matmuls with ±1
   matrices, which map onto `Q1_0` + dp4a. No mma, tensor cores or FP16 needed.
3. **CPU vs GPU for extreme low-bit?** For models that fit in 6 GB, the GPU should win (about 80 GB/s
   achieved vs at most 43 GB/s theoretical RAM). For BitNet TQ types, the CPU is the only option in
   llama.cpp. To be measured.
4. **Streaming compressed weights?** Over PCIe x8 for decode, no [estimate]. Memory-mapped weights on the
   CPU side, yes; that is strategy A.
5. **13B in 8 GB RAM + 6 GB VRAM?** A ternary 13B (about 3.7 GB) would fit in VRAM, but none of the
   ternary families has a 13B. A conventional 13 to 14B at Q4_K_M (about 8 to 9 GB) would put 3 GB in RAM:
   plausible but slow and tight next to the mind. At IQ2_XXS (about 4 GB) it fits in VRAM with a quality
   cost to be measured.
6. **Largest useful model?** Hypothesis: Ternary-Bonsai-2-27B (`Q2_0`, 7.63 GB) with about 2.3 GB on the CPU,
   or Ternary-Bonsai-2-9B entirely on the GPU. "Useful" will be decided by the prompt set in section 7.
7. **Layer-wise mixing?** Yes: llama.cpp supports per-tensor types; the open question is whether Bonsai's
   ternary weights tolerate it.
8. **Custom kernels on Pascal?** Possibly: MMVQ uses GENERIC parameters on Pascal and the new
   `Q1_0`/`Q2_0` kernels have never been tuned for it. Profile first.
9. **30B through offloading?** A ternary 27B is the realistic ceiling: 7.63 GB `Q2_0` (or 5.95 GB `PTQ1_0`
   with the fork, which would nearly fit in VRAM). A 30B+ conventional 4-bit model (17 GB+) does not fit
   in RAM + VRAM without disk paging, which would be far below 1 tok/s.
10. **What to borrow:** llama.cpp (already pinned) for everything; PrismML Bonsai models; the PrismML fork
    only if `PTQ1_0` is needed; bitnet.cpp for CPU comparisons; LittleBit's training code for any future
    QAT on user servers.

## 7. Phase 2 baseline plan (short)

- Baseline model: **Qwen3.5-4B Q4_K_M**, already validated [measured: tg128 29.3, pp512 573, KLD 0.024,
  3.0 GB VRAM].
- Add a fixed quality prompt set to the repo: about 30 prompts across general knowledge, reasoning,
  instruction following, coding, a 3-turn conversation, and factual questions with known answers;
  greedy decoding, fixed context, thinking off, scored automatically where possible (exact answers, unit
  tests for code) and the transcripts kept for reading. This is what lets different models be compared;
  KLD cannot.
- Measure RAM bandwidth (a 1-minute read-only test) and record the DIMM layout if the user can read it
  with `sudo dmidecode -t memory`.
- Load time, TTFT, peak RAM/VRAM and CPU/GPU utilisation already come from `lab smoke` and the
  benchmark telemetry.

## 8. First experiment proposal: E1, 1-bit and 2-bit weights on Pascal

**Hypothesis.** The pinned llama.cpp build runs `Q1_0` and `Q2_0` matmuls correctly on the GTX 1060, and
Ternary-Bonsai-2-4B (`Q2_0`, 1.21 GB) runs fully on the GPU with at least 1.3 times the decode speed and
at most 60% of the VRAM of Qwen3.5-4B Q4_K_M, with output that is still coherent.

**Requirements.** The existing build `d2db58ba90808f32`. No rebuild, driver or config change. One
download of about 1.2 GB (Ternary-Bonsai-2-4B `Q2_0` from `prism-ml/Ternary-Bonsai-2-4B-gguf`). The GPU
must be free of the mind's deep tier while it runs.

**Steps.**

1. Kernel check, no download (about 2 to 5 min):
   `test-backend-ops -b CUDA0 -o MUL_MAT -p "type_a=(q1_0|q2_0|iq1_s|iq2_xxs)"` and the same for
   `GET_ROWS`. Answers "do the 1-bit/2-bit kernels give the same results as the CPU on Pascal".
2. Download the 4B `Q2_0`, check its sha256 and metadata (architecture `qwen35`, tokenizer, vocab size
   against Qwen3.5-4B).
3. `lab smoke` with a temporary workload entry (not merged into the protected benchmark until the user
   approves): offload of all layers, VRAM, RAM, load time, sample output.
4. llama-bench tg128 / pp512 with the qwen35-4b winning flags, interleaved with the Q4_K_M baseline.
5. Quality: the decode-path perplexity on the lab's WikiText slice for both models, KLD against the Qwen
   Q8_0 reference only if the vocabularies match (it then measures distance from the base model, not
   quality), and 5 to 10 fixed prompts read side by side.

**Success:** step 1 passes; all layers on CUDA; tg128 ≥ 38 tok/s; VRAM ≤ 1.8 GB; coherent answers.
**Failure is also a result:** kernel mismatches (a Pascal bug to report upstream), CPU fallbacks, or
garbage output (format mismatch) each get recorded in `experiments/`.

**Cost.** About 20 to 30 minutes on the laptop, one 1.2 GB download, a few Claude turns. Nothing runs
longer than a few minutes at a time.

**What it unlocks.** If E1 passes: E2 is Ternary-Bonsai-2-9B (2.56 GB, fully on the GPU) against a 9B
conventional quant, E3 the RAM bandwidth and CPU-only decode test, E4 the 27B with partial offload.

## Sources

- LittleBit: Lee et al., "LittleBit: Ultra Low-Bit Quantization via Latent Factorization", NeurIPS 2025,
  [arXiv:2506.13771](https://arxiv.org/abs/2506.13771). Code: [SamsungLabs/LittleBit](https://github.com/SamsungLabs/LittleBit)
  (read: `quantization/modules/littlebit.py`, `quantization/utils/binary_packer.py`, `LICENSE`, `requirements.txt`).
- LittleBit-2: "Maximizing the Spectral Energy Gain in Sub-1-Bit LLMs via Latent Geometry Alignment",
  ICML 2026, [arXiv:2603.00042](https://arxiv.org/abs/2603.00042),
  [PMLR](https://proceedings.mlr.press/v306/lee26y.html),
  [Samsung Research blog](https://research.samsung.com/blog/LittleBit-2-Maximizing-the-Spectral-Energy-Gain-in-Sub-1-Bit-LLMs-via-Latent-Geometry-Alignment).
- BitNet GPU kernel: [microsoft/BitNet gpu/README.md](https://github.com/microsoft/BitNet/blob/main/gpu/README.md).
- llama.cpp ternary packing PR #8151 (TQ1_0/TQ2_0); llama.cpp b12021 source: `ggml/include/ggml.h`,
  `ggml/src/ggml-common.h`, `ggml/src/ggml-cuda/{vecdotq.cuh,mmvq.cu,mmq.cu,mmq-config-pascal-dp4a.cuh,ggml-cuda.cu}`,
  `tools/quantize/quantize.cpp`, `src/llama-arch.cpp`, `tests/test-backend-ops.cpp`.
- PrismML: [model formats](https://docs.prismml.com/download/formats.md),
  [models](https://docs.prismml.com/download/models), [llama.cpp guide](https://docs.prismml.com/run/llamacpp),
  [Bonsai-8B-gguf](https://huggingface.co/prism-ml/Bonsai-8B-gguf).
- Independent Bonsai tests: [ai-muninn, 2080 Ti](https://ai-muninn.com/en/blog/ternary-bonsai-2-27b-one-2080ti),
  [InsiderLLM, RTX 3090](https://insiderllm.com/guides/bonsai-2-27b-vs-qwen3-8-27b-rtx-3090/),
  [MindStudio, 9B](https://www.mindstudio.ai/blog/ternary-bonsai-2-9b-benchmark-tests),
  [RunAIHome overview, Sep 2026](https://runaihome.com/blog/ternary-llm-1-58-bit-quantization-home-lab-2026/).

Not verified in this audit (from general knowledge, to be checked before relying on it): exact Pascal
support status of QuIP#, AQLM, QTIP and EXL3 kernels; IQ-type quality curves on Qwen3.5.
