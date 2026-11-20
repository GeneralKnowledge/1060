# E2 — trained ternary (Bonsai Q2_0) vs 2-bit PTQ of the same base (Qwen3.5-4B IQ2_XXS / IQ2_M)

Run: 2026-11-20 11:22–12:41 UTC, `experiments/lowbit/e2_run.sh` at lab commit `f68a306`, plus one local edit (imatrix `-b 512`, see
"Deviations"). GTX 1060 Max-Q 6 GB (CC 6.1), i7-8750H, 8 GB RAM. The GPU was idle at the start, and nothing hung or was killed during this run.

- **Inference binaries:** the pinned CUDA build `d2db58ba90808f32` (llama.cpp `7bd180830`). It was not modified.
- **Quantise/imatrix binaries:** a separate CPU-only build of the same commit `7bd180830` (`~/1060/backends/builds/e2-tools`,
  `-DGGML_CUDA=OFF -DCMAKE_BUILD_TYPE=Release`, about 2.5 min at -j4).
- **Runtime flags (all models):** `-ngl 99 -fa 1 -ctk f16 -ctv f16 -b 2048 -ub 256`.
- **Task eval:** greedy, thinking off, `max_tokens` 512, llama-server c=4096.

## Models

| model | file bytes | bits/weight* | sha256 (first 16) | source |
|---|---|---|---|---|
| Qwen3.5-4B Q4_K_M | 2,740,937,888 | 5.19 | — (E1 baseline) | lab baseline |
| Ternary-Bonsai-2-4B Q2_0 | 1,212,103,520 | **2.28** | `64a5183b44921bdb` | prism-ml @ `50ebba61` (E1) |
| Qwen3.5-4B IQ2_XXS | 1,538,393,280 | 2.91 | `e7d0834e48cb27fc` | requantised from Q8_0 + imatrix |
| Qwen3.5-4B IQ2_M | 1,741,554,880 | 3.29 | `8427e2c128c09eba` | requantised from Q8_0 + imatrix |

\*bits/weight = llama-bench model_size × 8 / 4,205,751,296 params, which includes embeddings and output. The llama.cpp IQ2_XXS mix keeps some tensors at
higher precision, so it is **27 % larger than Bonsai**, not the same size. That size gap favours IQ2_XXS. Full hashes are in `models.txt`.

## Speed and VRAM (llama-bench r3, tok/s; peak VRAM = whole-GPU `memory.used` per MARK section)

| model | pp512 | tg128 | VRAM bench | VRAM PPL/KLD (c=512) | VRAM eval server (c=4096) |
|---|---|---|---|---|---|
| Q4_K_M | 581.3 | 29.82 | 3,009 MiB | 3,209 (E1) | 3,063 MiB |
| Bonsai Q2_0 | 604.8 | **50.67** | 1,565 MiB | 1,765 (E1) | 1,623 MiB |
| IQ2_XXS | 606.3 | 31.59 | 1,863 MiB | 2,063 MiB | 1,917 MiB |
| IQ2_M | 582.5 | 29.30 | 2,057 MiB | 2,257 MiB | 2,111 MiB |

The IQ2 quants are barely faster than Q4_K_M on Pascal because the codebook dequant costs about as much as the bytes saved. Bonsai's Q2_0 is a plain 2-bit
format and decodes 1.7× faster.

## Quality — WikiText-2 slice, c=512, 20 chunks, vs Qwen3.5-4B Q8_0 logits

| model | PPL | mean KLD | same top token |
|---|---|---|---|
| Q4_K_M (E1) | 10.24 ± 0.39 | 0.024 | 92.5 % |
| Bonsai Q2_0 (E1) | 18.41 ± 0.83 | 0.509 | 70.4 % |
| IQ2_XXS | 15.85 ± 0.63 | 0.559 | 66.7 % |
| IQ2_M | 12.47 ± 0.48 | 0.287 | 76.1 % |

**Caveat:** the imatrix was calibrated on this same WikiText slice (40 × 512 tokens), so PPL/KLD flatter the IQ2 quants. Bonsai is also a separately
trained model, so its KLD includes genuine model differences on top of quantisation error. The task scores below are the fair comparison.

## Tasks — GSM8K test first 50 + 10 coding tasks with unit tests

| model | GSM8K | code | answers at 512-token cap (of 60) | answers with a line repeated ≥ 5× | median gen tok/s | eval wall |
|---|---|---|---|---|---|---|
| Q4_K_M | **46/50 (92 %)** | **10/10** | 4 | 0 | 28.6 | 641 s |
| Bonsai Q2_0 | **44/50 (88 %)** | **10/10** | 3 | 0 | **49.1** | **356 s** |
| IQ2_XXS | 13/50 (26 %) | 3/10 | **52** | **45** | 30.5 | 998 s |
| IQ2_M | 40/50 (80 %) | 6/10 | 13 | 2 | 28.5 | 682 s |

At n=50 one standard error is about ±4–6 points. The differences among Bonsai, Q4_K_M and IQ2_M are within noise. The gap to IQ2_XXS is not.

### Representative wrong answers

- **Q4_K_M** (g12, g37, g41, g43). These are ordinary reasoning slips. g12 answers the break-even year 12 instead of 13. g37 and g43 hit the cap while
  double-checking after reaching the right value, so the lenient last-number grader picked a stray number.
- **Bonsai Q2_0** (g5, g8, g12, g21, g38, g39). These are coherent reasoning errors, with no loops or garbage:
  - g5 ignores "60 % of the price" and uses $3 per glass (50 vs 64).
  - g21 misreads the ages (2 vs 14).
  - g8, g38 and g39 ran out of tokens mid-way through a long, sensible derivation.
- **IQ2_XXS.** It degenerates on most prompts:
  - g3 prints `Answer: 180` 66 times until the cap.
  - g2 repeats a "The profit is calculated as follows:" block 4×.
  - g4 rambles "the problem does not specify…".
  - g5 computes a geometric series glass by glass until it is cut off.
  - On the code tasks, 7 of 10 answers hit the cap. 3 of those failures are only because the unclosed code fence became a SyntaxError (a grader artefact
    when the answer is truncated), and 3 are wrong logic (to_roman, rle, second_largest).
- **IQ2_M** (10 GSM8K and 4 code). Mostly reasoning errors. 2 answers loop: g7 prints `Answer: 120 minutes.` 20× and g12 repeats per-year cash-flow
  blocks. g5 makes the same "flat $3" error as Bonsai.

## E2 success criterion (experiments/lowbit/README.md)

| criterion | values | verdict |
|---|---|---|
| Bonsai ≥ IQ2_XXS + 10 points on GSM8K | 88 % vs 26 % (+62) | **PASS** |
| Bonsai within 15 points of Q4_K_M on GSM8K | 88 % vs 92 % (−4) | **PASS** |

**E2 passes.** At 1.2 GB (2.3 bits per weight, 1.6 GB VRAM) the trained ternary model holds Q4_K_M-level task accuracy on this set, at 1.7× the decode speed.
Post-training 2-bit of the same base collapses at IQ2_XXS and is still clearly worse at IQ2_M (3.3 bits per weight, 2.1 GB). PPL/KLD rank Bonsai below
IQ2_M, which is the opposite of the task scores. That fits both the calibration overlap and the fact that Bonsai is a different trained model.

## Deviations and failed attempts (kept in this directory's siblings)

1. `e2/20261120T104956Z/`: the first attempt tried to add llama-quantize/imatrix to the pinned build. A 5 Nov glibc-header update
   (`/usr/include`, 477 files newer than the build) made make treat every object as stale, including all CUDA objects, so I stopped it after about 1 min.
   Some ggml-base/common objects were recompiled inside the build tree, but no library or binary in `build/bin` was relinked (timestamps still 1 Nov).
2. `e2/20261120T105419Z/`: the CPU imatrix at the script's default batch (2048 tokens, 4 seqs) reached 6.6 GB RSS on the 7.8 GB machine, mostly 248k-vocab
   logits on top of the 4.5 GB Q8_0. It sat in disk wait and swap at 418 s per pass (ETA 70+ min). I stopped it before any imatrix file was written.
3. For this run I added `-b 512` to the imatrix call in `e2_run.sh`. That gave 16.6 s per chunk and a 16 min total. Batch size does not change the collected
   statistics: chunk-1 PPL was 5.9906 in both runs.
4. The quantise step itself is CPU-only: IQ2_XXS took 366 s and IQ2_M 255 s.
5. GGUFs and the imatrix are in `~/1060/models/` (git-ignored, not committed).
