# E3 — Ternary-Bonsai-2-9B Q2_0 on the GTX 1060

Run: 2026-11-20 16:11–16:21 UTC, `experiments/lowbit/e3_run.sh` at lab commit `3e82459`, pinned CUDA build `d2db58ba90808f32`
(llama.cpp `7bd180830`). Runtime flags are the lab's: `-ngl 99 -fa 1 -ctk f16 -ctv f16 -b 2048 -ub 256`. The task eval is greedy, with thinking off,
512 max tokens and c=4096, the same as E2. The GPU was idle at the start. There were no OOM, no retry at -c 2048 and no hangs.

## Model

| | |
|---|---|
| Repo | `prism-ml/Ternary-Bonsai-2-9B-gguf` @ `82265ccc78ddaac4700286a35e9badcc32376dd9` |
| File | `Ternary-Bonsai-2-9B-Q2_0.gguf` (upstream Q2_0), 2,556,640,128 bytes |
| sha256 | `1f884aa56aacd2bbbb2e4b7abaead2b1102b74a9c75a23fc33dab916813a384d` (matches HF LFS oid) |
| Architecture | `qwen35`, 32 blocks, n_embd 4096, vocab 248,320 (same tokenizer as Qwen3.5-4B / Bonsai-4B) |
| Params / bits per weight | 8,953,803,264 / **2.27** (model_size 2,545,600,512 B) |

## Memory and offload

| | value |
|---|---|
| Offloaded layers | **33/33** (full GPU) |
| CUDA0 buffers @ c=4096 | model 2,154.9 MiB, KV 128 MiB, compute 260.5 MiB |
| Peak VRAM (whole GPU) | smoke 2,751 MiB, bench 2,693 MiB, eval server 2,707 MiB → **2.69 GiB** |
| Max RSS (smoke) | 2,817,180 kB (2.69 GiB) |
| Load time (smoke, warm cache) | 1.09 s |

## Speed (llama-bench r3, tok/s)

| test | Bonsai-2-9B Q2_0 |
|---|---|
| pp512 | 352.2 ± 0.8 |
| pp2048 | 344.4 ± 0.2 |
| tg128 | **29.47 ± 0.01** |

## Tasks (GSM8K test first 50 + 10 coding tasks)

| model | GSM8K | code | 512-token cap hits (of 60) | answers with a line repeated ≥ 5× | median gen tok/s | tg128 | peak VRAM (eval) |
|---|---|---|---|---|---|---|---|
| **Bonsai-2-9B Q2_0 (E3)** | **47/50 (94 %)** | **10/10** | 2 | 0 | 28.5 | 29.5 | 2,707 MiB |
| Qwen3.5-4B Q4_K_M (E2) | 44/50 (88 %) | 10/10 | 4 | 0 | 28.6 | 29.8 | 3,063 MiB |
| Bonsai-2-4B Q2_0 (E2) | 44/50 (88 %) | 10/10 | 3 | 0 | 49.1 | 50.7 | 1,623 MiB |

Wrong answers (3). All are coherent and there were no loops:
- **g21** (want 14, got 2). It misreads the ages ("Raymond had his son when he was 23"), the same mistake Bonsai-4B made.
- **g37** (want 2, cut off at 512 tokens). It keeps re-reading an inconsistent-looking premise ("has $5 left"). Q4_K_M also missed this one at
  the cap. The grader's last-number fallback picked 13.
- **g45** (want 104, cut off at 512 tokens). It debates what "2/5 times more" means and never reaches a final answer.

Raw greedy completion (no chat template) repeats a short coherent list ("Algorithms / Data Structures / Data Representation …").
That is milder than Bonsai-4B's "Abstraction ×3" collapse, and chat-mode answers do not loop.

## E3 success criterion (experiments/lowbit/README.md)

| criterion | result | verdict |
|---|---|---|
| Runs fully on the GPU | 33/33 layers | **PASS** |
| Beats Bonsai-2-4B and Q4_K_M on GSM8K-50 + code (≥ 47/50 or clearly better on code) | 47/50 and 10/10, vs 44 and 44 (code tied at 10/10; Q4_K_M rescored from 46, see E2) | **PASS** (exactly at the threshold) |
| ≥ 20 tok/s | 29.5 tok/s | **PASS** |
| VRAM ≤ 4.5 GiB | 2.69 GiB peak | **PASS** |

**E3 passes.** The caveat is that 47 vs 44 vs 44 on 50 questions is within one standard error (about ±4 points). So the 9B is *not demonstrably* better
than Qwen3.5-4B Q4_K_M on this set. It is about as good at the same decode speed, with 12 % less VRAM, fewer cap hits and no loops. The set is near
ceiling (all three are ≥ 88 % and code is 10/10 everywhere), so a harder or larger eval is needed to separate them.
VRAM headroom is large: 2.7 of 6 GiB leaves about 3 GiB for longer context.
