# E1 — Ternary-Bonsai-2-4B Q2_0 vs Qwen3.5-4B Q4_K_M on the GTX 1060

Run: 2026-11-20 09:03–09:15 UTC, `experiments/lowbit/e1_run.sh`, lab commit `07ae4c8`, llama.cpp build 12021 (`7bd180830`,
pinned build `d2db58ba90808f32`), driver 550.163.01, GTX 1060 Max-Q 6 GB (CC 6.1), i7-8750H, 8 GB RAM.
Runtime flags for both models (= `winners/qwen35-4b`): `-ngl 99 -fa 1 -ctk f16 -ctv f16 -b 2048 -ub 256`.
The GPU was idle at the start (no compute apps). Nothing was killed and no step hung.

## Candidate provenance

| | |
|---|---|
| Repo | `prism-ml/Ternary-Bonsai-2-4B-gguf` @ `50ebba61be85e5be07e8083db468ace0fc1bac43` |
| File | `Ternary-Bonsai-2-4B-Q2_0.gguf` (upstream Q2_0, not PQ2_0/PTQ1_0/MTP/mmproj) |
| Size | 1,212,103,520 bytes (baseline Q4_K_M: 2,740,937,888) |
| sha256 | `64a5183b44921bdb5ac4a6f5f92007821d6ebe5c66997d3c147fd214ab9433d4` (matches HF LFS oid) |

## GGUF metadata

| | Ternary-Bonsai-2-4B Q2_0 | Qwen3.5-4B Q4_K_M |
|---|---|---|
| arch | `qwen35`, 32 blocks, d=2560 | `qwen35`, 32 blocks, d=2560 |
| vocab | 248,320 (gpt2 / pre `qwen35`) | 248,320 (gpt2 / pre `qwen35`) |
| file_type | 41 (Q2_0) | 15 (Q4_K_M) |
| tensor types | 201 × Q2_0, 225 × F32 | 131 Q4_K, 48 Q5_K, 48 Q8_0, 22 Q6_K, 177 F32 |
| params (llama-bench) | 4,205,751,296 | — |

The vocab matches, so KLD against the Qwen3.5-4B Q8_0 logits is valid. Note that Bonsai is a separately trained ternary model,
not a PTQ of the reference, so its KLD mixes quantisation error with real model differences.

## Step 1 — kernel correctness on Pascal (`test-backend-ops -b CUDA0`)

| type | MUL_MAT ok / total | GET_ROWS | status |
|---|---|---|---|
| q1_0 | 85 / 85 | 8 / 8 OK | **pass** |
| q2_0 | 85 / 85 | 8 / 8 OK | **pass** |
| iq1_s | 49 / 49 | 8 / 8 OK | **pass** |
| iq2_xxs | 84 / 84 | 8 / 8 OK | **pass** |
| tq2_0 | 0 / 48 | 8 not supported | **not supported** on CUDA (skipped, 0 failures) |

Overall: 303/303 run tests passed and 0 failed. tq2_0 has no CUDA kernel in this build and would fall back to CPU.

## Step 2 — load / offload / memory

| | Bonsai Q2_0 | Qwen Q4_K_M |
|---|---|---|
| offloaded layers | 33/33 (full GPU) | 33/33 (full GPU) |
| KV / recurrent-state / compute buffers @ c=4096 | 128 / 50.25 / 254.5 MiB | 128 / 50.25 / 254.5 MiB |
| max RSS (smoke, `/usr/bin/time -v`) | 1,504,964 kB (1.44 GiB) | 2,997,928 kB (2.86 GiB) |
| smoke load time | 0.90 s | 23.27 s |
| llama-server ready (prompt step) | 11.4 s | 9.7 s |
| smoke decode | 50.1 tok/s | 29.7 tok/s |

Load time is confounded by the page cache. Bonsai had just been downloaded and hashed, so it was in cache, and the baseline was
read cold from disk. The server-ready times (both warm) show no real load-time advantage.

## Peak VRAM (whole-GPU `memory.used`, idle ≈ 2 MiB)

| section | Bonsai Q2_0 | Qwen Q4_K_M |
|---|---|---|
| kernels | 687 MiB (test harness) | — |
| smoke (c=4096) | 1,671 MiB | 3,115 MiB |
| bench round 1 / 2 | 1,613 / 1,613 MiB | 3,057 / 3,057 MiB |
| perplexity + KLD (c=512) | 1,765 MiB | 3,209 MiB |
| prompts (server, c=4096×4 slots) | ≤ 3,061 MiB combined; not split per model | |

Section boundaries come from the `run.log` timestamps. The MARK lines in `gpu.csv` were lost because `nvidia-smi` holds the file
open without O_APPEND and overwrites the appended marks (a script bug, see below).

## Step 3 — speed (`llama-bench -r 5`, interleaved A B A B, tok/s)

| test | Bonsai r1 | Bonsai r2 | **Bonsai mean** | Qwen r1 | Qwen r2 | **Qwen mean** | ratio |
|---|---|---|---|---|---|---|---|
| pp512 | 599.30 | 599.05 | **599.2** | 573.56 | 571.87 | **572.7** | 1.05× |
| pp2048 | 575.75 | 575.44 | **575.6** | 553.74 | 552.00 | **552.9** | 1.04× |
| tg128 | 50.69 | 50.65 | **50.67** | 29.44 | 29.32 | **29.38** | 1.72× |

The rounds are very consistent (<0.5 %). Decode scales roughly with weight bytes, and prefill is compute-bound and nearly unchanged.

## Step 4 — quality (wikitext-2 slice, c=512)

| | Bonsai Q2_0 | Qwen Q4_K_M |
|---|---|---|
| PPL (20 chunks, batched) | **18.41 ± 0.83** | **10.24 ± 0.39** |
| decode PPL (2 chunks, `-ub 1`) | 16.70 ± 2.59 | 8.57 ± 1.11 |
| mean KLD vs Qwen3.5-4B Q8_0 | **0.509 ± 0.010** | 0.0242 ± 0.0015 |
| 99 % KLD / max KLD | 3.55 / 10.39 | 0.19 / 4.40 |
| same top token | 70.4 % | 92.5 % |
| RMS Δp | 17.3 % | 4.2 % |

The batched and decode paths agree within error for both models, so the decode path shows no kernel-specific breakage.

## Step 5 — fixed prompts (`prompts-v0.jsonl`, greedy, thinking off)

| id | cat | Bonsai | Qwen | notes |
|---|---|---|---|---|
| know-1 | knowledge | ✓ Canberra, 1927 | ✓ Canberra, 1927 | Qwen adds a wrong detail ("Commonwealth of Australia Act was passed") |
| know-2 | knowledge | ✓ García Márquez | ✓ García Márquez | |
| reason-1 | reasoning | ✓ 12 | ✓ 12 | |
| reason-2 | reasoning | ✓ No | ✓ No | |
| instr-1 | instruction | ✓ 3 items, no extra text | ✓ same | |
| code-1 | coding | ✓ correct `is_prime` | ✓ identical code | |
| fact-1 | hallucination | ✓ "no one has" | ✓ "no one has", but calls Viking a Soviet lander (wrong) | |
| multi-1 | multi-turn | ✓ recalls Priya/green, but says "*My* name is Priya" (pronoun slip) | ✓ | |
| decode tok/s | | 46.7–49.5 | 28.4–29.0 | |

Expectation substring hits: Bonsai 6/6 and Qwen 6/6 (2 prompts have no expectation). Bonsai's chat answers are coherent and shorter.

**Raw completion (no chat template) degenerates for Bonsai.** The greedy smoke prompt "The three most important ideas in computer science
are" produced `1. **Abstraction** 2. **Abstraction** 3. **Abstraction**` and then repeated the prompt in a loop. The baseline produced a coherent
list. It is not gibberish, but this is a repetition collapse under greedy raw completion.

## Host / RAM bandwidth (read-only, numpy 1.26.4)

`np.copyto` on 256 MiB ×10: **13.5 GB/s copy (≈27 GB/s read+write)**. The download was running concurrently, so treat this as a lower bound.
CPU governor `powersave`.
```
MemTotal:        7969020 kB
MemFree:          234804 kB
MemAvailable:    6470836 kB
Buffers:           34936 kB
Cached:          5757340 kB
```

## Success criteria (experiments/lowbit/README.md)

| criterion | result | verdict |
|---|---|---|
| Q1_0/Q2_0 kernels pass test-backend-ops on Pascal | q1_0 85/85, q2_0 85/85 MUL_MAT, GET_ROWS OK | **PASS** |
| Runs fully on the GPU | 33/33 layers offloaded, CUDA backend | **PASS** |
| tg128 ≥ 38 tok/s | 50.67 tok/s (baseline 29.38) | **PASS** |
| VRAM ≤ 1.8 GB | peak 1,765 MiB = 1.72 GiB (ppl); 1,613 MiB in bench; 1,671 MiB at c=4096 | **PASS** (margin: 1,765 MiB is 1.85 decimal GB; README uses GiB for the 3.0 GB baseline) |
| Coherent answers | 8/8 chat prompts coherent and correct; raw greedy completion loops | **PASS with caveat** |

**Not a README criterion, but it decides usefulness:** quality is clearly below Q4_K_M. PPL is +80 % (18.4 vs 10.2), mean KLD is 21× higher,
and the top token agrees only 70 % of the time. On this short prompt set the gap does not show, so `prompts-v0` (8 prompts) is too easy to
separate these models. A harder or longer eval is needed before calling it a replacement.

## Errors / anomalies

1. `gpu.csv` MARK lines are missing. `nvidia-smi ... > gpu.csv` truncates and writes at its own offset, so `mark()`'s `>>` appends are
   overwritten. Fix: start the sampler with `>>` (O_APPEND) or write marks to a separate file.
2. The script's smoke grep finds no `offloaded`/`n_vocab`/`arch` lines at default verbosity. Offload was captured afterwards with `-v`
   runs (`offload_*.log`) and metadata with gguf-py.
3. The script needs `benchmarks/workloads/wikitext-2-test-slice.txt`, which is untracked in `~/1060`. It was copied into the worktree for
   this run (not committed).
4. No CPU fallback, no kernel failures, no OOM, no hangs.
