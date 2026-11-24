# E4b — Ternary-Bonsai-2-27B Q2_0 with partial CPU offload on the GTX 1060

Run: 2026-11-20 18:40–19:42 UTC, `experiments/lowbit/e4_run.sh offload` at lab commit `d6fb81f`, pinned CUDA build `d2db58ba90808f32`
(llama.cpp `7bd180830`, i.e. b12021, which is newer than the b11556 that PrismML says the 27B Q2_0 needs). Flags are `-fa 1 -ctk f16 -ctv f16 -b 2048 -ub 256`,
with mmap on and `-ngl` swept. The eval subset was rerun by hand with `-np 1` (see "Deviations").

## Model

| | |
|---|---|
| Repo | `prism-ml/Ternary-Bonsai-2-27B-gguf` @ `839bd57023c1558945afea70a74fa83c054f35f9` (upstream repo; it has Q2_0, so the `-dev` fallback was not needed) |
| File | `Ternary-Bonsai-2-27B-Q2_0.gguf`, 7,626,008,928 bytes |
| sha256 | `4f99aed01b8a877e153f9aa6569a4440fe17c59701cc0953e36d7f460549e70e` (matches HF LFS oid and the `-dev` copy) |
| Arch | `qwen35`, 64 layers (65 offloadable incl. output), n_embd 5120, vocab 248,320 |
| Params / bits per weight | 26,895,998,464 / **2.27** |

## -ngl sweep (llama-bench pp512 / tg64, r2)

The VRAM figures are whole-GPU `memory.used` peaks, out of 6,144 MiB. Before the run: RAM used 1,120 MiB, MemAvailable 6,662 MiB, swap 518 MiB.

| ngl | pp512 tok/s | tg64 tok/s | peak VRAM | max RAM used (anon) | min MemAvailable | max swap |
|---|---|---|---|---|---|---|
| 36 | 54.5 ± 10.8 | 3.61 ± 0.25 | 4,409 MiB | 1,433 MiB | 6,348 MiB | 715 MiB |
| 42 | 53.6 ± 11.1 | 4.14 ± 0.18 | 5,039 MiB | 1,451 MiB | 6,331 MiB | 751 MiB |
| 46 | 54.0 ± 8.1 | 4.41 ± 0.29 | 5,463 MiB | 1,324 MiB | 6,458 MiB | 796 MiB |
| **50** | **59.5 ± 9.7** | **5.05 ± 0.40** | 5,885 MiB | 1,333 MiB | 6,448 MiB | 822 MiB |
| 54 | — | — | 6,011 MiB | | | 866 MiB |

At ngl 54 context creation fails: `cudaMalloc` of the 262.5 MiB compute buffer runs out of memory. **The chosen ngl is 50 of 65.** At that setting the buffers are
CUDA0 model 5,376 MiB, CPU_Mapped model 1,886 MiB, KV 208 MiB and compute 262.5 MiB. Generation speed grows roughly linearly with the share of layers
on the GPU. Prompt processing stays around 55 tok/s because it is limited by the CPU-side layers.

## Smoke (ngl 50, c=4096, llama-completion)

- Load time: **116 s** (cold start, weights read from disk through mmap). Prompt eval was 1.0 tok/s on 9 tokens and generation **4.44 tok/s**.
- Max RSS was 7,012,536 kB. Most of that is mmapped file pages shared with the page cache, not anonymous memory: "used" stayed at ≤ 1.45 GiB.
- Peak VRAM was 6,055 MiB, leaving only about 90 MiB of headroom.
- Output: `1. Abstraction 2. Algorithms 3. Data Structures`, then it repeats the prompt (greedy raw completion without the chat template). This is coherent and the same
  pattern as the 9B. Chat-mode answers below are fully coherent.

## Hard-item subset (first 6 MATH-500 L4–5 + first 3 hard coding tasks, 1024-token cap, greedy, thinking off)

The server ran at ngl 50 with `-np 1` and c=4096. Server load took 3 min 35 s (cold mmap plus CUDA setup), and the eval wall time was 1,593 s (26.5 min). Median generation was **4.90 tok/s**
(range 4.87–5.10). There were 4 cap hits and 0 loops.

| item | 27B | 9B (E4a) |
|---|---|---|
| prealgebra/1139 | ✗ cap (1024 tok) | ✓ (also at the cap; the last-number fallback happened to be right) |
| number_theory/737 | ✗ cap (on track: 284 → 220 → …) | ✗ (280, an arithmetic slip) |
| algebra/1837 | ✓ | ✓ |
| algebra/2427 | ✗ cap (simulating balances year by year) | ✓ |
| counting_and_probability/525 | ✗ cap | ✗ cap |
| counting_and_probability/134 | ✓ | ✓ |
| hcode-lru_cache | ✓ | ✓ |
| hcode-dijkstra | ✓ | ✗ (tuple-unpacking bug) |
| hcode-calc | ✓ | ✗ (tokenizer infinite loop) |
| **total** | **5/9** (maths 2/6, code 3/3) | **5/9** (maths 4/6, code 1/3) |

All 4 of the 27B's maths misses ran out of tokens while reasoning coherently. It writes longer derivations than the 9B, and with thinking off and a 1024-token
cap that costs it on maths. It solved all three code tasks, including the two the 9B failed.

Memory during the eval: RAM "used" rose to 4,447 MiB because llama-server holds more anonymous memory than llama-bench. MemAvailable went down to **3,334 MiB**,
above the script's 3 GB abort line, and swap peaked at **873 MiB**. Swap grew by at most 370 MiB over the 518 MiB starting level during the whole run, well under the ~1.5 GB stop
threshold, and the machine stayed responsive. The mind was not touched.

## E4b success criterion (experiments/lowbit/README.md)

| criterion | result | verdict |
|---|---|---|
| Runs with part of its layers on the CPU, without swapping | 50/65 on the GPU, 15 on the CPU; swap +370 MiB at most, MemAvailable ≥ 3.3 GiB | **PASS** |
| ≥ 3 tok/s generation | 5.05 (bench), 4.44 (smoke), 4.90 median (eval) | **PASS** |
| Coherent | all answers coherent, no loops | **PASS** |
| At least as good as the 9B on the 6+3 subset | 5/9 vs 5/9 | **PASS (tie)** |

**E4b passes, narrowly.** A 27B model (7.6 GB of weights) runs on a 6 GB GPU plus 8 GB RAM at about 5 tok/s without swapping. That is 6× slower than the 9B,
and on 9 items it is not better: it trades maths that hits the cap for stronger code. Nine items cannot show a quality difference. The 1024-token cap with
thinking off penalises the 27B's longer reasoning, so a fair test needs a bigger cap, which at 5 tok/s means about 3.5 min per long answer.
The memory margin is thin: 90 MiB of VRAM at ngl 50 with c=4096, and 3.3 GiB MemAvailable while serving. Longer contexts would need a lower ngl (slower).

## Deviations

1. The script's eval subset failed at first: llama-server's default 4 slots needed a further 449 MiB recurrent-state cache on the GPU and ran out of memory at ngl 50.
   Logs are kept as `eval_hard_server_np4_OOM.log` and `eval_progress_np4_OOM.log`. I reran the same subset by hand with
   `eval_tasks.py ... --math-limit 6 --code-limit 3 --server-args "-ngl 50 -np 1 -fa 1 -ctk f16 -ctv f16 -b 2048 -ub 256"`. During the rerun I sampled
   GPU and memory myself (MARK `eval-np1` in `gpu.csv` and `mem.log`). One slot matches single-user use, and PrismML recommends it too.
2. The projected subset time was about 28 min (from the 9B's token counts at 4.4 tok/s). It was under 30 min, so MATH_N/CODE_N stayed at 6/3. The actual time was 26.5 min plus 3.6 min load.
3. The 27B download (7.63 GB) overlapped with the start of E4a, as the brief planned.
