# E5 — PrismML llama.cpp fork + Ternary-Bonsai-2-27B PTQ1_0 (1.75 bpw) on the GTX 1060

Run: 2026-11-20 22:09–23:30 UTC. Lab commit `03e3ba9` (includes the `-np 1` eval fix). The pinned build `d2db58ba90808f32` was not touched.
The sweep folder written by `e4_run.sh offload` was moved here as `offload_sweep/`.

## Fork build

| | |
|---|---|
| Source | https://github.com/PrismML-Eng/llama.cpp, newest release tag `prism-b12128-d7b2896` = commit `d7b28968306207fd176e7ff10796ccb6be607865` (2026-10-30), cloned to `~/1060/backends/src/prism-llama.cpp` |
| Build dir | `~/1060/backends/builds/prism-d7b289683` |
| Configure | `CUDACXX=/usr/lib/nvidia-cuda-toolkit/bin/nvcc CUDAHOSTCXX=g++-12 cmake -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=61 -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF` (CUDA 12.0.140) |
| Targets | llama-bench, llama-server, llama-completion, test-backend-ops, at `-j3` |
| Build time | **1,060 s (17.7 min)**, clean, 4 compiler warnings, no RAM pressure |
| Version | `0.7.0-dev (build 12128, commit d7b289683)` |

The PTQ1_0 CUDA path in the source has a generic `vec_dot_ptq1_0_q8_1` kernel, which uses dp4a and so works on Pascal. It also has a faster mmvq path that is gated
to Turing or newer (`mmvq.cu:329`), which the 1060 does not get.

## Kernel tests (`test-backend-ops -b CUDA0`, fork binaries)

| type | MUL_MAT ok / total | GET_ROWS | status |
|---|---|---|---|
| ptq1_0 | **185 / 185** | 12 OK | **pass** |
| pq2_0 | **260 / 260** | 8 OK | **pass** |
| q2_0 | 82 / 82 | 8 OK | **pass** |
| tq1_0 | 0 / 13 | 8 not supported | not supported on CUDA |
| tq2_0 | 0 / 48 | 8 not supported | not supported on CUDA |

## Model

`prism-ml/Ternary-Bonsai-2-27B-gguf` @ `839bd57023c1558945afea70a74fa83c054f35f9`, `Ternary-Bonsai-2-27B-PTQ1_0.gguf`, 5,946,648,928 bytes.
sha256 is `53107f530aa52eb00912263ab1ee29bd199261c87cd7b4ad4ca1318c1fe33ee3` (matches HF). llama-bench reports it as "qwen35 27B PTQ1_0 - 1.75 bpw ternary (group 128)",
with model_size 5,935,527,936 B for 26.9B params (**1.77 bits per weight overall**). Q2_0 is 7.61 GB and 2.27 bits per weight.

## Speed sweep, fork binaries (llama-bench pp512 / tg64, r2, `-fa 1 -ctk f16 -ctv f16 -b 2048 -ub 256`, mmap on)

Before the run: RAM used 1,059 MiB, MemAvailable 6,722 MiB, swap 614 MiB.

| model | ngl | pp512 tok/s | tg64 tok/s | peak VRAM | max RAM used | min MemAvailable | max swap |
|---|---|---|---|---|---|---|---|
| Q2_0 (fork baseline) | 50 | 59.4 ± 7.3 | **5.16 ± 0.07** | — | — | — | — |
| PTQ1_0 | 50 | 40.9 ± 2.1 | 4.91 ± 0.01 | 5,027 MiB | 1,419 | 6,362 | 823 |
| PTQ1_0 | 54 | 47.8 | 5.97 ± 0.11 | 5,357 MiB | 1,500 | 6,281 | 642 |
| PTQ1_0 | 58 | 48.6 | 7.13 ± 0.08 | 5,689 MiB | 1,504 | 6,277 | 639 |
| PTQ1_0 | **62** | 49.6 | **8.77 ± 0.01** | 6,023 MiB | 1,511 | 6,270 | 639 |
| PTQ1_0 | 65 | — | — | CUDA OOM (abort in VMM alloc) | | | |

- The fork's Q2_0 at ngl 50 (5.16 tok/s) matches the pinned build's 5.05 within noise, so the engines are comparable.
- At equal ngl, PTQ1_0 is slightly *slower* than Q2_0: 4.91 vs 5.16 generation and 41 vs 59 prompt tokens/s. The Pascal kernel for the 1.75-bit packing costs more per weight.
- The gain comes only from the 1.7 GB smaller file. More layers fit on the GPU, so 62 instead of 50 of 65, and generation rises about linearly with the GPU share.
- Swap never grew more than about 200 MiB, and RAM "used" stayed at or below 1.5 GiB during the benchmarks.

## Fit at real context (c=4096)

llama-bench only allocates a context of about 576 tokens. At c=4096:

| tool | ngl 62 | ngl 60 | ngl 58 |
|---|---|---|---|
| llama-completion (smoke, 1 seq) | **OOM** (`cuMemCreate`, during the first prompt) | **OOM** | **OK**: 7.08 tok/s, load 5.5 s (warm), peak VRAM 5,899 MiB |
| llama-server `-np 1` (eval) | **OK**: 8.64–8.68 tok/s, peak VRAM **6,061 MiB** (about 80 MiB free), MemAvailable ≥ 4.8 GiB, swap 639 MiB | not tried | not tried |

The server fits at 62 where llama-completion does not, because the completion run reserves more VRAM for its first prompt. Either way, ngl 62 has essentially no
headroom. **ngl 58 is the robust choice** (about 245 MiB free at c=4096, 7.1 tok/s), and ngl 62 works in the server for c ≤ 4096 and one user only.

## Equality check (greedy, 64 tokens, chat template off, fork binaries)

Prompt: "The three most important ideas in computer science are"

| | output |
|---|---|
| PTQ1_0, ngl 58 | `:\n1. Abstraction\n2. Algorithms\n3. Data Structures\n\nI think this is a good way to think about it.\n\nAbstraction is the idea that you can hide complexity behind a simple interface. For example, a function is an abstraction: you call it with some inputs and it gives you some` |
| Q2_0, ngl 50 | `:\n1. Abstraction\n2. Algorithms\n3. Data Structures\n\nThe three most important ideas in computer science are:\n1. Abstraction …` (repeats the prompt) |
| Q2_0, pinned build (E4) | byte-identical to the fork's Q2_0 output |

The outputs are identical for the first ~20 tokens and diverge at byte 106, after the list. Both are coherent; PTQ1_0 continues with prose instead of repeating the prompt.
The hard-item answers below agree more strongly: on the same items the PTQ1_0 and Q2_0 transcripts contain the same derivations, the same intermediate values
(220 → σ(220) for 737, the 3 = 3 ratio check for 1837) and the same final answers. So the weights behave the same, and the text differs only by small numeric
differences between kernels.

## Sanity items (3 MATH-500 + 1 code, 1024-token cap, server ngl 62 `-np 1`, 380 s)

| item | PTQ1_0 27B | Q2_0 27B (E4b) | Bonsai-9B (E4a) |
|---|---|---|---|
| prealgebra/1139 | ✗ cap | ✗ cap | ✓ (cap + fallback) |
| number_theory/737 | ✗ cap (same path as Q2_0) | ✗ cap | ✗ |
| algebra/1837 | ✓ (805 tok) | ✓ (806 tok) | ✓ |
| hcode-lru_cache | ✓ | ✓ | ✓ |
| total | 2/4 | 2/4 | 3/4 |

Median generation was **8.68 tok/s**, with 2 cap hits and no loops.

## Verdict

| question | answer |
|---|---|
| Does PTQ1_0 run on Pascal? | **Yes.** The fork builds with CUDA 12.0 for sm_61, the kernels pass 185/185, and the output is coherent and matches Q2_0's. |
| Meaningfully faster than Q2_0's 5.1 tok/s? | **Yes, through fit rather than kernel speed.** The same-engine Q2_0 is 5.16 tok/s at its best ngl (50). PTQ1_0 gives 7.1 tok/s at ngl 58 (+38 %, robust) and 8.7 tok/s at ngl 62 (+70 %, server with one slot at c=4096, about 80 MiB VRAM free). |
| Target ≥ 8 tok/s | **Met only at ngl 62** (8.77 bench, 8.68 in the server eval), which is at the VRAM limit. At the robust setting (ngl 58) it is 7.1 tok/s, **below target**. |

PTQ1_0 is the better 27B format on this card: it uses 22 % less disk and RAM, is up to 1.7× faster to generate, and its outputs match. The ≥ 8 tok/s target is
only reached with no VRAM margin, so the remaining gap would have to come from a smaller context, a quantized KV cache or the Turing-only mmvq path, which this
GPU can't use. Prompt processing (41–50 tok/s) is slower than Q2_0's 59 because the generic Pascal PTQ1_0 kernel is more expensive per weight.

## Deviations

1. llama-completion ran out of VRAM at ngl 62 inside `e4_run.sh`, so its `offload_sweep/smoke.*` files record that failure. The script still went on to the
   eval subset, where the server at ngl 62 with one slot worked. I then ran smoke tests by hand at ngl 60 (OOM) and 58 (OK): `smoke_ptq1_0_ngl{60,58}.*` and `vram_smoke_ngl*.csv`.
2. The Q2_0 baseline and equality check used the fork's binaries (`bench_fork_q2_0_ngl50.*`, `smoke_q2_0_ngl50.*`). The Q2_0 smoke test reloaded the file cold (load 255 s)
   because the PTQ1_0 runs had pushed it out of the page cache.
3. GGUFs are in `~/1060/models/` (git-ignored).
