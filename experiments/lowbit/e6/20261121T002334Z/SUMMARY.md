# E6 — fitting more 27B PTQ1_0 layers on the 6 GB GTX 1060 (KV q8_0/q4_0, -ub 128, ctx 2048)

Run: 2026-11-21 00:23–01:20 UTC, `experiments/lowbit/e6_run.sh` (this commit). Binaries are from the PrismML fork build `prism-d7b289683`
(tag prism-b12128). Model is `Ternary-Bonsai-2-27B-PTQ1_0.gguf` (5.95 GB, from E5). Flags are `-fa 1 -b 2048` plus the config's KV types, `-ub` and ctx, with mmap on.
The pinned build, config/ and the mind were not touched.

**Method.** For each config, I looked for the highest `-ngl` at which `llama-completion` survives a near-full prompt (3,212 tokens at ctx 4096, 1,621 at ctx 2048)
followed by 64 generated tokens. A near-full prompt is needed because the CUDA VMM pool grows with the prompt, and in E5 a short prompt passed where real use failed.
At that ngl I then ran `llama-bench` (pp512/tg64, r2) and `llama-server -np 1` with one near-full-context request. Peak VRAM is whole-GPU `memory.used`
sampled every 200 ms, against **6,144 MiB total**.

**Contention.** Other jobs ran on the laptop during this experiment: an ixOS Swift compile (finished about 00:50), a TimeBalance Polymarket ingest and other Claude
sessions. They caused 20–40 % I/O wait and evicted the model from the page cache. **The fit results are valid. The speeds are slightly pessimistic**, because the
3 CPU-side layers compete for CPU and memory bandwidth. Both configs that were measured in E5 as well came out within 1–2 % of E5, so the effect is small.

## Results (ctx 4096 unless noted)

| config | KV | -ub | ctx | **max ngl** (fit, near-full prompt) | completion tg / pp (tok/s) | peak VRAM, completion → spare | server `-np 1` tg / pp | server peak → spare | bench tg64 / pp512 |
|---|---|---|---|---|---|---|---|---|---|
| A (E5 baseline) | f16 | 256 | 4096 | **59** (60 → OOM) | 7.00 / 46.7 | 5,985 → **159** | **7.05** / 46.4 | 5,813 → 331 | 7.43 / 48.7 |
| B | q8_0 | 256 | 4096 | **61** (62, 63 → OOM) | 6.58* / 41.9 | 6,041 → 103 | 7.78 / 46.8 | 5,863 → 281 | 8.30 / 49.4 |
| **C** | **q8_0** | **128** | **4096** | **62** (63 → OOM) | **8.32** / 40.4 | 5,995 → **149** | **8.25** / 40.8 | 5,909 → **235** | **8.66** / 41.7 |
| D | q4_0 | 128 | 4096 | **63** | 8.54 / 40.7 | 6,013 → 131 | 7.81 / 41.2 | 5,927 → 217 | 9.26 / 42.0 |
| E | q8_0 | 128 | **2048** | **63** | 9.00 / 40.9 | 6,009 → 135 | 8.95 / 40.5 | 5,923 → 221 | 9.26 / 42.0 |

\*B's completion run was in pass 1, during the heaviest I/O contention. Its server (7.78) and bench (8.30) figures are the better guide.

Observations:
- **Each change buys about 1 layer.** q8_0 KV gives +2 layers (59 → 61). `-ub 128` gives +1 more (→ 62). q4_0 KV or ctx 2048 each give +1 more (→ 63). All 65 never fit.
- **q4_0 KV is slower at long context on Pascal.** D fits one more layer than C, but its server generation at a 3.2k-token context is 7.81 against C's 8.25, because flash attention has to dequantize
  q4_0 K/V. q8_0 does not show this. With an empty context D benchmarks at 9.26, so the cost grows with context length.
- **`-ub 128` costs prompt speed**: about 41 against 46–49 tok/s (−13 %).
- `llama-completion` consistently peaks 80–180 MiB higher than `llama-server` at the same settings, so it is the stricter fit test.

## Quality sanity (greedy, 64 tokens, ngl 58, ctx 4096, `-ub 256`)

| KV | output after "The three most important ideas in computer science are" | tg |
|---|---|---|
| f16 | `:\n1. Abstraction\n2. Algorithms\n3. Data Structures\n\nI think this is a good way to think about it.\n\nAbstraction is the idea that …` | 6.98 |
| q8_0 | `:\n1. Abstraction\n2. Algorithms\n3. Data Structures\n\nThe three most important ideas in computer science are: …` (repeats) | 7.01 |
| q4_0 | identical to q8_0, character for character | 6.92 |

All three are identical up to character 105, just after the list. That is the same near-tie token where PTQ1_0 and Q2_0 diverged in E5. After it, f16 continues with prose,
while q8_0 and q4_0 repeat the prompt, as Q2_0 did. This is one near-tie greedy token, not garbage. The q8_0 and q4_0 outputs are identical to each other. A real quality comparison
would need KLD or task scores at quantized KV, which I did not run here.

## Recommendation

**Config C is the setting to use: `-ngl 62 -fa 1 -ctk q8_0 -ctv q8_0 -b 2048 -ub 128 -c 4096 -np 1`.**

| | A: f16, ngl 59 (old robust) | **C: q8_0 + ub 128, ngl 62** | gain |
|---|---|---|---|
| server generation at 3.2k context | 7.05 tok/s | **8.25 tok/s** | **+17 %** |
| bench tg64 | 7.43 | 8.66 | +17 % |
| completion generation, near-full context | 7.00 | 8.32 | +19 % |
| prompt processing | 46 | 41 | −13 % |
| VRAM spare, server / completion | 331 / 159 MiB | **235** / 149 MiB | |

Against the 7.1 tok/s E5 baseline (ngl 58, f16), C is **+16–17 %, at about 8.3 tok/s**, and it clears 8 tok/s at a real 4096-token context.
- **The 150 MiB rule:** C meets it in the server (235 MiB spare). In llama-completion it falls 1 MiB short (149). If that margin matters, ngl 61 with the same flags should add about 80 MiB, at roughly 7.9 tok/s (not measured).
- **Shorter contexts:** for ≤ 2048, E (ngl 63) gives about 9.0 tok/s.
- **Avoid q4_0 KV.** It fits one more layer than C but is slower at long context.

**Against my estimate.** I had projected 9–9.5 tok/s robust and an 11 tok/s ceiling. The measured robust gain is smaller: about 8.3 tok/s at full context, and about 9.0 only at
ctx 2048. The memory tricks freed 3–4 layers as expected, but generation at a 3k-token context is slower than the empty-context benchmark numbers my linear model was fitted on.

## Deviations and failed attempts

1. `e6/20261120T234228Z-aborted-backtrace-hang/`: on CUDA OOM the fork's crash handler attached gdb/lldb for a backtrace and hung until the 300 s timeout.
   That run did record A at 59 = ok, 7.27 tok/s. The runner now exports `GGML_NO_BACKTRACE=1`.
2. `e6/20261121T000940Z-aborted-io-contention/`: under the other jobs' I/O load, model loads took about 4 min, so 240 s timeouts were recorded as failures (`fail_rc124`
   with peak VRAM 67 MiB means the model never even reached the GPU). In this directory, the pass-1 rows above the `--- pass 2` marker in `fit.csv` have the same problem,
   except B at 61, which is ok. Pass 2 used a 900 s timeout and re-tested every inconclusive point.
3. OOMs mostly show as `fail_rc134` (SIGABRT from the CUDA allocation abort). With the backtrace off, the log line my `oom` check looks for is not printed. All rc134 rows are OOMs.
4. B at 61 was measured in pass 1 (under heavy contention). B at 63 and 62 were re-tested in pass 2 (OOM).
