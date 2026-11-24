# E7 — stacking: PrismML packing + E6 memory settings + speculative decoding

Run: 2026-11-24 10:36–12:06 UTC, `experiments/lowbit/e7_run.sh` + `e7_spec.py` (this commit), PrismML fork build `prism-d7b289683`
(prism-b12128). The pinned build, config/, winners/, the mind and the other projects' jobs were not touched.

**Workload.** Every variant ran the same workload: `llama-server -np 1 -c 4096 --jinja`, greedy (temperature 0, seed 1), thinking off, `max_tokens` 256.
There were 6 prompts: the first 3 GSM8K-50 items, the first 2 hard MATH-500 items and the first hard code task (lru_cache), from `experiments/lowbit/*.jsonl`.
Speeds are llama-server's own `predicted_per_second`. "agg" is total tokens / total generation time over the 6 prompts.
Peak VRAM is whole-GPU `memory.used` (6,144 MiB total). B2 variants first had to survive a 3,212-token "fit" prompt.

**Downloads** (as briefed; both match the HF LFS sha256):
- `prism-ml/Ternary-Bonsai-2-9B-gguf` @ `82265ccc78ddaac4700286a35e9badcc32376dd9`, `Ternary-Bonsai-2-9B-PQ2_0-MTP-Q8_0.gguf`, 2,675,410,720 B, sha256 `c79fcdfaf6b7bf47089b50dadcfd11f9ce7d3e96adf3bbefde7912e5ef69e22b`
- `prism-ml/Ternary-Bonsai-2-2B-gguf` @ `a6032a5089ac27e96be3a80c7b42d1ad0bd89887`, `Ternary-Bonsai-2-2B-PTQ1_0.gguf`, 429,183,456 B, sha256 `36061ad406df8a4903a6db005be350c50fabfaf717f9ef0acf4b76adcd9929fa`

**Contention.** An ixOS `swift-build`, a TimeBalance Polymarket ingest and other Claude sessions ran throughout. The load average reached 10 and swap was used.
The 9B results were unaffected: the model stays cached and loads in about 2 s. The 27B loads took **6–15 min** (vs about 1 min when quiet), and the 27B speeds are slightly pessimistic
(B0 7.6 tok/s here vs 8.25 in E6).

## Part A — Ternary-Bonsai-2-9B (fully on the GPU, `-ngl 99`)

| variant | model / flags | gsm8k | hard maths | code | **agg tok/s** | draft acceptance | peak VRAM | same text as baseline |
|---|---|---|---|---|---|---|---|---|
| A0 | Q2_0, f16 KV, ub 256 | 32.3 | 32.0 | 32.0 | **32.1** | — | 2,549 | (baseline) |
| A3a | A0 + `--spec-type ngram-mod` | 30.5 | 30.5 | 31.7 | 30.8 | 3/256 = 1 % | 2,573 | 5/6 vs A0 |
| A1n | **PQ2_0-MTP file, no spec**, f16 KV | 38.7 | 38.3 | 38.4 | **38.5** | — | 2,431 | **6/6 vs A0** |
| **A1m2** | A1n + `--spec-type draft-mtp --spec-draft-n-max 2` | 71.9 | 71.7 | 77.0 | **72.5** | 784/900 = **87 %** | 2,847 | 5/6 vs A1n |
| A1m3 | A1n + draft-mtp n-max 3 | 64.8 | 63.6 | 70.8 | 65.1 | 862/1107 = 78 % | 2,897 | 4/6 vs A1n |
| A2n | PQ2_0-MTP, q8_0 KV, no spec | 37.8 | 37.7 | 37.8 | 37.8 | — | 2,371 | 3/6 vs A1n |
| **A2m2** | A2n + draft-mtp n-max 2 | 71.5 | 70.7 | 75.9 | **71.8** | 785/898 = 87 % | 2,787 | 4/6 vs A2n; **6/6 vs A1m2** |
| A2m3 | A2n + draft-mtp n-max 3 | 61.4 | 61.3 | 70.2 | 61.9 | 852/1145 = 74 % | 2,837 | 5/6 vs A2n |
| A3b | A2n + `--spec-type draft-mtp,ngram-mod` n-max 2 | 65.1 | 64.4 | 76.0 | 66.5 | 782/1144 = 68 % | 2,811 | 4/6 vs A2n |

- **The PQ2_0 packing alone is +20 %** (32.1 → 38.5 tok/s) and gives **identical text** to Q2_0 on all 6 prompts, because they hold the same ternary weights. It also uses 120 MiB less VRAM.
- **The built-in MTP head roughly doubles it: 72.5 tok/s, 2.26× the Q2_0 baseline.** 87 % of drafted tokens are accepted.
  Drafting 2 tokens beats 3, because the third guess is accepted less often and the extra verification costs more than it saves.
- q8_0 KV does nothing for speed on the 9B (it already fits with 3 GB spare). It saves 60 MiB.
- N-gram speculation does not help on this workload (about 1 % acceptance; maths and fresh code have little verbatim repetition). Stacked on top of MTP it *costs* 8 %, because its
  rejected drafts lower the combined acceptance.

## Part B — Ternary-Bonsai-2-27B PTQ1_0 (`-fa 1 -ctk q8_0 -ctv q8_0 -b 2048 -ub 128`)

| variant | flags | result | agg tok/s | acceptance | peak VRAM | same text as B0 |
|---|---|---|---|---|---|---|
| **B0** | `-ngl 62` (E6 config C) | ok | **7.6** (6.5–8.1; E6 quiet: 8.25) | — | 5,909 | (baseline) |
| B1 | B0 + ngram-mod | **CUDA error (OOM)** on prompt 2 | 7.3 (1 prompt) | — | 5,935 | 1/1 |
| B2cpu3 | `-ngl 62` + 2B draft `-ngld 0` (draft on the CPU), n-max 3 | **CUDA error (OOM)** on the fit prompt | — | — | 6,011 | — |
| B2g56n3 | `-ngl 56` + 2B draft `-ngld all -ctkd q8_0 -ctvd q8_0`, n-max 3 | **CUDA error (OOM)** on the fit prompt | — | — | 5,975 | — |
| B2g54n3 | `-ngl 54` + same draft | **CUDA error (OOM)** on the fit prompt | — | — | 5,987 | — |
| B2g52n3 | `-ngl 52` + same draft | ok (survives the 3,212-token prompt) | **3.3** | 665/1767 = **38 %** | 6,007 | 5/6 |
| B2g52n5 | `-ngl 52`, n-max 5 | **skipped** | | | | |

- **No speculative setup beats B0 on the 27B.** Every speculative variant needs extra GPU memory for the verification batch and the draft's own buffers.
  At 62 GPU layers there is no room left. Even with the draft on the CPU, the server still allocates GPU compute buffers for the draft path, and it runs out of memory.
- The first layer count where the 2B draft fits fully on the GPU is 52. There the 27B alone would run at only about 4.9 tok/s (E5 sweep), and the 2B draft is accepted only 38 % of the time.
  That is too low to recover even that, so the result is 3.3 tok/s, **2.3× slower than B0**.
  The 2B's guesses agree with the 27B far less often than the 9B's own MTP head agrees with the 9B: 38 % vs 87 %.
- I skipped B2g52n5. At 38 % acceptance a longer draft can only add rejected work, and each 27B reload took about 12 min on the busy laptop.
- I did not try the 4B (Q2_0, 1.2 GB) as a draft. It is 3× the 2B's size, so it would force the 27B even lower than 52 layers, and the 2B already lost.

## Losslessness (greedy)

The speculative outputs are **not bit-identical** to plain decoding. In each speculative variant, 1–2 of the 6 answers diverge once, at a near-tie token, and continue with a
different but coherent wording. Two examples:
- "4 multiplication operators and 1 addition operator" vs "4 operators: $2·3·4·5+1$".
- "**Calculate total eggs used:**" vs "**Calculate remaining eggs:**".

Plain decoding with q8_0 instead of f16 KV diverges at the same spots (A2n vs A1n: 3/6 identical). So this is batch- and precision-dependent floating-point rounding at near-ties, not
quality loss from speculation: the verification step evaluates several tokens in one batch, which rounds slightly differently. Two things are exactly reproducible:
- A1m2 and A2m2 (the same speculation with f16 vs q8_0 KV) produce identical text on all 6 prompts.
- Q2_0 and PQ2_0 (A0 vs A1n) produce identical text on all 6 prompts.

## Best stacked settings

**9B (recommended for everyday use):**
```
llama-server -m Ternary-Bonsai-2-9B-PQ2_0-MTP-Q8_0.gguf -ngl 99 -fa 1 -ctk q8_0 -ctv q8_0 -b 2048 -ub 256 -np 1 -c 4096 \
  --spec-type draft-mtp --spec-draft-n-max 2
```
This gives about **72 tok/s** (vs 29.8 for the lab's Qwen3.5-4B Q4_K_M and 32 for the 9B Q2_0) at a 2.8 GB VRAM peak, with about 3.3 GB free for longer contexts.
f16 KV gives the same speed and 60 MiB more VRAM.

**27B:** unchanged from E6, with no speculation:
```
llama-server -m Ternary-Bonsai-2-27B-PTQ1_0.gguf -ngl 62 -fa 1 -ctk q8_0 -ctv q8_0 -b 2048 -ub 128 -np 1 -c 4096
```
This gives about 8.25 tok/s when the laptop is quiet. Speculation would need a 27B with a built-in MTP head, which PrismML does not publish (the 9B and 4B have one). It would also need
a few hundred MiB of VRAM that this card does not have at 62 layers.

## What didn't work / notes

1. B1 (n-gram on the 27B): CUDA OOM at ngl 62. I didn't retry it at a lower ngl, because n-gram gave no benefit even where it fit (the 9B).
2. 27B + 2B draft: OOM at ngl 62 (draft on the CPU), 56 and 54. It runs at 52 but is slower than plain decoding.
3. The server log only records `ggml-cuda.cu:111: CUDA error` (with `GGML_NO_BACKTRACE=1` the message text is not printed). OOM is inferred from the context: it happens on the first
   allocation-heavy request at the VRAM limit, and llama.cpp's own fit check warned "failed to fit params to free device memory" at load.
4. My E7a proposal (Bonsai-4B as a CPU draft) was folded into this brief's design. The 2B draft on the CPU already failed on VRAM, and the 4B would not change that.
