# E4a — harder eval: Qwen3.5-4B Q4_K_M vs Bonsai-2-4B Q2_0 vs Bonsai-2-9B Q2_0

Run: 2026-11-20 17:58–18:40 UTC, `experiments/lowbit/e4_run.sh hard` at lab commit `d6fb81f`, pinned CUDA build `d2db58ba90808f32`
(llama.cpp `7bd180830`), full GPU offload, lab flags `-ngl 99 -fa 1 -ctk f16 -ctv f16 -b 2048 -ub 256`.
The eval is greedy, with thinking off, **1024-token cap** and c=4096.
- Maths: MATH-500 levels 4–5 with integer answers, first 30 (`math500-l45-int-first30.jsonl`).
- Code: 8 harder tasks with unit tests (`code-tasks-hard-v0.jsonl`).

The 27B download ran concurrently from 17:58 to about 18:25, and only for the first part of the run. Decode speeds match E2/E3, so it had no visible effect.

## Results

| model | maths | code | total | answers at the 1024 cap (of 38) | real loops | median gen tok/s | wall |
|---|---|---|---|---|---|---|---|
| Qwen3.5-4B Q4_K_M | 15/30 (50 %) | **6/8** | 21/38 | 14 | 0 | 28.5 | 1013 s |
| Bonsai-2-4B Q2_0 | 14/30 (47 %) | 5/8 | 19/38 | 15 | 0 | **48.8** | **556 s** |
| Bonsai-2-9B Q2_0 | **17/30 (57 %)** | 5/8 | **22/38** | **12** | 0 | 28.2 | 977 s |

All cap hits are maths items, with none in code. The script's "line repeated ≥ 5×" check flags 1 / 1 / 2 answers, but these are false positives:
repeated `$$` or `\[` LaTeX delimiters, and `i += 1` inside generated code. None is a real loop.

## What the set actually measures

The cap dominates. **Every model's maths misses except one are answers that ran out of the 1024 tokens**:
- Q4_K_M: 15 misses, 14 at the cap.
- Bonsai-4B: 16 misses, 15 at the cap.
- Bonsai-9B: 13 misses, 12 at the cap.

**Rescored 2026-11-24 (PR #20 review):** maths is now graded only from the last `Answer:` line, with no fallback to the last number in the text, and fractions are evaluated. Answers that ran out of tokens before an `Answer:` line now count as wrong. Before rescoring, a lenient last-number fallback credited a few cap-hit answers (maths was 17 / 16 / 18).

Seven items (intermediate algebra 279/1849/582/362/345/232 and geometry 283) hit the cap for **all three** models. Without thinking mode and with a
1024-token budget, these level 4–5 problems mostly test whether the model finishes in time. The 9B hits the cap least, which is where its +2–3 comes from.

Only one maths answer per model was finished but wrong:
- Q4_K_M, prealgebra/1646: answered angle 110 (want 80).
- Bonsai-4B, algebra/873: sign error, gave c = −35 (want 33).
- Bonsai-9B, number_theory/737: summed the proper divisors of 224 and left out one of them (280, want 284).

Code failures (all finished; none at the cap):
- **Q4_K_M:** interval_rooms gives the wrong room count; sudoku_valid rejects a valid board.
- **Bonsai-4B:** lru_cache fails to update a value on `put`; calc fails to parse a valid expression; sudoku_valid rejects a valid board.
- **Bonsai-9B:** dijkstra treats the adjacency list's (v, w) tuples wrongly ("too many values to unpack"); calc has an infinite loop in the tokenizer
  (10 s timeout); interval_rooms gives the wrong room count.

Each model solves a different subset of the code tasks, and 6 of the 8 tasks split the models.

## E4a success criterion (experiments/lowbit/README.md)

| criterion | result | verdict |
|---|---|---|
| Spread ≥ 15 points between the three models | maths 53–60 % (7 points); total 55–61 % (5 points) | **FAIL** |
| Expected order 9B > Q4_K_M > Bonsai-4B | maths 17 > 15 > 14 (matches, after rescoring); total 22 > 21 > 19 | order matches on maths, but the gaps are 1–2 items each, well inside noise (±9 points at n=30) |

**E4a fails to separate the models.** The harder set moved all three from about 90 % to about 57 %, but they moved together. The remaining gap is
1–2 items, and most of it comes from finishing within the 1024-token cap rather than from being right more often. To separate them, a next eval
would need a larger cap (2048+, or thinking on with a reasoning budget) so that cap hits stop dominating, and more items (≥ 100) to bring the noise
below the size of the effect. On this hardware Bonsai-4B remains the speed pick (1.7× decode), and on quality the 9B ≈ Q4_K_M.
