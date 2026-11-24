# E9 — 27B PTQ1_0 vs 9B PQ2_0+MTP on the hard set with a 2048-token cap

Run: 2026-11-24 13:41–14:37 UTC, `eval_tasks.py` at lab commit `a406b46`, PrismML fork build `prism-d7b289683`.

**Hard set:** the first 30 MATH-500 level 4–5 problems with integer answers, plus 8 hard coding tasks. Greedy decoding, thinking off, c=4096, **`--max-tokens 2048`** (E4 used 1024).

| run | model | server args |
|---|---|---|
| 9b | `Ternary-Bonsai-2-9B-PQ2_0-MTP-Q8_0.gguf` | `-ngl 99 -fa on -np 1 --spec-type draft-mtp --spec-draft-n-max 2` |
| 27b | `Ternary-Bonsai-2-27B-PTQ1_0.gguf` | `-ngl 62 -fa on -ctk q8_0 -ctv q8_0 -b 2048 -ub 128 -np 1` (E6 best) |

**The 27B run was stopped by the memory watchdog at 14:36:56 after 17 of 38 items** (all maths). The watchdog checks every 60 s and stops the run if MemAvailable < 300 MiB or swap grows by more than 1.5 GiB.
Swap had grown from 1,072 to 2,672 MiB, and MemAvailable fell to about 420–500 MiB from 14:30. Most of the pressure came from outside the run. Another thread's ixOS build was running
`dsymutil` (850 MB RSS, running for 54 min). A TimeBalance ingest and several Claude sessions were also running. The 27B itself had held MemAvailable at about 3.2 GiB until 14:15.
As briefed, I stopped and did not restart it. `mem.log` has the trace.

## Results

| | **9B PQ2_0 + MTP** (complete) | **27B PTQ1_0** (stopped at 17/38) |
|---|---|---|
| maths | **21/30** (70 %) | **16/17** run (13/14 in `eval_27b.jsonl` + 3 more ok in the progress log, see note) |
| code | 5/8 | not reached |
| total | 26/38 | — |
| 2048-cap hits | 6 (all maths) | 1 of 14 recorded |
| loops (a line repeated ≥ 5×) | 0 real (2 flagged: a repeated `$$` and `i += 1` in code) | 0 |
| median generation tok/s | **67.4** | **8.30** (7.98–8.39) |
| decode time | 533 s for 38 items | 1,549 s for 14 items |
| load | about 2 s (cached) | 4 min 38 s |

Grading uses the current strict grader. `rescore_math.py` changed nothing for either run.

**Head-to-head on the 17 items both models finished:**

| | 27B | 9B |
|---|---|---|
| correct | **16/17** | **14/17** |
| differs | number_theory/737 ✓ (9B: 280, a divisor missed); counting_and_probability/525 ✓ in 1,796 tokens (9B: hit the 2048 cap) | — |
| both wrong | prealgebra/1139 (both hit the 2048 cap) | |

Per item (first 14 from the jsonl; the last 3 from the progress log):

| item | 9B | 27B |
|---|---|---|
| prealgebra/1139 | cap (2048) | cap (2048) |
| number_theory/737 | ✗ 280 (818 tok) | **✓** (1,094 tok) |
| algebra/1837 | ✓ | ✓ |
| algebra/2427 | ✓ | ✓ |
| counting_and_probability/525 | cap (2048) | **✓** (1,796 tok) |
| counting_and_probability/134, /119; geometry/627; algebra/305, /187; prealgebra/1388, /951; number_theory/1055; precalculus/675 | ✓ | ✓ |
| intermediate_algebra/279; prealgebra/1233; precalculus/1313 | ✓ | ✓ (progress log) |

The 9B's other misses fall in the part the 27B never reached: intermediate_algebra/1849, /582 and /345 and geometry/283 (all cap hits); algebra/892 and prealgebra/1646 (finished, wrong); and code dijkstra, calc and interval_rooms.

## Verdict

- **Not enough to call it.** The 27B was +2 on 17 items (16 vs 14). At n=17 the noise is about ±11 points (about 2 items), so this is a hint, not a result. The full 38-item comparison did not run.
- The hint points the expected way. The 27B fixed one arithmetic slip and finished one problem the 9B could not finish within 2048 tokens, while being about as verbose (12,785 vs 11,644
  tokens on the same 14 items).
- **Raising the cap from 1024 to 2048 mattered for the 9B.** On the same 30 maths items it scores 21/30 here (6 cap hits) against 17/30 with the strict grader at 1024 in E4 (Q2_0 file). PQ2_0 and Q2_0
  give identical text, so the difference comes from the cap and the MTP setting, not the packing.
- **Cost:** the 27B is 8.1× slower to decode (8.3 vs 67 tok/s). On this set an average answer takes about 2 min on the 27B against about 14 s on the 9B.
- To finish the comparison, the 27B needs the remaining 21 items (about 50 min of decoding plus a 5-min load) on a machine with **about 3 GiB of RAM free for the whole run**, so no concurrent Swift or
  dsymutil builds. Running it in two chunks (`--math-limit` / `--code-limit`) would also bound the run time.

## Notes

1. Both `eval_tasks.py` runs ended non-zero. For the 9B, llama-server took more than 30 s to exit after SIGTERM, so `proc.wait(timeout=30)` raised (the same as E8) and `eval_9b_summary.json` is missing.
   All 38 items are in `eval_9b.jsonl`. The leftover 9B server held 68 MiB of VRAM while the 27B was loading, so I killed it with SIGKILL (it was my own process).
   For the 27B, the watchdog killed the run (exit 143).
2. `eval_27b.jsonl` has 14 lines, but the progress log shows 17 finished items. The last 3 rows were still in the file buffer when the process was killed. Their verdicts (all ok) come
   from `eval_progress_27b.log`, which uses the same grader at run time. A small follow-up would be to flush `eval.jsonl` after each row.
