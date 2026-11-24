# E8 — quality check: Bonsai-2-9B PQ2_0 + built-in MTP head (the E7 best setting) vs E3 (9B Q2_0)

Run: 2026-11-24 13:28–13:41 UTC. `eval_tasks.py` at lab commit `a406b46` (grader tests 4/4 pass), PrismML fork build `prism-d7b289683`.
The task sets are the same as E3: GSM8K test first 50 + 10 coding tasks, greedy, thinking off, 512-token cap, c=4096.
Server: `-ngl 99 -fa on -ctk f16 -ctv f16 -b 2048 -ub 256 -np 1 --spec-type draft-mtp --spec-draft-n-max 2`,
model `Ternary-Bonsai-2-9B-PQ2_0-MTP-Q8_0.gguf` (sha256 c79fcdfa…, from E7).

The run started only after another thread's C0 agent benchmark had released the GPU (queued; nothing of that job was touched).

## Results (maths graded with the current strict grader; E3 rescored with it)

| | **E8: 9B PQ2_0 + MTP n=2** | E3: 9B Q2_0, no speculation |
|---|---|---|
| GSM8K-50 | **46/50** (92 %) | 47/50 (94 %); rescore unchanged (47) |
| code-10 | **10/10** | 10/10 |
| 512-token cap hits (of 60) | 3 | 2 |
| loops (a line repeated ≥ 5×) | 0 | 0 |
| median generation tok/s | **61.4** | 28.5 |
| total decode time (sum of tokens / tok/s) | **263 s** | 497 s |
| eval wall time | 615 s (plus a 58 s load) | 543.5 s |
| generated tokens | 14,199 | 14,114 |
| draft acceptance | 85–96 % per answer (mean accepted length about 2.7–2.9) | — |

- **Quality: no measurable change.** 46 vs 47 is one item, well inside the ±4-point noise of n=50.
- 45 of 60 answers are byte-identical to E3. The rest diverge at near-tie tokens, as seen in E7, because the batched verification rounds slightly differently.
- **Speed:** decode is 2.15× faster (median 61 vs 28.5 tok/s; decode time 263 vs 497 s). The median is below E7's 72 because these answers are longer and acceptance varies more on them.
- **Wall time did not drop.** About 350 s of the E8 wall time was spent outside decoding, against about 46 s in E3. Some prompt evaluations ran at about 40 tok/s instead of about 240 (for example, 2.6 s for a
  105-token prompt). This was on a busy laptop (load average 6–8 from other jobs), and I did not isolate the cause. Decode speed is the reliable number here. A wall-clock
  rerun on a quiet machine would settle it.

## Items whose verdict differs from E3

| id | E8 | E3 | want |
|---|---|---|---|
| gsm8k-test-8 | ✗: hit the 512 cap one step from the answer (`Final Distance = 180 - 135` …) | ✓ 45 | 45 |

The other misses are the same in both runs: g21 (answers 2, want 14; the same age misreading), and g37 and g45 (both hit the cap in both runs).

## Notes

- `eval_tasks.py` exited 1 after all 60 items were written. llama-server took longer than 30 s to exit after SIGTERM (`proc.wait(timeout=30)` raised), so `eval_summary.json`
  was not written. The numbers above come from `eval.jsonl`. The server did exit shortly afterwards, so it did not block E9.
- Verdict: the E7 setting keeps E3's quality (46/50 vs 47/50, 10/10 code) at about 2.15× the decode speed. It is a good candidate for the everyday 9B setting.
