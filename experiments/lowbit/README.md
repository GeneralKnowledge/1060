# Low-bit experiments

Experiments for the extreme low-bit mission. Design and hypotheses: [`EXTREME_COMPRESSION_RESEARCH.md`](../../EXTREME_COMPRESSION_RESEARCH.md).
They call the pinned llama.cpp build directly and never touch `config/`, `winners/` or the benchmark lock.

| Experiment | Script | Hypothesis / success criterion |
|---|---|---|
| E1 (done, all criteria passed; quality below Q4_K_M) | `e1_run.sh MODEL.gguf` | Q1_0/Q2_0 kernels pass test-backend-ops on Pascal; Ternary-Bonsai-2-4B Q2_0 runs fully on the GPU with tg128 ≥ 38 tok/s, VRAM ≤ 1.8 GB and coherent answers, against Qwen3.5-4B Q4_K_M (29.3 tok/s, 3.0 GB) |
| E2 | `e2_run.sh BONSAI.gguf` | At the same ~1.2 GB, trained ternary (Bonsai Q2_0) beats post-training 2-bit of the same base (Qwen3.5-4B IQ2_XXS, imatrix) on GSM8K-50 + 10 coding tasks; IQ2_M and Q4_K_M give the curve. Success: Bonsai ≥ IQ2_XXS + 10 points on GSM8K, and within 15 points of Q4_K_M |

`gsm8k-test-first50.jsonl` (first 50 of the GSM8K test split, MIT) and `code-tasks-v0.jsonl` (10 tasks with unit tests) are scored by `eval_tasks.py`.
`prompts-v0.jsonl` is the fixed cross-model prompt set (greedy, thinking off), run by `run_prompts.py`.
Raw outputs go to `e1/<UTC timestamp>/` (env, kernel logs, smoke logs, llama-bench JSON, perplexity/KLD,
GPU telemetry at 0.5 s, prompt transcripts).
