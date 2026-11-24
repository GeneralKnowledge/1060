# C0: on-device coding agent benchmark

Question: can a small low-bit model (Ternary-Bonsai 4B / 9B, Q2_0) work as a Python coding
agent, the kind that would run on an iPhone in ixOS Device Lab: write files, write tests,
run them, fix, and finish?

- `tasks.py`: 8 small build-with-tests tasks in three levels (1 = one function/class,
  2 = a small module with edge cases or sqlite, 3 = several parts, state, two files).
  Each has a hidden acceptance test the agent never sees, and a reference solution.
- `agent_bench.py`: the loop. Each reply is one JSON action
  (`write | read | run | test | done`), forced with llama-server's JSON-schema output.
  Code runs with plain `python3` in a temp folder (10 s per run, 30 s per test run).
  At most 14 steps per task, context 16k, thinking off. The score is the hidden test.
- `c0_run.sh`: runs both Bonsai models on the laptop GPU; refuses if the GPU is busy.

Self-check without a model: `python3 experiments/agent/agent_bench.py --fake --out /tmp/c0-fake`
(writes each reference solution and passes 8/8; hidden tests fail on an empty project).

Outputs per run: `summary.json` (hidden pass rate by model and level, said-done, own tests
passing, invalid actions, steps, tokens, time), `runs.jsonl` (per task, with final files),
`transcripts.jsonl`, `gpu.csv`.
