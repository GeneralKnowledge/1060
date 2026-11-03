# The continuous mind (milestone A)

Design and plan: [Continuous mind: milestone A plan](https://claude.ai/artifact/CyT5UijS7eJ42CzU1XeLXR).

One long-lived process (`python3 -m mind`) around a persistent shared state in `mind/state/mind.sqlite`.
It has an identity from its first boot that is never regenerated, an append-only life stream,
episodic memory, beliefs with evidence, interruptible thoughts, a workspace, drives (arousal, fatigue,
curiosity), five overlapping states (awake, background, resting, dreaming, deep sleep) and settings it
tunes for itself within hard limits.

The code holds mechanisms only. There is no persona, no seeded interests or beliefs, and the world it
lives in is generated from a random seed over made-up symbols. Interests follow its own learning
progress; thoughts start only on what wins its workspace.

## Status

A1 (core without language models) is done. A0 (`scripts/mind/a0_feasibility.py`) measures the
LFM2.5 tiers and Qwen on the laptop. A2 to A6 follow the plan.

## Commands

```bash
python3 -m mind run --fast 30000 --state /tmp/m   # a quick life on a generated world (about 12 s)
python3 -m mind report --state /tmp/m             # what it learned vs the world's hidden rules
python3 -m mind run                               # live in real time (5 s ticks) with real signals
python3 -m mind status                            # identity, state, workspace, recent life stream
python3 -m mind stop                              # clean shutdown (or: touch mind/STOP)
```

Real time runs in tmux session `mind`. It uses no GPU, no network and no Claude calls in A1, and
runs no commands except `nvidia-smi` to read its own body temperature.

## First quick run (cloud, 2026-11-03)

Three minds, each in its own generated world, 30,000 ticks each (about 42 hours at 5 s per tick, run
in about 12 seconds):

| | delayed causes learned | cycles found | mean surprise first to last window (bits) | dream vs random-pair pass rate |
|---|---|---|---|---|
| mind 1 | 4 of 5 | 3 of 3 | 2.52 to 0.91 | 0.027 vs 0.032 |
| mind 2 | 4 of 4 | 3 of 3 | 2.03 to 0.43 | 0.035 vs 0.038 |
| mind 3 | 2 of 3 | 3 of 3 | 2.55 to 0.86 | 0.034 vs 0.065 |

Each also doubted a belief when its world's switching rule changed, and kept other true predictive
regularities that are consequences of the rules. Algorithmic dreaming is no better than random
recombination yet; that is the bar for the language tiers in A2.
