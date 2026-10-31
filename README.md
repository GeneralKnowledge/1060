# 1060 Research Lab

An autonomous research environment for extracting the best practical AI-inference performance from one specific
NVIDIA GTX 1060 laptop (Dell G3 17, Ubuntu Server, headless). Inspired by Karpathy's *autoresearch*: a small
experiment loop, measurable results, and a knowledge base that separates verified facts from hypotheses.

**Status:** M0 to M4 are built and tested against fake llama.cpp binaries. Nothing has run on the
laptop yet. See [`docs/assessment.md`](docs/assessment.md) for the technical assessment and
[`docs/running.md`](docs/running.md) for the steps to run it on the hardware.

| Milestone | What exists |
|---|---|
| M0 hardware discovery | `scripts/hardware_audit.sh` (read-only) |
| M1 benchmark environment | pinned llama.cpp builds (CUDA/Vulkan/CPU), `benchmarks/` (llama-bench + KL-divergence quality, telemetry), approval lock |
| M2 experiment controller | git-worktree isolation, interleaved A/B measurement, noise-aware multi-objective evaluator, winners + profiles, records + `results.tsv` |
| M3 research agent | provider-independent (`anthropic`, `openai_compatible`, `command`), proposes and interprets experiments |
| M4 continuous research | `python3 -m lab session` with experiment, time, failure and STOP-file limits |
| Safety | independent limits (temperature, VRAM, host RAM, time, stale telemetry), command allow-list, hardware/privileged actions diverted to human-approval proposals |

## M0: hardware audit

Read-only. It changes nothing on the machine.

```bash
scripts/hardware_audit.sh                 # writes research/hardware_profile.json
scripts/hardware_audit.sh --sudo          # also reads DIMM layout via `sudo -n dmidecode` (no prompt)
scripts/hardware_audit.sh --idle-seconds 0  # skip the 10 s passive idle telemetry sample
```

Every fact in the profile is `{"value", "status", "source"}` with status `verified`, `derived`,
`not_detected` or `unavailable`. Raw command output is kept under `research/audit_raw/<timestamp>/` (git-ignored)
as evidence. The profile also contains `spec_claims_check` (the project's assumptions vs what was observed) and
`unknowns`.

Requirements: Python 3.11+ for the lab (3.8+ for the audit), stdlib only. Optional tools improve coverage: `nvidia-smi`, `lspci` (pciutils),
`vulkaninfo` (vulkan-tools), `sensors` (lm-sensors). The audit never installs them.

## Tests

```bash
pip install pytest && python3 -m pytest -q
```

## Layout

```
config/          benchmark.toml + benchmark.lock (protected), safety.toml, research.toml
benchmarks/      benchmark engine and telemetry (protected from experiments)
lab/             audit, controller, evaluator, safety, build, agent, providers/, CLI (python3 -m lab)
winners/         best_config.json, profiles/, history/
experiments/     records/, results.tsv, configurations/ (editable), proposals/ (human approval)
backends/        experimental/ patches (editable); builds and llama.cpp source are git-ignored
research/        hardware profile, hypotheses, notes, limitations, references
scripts/         hardware_audit.sh, build_llama.sh, run_experiment.sh, restore_best.sh
tests/           unit + end-to-end tests with fake llama.cpp binaries
```

## Operating rules

No driver installs, persistent configuration, clock/power/fan changes, firmware work or privileged commands
without explicit human approval. Measure before modifying; never claim an optimisation without measurements.
