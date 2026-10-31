# 1060 Research Lab

An autonomous research environment for extracting the best practical AI-inference performance from one specific
NVIDIA GTX 1060 laptop (Dell G3 17, Ubuntu Server, headless). Inspired by Karpathy's *autoresearch*: a small
experiment loop, measurable results, and a knowledge base that separates verified facts from hypotheses.

**Status: M0 (hardware discovery).** See [`docs/assessment.md`](docs/assessment.md) for the technical
assessment and the M0/M1 plan.

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

Requirements: Python 3.8+ (stdlib only). Optional tools improve coverage: `nvidia-smi`, `lspci` (pciutils),
`vulkaninfo` (vulkan-tools), `sensors` (lm-sensors). The audit never installs them.

## Tests

```bash
pip install pytest && python3 -m pytest -q
```

## Layout

```
docs/assessment.md        technical assessment, risks, benchmark workload, M0/M1 plan
lab/hardware_audit.py     M0 audit (stdlib, read-only whitelist of commands)
scripts/hardware_audit.sh entry point
research/                 knowledge base: hardware_profile.json (generated), hypotheses, references
tests/                    parser and safety tests (synthetic fixtures)
```

## Operating rules

No driver installs, persistent configuration, clock/power/fan changes, firmware work or privileged commands
without explicit human approval. Measure before modifying; never claim an optimisation without measurements.
