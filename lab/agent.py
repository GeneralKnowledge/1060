"""Research agent (M3) and unattended sessions (M4).

The agent only reasons and proposes. It never runs commands itself: every
proposal goes through the Controller, which owns isolation, safety,
benchmarking and the keep/discard decision. The agent's own writing lands in
research/research_notes.md, always tied to the experiment record it is about,
so interpretation never masquerades as measurement.
"""
from __future__ import annotations

import json
import re
import time
import tomllib
from pathlib import Path

from lab.controller import Controller
from lab.experiment import ExperimentSpec, load_records, now, save_record
from lab.providers import Provider, ProviderError, make_provider

ROOT = Path(__file__).resolve().parents[1]

SYSTEM = """You are the research agent of an autonomous GPU-optimisation lab. The target is one
specific laptop GTX 1060 (Pascal, SM 6.1) running llama.cpp for local LLM inference. Your job is
to find configurations and code changes that improve *balanced* inference performance (decode
speed, prompt speed, time to first token, VRAM, host RAM, efficiency) without losing model quality
or stability.

How the lab works:
- You propose ONE experiment at a time. A separate controller builds it in an isolated git
  worktree, benchmarks it against the current best config with interleaved A/B rounds, checks
  quality (KL divergence / perplexity) and stability, and decides promote / discard /
  inconclusive / rejected using measured noise. You cannot change the benchmark, the safety
  limits, or the decision rule.
- An experiment = `changes` deep-merged into the best config (keys: backend [cuda|vulkan|cpu],
  build.cmake_flags, build.patches, build.llama_cpp_commit, runtime.*, env.*, model) plus optional
  `files` you write. You may only write files under the editable paths you are given. A patch to
  llama.cpp goes in backends/experimental/<name>.patch (git diff format, applied at the pinned
  commit) and is referenced from build.patches.
- runtime keys map to llama-bench flags: n_gpu_layers, flash_attn, cache_type_k, cache_type_v,
  batch_size, ubatch_size, threads, mmap, split_mode. env sets runtime environment variables
  (e.g. GGML_CUDA_GRAPH_OPT=1). Compile options such as GGML_CUDA_FORCE_CUBLAS go in
  build.cmake_flags (e.g. "-DGGML_CUDA_FORCE_CUBLAS=ON"); they have no effect as env.
- Correctness gates: batched KL divergence, decode-path perplexity at ubatch 1, and
  test-backend-ops (each CUDA op vs the CPU backend). A code change that breaks any op the
  baseline passes is rejected.
- Clock, power, fan, firmware, driver or any privileged change: set kind to "hardware" or
  "deep_hardware". It will be written up for a human to approve, not executed.

Workloads: each model family is a separate workload with its own best config and history.
Results from another workload are priors for this one, never measurements: carry a setting over
only as a hypothesis, and never compare speeds across models. The workload brief describes the
model's architecture and the questions that are open for it.

Research discipline:
- Start from the workload brief and the prior-knowledge file. Do not spend an experiment re-establishing a result it
  lists as known or measured; knobs it lists as settled by sweep are not yours to re-test. If
  you have new evidence against a prior, say so in revisit_reason.
- Spend the budget on its open questions. Say which one an experiment addresses.
- Prefer the smallest experiment that isolates one variable. Combine changes only once their
  individual effects are known.
- Do not repeat a conclusively tested config. To revisit one, give revisit_reason with the new
  evidence.
- A failed or inconclusive experiment is information; use it.
- Never claim an effect that the records do not show.

Reply with exactly one fenced ```json block and nothing else after it:
{"action": "experiment", "hypothesis": "...", "objective": "...", "description": "...",
 "kind": "software", "changes": {...}, "files": {"path": "content"},
 "benchmark_overrides": {}, "revisit_reason": ""}
or {"action": "stop", "reason": "..."} if nothing worthwhile remains."""

INTERPRET = """Here is the full record of the experiment you proposed. Write a short research note
in Markdown (no top-level heading) covering: what was measured (cite numbers from the record
only), whether the hypothesis is supported, refuted or unresolved and why, what it suggests
about the bottleneck, and one or two follow-up experiments. Label anything not directly measured
as a hypothesis."""


def extract_json(text: str) -> dict:
    blocks = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    candidates = blocks or [text[text.find("{"): text.rfind("}") + 1]]
    for c in reversed(candidates):
        try:
            return json.loads(c)
        except ValueError:
            continue
    raise ValueError("no JSON object found in reply")


def _read(p: Path, limit: int = 6000) -> str:
    if not p.exists():
        return "(none yet)"
    t = p.read_text()
    return t if len(t) <= limit else "...(truncated)...\n" + t[-limit:]


def record_workload(r: dict) -> str:
    env = r.get("execution_environment") or {}
    return env.get("workload") or (r.get("baseline_config") or {}).get("workload") or "small"


def record_summary(r: dict) -> dict:
    ev = r.get("evaluation") or {}
    return {"id": r["id"], "hypothesis": r.get("hypothesis"), "decision": r.get("decision"),
            "changes": _diff(r.get("baseline_config") or {}, r.get("experimental_config") or {}),
            "balanced_score": ev.get("balanced_score"), "threshold": ev.get("threshold"),
            "ratios": {k: v.get("ratio") for k, v in (ev.get("metrics") or {}).items()},
            "gate_failures": ev.get("gate_failures"), "tradeoffs": ev.get("tradeoffs"),
            "error": r.get("error"), "observations": (r.get("research_observations") or "")[:600]}


def _diff(a: dict, b: dict, prefix="") -> dict:
    out = {}
    for k in sorted(set(a) | set(b)):
        if k in ("id", "metrics", "validated", "promoted_utc", "previous", "evaluation_summary",
                 "validated_utc", "baseline_quality"):
            continue
        va, vb = a.get(k), b.get(k)
        if isinstance(va, dict) and isinstance(vb, dict):
            out.update(_diff(va, vb, f"{prefix}{k}."))
        elif va != vb:
            out[f"{prefix}{k}"] = [va, vb]
    return out


class ResearchAgent:
    def __init__(self, controller: Controller, provider: Provider | None = None, root: Path = ROOT):
        self.c = controller
        self.root = Path(root)
        self.research = controller.research
        self.provider = provider or make_provider(self.research["provider"])
        self.notes = self.root / "research" / "research_notes.md"

    def context(self, history_limit: int = 30) -> str:
        # Only this workload's experiments: another model's numbers are not comparable.
        # Records from before workloads were recorded belong to the original "small" one.
        records = [r for r in load_records(self.c.records)
                   if record_workload(r) == self.c.workload][-history_limit:]
        wl = self.c.cfg["workloads"][self.c.workload]
        brief = wl.get("research_brief")
        profile = self.root / "research" / "hardware_profile.json"
        prof = "(hardware audit not run yet: treat all hardware facts as unverified)"
        if profile.exists():
            d = json.loads(profile.read_text())
            prof = json.dumps({k: d.get(k) for k in ("gpu", "driver", "cuda", "clocks", "power",
                                                     "thermal", "memory", "spec_claims_check",
                                                     "unknowns")}, default=str)[:12000]
        props = sorted((self.root / "experiments" / "proposals").glob("*.json"))
        parts = [
            "## Hardware profile\n" + prof,
            f"## Workload\n{self.c.workload} (model family {wl.get('family')}, model "
            f"{wl.get('model')}, architecture {wl.get('arch', 'see brief')})",
            "## Workload research brief (read first)\n" +
            (_read(self.root / brief, 9000) if brief else "(none: use the general priors)"),
            "## Current best config\n" + json.dumps(self.c.best(), indent=1),
            "## Editable paths\n" + json.dumps(self.research["permissions"]["editable"]),
            "## Experiment history (oldest first)\n" +
            "\n".join(json.dumps(record_summary(r), default=str) for r in records),
            "## Prior knowledge (read first)\n" + _read(self.root / "research" / "priors.md", 9000),
            "## Hypotheses\n" + _read(self.root / "research" / "hypotheses.md"),
            "## Known limitations\n" + _read(self.root / "research" / "known_limitations.md", 3000),
            "## Research notes (latest)\n" + _read(self.notes),
            "## Pending human-approval proposals\n" + (", ".join(p.stem for p in props) or "none"),
        ]
        return "\n\n".join(parts)

    def propose(self) -> dict:
        messages = [{"role": "user", "content": self.context() + "\n\nPropose the next experiment."}]
        reply = self.provider.complete(SYSTEM, messages)
        try:
            return extract_json(reply)
        except ValueError as e:
            messages += [{"role": "assistant", "content": reply},
                         {"role": "user", "content": f"That reply could not be parsed ({e}). "
                                                     "Reply with exactly one ```json block."}]
            return extract_json(self.provider.complete(SYSTEM, messages))

    def interpret(self, rec: dict) -> str:
        msg = [{"role": "user", "content": INTERPRET + "\n\n```json\n" +
                json.dumps(rec, indent=1, default=str)[:30000] + "\n```"}]
        try:
            return self.provider.complete(SYSTEM.split("Reply with exactly")[0], msg).strip()
        except ProviderError as e:
            return f"(interpretation unavailable: {e})"

    def step(self) -> dict:
        prop = self.propose()
        if prop.get("action") == "stop":
            return {"decision": "agent_stop", "reason": prop.get("reason", "")}
        spec = ExperimentSpec.from_dict({**prop, "source": "agent"})
        rec = self.c.run(spec)
        note = self.interpret(rec)
        rec["research_observations"] = note
        save_record(rec, self.c.records)
        self.notes.parent.mkdir(parents=True, exist_ok=True)
        with open(self.notes, "a") as fh:
            fh.write(f"\n### {rec['id']}: {rec.get('hypothesis', '')[:120]}\n"
                     f"_Decision: **{rec.get('decision')}** (record experiments/records/{rec['id']}.json). "
                     f"Agent interpretation below; only the record's numbers are measurements._\n\n"
                     f"{note}\n")
        if self.c.auto_commit:
            self.c._commit(["research/research_notes.md", f"experiments/records/{rec['id']}.json"],
                           f"notes: {rec['id']}")
        return rec


def run_session(agent: ResearchAgent, *, max_experiments: int | None = None,
                max_hours: float | None = None, log=print) -> dict:
    """M4: unattended research with hard limits. Stops on: experiment count,
    wall-clock, STOP file, agent stop, or repeated failures."""
    s = agent.research["session"]
    max_experiments = max_experiments or s["max_experiments"]
    max_hours = max_hours or s["max_hours"]
    stop_file = agent.root / s["stop_file"]
    limit_fail = agent.c.limits.max_consecutive_failures
    t0 = time.time()
    done, fails, results = 0, 0, []
    reason = "max_experiments"
    while done < max_experiments:
        if stop_file.exists():
            reason = "stop_file"
            break
        if (time.time() - t0) / 3600 >= max_hours:
            reason = "max_hours"
            break
        try:
            rec = agent.step()
        except ProviderError as e:
            fails += 1
            log(f"[{now()}] provider error: {e}")
            if fails >= limit_fail:
                reason = "provider_failures"
                break
            time.sleep(min(60 * fails, 300))
            continue
        if rec.get("decision") == "agent_stop":
            reason = "agent_stop: " + rec.get("reason", "")
            break
        done += 1
        results.append({"id": rec["id"], "decision": rec["decision"]})
        log(f"[{now()}] {rec['id']} -> {rec['decision']}")
        fails = fails + 1 if rec["decision"] in ("failed",) or \
            (rec.get("stability") or {}).get("candidate_safety_aborts") else 0
        if fails >= limit_fail:
            reason = "consecutive_failures"
            break
        if done < max_experiments:
            time.sleep(s.get("pause_between_s", 0))
    return {"experiments": done, "stopped_because": reason, "results": results,
            "hours": round((time.time() - t0) / 3600, 3)}
