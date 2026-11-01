"""Scripted sweeps: settle well-known knobs without spending research-agent calls.

A sweep file lists steps, each a `changes` block like an agent proposal. Every
step runs through the normal Controller (isolated worktree, locked benchmark,
interleaved A/B against the *current* best, quality and kernel gates), so a
promoted step becomes the base for the next one. Already-tested configs come
back as `duplicate` at no cost.
"""
from __future__ import annotations

import json
from pathlib import Path

from lab.controller import Controller
from lab.experiment import ExperimentSpec, now


def load_sweep(path: Path) -> dict:
    d = json.loads(Path(path).read_text())
    if not d.get("name") or not isinstance(d.get("steps"), list) or not d["steps"]:
        raise ValueError(f"{path}: a sweep needs a name and a non-empty steps list")
    for i, s in enumerate(d["steps"]):
        if not s.get("hypothesis") or not isinstance(s.get("changes"), dict):
            raise ValueError(f"{path}: step {i} needs a hypothesis and a changes object")
    return d


def run_sweep(c: Controller, sweep: dict, *, log=print) -> dict:
    stop_file = c.root / c.research["session"]["stop_file"]
    limit = c.limits.max_consecutive_failures
    results, fails, reason = [], 0, "complete"
    for i, step in enumerate(sweep["steps"]):
        if stop_file.exists():
            reason = "stop_file"
            break
        spec = ExperimentSpec(
            hypothesis=step["hypothesis"],
            objective=step.get("objective", f"sweep {sweep['name']} step {i + 1}"),
            description=step.get("description", f"scripted sweep {sweep['name']}"),
            changes=step["changes"], revisit_reason=step.get("revisit_reason", ""),
            source=f"sweep:{sweep['name']}")
        rec = c.run(spec)
        results.append({"step": i + 1, "id": rec["id"], "decision": rec["decision"],
                        "balanced_score": (rec.get("evaluation") or {}).get("balanced_score")})
        log(f"[{now()}] sweep {sweep['name']} step {i + 1}: {rec['id']} -> {rec['decision']}")
        fails = fails + 1 if rec["decision"] == "failed" else 0
        if fails >= limit:
            reason = "consecutive_failures"
            break
    return {"sweep": sweep["name"], "stopped_because": reason, "results": results,
            "best": c.best()["id"]}
