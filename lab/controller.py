"""Experiment controller: isolated workspace -> build -> measure -> evaluate ->
promote or discard -> permanent record.

Invariants
  * The validated best config (winners/best_config.json) changes only through
    promote(), which archives the previous winner and commits both.
  * Benchmark code always runs from this checkout, never from the worktree,
    and only when its fingerprint matches the approved lock.
  * The worktree may only change paths listed as editable in research.toml.
  * Anything needing hardware/privileged access becomes a proposal file for a
    human; it is never executed here.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import tomllib
from pathlib import Path

from benchmarks import benchmark as bench
from benchmarks.telemetry import NvidiaSmiSource, Sampler
from lab import evaluator
from lab.experiment import (ExperimentSpec, append_tsv, config_hash, deep_merge, load_records,
                            now, save_record)
from lab.safety import Limits, PolicyViolation, approval_reasons, load_limits, power_preflight

ROOT = Path(__file__).resolve().parents[1]


class ControllerError(Exception):
    pass


def git(root: Path, *args, check=True) -> str:
    p = subprocess.run(["git", *args], cwd=root, capture_output=True, text=True,
                       stdin=subprocess.DEVNULL)
    if check and p.returncode != 0:
        raise ControllerError(f"git {' '.join(args)}: {p.stderr.strip()}")
    return p.stdout


def default_builder(build_spec: dict, limits: Limits, patch_root: Path, log_dir: Path) -> Path:
    from lab.build import build
    return build(build_spec, limits, patch_root=patch_root, log_dir=log_dir)


class Controller:
    def __init__(self, root: Path = ROOT, limits: Limits | None = None, builder=default_builder,
                 telemetry_source=None, auto_commit: bool = True, use_telemetry: bool = True):
        self.root = Path(root)
        self.limits = limits or load_limits(self.root / "config" / "safety.toml")
        self.cfg = bench.load_config(self.root / "config" / "benchmark.toml")
        self.research = tomllib.loads((self.root / "config" / "research.toml").read_text())
        self.builder = builder
        self.telemetry_source = telemetry_source
        self.use_telemetry = use_telemetry
        self.auto_commit = auto_commit
        self.records = self.root / "experiments" / "records"
        self.logs = self.root / "experiments" / "logs"
        self.best_path = self.root / "winners" / "best_config.json"

    # ------------------------------------------------------------ state
    def best(self) -> dict:
        return json.loads(self.best_path.read_text())

    def tested_hashes(self) -> dict:
        return {r.get("candidate_config_hash"): r["id"] for r in load_records(self.records)
                if r.get("candidate_config_hash")}

    # ------------------------------------------------------------ workspace
    def worktree(self, exp_id: str) -> Path:
        return self.root / "experiments" / "worktrees" / exp_id

    def create_workspace(self, exp_id: str) -> Path:
        wt = self.worktree(exp_id)
        wt.parent.mkdir(parents=True, exist_ok=True)
        git(self.root, "worktree", "add", "-q", "-b", f"exp/{exp_id}", str(wt), "HEAD")
        return wt

    def _allowed(self, rel: str) -> bool:
        perms = self.research["permissions"]
        if any(rel.startswith(p) or rel == p.rstrip("/") for p in perms["protected"]):
            return False
        return any(rel.startswith(p) for p in perms["editable"])

    def write_files(self, wt: Path, files: dict):
        for rel, content in files.items():
            rel = os.path.normpath(rel)
            if rel.startswith("..") or os.path.isabs(rel) or not self._allowed(rel):
                raise PolicyViolation(f"experiment may not write {rel}")
            p = wt / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content)

    def validate_and_commit(self, wt: Path, exp_id: str) -> tuple[list[str], str]:
        status = git(wt, "status", "--porcelain", "--untracked-files=all")
        changed = [line[3:].strip() for line in status.splitlines() if line.strip()]
        bad = [c for c in changed if not self._allowed(c)]
        if bad:
            raise PolicyViolation(f"experiment modified protected paths: {bad}")
        diff = ""
        if changed:
            git(wt, "add", "-A")
            git(wt, "-c", "user.name=lab", "-c", "user.email=lab@localhost", "commit", "-q",
                "-m", f"experiment {exp_id}")
            diff = git(wt, "show", "--format=", "HEAD")
        return changed, diff

    def cleanup(self, exp_id: str, keep_branch: bool = False):
        wt = self.worktree(exp_id)
        if wt.exists():
            git(self.root, "worktree", "remove", "--force", str(wt), check=False)
            shutil.rmtree(wt, ignore_errors=True)
        git(self.root, "worktree", "prune", check=False)
        if not keep_branch:
            git(self.root, "branch", "-D", f"exp/{exp_id}", check=False)

    # ------------------------------------------------------------ measurement
    def _sampler(self):
        if not self.use_telemetry:
            return None
        return Sampler(self.telemetry_source or NvidiaSmiSource(), interval=1.0)

    def _prepare(self, config: dict, patch_root: Path, log_dir: Path):
        build_spec = {"backend": config["backend"], **config["build"]}
        bin_dir = self.builder(build_spec, self.limits, patch_root, log_dir)
        wl = bench.resolve_workload(self.cfg, config["workload"], config.get("model"), self.root)
        return Path(bin_dir), wl

    def measure_pair(self, base_cfg: dict, cand_cfg: dict | None, patch_root: Path,
                     exp_id: str, overrides: dict | None = None) -> dict:
        """Interleaved A/B measurement. cand_cfg=None measures the baseline alone."""
        defaults = dict(self.cfg["defaults"])
        for k in ("rounds", "repetitions"):
            if overrides and k in overrides:
                defaults[k] = int(overrides[k])
        log_dir = self.logs / exp_id
        arms = {"baseline": (base_cfg, self.root)}
        if cand_cfg is not None:
            arms["candidate"] = (cand_cfg, patch_root)
        prepared = {name: self._prepare(c, pr, log_dir) for name, (c, pr) in arms.items()}
        ok, why, power_state = power_preflight(self.limits)
        if not ok:
            raise ControllerError(f"power preflight failed: {why}")
        runs = {name: [] for name in arms}
        quality = {}
        sampler = self._sampler()
        ctx = sampler if sampler else _Null()
        with ctx:
            idle = sampler.sample_once().temp_c if sampler else None
            if defaults.get("warmup"):
                for name in arms:
                    b, wl = prepared[name]
                    warm = dict(defaults, repetitions=1, n_prompt=[defaults["n_prompt"][0]],
                                n_gen=[16])
                    bench.run_bench_once(b, arms[name][0]["runtime"], arms[name][0].get("env", {}),
                                         wl, warm, limits=self.limits, sampler=sampler)
            order = list(arms)
            for rnd in range(defaults["rounds"]):
                ok, why, _ = power_preflight(self.limits)
                if not ok:
                    raise ControllerError(f"power check failed before round {rnd}: {why}")
                for name in (order if rnd % 2 == 0 else order[::-1]):
                    bench.wait_cooldown(sampler, idle, defaults["cooldown_max_delta_c"],
                                        defaults["cooldown_timeout_s"])
                    b, wl = prepared[name]
                    c = arms[name][0]
                    runs[name].append(bench.run_bench_once(
                        b, c["runtime"], c.get("env", {}), wl, defaults, limits=self.limits,
                        sampler=sampler, log_path=log_dir / f"{name}-bench-r{rnd}.log"))
            for name in arms:
                b, wl = prepared[name]
                c = arms[name][0]
                quality[name] = bench.run_quality(b, c["runtime"], c.get("env", {}), wl, self.cfg,
                                                  limits=self.limits, sampler=sampler,
                                                  log_path=log_dir / f"{name}-quality.log",
                                                  root=self.root)
        capped = sorted({f"{name}: {ev['detail']}" for name, rs in runs.items() for r in rs
                         for ev in r.get("safety_events", []) if ev.get("kind") == "power_capped"})
        # Short runs can finish between monitor ticks, so also check every sample
        # the sampler took during the whole measurement (including the idle one).
        floor = self.limits.power_limit_min_w
        if sampler and floor:
            low = [s.power_limit_w for s in sampler.samples
                   if s.power_limit_w is not None and s.power_limit_w < floor]
            if low:
                capped = sorted(set(capped) | {f"sampler: GPU power limit {min(low):.0f} W "
                                               f"< {floor:.0f} W in {len(low)} sample(s)"})
        return {"runs": runs, "quality": quality, "idle_temp_c": idle,
                "power_state": power_state, "power_capped": capped,
                "trials": {n: bench.derive_metrics(r, defaults) for n, r in runs.items()},
                "defaults": defaults,
                "bin_dirs": {n: str(p[0]) for n, p in prepared.items()}}

    # ------------------------------------------------------------ lifecycle
    def run(self, spec: ExperimentSpec) -> dict:
        lock = bench.verify_lock(self.root)
        best = self.best()
        if spec.workload and spec.workload != best["workload"]:
            raise ControllerError("candidate workload differs from the best config's; run it as a "
                                  "separate investigation with its own baseline")
        rec = {"id": spec.id, "hypothesis": spec.hypothesis, "research_objective": spec.objective,
               "description": spec.description, "kind": spec.kind, "started_utc": now(),
               "baseline_config": best, "benchmark": lock,
               "execution_environment": {"workload": best["workload"],
                                         "hardware_profile": _profile_digest(self.root)}}

        # Approval gate: hardware experiments and anything privileged go to a human.
        text = json.dumps(spec.changes) + "\n" + "\n".join(spec.files.values())
        reasons = approval_reasons(text, self.limits)
        if spec.kind != "software" or reasons:
            prop = self.root / "experiments" / "proposals" / f"{spec.id}.json"
            prop.parent.mkdir(parents=True, exist_ok=True)
            prop.write_text(json.dumps({**spec.__dict__, "approval_reasons": reasons,
                                        "status": "awaiting_human_approval"}, indent=2))
            rec.update(decision="awaiting_approval", finished_utc=now(),
                       research_observations=f"proposal written to {prop.relative_to(self.root)}")
            return self._finish(rec)

        cand = deep_merge(best, spec.changes)
        for stale in ("validated", "metrics", "note", "validated_utc", "baseline_quality"):
            cand.pop(stale, None)
        cand["id"] = spec.id
        rec["experimental_config"] = cand
        h = config_hash(cand, self._patch_contents(cand, spec.files))
        rec["candidate_config_hash"] = h
        prior = self.tested_hashes().get(h)
        if prior and not spec.revisit_reason:
            rec.update(decision="duplicate", finished_utc=now(),
                       research_observations=f"identical config already tested in {prior}")
            return self._finish(rec)
        if h == config_hash(best, self._patch_contents(best, {})) and not spec.files:
            rec.update(decision="no_change", finished_utc=now())
            return self._finish(rec)

        wt = self.create_workspace(spec.id)
        promoted = False
        try:
            self.write_files(wt, spec.files)
            changed, diff = self.validate_and_commit(wt, spec.id)
            rec["files_modified"] = changed
            if diff:
                (self.records).mkdir(parents=True, exist_ok=True)
                (self.records / f"{spec.id}.diff").write_text(diff)
            m = self.measure_pair(best, cand, wt, spec.id, spec.benchmark_overrides)
            rec["benchmark_parameters"] = m["defaults"]
            rec["execution_environment"]["power_state"] = m["power_state"]
            rec["results"] = {"trials": m["trials"], "idle_temp_c": m["idle_temp_c"],
                              "telemetry": {n: [r.get("telemetry") for r in rs]
                                            for n, rs in m["runs"].items()}}
            rec["correctness"] = m["quality"]
            errors = [e for r in m["runs"]["candidate"] for e in r["errors"]]
            aborts = [r["aborted"] for r in m["runs"]["candidate"] if r.get("aborted")]
            base_errors = [e for r in m["runs"]["baseline"] for e in r["errors"]]
            rec["stability"] = {"candidate_errors": errors, "candidate_safety_aborts": aborts,
                                "baseline_errors": base_errors,
                                "safety_events": [ev for r in m["runs"]["candidate"]
                                                  for ev in r.get("safety_events", [])]}
            if base_errors:
                raise ControllerError(f"baseline failed during experiment: {base_errors[:3]}")
            if m["power_capped"]:
                # Firmware power throttling in either arm makes the A/B comparison meaningless.
                rec["stability"]["power_capped"] = m["power_capped"]
                rec.update(decision="invalid_power_state",
                           research_observations="GPU power limit dropped during measurement; "
                                                 "result not comparable, re-run when power is stable")
                raise _Stop()
            ev = evaluator.evaluate(m["trials"]["baseline"], m["trials"]["candidate"],
                                    base_quality=m["quality"].get("baseline"),
                                    cand_quality=m["quality"].get("candidate"),
                                    errors=errors, safety_aborts=aborts, cfg=self.cfg)
            rec["evaluation"] = ev
            rec["decision"] = ev["decision"]
            if ev["decision"] == "promote":
                self.promote(spec.id, cand, ev, m["trials"]["candidate"])
                promoted = True
            elif ev["profile_wins"]:
                self.update_profiles(spec.id, cand, ev, m["trials"]["candidate"])
        except _Stop:
            pass
        except PolicyViolation as e:
            rec.update(decision="rejected", error=f"policy: {e}")
        except (ControllerError, bench.BenchmarkError, Exception) as e:  # record every failure
            rec.update(decision="failed", error=f"{type(e).__name__}: {e}")
        finally:
            self.cleanup(spec.id)
        rec["finished_utc"] = now()
        rec["promoted"] = promoted
        return self._finish(rec)

    def _patch_contents(self, cfg: dict, files: dict) -> dict:
        """Text of each patch cfg references: the experiment's own files first, else this checkout."""
        out = {}
        for p in (cfg.get("build") or {}).get("patches") or []:
            if p in files:
                out[p] = files[p]
            elif (self.root / p).is_file():
                out[p] = (self.root / p).read_text()
        return out

    def _finish(self, rec: dict) -> dict:
        save_record(rec, self.records)
        append_tsv(rec, self.root / "experiments" / "results.tsv")
        if self.auto_commit:
            self._commit([f"experiments/records/{rec['id']}.json", "experiments/results.tsv",
                          f"experiments/records/{rec['id']}.diff",
                          f"experiments/proposals/{rec['id']}.json"],
                         f"experiment {rec['id']}: {rec['decision']}")
        return rec

    def _commit(self, paths, msg):
        existing = [p for p in paths if (self.root / p).exists()]
        if not existing:
            return
        git(self.root, "add", "--", *existing)
        if git(self.root, "diff", "--cached", "--name-only").strip():
            git(self.root, "-c", "user.name=lab", "-c", "user.email=lab@localhost",
                "commit", "-q", "-m", msg)

    # ------------------------------------------------------------ winners
    def promote(self, exp_id: str, cand: dict, ev: dict, trials: dict):
        branch = f"exp/{exp_id}"
        if git(self.root, "rev-list", "--count", f"HEAD..{branch}").strip() != "0":
            out = subprocess.run(["git", "-c", "user.name=lab", "-c", "user.email=lab@localhost",
                                  "merge", "--no-ff", "-q", "-m", f"promote {exp_id}", branch],
                                 cwd=self.root, capture_output=True, text=True)
            if out.returncode != 0:
                git(self.root, "merge", "--abort", check=False)
                raise ControllerError(f"could not merge {branch}: {out.stderr.strip()}")
        old = self.best()
        hist = self.root / "winners" / "history"
        hist.mkdir(parents=True, exist_ok=True)
        (hist / f"{old['id']}.json").write_text(json.dumps(old, indent=2) + "\n")
        new = dict(cand, validated=True, promoted_utc=now(), previous=old["id"],
                   metrics=_medians(trials), evaluation_summary={
                       "balanced_score": ev["balanced_score"], "tradeoffs": ev["tradeoffs"]})
        self.best_path.write_text(json.dumps(new, indent=2) + "\n")
        self.update_profiles(exp_id, cand, ev, trials, commit=False)
        if self.auto_commit:
            self._commit(["winners"], f"winners: promote {exp_id}")

    def update_profiles(self, exp_id, cand, ev, trials, commit=True):
        pdir = self.root / "winners" / "profiles"
        pdir.mkdir(parents=True, exist_ok=True)
        for prof in ev["profile_wins"]:
            (pdir / f"{prof}.json").write_text(json.dumps(
                dict(cand, profile=prof, from_experiment=exp_id, metrics=_medians(trials),
                     updated_utc=now()), indent=2) + "\n")
        if commit and self.auto_commit:
            self._commit(["winners/profiles"], f"profiles: {exp_id} wins {ev['profile_wins']}")

    def validate_baseline(self) -> dict:
        """Measure the current best config alone and mark it validated."""
        bench.verify_lock(self.root)
        best = self.best()
        exp_id = "baseline-" + now().replace(":", "")
        m = self.measure_pair(best, None, self.root, exp_id)
        errs = [e for r in m["runs"]["baseline"] for e in r["errors"]]
        q = m["quality"]["baseline"]
        rec = {"id": exp_id, "hypothesis": "baseline validation", "kind": "baseline",
               "started_utc": now(), "baseline_config": best,
               "execution_environment": {"workload": best["workload"],
                                         "hardware_profile": _profile_digest(self.root)},
               "results": {"trials": m["trials"]}, "correctness": {"baseline": q},
               "stability": {"errors": errs}}
        rec["execution_environment"]["power_state"] = m["power_state"]
        rec["stability"]["power_capped"] = m["power_capped"]
        ok = not errs and q.get("returncode") == 0 and not m["power_capped"]
        rec["decision"] = "baseline_validated" if ok else "baseline_failed"
        if ok:
            best.pop("note", None)  # e.g. "not yet measured": no longer true once validated
            best.update(validated=True, metrics=_medians(m["trials"]["baseline"]),
                        validated_utc=now(), baseline_quality=q)
            self.best_path.write_text(json.dumps(best, indent=2) + "\n")
            if self.auto_commit:
                self._commit(["winners/best_config.json"], f"winners: validate baseline {exp_id}")
        rec["finished_utc"] = now()
        return self._finish(rec)


class _Stop(Exception):
    """Ends an experiment early with the decision already set on the record."""


class _Null:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _medians(trials: dict) -> dict:
    import statistics
    return {k: round(statistics.median(v), 4) for k, v in trials.items() if v}


def _profile_digest(root: Path) -> dict | None:
    p = root / "research" / "hardware_profile.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    g = d.get("gpu", {})
    return {"audit_time": d.get("audit", {}).get("timestamp_utc"),
            "gpu": g.get("name", {}).get("value"), "vbios": g.get("vbios_version", {}).get("value"),
            "driver": d.get("driver", {}).get("nvidia_driver_version", {}).get("value")}
