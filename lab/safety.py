"""Safety controller. Independent of the research agent.

* load_limits(): reads config/safety.toml (a protected path).
* SafetyMonitor: subscribes to telemetry; trips on over-temperature, VRAM or
  host-memory exhaustion, stale telemetry (driver hang) or time budget, and
  kills the guarded process group.
* check_command()/approval_reasons(): command policy. Only allow-listed
  executables run; anything touching privileged or hardware interfaces is
  diverted to a human-approval proposal instead of executing.
* run_guarded(): run one subprocess under all of the above.
"""
from __future__ import annotations

import os
import re
import shutil
import signal
import subprocess
import threading
import time
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


@dataclass
class Limits:
    gpu_temp_max_c: float = 87
    gpu_temp_warn_c: float = 82
    vram_max_fraction: float = 0.98
    host_mem_available_min_mb: float = 512
    disk_free_min_gb: float = 5
    experiment_time_budget_s: float = 300
    build_time_budget_s: float = 3600
    telemetry_stale_s: float = 15
    max_consecutive_failures: int = 3
    require_ac: bool = False
    battery_min_pct: float = 0
    power_limit_min_w: float = 0
    allowed_executables: list = field(default_factory=list)
    approval_required_patterns: list = field(default_factory=list)


def load_limits(path: Path | None = None) -> Limits:
    path = path or ROOT / "config" / "safety.toml"
    data = tomllib.loads(path.read_text())
    return Limits(**data.get("limits", {}), **data.get("policy", {}))


class PolicyViolation(Exception):
    pass


def approval_reasons(text: str, limits: Limits) -> list[str]:
    """Patterns in a command line / config / patch that need a human."""
    low = f" {text.lower()} "
    hits = []
    for p in limits.approval_required_patterns:
        pl = p.lower()
        # Word-ish match for short tokens so "apt" doesn't hit "adapter".
        if len(pl) <= 4 and pl.isalpha():
            if re.search(rf"(?<![\w-]){re.escape(pl)}(?![\w-])", low):
                hits.append(p)
        elif pl in low:
            hits.append(p)
    return hits


def check_command(argv: list[str], limits: Limits) -> None:
    exe = os.path.basename(argv[0])
    if exe not in limits.allowed_executables:
        raise PolicyViolation(f"executable not allowed: {exe}")
    reasons = approval_reasons(" ".join([exe, *argv[1:]]), limits)
    if reasons:
        raise PolicyViolation(f"needs human approval: {reasons}")


@dataclass
class SafetyEvent:
    t: float
    kind: str
    detail: str
    fatal: bool


class SafetyMonitor:
    def __init__(self, limits: Limits, budget_s: float):
        self.limits = limits
        self.budget_s = budget_s
        self.events: list[SafetyEvent] = []
        self.tripped = threading.Event()
        self._kill = None
        self._t0 = time.time()
        self._last_gpu_ok = time.time()
        self._watch = threading.Event()

    # --- wiring
    def arm(self, kill_fn):
        self._kill = kill_fn
        self._t0 = time.time()
        self._last_gpu_ok = time.time()
        self._watch.clear()
        threading.Thread(target=self._watchdog, daemon=True, name="safety-watchdog").start()

    def disarm(self):
        self._watch.set()
        self._kill = None

    def _event(self, kind, detail, fatal):
        self.events.append(SafetyEvent(time.time(), kind, detail, fatal))
        if fatal and not self.tripped.is_set():
            self.tripped.set()
            if self._kill:
                self._kill()

    # --- telemetry listener
    def on_sample(self, s):
        L = self.limits
        if s.gpu_ok:
            self._last_gpu_ok = s.t
            if s.temp_c is not None and s.temp_c >= L.gpu_temp_max_c:
                self._event("gpu_overtemp", f"{s.temp_c} C >= {L.gpu_temp_max_c}", True)
            elif s.temp_c is not None and s.temp_c >= L.gpu_temp_warn_c:
                self._event("gpu_temp_warn", f"{s.temp_c} C", False)
            if L.power_limit_min_w and s.power_limit_w is not None \
                    and s.power_limit_w < L.power_limit_min_w:
                self._event("power_capped", f"GPU power limit {s.power_limit_w} W < "
                            f"{L.power_limit_min_w} W (firmware throttling?)", False)
            if s.vram_used_mib and s.vram_total_mib and \
                    s.vram_used_mib / s.vram_total_mib > L.vram_max_fraction:
                self._event("vram_exhausted", f"{s.vram_used_mib}/{s.vram_total_mib} MiB", True)
        if L.require_ac and s.ac_online is False:
            self._event("ac_lost", "AC adapter disconnected during run", True)
        if s.host_mem_available_mb is not None and s.host_mem_available_mb < L.host_mem_available_min_mb:
            self._event("host_mem_low", f"{s.host_mem_available_mb:.0f} MB available", True)

    def _watchdog(self):
        while not self._watch.wait(1.0):
            now = time.time()
            if now - self._t0 > self.budget_s:
                self._event("time_budget", f"exceeded {self.budget_s}s", True)
                return
            if now - self._last_gpu_ok > self.limits.telemetry_stale_s and self._gpu_expected:
                self._event("telemetry_stale", f"no GPU telemetry for {now - self._last_gpu_ok:.0f}s "
                            "(driver hang?)", True)
                return

    _gpu_expected = True


def power_preflight(limits: Limits) -> tuple[bool, str, dict]:
    """Is the machine in a fit power state to start a measurement?"""
    from benchmarks.telemetry import read_power_supply
    ps = read_power_supply()
    if limits.require_ac and ps["ac_online"] is False:
        return False, "on battery power (AC adapter not connected)", ps
    if limits.battery_min_pct and ps["battery_pct"] is not None \
            and ps["battery_pct"] < limits.battery_min_pct:
        return False, f"battery at {ps['battery_pct']:.0f}% < {limits.battery_min_pct}%", ps
    return True, "ok", ps


def disk_ok(limits: Limits, path: Path = ROOT) -> bool:
    return shutil.disk_usage(path).free / 2**30 >= limits.disk_free_min_gb


@dataclass
class GuardedResult:
    argv: list
    returncode: int | None
    stdout: str
    stderr: str
    duration_s: float
    aborted: str | None
    events: list
    max_rss_kb: int | None = None


def run_guarded(argv, *, limits: Limits, sampler=None, budget_s: float, cwd=None, env=None,
                gpu_expected: bool = True, log_path: Path | None = None) -> GuardedResult:
    """Run argv in its own process group, under telemetry + safety monitor."""
    check_command(argv, limits)
    if not disk_ok(limits):
        raise PolicyViolation("free disk below safety limit")
    mon = SafetyMonitor(limits, budget_s)
    mon._gpu_expected = gpu_expected and sampler is not None
    full_env = {**os.environ, **(env or {})}
    t0 = time.time()
    proc = subprocess.Popen(argv, cwd=cwd, env=full_env, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True, start_new_session=True,
                            stdin=subprocess.DEVNULL)

    def kill():
        try:
            os.killpg(proc.pid, signal.SIGTERM)
            time.sleep(2)
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass

    peak = {"rss_kb": None}
    done = threading.Event()

    def watch_rss():  # VmHWM = peak resident set of the child itself
        while not done.wait(0.5):
            try:
                for line in Path(f"/proc/{proc.pid}/status").read_text().splitlines():
                    if line.startswith("VmHWM:"):
                        peak["rss_kb"] = max(peak["rss_kb"] or 0, int(line.split()[1]))
            except (OSError, ValueError):
                pass

    threading.Thread(target=watch_rss, daemon=True).start()
    if sampler is not None:
        sampler.listeners.append(mon.on_sample)
    mon.arm(kill)
    try:
        out, err = proc.communicate()
    finally:
        done.set()
        mon.disarm()
        if sampler is not None and mon.on_sample in sampler.listeners:
            sampler.listeners.remove(mon.on_sample)
    rss = peak["rss_kb"]
    fatal = [e for e in mon.events if e.fatal]
    res = GuardedResult(argv=list(argv), returncode=proc.returncode, stdout=out, stderr=err,
                        duration_s=round(time.time() - t0, 2),
                        aborted=fatal[0].kind if fatal else None,
                        events=[e.__dict__ for e in mon.events], max_rss_kb=rss)
    if log_path:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(f"$ {' '.join(argv)}\n--- stdout ---\n{out}\n--- stderr ---\n{err}\n"
                            f"--- rc={proc.returncode} aborted={res.aborted} ---\n")
    return res
