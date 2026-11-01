"""Hardware telemetry sampling. Read-only.

A Sampler runs in a background thread and collects one Sample per interval
from a source. The default source shells out to nvidia-smi once per tick
(robust to driver hiccups: a failed read is recorded, not fatal) and reads
host memory / load from /proc. Tests inject a fake source.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import threading
import time
from dataclasses import asdict, dataclass, field

GPU_FIELDS = ("temperature.gpu,utilization.gpu,memory.used,memory.total,clocks.sm,"
              "clocks.mem,power.draw,pstate,clocks_event_reasons.active,power.limit")


@dataclass
class Sample:
    t: float
    gpu_ok: bool = False
    temp_c: float | None = None
    util_pct: float | None = None
    vram_used_mib: float | None = None
    vram_total_mib: float | None = None
    sm_mhz: float | None = None
    mem_mhz: float | None = None
    power_w: float | None = None
    pstate: str | None = None
    event_reasons: str | None = None
    host_mem_available_mb: float | None = None
    load1: float | None = None
    power_limit_w: float | None = None
    ac_online: bool | None = None
    battery_pct: float | None = None
    error: str | None = None


def _f(s: str):
    s = s.strip()
    try:
        return float(s)
    except ValueError:
        return None


def read_power_supply() -> dict:
    """AC and battery state from /sys/class/power_supply (laptops only)."""
    out = {"ac_online": None, "battery_pct": None, "battery_status": None}
    base = "/sys/class/power_supply"
    try:
        names = os.listdir(base)
    except OSError:
        return out
    for n in names:
        try:
            kind = open(f"{base}/{n}/type").read().strip()
            if kind == "Mains":
                online = open(f"{base}/{n}/online").read().strip() == "1"
                out["ac_online"] = bool(out["ac_online"]) or online
            elif kind == "Battery":
                out["battery_pct"] = float(open(f"{base}/{n}/capacity").read().strip())
                out["battery_status"] = open(f"{base}/{n}/status").read().strip()
        except (OSError, ValueError):
            continue
    return out


def read_host(sample: Sample) -> None:
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    sample.host_mem_available_mb = int(line.split()[1]) / 1024
                    break
        sample.load1 = os.getloadavg()[0]
    except OSError:
        pass
    ps = read_power_supply()
    sample.ac_online, sample.battery_pct = ps["ac_online"], ps["battery_pct"]


class NvidiaSmiSource:
    def __init__(self, timeout: float = 5.0):
        self.timeout = timeout
        self.available = shutil.which("nvidia-smi") is not None

    def __call__(self) -> Sample:
        s = Sample(t=time.time())
        read_host(s)
        if not self.available:
            s.error = "nvidia-smi not installed"
            return s
        try:
            p = subprocess.run(["nvidia-smi", f"--query-gpu={GPU_FIELDS}",
                                "--format=csv,noheader,nounits", "-i", "0"],
                               capture_output=True, text=True, timeout=self.timeout,
                               stdin=subprocess.DEVNULL)
        except (OSError, subprocess.TimeoutExpired) as e:
            s.error = f"nvidia-smi failed: {e}"
            return s
        if p.returncode != 0:
            s.error = f"nvidia-smi exit {p.returncode}: {p.stderr.strip()[:200]}"
            return s
        cols = [c.strip() for c in p.stdout.strip().split(",")]
        if len(cols) < 9:
            s.error = f"unexpected nvidia-smi output: {p.stdout.strip()[:200]}"
            return s
        (s.temp_c, s.util_pct, s.vram_used_mib, s.vram_total_mib, s.sm_mhz,
         s.mem_mhz, s.power_w) = (_f(c) for c in cols[:7])
        s.pstate, s.event_reasons = cols[7], cols[8]
        s.power_limit_w = _f(cols[9]) if len(cols) > 9 else None
        s.gpu_ok = True
        return s


class Sampler:
    """Background sampler. Listeners (e.g. the safety monitor) get every sample."""

    def __init__(self, source=None, interval: float = 1.0):
        self.source = source or NvidiaSmiSource()
        self.interval = interval
        self.samples: list[Sample] = []
        self.listeners: list = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    def sample_once(self) -> Sample:
        s = self.source()
        with self._lock:
            self.samples.append(s)
        for fn in list(self.listeners):
            fn(s)
        return s

    def _run(self):
        while not self._stop.is_set():
            self.sample_once()
            self._stop.wait(self.interval)

    def __enter__(self):
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="telemetry")
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=self.interval + 10)

    def window(self, t0: float, t1: float) -> list[Sample]:
        with self._lock:
            return [s for s in self.samples if t0 <= s.t <= t1]


def summarize(samples: list[Sample]) -> dict:
    ok = [s for s in samples if s.gpu_ok]
    out: dict = {"samples": len(samples), "gpu_samples": len(ok),
                 "errors": sorted({s.error for s in samples if s.error})}

    def stats(name):
        v = [getattr(s, name) for s in ok if getattr(s, name) is not None]
        if v:
            out[name] = {"min": min(v), "max": max(v), "mean": round(sum(v) / len(v), 2)}

    for n in ("temp_c", "util_pct", "vram_used_mib", "sm_mhz", "mem_mhz", "power_w", "power_limit_w"):
        stats(n)
    hm = [s.host_mem_available_mb for s in samples if s.host_mem_available_mb is not None]
    if hm:
        out["host_mem_available_mb_min"] = min(hm)
    reasons = {}
    for s in ok:
        if s.event_reasons and s.event_reasons not in ("0x0000000000000000", "Not Active"):
            reasons[s.event_reasons] = reasons.get(s.event_reasons, 0) + 1
    out["clock_event_reasons_seen"] = reasons
    ac = [s.ac_online for s in samples if s.ac_online is not None]
    if ac:
        out["ac_online_all"] = all(ac)
    bat = [s.battery_pct for s in samples if s.battery_pct is not None]
    if bat:
        out["battery_pct"] = {"min": min(bat), "max": max(bat)}
    if len(ok) >= 2 and all(s.power_w is not None for s in ok):
        # Trapezoidal energy estimate over the window.
        out["energy_j"] = round(sum((b.t - a.t) * (a.power_w + b.power_w) / 2
                                    for a, b in zip(ok, ok[1:])), 2)
    return out


def to_rows(samples: list[Sample]) -> list[dict]:
    return [asdict(s) for s in samples]
