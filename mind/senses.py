"""Percept sources. Each sense has a name and poll(tick) -> list of "feature=value" concepts.

Real signals arrive as raw, unlabelled values (binned numbers), the same way as
the generated world: the mind is told nothing about what they mean.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import time
from pathlib import Path

from mind.world import World


class WorldSense:
    name = "world"

    def __init__(self, d: Path, seed: int):
        self.d = d
        self.world = World.load_or_create(d, seed)

    def poll(self, tick):
        return [f"{f}={v}" for f, v in self.world.step()]

    def save(self):
        self.world.save(self.d)


class _Changes:
    """Emit a value only when it changes (like the world does)."""

    def __init__(self):
        self.last = {}

    def diff(self, values: dict):
        out = []
        for k, v in values.items():
            if v is not None and self.last.get(k) != v:
                self.last[k] = v
                out.append(f"{k}={v}")
        return out


class ClockSense(_Changes):
    name = "clock"

    def poll(self, tick):
        t = time.localtime()
        return self.diff({"c0.a": t.tm_hour, "c0.b": t.tm_wday})


def _bin(x, edges):
    if x is None:
        return None
    return sum(x >= e for e in edges)


class HostSense(_Changes):
    """Laptop body signals, sampled every `every` ticks, binned to a few levels."""
    name = "host"

    def __init__(self, every=6):
        super().__init__()
        self.every = every
        self.smi = shutil.which("nvidia-smi")

    def poll(self, tick):
        if tick % self.every:
            return []
        vals = {}
        try:
            vals["h0.a"] = _bin(float(open("/proc/loadavg").read().split()[0]), [0.5, 1.5, 3, 6])
            mem = open("/proc/meminfo").read()
            avail = int(re.search(r"MemAvailable:\s+(\d+)", mem).group(1)) / 1024
            vals["h0.b"] = _bin(avail, [1000, 2000, 4000])
        except (OSError, AttributeError, ValueError):
            pass
        if self.smi:
            try:
                out = subprocess.run([self.smi, "--query-gpu=temperature.gpu,utilization.gpu,memory.used",
                                      "--format=csv,noheader,nounits"], capture_output=True, text=True,
                                     timeout=5).stdout.split(",")
                vals["h1.a"] = _bin(float(out[0]), [45, 60, 75])
                vals["h1.b"] = _bin(float(out[1]), [5, 50, 90])
                vals["h1.c"] = _bin(float(out[2]), [500, 2000, 4000])
            except (OSError, subprocess.SubprocessError, ValueError, IndexError):
                pass
        return self.diff(vals)


class FileTailSense:
    """New lines appended to a file become percepts (only a coarse token of each line)."""

    def __init__(self, name, path: Path, column: int, every=12):
        self.name, self.path, self.column, self.every = name, path, column, every
        self.pos = path.stat().st_size if path.exists() else 0

    def poll(self, tick):
        if tick % self.every or not self.path.exists():
            return []
        size = self.path.stat().st_size
        if size < self.pos:
            self.pos = 0
        if size == self.pos:
            return []
        with open(self.path) as f:
            f.seek(self.pos)
            lines = f.read().splitlines()
            self.pos = f.tell()
        out = []
        for ln in lines:
            cols = ln.split("\t")
            if len(cols) > self.column:
                out.append(f"{self.name}.a={cols[self.column].strip()[:24]}")
        return out


class InboxSense:
    """Messages dropped into state/inbox/ arrive as word percepts, with no built-in priority."""
    name = "inbox"

    def __init__(self, d: Path):
        self.d = d
        d.mkdir(parents=True, exist_ok=True)

    def poll(self, tick):
        out = []
        for p in sorted(self.d.glob("*.txt")):
            words = re.findall(r"[a-z0-9']+", p.read_text().lower())[:40]
            out += [f"m0.w={w}" for w in words]
            p.rename(p.with_suffix(".read"))
        return out


def build(state_dir: Path, seed: int, real: bool, repo: Path):
    senses = [WorldSense(state_dir / "world", seed)]
    if real:
        senses += [ClockSense(), HostSense(), InboxSense(state_dir / "inbox"),
                   FileTailSense("r0", repo / "experiments" / "results.tsv", column=4)]
    return senses
