"""Build llama.cpp variants reproducibly.

A build spec is {"backend": cuda|vulkan|cpu, "llama_cpp_commit": sha,
"cmake_flags": [...], "patches": [paths relative to the experiment root]}.
Builds are content-addressed under backends/builds/<key>/ and reused, so
the baseline is built once and a patched candidate never touches it.
Nothing is installed system-wide.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path

from lab.safety import Limits, PolicyViolation, approval_reasons, run_guarded

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = "https://github.com/ggml-org/llama.cpp"
SRC = ROOT / "backends" / "llama.cpp-src"
BUILDS = ROOT / "backends" / "builds"
TARGETS = ["llama-bench", "llama-perplexity", "llama-cli", "llama-server", "test-backend-ops"]
REQUIRED = ["llama-bench", "llama-perplexity", "test-backend-ops"]

BACKEND_FLAGS = {
    "cuda": ["-DGGML_CUDA=ON", "-DCMAKE_CUDA_ARCHITECTURES=61"],
    "vulkan": ["-DGGML_VULKAN=ON"],
    "cpu": [],
}


class BuildError(Exception):
    pass


def build_key(spec: dict, patch_root: Path) -> str:
    h = hashlib.sha256()
    h.update(json.dumps({k: spec.get(k) for k in ("backend", "llama_cpp_commit", "cmake_flags")},
                        sort_keys=True).encode())
    for p in spec.get("patches", []):
        h.update((patch_root / p).read_bytes())
    return h.hexdigest()[:16]


def bin_dir_for(key: str) -> Path:
    return BUILDS / key / "build" / "bin"


def _git(args, limits, cwd=None, budget=600):
    r = run_guarded(["git", *args], limits=limits, budget_s=budget, cwd=cwd, gpu_expected=False)
    if r.returncode != 0:
        raise BuildError(f"git {' '.join(args)} failed: {r.stderr.strip()[-400:]}")
    return r.stdout


def ensure_source(commit: str, limits: Limits):
    if not (SRC / ".git").exists():
        SRC.parent.mkdir(parents=True, exist_ok=True)
        _git(["clone", "--filter=blob:none", UPSTREAM, str(SRC)], limits, budget=1800)
    probe = run_guarded(["git", "cat-file", "-e", f"{commit}^{{commit}}"], limits=limits,
                        budget_s=60, cwd=SRC, gpu_expected=False)
    if probe.returncode != 0:
        _git(["fetch", "origin", commit], limits, cwd=SRC, budget=1800)


def build(spec: dict, limits: Limits, patch_root: Path = ROOT, jobs: int = 2,
          log_dir: Path | None = None) -> Path:
    """Return the bin dir for spec, building it if needed."""
    backend = spec.get("backend", "cuda")
    if backend not in BACKEND_FLAGS:
        raise BuildError(f"unknown backend {backend}")
    flags = list(spec.get("cmake_flags", []))
    bad = approval_reasons(" ".join(flags), limits)
    if bad:
        raise PolicyViolation(f"cmake flags need approval: {bad}")
    key = build_key(spec, patch_root)
    out = bin_dir_for(key)
    if all((out / t).exists() for t in REQUIRED):
        return out
    work = BUILDS / key
    if all((out / t).exists() for t in ("llama-bench", "llama-perplexity")) and \
            (work / "build" / "CMakeCache.txt").exists():
        # Built before test-backend-ops was required: add only the missing targets.
        missing = [t for t in REQUIRED if not (out / t).exists()]
        r = run_guarded(["cmake", "--build", str(work / "build"), "--config", "Release",
                         f"-j{jobs}", "--target", *missing],
                        limits=limits, budget_s=limits.build_time_budget_s, gpu_expected=False,
                        log_path=(log_dir / f"build-{key}-add-targets.log") if log_dir else None)
        if r.returncode == 0 and not r.aborted and all((out / t).exists() for t in REQUIRED):
            return out
    commit = spec["llama_cpp_commit"]
    ensure_source(commit, limits)
    src = work / "src"
    if work.exists():
        shutil.rmtree(work)  # half-finished build of this exact key; safe to redo
    work.mkdir(parents=True)
    _git(["worktree", "add", "--detach", str(src), commit], limits, cwd=SRC)
    for p in spec.get("patches", []):
        _git(["apply", "--whitespace=nowarn", str((patch_root / p).resolve())], limits, cwd=src)
    (work / "spec.json").write_text(json.dumps(spec, indent=2))
    cfg = ["cmake", "-S", str(src), "-B", str(work / "build"), "-DCMAKE_BUILD_TYPE=Release",
           "-DLLAMA_CURL=OFF", "-DLLAMA_BUILD_TESTS=ON", *BACKEND_FLAGS[backend], *flags]
    r = run_guarded(cfg, limits=limits, budget_s=limits.build_time_budget_s, gpu_expected=False,
                    log_path=(log_dir / f"build-{key}-configure.log") if log_dir else None)
    if r.returncode != 0:
        raise BuildError(f"cmake configure failed: {r.stderr.strip()[-800:]}")
    r = run_guarded(["cmake", "--build", str(work / "build"), "--config", "Release",
                     f"-j{jobs}", "--target", *TARGETS],
                    limits=limits, budget_s=limits.build_time_budget_s, gpu_expected=False,
                    log_path=(log_dir / f"build-{key}-compile.log") if log_dir else None)
    if r.returncode != 0 or r.aborted:
        raise BuildError(f"build failed ({r.aborted or r.returncode}): {r.stderr.strip()[-800:]}")
    return out


def remove_worktrees(limits: Limits):
    """Housekeeping: prune worktree metadata for deleted build dirs."""
    if (SRC / ".git").exists():
        run_guarded(["git", "worktree", "prune"], limits=limits, budget_s=60, cwd=SRC,
                    gpu_expected=False)
