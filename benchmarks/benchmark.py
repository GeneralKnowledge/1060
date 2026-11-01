"""Benchmark engine: the objective evaluation mechanism.

Independent of the research agent. The controller always imports this module
from the main checkout (never from an experiment worktree), and refuses to
run if benchmarks/ or config/benchmark.toml differ from the approved lock.

A "lab config" (see winners/best_config.json) says which build to use and
which runtime flags to pass. measure() runs llama-bench for it under
telemetry and safety monitoring and returns per-metric trial values so the
evaluator can account for noise.
"""
from __future__ import annotations

import hashlib
import json
import re
import time
import tomllib
from pathlib import Path

from benchmarks.telemetry import Sampler, summarize

ROOT = Path(__file__).resolve().parents[1]
BENCH_CONFIG = ROOT / "config" / "benchmark.toml"
LOCK = ROOT / "config" / "benchmark.lock"

# runtime key -> llama-bench flag
BENCH_FLAGS = {
    "n_gpu_layers": "-ngl", "flash_attn": "-fa", "cache_type_k": "-ctk",
    "cache_type_v": "-ctv", "batch_size": "-b", "ubatch_size": "-ub",
    "threads": "-t", "split_mode": "-sm", "main_gpu": "-mg",
}
# llama.cpp b12021 replaced llama-bench's -mmp 0|1 with -lm/--load-mode; the runtime
# key stays "mmap" (0/1) and is translated here.
LOAD_MODE = {0: "none", 1: "mmap"}
# runtime key -> llama-perplexity flag (subset that affects numerics/fit)
PPL_FLAGS = {
    "n_gpu_layers": "-ngl", "flash_attn": "-fa", "cache_type_k": "-ctk",
    "cache_type_v": "-ctv", "batch_size": "-b", "ubatch_size": "-ub", "threads": "-t",
}


class BenchmarkError(Exception):
    pass


# ------------------------------------------------------------ versioning

def load_config(path: Path = BENCH_CONFIG) -> dict:
    return tomllib.loads(path.read_text())


def fingerprint(root: Path = ROOT) -> str:
    """Hash of everything that defines the benchmark methodology."""
    h = hashlib.sha256()
    files = sorted((root / "benchmarks").rglob("*.py")) + [root / "config" / "benchmark.toml"]
    for f in files:
        if "__pycache__" in f.parts:
            continue
        h.update(str(f.relative_to(root)).encode())
        h.update(f.read_bytes())
    return h.hexdigest()


def verify_lock(root: Path = ROOT) -> dict:
    lock_path = root / "config" / "benchmark.lock"
    if not lock_path.exists():
        raise BenchmarkError("no benchmark lock; a human must run `python3 -m lab approve-benchmark`")
    lock = json.loads(lock_path.read_text())
    fp = fingerprint(root)
    if lock.get("fingerprint") != fp:
        raise BenchmarkError("benchmark code/config changed since it was approved "
                             f"(lock {lock.get('fingerprint', '')[:12]} != current {fp[:12]}); "
                             "re-approve with `python3 -m lab approve-benchmark`")
    return lock


def approve(note: str, root: Path = ROOT) -> dict:
    cfg = load_config(root / "config" / "benchmark.toml")
    lock = {"version": cfg["version"], "fingerprint": fingerprint(root),
            "approved_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "note": note}
    (root / "config" / "benchmark.lock").write_text(json.dumps(lock, indent=2) + "\n")
    return lock


# ------------------------------------------------------------ inputs

def sha256_file(path: Path, cache_path: Path | None = None) -> str:
    st = path.stat()
    key = f"{path.resolve()}:{st.st_size}:{st.st_mtime_ns}"
    cache = {}
    if cache_path and cache_path.exists():
        try:
            cache = json.loads(cache_path.read_text())
        except ValueError:
            cache = {}
    if key in cache:
        return cache[key]
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    digest = h.hexdigest()
    if cache_path:
        cache[key] = digest
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(cache))
    return digest


def resolve_workload(cfg: dict, name: str, model_override: str | None = None,
                     root: Path = ROOT) -> dict:
    """Return the workload with an absolute, hash-verified model path.

    model_override lets a quantisation experiment swap the model file, but only
    to another file of the same family (same reference logits), declared in
    [workloads.*] with its own sha256."""
    if name not in cfg.get("workloads", {}):
        raise BenchmarkError(f"unknown workload {name!r}")
    wl = dict(cfg["workloads"][name], name=name)
    if model_override and model_override != wl["model"]:
        same_family = [w for w in cfg["workloads"].values()
                       if w["model"] == model_override and w["family"] == wl["family"]]
        if not same_family:
            raise BenchmarkError(f"model {model_override} is not a declared workload of family "
                                 f"{wl['family']}; different models are not comparable")
        wl.update(same_family[0], name=name)
    path = (root / wl["model"]).resolve()
    if not path.exists():
        raise BenchmarkError(f"model file missing: {path}")
    if not wl.get("sha256"):
        raise BenchmarkError(f"workload {name} has no sha256 pinned in config/benchmark.toml")
    digest = sha256_file(path, root / "experiments" / ".hashcache.json")
    if digest != wl["sha256"]:
        raise BenchmarkError(f"model hash mismatch for {path.name}")
    wl["model_path"] = str(path)
    return wl


# ------------------------------------------------------------ llama-bench

def bench_argv(bin_dir: Path, runtime: dict, wl: dict, defaults: dict) -> list[str]:
    argv = [str(bin_dir / "llama-bench"), "-m", wl["model_path"],
            "-p", ",".join(map(str, defaults["n_prompt"])),
            "-n", ",".join(map(str, defaults["n_gen"])),
            "-r", str(defaults["repetitions"]), "-o", "json"]
    for key, flag in BENCH_FLAGS.items():
        if runtime.get(key) is not None:
            argv += [flag, str(runtime[key])]
    if runtime.get("mmap") is not None:
        argv += ["-lm", LOAD_MODE[int(runtime["mmap"])]]
    return argv


def parse_bench_json(text: str) -> dict:
    """llama-bench -o json -> {"pp512": {"avg":..,"std":..,"samples":[..]}, "tg128": {...}}"""
    m = re.search(r"\[\s*{.*}\s*\]", text, re.S)
    if not m:
        raise BenchmarkError("llama-bench produced no JSON")
    out = {}
    for row in json.loads(m.group(0)):
        npp, ngen = row.get("n_prompt", 0), row.get("n_gen", 0)
        name = f"pp{npp}" if ngen == 0 else (f"tg{ngen}" if npp == 0 else f"pp{npp}+tg{ngen}")
        out[name] = {"avg": row.get("avg_ts"), "std": row.get("stddev_ts"),
                     "samples": row.get("samples_ts") or [row.get("avg_ts")]}
        out.setdefault("_meta", {"build_commit": row.get("build_commit"),
                                 "backends": row.get("backends") or row.get("gpu_info"),
                                 "model_type": row.get("model_type"),
                                 "model_size": row.get("model_size")})
    return out


def wait_cooldown(sampler: Sampler | None, idle_temp: float | None, delta: float, timeout: float):
    if sampler is None or idle_temp is None:
        return None
    t0 = time.time()
    while time.time() - t0 < timeout:
        s = sampler.sample_once()
        if not s.gpu_ok or s.temp_c is None or s.temp_c <= idle_temp + delta:
            return round(time.time() - t0, 1)
        time.sleep(5)
    return round(time.time() - t0, 1)


def run_bench_once(bin_dir: Path, runtime: dict, env: dict, wl: dict, defaults: dict, *,
                   limits, sampler, log_path: Path | None = None) -> dict:
    from lab.safety import run_guarded  # safety lives with the controller
    argv = bench_argv(bin_dir, runtime, wl, defaults)
    t0 = time.time()
    res = run_guarded(argv, limits=limits, sampler=sampler, budget_s=limits.experiment_time_budget_s,
                      env=env, gpu_expected=runtime.get("n_gpu_layers", 0) not in (0, None),
                      log_path=log_path)
    t1 = time.time()
    tele = summarize(sampler.window(t0, t1)) if sampler else {}
    out = {"argv": argv, "returncode": res.returncode, "aborted": res.aborted,
           "safety_events": res.events, "duration_s": res.duration_s,
           "peak_rss_mb": round(res.max_rss_kb / 1024, 1) if res.max_rss_kb else None,
           "telemetry": tele, "errors": []}
    if res.aborted or res.returncode != 0:
        out["errors"].append(res.aborted or f"exit {res.returncode}: {res.stderr.strip()[-500:]}")
        return out
    for pat in ("CUDA error", "out of memory", "Xid", "illegal memory access", "failed to allocate"):
        if pat.lower() in (res.stderr + res.stdout).lower():
            out["errors"].append(f"stderr contains {pat!r}")
    try:
        out["tests"] = parse_bench_json(res.stdout)
    except (BenchmarkError, ValueError) as e:
        out["errors"].append(str(e))
    return out


def derive_metrics(runs: list[dict], defaults: dict) -> dict:
    """Collapse rounds into per-metric trial lists (higher-is-better unless noted)."""
    trials: dict[str, list[float]] = {}
    pp_key = f"pp{defaults['n_prompt'][0]}"
    tg_key = f"tg{defaults['n_gen'][0]}"
    for r in runs:
        tests = r.get("tests") or {}
        for name, t in tests.items():
            if name.startswith("_"):
                continue
            trials.setdefault(f"{name}_ts", []).extend(x for x in t["samples"] if x)
        if pp_key in tests and tg_key in tests:
            pp, tg = tests[pp_key]["avg"], tests[tg_key]["avg"]
            # Derived, not measured end-to-end: prompt ingest + first decode step.
            trials.setdefault("ttft_s", []).append(int(pp_key[2:]) / pp + 1 / tg)
        tele = r.get("telemetry") or {}
        if "vram_used_mib" in tele:
            trials.setdefault("vram_peak_mib", []).append(tele["vram_used_mib"]["max"])
        if r.get("peak_rss_mb"):
            trials.setdefault("host_rss_peak_mb", []).append(r["peak_rss_mb"])
        if "power_w" in tele and tg_key in tests and tele["power_w"]["mean"]:
            # Approximate: decode tok/s per mean watt over the whole run.
            trials.setdefault("tokens_per_joule", []).append(tests[tg_key]["avg"] / tele["power_w"]["mean"])
        if "temp_c" in tele:
            trials.setdefault("temp_max_c", []).append(tele["temp_c"]["max"])
    return trials


# ------------------------------------------------------------ quality

def ppl_argv(bin_dir: Path, runtime: dict, wl: dict, quality: dict, ref_logits: Path | None,
             root: Path = ROOT) -> list[str]:
    argv = [str(bin_dir / "llama-perplexity"), "-m", wl["model_path"],
            "-f", str(root / quality["dataset"]), "--chunks", str(quality["chunks"]),
            "-c", "512"]
    if ref_logits and ref_logits.exists():
        argv += ["--kl-divergence-base", str(ref_logits), "--kl-divergence"]
    for key, flag in PPL_FLAGS.items():
        if runtime.get(key) is not None:
            argv += [flag, str(runtime[key])]
    return argv


def parse_ppl(text: str) -> dict:
    out = {}
    m = re.search(r"Mean\s+KLD:\s*([0-9.]+)", text)
    if m:
        out["mean_kld"] = float(m.group(1))
    m = re.search(r"Mean PPL\(Q\)\s*:\s*([0-9.]+)", text) or \
        re.search(r"Final estimate:\s*PPL\s*=\s*([0-9.]+)", text)
    if m:
        out["ppl"] = float(m.group(1))
    return out


def run_quality(bin_dir: Path, runtime: dict, env: dict, wl: dict, cfg: dict, *, limits,
                sampler, log_path: Path | None = None, root: Path = ROOT) -> dict:
    from lab.safety import run_guarded
    q = cfg["quality"]
    ref = cfg.get("families", {}).get(wl["family"], {}).get("reference_logits")
    ref_path = (root / ref) if ref else None
    argv = ppl_argv(bin_dir, runtime, wl, q, ref_path, root)
    res = run_guarded(argv, limits=limits, sampler=sampler, budget_s=limits.experiment_time_budget_s,
                      env=env, gpu_expected=runtime.get("n_gpu_layers", 0) not in (0, None),
                      log_path=log_path)
    out = {"argv": argv, "returncode": res.returncode, "aborted": res.aborted,
           "has_reference": bool(ref_path and ref_path.exists())}
    if res.returncode == 0 and not res.aborted:
        out.update(parse_ppl(res.stdout + res.stderr))
    return out


# ------------------------------------------------------------ decode-path quality

def decode_quality_argv(bin_dir: Path, runtime: dict, wl: dict, cfg: dict,
                        root: Path = ROOT) -> list[str] | None:
    """llama-perplexity with one token per ubatch, so batch-1 (decode) kernels make the logits.

    The batched quality run above uses 512-token ubatches, which only exercises the
    prompt-processing kernels (MMQ, flash-attention tile). Decode uses different
    kernels (MMVQ, flash-attention vec), so a patch could break generation and still
    pass. No reference logits: in --kl-divergence mode llama-perplexity always
    replays every chunk in the reference file, which is too slow at ubatch 1, so the
    check compares perplexity with the baseline's own decode run instead."""
    d = (cfg.get("quality") or {}).get("decode") or {}
    if not d.get("enabled", False):
        return None
    rt = dict(runtime, ubatch_size=int(d.get("ubatch_size", 1)))
    return ppl_argv(bin_dir, rt, wl, dict(cfg["quality"], chunks=int(d.get("chunks", 2))),
                    None, root)


def run_decode_quality(bin_dir: Path, runtime: dict, env: dict, wl: dict, cfg: dict, *, limits,
                       sampler, log_path: Path | None = None, root: Path = ROOT) -> dict | None:
    from lab.safety import run_guarded
    argv = decode_quality_argv(bin_dir, runtime, wl, cfg, root)
    if argv is None:
        return None
    res = run_guarded(argv, limits=limits, sampler=sampler, budget_s=limits.experiment_time_budget_s,
                      env=env, gpu_expected=runtime.get("n_gpu_layers", 0) not in (0, None),
                      log_path=log_path)
    out = {"argv": argv, "returncode": res.returncode, "aborted": res.aborted,
           "duration_s": res.duration_s}
    if res.returncode == 0 and not res.aborted:
        out.update(parse_ppl(res.stdout + res.stderr))
    return out


# ------------------------------------------------------------ kernel check

_ANSI = re.compile(r"\x1b\[[0-9;]*m")
_CASE = re.compile(r"^\s+([A-Za-z0-9_]+)\((.*)\): (.*)$")


def parse_backend_ops(text: str) -> dict:
    """test-backend-ops console output -> passed count, failed case names, completion."""
    passed, failed, unsupported, complete = 0, [], 0, False
    for raw in text.splitlines():
        line = _ANSI.sub("", raw).rstrip()
        if re.search(r"\d+/\d+ tests passed", line):
            complete = True
            continue
        m = _CASE.match(line)
        if not m:
            continue
        tail = m.group(3)
        if tail.endswith("FAIL"):
            failed.append(f"{m.group(1)}({m.group(2)})")
        elif tail.endswith("OK"):
            passed += 1
        elif tail.startswith("not supported"):
            unsupported += 1
    return {"passed": passed, "failed": failed, "not_supported": unsupported, "complete": complete}


def kernel_check_argvs(bin_dir: Path, cfg: dict) -> list[list[str]]:
    kc = cfg.get("kernel_check") or {}
    if not kc.get("enabled", False):
        return []
    out = []
    for suite in kc.get("suites", []):
        argv = [str(bin_dir / "test-backend-ops"), "test", "-b", kc.get("backend", "CUDA0"),
                "-o", suite["ops"]]
        if suite.get("params"):
            argv += ["-p", suite["params"]]
        out.append(argv)
    return out


def run_kernel_check(bin_dir: Path, env: dict, cfg: dict, *, limits, sampler,
                     log_dir: Path | None = None, name: str = "") -> dict | None:
    """Every configured op compared with the CPU backend (NMSE), per test-backend-ops."""
    from lab.safety import run_guarded
    argvs = kernel_check_argvs(bin_dir, cfg)
    if not argvs:
        return None
    suites, failed, passed, complete = [], [], 0, True
    for i, argv in enumerate(argvs):
        res = run_guarded(argv, limits=limits, sampler=sampler,
                          budget_s=limits.experiment_time_budget_s, env=env, gpu_expected=True,
                          log_path=(log_dir / f"{name}-kernels-{i}.log") if log_dir else None)
        r = parse_backend_ops(res.stdout)
        ok = r["complete"] and not res.aborted
        complete = complete and ok
        failed += r["failed"]
        passed += r["passed"]
        suites.append({"argv": argv[1:], "returncode": res.returncode, "aborted": res.aborted,
                       "duration_s": res.duration_s, "passed": r["passed"],
                       "failed": len(r["failed"]), "not_supported": r["not_supported"],
                       "complete": ok})
    return {"complete": complete, "passed": passed, "failed": sorted(set(failed)), "suites": suites}


# ------------------------------------------------------------ top level

def benchmark_context(cfg: dict) -> dict:
    lock = verify_lock()
    return {"benchmark_version": cfg["version"], "benchmark_fingerprint": lock["fingerprint"]}
