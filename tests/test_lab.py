"""Controller / evaluator / safety / agent tests using fake llama.cpp binaries.

No GPU needed: telemetry is disabled or faked, and the build step is replaced
by a function that returns tests/fake_bin."""
import json
import re
import shutil
import subprocess
import time
from pathlib import Path

import pytest

from benchmarks import benchmark as bench
from benchmarks.telemetry import Sample, Sampler, summarize
from lab import evaluator, safety
from lab.agent import ResearchAgent, extract_json, run_session
from lab.controller import Controller
from lab.experiment import ExperimentSpec
from lab.providers.scripted import ScriptedProvider

REPO = Path(__file__).resolve().parents[1]
FAKE_BIN = Path(__file__).parent / "fake_bin"


def sh(cwd, *args):
    return subprocess.run(args, cwd=cwd, check=True, capture_output=True, text=True).stdout


INITIAL_BEST = {
    "id": "initial", "validated": False, "note": "Starting point, not yet measured.",
    "workload": "small", "backend": "cuda",
    "build": {"llama_cpp_commit": "7bd18083082126c08d375cefaa6a5aa1c170ecde",
              "llama_cpp_tag": "b12021", "cmake_flags": [], "patches": []},
    "runtime": {"n_gpu_layers": 99, "flash_attn": 0, "cache_type_k": "f16", "cache_type_v": "f16",
                "batch_size": 2048, "ubatch_size": 512, "mmap": 1},
    "env": {},
}


@pytest.fixture
def lab(tmp_path):
    root = tmp_path / "lab"
    for d in ("config", "benchmarks", "lab", "winners", "research", "experiments", "backends"):
        shutil.copytree(REPO / d, root / d, ignore=shutil.ignore_patterns("__pycache__", "*.lock"))
    shutil.copy(REPO / ".gitignore", root / ".gitignore")
    # Start every test from a fixed, unmeasured winner and an empty history, so the
    # tests don't depend on what the real lab has promoted or recorded so far.
    for d in ("winners/history", "winners/profiles", "experiments/records", "experiments/proposals"):
        shutil.rmtree(root / d, ignore_errors=True)
    (root / "experiments" / "results.tsv").unlink(missing_ok=True)
    (root / "winners" / "best_config.json").write_text(json.dumps(INITIAL_BEST, indent=2) + "\n")
    (root / "models").mkdir()
    model = root / "models" / "Llama-3.2-3B-Instruct-Q4_K_M.gguf"
    model.write_bytes(b"fake gguf")
    data = root / "benchmarks" / "workloads" / "wikitext-2-test-slice.txt"
    data.parent.mkdir(parents=True, exist_ok=True)
    data.write_text("hello world\n")
    (root / "models" / "llama-3.2-3b-q8_0.kld").write_bytes(b"ref")
    cfg = (root / "config" / "benchmark.toml").read_text()
    # Pin the fake model's hash and clear any real hashes pinned on the lab machine,
    # so the fixture works whether or not config/benchmark.toml has been filled in.
    cfg = re.sub(r'^(dataset_)?sha256 = "[^"]*"', lambda m: f'{m.group(1) or ""}sha256 = ""',
                 cfg, flags=re.M)
    cfg = re.sub(r'^sha256 = ""', f'sha256 = "{bench.sha256_file(model)}"', cfg, count=1,
                 flags=re.M)
    cfg = re.sub(r"repetitions = \d+", "repetitions = 3", cfg)
    cfg = cfg.replace("warmup = true", "warmup = false")
    (root / "config" / "benchmark.toml").write_text(cfg)
    rs = (root / "config" / "research.toml").read_text().replace("pause_between_s = 30",
                                                                    "pause_between_s = 0")
    (root / "config" / "research.toml").write_text(rs)
    bench.approve("test", root)
    sh(root, "git", "init", "-q", "-b", "research")
    sh(root, "git", "add", "-A")
    sh(root, "git", "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "init")
    c = Controller(root=root, builder=lambda spec, limits, patch_root, log_dir: FAKE_BIN,
                   use_telemetry=False)
    return c


# ---------------------------------------------------------------- parsers

def test_parse_bench_and_ppl():
    out = subprocess.run([str(FAKE_BIN / "llama-bench"), "-p", "512,2048", "-n", "128", "-r", "3"],
                         capture_output=True, text=True).stdout
    t = bench.parse_bench_json(out)
    assert set(t) >= {"pp512", "pp2048", "tg128"} and len(t["tg128"]["samples"]) == 3
    q = bench.parse_ppl("Mean PPL(Q) : 7.5\nMean    KLD:   0.0123 ± 0.001\n")
    assert q == {"mean_kld": 0.0123, "ppl": 7.5}


def test_extract_json():
    assert extract_json('blah\n```json\n{"action": "stop", "reason": "x"}\n```')["action"] == "stop"


# ---------------------------------------------------------------- evaluator

def _cfg():
    return bench.load_config(REPO / "config" / "benchmark.toml")


def _trials(tg, pp, n=6, jitter=0.002):
    return {"tg128_ts": [tg * (1 + jitter * ((i % 3) - 1)) for i in range(n)],
            "pp512_ts": [pp * (1 + jitter * ((i % 3) - 1)) for i in range(n)],
            "ttft_s": [512 / pp + 1 / tg] * n}


Q = {"returncode": 0, "ppl": 7.5, "mean_kld": 0.004}


@pytest.mark.parametrize("cand,expected", [
    ((22, 330), "promote"), ((18, 270), "discard"), ((20.02, 300.1), "inconclusive")])
def test_evaluator_decisions(cand, expected):
    ev = evaluator.evaluate(_trials(20, 300), _trials(*cand), base_quality=Q, cand_quality=Q,
                            errors=[], safety_aborts=[], cfg=_cfg())
    assert ev["decision"] == expected, ev


def test_evaluator_noise_blocks_small_gain():
    noisy = lambda x: [x * f for f in (0.9, 1.1, 0.95, 1.05, 1.0, 1.0)]
    ev = evaluator.evaluate({"tg128_ts": noisy(20), "pp512_ts": noisy(300)},
                            {"tg128_ts": noisy(20.8), "pp512_ts": noisy(312)},
                            base_quality=Q, cand_quality=Q, errors=[], safety_aborts=[], cfg=_cfg())
    assert ev["decision"] == "inconclusive"


def test_evaluator_gates():
    bad_q = dict(Q, mean_kld=0.5)
    ev = evaluator.evaluate(_trials(20, 300), _trials(30, 400), base_quality=Q, cand_quality=bad_q,
                            errors=[], safety_aborts=[], cfg=_cfg())
    assert ev["decision"] == "rejected" and "KLD" in ev["gate_failures"][0]
    ev = evaluator.evaluate(_trials(20, 300), _trials(30, 400), base_quality=Q, cand_quality=Q,
                            errors=[], safety_aborts=["gpu_overtemp"], cfg=_cfg())
    assert ev["decision"] == "rejected"


# ---------------------------------------------------------------- safety

def test_policy():
    L = safety.load_limits(REPO / "config" / "safety.toml")
    safety.check_command(["/x/llama-bench", "-m", "m.gguf"], L)
    with pytest.raises(safety.PolicyViolation):
        safety.check_command(["bash", "-c", "echo"], L)
    with pytest.raises(safety.PolicyViolation):
        safety.check_command(["nvidia-smi", "-pl", "60"], L)
    assert safety.approval_reasons("sudo nvflash --save", L)
    assert not safety.approval_reasons("GGML_CUDA_FORCE_MMQ=1 adapter", L)


def test_monitor_trips_on_overtemp():
    L = safety.load_limits(REPO / "config" / "safety.toml")
    m = safety.SafetyMonitor(L, budget_s=60)
    killed = []
    m.arm(lambda: killed.append(1))
    m.on_sample(Sample(t=time.time(), gpu_ok=True, temp_c=95, vram_used_mib=1, vram_total_mib=6144))
    m.disarm()
    assert killed and m.events[0].kind == "gpu_overtemp"


def test_run_guarded_time_budget():
    L = safety.load_limits(REPO / "config" / "safety.toml")
    L.allowed_executables.append("sleep")
    r = safety.run_guarded(["sleep", "30"], limits=L, budget_s=1, gpu_expected=False)
    assert r.aborted == "time_budget" and r.duration_s < 10


def test_telemetry_summary_with_fake_source():
    temps = iter([40, 60, 70])
    src = lambda: Sample(t=time.time(), gpu_ok=True, temp_c=next(temps), power_w=30.0,
                         vram_used_mib=1000, vram_total_mib=6144)
    s = Sampler(src)
    for _ in range(3):
        s.sample_once()
    out = summarize(s.samples)
    assert out["temp_c"]["max"] == 70 and out["vram_used_mib"]["max"] == 1000


# ---------------------------------------------------------------- controller

def test_baseline_then_promote_with_patch(lab):
    rec = lab.validate_baseline()
    assert rec["decision"] == "baseline_validated" and lab.best()["validated"]
    spec = ExperimentSpec(hypothesis="FA helps prompt processing", objective="pp", description="fa on",
                          changes={"runtime": {"flash_attn": 1},
                                   "build": {"patches": ["backends/experimental/noop.patch"]}},
                          files={"backends/experimental/noop.patch": "# placeholder\n"})
    rec = lab.run(spec)
    assert rec["decision"] == "promote", rec.get("evaluation") or rec.get("error")
    best = lab.best()
    assert best["id"] == spec.id and best["runtime"]["flash_attn"] == 1 and best["previous"] == "initial"
    assert (lab.root / "winners" / "history" / "initial.json").exists()
    assert (lab.root / "backends" / "experimental" / "noop.patch").exists()  # merged from worktree
    assert (lab.root / f"experiments/records/{spec.id}.diff").exists()
    log = sh(lab.root, "git", "log", "--oneline")
    assert f"promote {spec.id}" in log
    assert "exp/" not in sh(lab.root, "git", "branch")
    assert not (lab.root / "experiments" / "worktrees" / spec.id).exists()
    tsv = (lab.root / "experiments" / "results.tsv").read_text()
    assert spec.id in tsv and "promote" in tsv
    # Same config again is refused as a duplicate.
    again = lab.run(ExperimentSpec(hypothesis="again", objective="", description="",
                                   changes={"runtime": {"flash_attn": 0}}))
    assert again["decision"] in ("discard", "inconclusive")
    dup = lab.run(ExperimentSpec(hypothesis="again", objective="", description="",
                                 changes={"runtime": {"flash_attn": 0}}))
    assert dup["decision"] == "duplicate"


def test_rewritten_patch_is_not_a_duplicate(lab):
    """Same patch path, different content: a new experiment, not a duplicate."""
    lab.validate_baseline()
    assert "note" not in lab.best()
    patch = "backends/experimental/kernel.patch"
    first = lab.run(ExperimentSpec(hypothesis="v1", objective="", description="",
                                   changes={"build": {"patches": [patch]}},
                                   files={patch: "# version 1\n"}))
    assert first["decision"] != "duplicate"
    second = lab.run(ExperimentSpec(hypothesis="v2", objective="", description="",
                                    changes={"build": {"patches": [patch]}},
                                    files={patch: "# version 2\n"}))
    assert second["decision"] != "duplicate"
    assert second["candidate_config_hash"] != first["candidate_config_hash"]
    assert "note" not in second["experimental_config"]


def test_parse_backend_ops():
    out = subprocess.run([str(FAKE_BIN / "test-backend-ops"), "test", "-o", "MUL_MAT"],
                         capture_output=True, text=True).stdout
    r = bench.parse_backend_ops(out)
    assert r["complete"] and r["passed"] == 2 and len(r["failed"]) == 1
    assert r["failed"][0].startswith("MUL_MAT(type_a=f16")
    assert not bench.parse_backend_ops("  MUL_MAT(x): OK\n")["complete"]  # cut off: no summary


def test_decode_quality_runs_at_ubatch_1(lab):
    wl = {"model_path": "m.gguf"}
    a = bench.decode_quality_argv(Path("/x"), {"ubatch_size": 512, "flash_attn": 1}, wl, lab.cfg,
                                  lab.root)
    assert a[a.index("-ub") + 1] == "1" and "--kl-divergence" not in a
    assert a[a.index("--chunks") + 1] == str(lab.cfg["quality"]["decode"]["chunks"])


def test_known_baseline_kernel_failure_does_not_block(lab):
    lab.validate_baseline()
    rec = lab.run(ExperimentSpec(hypothesis="FA", objective="", description="",
                                 changes={"runtime": {"flash_attn": 1}}))
    assert rec["decision"] == "promote", rec.get("evaluation")
    k = rec["kernel_check"]
    assert k["baseline"]["failed_total"] == len(lab.cfg["kernel_check"]["suites"])  # one per suite
    assert k["candidate"]["failed"] == k["baseline"]["failed"]
    assert rec["decode_correctness"]["candidate"]["ppl"] == 7.5


def test_decode_only_breakage_is_rejected(lab):
    """Batched quality passes, but decode (ubatch 1) output is broken: must not promote."""
    lab.validate_baseline()
    rec = lab.run(ExperimentSpec(hypothesis="faster decode", objective="", description="",
                                 changes={"runtime": {"flash_attn": 1},
                                          "env": {"FAKE_DECODE_BROKEN": "1"}}))
    assert rec["correctness"]["candidate"]["mean_kld"] < 0.03  # the old gate alone would pass
    assert rec["decision"] == "rejected"
    assert any("decode perplexity" in g for g in rec["evaluation"]["gate_failures"])
    assert lab.best()["id"] == "initial"


def test_new_kernel_failure_is_rejected(lab):
    lab.validate_baseline()
    rec = lab.run(ExperimentSpec(hypothesis="new mmvq", objective="", description="",
                                 changes={"runtime": {"flash_attn": 1},
                                          "env": {"FAKE_KERNEL_BROKEN": "1"}}))
    assert rec["decision"] == "rejected"
    assert any("kernel check" in g for g in rec["evaluation"]["gate_failures"])


def test_regression_is_discarded_and_best_untouched(lab):
    before = lab.best()
    rec = lab.run(ExperimentSpec(hypothesis="partial offload", objective="", description="",
                                 changes={"runtime": {"n_gpu_layers": 10}}))
    assert rec["decision"] == "discard"
    assert lab.best() == before


def test_quality_gate_rejects(lab):
    rec = lab.run(ExperimentSpec(hypothesis="q4 K cache", objective="", description="",
                                 changes={"runtime": {"cache_type_k": "q4_0", "flash_attn": 1}}))
    assert rec["decision"] == "rejected" and any("KLD" in g for g in rec["evaluation"]["gate_failures"])


def test_crash_is_rejected(lab, monkeypatch):
    rec = lab.run(ExperimentSpec(hypothesis="crash", objective="", description="",
                                 changes={"env": {"FAKE_CRASH": "1"}}))
    assert rec["decision"] in ("rejected", "failed")
    assert lab.best()["id"] == "initial"


def test_protected_paths_and_hardware(lab):
    rec = lab.run(ExperimentSpec(hypothesis="sneaky", objective="", description="",
                                 changes={"runtime": {"threads": 4}},
                                 files={"benchmarks/benchmark.py": "print('win')"}))
    assert rec["decision"] == "rejected" and "may not write" in rec["error"]
    rec = lab.run(ExperimentSpec(hypothesis="lower power", objective="", description="",
                                 kind="hardware", changes={"env": {"X": "nvidia-smi -pl 50"}}))
    assert rec["decision"] == "awaiting_approval"
    assert (lab.root / "experiments" / "proposals" / f"{rec['id']}.json").exists()


def test_benchmark_tamper_detected(lab):
    (lab.root / "benchmarks" / "benchmark.py").write_text("# changed\n", )
    with pytest.raises(bench.BenchmarkError):
        bench.verify_lock(lab.root)


def test_model_swap_must_be_same_family(lab):
    with pytest.raises(bench.BenchmarkError):
        bench.resolve_workload(lab.cfg, "small", "models/other.gguf", lab.root)


# ---------------------------------------------------------------- agent

def test_agent_step_and_session(lab):
    proposal = json.dumps({"action": "experiment", "hypothesis": "FA on", "objective": "pp",
                           "description": "flash attention", "kind": "software",
                           "changes": {"runtime": {"flash_attn": 1}}})
    provider = ScriptedProvider([f"```json\n{proposal}\n```", "FA improved pp512 in this record.",
                                 '```json\n{"action": "stop", "reason": "done"}\n```'])
    agent = ResearchAgent(lab, provider=provider, root=lab.root)
    out = run_session(agent, max_experiments=5, max_hours=1, log=lambda *a: None)
    assert out["experiments"] == 1 and out["stopped_because"].startswith("agent_stop")
    assert out["results"][0]["decision"] == "promote"
    notes = (lab.root / "research" / "research_notes.md").read_text()
    assert "FA improved pp512" in notes and out["results"][0]["id"] in notes
    # The agent saw the history and the best config in its context.
    assert "Current best config" in provider.calls[0][1][0]["content"]


def test_sweep_runs_steps_against_current_best(lab):
    from lab.sweep import load_sweep, run_sweep
    lab.validate_baseline()
    f = lab.root / "experiments" / "configurations" / "sweeps" / "t.json"
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps({"name": "t", "steps": [
        {"hypothesis": "FA on", "changes": {"runtime": {"flash_attn": 1}}},
        {"hypothesis": "partial offload", "changes": {"runtime": {"n_gpu_layers": 10}}},
        {"hypothesis": "FA on again", "changes": {"runtime": {"flash_attn": 1}}}]}))
    out = run_sweep(lab, load_sweep(f), log=lambda *a: None)
    assert [r["decision"] for r in out["results"]] == ["promote", "discard", "duplicate"]
    rec = json.loads((lab.root / "experiments" / "records" / f"{out['results'][1]['id']}.json").read_text())
    assert rec["source"] == "sweep:t" and rec["baseline_config"]["runtime"]["flash_attn"] == 1


def test_shipped_sweep_files_are_valid():
    from lab.sweep import load_sweep
    files = sorted((REPO / "experiments" / "configurations" / "sweeps").glob("*.json"))
    assert files
    for f in files:
        load_sweep(f)


def test_agent_context_includes_priors(lab):
    agent = ResearchAgent(lab, provider=ScriptedProvider([]), root=lab.root)
    ctx = agent.context()
    assert "Prior knowledge (read first)" in ctx and "Open questions" in ctx


def test_session_stop_file(lab):
    (lab.root / "experiments" / "STOP").write_text("")
    agent = ResearchAgent(lab, provider=ScriptedProvider([]), root=lab.root)
    assert run_session(agent, log=lambda *a: None)["stopped_because"] == "stop_file"


# ---------------------------------------------------------------- power state

def test_power_capped_run_is_invalid(lab):
    """Firmware power throttling (e.g. worn Dell battery) must not produce a result."""
    lab.use_telemetry = True
    lab.limits.power_limit_min_w = 65
    lab.telemetry_source = lambda: Sample(t=time.time(), gpu_ok=True, temp_c=50, power_w=50,
                                          power_limit_w=55.0, vram_used_mib=1000,
                                          vram_total_mib=6144, ac_online=True, battery_pct=40)
    rec = lab.run(ExperimentSpec(hypothesis="FA", objective="", description="",
                                 changes={"runtime": {"flash_attn": 1}}))
    assert rec["decision"] == "invalid_power_state", rec.get("error")
    assert lab.best()["id"] == "initial"


def test_power_preflight_refuses_battery(lab, monkeypatch):
    import benchmarks.telemetry as tel
    monkeypatch.setattr(tel, "read_power_supply",
                        lambda: {"ac_online": False, "battery_pct": 80, "battery_status": "Discharging"})
    lab.limits.require_ac = True
    rec = lab.run(ExperimentSpec(hypothesis="FA", objective="", description="",
                                 changes={"runtime": {"flash_attn": 1}}))
    assert rec["decision"] == "failed" and "battery power" in rec["error"]


def test_audit_accepts_g3_3779():
    from lab import hardware_audit as ha
    p = {"machine": {"dmi": {"value": {"product_name": "G3 3779"}}},
         "gpu": {"name": {"value": "NVIDIA GeForce GTX 1060 with Max-Q Design"},
                 "cuda_device_attributes": {"value": None}, "architecture": {"value": "Pascal"},
                 "compute_capability": {"value": "6.1"}, "vram_total_mib": {"value": 6144}},
         "memory": {"total_bytes": {"value": 8_100_000_000}},
         "os": {"distribution": {"value": "Ubuntu 24.04.3 LTS"}}}
    assert ha.check_claims(p)["machine"]["result"] == "confirmed"


def test_bench_argv_uses_load_mode_not_mmp():
    wl = {"model_path": "m.gguf"}
    d = {"n_prompt": [512], "n_gen": [128], "repetitions": 1}
    a = bench.bench_argv(Path("/x"), {"mmap": 1, "n_gpu_layers": 99}, wl, d)
    assert "-mmp" not in a and a[a.index("-lm") + 1] == "mmap"
    a = bench.bench_argv(Path("/x"), {"mmap": 0}, wl, d)
    assert a[a.index("-lm") + 1] == "none"
