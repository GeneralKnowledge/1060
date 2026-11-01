"""Workload smoke test and reference-logit generation.

`smoke` answers "does this model work on this card at all?" before anything is
benchmarked: the model file loads, the CUDA backend is used, every layer is
offloaded, text comes out, and how much VRAM / host RAM / temperature / clock it
took. It works before the model hash is pinned (it prints the hash to pin) and
writes a record to experiments/smoke/. It is not a measurement: nothing is
promoted or compared.

`reference` writes the family's Q8_0 reference logits for the KL-divergence gate,
once per family. Each family has its own file; another family's is never reused.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from benchmarks import benchmark as bench
from benchmarks.telemetry import summarize
from lab.experiment import now

# print_info / model-loader lines worth keeping: architecture, sizes, tokenizer identity.
INFO_KEYS = ("arch", "file type", "file size", "model type", "model params", "n_layer", "n_head",
             "n_head_kv", "n_embd", "n_embd_head_k", "n_embd_head_v", "n_ctx_train", "vocab type",
             "n_vocab", "general.name", "ssm_d_conv", "ssm_d_inner", "ssm_d_state",
             "ssm_n_group", "full_attention_interval")
KV_KEYS = ("general.architecture", "general.name", "general.basename", "general.size_label",
           "tokenizer.ggml.model", "tokenizer.ggml.pre")


class SmokeError(Exception):
    pass


def _model_file(c, rel: str, pinned: str | None) -> dict:
    path = (c.root / rel).resolve()
    if not path.exists():
        raise SmokeError(f"model file missing: {path}")
    digest = bench.sha256_file(path, c.root / "experiments" / ".hashcache.json")
    if pinned and digest != pinned:
        raise SmokeError(f"model hash mismatch for {path.name}: file {digest}, pinned {pinned}")
    return {"path": str(path), "sha256": digest, "pinned": bool(pinned),
            "size_bytes": path.stat().st_size}


def smoke_argv(bin_dir: Path, model_path: str, runtime: dict, ctx: int, prompt: str,
               n_predict: int) -> list[str]:
    argv = [str(bin_dir / "llama-completion"), "-m", model_path, "-p", prompt,
            "-n", str(n_predict), "-c", str(ctx), "--temp", "0", "--seed", "1", "-no-cnv",
            "-ngl", str(runtime.get("n_gpu_layers", 99)),
            "-fa", "on" if runtime.get("flash_attn") else "off"]
    for key, flag in (("cache_type_k", "-ctk"), ("cache_type_v", "-ctv"),
                      ("batch_size", "-b"), ("ubatch_size", "-ub"), ("threads", "-t")):
        if runtime.get(key) is not None:
            argv += [flag, str(runtime[key])]
    return argv


def parse_load_log(text: str) -> dict:
    out: dict = {"model_info": {}, "gguf_kv": {}}
    for line in text.splitlines():
        m = re.match(r"\s*print_info:\s*([^=]+?)\s*=\s*(.*)$", line)
        if m and m.group(1).strip() in INFO_KEYS:
            out["model_info"][m.group(1).strip()] = m.group(2).strip()
        m = re.match(r".*-\s*kv\s+\d+:\s+(\S+)\s+\S+\s+=\s+(.*)$", line)
        if m and m.group(1) in KV_KEYS:
            out["gguf_kv"][m.group(1)] = m.group(2).strip()
        if "tokenizer.chat_template" in line:
            out["has_chat_template"] = True
    m = re.search(r"offloaded\s+(\d+)\s*/\s*(\d+)\s+layers to GPU", text)
    if m:
        out["offloaded_layers"], out["total_layers"] = int(m.group(1)), int(m.group(2))
    out["cuda_devices"] = [d.strip() for d in
                           re.findall(r"Device \d+:\s*([^\n]+)", text)]
    out["cuda_backend"] = bool(re.search(r"ggml_cuda_init:\s*found\s+[1-9]", text)) and \
        bool(re.search(r"CUDA0", text))
    bufs = {}
    for name, size in re.findall(r"(CUDA0[^=\n]*?buffer size|CPU[^=\n]*?buffer size)\s*=\s*([0-9.]+)\s*MiB",
                                 text):
        key = re.sub(r"\s+", " ", name.strip())
        bufs[key] = round(bufs.get(key, 0.0) + float(size), 2)
    out["buffers_mib"] = bufs
    for kind, pat in (("prompt", r"prompt eval time\s*=.*?([0-9.]+)\s*tokens per second"),
                      ("eval", r"\beval time\s*=.*?([0-9.]+)\s*tokens per second")):
        hits = re.findall(pat, text)
        if hits:
            out[f"{kind}_tokens_per_s"] = float(hits[-1])
    out["cuda_error"] = bool(re.search(r"CUDA error|out of memory|GGML_ASSERT", text))
    return out


def smoke(c, *, prompt: str = "The three primary colours are", n_predict: int = 64) -> dict:
    from lab.safety import run_guarded
    best = c.best()
    wl = c.cfg["workloads"][c.workload]
    ts = now().replace(":", "")
    log_dir = c.root / "experiments" / "smoke"
    log_dir.mkdir(parents=True, exist_ok=True)
    rec = {"id": f"smoke-{c.workload}-{ts}", "kind": "smoke", "started_utc": now(),
           "workload": c.workload, "family": wl.get("family"), "arch_expected": wl.get("arch"),
           "config": {k: best.get(k) for k in ("backend", "build", "runtime", "env")},
           "prompt": prompt, "n_predict": n_predict}
    try:
        rec["model"] = _model_file(c, best.get("model") or wl["model"], wl.get("sha256"))
        bin_dir = Path(c.builder({"backend": best["backend"], **best["build"]}, c.limits, c.root,
                                 log_dir))
        argv = smoke_argv(bin_dir, rec["model"]["path"], best["runtime"], int(wl.get("ctx", 4096)),
                          prompt, n_predict)
        sampler = c._sampler()
        if sampler:
            with sampler:
                res = run_guarded(argv, limits=c.limits, sampler=sampler, budget_s=600,
                                  env=best.get("env", {}), gpu_expected=True,
                                  log_path=log_dir / f"{rec['id']}.log")
            rec["telemetry"] = summarize(sampler.samples)
        else:
            res = run_guarded(argv, limits=c.limits, sampler=None, budget_s=600,
                              env=best.get("env", {}), gpu_expected=True,
                              log_path=log_dir / f"{rec['id']}.log")
        rec["argv"] = argv
        rec["returncode"], rec["aborted"] = res.returncode, res.aborted
        rec["load"] = parse_load_log(res.stderr + "\n" + res.stdout)
        text = res.stdout.strip()
        rec["generated_text"] = text[-2000:]
        rec["safety_events"] = getattr(res, "safety_events", None)
        ld = rec["load"]
        arch = ld["model_info"].get("arch") or ld["gguf_kv"].get("general.architecture")
        checks = {
            "exit_ok": res.returncode == 0 and not res.aborted,
            "arch_matches": wl.get("arch") is None or arch == wl.get("arch"),
            "cuda_backend": best["backend"] != "cuda" or ld["cuda_backend"],
            "all_layers_offloaded": best["runtime"].get("n_gpu_layers", 0) < 99 or (
                "offloaded_layers" in ld and ld["offloaded_layers"] == ld["total_layers"]),
            "generated_text": len(text) > len(prompt) and _mostly_printable(text),
            "no_cuda_error": not ld["cuda_error"],
        }
        rec["checks"] = checks
        rec["ok"] = all(checks.values())
    except (SmokeError, bench.BenchmarkError) as e:
        rec.update(ok=False, error=str(e))
    rec["finished_utc"] = now()
    (log_dir / f"{rec['id']}.json").write_text(json.dumps(rec, indent=2, default=str) + "\n")
    return rec


def _mostly_printable(text: str) -> bool:
    if not text:
        return False
    good = sum(ch.isprintable() or ch in "\n\t" for ch in text)
    return good / len(text) > 0.95


def summary(rec: dict) -> dict:
    keep = ("id", "ok", "error", "checks", "returncode")
    out = {k: rec.get(k) for k in keep if k in rec}
    if "model" in rec:
        out["model"] = {k: rec["model"][k] for k in ("path", "sha256", "pinned")}
        if not rec["model"]["pinned"]:
            out["pin_this_sha256"] = rec["model"]["sha256"]
    ld = rec.get("load") or {}
    out["load"] = {k: ld.get(k) for k in ("offloaded_layers", "total_layers", "buffers_mib",
                                          "prompt_tokens_per_s", "eval_tokens_per_s")}
    out["model_info"] = ld.get("model_info")
    t = rec.get("telemetry") or {}
    out["telemetry"] = {k: t.get(k) for k in ("vram_used_mib", "temp_c", "sm_mhz", "mem_mhz",
                                              "power_w", "host_mem_available_mb_min")}
    out["generated_text_tail"] = (rec.get("generated_text") or "")[-300:]
    return out


def reference(c, *, ngl: int = 99, force: bool = False) -> dict:
    """Write the family's reference logits from its Q8_0 model (llama-perplexity
    --kl-divergence-base), with the same dataset, chunks and context as the gate."""
    from lab.safety import run_guarded
    best = c.best()
    wl = c.cfg["workloads"][c.workload]
    fam = c.cfg.get("families", {}).get(wl["family"], {})
    if not fam.get("reference_logits") or not fam.get("reference_model"):
        raise SmokeError(f"family {wl['family']} needs reference_logits and reference_model "
                         "in config/benchmark.toml")
    out = c.root / fam["reference_logits"]
    if out.exists() and not force:
        raise SmokeError(f"{out} exists; references are written once (use --force to redo)")
    model = _model_file(c, fam["reference_model"], fam.get("reference_model_sha256"))
    q = c.cfg["quality"]
    bin_dir = Path(c.builder({"backend": best["backend"], **best["build"]}, c.limits, c.root,
                             c.root / "experiments" / "logs"))
    argv = [str(bin_dir / "llama-perplexity"), "-m", model["path"], "-f", str(c.root / q["dataset"]),
            "--chunks", str(q["chunks"]), "-c", "512", "-ngl", str(ngl),
            "--kl-divergence-base", str(out)]
    log = c.root / "experiments" / "logs" / f"reference-{wl['family']}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    res = run_guarded(argv, limits=c.limits, sampler=None,
                      budget_s=c.limits.experiment_time_budget_s, gpu_expected=ngl > 0,
                      log_path=log)
    ok = res.returncode == 0 and not res.aborted and out.exists()
    return {"ok": ok, "argv": argv, "reference_model": model, "output": str(out),
            "output_bytes": out.stat().st_size if out.exists() else 0,
            "ppl": bench.parse_ppl(res.stdout + res.stderr).get("ppl"), "log": str(log)}
