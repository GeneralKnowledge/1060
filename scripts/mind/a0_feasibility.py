#!/usr/bin/env python3
"""A0 feasibility for the mind's language tiers. Measurement only; run on the laptop.

  python3 scripts/mind/a0_feasibility.py            # everything (Qwen part only if the GPU is free)
  python3 scripts/mind/a0_feasibility.py --no-gpu   # CPU tiers only

1. Downloads LFM2.5-350M Q8_0 and LFM2.5-1.2B-Thinking Q4_K_M (about 1.1 GB) into models/mind/.
2. For each, starts llama-server on the CPU (-ngl 0) with 2 and 4 threads and measures
   generation and prompt speed, resident RAM, JSON-schema constrained output, and (Thinking)
   the length of the reasoning trace.
3. If no other process uses the GPU: starts Qwen3.5-4B with the lab's winning settings and
   measures load time, VRAM, generation speed and how fast VRAM is released after stopping.
Writes docs/mind-feasibility.md and mind/reports/a0.json. Touches nothing else.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BIN = ROOT / "backends" / "builds" / "d2db58ba90808f32" / "build" / "bin"
MODELS = ROOT / "models" / "mind"
PORT = 8101
TIERS = [("LiquidAI/LFM2.5-350M-GGUF", "Q8_0"), ("LiquidAI/LFM2.5-1.2B-Thinking-GGUF", "Q4_K_M")]
QWEN = ROOT / "models" / "Qwen3.5-4B-Q4_K_M.gguf"
SCHEMA = {"type": "object", "properties": {"relation": {"type": "string"},
                                           "confidence": {"type": "number"}},
          "required": ["relation", "confidence"]}


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def fetch(repo, quant):
    MODELS.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(f"https://huggingface.co/api/models/{repo}", timeout=60) as r:
        files = [s["rfilename"] for s in json.load(r)["siblings"]]
    name = next(f for f in files if f.endswith(f"-{quant}.gguf") and "/" not in f)
    dest = MODELS / name
    if not dest.exists():
        log("downloading", name)
        tmp = dest.with_suffix(".part")
        subprocess.run(["curl", "-fL", "--retry", "4", "-o", str(tmp),
                        f"https://huggingface.co/{repo}/resolve/main/{name}"], check=True)
        tmp.rename(dest)
    return dest


def post(path, body, timeout=600):
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}{path}", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def server(model, args):
    p = subprocess.Popen([str(BIN / "llama-server"), "-m", str(model), "--port", str(PORT), *args],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    t0 = time.time()
    while time.time() - t0 < 300:
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2)
            return p, time.time() - t0
        except Exception:
            if p.poll() is not None:
                raise RuntimeError("llama-server exited")
            time.sleep(0.5)
    p.kill()
    raise RuntimeError("llama-server did not become healthy")


def rss_mb(pid):
    for ln in open(f"/proc/{pid}/status"):
        if ln.startswith("VmRSS"):
            return int(ln.split()[1]) / 1024


def measure(p):
    prompt = "Below are notes from a memory. " + " ".join(f"item {i} follows item {i - 1}." for i in range(1, 60))
    r = post("/completion", {"prompt": prompt, "n_predict": 128, "temperature": 0.1, "ignore_eos": True,
                             "cache_prompt": False})
    t = r["timings"]
    js = post("/completion", {"prompt": "Return a relation between 'a then b' and 'b then c' as JSON.",
                              "n_predict": 96, "temperature": 0.1, "json_schema": SCHEMA})
    try:
        ok = set(SCHEMA["required"]) <= set(json.loads(js["content"]))
    except (ValueError, TypeError):
        ok = False
    return dict(pp_tok_s=round(t["prompt_per_second"], 1), tg_tok_s=round(t["predicted_per_second"], 1),
                prompt_tokens=t["prompt_n"], rss_mb=round(rss_mb(p.pid)), json_schema_ok=ok)


def thinking_len():
    r = post("/v1/chat/completions", {"messages": [{"role": "user", "content":
              "Two events: x happens, then 3 steps later y happens, most of the time. What could explain it?"}],
              "max_tokens": 2048, "temperature": 0.05})
    msg = r["choices"][0]["message"]
    return dict(completion_tokens=r["usage"]["completion_tokens"],
                reasoning_chars=len(msg.get("reasoning_content") or ""), answer_chars=len(msg.get("content") or ""))


def gpu_busy():
    out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout.strip()
    return out


def vram_used():
    return int(subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                              capture_output=True, text=True).stdout.strip())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-gpu", action="store_true")
    a = ap.parse_args()
    res = dict(started=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), cpu=os.cpu_count(), tiers={})
    for repo, quant in TIERS:
        model = fetch(repo, quant)
        res["tiers"][model.name] = entry = dict(size_mb=round(model.stat().st_size / 1e6))
        for threads in (2, 4):
            log("measuring", model.name, "threads", threads)
            p, load_s = server(model, ["-ngl", "0", "-t", str(threads), "-c", "4096", "-np", "1"])
            try:
                m = measure(p)
                m["load_s"] = round(load_s, 1)
                if "Thinking" in model.name and threads == 4:
                    m["thinking"] = thinking_len()
                entry[f"t{threads}"] = m
            finally:
                p.terminate()
                p.wait(30)
    if a.no_gpu or not shutil.which("nvidia-smi"):
        res["qwen"] = "skipped (--no-gpu)"
    elif gpu_busy():
        res["qwen"] = f"skipped: GPU in use by {gpu_busy()!r}"
    else:
        best = json.loads((ROOT / "winners" / "qwen35-4b" / "best_config.json").read_text())["runtime"]
        args = ["-ngl", str(best["n_gpu_layers"]), "-fa", "on" if best["flash_attn"] else "off",
                "-b", str(best["batch_size"]), "-ub", str(best["ubatch_size"]), "-c", "8192", "-np", "1"]
        if not best.get("mmap", 1):
            args += ["-lm", "none"]  # b12021 replaced --no-mmap with --load-mode
        before = vram_used()
        log("measuring Qwen3.5-4B on the GPU")
        p, load_s = server(QWEN, args)
        q = {}
        try:
            q.update(measure(p))
            q.update(load_s=round(load_s, 1), vram_mib=vram_used() - before)
        finally:
            t0 = time.time()
            p.terminate()
            p.wait(30)
            while vram_used() > before + 100 and time.time() - t0 < 30:
                time.sleep(0.2)
            q["unload_s"] = round(time.time() - t0, 2)
        res["qwen"] = q
    (ROOT / "mind" / "reports").mkdir(parents=True, exist_ok=True)
    (ROOT / "mind" / "reports" / "a0.json").write_text(json.dumps(res, indent=1))
    lines = ["# Mind A0 feasibility (language tiers)", "", f"Run {res['started']} on the laptop "
             f"({res['cpu']} logical CPUs), llama.cpp b12021 build, measurement only.", "",
             "| model | threads | load s | prompt tok/s | gen tok/s | RSS MB | JSON schema |", "|---|---|---|---|---|---|---|"]
    for name, e in res["tiers"].items():
        for th in (2, 4):
            m = e[f"t{th}"]
            lines.append(f"| {name} ({e['size_mb']} MB) | {th} | {m['load_s']} | {m['pp_tok_s']} | {m['tg_tok_s']} | "
                         f"{m['rss_mb']} | {'ok' if m['json_schema_ok'] else 'FAILED'} |")
    for name, e in res["tiers"].items():
        if "thinking" in e.get("t4", {}):
            lines += ["", f"{name} reasoning on a short question: {e['t4']['thinking']}"]
    q = res["qwen"]
    lines += ["", "Qwen3.5-4B (GPU, lab winning settings): " + (json.dumps(q) if isinstance(q, dict) else q)]
    (ROOT / "docs" / "mind-feasibility.md").write_text("\n".join(lines) + "\n")
    log("done; see docs/mind-feasibility.md")


if __name__ == "__main__":
    sys.exit(main())
