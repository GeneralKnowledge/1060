"""E7: one llama-server variant through the fixed speed workload (speculative-decoding stacking).

Starts llama-server (-np 1 -c 4096 plus the variant's args), optionally sends a near-full-context "fit" request
first (the CUDA VMM pool grows with the prompt, so a variant only counts as fitting if it survives that), then
sends the 6 fixed prompts greedily with thinking off (max_tokens 256) and appends one JSON line per prompt to
OUT/runs.jsonl, with server timings (incl. draft_n / draft_n_accepted when reported), peak VRAM and min MemAvailable.

  GGML_NO_BACKTRACE=1 python3 experiments/lowbit/e7_spec.py --bin BIN --out DIR --name A0 --model M.gguf \
      --args "-ngl 99 -fa 1 ..." [--fit-prompt FILE]
Stdlib only.
"""
import argparse
import json
import os
import subprocess
import threading
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).parent
PORT = 8099
MATH_SUFFIX = ("\n\nSolve the problem. Think step by step briefly, then give the final answer on the "
               "last line in the form 'Answer: <number>'.")


def workload():
    def first(f, n):
        return [json.loads(l) for l in (HERE / f).read_text().splitlines() if l.strip()][:n]
    p = [(q["id"], "gsm8k", q["question"] + MATH_SUFFIX) for q in first("gsm8k-test-first50.jsonl", 3)]
    p += [(q["id"], "math-hard", q["question"] + MATH_SUFFIX) for q in first("math500-l45-int-first30.jsonl", 2)]
    p += [(q["id"], "code-hard", q["prompt"]) for q in first("code-tasks-hard-v0.jsonl", 1)]
    return p


def post(path, body, timeout=1800):
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}{path}", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


class Sampler(threading.Thread):
    """peak VRAM (nvidia-smi) and min MemAvailable, every 0.25 s"""
    def __init__(self):
        super().__init__(daemon=True)
        self.vram, self.avail, self.stop = 0, 10 ** 9, False

    def run(self):
        while not self.stop:
            try:
                v = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                                   capture_output=True, text=True, timeout=5).stdout.strip()
                self.vram = max(self.vram, int(v.splitlines()[0]))
                for line in open("/proc/meminfo"):
                    if line.startswith("MemAvailable"):
                        self.avail = min(self.avail, int(line.split()[1]) // 1024)
            except Exception:
                pass
            time.sleep(0.25)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--args", required=True)
    ap.add_argument("--fit-prompt")
    ap.add_argument("--max-tokens", type=int, default=256)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cmd = [str(Path(a.bin) / "llama-server"), "-m", a.model, "-np", "1", "-c", "4096", "--jinja",
           "--host", "127.0.0.1", "--port", str(PORT), *a.args.split()]
    env = dict(os.environ, GGML_NO_BACKTRACE="1")
    log = open(out / f"server_{a.name}.log", "w")
    smp = Sampler(); smp.start()
    t0 = time.time()
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)
    base = {"variant": a.name, "model": Path(a.model).name, "args": a.args}
    status, ready_s, fit = "ok", None, None
    try:
        while True:
            if proc.poll() is not None:
                raise RuntimeError(f"llama-server exited with {proc.returncode} during load")
            if time.time() - t0 > 900:
                raise TimeoutError("not ready after 900 s")
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2) as r:
                    if r.status == 200:
                        break
            except Exception:
                pass
            time.sleep(1)
        ready_s = round(time.time() - t0, 1)
        if a.fit_prompt:
            r = post("/completion", {"prompt": Path(a.fit_prompt).read_text(), "n_predict": 16, "temperature": 0, "seed": 1})
            fit = r.get("timings", {}).get("prompt_n")
        with open(out / "runs.jsonl", "a") as fo:
            for pid, cat, prompt in workload():
                t1 = time.time()
                r = post("/v1/chat/completions", {
                    "messages": [{"role": "user", "content": prompt}], "temperature": 0, "seed": 1,
                    "max_tokens": a.max_tokens, "chat_template_kwargs": {"enable_thinking": False}})
                tm = r.get("timings", {})
                row = {**base, "id": pid, "cat": cat, "wall_s": round(time.time() - t1, 2),
                       "text": r["choices"][0]["message"].get("content") or "",
                       "finish": r["choices"][0].get("finish_reason"), **{k: tm.get(k) for k in (
                           "prompt_n", "prompt_per_second", "predicted_n", "predicted_per_second",
                           "draft_n", "draft_n_accepted")}}
                fo.write(json.dumps(row) + "\n"); fo.flush()
                acc = (f" acc {tm['draft_n_accepted']}/{tm['draft_n']}" if tm.get("draft_n") else "")
                print(f"  {a.name} {pid}: {tm.get('predicted_n')} tok, {tm.get('predicted_per_second', 0):.2f} tok/s{acc}", flush=True)
    except Exception as e:
        status = f"FAILED: {e}"
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
        smp.stop = True; smp.join(timeout=5)
        log.close()
    summ = {**base, "status": status, "ready_s": ready_s, "fit_prompt_tokens": fit,
            "peak_vram_mib": smp.vram, "min_mem_available_mib": smp.avail}
    with open(out / "variants.jsonl", "a") as fo:
        fo.write(json.dumps(summ) + "\n")
    print(f"[{a.name}] {status} ready {ready_s}s peak VRAM {smp.vram} MiB min MemAvailable {smp.avail} MiB", flush=True)


if __name__ == "__main__":
    main()
