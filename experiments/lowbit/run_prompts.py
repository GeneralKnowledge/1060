"""Fixed prompt set for comparing different models (KLD cannot do that).

Starts llama-server per model with the lab's qwen35-4b runtime flags, sends every prompt greedily
with thinking off, and writes prompts.jsonl plus a side-by-side prompts.md. Stdlib only.

  python3 experiments/lowbit/run_prompts.py --bin BIN --out DIR --models A.gguf B.gguf \
      --prompts experiments/lowbit/prompts-v0.jsonl
"""
import argparse
import json
import subprocess
import time
import urllib.request
from pathlib import Path

def _free_port():
    import socket
    with socket.socket() as sk:
        sk.bind(("127.0.0.1", 0))
        return sk.getsockname()[1]


# A port of our own, so a server someone else left on 8099 can never answer for ours.
PORT = _free_port()
RT = ["-ngl", "99", "-fa", "on", "-ctk", "f16", "-ctv", "f16", "-b", "2048", "-ub", "256"]


def post(path, body, timeout=600):
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}{path}", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def wait_ready(proc, timeout=180):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            raise RuntimeError(f"llama-server exited with {proc.returncode}")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2) as r:
                if r.status == 200 and proc.poll() is None:
                    return time.time() - t0
        except Exception:
            pass
        time.sleep(1)
    raise TimeoutError("llama-server did not become ready")


def chat(messages):
    t0 = time.time()
    r = post("/v1/chat/completions", {
        "messages": messages, "temperature": 0, "seed": 1, "max_tokens": 384,
        "chat_template_kwargs": {"enable_thinking": False}})
    return r["choices"][0]["message"].get("content") or "", r.get("timings", {}), time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--models", nargs="+", required=True)
    a = ap.parse_args()
    prompts = [json.loads(l) for l in Path(a.prompts).read_text().splitlines() if l.strip()]
    out = Path(a.out)
    rows = []
    stream = open(out / "prompts.jsonl", "w")  # one row per finished prompt, kept if a later model fails
    for model in a.models:
        tag = Path(model).stem
        log = open(out / f"server_{tag}.log", "w")
        proc = subprocess.Popen([str(Path(a.bin) / "llama-server"), "-m", model, "-c", "4096", "--jinja",
                                 "--host", "127.0.0.1", "--port", str(PORT), *RT],
                                stdout=log, stderr=subprocess.STDOUT)
        try:
            ready_s = wait_ready(proc)
            for p in prompts:
                turns = p.get("turns") or [p["prompt"]]
                msgs, answers, timings = [], [], []
                for t in turns:
                    msgs.append({"role": "user", "content": t})
                    text, tm, wall = chat(msgs)
                    msgs.append({"role": "assistant", "content": text})
                    answers.append(text)
                    timings.append({"wall_s": round(wall, 2), **{k: tm.get(k) for k in (
                        "prompt_n", "prompt_ms", "predicted_n", "predicted_per_second")}})
                final = answers[-1]
                exp = p.get("expect", "")
                rows.append({"model": tag, "id": p["id"], "cat": p["cat"], "answers": answers,
                             "expect": exp, "contains_expect": (exp.lower() in final.lower()) if exp else None,
                             "timings": timings, "server_ready_s": round(ready_s, 1)})
                stream.write(json.dumps(rows[-1]) + "\n")
                stream.flush()
                print(tag, p["id"], "ok" if rows[-1]["contains_expect"] in (True, None) else "MISS", flush=True)
        finally:
            proc.terminate()
            proc.wait(timeout=30)
            log.close()
    stream.close()
    md = ["# Fixed prompts, side by side", ""]
    for p in prompts:
        md += [f"## {p['id']} ({p['cat']})", "", "> " + " / ".join(p.get("turns") or [p["prompt"]]), ""]
        for r in rows:
            if r["id"] == p["id"]:
                md += [f"**{r['model']}** (expect `{r['expect']}`: {r['contains_expect']})", "", "```",
                       r["answers"][-1].strip(), "```", ""]
    (out / "prompts.md").write_text("\n".join(md))
    print("wrote", out / "prompts.md")


if __name__ == "__main__":
    main()
