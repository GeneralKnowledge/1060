"""Checkable cross-model quality: GSM8K maths (exact numeric answer) and coding tasks (unit tests).

Greedy, thinking off, same prompts and limits for every model. Starts llama-server per model with the
qwen35-4b runtime flags (see run_prompts.py). Writes eval.jsonl and eval_summary.json. Stdlib only.

  python3 experiments/lowbit/eval_tasks.py --bin BIN --out DIR --models A.gguf B.gguf ...

Model-written code runs in a subprocess (python3 -I, temp dir, 10 s timeout).
"""
import argparse
import json
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import run_prompts as rp  # noqa: E402

HERE = Path(__file__).parent
MATH_SUFFIX = ("\n\nSolve the problem. Think step by step briefly, then give the final answer on the "
               "last line in the form 'Answer: <number>'.")
MAX_TOKENS = 512


def chat(prompt, max_tokens=MAX_TOKENS):
    t0 = time.time()
    r = rp.post("/v1/chat/completions", {
        "messages": [{"role": "user", "content": prompt}], "temperature": 0, "seed": 1,
        "max_tokens": max_tokens, "chat_template_kwargs": {"enable_thinking": False}})
    ch = r["choices"][0]
    tm = r.get("timings", {})
    return (ch["message"].get("content") or "", ch.get("finish_reason"),
            tm.get("predicted_n"), tm.get("predicted_per_second"), time.time() - t0)


def num(s):
    try:
        return float(s.replace(",", "").replace("$", "").rstrip("."))
    except ValueError:
        return None


ANSWER_RE = re.compile(r"Answer:\s*\**\s*\$?\s*(-?[\d,]*\.?\d+)(\s*/\s*-?[\d,]*\.?\d+)?(?!\d|[.,]\d|\s*/)")


def math_answer(text):
    """Value on the last 'Answer:' line, or None. A fraction such as 18/5 is evaluated, never read as its
    numerator. There is no fallback to the last number in the text: a reply cut off before its
    'Answer:' line scores as wrong."""
    i = text.rfind("Answer:")
    m = ANSWER_RE.match(text, i) if i >= 0 else None
    if not m:
        return None
    whole, frac = m.groups()
    v = num(whole)
    if v is not None and frac:
        d = num(frac.strip().lstrip("/").strip())
        v = v / d if d else None
    return v


PASS_MARK = "__ALL_TESTS_COMPLETED_7f3a__"


def _limits():
    import resource
    resource.setrlimit(resource.RLIMIT_AS, (1 << 30, 1 << 30))       # 1 GiB address space
    resource.setrlimit(resource.RLIMIT_FSIZE, (16 << 20, 16 << 20))  # 16 MiB per written file
    resource.setrlimit(resource.RLIMIT_NPROC, (64, 64))


def run_code(text, tests):
    """Run model-written code plus the unit tests. Passes only if the trusted tail runs to the end and
    prints PASS_MARK, so an early sys.exit(0) in the generated code is a failure. Output goes to files in
    the temp dir (bounded by RLIMIT_FSIZE), and the child has memory and process limits. This is not
    a filesystem or network sandbox: run it only on prompts you trust the model to answer benignly."""
    m = re.findall(r"```(?:python)?\n(.*?)```", text, re.S)
    code = m[0] if m else text
    with tempfile.TemporaryDirectory() as d:
        f = Path(d) / "t.py"
        f.write_text(code + "\n\n" + tests + f"\nprint({PASS_MARK!r})\n")
        out, err = Path(d) / "out.txt", Path(d) / "err.txt"
        try:
            with open(out, "w") as fo, open(err, "w") as fe:
                p = subprocess.run([sys.executable, "-I", str(f)], cwd=d, stdout=fo, stderr=fe,
                                   timeout=10, preexec_fn=_limits)
        except subprocess.TimeoutExpired:
            return False, "timeout"
        tail = out.read_text(errors="replace")[-2000:]
        ok = p.returncode == 0 and tail.rstrip().endswith(PASS_MARK)
        return ok, err.read_text(errors="replace")[-300:] or ("" if ok else "tests did not complete")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--math", default=str(HERE / "gsm8k-test-first50.jsonl"))
    ap.add_argument("--code", default=str(HERE / "code-tasks-v0.jsonl"))
    ap.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    ap.add_argument("--math-limit", type=int, default=0, help="only the first N maths items (0 = all)")
    ap.add_argument("--code-limit", type=int, default=0, help="only the first N coding tasks (0 = all)")
    ap.add_argument("--server-args", default=" ".join(rp.RT + ["-np", "1"]),
                    help="llama-server runtime flags (default: the qwen35-4b winner flags)")
    ap.add_argument("--ctx", type=int, default=4096)
    ap.add_argument("--tag", default="", help="suffix for output files, e.g. hard")
    a = ap.parse_args()
    maths = [json.loads(l) for l in Path(a.math).read_text().splitlines() if l.strip()]
    codes = [json.loads(l) for l in Path(a.code).read_text().splitlines() if l.strip()]
    stems = [Path(m).stem for m in a.models]
    if len(set(stems)) != len(stems):
        sys.exit(f"duplicate model file names: {stems}")
    maths = maths[:a.math_limit] if a.math_limit else maths
    codes = codes[:a.code_limit] if a.code_limit else codes
    sfx = f"_{a.tag}" if a.tag else ""
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    summary = {}
    with open(out / f"eval{sfx}.jsonl", "w") as fo:
        for model in a.models:
            tag = Path(model).stem
            log = open(out / f"eval{sfx}_server_{tag}.log", "w")
            proc = subprocess.Popen([str(Path(a.bin) / "llama-server"), "-m", model, "-c", str(a.ctx), "--jinja",
                                     "--host", "127.0.0.1", "--port", str(rp.PORT), *a.server_args.split()],
                                    stdout=log, stderr=subprocess.STDOUT)
            s = {"math_correct": 0, "math_n": len(maths), "code_pass": 0, "code_n": len(codes),
                 "hit_max_tokens": 0, "gen_tokens": 0, "tps": []}
            t_start = time.time()
            try:
                rp.wait_ready(proc, timeout=600)
                for q in maths:
                    text, fin, n, tps, wall = chat(q["question"] + MATH_SUFFIX, a.max_tokens)
                    got = math_answer(text)
                    ok = got is not None and abs(got - float(q["answer"])) < 1e-6
                    s["math_correct"] += ok
                    s["hit_max_tokens"] += fin == "length"
                    s["gen_tokens"] += n or 0
                    s["tps"].append(tps or 0)
                    fo.write(json.dumps({"model": tag, "id": q["id"], "ok": ok, "got": got, "want": q["answer"],
                                         "finish": fin, "tokens": n, "tps": tps, "text": text}) + "\n")
                    print(tag, q["id"], "ok" if ok else "WRONG", flush=True)
                for c in codes:
                    text, fin, n, tps, wall = chat(c["prompt"], a.max_tokens)
                    ok, err = run_code(text, c["tests"])
                    s["code_pass"] += ok
                    s["hit_max_tokens"] += fin == "length"
                    s["gen_tokens"] += n or 0
                    s["tps"].append(tps or 0)
                    fo.write(json.dumps({"model": tag, "id": c["id"], "ok": ok, "err": err, "finish": fin,
                                         "tokens": n, "tps": tps, "text": text}) + "\n")
                    print(tag, c["id"], "pass" if ok else "FAIL", flush=True)
            finally:
                proc.terminate()
                proc.wait(timeout=30)
                log.close()
            tps = sorted(x for x in s.pop("tps") if x)
            s["median_gen_tps"] = round(tps[len(tps) // 2], 2) if tps else None
            s["wall_s"] = round(time.time() - t_start, 1)
            summary[tag] = s
            print(tag, json.dumps(s), flush=True)
    (out / f"eval{sfx}_summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
