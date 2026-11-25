"""C0: can a small local model work as a Python coding agent?

The agent gets an empty project folder and a task, and acts one JSON action at a time
(write / read / run / test / done), each reply forced into that shape by llama-server's
JSON-schema grammar. After it says done or runs out of steps, hidden acceptance tests it
never saw decide the score. This is the loop planned for the iPhone (ixOS Device Lab), run
here on the GTX 1060 first to choose the model and the prompt.

  python3 experiments/agent/agent_bench.py --bin BIN --out DIR --models A.gguf B.gguf
  python3 experiments/agent/agent_bench.py --fake --out DIR      # harness self-check, no model

Code the agent writes runs in a subprocess in a temp dir (10 s timeout). Stdlib only.
"""
import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from tasks import TASKS  # noqa: E402

PORT = 8098
RT = ["-ngl", "99", "-fa", "on", "-b", "2048", "-ub", "256", "-np", "1"]
RESULT_CHARS = 1500

ACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "thought": {"type": "string", "maxLength": 400},
        "action": {"enum": ["write", "read", "run", "test", "done"]},
        "path": {"type": "string", "maxLength": 80},
        "content": {"type": "string"},
    },
    "required": ["thought", "action", "path", "content"],
}

SYSTEM = """You are a coding agent. You work in an empty project folder, and only the Python 3 standard library is available.
Build what the user asks as small Python files, and write unittest tests for it in files named test_*.py.

Every reply is exactly one JSON action:
{"thought": "<one or two short sentences>", "action": "<write|read|run|test|done>", "path": "<file name or empty>", "content": "<file text or empty>"}

- write: create or replace the file `path` with the complete text in `content`.
- read: show the file `path`.
- run: run `python3 path` and show its output (`path` is just the file name, e.g. "demo.py").
- test: run all the unittest tests (test_*.py) and show the result.
- done: you have finished. Say done once your tests pass.

Work in small steps: write the module, write the tests, run test, then fix what fails and test again.
When a test fails, first decide which side is wrong: re-read the task. Your own test can be the mistake
(a wrong expected value); then fix the test, not the code. Never repeat an action that just failed the same way."""


# --- the model ---------------------------------------------------------------

def post(body, timeout=900):
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def wait_ready(proc, timeout=600):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            raise RuntimeError(f"llama-server exited with {proc.returncode}")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2) as r:
                if r.status == 200:
                    return time.time() - t0
        except Exception:
            pass
        time.sleep(1)
    raise TimeoutError("llama-server did not become ready")


class ServerModel:
    def __init__(self, think, max_tokens):
        self.think, self.max_tokens = think, max_tokens

    def __call__(self, messages):
        body = {"messages": messages, "temperature": 0, "seed": 1, "max_tokens": self.max_tokens,
                "chat_template_kwargs": {"enable_thinking": self.think},
                "response_format": {"type": "json_schema", "json_schema": {"name": "action", "schema": ACTION_SCHEMA}}}
        r = post(body)
        ch = r["choices"][0]
        tm = r.get("timings", {})
        return (ch["message"].get("content") or "", ch.get("finish_reason"),
                {"prompt_n": tm.get("prompt_n"), "predicted_n": tm.get("predicted_n"),
                 "predicted_per_second": tm.get("predicted_per_second")})


class FakeModel:
    """Writes the task's reference solution, runs the tests, says done: checks the harness."""

    def __init__(self, task, tests_first=False):
        files = list(task["reference"].items())
        if tests_first:
            files.insert(0, ("test_own.py", task["hidden"]))
        else:
            files.append(("test_own.py", task["hidden"]))
        self.script = [{"thought": "write", "action": "write", "path": p, "content": c} for p, c in files]
        self.script += [{"thought": "test", "action": "test", "path": "", "content": ""},
                        {"thought": "finished", "action": "done", "path": "", "content": ""}]

    def __call__(self, messages):
        return json.dumps(self.script.pop(0)), "stop", {}


# --- the tools ---------------------------------------------------------------

def safe_path(ws, name):
    p = (ws / name).resolve()
    if not name or ws.resolve() not in p.parents:
        raise ValueError(f"bad path {name!r}: use a plain file name inside the project")
    return p


def run_cmd(ws, args, timeout=10):
    try:
        p = subprocess.run([sys.executable, "-E", "-s", *args], cwd=ws, capture_output=True, text=True,
                           timeout=timeout)
        out = (p.stdout + p.stderr).strip()
        return p.returncode, out
    except subprocess.TimeoutExpired:
        return -1, f"timed out after {timeout} s"


def tail(text, n=RESULT_CHARS):
    return text if len(text) <= n else "...\n" + text[-n:]


def files_line(ws):
    names = sorted(p.name for p in ws.iterdir() if p.is_file() and not p.name.startswith("_hidden"))
    return "Files: " + (", ".join(f"{n} ({len((ws / n).read_text().splitlines())} lines)" for n in names) or "none")


def act(ws, a):
    kind, path = a["action"], a.get("path", "")
    if kind == "write":
        p = safe_path(ws, path)
        p.write_text(a.get("content", ""))
        return f"Wrote {path}."
    if kind == "read":
        p = safe_path(ws, path)
        return tail(p.read_text(), 3000) if p.is_file() else f"No file {path}."
    if kind == "run":
        words = path.split()
        if len(words) > 1:  # "python3 test_x.py": the model wrote a command, not a file name
            path = next((w for w in words if w.endswith(".py")), path)
        p = safe_path(ws, path)
        if not p.is_file():
            return f"No file {path}."
        code, out = run_cmd(ws, [p.name])
        return f"Exit code {code}.\n{tail(out)}"
    if kind == "test":
        code, out = run_cmd(ws, ["-m", "unittest", "discover", "-p", "test*.py"], timeout=30)
        return f"Tests {'passed' if code == 0 else 'FAILED'}.\n{tail(out)}"
    return "Done."


def hidden_score(ws, task):
    (ws / "_hidden_test.py").write_text(task["hidden"])
    code, out = run_cmd(ws, ["-m", "unittest", "-v", "_hidden_test"], timeout=60)
    ran = re.search(r"Ran (\d+) test", out)
    n = int(ran.group(1)) if ran else 0
    bad = re.search(r"FAILED \((.*?)\)", out)
    failed = sum(int(x) for x in re.findall(r"=(\d+)", bad.group(1))) if bad else 0
    return {"hidden_pass": code == 0 and n > 0, "hidden_tests": n, "hidden_failed": failed if n else None,
            "hidden_tail": tail(out, 600)}


def compact(history):
    """Older full copies of a file the agent rewrote are replaced by a note, to save context."""
    last_write = {}
    for i, m in enumerate(history):
        if m["role"] == "assistant" and m.get("_action", {}).get("action") == "write":
            last_write[m["_action"]["path"]] = i
    out = []
    for i, m in enumerate(history):
        a = m.get("_action")
        if a and a.get("action") == "write" and last_write.get(a["path"]) != i:
            short = dict(a, content=f"<{len(a['content'].splitlines())} lines, since replaced>")
            out.append({"role": "assistant", "content": json.dumps(short)})
        else:
            out.append({"role": m["role"], "content": m["content"]})
    return out


TEST_SYSTEM = """You write unittest tests for a Python task, before anyone writes the code. Only the Python 3
standard library is available. Write ONE file named test_<module>.py that imports the module the task names.
Check only what the task states, with small examples whose expected values you have worked out by hand.
Do not write the module itself.

Reply with exactly one JSON action:
{"thought": "<one or two short sentences>", "action": "write", "path": "test_<module>.py", "content": "<the test file>"}"""

CODE_SYSTEM = """You are a coding agent. You work in a project folder, and only the Python 3 standard library is available.
The tests for the task are already written and locked: you cannot change them. Write the code that makes them pass.

Every reply is exactly one JSON action:
{"thought": "<one or two short sentences>", "action": "<write|read|run|test|done>", "path": "<file name or empty>", "content": "<file text or empty>"}

- write: create or replace the file `path` with the complete text in `content` (not a test_*.py file).
- read: show the file `path`.
- run: run `python3 path` and show its output (`path` is just the file name, e.g. "demo.py").
- test: run all the tests and show the result.
- done: you have finished. Say done once the tests pass.

Work in small steps: write the module, run test, read the failures, fix the code and test again.
If you are sure after a few tries that a test contradicts the task, say done and explain why in thought.
Never repeat an action that just failed the same way."""


def agent_loop(model, ws, history, rec, steps, label, locked_tests=False, stop_after_test_write=False):
    """Run up to `steps` actions. Returns how it ended: done, tests_written, steps or a server error."""
    for _ in range(steps):
        rec["steps"] += 1
        try:
            text, finish, tm = model(compact(history))
        except urllib.error.HTTPError as e:
            return f"server error {e.code}: {e.read()[:200]!r}"
        rec["gen_tokens"] += tm.get("predicted_n") or 0
        rec["max_prompt"] = max(rec["max_prompt"], tm.get("prompt_n") or 0)
        try:
            a = json.loads(text)
            assert a.get("action") in ("write", "read", "run", "test", "done")
        except Exception:
            rec["invalid"] += 1
            history += [{"role": "assistant", "content": text[:2000]},
                        {"role": "user", "content": f"That was not one valid JSON action (finish: {finish}). Reply with one JSON action."}]
            continue
        rec["actions"].append(a["action"] + (f" {a['path']}" if a.get("path") else ""))
        history.append({"role": "assistant", "content": json.dumps(a), "_action": a})
        if a["action"] == "done" and not stop_after_test_write:
            return "done"
        is_test = Path(a.get("path", "")).name.startswith("test")
        if stop_after_test_write and not (a["action"] == "write" and is_test):
            result = "Write the test file now: action write, path test_<module>.py."
        elif locked_tests and a["action"] == "write" and is_test:
            result = "Test files are locked. Change the code, not the tests."
        else:
            try:
                result = act(ws, a)
            except Exception as e:
                result = f"Error: {e}"
            if stop_after_test_write:
                code, out = run_cmd(ws, ["-m", "py_compile", a["path"]])
                if code == 0:
                    print(f"  {label}: wrote {a['path']}", flush=True)
                    return "tests_written"
                result = f"That test file has a syntax error, write it again:\n{tail(out, 600)}"
        prev = [m.get("_action") for m in history[:-1] if m.get("_action")]
        if prev and prev[-1] == a and a["action"] != "write":
            result += "\nYou just did exactly this. Do something different."
        history.append({"role": "user", "content": f"{result}\n{files_line(ws)}"})
        print(f"  {label} step {rec['steps']}: {rec['actions'][-1]} -> {result.splitlines()[0][:80]}", flush=True)
    return "steps"


def run_task(model, task, max_steps, log, tests_first=False):
    ws = Path(tempfile.mkdtemp(prefix=f"agent-{task['id']}-"))
    rec = {"task": task["id"], "level": task["level"], "steps": 0, "invalid": 0, "finished": "steps",
           "gen_tokens": 0, "max_prompt": 0, "actions": [], "mode": "tests_first" if tests_first else "single"}
    transcripts = []
    t0 = time.time()
    try:
        if tests_first:
            # Step 1: tests from the task alone, in their own conversation (at most 3 tries).
            h1 = [{"role": "system", "content": TEST_SYSTEM}, {"role": "user", "content": task["spec"]}]
            end1 = agent_loop(model, ws, h1, rec, 3, f"{task['id']} tests", stop_after_test_write=True)
            transcripts.append(h1)
            rec["test_steps"] = rec["steps"]
            tests = {p.name: p.read_text() for p in sorted(ws.glob("test*.py"))}
            if end1 != "tests_written" or not tests:
                rec["finished"] = "no tests" if end1 == "steps" else end1
            else:
                # Step 2: a fresh conversation that sees the task and the locked tests, and writes the code.
                shown = "\n\n".join(f"{n} (locked):\n```python\n{c}\n```" for n, c in tests.items())
                h2 = [{"role": "system", "content": CODE_SYSTEM},
                      {"role": "user", "content": f"{task['spec']}\n\n{shown}"}]
                rec["finished"] = agent_loop(model, ws, h2, rec, max_steps - rec["steps"], task["id"], locked_tests=True)
                transcripts.append(h2)
                if any((ws / n).read_text() != c for n, c in tests.items() if (ws / n).exists()):
                    rec["tests_changed"] = True  # should never happen: writes to tests are refused
        else:
            h = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": task["spec"]}]
            rec["finished"] = agent_loop(model, ws, h, rec, max_steps, task["id"])
            transcripts.append(h)
        rec["wall_s"] = round(time.time() - t0, 1)
        code, out = run_cmd(ws, ["-m", "unittest", "discover", "-p", "test*.py"], timeout=30)
        rec["own_tests_pass"] = code == 0 and "Ran 0 tests" not in out
        rec.update(hidden_score(ws, task))
        rec["files"] = {p.name: p.read_text() for p in ws.iterdir()
                        if p.is_file() and p.suffix == ".py" and not p.name.startswith("_hidden")}
        log.write(json.dumps({"task": task["id"], "transcript": [[{"role": m["role"], "content": m["content"]}
                                                                    for m in h] for h in transcripts]}) + "\n")
    finally:
        shutil.rmtree(ws, ignore_errors=True)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", help="llama.cpp build/bin with llama-server")
    ap.add_argument("--out", required=True)
    ap.add_argument("--models", nargs="*", default=[])
    ap.add_argument("--tasks", nargs="*", help="task ids (default: all)")
    ap.add_argument("--max-steps", type=int, default=14)
    ap.add_argument("--max-tokens", type=int, default=3000, help="per reply")
    ap.add_argument("--ctx", type=int, default=16384)
    ap.add_argument("--think", action="store_true", help="thinking on (default off)")
    ap.add_argument("--server-args", default=" ".join(RT))
    ap.add_argument("--fake", action="store_true", help="scripted reference solutions, no model")
    ap.add_argument("--tests-first", action="store_true",
                    help="write the tests from the task first, then the code against locked tests")
    a = ap.parse_args()
    tasks = [t for t in TASKS if not a.tasks or t["id"] in a.tasks]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    runs = ["fake"] if a.fake else a.models
    summary = {}
    with open(out / "runs.jsonl", "a") as fo, open(out / "transcripts.jsonl", "a") as log:
        for model_path in runs:
            tag = ("fake" if a.fake else Path(model_path).stem + ("-think" if a.think else "")) + \
                ("-testsfirst" if a.tests_first else "")
            proc = None
            if not a.fake:
                slog = open(out / f"server_{tag}.log", "w")
                proc = subprocess.Popen([str(Path(a.bin) / "llama-server"), "-m", model_path, "-c", str(a.ctx), "--jinja",
                                         "--host", "127.0.0.1", "--port", str(PORT), *a.server_args.split()],
                                        stdout=slog, stderr=subprocess.STDOUT)
            t0 = time.time()
            recs = []
            try:
                load_s = wait_ready(proc) if proc else 0
                for task in tasks:
                    model = FakeModel(task, a.tests_first) if a.fake else ServerModel(a.think, a.max_tokens)
                    print(f"[{tag}] {task['id']}", flush=True)
                    rec = run_task(model, task, a.max_steps, log, a.tests_first)
                    rec["model"] = tag
                    recs.append(rec)
                    fo.write(json.dumps(rec) + "\n")
                    fo.flush()
                    print(f"[{tag}] {task['id']}: hidden {'PASS' if rec['hidden_pass'] else 'FAIL'} "
                          f"({rec['hidden_tests']} tests, {rec['hidden_failed']} failed), {rec['steps']} steps, "
                          f"{rec['finished']}, {rec['wall_s']} s", flush=True)
            finally:
                if proc:
                    proc.terminate()
                    try:
                        proc.wait(timeout=120)  # the GPU driver can hold it for a while
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait(timeout=120)
            summary[tag] = {
                "hidden_pass": sum(r["hidden_pass"] for r in recs), "tasks": len(recs),
                "by_level": {lv: f"{sum(r['hidden_pass'] for r in recs if r['level'] == lv)}/{sum(1 for r in recs if r['level'] == lv)}"
                             for lv in sorted({r["level"] for r in recs})},
                "said_done": sum(r["finished"] == "done" for r in recs),
                "no_tests": sum(r["finished"] == "no tests" for r in recs),
                "own_tests_pass": sum(r["own_tests_pass"] for r in recs),
                "invalid_actions": sum(r["invalid"] for r in recs),
                "mean_steps": round(sum(r["steps"] for r in recs) / max(len(recs), 1), 1),
                "gen_tokens": sum(r["gen_tokens"] for r in recs),
                "max_prompt_tokens": max((r["max_prompt"] for r in recs), default=0),
                "load_s": round(load_s, 1), "wall_s": round(time.time() - t0, 1),
            }
            print(tag, json.dumps(summary[tag]), flush=True)
            (out / "summary.json").write_text(json.dumps(summary, indent=2))  # after each model


if __name__ == "__main__":
    main()
