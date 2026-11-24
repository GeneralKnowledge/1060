"""Rescore committed maths results with the current eval_tasks.math_answer (strict 'Answer:' line, no fallback).

Raw eval*.jsonl files are kept as recorded at run time. This writes <name>_rescored.json next to each, with per-model
strict maths counts and the ids whose verdict changed.

  python3 experiments/lowbit/rescore_math.py experiments/lowbit/e2/*/eval.jsonl ...
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from eval_tasks import math_answer  # noqa: E402

for f in map(Path, sys.argv[1:]):
    out = {}
    for line in f.read_text().splitlines():
        r = json.loads(line)
        if "want" not in r:
            continue
        got = math_answer(r["text"])
        ok = got is not None and abs(got - float(r["want"])) < 1e-6
        m = out.setdefault(r["model"], {"math_correct_strict": 0, "math_correct_recorded": 0, "math_n": 0, "changed": []})
        m["math_n"] += 1
        m["math_correct_strict"] += ok
        m["math_correct_recorded"] += bool(r["ok"])
        if ok != bool(r["ok"]):
            m["changed"].append({"id": r["id"], "recorded": r["ok"], "strict": ok, "finish": r.get("finish")})
    dest = f.with_name(f.stem + "_rescored.json")
    dest.write_text(json.dumps(out, indent=2) + "\n")
    print(dest, {k: (v["math_correct_recorded"], v["math_correct_strict"]) for k, v in out.items()})
