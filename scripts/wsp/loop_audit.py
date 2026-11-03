"""GSM8K loop audit against a running llama-server (OpenAI chat API).

Counts: accuracy, generations hitting max_tokens, generations with repeated n-gram loops.
Usage: loop_audit.py <parquet> <port> <out_jsonl> [--n 32] [--max-tokens 8192] [--workers 4]
"""
import argparse, json, re, time, urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import pyarrow.parquet as pq

ap = argparse.ArgumentParser()
ap.add_argument("parquet"); ap.add_argument("port", type=int); ap.add_argument("out")
ap.add_argument("--n", type=int, default=32); ap.add_argument("--max-tokens", type=int, default=8192)
ap.add_argument("--workers", type=int, default=4); ap.add_argument("--seed", type=int, default=1234)
a = ap.parse_args()

rows = pq.read_table(a.parquet).to_pylist()[: a.n]
PROMPT = "{q}\n\nPlease reason step by step, and put your final answer within \\boxed{{}}."

def num(s):
    s = s.replace(",", "").replace("$", "").strip()
    m = re.findall(r"-?\d+(?:\.\d+)?", s)
    return float(m[-1]) if m else None

def extract(text):
    b = re.findall(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", text)
    if b: return num(b[-1])
    return num(text[-300:]) if text else None

def loop_score(text, n=12, window=3000):
    """max repeat count of any word n-gram within the last `window` words."""
    w = text.split()[-window:]
    if len(w) < n: return 0
    c = Counter(tuple(w[i:i + n]) for i in range(len(w) - n + 1))
    return max(c.values())

def run(i):
    r = rows[i]
    body = {"messages": [{"role": "user", "content": PROMPT.format(q=r["question"])}],
            "max_tokens": a.max_tokens, "temperature": 0.6, "top_p": 0.95, "top_k": 20,
            "seed": a.seed + i, "chat_template_kwargs": {"enable_thinking": True}}
    req = urllib.request.Request(f"http://127.0.0.1:{a.port}/v1/chat/completions",
                                 json.dumps(body).encode(), {"Content-Type": "application/json"})
    t = time.time()
    resp = json.load(urllib.request.urlopen(req, timeout=7200))
    ch = resp["choices"][0]; msg = ch["message"]
    reasoning = msg.get("reasoning_content") or ""; content = msg.get("content") or ""
    full = reasoning + "\n" + content
    gold = num(r["answer"].split("####")[-1])
    pred = extract(content) if content.strip() else None
    rec = {"i": i, "finish": ch["finish_reason"], "completion_tokens": resp["usage"]["completion_tokens"],
           "secs": round(time.time() - t, 1), "gold": gold, "pred": pred,
           "correct": pred is not None and gold is not None and abs(pred - gold) < 1e-6,
           "loop_max_repeat_12gram": loop_score(full), "has_answer_section": bool(content.strip()),
           "tail": full[-400:]}
    print(json.dumps({k: rec[k] for k in ("i", "finish", "completion_tokens", "secs", "pred", "gold", "correct", "loop_max_repeat_12gram")}), flush=True)
    return rec

with ThreadPoolExecutor(a.workers) as ex:
    recs = list(ex.map(run, range(len(rows))))
with open(a.out, "w") as f:
    for r in recs: f.write(json.dumps(r) + "\n")
cap = sum(r["finish"] == "length" for r in recs)
loops = sum(r["loop_max_repeat_12gram"] >= 5 for r in recs)
print(json.dumps({"n": len(recs), "accuracy": sum(r["correct"] for r in recs) / len(recs), "hit_cap": cap,
                  "loop_ge5": loops, "loop_or_cap": sum((r["finish"] == "length") or r["loop_max_repeat_12gram"] >= 5 for r in recs),
                  "mean_tokens": sum(r["completion_tokens"] for r in recs) / len(recs)}))
