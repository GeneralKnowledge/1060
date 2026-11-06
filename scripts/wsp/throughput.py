"""Aggregate generation throughput against a running llama-server with N parallel slots."""
import json, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
port, n, gen = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
PROMPTS = ["Write a detailed essay about the history of bridges.", "Explain how a refrigerator works, step by step.",
           "Describe a walk through a busy market in a coastal town.", "Summarise the main ideas of thermodynamics."]
def one(i):
    body = {"prompt": PROMPTS[i % 4], "n_predict": gen, "temperature": 0.0, "ignore_eos": True, "cache_prompt": False}
    r = urllib.request.Request(f"http://127.0.0.1:{port}/completion", json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(r, timeout=3600))
one(0)  # warm-up
t = time.time()
with ThreadPoolExecutor(n) as ex: res = list(ex.map(one, range(n)))
wall = time.time() - t
toks = sum(r["timings"]["predicted_n"] for r in res)
print(json.dumps({"np": n, "gen_tokens_each": gen, "wall_s": round(wall, 2), "aggregate_tok_s": round(toks / wall, 2),
                  "per_slot_tok_s": [round(r["timings"]["predicted_per_second"], 2) for r in res]}))
