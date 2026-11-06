"""Timestral-28M / 209M feasibility benchmark on the GTX 1060.

Measures load time, VRAM, per-forecast latency (context 2048, horizon 64/256) and a small fixed zero-shot eval
(last-H holdout, context = up to 2048 preceding steps) with MASE and WQL vs seasonal-naive.
Usage: timestral_bench.py <chronos_data_dir> <out_json>
"""
import glob, json, os, sys, time
import numpy as np, pyarrow.parquet as pq, torch
from timestral import TimestralForecaster

DATA, OUT = sys.argv[1], sys.argv[2]
SEED, NSER, CTX, HS, BS = 0, 100, 2048, (64, 256), 16
DATASETS = {"m4_hourly": ("target", 24), "electricity_15min": ("consumption_kW", 96), "exchange_rate": ("target", 1),
            "nn5": ("target", 7), "monash_traffic": ("target", 24)}
QL = np.round(np.arange(0.1, 0.91, 0.1), 2)

def load_series(name):
    col, m = DATASETS[name]
    t = pq.read_table(glob.glob(f"{DATA}/{name}/*.parquet")[0], columns=["id", col])
    ids, ys = t.column("id").to_pylist(), t.column(col).to_pylist()
    rng = np.random.default_rng(SEED)
    idx = sorted(rng.choice(len(ys), min(NSER, len(ys)), replace=False).tolist())
    return [(ids[i], np.asarray(ys[i], dtype=np.float64)) for i in idx], m

def tasks(series, m, H):
    out = []
    for sid, y in series:
        if len(y) < H + max(3 * m, 64) or np.isnan(y[-H:]).any(): continue
        ctx, tgt = y[:-H][-CTX:], y[-H:]
        d = np.abs(ctx[m:] - ctx[:-m]); scale = np.nanmean(d)
        if not np.isfinite(scale) or scale == 0: continue
        out.append((sid, ctx, tgt, scale))
    return out

def snaive(ctx, m, H):
    last = ctx[-m:]
    return np.array([last[i % m] for i in range(H)])

def metrics(pred_q, levels, tgt, scale):  # pred_q (H, nq)
    li = [int(np.argmin(np.abs(np.asarray(levels) - q))) for q in QL]
    med = pred_q[:, int(np.argmin(np.abs(np.asarray(levels) - 0.5)))]
    ql = sum(2 * np.sum(np.abs((tgt - pred_q[:, j]) * ((tgt <= pred_q[:, j]) - q))) for q, j in zip(QL, li)) / len(QL)
    return np.mean(np.abs(tgt - med)) / scale, ql, np.sum(np.abs(tgt))

def forecast_batch(fc, ctxs, H):
    T = max(len(c) for c in ctxs)
    arr = np.full((len(ctxs), 1, T), np.nan, dtype=np.float32)
    for i, c in enumerate(ctxs): arr[i, 0, T - len(c):] = c
    f = fc.forecast(arr, prediction_length=H)
    q = f.quantiles.detach().cpu().numpy() if torch.is_tensor(f.quantiles) else np.asarray(f.quantiles)
    return q[:, 0], f.quantile_levels  # (B, H, nq)

res = {"seed": SEED, "max_series_per_dataset": NSER, "context": CTX, "torch": torch.__version__,
       "gpu": torch.cuda.get_device_name(0), "datasets": {k: {"hf": f"autogluon/chronos_datasets/{k}", "season": v[1]} for k, v in DATASETS.items()},
       "models": {}, "eval": []}
data = {k: load_series(k) for k in DATASETS}
TASKS = {(k, H): tasks(data[k][0], data[k][1], H) for k in DATASETS for H in HS}
for (k, H), ts in TASKS.items():  # seasonal-naive baseline
    mm, qq, ss = zip(*[metrics(np.repeat(snaive(c, DATASETS[k][1], H)[:, None], len(QL), 1), list(QL), t, s) for _, c, t, s in ts]) if ts else ((), (), ())
    res["eval"].append({"model": "seasonal_naive", "dataset": k, "H": H, "n": len(ts), "MASE": float(np.mean(mm)) if ts else None,
                        "WQL": float(sum(qq) / sum(ss)) if ts else None})

for repo in ("mistralai/Timestral-28M", "mistralai/Timestral-209M"):
    name = repo.split("/")[1]; r = {}
    TimestralForecaster.from_pretrained(repo, device="cpu")  # ensure downloaded, so load time excludes network
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    t = time.time(); fc = TimestralForecaster.from_pretrained(repo, device="cuda"); torch.cuda.synchronize()
    r["load_s"] = round(time.time() - t, 2); r["vram_weights_mib"] = round(torch.cuda.memory_allocated() / 2**20)
    rng = np.random.default_rng(1)
    for B in (1, 16):
        x = rng.standard_normal((B, 1, CTX)).cumsum(-1).astype(np.float32)
        for H in HS:
            fc.forecast(x, prediction_length=H); torch.cuda.synchronize()
            ts = []
            for _ in range(10):
                t = time.time(); fc.forecast(x, prediction_length=H); torch.cuda.synchronize(); ts.append(time.time() - t)
            r[f"latency_ms_B{B}_ctx{CTX}_H{H}"] = round(1000 * float(np.median(ts)), 1)
    torch.cuda.reset_peak_memory_stats(); t0 = time.time()
    for (k, H), ts in TASKS.items():
        t1 = time.time(); mm, qq, ss = [], [], []
        for i in range(0, len(ts), BS):
            chunk = ts[i:i + BS]
            q, lv = forecast_batch(fc, [c for _, c, _, _ in chunk], H)
            for (_, c, tg, s), pq_ in zip(chunk, q):
                a, b, cc = metrics(pq_, lv, tg, s); mm.append(a); qq.append(b); ss.append(cc)
        res["eval"].append({"model": name, "dataset": k, "H": H, "n": len(ts), "MASE": float(np.mean(mm)) if ts else None,
                            "WQL": float(sum(qq) / sum(ss)) if ts else None, "secs": round(time.time() - t1, 2)})
    torch.cuda.synchronize(); r["full_eval_s"] = round(time.time() - t0, 1)
    r["vram_peak_eval_mib"] = round(torch.cuda.max_memory_allocated() / 2**20)
    r["params_m"] = round(sum(p.numel() for p in fc.model.parameters()) / 1e6, 1) if hasattr(fc, "model") else None
    res["models"][name] = r; print(name, json.dumps(r), flush=True)
    del fc; torch.cuda.empty_cache()

json.dump(res, open(OUT, "w"), indent=1)
# summary table + weak spots for 28M
E = {(e["model"], e["dataset"], e["H"]): e for e in res["eval"]}
lines = ["| dataset (season) | H | n | MASE snaive / 28M / 209M | WQL snaive / 28M / 209M |", "|---|---|---|---|---|"]
weak = []
f = lambda v: "-" if v is None else f"{v:.3f}"
for k in DATASETS:
    for H in HS:
        s, a, b = (E[(mn, k, H)] for mn in ("seasonal_naive", "Timestral-28M", "Timestral-209M"))
        lines.append(f"| {k} ({DATASETS[k][1]}) | {H} | {s['n']} | {f(s['MASE'])} / {f(a['MASE'])} / {f(b['MASE'])} | {f(s['WQL'])} / {f(a['WQL'])} / {f(b['WQL'])} |")
        if a["MASE"] is not None:
            if a["MASE"] > 1.10 * b["MASE"]: weak.append(f"{k} H={H}: 28M MASE {a['MASE']:.3f} is {100*(a['MASE']/b['MASE']-1):.0f}% worse than 209M")
            if a["MASE"] > s["MASE"]: weak.append(f"{k} H={H}: 28M MASE {a['MASE']:.3f} worse than seasonal-naive {s['MASE']:.3f}")
open(OUT.replace(".json", ".md"), "w").write("\n".join(lines) + "\n\nWeak spots for 28M (>10% worse MASE than 209M, or worse than seasonal-naive):\n\n"
                                            + ("\n".join(f"- {w}" for w in weak) if weak else "- none") + "\n")
print("\n".join(lines)); print(weak)
