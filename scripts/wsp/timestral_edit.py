"""First Timestral-28M weight-edit experiment: rank-1 output-space edits W <- W + alpha u u^T W.

Arms: 'struct' (u = top-8 left singular vectors of W, or the layer's m4_hourly-vs-rest residual mean-difference
direction) and 'random' (9 seeded random unit directions per matrix, same search-space size).
Optuna TPE, N trials per arm, on the optimisation split. Then best trial per arm on validation with paired
per-series bootstrap CIs vs the unedited model; holdout once for any arm whose m4_hourly H64 WQL CI excludes 0.
Original weights: read-only, hash-checked; every trial is a delta spec (layer, matrix, dir, alpha) only.
Usage: timestral_edit.py <chronos_dir> <out_dir> [--trials 200]
"""
import argparse, glob, hashlib, json, os, time
import numpy as np, optuna, pyarrow.parquet as pq, torch
from timestral import TimestralForecaster

ap = argparse.ArgumentParser(); ap.add_argument("data"); ap.add_argument("out"); ap.add_argument("--trials", type=int, default=200)
A = ap.parse_args(); os.makedirs(A.out, exist_ok=True)
REPO, SEED, NSER, CTX, HS, BS, LAMBDA, NB = "mistralai/Timestral-28M", 0, 300, 2048, (64, 256), 32, 1.0, 2000
DATASETS = {"m4_hourly": "target", "electricity_15min": "consumption_kW", "exchange_rate": "target", "nn5": "target", "monash_traffic": "target"}
SEASON = {"m4_hourly": 24, "electricity_15min": 96, "exchange_rate": 1, "nn5": 7, "monash_traffic": 24}
TARGET = ("m4_hourly", 64); QL = np.round(np.arange(0.1, 0.91, 0.1), 2)
log = open(os.path.join(A.out, "log.txt"), "a")
def P(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); log.write(s + "\n"); log.flush()

# ---------- data and splits
splits, TASKS = {}, {}
for k, col in DATASETS.items():
    t = pq.read_table(glob.glob(f"{A.data}/{k}/*.parquet")[0], columns=["id", col])
    ids, ys = t.column("id").to_pylist(), t.column(col).to_pylist()
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(ys), min(NSER, len(ys)), replace=False)
    n = len(idx); a, b = int(round(0.4 * n)), int(round(0.8 * n))
    sp = {"opt": idx[:a], "val": idx[a:b], "hold": idx[b:]}
    splits[k] = {s: [str(ids[i]) for i in v] for s, v in sp.items()}
    m = SEASON[k]
    for s, v in sp.items():
        for H in HS:
            L = []
            for i in sorted(v.tolist()):
                y = np.asarray(ys[i], float)
                if len(y) < H + max(3 * m, 64) or np.isnan(y[-H:]).any(): continue
                ctx, tg = y[:-H][-CTX:], y[-H:]
                sc = np.nanmean(np.abs(ctx[m:] - ctx[:-m]))
                if np.isfinite(sc) and sc > 0: L.append((str(ids[i]), ctx.astype(np.float32), tg, sc))
            if L: TASKS[(s, k, H)] = L
json.dump(splits, open(os.path.join(A.out, "splits.json"), "w"))
P("tasks", {f"{s}/{k}/{H}": len(v) for (s, k, H), v in TASKS.items()})

# ---------- model, read-only originals
fc = TimestralForecaster.from_pretrained(REPO, device="cuda")
blob = glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--mistralai--Timestral-28M/snapshots/*/model.safetensors"))[0]
def sha(p): return hashlib.sha256(open(os.path.realpath(p), "rb").read()).hexdigest()
H0 = sha(blob); assert H0 == os.path.basename(os.readlink(blob)), "checkpoint hash mismatch"
layers = fc.model.transformer.layers
MATS = {}
for li, L in enumerate(layers):
    MATS[(li, "mlp")] = L.mlp.w3.weight
    MATS[(li, "attn")] = L.attns[-1].wO.weight  # last attention of the block (variate attn on time_variate layers)
ORIG = {k: w.detach().clone() for k, w in MATS.items()}

def batched(ts, H):
    out = []
    for i in range(0, len(ts), BS):
        ch = ts[i:i + BS]; T = max(len(c) for _, c, _, _ in ch)
        arr = np.full((len(ch), 1, T), np.nan, np.float32)
        for j, (_, c, _, _) in enumerate(ch): arr[j, 0, T - len(c):] = c
        f = fc.forecast(arr, prediction_length=H)
        q = f.quantiles.detach().float().cpu().numpy()[:, 0]; lv = np.asarray(f.quantile_levels)
        out.append(q)
    return np.concatenate(out), lv

def per_series(split):
    """dict (k,H) -> arrays (mase[n], ql[n], absy[n]) for the current weights."""
    R = {}
    for (s, k, H), ts in TASKS.items():
        if s != split: continue
        q, lv = batched(ts, H)
        li = [int(np.argmin(np.abs(lv - x))) for x in QL]; mi = int(np.argmin(np.abs(lv - 0.5)))
        tg = np.stack([t for _, _, t, _ in ts]); sc = np.array([s_ for _, _, _, s_ in ts])
        ql = np.mean([2 * np.sum(np.abs((tg - q[:, :, j]) * ((tg <= q[:, :, j]) - x)), 1) for x, j in zip(QL, li)], 0)
        R[(k, H)] = (np.mean(np.abs(tg - q[:, :, mi]), 1) / sc, ql, np.sum(np.abs(tg), 1))
    return R
def agg(R): return {kh: (float(np.mean(m)), float(q.sum() / a.sum())) for kh, (m, q, a) in R.items()}

# ---------- directions
@torch.no_grad()
def meandiff():
    acts = {li: {"m4": [], "rest": []} for li in range(len(layers))}
    cur = {}
    hs = [L.register_forward_hook(lambda m, i, o, li=li: acts[li][cur["g"]].append(o.detach().float().reshape(-1, o.shape[-1]).mean(0))) for li, L in enumerate(layers)]
    for (s, k, H), ts in TASKS.items():
        if s != "opt" or H != 64: continue
        cur["g"] = "m4" if k == "m4_hourly" else "rest"; batched(ts, H)
    for h in hs: h.remove()
    return {li: torch.nn.functional.normalize(torch.stack(v["m4"]).mean(0) - torch.stack(v["rest"]).mean(0), dim=0) for li, v in acts.items()}
MD = meandiff()
DIRS = {"struct": {}, "random": {}}
g = torch.Generator().manual_seed(1234)
for (li, mt), W in ORIG.items():
    U = torch.linalg.svd(W.float(), full_matrices=False).U  # [out, r]
    DIRS["struct"][(li, mt)] = [U[:, j] for j in range(8)] + [MD[li].to(W.device)]
    DIRS["random"][(li, mt)] = [torch.nn.functional.normalize(torch.randn(W.shape[0], generator=g), dim=0).to(W.device) for _ in range(9)]

@torch.no_grad()
def apply(spec, arm):
    for k, w in MATS.items(): w.copy_(ORIG[k])
    if spec is None: return
    W = MATS[(spec["layer"], spec["matrix"])]; u = DIRS[arm][(spec["layer"], spec["matrix"])][spec["dir"]].to(W.dtype)
    W.add_(spec["alpha"] * torch.outer(u, u @ W))

apply(None, "struct"); BASE = {s: per_series(s) for s in ("opt", "val")}; BASEA = agg(BASE["opt"])
P("base opt", json.dumps({f"{k}/{H}": v for (k, H), v in BASEA.items()}))

def objective_from(a):
    gain = (BASEA[TARGET][1] - a[TARGET][1]) / BASEA[TARGET][1]
    regs = [max(0.0, (a[kh][1] - BASEA[kh][1]) / BASEA[kh][1]) for kh in a if kh != TARGET]
    return gain - LAMBDA * float(np.mean(regs)), gain, float(np.mean(regs))

results = {}
for arm in ("struct", "random"):
    t0 = time.time(); trials = []
    def obj(tr):
        spec = {"layer": tr.suggest_int("layer", 0, len(layers) - 1), "matrix": tr.suggest_categorical("matrix", ["mlp", "attn"]),
                "dir": tr.suggest_int("dir", 0, 8), "alpha": tr.suggest_float("alpha", -1.0, 1.0)}
        apply(spec, arm); a = agg(per_series("opt")); o, gain, reg = objective_from(a)
        trials.append({**spec, "objective": o, "target_gain": gain, "mean_regression": reg}); return o
    st = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED),
                             storage=f"sqlite:///{A.out}/optuna.db", study_name=f"timestral28m-{arm}", load_if_exists=False)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    st.optimize(obj, n_trials=A.trials)
    json.dump(trials, open(os.path.join(A.out, f"trials-{arm}.json"), "w"), indent=0)
    best = {k: st.best_params[k] for k in ("layer", "matrix", "dir", "alpha")}
    P(arm, "best", best, "obj", round(st.best_value, 4), "secs", round(time.time() - t0))
    results[arm] = {"best_spec": best, "best_opt_objective": st.best_value, "opt_secs": round(time.time() - t0),
                    "best_opt_target_gain": max(trials, key=lambda t: t["objective"])["target_gain"]}

def boot(Rb, Re, split_name):
    rng = np.random.default_rng(SEED); out = {}
    for kh in Rb:
        mb, qb, ab = Rb[kh]; me, qe, _ = Re[kh]; n = len(mb)
        I = rng.integers(0, n, (NB, n))
        dm = (me[I].mean(1) - mb[I].mean(1)) / mb[I].mean(1)
        dq = (qe[I].sum(1) - qb[I].sum(1)) / qb[I].sum(1)
        pt_m = (me.mean() - mb.mean()) / mb.mean(); pt_q = (qe.sum() - qb.sum()) / qb.sum()
        out[f"{kh[0]}/H{kh[1]}"] = {"n": n, "dMASE_rel": float(pt_m), "dMASE_ci": [float(x) for x in np.percentile(dm, [2.5, 97.5])],
                                    "dWQL_rel": float(pt_q), "dWQL_ci": [float(x) for x in np.percentile(dq, [2.5, 97.5])],
                                    "regression_flag": bool(np.percentile(dq, 2.5) > 0 or np.percentile(dm, 2.5) > 0)}
    return out

for arm, r in results.items():
    apply(r["best_spec"], arm); r["val"] = boot(BASE["val"], per_series("val"), "val")
    t = r["val"][f"{TARGET[0]}/H{TARGET[1]}"]; r["val_target_ci_excludes_zero"] = t["dWQL_ci"][1] < 0
    P(arm, "val target", t)
    if r["val_target_ci_excludes_zero"]:
        apply(None, arm); Bh = per_series("hold"); apply(r["best_spec"], arm)
        r["holdout"] = boot(Bh, per_series("hold"), "hold"); P(arm, "holdout target", r["holdout"][f"{TARGET[0]}/H{TARGET[1]}"])
apply(None, "struct")
assert sha(blob) == H0, "checkpoint changed on disk!"
results.update({"checkpoint_sha256": H0, "checkpoint_unchanged": True, "trials_per_arm": A.trials, "lambda": LAMBDA,
                "base_opt": {f"{k}/H{H}": {"MASE": v[0], "WQL": v[1]} for (k, H), v in BASEA.items()},
                "n_series": {f"{s}/{k}/H{H}": len(v) for (s, k, H), v in TASKS.items()}})
json.dump(results, open(os.path.join(A.out, "results.json"), "w"), indent=1)
P("DONE")
