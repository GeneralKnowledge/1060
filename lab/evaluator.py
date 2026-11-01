"""Multi-objective evaluation with noise accounting.

Inputs are per-metric trial lists for the baseline and the candidate,
measured in the same session with interleaved rounds. Output is a decision:

  rejected      a mandatory gate failed (errors, safety abort, quality, decode-path
                quality, or an op test that newly fails against the CPU backend)
  promote       balanced score improved by more than the noise threshold
  discard       balanced score got worse by more than the threshold
  inconclusive  the difference is within measurement noise
"""
from __future__ import annotations

import math
import statistics

# metric -> (direction, weight key)
METRICS = {
    "tg128_ts": (+1, "tg"),
    "pp512_ts": (+1, "pp"),
    "ttft_s": (-1, "ttft"),
    "vram_peak_mib": (-1, "vram"),
    "host_rss_peak_mb": (-1, "host_ram"),
    "tokens_per_joule": (+1, "efficiency"),
}

PROFILES = {
    "best_generation": ["tg128_ts"],
    "best_prompt_processing": ["pp512_ts", "pp2048_ts"],
    "best_interactive": ["ttft_s", "tg128_ts"],
    "best_efficiency": ["tokens_per_joule"],
    "best_model_capacity": ["vram_peak_mib"],
}
DIRECTION = {**{k: v[0] for k, v in METRICS.items()}, "pp2048_ts": +1}


def _cv(xs):
    if len(xs) < 2:
        return 0.0
    m = statistics.mean(xs)
    return statistics.stdev(xs) / abs(m) if m else 0.0


def compare_metric(base, cand, direction):
    """Ratio > 1 always means 'candidate is better'."""
    if not base or not cand:
        return None
    mb, mc = statistics.median(base), statistics.median(cand)
    if mb <= 0 or mc <= 0:
        return None
    ratio = mc / mb if direction > 0 else mb / mc
    return {"baseline": mb, "candidate": mc, "ratio": round(ratio, 4),
            "noise": round(math.sqrt(_cv(base) ** 2 + _cv(cand) ** 2), 4),
            "n": [len(base), len(cand)]}


def quality_gate(base_q: dict, cand_q: dict, qcfg: dict) -> list[str]:
    fails = []
    if not cand_q or cand_q.get("returncode") not in (0, None) or cand_q.get("aborted"):
        return ["quality run failed"]
    if "mean_kld" in cand_q and cand_q["mean_kld"] > qcfg["max_mean_kld"]:
        fails.append(f"mean KLD {cand_q['mean_kld']} > {qcfg['max_mean_kld']}")
    if base_q and base_q.get("ppl") and cand_q.get("ppl"):
        if cand_q["ppl"] / base_q["ppl"] > qcfg["max_ppl_ratio"]:
            fails.append(f"perplexity {cand_q['ppl']} vs baseline {base_q['ppl']}")
    if "ppl" not in cand_q and "mean_kld" not in cand_q:
        fails.append("no quality measurement parsed")
    return fails


def decode_gate(base_d: dict | None, cand_d: dict | None, dcfg: dict) -> list[str]:
    """Decode-path quality (ubatch 1) relative to the baseline's own decode run."""
    if cand_d is None:
        return []
    if cand_d.get("returncode") not in (0, None) or cand_d.get("aborted"):
        return ["decode quality run failed"]
    if "ppl" not in cand_d:
        return ["decode quality: no perplexity parsed"]
    if not base_d or "ppl" not in base_d:
        return ["decode quality: baseline decode run has no perplexity to compare with"]
    limit = dcfg.get("max_ppl_ratio", 1.01)
    if cand_d["ppl"] / base_d["ppl"] > limit:
        return [f"decode perplexity {cand_d['ppl']} vs baseline {base_d['ppl']} (limit x{limit})"]
    return []


def kernel_gate(base_k: dict | None, cand_k: dict | None) -> list[str]:
    """No op may fail test-backend-ops on the candidate that passes on the baseline."""
    if cand_k is None:
        return []
    if not cand_k.get("complete"):
        return ["kernel check did not complete on the candidate"]
    if not base_k or not base_k.get("complete"):
        return ["kernel check did not complete on the baseline; nothing to compare with"]
    new = sorted(set(cand_k.get("failed", [])) - set(base_k.get("failed", [])))
    if new:
        return [f"kernel check: {len(new)} op test(s) fail that pass on the baseline, e.g. "
                + "; ".join(new[:3])]
    return []


def evaluate(base: dict, cand: dict, *, base_quality: dict | None, cand_quality: dict | None,
             errors: list[str], safety_aborts: list[str], cfg: dict,
             base_decode: dict | None = None, cand_decode: dict | None = None,
             base_kernels: dict | None = None, cand_kernels: dict | None = None) -> dict:
    ev = cfg["evaluation"]
    weights = ev["weights"]
    per_metric = {}
    for name, direction in DIRECTION.items():
        c = compare_metric(base.get(name), cand.get(name), direction)
        if c:
            c["significant"] = abs(c["ratio"] - 1) > max(ev["min_effect"], ev["noise_k"] * c["noise"])
            per_metric[name] = c

    gates = []
    gates += [f"error: {e}" for e in errors]
    gates += [f"safety: {s}" for s in safety_aborts]
    gates += quality_gate(base_quality or {}, cand_quality or {}, cfg["quality"])
    gates += decode_gate(base_decode, cand_decode, cfg["quality"].get("decode") or {})
    gates += kernel_gate(base_kernels, cand_kernels)
    if "tg128_ts" not in per_metric and "pp512_ts" not in per_metric:
        gates.append("no throughput measurements")

    num = den = noise_sq = 0.0
    for name, (direction, wkey) in METRICS.items():
        if name in per_metric and weights.get(wkey):
            w = weights[wkey]
            num += w * math.log(per_metric[name]["ratio"])
            noise_sq += (w * per_metric[name]["noise"]) ** 2
            den += w
    score = math.exp(num / den) if den else None
    score_noise = math.sqrt(noise_sq) / den if den else None
    threshold = max(ev["min_effect"], ev["noise_k"] * (score_noise or 0))

    if gates:
        decision = "rejected"
    elif score is None:
        decision = "inconclusive"
    elif score - 1 > threshold:
        decision = "promote"
    elif 1 - score > threshold:
        decision = "discard"
    else:
        decision = "inconclusive"

    tradeoffs = [f"{n} worse by {round((1 - m['ratio']) * 100, 1)}%"
                 for n, m in per_metric.items() if m["significant"] and m["ratio"] < 1]
    profile_wins = [p for p, ms in PROFILES.items()
                    if not gates and ms and all(m in per_metric and per_metric[m]["significant"]
                                                and per_metric[m]["ratio"] > 1 for m in ms)]
    return {"decision": decision, "balanced_score": round(score, 4) if score else None,
            "threshold": round(threshold, 4), "metrics": per_metric, "gate_failures": gates,
            "tradeoffs": tradeoffs, "profile_wins": profile_wins}
