"""Experiment records: the permanent, reproducible log of every attempt."""
from __future__ import annotations

import copy
import datetime as dt
import hashlib
import json
import re
import secrets
from dataclasses import asdict, dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RECORDS = ROOT / "experiments" / "records"
RESULTS_TSV = ROOT / "experiments" / "results.tsv"
TSV_COLUMNS = ["id", "finished_utc", "workload", "kind", "decision", "balanced_score",
               "tg128_ratio", "pp512_ratio", "ttft_ratio", "vram_ratio", "cand_tg128_ts",
               "cand_pp512_ts", "mean_kld", "hypothesis"]


def new_id() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d-%H%M%S-") + secrets.token_hex(2)


@dataclass
class ExperimentSpec:
    """What the researcher (human or agent) proposes."""
    hypothesis: str
    objective: str
    description: str
    changes: dict = field(default_factory=dict)  # deep-merged into the baseline config
    files: dict = field(default_factory=dict)    # path -> content, written into the worktree
    kind: str = "software"                        # software | hardware | deep_hardware
    workload: str | None = None
    benchmark_overrides: dict = field(default_factory=dict)  # only rounds/repetitions
    revisit_reason: str = ""                      # required to re-test a tested config
    id: str = field(default_factory=new_id)

    @classmethod
    def from_dict(cls, d: dict) -> "ExperimentSpec":
        known = {k: d[k] for k in cls.__dataclass_fields__ if k in d}
        return cls(**known)


def deep_merge(base: dict, changes: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in changes.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        elif v is None:
            out.pop(k, None)
        else:
            out[k] = v
    return out


def config_hash(cfg: dict) -> str:
    keep = {k: cfg.get(k) for k in ("workload", "model", "backend", "build", "runtime", "env")}
    return hashlib.sha256(json.dumps(keep, sort_keys=True).encode()).hexdigest()[:16]


def save_record(rec: dict, records: Path = RECORDS) -> Path:
    records.mkdir(parents=True, exist_ok=True)
    p = records / f"{rec['id']}.json"
    p.write_text(json.dumps(rec, indent=2, default=str) + "\n")
    return p


def load_records(records: Path = RECORDS) -> list[dict]:
    out = []
    for p in sorted(records.glob("*.json")):
        try:
            out.append(json.loads(p.read_text()))
        except ValueError:
            pass
    return out


def append_tsv(rec: dict, path: Path = RESULTS_TSV):
    path.parent.mkdir(parents=True, exist_ok=True)
    ev = rec.get("evaluation") or {}
    m = ev.get("metrics") or {}

    def r(name):
        return m.get(name, {}).get("ratio", "")

    def c(name):
        return m.get(name, {}).get("candidate", "")

    row = {
        "id": rec["id"], "finished_utc": rec.get("finished_utc", ""),
        "workload": rec.get("execution_environment", {}).get("workload", ""),
        "kind": rec.get("kind", ""), "decision": rec.get("decision", ""),
        "balanced_score": ev.get("balanced_score", ""),
        "tg128_ratio": r("tg128_ts"), "pp512_ratio": r("pp512_ts"), "ttft_ratio": r("ttft_s"),
        "vram_ratio": r("vram_peak_mib"), "cand_tg128_ts": c("tg128_ts"), "cand_pp512_ts": c("pp512_ts"),
        "mean_kld": (rec.get("correctness") or {}).get("candidate", {}).get("mean_kld", ""),
        "hypothesis": re.sub(r"\s+", " ", rec.get("hypothesis", ""))[:200],
    }
    new = not path.exists()
    with open(path, "a") as fh:
        if new:
            fh.write("\t".join(TSV_COLUMNS) + "\n")
        fh.write("\t".join(str(row[k]) for k in TSV_COLUMNS) + "\n")


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
