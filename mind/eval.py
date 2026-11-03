"""Measures of the mechanisms (not of whether it thinks what we would).

Reads the world's hidden rules (which the mind never sees) and the mind's own
store, and reports: how much of the world it has learned, how surprise changed,
whether dreaming beats random recombination, and what it cost.
"""
from __future__ import annotations

import json
from pathlib import Path

from mind.store import Store


def _cause_truth(rules):
    """(trigger, effect) -> delays that are true at some point in the world's life."""
    truth = {}
    for r in rules:
        if r["kind"] == "cause":
            a = "=".join(r["trigger"])
            b = "=".join(r["effect"])
            lags = {r["delay"]} | ({r["delay_after"]} if "delay_after" in r else set())
            truth[(a, b)] = lags
    return truth


def evaluate(state_dir: Path) -> dict:
    rules = json.loads((state_dir / "world" / "rules.json").read_text())["rules"]
    st = Store(state_dir / "mind.sqlite")
    truth = _cause_truth(rules)
    beliefs = st.beliefs()
    found, cause_rows = set(), []
    for b in beliefs:
        if b["kind"] != "cause":
            continue
        true_lags = truth.get((b["a"], b["b"]))
        match = bool(true_lags) and any(abs(b["lag"] - L) <= 0 for L in true_lags)
        if match and b["status"] == "belief":
            found.add((b["a"], b["b"]))
        cause_rows.append(dict(**b, true_rule=match))
    by_status = {}
    for r in cause_rows:
        s = by_status.setdefault(r["status"], dict(n=0, true_rule=0))
        s["n"] += 1
        s["true_rule"] += r["true_rule"]
    cycles = [r for r in rules if r["kind"] == "cycle"]
    cyc_beliefs = [b for b in beliefs if b["kind"] == "cycle"]
    cyc_found = sum(1 for r in cycles if any(b["a"].split("=")[0] == r["feature"] and
                                             abs(b["lag"] - r["period"]) <= 1 for b in cyc_beliefs))

    def rate(kind):
        rows = st.db.execute("SELECT data FROM stream WHERE kind=?", (kind,)).fetchall()
        n = len(rows)
        p = sum(1 for (d,) in rows if json.loads(d).get("passed"))
        return dict(n=n, passed=p, rate=round(p / n, 3) if n else None)

    dream_true = sum(1 for r in cause_rows if r["origin"] == "dream" and r["true_rule"])
    # surprise trajectory per param window
    surprise = [json.loads(d)["mean_surprise"] for (d,) in st.db.execute(
        "SELECT data FROM stream WHERE kind='self_tune' ORDER BY id")]
    states = dict(st.db.execute("SELECT kind, count(*) FROM stream GROUP BY kind").fetchall())
    ck = st.get("checkpoint") or {}
    occupancy = dict(st.db.execute(
        "SELECT state, count(*) FROM stream WHERE kind='state' GROUP BY state").fetchall())
    out = dict(
        identity=st.get("identity"), boots=st.get("boots"), ticks=ck.get("tick"),
        causes_in_world=len(truth), causes_believed_correctly=len(found),
        cause_beliefs_by_status=by_status,
        cycles_in_world=len(cycles), cycles_found=cyc_found, cycle_beliefs=len(cyc_beliefs),
        dream=rate("dream"), dream_control=rate("dream_control"), dream_hypotheses_true=dream_true,
        surprise_by_window=surprise, events=states, state_entries=occupancy,
        params=(st.get("params") or {}).get("values"),
        param_changes_kept=sum(1 for h in (st.get("params") or {}).get("history", []) if h.get("kept")),
        param_changes_tried=len((st.get("params") or {}).get("history", [])),
    )
    st.close()
    return out


def report_md(r: dict) -> str:
    s = r["surprise_by_window"]
    lines = [
        f"# Mind report: {r['identity']['id'][:8]}", "",
        f"Born {r['identity']['born']}, {r['boots']} boots, {r['ticks']} ticks lived.", "",
        "| measure | value |", "|---|---|",
        f"| delayed causes in the world / believed correctly | {r['causes_in_world']} / {r['causes_believed_correctly']} |",
        f"| cycles in the world / found | {r['cycles_in_world']} / {r['cycles_found']} |",
        f"| cause beliefs by status (n, matching a true rule) | "
        + ", ".join(f"{k}: {v['n']} ({v['true_rule']})" for k, v in sorted(r['cause_beliefs_by_status'].items())) + " |",
        f"| dream pairs passing the memory test vs random-pair control | "
        f"{r['dream']['rate']} ({r['dream']['n']}) vs {r['dream_control']['rate']} ({r['dream_control']['n']}) |",
        f"| dream hypotheses that are true rules | {r['dream_hypotheses_true']} |",
        f"| mean surprise, first window / last window (bits) | "
        f"{s[0] if s else '-'} / {s[-1] if s else '-'} |",
        f"| self-tuning changes kept / tried | {r['param_changes_kept']} / {r['param_changes_tried']} |",
        f"| thoughts started / attention entries | {r['events'].get('thought_start', 0)} / {r['events'].get('attend', 0)} |",
        f"| state entries | " + ", ".join(f"{k} {v}" for k, v in sorted(r['state_entries'].items())) + " |",
        "", "Current self-tuned settings: " + ", ".join(f"{k}={v}" for k, v in (r["params"] or {}).items()),
    ]
    return "\n".join(lines) + "\n"
