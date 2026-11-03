"""Processes that recruit language tiers (A2).

Each function builds one question from the mind's own material and submits it to
a tier; collect() applies answers when they arrive. Every call is logged in the
life stream with its tier, seconds, tokens and any error. Nothing here tells the
mind who it is or what to care about: prompts carry only the task and its own
memories, beliefs and earlier notes.

  dream      reflective tier proposes relations between memories (compared with
             graph-walk dreams and the random control by the same memory test)
  reflect    reflective tier writes a short note on recent changes and may ask
             questions; notes are kept and shown back to it in later prompts
  interpret  fast tier turns incoming text into a few keyword percepts
"""
from __future__ import annotations

from mind.predict import MAX_LAG, fv

PAIR = {"type": "object", "properties": {"a": {"type": "string"}, "b": {"type": "string"},
                                         "lag": {"type": "integer"}}, "required": ["a", "b", "lag"]}
DREAM_SCHEMA = {"type": "object", "properties": {"proposals": {"type": "array", "items": PAIR, "maxItems": 3}},
                "required": ["proposals"]}
REFLECT_SCHEMA = {"type": "object", "properties": {"note": {"type": "string"},
                                                   "questions": {"type": "array", "items": PAIR, "maxItems": 3}},
                  "required": ["note"]}
READ_SCHEMA = {"type": "object", "properties": {"keywords": {"type": "array", "items": {"type": "string"},
                                                             "maxItems": 5}}, "required": ["keywords"]}


def _allowed(m):
    """Hard cap on tier calls per self-tuning window (config, not self-tunable)."""
    if m.tiers.pending or m.win.get("tier", 0) >= m.cfg["max_tier_calls_per_window"]:
        return False
    m.win["tier"] = m.win.get("tier", 0) + 1
    return True


def _notes(m, n=3):
    return (m.store.get("notes") or [])[-n:]


def _neighbours(m, c, k=2):
    nb = []
    for key, w in m.edges.items():
        a, b = key.split("\t")
        if c in (a, b):
            nb.append((w, b if a == c else a))
    return [x for _, x in sorted(nb, reverse=True)[:k]]


def _beliefs_text(m, limit=6):
    bs = sorted(m.store.beliefs("belief"), key=lambda b: -b["confidence"])[:limit]
    return [f"- {b['a']} then {b['b']} after {b['lag']} ticks (confidence {b['confidence']:.2f})"
            if b["kind"] == "cause" else f"- {b['a']} repeats every {b['lag']} ticks" for b in bs]


def dream(m, t):
    if not m.tiers.has("reflective") or not _allowed(m):
        return
    items = []
    for _ in range(12):
        r = m.store.random_percept(m.rng)
        if r and r[0] not in items:
            items.append(r[0])
    if len(items) < 4:
        return
    lines = [f"- {c} (seen {m.pred.count.get(c, 0)} times; often near: {', '.join(_neighbours(m, c)) or 'nothing yet'})"
             for c in items]
    prompt = "\n".join([
        "Items from a memory, written as feature=value:", *lines, "",
        "Relations already known:", *(_beliefs_text(m) or ["- none"]), "",
        *(["Earlier notes:", *[f"- {n}" for n in _notes(m)], ""] if _notes(m) else []),
        f"Propose up to 3 new relations of the form: item a happens, then item b happens lag ticks later "
        f"(lag 1 to {MAX_LAG}). Use items exactly as written. Answer with JSON only: "
        '{"proposals": [{"a": "...", "b": "...", "lag": 1}]}'])
    m.tiers.submit("reflective", "dream", prompt, DREAM_SCHEMA, dict(items=items))


def reflect(m, t):
    if not m.tiers.has("reflective"):
        return
    recent = m.store.stream(kinds=["belief_belief", "belief_doubted", "belief_rejected", "thought_end",
                                   "hypothesis"], limit=10)
    if not recent or recent[0]["tick"] <= (m.store.get("last_reflect_tick") or 0) or not _allowed(m):
        return   # nothing new since the last reflection
    m.store.put("last_reflect_tick", t)
    lines = []
    for e in reversed(recent):
        d = e["data"]
        if e["kind"] == "thought_end":
            lines.append(f"- tick {e['tick']}: thought about {d.get('focus')} ended: {d.get('outcome')}")
        else:
            lines.append(f"- tick {e['tick']}: {e['kind'].replace('belief_', '')}: {d.get('a')} then {d.get('b')} "
                         f"after {d.get('lag')} ticks")
    prompt = "\n".join([
        "Recent changes in a memory:", *lines, "",
        "Relations currently held:", *(_beliefs_text(m) or ["- none"]), "",
        *(["Earlier notes:", *[f"- {n}" for n in _notes(m)], ""] if _notes(m) else []),
        "Write a short note (one or two sentences) about what these changes suggest, and up to 3 questions in the "
        "form of relations worth testing. Answer with JSON only: "
        '{"note": "...", "questions": [{"a": "...", "b": "...", "lag": 1}]}'])
    m.tiers.submit("reflective", "reflect", prompt, REFLECT_SCHEMA)


def interpret(m, t, words):
    if not words or not m.tiers.has("fast"):
        return
    text = " ".join(w.split("=", 1)[1] for w in words)
    prompt = ("Text: " + text + "\nList up to 5 short lowercase keywords for what this text is about. "
              'Answer with JSON only: {"keywords": ["..."]}')
    m.tiers.submit("fast", "interpret", prompt, READ_SCHEMA)


def _test_proposals(m, t, props, origin):
    out = []
    for p in (props or [])[:3]:
        if not isinstance(p, dict):
            continue
        a, b = str(p.get("a", "")).strip(), str(p.get("b", "")).strip()
        try:
            lag = int(p.get("lag", 0))
        except (TypeError, ValueError):
            lag = 0
        valid = (a in m.pred.count and b in m.pred.count and a != b and fv(a)[0] != fv(b)[0]
                 and 1 <= lag <= MAX_LAG)
        if not valid:
            m.store.log(t, "dream_llm" if origin == "dream_llm" else "question", m.state,
                        dict(a=a, b=b, lag=lag, valid=False, passed=False))
            continue
        res = m._replay_test(a, b, lag)
        m.store.log(t, "dream_llm" if origin == "dream_llm" else "question", m.state,
                    dict(a=a, b=b, lag=lag, valid=True, lift=res["lift"], passed=res["pass"]))
        if res["pass"]:
            m._hypothesis(t, a, b, lag, res["conf"], origin)
        out.append(res["pass"])
    return out


def collect(m, t):
    wait = m.cfg["tier_wait_s"] if (m.fast_mode and m.tiers.pending) else 0.0
    for r in m.tiers.drain(wait):
        m.store.log(t, "tier_call", m.state, dict(tier=r["tier"], kind=r["kind"], seconds=r.get("seconds"),
                                                  tokens=r.get("tokens"), reasoning_chars=r.get("reasoning_chars"),
                                                  parsed=r.get("json") is not None, error=r.get("error")),
                    cost=r.get("seconds") or 0.0)
        j = r.get("json")
        if r.get("error") or not isinstance(j, dict):
            continue
        if r["kind"] == "dream":
            _test_proposals(m, t, j.get("proposals"), "dream_llm")
        elif r["kind"] == "reflect":
            note = str(j.get("note", "")).strip()[:400]
            if note:
                notes = (m.store.get("notes") or []) + [note]
                m.store.put("notes", notes[-20:])
                m.store.log(t, "reflection", m.state, dict(note=note))
            _test_proposals(m, t, j.get("questions"), "reflection")
        elif r["kind"] == "interpret":
            kws = [str(k).strip().lower()[:24] for k in (j.get("keywords") or [])][:5]
            m.extra_percepts += [("inbox", f"m0.k={k}") for k in kws if k]
