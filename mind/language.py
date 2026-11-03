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

PAIR = {"type": "object", "properties": {"a": {"type": "integer"}, "b": {"type": "integer"},
                                         "lag": {"type": "integer"}}, "required": ["a", "b", "lag"]}
DREAM_SCHEMA = {"type": "object", "properties": {"proposals": {"type": "array", "items": PAIR, "maxItems": 3}},
                "required": ["proposals"]}
REFLECT_SCHEMA = {"type": "object", "properties": {"note": {"type": "string"},
                                                   "questions": {"type": "array", "items": PAIR, "maxItems": 3}},
                  "required": ["note"]}
PAIR_FORMAT = (f'Refer to items only by their number. Each relation means: item a happens, then item b happens '
               f'lag ticks later (lag 1 to {MAX_LAG}).')
READ_SCHEMA = {"type": "object", "properties": {"keywords": {"type": "array", "items": {"type": "string"},
                                                             "maxItems": 5}}, "required": ["keywords"]}


def _allowed(m, kind):
    """Hard cap on tier calls per self-tuning window, per kind (config, not self-tunable)."""
    key = "tier_" + kind
    if m.tiers.pending or m.win.get(key, 0) >= m.cfg["max_tier_calls_per_window"]:
        return False
    m.win[key] = m.win.get(key, 0) + 1
    return True


def _numbered(items):
    return [f"{i}. {c}" for i, c in enumerate(items, 1)]


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
    tier = m.cfg.get("dream_tier", "reflective")
    if not m.tiers.has(tier) or not _allowed(m, "dream"):
        return
    items = []
    for _ in range(12):
        r = m.store.random_percept(m.tier_rng)
        if r and r[0] not in items:
            items.append(r[0])
    if len(items) < 4:
        return
    lines = [f"{i}. {c} (seen {m.pred.count.get(c, 0)} times; often near: "
             f"{', '.join(_neighbours(m, c)) or 'nothing yet'})" for i, c in enumerate(items, 1)]
    prompt = "\n".join([
        "Numbered items from a memory (feature=value):", *lines, "",
        "Relations already known:", *(_beliefs_text(m) or ["- none"]), "",
        "Propose up to 3 new relations between the numbered items. " + PAIR_FORMAT + " Answer with JSON only, "
        'for example: {"proposals": [{"a": 3, "b": 7, "lag": 2}]}'])
    m.tiers.submit(tier, "dream", prompt, DREAM_SCHEMA, dict(items=items))
    m.store.log(t, "tier_ask", m.state, dict(kind="dream", items=items))


def reflect(m, t):
    if not m.tiers.has("reflective"):
        return
    recent = m.store.stream(kinds=["belief_belief", "belief_doubted", "belief_rejected", "thought_end",
                                   "hypothesis"], limit=10)
    if not recent or recent[0]["tick"] <= (m.store.get("last_reflect_tick") or 0) or not _allowed(m, "reflect"):
        return   # nothing new since the last reflection
    m.store.put("last_reflect_tick", t)
    lines, items = [], []

    def ref(c):
        if c and c not in items:
            items.append(c)
        return f"[{items.index(c) + 1}]" if c else "?"

    for e in reversed(recent):
        d = e["data"]
        if e["kind"] == "thought_end":
            focus = str(d.get("focus", ""))
            lines.append(f"- tick {e['tick']}: thought about {ref(focus.removeprefix('missing:'))} ended: "
                         f"{d.get('outcome')}")
        else:
            lines.append(f"- tick {e['tick']}: {e['kind'].replace('belief_', '')}: {ref(d.get('a'))} then "
                         f"{ref(d.get('b'))} after {d.get('lag')} ticks")
    last = _notes(m, 1)
    prompt = "\n".join([
        "Numbered items:", *_numbered(items), "",
        "Recent changes in a memory (items in brackets):", *lines, "",
        "Relations currently held:", *(_beliefs_text(m) or ["- none"]), "",
        *(["Previous note: " + last[0], ""] if last else []),
        "Write a short note (one or two sentences) about what these specific changes suggest, and up to 3 "
        "relations between the numbered items worth testing. " + PAIR_FORMAT + " Answer with JSON only, "
        'for example: {"note": "...", "questions": [{"a": 1, "b": 2, "lag": 3}]}'])
    m.tiers.submit("reflective", "reflect", prompt, REFLECT_SCHEMA, dict(items=items))


def interpret(m, t, words):
    if not words or not m.tiers.has("fast"):
        return
    text = " ".join(w.split("=", 1)[1] for w in words)
    prompt = ("Text: " + text + "\nList up to 5 short lowercase keywords for what this text is about. "
              'Answer with JSON only: {"keywords": ["..."]}')
    m.tiers.submit("fast", "interpret", prompt, READ_SCHEMA)


def _item(items, x):
    try:
        i = int(x)
    except (TypeError, ValueError):
        return str(x or "").strip()   # a name instead of a number: validated below like anything else
    return items[i - 1] if 1 <= i <= len(items) else f"#{i}"


def _test_proposals(m, t, props, origin, items):
    out = []
    for p in (props or [])[:3]:
        if not isinstance(p, dict):
            continue
        a, b = _item(items, p.get("a")), _item(items, p.get("b"))
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


EXPECTED = {"dream": "proposals", "reflect": "note", "interpret": "keywords"}


def _usable(r):
    j = r.get("json")
    return isinstance(j, dict) and EXPECTED.get(r["kind"], "") in j


def collect(m, t):
    wait = m.cfg["tier_wait_s"] if (m.fast_mode and m.tiers.pending) else 0.0
    for r in m.tiers.drain(wait):
        m.store.log(t, "tier_call", m.state, dict(tier=r["tier"], kind=r["kind"], seconds=r.get("seconds"),
                                                  tokens=r.get("tokens"), reasoning_chars=r.get("reasoning_chars"),
                                                  parsed=_usable(r), error=r.get("error"),
                                                  reply=(r.get("content") or "")[-1500:]),
                    cost=r.get("seconds") or 0.0)
        j = r.get("json")
        if r.get("error") or not _usable(r):
            continue
        if r["kind"] == "dream":
            _test_proposals(m, t, j.get("proposals"), "dream_llm", r["context"].get("items", []))
        elif r["kind"] == "reflect":
            note = str(j.get("note", "")).strip()[:400]
            if note:
                notes = (m.store.get("notes") or []) + [note]
                m.store.put("notes", notes[-20:])
                m.store.log(t, "reflection", m.state, dict(note=note))
            _test_proposals(m, t, j.get("questions"), "reflection", r["context"].get("items", []))
        elif r["kind"] == "interpret":
            kws = [str(k).strip().lower()[:24] for k in (j.get("keywords") or [])][:5]
            m.extra_percepts += [("inbox", f"m0.k={k}") for k in kws if k]
