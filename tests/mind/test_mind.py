from pathlib import Path

import random

from mind import senses as S
from mind.core import Mind
from mind.eval import evaluate
from mind.params import PARAMS, Params
from mind.store import Store
from mind.world import World


def make(tmp: Path, cfg=None):
    ss = []
    m = Mind(tmp, ss, cfg or {})
    ss += S.build(tmp, m.identity["seed"], real=False, repo=tmp)
    return m


def test_world_is_generated_and_deterministic(tmp_path):
    a, b = World(123), World(123)
    assert a.rules == b.rules and a.features == b.features
    assert [a.step() for _ in range(300)] == [b.step() for _ in range(300)]
    kinds = {r["kind"] for r in a.rules}
    assert {"cycle", "cause", "noise"} <= kinds
    assert World(124).features != a.features


def test_world_continues_across_restart(tmp_path):
    w = World.load_or_create(tmp_path, 7)
    for _ in range(50):
        w.step()
    w.save(tmp_path)
    w2 = World.load_or_create(tmp_path, 7)
    assert w2.tick == 50
    assert [w.step() for _ in range(100)] == [w2.step() for _ in range(100)]


def test_identity_survives_restarts_and_history_is_kept(tmp_path):
    m = make(tmp_path)
    ident = m.identity
    m.run(ticks=200, fast=True)
    m.store.close()
    m2 = make(tmp_path)
    assert m2.identity == ident
    assert m2.boots == 2 and m2.tick == 200
    kinds = [e["kind"] for e in m2.store.stream(limit=10000)]
    assert kinds.count("birth") == 1 and kinds.count("boot") == 2


def test_hard_kill_loses_at_most_uncommitted_ticks(tmp_path):
    m = make(tmp_path, {"checkpoint_every": 1000})
    for _ in range(30):
        m.step()
        m.store.commit()          # real-time mode commits every tick
    n = m.store.db.execute("SELECT count(*) FROM percepts").fetchone()[0]
    m.store.db.close()            # no checkpoint, no shutdown: like kill -9
    st = Store(tmp_path / "mind.sqlite")
    assert st.db.execute("SELECT count(*) FROM percepts").fetchone()[0] == n
    st.close()
    m2 = make(tmp_path)           # restarts from the last checkpoint, history intact
    assert m2.identity["id"] == m.identity["id"]


def test_no_seeded_content(tmp_path):
    m = make(tmp_path)
    assert m.store.beliefs() == []
    assert m.interests == {} and m.workspace == {}
    assert m.params.values == {k: v[0] for k, v in PARAMS.items()}


def test_self_tuning_stays_in_bounds():
    p, rng = Params(), random.Random(0)
    for i in range(500):
        p.evaluate(rng.random(), rng)
        for k, (_, lo, hi, _) in PARAMS.items():
            assert lo <= p[k] <= hi


def test_a_short_life_learns_its_world(tmp_path):
    m = make(tmp_path)
    m.run(ticks=15000, fast=True)
    m.store.close()
    r = evaluate(tmp_path)
    assert r["cycles_found"] >= 1
    assert r["surprise_by_window"][-1] < r["surprise_by_window"][0]
    states = set(r["state_entries"])
    assert {"deep_sleep", "dreaming"} <= states


def test_backup_is_a_readable_copy(tmp_path):
    m = make(tmp_path / "s")
    m.run(ticks=50, fast=True)
    m.store.backup(tmp_path / "b" / "x.sqlite")
    st = Store(tmp_path / "b" / "x.sqlite")
    assert st.get("identity") == m.identity
    st.close()


# ---------------------------------------------------------------- A2: language tiers (fake server)
import json as _json
import re as _re
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

from mind.tiers import TierPool, parse_json


class _Fake(BaseHTTPRequestHandler):
    prompts = []

    def log_message(self, *a):
        pass

    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"ok")

    def do_POST(self):
        body = _json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        prompt = body["messages"][0]["content"]
        _Fake.prompts.append(prompt)
        items = _re.findall(r"^\d+\. (\S+=\S+) \(seen", prompt, flags=_re.M)
        if "proposals" in prompt and len(items) >= 2:
            content = "<think>hmm</think> " + _json.dumps({"proposals": [{"a": 1, "b": 2, "lag": 2},
                                                                         {"a": 99, "b": 1, "lag": 1}]})
        elif "note" in prompt:
            content = _json.dumps({"note": "things that come before others keep coming before them",
                                   "questions": []})
        else:
            content = _json.dumps({"keywords": ["greeting"]})
        out = _json.dumps({"choices": [{"message": {"content": content}}],
                           "usage": {"completion_tokens": 12}}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(out)


def _fake_pool():
    srv = HTTPServer(("127.0.0.1", 0), _Fake)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{srv.server_port}"
    return srv, TierPool({"fast": {"url": url}, "reflective": {"url": url, "json_schema": False}}, Path("."))


def test_parse_json_takes_last_object_after_thinking():
    assert parse_json('<think>{"x": 1}</think> answer: {"a": {"b": 2}}') == {"a": {"b": 2}}
    assert parse_json("no json") is None


def test_tiers_dream_reflect_and_read(tmp_path):
    srv, pool = _fake_pool()
    ss = []
    m = Mind(tmp_path, ss, {"llm_dream_every": 3, "reflect_every": 5}, tiers=pool)
    ss += S.build(tmp_path, m.identity["seed"], real=False, repo=tmp_path)
    m.run(ticks=6000, fast=True)
    (tmp_path / "inbox").mkdir(exist_ok=True)
    m.extra_percepts = []
    from mind import language
    language.interpret(m, m.tick, ["m0.w=hello", "m0.w=there"])
    m.run(ticks=2, fast=True)
    kinds = {e["kind"] for e in m.store.stream(limit=100000)}
    assert {"tier_call", "dream_llm"} <= kinds
    calls = [e["data"] for e in m.store.stream(kinds=["tier_call"], limit=1000)]
    assert all(c["error"] is None for c in calls)
    llm = [e["data"] for e in m.store.stream(kinds=["dream_llm"], limit=1000)]
    assert any(d["valid"] for d in llm) and any(not d["valid"] for d in llm)   # invented items are rejected
    assert "m0.k=greeting" in m.pred.count
    # no persona: prompts carry only the task and the mind's own material
    assert not any(w in p.lower() for p in _Fake.prompts for w in ("you are", "your name", "assistant"))
    pool.close()
    srv.shutdown()


def test_missing_tiers_fall_back_to_algorithms(tmp_path):
    pool = TierPool({"reflective": {"model": "models/none.gguf"}}, tmp_path)
    ss = []
    m = Mind(tmp_path, ss, {"llm_dream_every": 1}, tiers=pool)
    ss += S.build(tmp_path, m.identity["seed"], real=False, repo=tmp_path)
    m.run(ticks=3000, fast=True)
    assert m.store.count("tier_call") == 0
    pool.close()
