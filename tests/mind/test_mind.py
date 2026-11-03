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
