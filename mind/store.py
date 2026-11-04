"""Persistent state of the mind: one SQLite file (WAL), never recreated.

Tables
  kv         identity, checkpoints of in-RAM structures (JSON)
  stream     append-only life stream: everything that happens, with cost
  percepts   episodic memory: what was perceived, when, and how surprising
  beliefs    hypotheses and beliefs about regularities, with evidence
  thoughts   thought objects (active / suspended / resolved) with continuations
"""
from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS kv(key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS stream(id INTEGER PRIMARY KEY, tick INTEGER, t REAL, kind TEXT,
    state TEXT, data TEXT, cost REAL DEFAULT 0);
CREATE INDEX IF NOT EXISTS stream_kind ON stream(kind, id);
CREATE TABLE IF NOT EXISTS percepts(id INTEGER PRIMARY KEY, tick INTEGER, source TEXT,
    concept TEXT, surprise REAL);
CREATE INDEX IF NOT EXISTS percepts_tick ON percepts(tick);
CREATE INDEX IF NOT EXISTS percepts_concept ON percepts(concept, tick);
CREATE TABLE IF NOT EXISTS beliefs(id INTEGER PRIMARY KEY, kind TEXT, a TEXT, b TEXT, lag INTEGER,
    confidence REAL, support INTEGER, origin TEXT, status TEXT, created INTEGER, updated INTEGER,
    UNIQUE(kind, a, b, lag));
CREATE TABLE IF NOT EXISTS thoughts(id INTEGER PRIMARY KEY, created INTEGER, updated INTEGER,
    status TEXT, focus TEXT, step INTEGER, data TEXT);
"""


class Store:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.db = sqlite3.connect(str(path), isolation_level=None, check_same_thread=False)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=NORMAL")
        self.db.executescript(SCHEMA)
        self.db.execute("BEGIN")

    # -- transactions: one per tick (or per batch of ticks in fast mode)
    def commit(self):
        self.db.execute("COMMIT")
        self.db.execute("BEGIN")

    def close(self):
        try:
            self.db.execute("COMMIT")
        except sqlite3.OperationalError:
            pass
        self.db.close()

    # -- key/value
    def get(self, key, default=None):
        r = self.db.execute("SELECT value FROM kv WHERE key=?", (key,)).fetchone()
        return json.loads(r[0]) if r else default

    def put(self, key, value):
        self.db.execute("INSERT OR REPLACE INTO kv(key, value) VALUES(?, ?)", (key, json.dumps(value)))

    # -- life stream
    def log(self, tick, kind, state, data=None, cost=0.0):
        self.db.execute("INSERT INTO stream(tick, t, kind, state, data, cost) VALUES(?,?,?,?,?,?)",
                        (tick, time.time(), kind, state, json.dumps(data or {}), cost))

    def stream(self, kinds=None, limit=50, exclude=()):
        q, args = "SELECT id, tick, t, kind, state, data, cost FROM stream WHERE 1", []
        if kinds:
            q += f" AND kind IN ({','.join('?' * len(kinds))})"
            args += list(kinds)
        if exclude:
            q += f" AND kind NOT IN ({','.join('?' * len(exclude))})"
            args += list(exclude)
        q += " ORDER BY id DESC LIMIT ?"
        rows = self.db.execute(q, (*args, limit)).fetchall()
        return [dict(id=r[0], tick=r[1], t=r[2], kind=r[3], state=r[4], data=json.loads(r[5]), cost=r[6])
                for r in rows]

    def count(self, kind):
        return self.db.execute("SELECT count(*) FROM stream WHERE kind=?", (kind,)).fetchone()[0]

    # -- episodic memory
    def add_percept(self, tick, source, concept, surprise):
        self.db.execute("INSERT INTO percepts(tick, source, concept, surprise) VALUES(?,?,?,?)",
                        (tick, source, concept, surprise))

    def occurrences(self, concept):
        return self.db.execute("SELECT count(*) FROM percepts WHERE concept=?", (concept,)).fetchone()[0]

    def co_occurrences(self, a, b, lag):
        """How often b was perceived exactly `lag` ticks after a (replay from memory)."""
        return self.db.execute(
            "SELECT count(*) FROM percepts pa JOIN percepts pb ON pb.tick = pa.tick + ? AND pb.concept = ? "
            "WHERE pa.concept = ?", (lag, b, a)).fetchone()[0]

    def memory_span(self):
        r = self.db.execute("SELECT min(tick), max(tick), count(DISTINCT tick) FROM percepts").fetchone()
        return (r[0] or 0, r[1] or 0, r[2] or 0)

    def percepts_at(self, tick):
        return [r[0] for r in self.db.execute("SELECT concept FROM percepts WHERE tick=?", (tick,))]

    def random_percept(self, rng):
        lo, hi = self.db.execute("SELECT min(id), max(id) FROM percepts").fetchone()
        if lo is None:
            return None
        r = self.db.execute("SELECT concept, tick FROM percepts WHERE id >= ? ORDER BY id LIMIT 1",
                            (rng.randint(lo, hi),)).fetchone()
        return r

    def forget(self, before_tick, surprise_below):
        cur = self.db.execute("DELETE FROM percepts WHERE tick < ? AND surprise < ?", (before_tick, surprise_below))
        return cur.rowcount

    # -- beliefs
    def belief(self, kind, a, b, lag):
        r = self.db.execute("SELECT id, confidence, support, origin, status FROM beliefs "
                            "WHERE kind=? AND a=? AND b=? AND lag=?", (kind, a, b, lag)).fetchone()
        return dict(id=r[0], confidence=r[1], support=r[2], origin=r[3], status=r[4]) if r else None

    def upsert_belief(self, tick, kind, a, b, lag, confidence, support, origin, status):
        old = self.belief(kind, a, b, lag)
        if old:
            self.db.execute("UPDATE beliefs SET confidence=?, support=?, status=?, updated=? WHERE id=?",
                            (confidence, support, status, tick, old["id"]))
        else:
            self.db.execute("INSERT INTO beliefs(kind, a, b, lag, confidence, support, origin, status, created, "
                            "updated) VALUES(?,?,?,?,?,?,?,?,?,?)",
                            (kind, a, b, lag, confidence, support, origin, status, tick, tick))
        return old

    def beliefs(self, status=None):
        q = "SELECT kind, a, b, lag, confidence, support, origin, status, created, updated FROM beliefs"
        rows = self.db.execute(q + (" WHERE status=?" if status else ""), (status,) if status else ()).fetchall()
        keys = ("kind", "a", "b", "lag", "confidence", "support", "origin", "status", "created", "updated")
        return [dict(zip(keys, r)) for r in rows]

    # -- thoughts
    def new_thought(self, tick, focus, data):
        cur = self.db.execute("INSERT INTO thoughts(created, updated, status, focus, step, data) "
                              "VALUES(?,?,?,?,?,?)", (tick, tick, "active", focus, 0, json.dumps(data)))
        return cur.lastrowid

    def save_thought(self, tid, tick, status, step, data):
        self.db.execute("UPDATE thoughts SET updated=?, status=?, step=?, data=? WHERE id=?",
                        (tick, status, step, json.dumps(data), tid))

    def thoughts(self, statuses):
        rows = self.db.execute(
            f"SELECT id, focus, step, data, status, created FROM thoughts WHERE status IN "
            f"({','.join('?' * len(statuses))}) ORDER BY updated", tuple(statuses)).fetchall()
        return [dict(id=r[0], focus=r[1], step=r[2], data=json.loads(r[3]), status=r[4], created=r[5])
                for r in rows]

    def set_thought_status(self, old, new):
        self.db.execute("UPDATE thoughts SET status=? WHERE status=?", (new, old))

    def backup(self, dest: Path):
        dest.parent.mkdir(parents=True, exist_ok=True)
        self.db.execute("COMMIT")
        try:
            out = sqlite3.connect(str(dest))
            self.db.backup(out)
            out.close()
        finally:
            self.db.execute("BEGIN")
