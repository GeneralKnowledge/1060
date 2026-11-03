"""Cheap predictors: no model needed.

A concept is "feature=value". Each tick the mind sees a set of concepts.
The predictor keeps:
  * value transitions per feature (Markov),
  * intervals between occurrences of each concept (periodicity),
  * a limited set of *trackers*: candidate regularities "a, then b after lag
    ticks" that the mind is currently paying attention to. Which pairs get a
    tracker is decided by thinking and dreaming, not enumerated: capacity is
    small, so what the mind attends to determines what it can learn.
Surprise of a percept = -log2 of the probability the predictor gave it.
"""
from __future__ import annotations

import math
from collections import defaultdict, deque

P_FLOOR = 0.01
MAX_LAG = 10


def fv(concept):
    f, _, v = concept.partition("=")
    return f, v


class Predictor:
    def __init__(self, capacity: int = 300):
        self.capacity = capacity
        self.trans = defaultdict(lambda: defaultdict(int))     # feature -> "prev>next" -> n
        self.last = {}                                          # feature -> last value
        self.count = defaultdict(int)                           # concept -> occurrences
        self.last_seen = {}                                     # concept -> tick
        self.intervals = defaultdict(lambda: deque(maxlen=8))   # concept -> recent intervals
        self.trackers = {}                                      # "a|b|lag" -> [n, hits]
        self.ticks = 0
        self.recent = deque(maxlen=MAX_LAG + 1)                 # sets of concepts, newest last

    # -- probabilities
    def p_markov(self, concept):
        f, v = fv(concept)
        prev = self.last.get(f)
        if prev is None:
            return None
        row = self.trans[f]
        tot = sum(n for k, n in row.items() if k.startswith(prev + ">"))
        return (row.get(f"{prev}>{v}", 0) + 0.5) / (tot + 1.0 + 0.5 * len(row)) if tot else None

    def p_periodic(self, concept, tick):
        iv = self.intervals.get(concept)
        if not iv or len(iv) < 3:
            return None
        mean = sum(iv) / len(iv)
        sd = (sum((x - mean) ** 2 for x in iv) / len(iv)) ** 0.5
        if sd > 0.1 * mean + 0.5:
            return None
        return 0.9 if abs((tick - self.last_seen[concept]) - mean) <= max(1.0, sd) else 0.05

    def tracker_conf(self, key):
        n, hits = self.trackers[key]
        return (hits + 0.5) / (n + 1.0)

    def p_trackers(self, concept, tick):
        best = None
        for key in self.trackers:
            a, b, lag = key.split("|")
            if b != concept:
                continue
            lag = int(lag)
            if lag <= len(self.recent) and a in self.recent[-lag]:
                c = self.tracker_conf(key)
                best = c if best is None else max(best, c)
        return best

    def expectations(self, tick, min_p=0.75):
        """Concepts the predictor confidently expects this tick (for missing-prediction errors)."""
        exp = {}
        for key, (n, hits) in self.trackers.items():
            if n < 5:
                continue
            a, b, lag = key.split("|")
            lag = int(lag)
            if lag <= len(self.recent) and a in self.recent[-lag]:
                c = self.tracker_conf(key)
                if c >= min_p:
                    exp[b] = max(exp.get(b, 0), c)
        return exp

    def surprise(self, concept, tick):
        ps = [p for p in (self.p_markov(concept), self.p_periodic(concept, tick), self.p_trackers(concept, tick))
              if p is not None]
        if not ps:
            p = 1.0 / (2 + self.count.get(concept, 0)) if self.count.get(concept) else P_FLOOR
        else:
            p = max(ps)
        return -math.log2(max(P_FLOOR, min(1.0, p)))

    # -- learning
    def observe(self, tick, concepts):
        """Score then learn from one tick of percepts. Returns {concept: surprise}, {missing: expected_p}."""
        expected = self.expectations(tick)
        scores = {c: self.surprise(c, tick) for c in concepts}
        missing = {b: p for b, p in expected.items() if b not in concepts}
        self.ticks += 1
        for key, rec in self.trackers.items():
            a, b, lag = key.split("|")
            lag = int(lag)
            if lag <= len(self.recent) and a in self.recent[-lag]:
                rec[0] += 1
                if b in concepts:
                    rec[1] += 1
        for c in concepts:
            f, v = fv(c)
            prev = self.last.get(f)
            if prev is not None:
                self.trans[f][f"{prev}>{v}"] += 1
            self.last[f] = v
            if c in self.last_seen:
                self.intervals[c].append(tick - self.last_seen[c])
            self.last_seen[c] = tick
            self.count[c] += 1
        self.recent.append(set(concepts))
        return scores, missing

    def base_rate(self, concept):
        return (self.count.get(concept, 0) + 0.5) / (self.ticks + 1.0)

    def track(self, a, b, lag, protected=()):
        key = f"{a}|{b}|{lag}"
        if key in self.trackers:
            return False
        if len(self.trackers) >= self.capacity:
            # evict the least useful tracker that is not backing a belief
            cands = [(self.tracker_conf(k) if self.trackers[k][0] >= 8 else 0.5, k)
                     for k in self.trackers if k not in protected]
            if not cands:
                return False
            del self.trackers[min(cands)[1]]
        self.trackers[key] = [0, 0]
        return True

    def periodic(self, concept):
        iv = self.intervals.get(concept)
        if not iv or len(iv) < 6:
            return None
        mean = sum(iv) / len(iv)
        sd = (sum((x - mean) ** 2 for x in iv) / len(iv)) ** 0.5
        return round(mean) if sd <= 0.05 * mean + 0.5 else None

    # -- checkpoint
    def dump(self):
        return dict(trans={f: dict(r) for f, r in self.trans.items()}, last=self.last, count=dict(self.count),
                    last_seen=self.last_seen, intervals={c: list(d) for c, d in self.intervals.items()},
                    trackers=self.trackers, ticks=self.ticks, recent=[sorted(s) for s in self.recent])

    def load(self, d):
        for f, r in d["trans"].items():
            self.trans[f].update(r)
        self.last, self.last_seen, self.trackers, self.ticks = d["last"], d["last_seen"], d["trackers"], d["ticks"]
        self.count.update(d["count"])
        for c, iv in d["intervals"].items():
            self.intervals[c].extend(iv)
        self.recent.extend(set(s) for s in d["recent"])
