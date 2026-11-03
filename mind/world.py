"""A generated world for the mind to live in.

Everything about it comes from a random seed: made-up symbols for entities,
properties and values, and rules of a few structural kinds (cycles, delayed
causes, drift, rare events, a rule that changes partway, unreliable reports).
Nobody authored what it is "about".

The rule list is written to rules.json for evaluation only. The mind never
reads it: the mind only receives percepts (feature, value) from step().
"""
from __future__ import annotations

import json
import random
from pathlib import Path

CONS = "bdfgklmnprstvz"
VOWS = "aeiou"


def _word(rng, syll):
    return "".join(rng.choice(CONS) + rng.choice(VOWS) for _ in range(syll))


class World:
    def __init__(self, seed: int, n_entities: int = 6, day: int = 240):
        self.seed = seed
        rng = random.Random(seed)
        self.rng = random.Random(seed + 1)
        used = set()

        def fresh(syll):
            while True:
                w = _word(rng, syll)
                if w not in used:
                    used.add(w)
                    return w

        self.day = day                       # ticks per world day; observation is sparse at night
        self.features: dict[str, list[str]] = {}
        for _ in range(n_entities):
            ent = fresh(2)
            for _ in range(rng.randint(1, 3)):
                self.features[f"{ent}.{fresh(2)}"] = [fresh(2) for _ in range(rng.randint(2, 4))]
        feats = list(self.features)
        rng.shuffle(feats)
        self.rules: list[dict] = []
        driven: set[str] = set()

        def take():
            for f in feats:
                if f not in driven:
                    driven.add(f)
                    return f
            return None

        # drivers: features that change on their own
        drivers = []
        for kind in ["cycle", "cycle", "drift", "rare", "cycle"]:
            f = take()
            if f is None:
                break
            vals = self.features[f]
            if kind == "cycle":
                durs = [rng.randint(3, 30) for _ in vals]
                self.rules.append(dict(kind="cycle", feature=f, values=vals, durations=durs, period=sum(durs)))
            elif kind == "drift":
                self.rules.append(dict(kind="drift", feature=f, values=vals, p=round(rng.uniform(0.03, 0.1), 3)))
            else:
                self.rules.append(dict(kind="rare", feature=f, value=vals[-1], default=vals[0],
                                       p=round(rng.uniform(0.004, 0.012), 4)))
            drivers.append(f)
        # causes: a driver's value triggers an effect on another feature after a delay
        sources = list(drivers)
        for i in range(rng.randint(3, 5)):
            f = take()
            if f is None:
                break
            src = rng.choice(sources)
            r = dict(kind="cause", trigger=[src, rng.choice(self.features[src])], effect=[f, self.features[f][-1]],
                     default=self.features[f][0], delay=rng.randint(1, 8), hold=rng.randint(2, 6),
                     p=round(rng.uniform(0.8, 1.0), 2))
            if i == 0:   # one rule changes partway through the world's life
                r["switch_tick"] = rng.randint(3 * day, 6 * day)
                r["delay_after"] = rng.choice([d for d in range(1, 11) if abs(d - r["delay"]) >= 2])
            self.rules.append(r)
            sources.append(f)
        # unreliable reports: occasionally a feature is reported with a wrong value
        noisy = rng.choice(feats)
        self.rules.append(dict(kind="noise", feature=noisy, p=0.01))
        self.state = {f: v[0] for f, v in self.features.items()}
        self.pending: list[tuple[int, str, str]] = []   # (tick, feature, value)
        self.tick = 0

    # -- persistence (the world continues across restarts of the mind)
    def save(self, d: Path):
        d.mkdir(parents=True, exist_ok=True)
        (d / "rules.json").write_text(json.dumps(dict(seed=self.seed, day=self.day, features=self.features,
                                                      rules=self.rules), indent=1))
        (d / "state.json").write_text(json.dumps(dict(tick=self.tick, state=self.state, pending=self.pending,
                                                      rng=_rng_state(self.rng))))

    @classmethod
    def load_or_create(cls, d: Path, seed: int, **kw):
        w = cls(seed, **kw)
        st = d / "state.json"
        if st.exists():
            s = json.loads(st.read_text())
            w.tick, w.state = s["tick"], s["state"]
            w.pending = [tuple(p) for p in s["pending"]]
            w.rng.setstate(_rng_from(s["rng"]))
        else:
            w.save(d)
        return w

    def night(self, tick):
        return (tick % self.day) >= int(self.day * 0.65)

    def step(self) -> list[tuple[str, str]]:
        """Advance one tick. Returns the percepts the mind receives this tick."""
        self.tick += 1
        t, rng = self.tick, self.rng
        changed: dict[str, str] = {}

        def setv(f, v):
            if self.state[f] != v:
                self.state[f] = v
                changed[f] = v

        for r in self.rules:
            k = r["kind"]
            if k == "cycle":
                pos, acc = t % r["period"], 0
                for v, dur in zip(r["values"], r["durations"]):
                    acc += dur
                    if pos < acc:
                        setv(r["feature"], v)
                        break
            elif k == "drift" and rng.random() < r["p"]:
                vals = r["values"]
                i = vals.index(self.state[r["feature"]]) + rng.choice([-1, 1])
                setv(r["feature"], vals[max(0, min(len(vals) - 1, i))])
            elif k == "rare":
                setv(r["feature"], r["value"] if rng.random() < r["p"] else r["default"])
        # delayed effects due now
        due = [p for p in self.pending if p[0] <= t]
        self.pending = [p for p in self.pending if p[0] > t]
        for _, f, v in due:
            setv(f, v)
        # triggers fire on the tick their condition becomes true
        for r in self.rules:
            if r["kind"] != "cause":
                continue
            src, val = r["trigger"]
            if changed.get(src) == val and rng.random() < r["p"]:
                delay = r["delay_after"] if t >= r.get("switch_tick", 1 << 60) else r["delay"]
                f, v = r["effect"]
                self.pending.append((t + delay, f, v))
                self.pending.append((t + delay + r["hold"], f, r["default"]))
        # observation: sparse at night, plus occasional wrong reports
        obs_p = 0.35 if self.night(t) else 0.97
        out = [(f, v) for f, v in changed.items() if rng.random() < obs_p]
        for r in self.rules:
            if r["kind"] == "noise" and rng.random() < r["p"]:
                vals = [v for v in self.features[r["feature"]] if v != self.state[r["feature"]]]
                if vals:
                    out.append((r["feature"], rng.choice(vals)))
        return out


def _rng_state(rng):
    a, b, c = rng.getstate()
    return [a, list(b), c]


def _rng_from(s):
    return (s[0], tuple(s[1]), s[2])
