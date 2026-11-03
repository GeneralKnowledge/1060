"""The mind's own settings, adjusted by the mind itself.

Every parameter starts at a neutral value inside hard bounds. Once per
evaluation window the mind compares how well it predicted its world (mean
surprise of percepts) against the compute it spent, keeps or reverts the last
change, and tries a new one: a (1+1) evolution strategy. Hard limits (budgets,
capacity, safety) live in config and are not parameters.
"""
from __future__ import annotations

PARAMS = {
    # name: (start, low, high, step)
    "think_threshold": (0.5, 0.15, 0.95, 0.08),     # workspace activation needed to start a thought
    "wake_threshold": (0.55, 0.25, 0.95, 0.08),     # arousal that wakes it from any state
    "sleep_fatigue": (0.7, 0.35, 0.95, 0.08),       # fatigue at which it falls asleep when quiet
    "ws_decay": (0.85, 0.6, 0.98, 0.04),            # workspace activation kept per tick
    "interest_rate": (0.05, 0.005, 0.3, 0.03),      # how fast interests follow learning progress
    "dream_walk": (3, 1, 8, 1),                     # graph steps between recombined memories
    "lift_min": (3.0, 1.5, 8.0, 0.5),               # how much better than chance a regularity must be
    "forget_surprise": (1.0, 0.2, 3.0, 0.3),        # percepts less surprising than this fade in deep sleep
}


class Params:
    def __init__(self, saved: dict | None = None):
        saved = saved or {}
        self.values = {k: saved.get("values", {}).get(k, v[0]) for k, v in PARAMS.items()}
        self.trial = saved.get("trial")            # {"name", "old", "baseline"} while a change is on trial
        self.history = saved.get("history", [])

    def __getitem__(self, k):
        return self.values[k]

    def dump(self):
        return dict(values=self.values, trial=self.trial, history=self.history[-200:])

    def evaluate(self, reward: float, rng) -> dict:
        """Called once per window with that window's reward. Returns what happened, for the life stream."""
        event = {}
        if self.trial:
            kept = reward >= self.trial["baseline"]
            if not kept:
                self.values[self.trial["name"]] = self.trial["old"]
            event = dict(param=self.trial["name"], old=self.trial["old"], new=self.trial["new"],
                         reward=round(reward, 4), baseline=round(self.trial["baseline"], 4), kept=kept)
            self.history.append(event)
            baseline = reward if kept else self.trial["baseline"]
        else:
            baseline = reward
        name = rng.choice(sorted(PARAMS))
        _, lo, hi, step = PARAMS[name]
        old = self.values[name]
        new = min(hi, max(lo, old + rng.choice([-step, step])))
        if isinstance(PARAMS[name][0], int):
            new = int(round(new))
        self.values[name] = new
        self.trial = dict(name=name, old=old, new=new, baseline=baseline)
        event["trying"] = dict(param=name, old=old, new=new)
        return event
