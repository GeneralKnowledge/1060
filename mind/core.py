"""The mind: one long-lived process around a persistent shared state.

Every tick, in every state: perceive -> predict (surprise) -> associate ->
update interests -> attend -> update drives -> choose state. Then the state
decides which other processes get time: thinking (awake, background),
replay (resting), recombination (dreaming), consolidation (deep sleep).

In milestone A1 every "think" step is a cheap algorithm (replaying memory to
test a candidate regularity). A2 lets the same steps recruit language tiers.
Nothing here says what to think about: thoughts start only on what wins the
workspace, and what wins depends on surprise, novelty and the mind's own
interests (which follow its own learning progress).
"""
from __future__ import annotations

import datetime as dt
import math
import random
import time
import uuid
from pathlib import Path

from mind import language
from mind.params import Params
from mind.predict import MAX_LAG, Predictor, fv
from mind.store import Store

STATES = ("awake", "background", "resting", "dreaming", "deep_sleep")
SLEEP = ("dreaming", "deep_sleep")
WS_CAP = 7

DEFAULTS = dict(
    tracker_capacity=300,        # regularities it can follow at once (attention is scarce)
    max_steps_per_window=400,    # hard compute cap per parameter window (not self-tunable)
    param_window=300,            # ticks between self-tuning decisions
    checkpoint_every=50,
    forget_age=3000,             # ticks before unsurprising percepts may fade
    dream_control=0.5,           # share of dreams paired with a random-pair control (for evaluation only)
    llm_dream_every=15,          # every Nth dream tick asks the reflective tier for proposals (if present)
    reflect_every=40,            # every Nth resting tick asks the reflective tier to reflect (if present)
    tier_wait_s=600,             # fast mode only: wait this long for a tier answer (keeps runs reproducible)
    max_tier_calls_per_window=8,  # hard cap per kind (dream, reflect) per param window (not self-tunable)
    min_forward=8,               # forward observations before a hypothesis can become a belief
)


class Mind:
    def __init__(self, state_dir: Path, senses: list, cfg: dict | None = None, tiers=None):
        self.cfg = {**DEFAULTS, **(cfg or {})}
        self.state_dir = state_dir
        self.store = Store(state_dir / "mind.sqlite")
        self.senses = senses
        self.tiers = tiers                                # mind.tiers.TierPool or None (algorithms only)
        self.fast_mode = False
        self.extra_percepts: list[tuple[str, str]] = []   # percepts produced by tiers (e.g. read text)
        self._identity()
        self.pred = Predictor(self.cfg["tracker_capacity"])
        if (p := self.store.get("predictor")):
            self.pred.load(p)
        self.params = Params(self.store.get("params"))
        ck = self.store.get("checkpoint") or {}
        self.tick = ck.get("tick", 0)
        self.state = ck.get("state", "background")
        self.arousal = ck.get("arousal", 0.0)
        self.fatigue = ck.get("fatigue", 0.0)
        self.quiet = ck.get("quiet", 0)
        self.sleep_phase = ck.get("sleep_phase", 0)
        self.workspace = ck.get("workspace", {})          # focus -> activation
        self.interests = ck.get("interests", {})          # concept -> [strength, fast_ema, slow_ema]
        self.edges = ck.get("edges", {})                  # "a\tb" (sorted) -> weight
        self.win = ck.get("win", dict(s=0.0, n=0, steps=0))
        self.curiosity = ck.get("curiosity", 0.0)
        self.rng = random.Random(self.identity["seed"] * 7919 + self.boots)
        self.steps_used = 0
        self.ws_seen = ck.get("ws_seen", {})               # focus -> tick it was noticed
        self.store.set_thought_status("active", "suspended")   # thoughts interrupted by the restart
        self.store.commit()

    # ------------------------------------------------------------------ identity
    def _identity(self):
        ident = self.store.get("identity")
        now = time.time()
        if not ident:
            ident = dict(id=str(uuid.uuid4()), born=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                         seed=self.cfg.get("birth_seed") or random.SystemRandom().randrange(1, 2 ** 31))
            self.store.put("identity", ident)
            self.store.log(0, "birth", "background", ident)
        self.identity = ident
        self.boots = self.store.get("boots", 0) + 1
        self.store.put("boots", self.boots)
        last = self.store.get("last_alive")
        self.store.log(self.store.get("checkpoint", {}).get("tick", 0), "boot", None,
                       dict(boot=self.boots, offline_s=round(now - last, 1) if last else None))

    # ------------------------------------------------------------------ main loop
    def step(self):
        self.tick += 1
        t = self.tick
        percepts = [(s.name, c) for s in self.senses for c in s.poll(t)] + self.extra_percepts
        self.extra_percepts = []
        if self.tiers:
            language.interpret(self, t, [c for src, c in percepts if src == "inbox" and c.startswith("m0.w=")])
        concepts = sorted({c for _, c in percepts})
        scores, missing = self.pred.observe(t, concepts)
        for src, c in percepts:
            self.store.add_percept(t, src, c, round(scores[c], 3))
            if src == "world":
                self.win["s"] += scores[c]
                self.win["n"] += 1
        self._associate(concepts)
        self._update_interests(scores)
        entering = self._attend(t, scores, missing)
        self._drives(scores, entering)
        self._choose_state(t)
        self._run_processes(t)
        if self.tiers:
            language.collect(self, t)
        if t % self.cfg["param_window"] == 0:
            self._self_tune(t)
        if t % self.cfg["checkpoint_every"] == 0:
            self.checkpoint()

    def run(self, ticks=None, tick_s=5.0, fast=False, stop_file: Path | None = None, on_tick=None):
        n = 0
        self.fast_mode = fast
        try:
            while ticks is None or n < ticks:
                t0 = time.time()
                self.step()
                n += 1
                if not fast:
                    self.store.commit()
                elif n % 100 == 0:
                    self.store.commit()
                if on_tick:
                    on_tick(self)
                if stop_file and stop_file.exists():
                    self.store.log(self.tick, "stop_requested", self.state)
                    break
                if not fast:
                    time.sleep(max(0.0, tick_s - (time.time() - t0)))
        finally:
            self.checkpoint()
            self.store.log(self.tick, "shutdown", self.state)
            self.store.commit()

    def checkpoint(self):
        self.store.put("checkpoint", dict(
            tick=self.tick, state=self.state, arousal=self.arousal, fatigue=self.fatigue, quiet=self.quiet,
            sleep_phase=self.sleep_phase, workspace=self.workspace, interests=self.interests, edges=self.edges,
            win=self.win, curiosity=self.curiosity, ws_seen={k: self.ws_seen[k] for k in self.workspace if k in self.ws_seen}))
        self.store.put("predictor", self.pred.dump())
        self.store.put("params", self.params.dump())
        self.store.put("last_alive", time.time())
        for s in self.senses:
            if hasattr(s, "save"):
                s.save()
        self.store.commit()

    # ------------------------------------------------------------------ always-on processes
    def _associate(self, concepts):
        prev = self.pred.recent[-2] if len(self.pred.recent) > 1 else set()
        pool = sorted(set(concepts) | prev)[:24]
        for b in concepts:
            for a in pool:
                if a == b:
                    continue
                k = "\t".join(sorted((a, b)))
                w = self.edges.get(k, 0.0)
                self.edges[k] = w + 0.1 * (1 - w)

    def _update_interests(self, scores):
        r = self.params["interest_rate"]
        for c, s in scores.items():
            rec = self.interests.get(c)
            if rec is None:
                rec = self.interests[c] = [0.0, s, s]
            rec[1] += 0.3 * (s - rec[1])        # recent surprise
            rec[2] += 0.03 * (s - rec[2])       # long-run surprise
            progress = max(0.0, rec[2] - rec[1])   # learning progress: surprise is falling
            rec[0] += r * (progress - rec[0])

    def _interest(self, c):
        rec = self.interests.get(c)
        return min(1.0, rec[0]) if rec else 0.0

    def _attend(self, t, scores, missing):
        d = self.params["ws_decay"]
        self.workspace = {k: v * d for k, v in self.workspace.items() if v * d > 0.05}
        entering = 0.0
        cands = {}
        for c, s in scores.items():
            novelty = 1.0 / (1 + self.pred.count.get(c, 0))
            cands[c] = 0.55 * min(1.0, s / 6.0) + 0.3 * self._interest(c) + 0.15 * novelty
        for b, p in missing.items():
            cands["missing:" + b] = 0.6 * p
        for k, a in cands.items():
            if a > self.workspace.get(k, 0.0):
                if k not in self.workspace and a >= self.params["think_threshold"]:
                    self.store.log(t, "attend", self.state, dict(focus=k, activation=round(a, 3)))
                entering = max(entering, a - self.workspace.get(k, 0.0))
                self.workspace[k] = a
                self.ws_seen[k] = t
        if len(self.workspace) > WS_CAP:
            self.workspace = dict(sorted(self.workspace.items(), key=lambda kv: -kv[1])[:WS_CAP])
        return entering

    def _drives(self, scores, entering):
        top = max(scores.values(), default=0.0)
        self.quiet = 0 if top > 3.0 else self.quiet + 1
        self.arousal = 0.8 * self.arousal + 0.2 * min(1.0, entering * 1.6)
        mean_s = sum(scores.values()) / len(scores) if scores else 0.0
        self.curiosity += 0.05 * (mean_s - self.curiosity)
        # sleep pressure builds while awake in any form (resting only slows it); only sleep clears it
        delta = dict(awake=0.004, background=0.002, resting=0.0007, dreaming=-0.004, deep_sleep=-0.008)
        self.fatigue = min(1.0, max(0.0, self.fatigue + delta[self.state]))

    def _choose_state(self, t):
        p, old = self.params, self.state
        sleeping = old in SLEEP
        if self.arousal > p["wake_threshold"] + (0.15 if sleeping else 0.0):
            new = "awake"
        elif sleeping:
            if self.fatigue < 0.08:
                new = "background"
            else:
                self.sleep_phase += 1
                new = "deep_sleep" if (self.sleep_phase % 60) < 40 else "dreaming"
        elif self.fatigue > p["sleep_fatigue"] and self.quiet >= 10:
            new, self.sleep_phase = "deep_sleep", 0
        elif old == "awake" and self.arousal > p["wake_threshold"] * 0.6:
            new = "awake"
        elif self.quiet >= 15:
            new = "resting"
        else:
            new = "background"
        if new != old:
            self.state = new
            self.store.log(t, "state", new, dict(old=old, arousal=round(self.arousal, 3),
                                                 fatigue=round(self.fatigue, 3)))
            if new in SLEEP and old not in SLEEP:
                self.store.set_thought_status("active", "suspended")
            elif old in SLEEP and new not in SLEEP:
                self.store.set_thought_status("suspended", "active")

    # ------------------------------------------------------------------ state-dependent processes
    def _budget(self, n=1):
        if self.win["steps"] + n > self.cfg["max_steps_per_window"]:
            return False
        self.win["steps"] += n
        return True

    def _run_processes(self, t):
        s = self.state
        if s == "awake":
            self._start_thoughts(t, extra=0.0, max_active=3)
            for _ in range(3):
                self._think_step(t)
        elif s == "background":
            self._start_thoughts(t, extra=0.2, max_active=1)
            if t % 3 == 0:
                self._think_step(t)
        elif s == "resting":
            if t % 2 == 0:
                self._replay(t)
            if t % 4 == 0:
                self._think_step(t)
        elif s == "dreaming":
            self._dream(t)
        elif s == "deep_sleep" and t % 10 == 0:
            self._consolidate(t)

    # thinking ---------------------------------------------------------------
    def _start_thoughts(self, t, extra, max_active):
        active = self.store.thoughts(["active"])
        busy = {th["focus"] for th in active}
        for focus, a in sorted(self.workspace.items(), key=lambda kv: -kv[1]):
            if len(active) >= max_active:
                break
            if a < self.params["think_threshold"] + extra or focus in busy:
                continue
            seen = self.ws_seen.get(focus, t)
            tid = self.store.new_thought(t, focus, dict(seen=seen, cands=[], passed=[], i=0))
            self.store.log(t, "thought_start", self.state, dict(thought=tid, focus=focus, activation=round(a, 3)))
            active.append(dict(id=tid, focus=focus))
            busy.add(focus)

    def _think_step(self, t):
        active = self.store.thoughts(["active"])
        if not active or not self._budget():
            return
        th = active[0]
        d, focus, step = th["data"], th["focus"], th["step"]
        missing = focus.startswith("missing:")
        b = focus[8:] if missing else focus
        if step == 0:
            # gather: what came shortly before the moment it was noticed?
            cands = {}
            for lag in range(1, MAX_LAG + 1):
                for a in self.store.percepts_at(d["seen"] - lag):
                    if a == b:
                        continue
                    k = "\t".join(sorted((a, b)))
                    score = self.edges.get(k, 0.0) + self._interest(a) + 0.01 * self.rng.random()
                    if (a, lag) not in cands or cands[(a, lag)] < score:
                        cands[(a, lag)] = score
            d["cands"] = [[a, lag] for (a, lag), _ in sorted(cands.items(), key=lambda kv: -kv[1])[:10]]
            status = "active" if d["cands"] else "resolved"
            if not d["cands"]:
                self.store.log(t, "thought_end", self.state, dict(thought=th["id"], focus=focus, outcome="nothing before it"))
        else:
            # test two candidates per step against memory
            for a, lag in d["cands"][d["i"]:d["i"] + 2]:
                res = self._replay_test(a, b, lag)
                if res["pass"]:
                    d["passed"].append([a, lag, res["conf"], res["lift"]])
            d["i"] += 2
            status = "active"
            if d["i"] >= len(d["cands"]):
                status = "resolved"
                for a, lag, conf, lift in d["passed"]:
                    self._hypothesis(t, a, b, lag, conf, "thought")
                per = self.pred.periodic(b)
                if per:
                    self._belief_cycle(t, b, per, "thought")
                outcome = (f"{len(d['passed'])} candidate regularities" if d["passed"]
                           else ("periodic" if per else "unexplained"))
                self.store.log(t, "thought_end", self.state, dict(thought=th["id"], focus=focus, outcome=outcome,
                                                                  passed=d["passed"][:5]))
                if missing:
                    self._update_beliefs(t)
        self.store.save_thought(th["id"], t, status, step + 1, d)

    def _replay_test(self, a, b, lag):
        lo, hi, n_ticks = self.store.memory_span()
        n = self.store.occurrences(a)
        if n < 4 or n_ticks == 0:
            return {"pass": False, "conf": 0.0, "lift": 0.0, "n": n}
        hits = self.store.co_occurrences(a, b, lag)
        conf = (hits + 0.5) / (n + 1.0)
        base = (self.store.occurrences(b) + 0.5) / max(1.0, hi - lo + 1.0)
        lift = conf / base
        return {"pass": lift >= self.params["lift_min"] and conf >= 0.3, "conf": round(conf, 3),
                "lift": round(lift, 2), "n": n}

    def _hypothesis(self, t, a, b, lag, conf, origin):
        old = self.store.belief("cause", a, b, lag)
        if old is None:
            self.store.upsert_belief(t, "cause", a, b, lag, conf, 0, origin, "hypothesis")
            self.store.log(t, "hypothesis", self.state, dict(a=a, b=b, lag=lag, conf=conf, origin=origin))
        self.pred.track(a, b, lag, protected=self._protected())

    def _protected(self):
        return {f"{x['a']}|{x['b']}|{x['lag']}" for x in self.store.beliefs("belief")}

    def _belief_cycle(self, t, c, period, origin):
        old = self.store.belief("cycle", c, c, period)
        self.store.upsert_belief(t, "cycle", c, c, period, 0.9, len(self.pred.intervals.get(c, [])), origin, "belief")
        if old is None:
            self.store.log(t, "belief_new", self.state, dict(kind="cycle", concept=c, period=period, origin=origin))

    def _update_beliefs(self, t):
        """Promote or doubt hypotheses using forward (out-of-sample) tracker evidence."""
        p = self.params
        for key, (n, hits) in list(self.pred.trackers.items()):
            a, b, lag = key.split("|")
            lag = int(lag)
            old = self.store.belief("cause", a, b, lag)
            if old is None or n < self.cfg["min_forward"]:
                continue
            conf = (hits + 0.5) / (n + 1.0)
            lift = conf / self.pred.base_rate(b)
            if lift >= p["lift_min"] and conf >= 0.5:
                status = "belief"
            elif old["status"] == "belief" and (conf < 0.3 or lift < p["lift_min"] / 2):
                status = "doubted"
            elif old["status"] == "hypothesis" and n >= 3 * self.cfg["min_forward"] and conf < 0.2:
                status = "rejected"
            else:
                status = old["status"]
            self.store.upsert_belief(t, "cause", a, b, lag, round(conf, 3), n, old["origin"], status)
            if status != old["status"]:
                self.store.log(t, "belief_" + status, self.state,
                               dict(a=a, b=b, lag=lag, conf=round(conf, 3), lift=round(lift, 2), n=n,
                                    origin=old["origin"]))
            if status == "rejected":
                self.pred.trackers.pop(key, None)

    # resting ---------------------------------------------------------------
    def _replay(self, t):
        if not self._budget():
            return
        self.rest_ticks = getattr(self, "rest_ticks", 0) + 1
        if self.tiers and self.rest_ticks % self.cfg["reflect_every"] == 0:
            language.reflect(self, t)
        r = self.store.random_percept(self.rng)
        if r:
            c, tick = r
            # re-experiencing a memory: it competes for the workspace again
            a = 0.5 * self.workspace.get(c, 0.0) + 0.4 * self._interest(c) + 0.2 * self.rng.random()
            if a > self.workspace.get(c, 0.0):
                self.workspace[c] = a
                self.ws_seen[c] = tick
        self._update_beliefs(t)

    # dreaming --------------------------------------------------------------
    def _walk(self, start, steps):
        cur = start
        for _ in range(steps):
            nbrs = [(k, w) for k, w in self.edges.items() if cur in k.split("\t")]
            if not nbrs:
                break
            # prefer weak links: far associations, not the obvious ones
            weights = [1.0 - w + 0.05 for _, w in nbrs]
            k = self.rng.choices([k for k, _ in nbrs], weights=weights)[0]
            a, b = k.split("\t")
            cur = b if a == cur else a
        return cur

    def _best_lag(self, a, b):
        best = None
        for lag in range(1, MAX_LAG + 1):
            r = self._replay_test(a, b, lag)
            if best is None or r["lift"] > best[1]["lift"]:
                best = (lag, r)
        return best

    def _dream(self, t):
        if not self._budget(2):
            return
        self.dream_ticks = getattr(self, "dream_ticks", 0) + 1
        if self.tiers and self.dream_ticks % self.cfg["llm_dream_every"] == 0:
            language.dream(self, t)
        r = self.store.random_percept(self.rng)
        if not r:
            return
        a = r[0]
        b = self._walk(a, int(self.params["dream_walk"]))
        if b == a or fv(a)[0] == fv(b)[0]:
            return
        lag, res = self._best_lag(a, b)
        self.store.log(t, "dream", self.state, dict(a=a, b=b, lag=lag, lift=res["lift"], passed=res["pass"]))
        if res["pass"]:
            self._hypothesis(t, a, b, lag, res["conf"], "dream")
        if self.rng.random() < self.cfg["dream_control"]:
            ra, rb = self.store.random_percept(self.rng), self.store.random_percept(self.rng)
            if ra and rb and fv(ra[0])[0] != fv(rb[0])[0]:
                clag, cres = self._best_lag(ra[0], rb[0])
                self.store.log(t, "dream_control", self.state,
                               dict(a=ra[0], b=rb[0], lag=clag, lift=cres["lift"], passed=cres["pass"]))

    # deep sleep ------------------------------------------------------------
    def _consolidate(self, t):
        if not self._budget(3):
            return
        self.edges = {k: w * 0.97 for k, w in self.edges.items() if w * 0.97 > 0.02}
        for rec in self.interests.values():
            rec[0] *= 0.98
        forgotten = self.store.forget(t - self.cfg["forget_age"], self.params["forget_surprise"])
        self._update_beliefs(t)
        for c in list(self.pred.count):
            per = self.pred.periodic(c)
            if per and self.store.belief("cycle", c, c, per) is None:
                self._belief_cycle(t, c, per, "sleep")
        self.store.log(t, "consolidate", self.state, dict(forgotten=forgotten, edges=len(self.edges),
                                                          trackers=len(self.pred.trackers)))

    # self-tuning -----------------------------------------------------------
    def _self_tune(self, t):
        w = self.win
        if w["n"] == 0:
            reward = None
        else:
            reward = -(w["s"] / w["n"]) - 0.001 * w["steps"]
        if reward is not None:
            ev = self.params.evaluate(reward, self.rng)
            ev["mean_surprise"] = round(w["s"] / w["n"], 4)
            ev["steps"] = w["steps"]
            self.store.log(t, "self_tune", self.state, ev)
        self.win = dict(s=0.0, n=0, steps=0)

    # ------------------------------------------------------------------ introspection
    def snapshot(self):
        return dict(id=self.identity["id"], born=self.identity["born"], boots=self.boots, tick=self.tick,
                    state=self.state, arousal=round(self.arousal, 3), fatigue=round(self.fatigue, 3),
                    curiosity=round(self.curiosity, 3),
                    workspace={k: round(v, 3) for k, v in sorted(self.workspace.items(), key=lambda kv: -kv[1])},
                    trackers=len(self.pred.trackers), params=self.params.values,
                    top_interests=sorted(((round(r[0], 4), c) for c, r in self.interests.items()), reverse=True)[:8])
