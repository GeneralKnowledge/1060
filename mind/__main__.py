"""python3 -m mind run | status | report | stop

  run            live in real time (tick_s from config/mind.toml), with real signals; runs until STOP
  run --fast N   live N ticks as fast as possible on the generated world only (quick look)
  status         identity, state, workspace, recent life stream
  report         measures against the world's hidden rules; writes mind/reports/<date>-<tick>.md
  stop           ask a running mind to shut down cleanly (touches mind/STOP)

--state DIR picks a different mind (default mind/state). A mind is never recreated:
deleting its state directory is the only way to end one, and no code here does that.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_cfg():
    p = ROOT / "config" / "mind.toml"
    return tomllib.loads(p.read_text()) if p.exists() else {}


def main(argv=None):
    ap = argparse.ArgumentParser(prog="mind")
    ap.add_argument("cmd", choices=["run", "status", "report", "stop"])
    ap.add_argument("--state", type=Path, default=ROOT / "mind" / "state")
    ap.add_argument("--fast", type=int, metavar="TICKS", help="run this many ticks without waiting")
    ap.add_argument("--progress", type=int, default=0, help="print a line every N ticks")
    a = ap.parse_args(argv)
    cfg = load_cfg()
    stop = ROOT / "mind" / "STOP"

    if a.cmd == "stop":
        stop.touch()
        print("STOP requested")
        return 0

    if a.cmd == "report":
        from mind.eval import evaluate, report_md
        r = evaluate(a.state)
        md = report_md(r)
        out = ROOT / "mind" / "reports"
        out.mkdir(parents=True, exist_ok=True)
        f = out / f"{dt.date.today().isoformat()}-{r['ticks']}.md"
        f.write_text(md)
        (f.with_suffix(".json")).write_text(json.dumps(r, indent=1))
        print(md)
        print(f"written {f}")
        return 0

    from mind import senses as S
    from mind.core import Mind
    from mind.store import Store

    if a.cmd == "status":
        if not (a.state / "mind.sqlite").exists():
            print("no mind has been born in", a.state)
            return 1
        st = Store(a.state / "mind.sqlite")
        ident, ck = st.get("identity"), st.get("checkpoint") or {}
        print(f"mind {ident['id']}  born {ident['born']}  boots {st.get('boots')}  tick {ck.get('tick')}")
        print(f"state {ck.get('state')}  arousal {ck.get('arousal', 0):.2f}  fatigue {ck.get('fatigue', 0):.2f}")
        print("workspace:", json.dumps({k: round(v, 2) for k, v in (ck.get("workspace") or {}).items()}))
        for e in reversed(st.stream(limit=15)):
            print(f"  {e['tick']:>7} {e['state'] or '':<11} {e['kind']:<15} {json.dumps(e['data'])[:110]}")
        st.close()
        return 0

    # run
    first = not (a.state / "mind.sqlite").exists()
    real = a.fast is None
    senses = []
    mind = Mind(a.state, senses, cfg.get("mind", {}))
    senses += S.build(a.state, mind.identity["seed"], real=real, repo=ROOT)
    if first:
        print(f"born: {mind.identity['id']}")
    if stop.exists():
        stop.unlink()

    def progress(m):
        if a.progress and m.tick % a.progress == 0:
            print(f"tick {m.tick:>6} {m.state:<11} arousal {m.arousal:.2f} fatigue {m.fatigue:.2f} "
                  f"trackers {len(m.pred.trackers):>3} ws {len(m.workspace)}", flush=True)

    t0 = time.time()
    mind.run(ticks=a.fast, tick_s=float(cfg.get("mind", {}).get("tick_s", 5.0)), fast=not real,
             stop_file=stop, on_tick=progress)
    print(f"stopped at tick {mind.tick} after {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
