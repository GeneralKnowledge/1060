"""CLI: python3 -m lab <command>

  audit                     M0 read-only hardware audit
  approve-benchmark NOTE    (human) approve current benchmark code/config
  hash FILE                 sha256 of a model/dataset, to pin in config/benchmark.toml
  build [--backend B]       build llama.cpp for the best config (or another backend)
  baseline                  measure + validate the current best config
  run SPEC.json             run one manually written experiment
  agent-step                let the research agent propose and run one experiment
  session [--max-experiments N] [--max-hours H]   unattended research
  sweep FILE.json           run a scripted list of known knobs (no research agent)
  status                    best config, recent results, pending proposals
  restore                   discard stray worktrees, reset winners/ to the last commit
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="lab", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("audit")
    a.add_argument("rest", nargs=argparse.REMAINDER)
    p = sub.add_parser("approve-benchmark")
    p.add_argument("note")
    p = sub.add_parser("hash")
    p.add_argument("file")
    p = sub.add_parser("build")
    p.add_argument("--backend", choices=["cuda", "vulkan", "cpu"])
    sub.add_parser("baseline")
    p = sub.add_parser("run")
    p.add_argument("spec")
    sub.add_parser("agent-step")
    p = sub.add_parser("session")
    p.add_argument("--max-experiments", type=int)
    p.add_argument("--max-hours", type=float)
    p = sub.add_parser("sweep")
    p.add_argument("file")
    sub.add_parser("status")
    sub.add_parser("restore")
    args = ap.parse_args(argv)

    if args.cmd == "audit":
        from lab import hardware_audit
        return hardware_audit.main(args.rest)
    if args.cmd == "approve-benchmark":
        from benchmarks.benchmark import approve
        print(json.dumps(approve(args.note), indent=2))
        return 0
    if args.cmd == "hash":
        from benchmarks.benchmark import sha256_file
        print(sha256_file(Path(args.file)))
        return 0
    if args.cmd == "build":
        from lab.build import build
        from lab.safety import load_limits
        best = json.loads((ROOT / "winners" / "best_config.json").read_text())
        spec = {"backend": args.backend or best["backend"], **best["build"]}
        print(build(spec, load_limits(), log_dir=ROOT / "experiments" / "logs" / "builds"))
        return 0
    if args.cmd == "status":
        return status()
    if args.cmd == "restore":
        return restore()

    from lab.controller import Controller
    c = Controller()
    if args.cmd == "baseline":
        rec = c.validate_baseline()
    elif args.cmd == "run":
        from lab.experiment import ExperimentSpec
        rec = c.run(ExperimentSpec.from_dict(json.loads(Path(args.spec).read_text())))
    elif args.cmd == "agent-step":
        from lab.agent import ResearchAgent
        rec = ResearchAgent(c).step()
    elif args.cmd == "sweep":
        from lab.sweep import load_sweep, run_sweep
        print(json.dumps(run_sweep(c, load_sweep(Path(args.file))), indent=2))
        return 0
    elif args.cmd == "session":
        from lab.agent import ResearchAgent, run_session
        print(json.dumps(run_session(ResearchAgent(c), max_experiments=args.max_experiments,
                                     max_hours=args.max_hours), indent=2))
        return 0
    print(json.dumps({k: rec.get(k) for k in ("id", "decision", "error", "evaluation")},
                     indent=2, default=str))
    return 0


def status() -> int:
    best = json.loads((ROOT / "winners" / "best_config.json").read_text())
    print("best config:", best["id"], "(validated)" if best.get("validated") else "(NOT validated)")
    print(json.dumps({k: best.get(k) for k in ("workload", "backend", "build", "runtime", "env",
                                               "metrics")}, indent=2))
    tsv = ROOT / "experiments" / "results.tsv"
    if tsv.exists():
        print("\nrecent results:")
        print("".join(tsv.read_text().splitlines(keepends=True)[-10:]))
    props = sorted((ROOT / "experiments" / "proposals").glob("*.json"))
    if props:
        print("proposals awaiting human approval:", ", ".join(p.stem for p in props))
    return 0


def restore() -> int:
    def git(*a):
        return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True)
    wt = ROOT / "experiments" / "worktrees"
    for d in sorted(wt.glob("*")) if wt.exists() else []:
        git("worktree", "remove", "--force", str(d))
    git("worktree", "prune")
    for b in git("branch", "--list", "exp/*", "--format=%(refname:short)").stdout.split():
        git("branch", "-D", b)
    git("checkout", "--", "winners")
    print("restored: worktrees removed, exp/* branches deleted, winners/ reset to HEAD")
    return 0


if __name__ == "__main__":
    sys.exit(main())
