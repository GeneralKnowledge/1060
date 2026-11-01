#!/usr/bin/env bash
# Return to the last committed best configuration: removes experiment worktrees and
# exp/* branches, and resets winners/ to HEAD. Does not touch drivers or hardware state.
set -euo pipefail
cd "$(dirname "$0")/.."
exec python3 -m lab restore
