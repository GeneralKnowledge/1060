#!/usr/bin/env bash
# Start the continuous mind in tmux session "mind", restarting it if it ever exits
# (a clean stop via `python3 -m mind stop --state ...` ends the loop).
#   scripts/mind/start.sh [STATE_DIR] [LAB_REPO]
set -euo pipefail
CODE="$(cd "$(dirname "$0")/../.." && pwd)"
STATE="${1:-$HOME/mind-state}"
REPO="${2:-$HOME/1060}"
mkdir -p "$STATE"
if tmux has-session -t =mind 2>/dev/null; then echo "tmux session 'mind' already exists"; exit 1; fi
tmux new-session -d -s mind -c "$CODE" "while true; do nice -n 10 python3 -m mind run --state '$STATE' --repo '$REPO' --progress 720 2>&1 | tee -a '$STATE/mind.log'; [ -f '$STATE/STOP' ] && break; echo restarting in 10 s; sleep 10; done"
echo "mind started in tmux session 'mind' (state $STATE); status: cd $CODE && python3 -m mind status --state $STATE"
