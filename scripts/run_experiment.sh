#!/usr/bin/env bash
# Run one experiment spec (JSON) through the controller. See docs/running.md.
set -euo pipefail
cd "$(dirname "$0")/.."
exec python3 -m lab run "$1"
