#!/usr/bin/env bash
# M0 hardware audit. Read-only: inspects the machine and writes
# research/hardware_profile.json plus raw evidence under research/audit_raw/.
# Usage: scripts/hardware_audit.sh [--sudo] [--idle-seconds N]
set -euo pipefail
cd "$(dirname "$0")/.."
exec python3 lab/hardware_audit.py "$@"
