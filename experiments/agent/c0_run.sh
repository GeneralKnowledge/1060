#!/usr/bin/env bash
# C0: can a small low-bit model act as a Python coding agent? Runs the agent loop on
# 8 build-with-tests tasks for each model, on the laptop GPU. Quick run: ~30-60 min.
# Usage: experiments/agent/c0_run.sh [extra agent_bench.py args, e.g. --tasks roman lru]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
BIN="${BIN:-$ROOT/backends/builds/d2db58ba90808f32/build/bin}"
MODELS_DIR="${MODELS_DIR:-$HOME/1060/models}"
OUT="${OUT:-$ROOT/experiments/agent/runs/c0-$(date -u +%Y%m%dT%H%MZ)}"

# Don't share the GPU with another benchmark: the timings would be meaningless.
busy=$(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader || true)
if [[ -n "$busy" ]]; then
  echo "GPU busy, not starting:"; echo "$busy"; exit 1
fi
[[ -f "$ROOT/experiments/STOP" ]] && { echo "experiments/STOP exists, not starting"; exit 1; }

mkdir -p "$OUT"
nvidia-smi --query-gpu=timestamp,temperature.gpu,power.draw,memory.used,utilization.gpu \
  --format=csv -l 5 > "$OUT/gpu.csv" &
SMI=$!
trap 'kill $SMI 2>/dev/null || true' EXIT

python3 "$ROOT/experiments/agent/agent_bench.py" --bin "$BIN" --out "$OUT" \
  --models "$MODELS_DIR/Ternary-Bonsai-2-4B-Q2_0.gguf" "$MODELS_DIR/Ternary-Bonsai-2-9B-Q2_0.gguf" \
  "$@" 2>&1 | tee "$OUT/log.txt"
echo "Results: $OUT/summary.json"
