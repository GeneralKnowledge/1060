#!/usr/bin/env bash
# E3: one candidate model through the E1/E2 measurements (speed, VRAM/RAM, offload, checkable tasks).
# Usage: experiments/lowbit/e3_run.sh /path/to/Ternary-Bonsai-2-9B-Q2_0.gguf
# Compared with E2's numbers for Qwen3.5-4B Q4_K_M and Bonsai-2-4B (same eval, greedy, so they are not re-run).
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
BIN="${BIN:-$ROOT/backends/builds/d2db58ba90808f32/build/bin}"
CAND="$1"
OUT="$ROOT/experiments/lowbit/e3/$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$OUT"
RT=(-ngl 99 -fa 1 -ctk f16 -ctv f16 -b 2048 -ub 256)
log() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$OUT/run.log"; }
{
  echo "date_utc: $(date -u +%FT%TZ)"; echo "lab_commit: $(git -C "$ROOT" rev-parse HEAD)"
  "$BIN/llama-cli" --version 2>&1 | head -3
  nvidia-smi --query-gpu=name,driver_version,memory.total,power.limit --format=csv; free -m
  stat -c '%n %s' "$CAND"; sha256sum "$CAND"
} > "$OUT/env.txt" 2>&1
busy=$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader)
if [ -n "$busy" ]; then log "GPU busy, aborting: $busy"; exit 2; fi
nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu,power.draw,temperature.gpu \
  --format=csv,noheader -lms 500 >> "$OUT/gpu.csv" &
SMI=$!
trap 'kill $SMI 2>/dev/null' EXIT
mark() { echo "MARK,$(date -u +%FT%T),$1" >> "$OUT/gpu.csv"; }
tag=$(basename "$CAND" .gguf)

log "smoke"; mark smoke
/usr/bin/time -v "$BIN/llama-completion" -v -m "$CAND" -p "The three most important ideas in computer science are" \
  -n 64 -c 4096 --temp 0 --seed 1 -no-cnv "${RT[@]}" > "$OUT/smoke.txt" 2> "$OUT/smoke.log"
grep -E "offloaded|general.architecture|n_vocab|model size|Maximum resident|out of memory" "$OUT/smoke.log" | tee -a "$OUT/run.log"

log "bench"; mark bench
"$BIN/llama-bench" -m "$CAND" -p 512,2048 -n 128 -r 3 -o json -lm none "${RT[@]}" > "$OUT/bench.json" 2> "$OUT/bench.log"

log "GSM8K-50 + 10 coding tasks"; mark eval
python3 "$ROOT/experiments/lowbit/eval_tasks.py" --bin "$BIN" --out "$OUT" --models "$CAND" > "$OUT/eval_progress.log" 2>&1
grep '^.* {' "$OUT/eval_progress.log" | tee -a "$OUT/run.log"
log "done: $OUT"
