#!/usr/bin/env bash
# E2: trained ternary (Ternary-Bonsai-2-4B Q2_0) vs ordinary 2-bit PTQ of the same base model at the same size.
# Usage: experiments/lowbit/e2_run.sh /path/to/Ternary-Bonsai-2-4B-Q2_0.gguf
# Makes Qwen3.5-4B IQ2_XXS (~Bonsai's size) and IQ2_M from the Q8_0 with an importance matrix,
# then measures speed, PPL/KLD and checkable task accuracy (GSM8K first 50 + 10 coding tasks) for all four.
# Read-only for the lab: does not touch config/, winners/ or the benchmark lock.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
BIN="${BIN:-$ROOT/backends/builds/d2db58ba90808f32/build/bin}"
MODELS="${MODELS:-$(dirname "$1")}"
BONSAI="$1"
BASE="${BASE:-$MODELS/Qwen3.5-4B-Q4_K_M.gguf}"
Q8="${Q8:-$MODELS/Qwen3.5-4B-Q8_0.gguf}"
REF="${REF:-$MODELS/qwen3.5-4b-q8_0.kld}"
DATA="${DATA:-$ROOT/benchmarks/workloads/wikitext-2-test-slice.txt}"
OUT="$ROOT/experiments/lowbit/e2/$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$OUT"
RT=(-ngl 99 -fa 1 -ctk f16 -ctv f16 -b 2048 -ub 256)
log() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$OUT/run.log"; }

{
  echo "date_utc: $(date -u +%FT%TZ)"; echo "lab_commit: $(git -C "$ROOT" rev-parse HEAD)"
  "$BIN/llama-cli" --version 2>&1 | head -3
  nvidia-smi --query-gpu=name,driver_version,memory.total,power.limit --format=csv; free -m
} > "$OUT/env.txt" 2>&1

busy=$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader)
if [ -n "$busy" ]; then log "GPU busy, aborting: $busy"; exit 2; fi
nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu,power.draw,temperature.gpu \
  --format=csv,noheader -lms 500 >> "$OUT/gpu.csv" &
SMI=$!
trap 'kill $SMI 2>/dev/null' EXIT
mark() { echo "MARK,$(date -u +%FT%T),$1" >> "$OUT/gpu.csv"; }

# --- step 1: add the quantize and imatrix tools to the existing build (no reconfigure, no full rebuild)
if [ ! -x "$BIN/llama-quantize" ] || [ ! -x "$BIN/llama-imatrix" ]; then
  log "step 1: build llama-quantize llama-imatrix"
  cmake --build "$BIN/.." -j2 --target llama-quantize llama-imatrix > "$OUT/build_tools.log" 2>&1 || { log "build failed"; exit 3; }
fi

# --- step 2: importance matrix from the Q8_0 (calibrated on the same WikiText slice the PPL uses:
#             this flatters IQ2 on PPL/KLD, so the task scores are the fair comparison)
IMAT="$MODELS/qwen3.5-4b-q8_0.imatrix"
if [ ! -f "$IMAT" ]; then
  log "step 2: imatrix"; mark imatrix
  "$BIN/llama-imatrix" -m "$Q8" -f "$DATA" -c 512 -ngl 99 -o "$IMAT" > "$OUT/imatrix.log" 2>&1 || { log "imatrix failed"; exit 4; }
fi

# --- step 3: quantise (requantised from Q8_0; no F16 on the laptop)
QUANTS=()
for t in IQ2_XXS IQ2_M; do
  q="$MODELS/Qwen3.5-4B-$t.gguf"
  if [ ! -f "$q" ]; then
    log "step 3: quantize $t"
    "$BIN/llama-quantize" --allow-requantize --imatrix "$IMAT" "$Q8" "$q" "$t" > "$OUT/quantize_$t.log" 2>&1 || { log "quantize $t failed"; continue; }
  fi
  QUANTS+=("$q")
done
ALL=("$BASE" "$BONSAI" "${QUANTS[@]}")
for m in "${ALL[@]}"; do stat -c '%n %s' "$m"; sha256sum "$m"; done | tee "$OUT/models.txt" | tee -a "$OUT/run.log"

# --- step 4: speed (one round, r3, same flags as the lab benchmark)
for m in "${ALL[@]}"; do
  tag=$(basename "$m" .gguf); log "step 4: bench $tag"; mark "bench-$tag"
  "$BIN/llama-bench" -m "$m" -p 512 -n 128 -r 3 -o json -lm none "${RT[@]}" > "$OUT/bench_$tag.json" 2> "$OUT/bench_$tag.log"
  sleep 15
done

# --- step 5: PPL and KLD vs the Qwen Q8_0 reference for the new quants (E1 has them for the other two)
for m in "${QUANTS[@]}"; do
  tag=$(basename "$m" .gguf); log "step 5: kld $tag"; mark "kld-$tag"
  "$BIN/llama-perplexity" -m "$m" -f "$DATA" --chunks 20 -c 512 --kl-divergence-base "$REF" --kl-divergence "${RT[@]}" > "$OUT/kld_$tag.txt" 2>&1
  grep -E "Mean +KLD|Same top|Mean PPL\(Q\)" "$OUT/kld_$tag.txt" | tee -a "$OUT/run.log"
done

# --- step 6: checkable tasks
log "step 6: GSM8K-50 + 10 coding tasks"; mark eval
python3 "$ROOT/experiments/lowbit/eval_tasks.py" --bin "$BIN" --out "$OUT" --models "${ALL[@]}" > "$OUT/eval_progress.log" 2>&1
grep '^.* {' "$OUT/eval_progress.log" | tee -a "$OUT/run.log"
log "done: $OUT"
