#!/usr/bin/env bash
# E1: 1-bit / 2-bit weight formats on the GTX 1060 (Pascal), see EXTREME_COMPRESSION_RESEARCH.md section 8.
# Usage: experiments/lowbit/e1_run.sh /path/to/Ternary-Bonsai-2-4B-Q2_0.gguf
# Read-only for the lab: does not touch config/, winners/ or the benchmark lock.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
BIN="${BIN:-$ROOT/backends/builds/d2db58ba90808f32/build/bin}"
CAND="$1"
BASE="${BASE:-$ROOT/models/Qwen3.5-4B-Q4_K_M.gguf}"
REF="${REF:-$ROOT/models/qwen3.5-4b-q8_0.kld}"
DATA="${DATA:-$ROOT/benchmarks/workloads/wikitext-2-test-slice.txt}"  # git-ignored; copy from ~/1060
OUT="$ROOT/experiments/lowbit/e1/$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$OUT"
# Same runtime flags as winners/qwen35-4b/best_config.json
RT=(-ngl 99 -fa 1 -ctk f16 -ctv f16 -b 2048 -ub 256)

log() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$OUT/run.log"; }

# --- environment record
{
  echo "date_utc: $(date -u +%FT%TZ)"
  echo "lab_commit: $(git -C "$ROOT" rev-parse HEAD)"
  echo "bin: $BIN"
  "$BIN/llama-cli" --version 2>&1 | head -3
  nvidia-smi --query-gpu=name,driver_version,memory.total,power.limit,clocks.max.sm,clocks.max.mem --format=csv
  free -m
  echo "candidate: $CAND"; sha256sum "$CAND"; stat -c 'bytes %s' "$CAND"
  echo "baseline: $BASE"; stat -c 'bytes %s' "$BASE"
} > "$OUT/env.txt" 2>&1

# --- GPU must be free (the mind's deep tier yields to other GPU users but may hold it now)
busy=$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader)
if [ -n "$busy" ]; then log "GPU busy, aborting: $busy"; exit 2; fi

# --- GPU telemetry sampler (VRAM, util, power, temp every 0.5 s)
nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu,power.draw,temperature.gpu,clocks.sm,clocks.mem \
  --format=csv,noheader -lms 500 >> "$OUT/gpu.csv" &
SMI=$!
trap 'kill $SMI 2>/dev/null' EXIT
mark() { echo "MARK,$(date -u +%FT%T),$1" >> "$OUT/gpu.csv"; }

# --- step 1: kernel correctness vs CPU on Pascal
log "step 1: test-backend-ops"
mark kernels
"$BIN/test-backend-ops" test -b CUDA0 -o MUL_MAT -p "type_a=(q1_0|q2_0|iq1_s|iq2_xxs|tq2_0),type_b=f32" > "$OUT/kernels_mul_mat.txt" 2>&1
"$BIN/test-backend-ops" test -b CUDA0 -o GET_ROWS > "$OUT/kernels_get_rows.txt" 2>&1
grep -E "passed|FAIL|not supported" "$OUT/kernels_mul_mat.txt" | tail -5 | tee -a "$OUT/run.log"

# --- step 2: smoke (load, offload, VRAM, RSS, sample text)
for m in "$CAND" "$BASE"; do
  tag=$(basename "$m" .gguf)
  log "step 2: smoke $tag"
  mark "smoke-$tag"
  /usr/bin/time -v "$BIN/llama-completion" -v -m "$m" -p "The three most important ideas in computer science are" \
    -n 96 -c 4096 --temp 0 --seed 1 -no-cnv "${RT[@]}" > "$OUT/smoke_$tag.txt" 2> "$OUT/smoke_$tag.log"
  grep -E "offloaded|n_vocab|general.architecture|file type|model size|Maximum resident|Elapsed|eval time|load time" "$OUT/smoke_$tag.log" | tee -a "$OUT/run.log"
done

# --- step 3: speed, interleaved A B A B (same flags as the lab benchmark: pp512, pp2048, tg128, r5, no mmap)
for round in 1 2; do
  for m in "$BASE" "$CAND"; do
    tag=$(basename "$m" .gguf)
    log "step 3: bench round $round $tag"
    mark "bench-$round-$tag"
    "$BIN/llama-bench" -m "$m" -p 512,2048 -n 128 -r 5 -o json -lm none "${RT[@]}" > "$OUT/bench_${round}_$tag.json" 2> "$OUT/bench_${round}_$tag.log"
    sleep 20
  done
done

# --- step 4: quality. PPL (batched + decode path) and KLD vs the Qwen3.5-4B Q8_0 reference
for m in "$CAND" "$BASE"; do
  tag=$(basename "$m" .gguf)
  log "step 4: perplexity $tag"
  mark "ppl-$tag"
  if [ -f "$REF" ]; then
    "$BIN/llama-perplexity" -m "$m" -f "$DATA" --chunks 20 -c 512 --kl-divergence-base "$REF" --kl-divergence "${RT[@]}" > "$OUT/kld_$tag.txt" 2>&1
  fi
  "$BIN/llama-perplexity" -m "$m" -f "$DATA" --chunks 20 -c 512 "${RT[@]}" > "$OUT/ppl_$tag.txt" 2>&1
  "$BIN/llama-perplexity" -m "$m" -f "$DATA" --chunks 2 -c 512 "${RT[@]}" -ub 1 > "$OUT/ppl_decode_$tag.txt" 2>&1
  grep -E "Final estimate|Mean +KLD|Mean KLD" "$OUT/kld_$tag.txt" "$OUT/ppl_$tag.txt" "$OUT/ppl_decode_$tag.txt" 2>/dev/null | tee -a "$OUT/run.log"
done

# --- step 5: fixed prompts (greedy, thinking off) for side-by-side reading
log "step 5: prompts"
python3 "$ROOT/experiments/lowbit/run_prompts.py" --bin "$BIN" --out "$OUT" --models "$CAND" "$BASE" \
  --prompts "$ROOT/experiments/lowbit/prompts-v0.jsonl" 2>&1 | tail -3 | tee -a "$OUT/run.log"

log "done: $OUT"
