#!/usr/bin/env bash
# E4a: harder checkable eval (MATH-500 level 4-5 integer answers, first 30; 8 harder coding tasks; 1024-token cap)
#   experiments/lowbit/e4_run.sh hard MODEL.gguf [MODEL.gguf ...]
# E4b: 27B partial offload: find the largest -ngl that fits, measure speed/memory, a handful of hard items
#   experiments/lowbit/e4_run.sh offload Ternary-Bonsai-2-27B-Q2_0.gguf
# Read-only for the lab: does not touch config/, winners/ or the benchmark lock.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
BIN="${BIN:-$ROOT/backends/builds/d2db58ba90808f32/build/bin}"
MODE="$1"; shift
OUT="$ROOT/experiments/lowbit/e4/$(date -u +%Y%m%dT%H%M%SZ)-$MODE"
mkdir -p "$OUT"
HARD=(--math "$ROOT/experiments/lowbit/math500-l45-int-first30.jsonl" --code "$ROOT/experiments/lowbit/code-tasks-hard-v0.jsonl" --max-tokens 1024 --tag hard)
log() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$OUT/run.log"; }
FAIL=0
fail() { FAIL=1; log "FAILED: $*"; }
{
  echo "date_utc: $(date -u +%FT%TZ)"; echo "lab_commit: $(git -C "$ROOT" rev-parse HEAD)"
  "$BIN/llama-cli" --version 2>&1 | head -3
  nvidia-smi --query-gpu=name,driver_version,memory.total,power.limit --format=csv; free -m
  for m in "$@"; do stat -c '%n %s' "$m"; done
} > "$OUT/env.txt" 2>&1
busy=$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader)
if [ -n "$busy" ]; then log "GPU busy, aborting: $busy"; exit 2; fi
nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu,power.draw,temperature.gpu \
  --format=csv,noheader -lms 500 >> "$OUT/gpu.csv" &
SMI=$!
# host memory/swap every 2 s (offload runs share 7.6 GiB with the OS and the mind)
( while true; do echo "$(date -u +%T) $(free -m | awk '/Mem:/{print "used",$3,"avail",$7} /Swap:/{print "swap",$3}' | tr '\n' ' ')"; sleep 2; done ) >> "$OUT/mem.log" &
MEMLOG=$!
trap 'kill $SMI $MEMLOG 2>/dev/null' EXIT
mark() { echo "MARK,$(date -u +%FT%T),$1" >> "$OUT/gpu.csv"; echo "MARK $(date -u +%T) $1" >> "$OUT/mem.log"; }

if [ "$MODE" = hard ]; then
  log "hard eval: $*"; mark hard
  python3 "$ROOT/experiments/lowbit/eval_tasks.py" --bin "$BIN" --out "$OUT" "${HARD[@]}" --models "$@" > "$OUT/eval_progress.log" 2>&1 || fail "hard eval"
  grep '^.* {' "$OUT/eval_progress.log" | tee -a "$OUT/run.log"

elif [ "$MODE" = offload ]; then
  M="$1"
  avail=$(free -m | awk '/Mem:/{print $7}')
  log "MemAvailable ${avail} MiB"
  if [ "$avail" -lt 3000 ]; then log "under 3 GB RAM available, aborting"; exit 5; fi
  # mmap on (default): CPU-side layers stay in the page cache instead of a second copy
  RT=(-fa 1 -ctk f16 -ctv f16 -b 2048 -ub 256)
  best=0
  for ngl in ${NGLS:-36 42 46 50 54 58 62 65}; do
    log "bench ngl=$ngl"; mark "bench-ngl$ngl"
    if timeout 900 "$BIN/llama-bench" -m "$M" -ngl "$ngl" -p 512 -n 64 -r 2 -o json "${RT[@]}" > "$OUT/bench_ngl$ngl.json" 2> "$OUT/bench_ngl$ngl.log" \
       && grep -q avg_ts "$OUT/bench_ngl$ngl.json"; then
      best=$ngl
      python3 -c "import json,sys;[print('  n_prompt',r['n_prompt'],'n_gen',r['n_gen'],'avg_ts',round(r['avg_ts'],2)) for r in json.load(open(sys.argv[1]))]" "$OUT/bench_ngl$ngl.json" | tee -a "$OUT/run.log"
    else
      log "ngl=$ngl failed (see bench_ngl$ngl.log)"; tail -3 "$OUT/bench_ngl$ngl.log" | tee -a "$OUT/run.log"; break
    fi
    swap=$(free -m | awk '/Swap:/{print $3}'); log "swap used ${swap} MiB"
  done
  log "largest working ngl: $best"
  [ "$best" -gt 0 ] || exit 6
  log "smoke ngl=$best"; mark smoke
  /usr/bin/time -v "$BIN/llama-completion" -v -m "$M" -ngl "$best" "${RT[@]}" -p "The three most important ideas in computer science are" \
    -n 64 -c 4096 --temp 0 --seed 1 -no-cnv > "$OUT/smoke.txt" 2> "$OUT/smoke.log" || fail "smoke ngl=$best"
  grep -E "offloaded|general.architecture|model size|Maximum resident|load time|eval time|out of memory" "$OUT/smoke.log" | tee -a "$OUT/run.log"
  log "hard eval subset at ngl=$best"; mark eval
  python3 "$ROOT/experiments/lowbit/eval_tasks.py" --bin "$BIN" --out "$OUT" "${HARD[@]}" \
    --math-limit "${MATH_N:-6}" --code-limit "${CODE_N:-3}" --server-args "-ngl $best -np 1 ${RT[*]}" --models "$M" > "$OUT/eval_progress.log" 2>&1 || fail "offload eval"
  grep '^.* {' "$OUT/eval_progress.log" | tee -a "$OUT/run.log"
fi
if [ "$FAIL" = 0 ]; then log "done: $OUT"; else log "INCOMPLETE, a required step failed: $OUT"; exit 1; fi
