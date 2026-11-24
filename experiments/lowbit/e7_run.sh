#!/usr/bin/env bash
# E7: stack PrismML packing + E6 memory settings + speculative decoding (MTP head, n-gram, small draft model).
#   BIN=<prism fork bin> experiments/lowbit/e7_run.sh [A|B|all]
# Every variant runs the same 6-prompt greedy workload through llama-server -np 1 -c 4096 (e7_spec.py).
# Read-only for the lab: does not touch config/, winners/ or the benchmark lock.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
BIN="${BIN:?set BIN to the PrismML fork bin dir}"
MD="${MODELS:-$HOME/1060/models}"
OUT="${OUT:-$ROOT/experiments/lowbit/e7/$(date -u +%Y%m%dT%H%M%SZ)}"
PART="${1:-all}"
mkdir -p "$OUT"
export GGML_NO_BACKTRACE=1
log() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$OUT/run.log"; }
{
  echo "date_utc: $(date -u +%FT%TZ)"; echo "lab_commit: $(git -C "$ROOT" rev-parse HEAD)"
  "$BIN/llama-server" --version 2>&1 | tail -2
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv; free -m; uptime
} >> "$OUT/env.txt" 2>&1
busy=$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader)
if [ -n "$busy" ]; then log "GPU busy, aborting: $busy"; exit 2; fi
head -c 14000 "$ROOT/benchmarks/workloads/wikitext-2-test-slice.txt" > "$OUT/fit_prompt.txt"   # ~3.2k tokens

run() {  # name model "args" [fit]
  [[ "$1" =~ ^(${ONLY:-.*})$ ]] || return 0
  log "variant $1: $(basename "$2") $3"
  python3 "$ROOT/experiments/lowbit/e7_spec.py" --bin "$BIN" --out "$OUT" --name "$1" --model "$2" --args "$3" \
    ${4:+--fit-prompt "$OUT/fit_prompt.txt"} 2>&1 | tee -a "$OUT/run.log"
  sleep 5
}

Q9="$MD/Ternary-Bonsai-2-9B-Q2_0.gguf"
MTP9="$MD/Ternary-Bonsai-2-9B-PQ2_0-MTP-Q8_0.gguf"
P27="$MD/Ternary-Bonsai-2-27B-PTQ1_0.gguf"
D2="$MD/Ternary-Bonsai-2-2B-PTQ1_0.gguf"
F16="-fa 1 -ctk f16 -ctv f16 -b 2048 -ub 256"
Q8="-fa 1 -ctk q8_0 -ctv q8_0 -b 2048 -ub 256"
C27="-fa 1 -ctk q8_0 -ctv q8_0 -b 2048 -ub 128"   # E6 config C

if [ "$PART" = A ] || [ "$PART" = all ]; then
  run A0      "$Q9"   "-ngl 99 $F16"
  run A3a     "$Q9"   "-ngl 99 $F16 --spec-type ngram-mod"
  run A1n     "$MTP9" "-ngl 99 $F16"
  run A1m2    "$MTP9" "-ngl 99 $F16 --spec-type draft-mtp --spec-draft-n-max 2"
  run A1m3    "$MTP9" "-ngl 99 $F16 --spec-type draft-mtp --spec-draft-n-max 3"
  run A2n     "$MTP9" "-ngl 99 $Q8"
  run A2m2    "$MTP9" "-ngl 99 $Q8 --spec-type draft-mtp --spec-draft-n-max 2"
  run A2m3    "$MTP9" "-ngl 99 $Q8 --spec-type draft-mtp --spec-draft-n-max 3"
  run A3b     "$MTP9" "-ngl 99 $Q8 --spec-type draft-mtp,ngram-mod --spec-draft-n-max ${A_BEST_N:-2}"
fi

if [ "$PART" = B ] || [ "$PART" = all ]; then
  run B0      "$P27" "-ngl 62 $C27"
  run B1      "$P27" "-ngl 62 $C27 --spec-type ngram-mod"
  # 2B draft on the CPU, target keeps 62 layers
  run B2cpu3  "$P27" "-ngl 62 $C27 -md $D2 -ngld 0 --spec-type draft-simple --spec-draft-n-max 3" fit
  # 2B draft fully on the GPU: lower the target's layers until both survive a near-full prompt
  for ngl in ${B2_NGLS:-56 54 52 50}; do
    run "B2g${ngl}n3" "$P27" "-ngl $ngl $C27 -md $D2 -ngld all -ctkd q8_0 -ctvd q8_0 --spec-type draft-simple --spec-draft-n-max 3" fit
    if tail -1 "$OUT/variants.jsonl" | grep -q '"status": "ok"'; then
      run "B2g${ngl}n5" "$P27" "-ngl $ngl $C27 -md $D2 -ngld all -ctkd q8_0 -ctvd q8_0 --spec-type draft-simple --spec-draft-n-max 5" fit
      break
    fi
  done
fi
log "done: $OUT"
