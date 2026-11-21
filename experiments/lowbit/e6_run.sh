#!/usr/bin/env bash
# E6: squeeze VRAM so more 27B PTQ1_0 layers fit on the 6 GB GTX 1060 at a real context.
# For each config (KV type, -ub, ctx) find the highest -ngl at which llama-completion survives a near-full prompt
# (the VMM pool grows with the prompt, so a short prompt is not enough), then bench it (pp512/tg64) and record peak VRAM.
#   BIN=<prism fork bin> experiments/lowbit/e6_run.sh MODEL.gguf
# Read-only for the lab: does not touch config/, winners/ or the benchmark lock.
set -uo pipefail
# an OOM abort otherwise attaches gdb/lldb for a backtrace and can hang until the timeout
export GGML_NO_BACKTRACE=1
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
BIN="${BIN:?set BIN to the PrismML fork bin dir}"
M="$1"
DATA="${DATA:-$ROOT/benchmarks/workloads/wikitext-2-test-slice.txt}"
OUT="${OUT:-$ROOT/experiments/lowbit/e6/$(date -u +%Y%m%dT%H%M%SZ)}"
mkdir -p "$OUT"
log() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$OUT/run.log"; }
{
  echo "date_utc: $(date -u +%FT%TZ)"; echo "lab_commit: $(git -C "$ROOT" rev-parse HEAD)"
  "$BIN/llama-bench" --version 2>&1 | tail -2
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv; free -m; stat -c '%n %s' "$M"
} >> "$OUT/env.txt" 2>&1
busy=$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader)
if [ -n "$busy" ]; then log "GPU busy, aborting: $busy"; exit 2; fi

# near-full prompts: ~3.3k tokens for ctx 4096, ~1.6k for ctx 2048 (64 tokens are generated after them)
head -c 14000 "$DATA" > "$OUT/prompt_4096.txt"
head -c 6800 "$DATA" > "$OUT/prompt_2048.txt"

peak() { sort -n "$1" | tail -1; }
# name ctk ctv ub ctx
CONFIGS=(
  "A_f16_ub256_c4096 f16 f16 256 4096"
  "B_q8_ub256_c4096 q8_0 q8_0 256 4096"
  "C_q8_ub128_c4096 q8_0 q8_0 128 4096"
  "D_q4_ub128_c4096 q4_0 q4_0 128 4096"
  "E_q8_ub128_c2048 q8_0 q8_0 128 2048"
)
[ -f "$OUT/fit.csv" ] || echo "config,ngl,status,prompt_tokens,pp_tps,tg_tps,peak_vram_mib" > "$OUT/fit.csv"
for c in "${CONFIGS[@]}"; do
  read -r name ctk ctv ub ctx <<< "$c"
  [[ "$name" =~ ${CONFIG_FILTER:-.} ]] || continue
  best=0
  for ngl in ${NGLS:-65 64 63 62 61 60 59 58}; do
    tag="${name}_ngl$ngl"
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -lms 200 > "$OUT/vram_$tag.csv" & SMI=$!
    timeout "${FIT_TIMEOUT:-900}" "$BIN/llama-completion" -m "$M" -ngl "$ngl" -fa 1 -ctk "$ctk" -ctv "$ctv" -b 2048 -ub "$ub" -c "$ctx" \
      -f "$OUT/prompt_$ctx.txt" -n 64 --temp 0 --seed 1 -no-cnv > "$OUT/fit_$tag.txt" 2> "$OUT/fit_$tag.log"
    rc=$?; kill $SMI 2>/dev/null; wait $SMI 2>/dev/null
    pk=$(peak "$OUT/vram_$tag.csv")
    pt=$(grep -oE "prompt eval time = .* / +[0-9]+ tokens" "$OUT/fit_$tag.log" | grep -oE "[0-9]+ tokens" | grep -oE "[0-9]+")
    pp=$(grep -E "prompt eval time" "$OUT/fit_$tag.log" | grep -oE "[0-9.]+ tokens per second" | grep -oE "^[0-9.]+")
    tg=$(grep -E "^.*[^t] eval time" "$OUT/fit_$tag.log" | grep -v prompt | grep -oE "[0-9.]+ tokens per second" | grep -oE "^[0-9.]+")
    if [ $rc -eq 0 ] && [ -n "$tg" ]; then st=ok
    elif grep -qiE "out of memory|failed to allocate" "$OUT/fit_$tag.log"; then st=oom
    elif [ $rc -eq 124 ]; then st=timeout   # inconclusive (slow load under I/O contention), not an OOM
    else st="fail_rc$rc"; fi
    echo "$name,$ngl,$st,$pt,$pp,$tg,$pk" | tee -a "$OUT/fit.csv"
    if [ "$st" = ok ]; then best=$ngl; break; fi
    sleep 3
  done
  log "$name: highest ngl surviving a ${ctx}-ctx prompt = $best"
  [ "$best" -gt 0 ] || continue
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -lms 200 > "$OUT/vram_bench_$name.csv" & SMI=$!
  "$BIN/llama-bench" -m "$M" -ngl "$best" -fa 1 -ctk "$ctk" -ctv "$ctv" -b 2048 -ub "$ub" -p 512 -n 64 -r 2 -o json \
    > "$OUT/bench_$name.json" 2> "$OUT/bench_$name.log"
  kill $SMI 2>/dev/null; wait $SMI 2>/dev/null
  # llama-server -np 1 at the same settings: load, then one near-full-context request
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -lms 200 > "$OUT/vram_server_$name.csv" & SMI=$!
  "$BIN/llama-server" -m "$M" -ngl "$best" -fa 1 -ctk "$ctk" -ctv "$ctv" -b 2048 -ub "$ub" -c "$ctx" -np 1 \
    --host 127.0.0.1 --port 8099 > "$OUT/server_$name.log" 2>&1 & SRV=$!
  python3 - "$OUT/prompt_$ctx.txt" "$OUT/server_$name.json" <<'PY' 2>&1 | tee -a "$OUT/run.log"
import json, sys, time, urllib.request
t0 = time.time()
while time.time() - t0 < 300:
    try:
        urllib.request.urlopen("http://127.0.0.1:8099/health", timeout=2); break
    except Exception:
        time.sleep(1)
body = {"prompt": open(sys.argv[1]).read(), "n_predict": 64, "temperature": 0, "seed": 1}
try:
    r = json.loads(urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8099/completion",
        data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}), timeout=600).read())
    t = r.get("timings", {}); json.dump(r, open(sys.argv[2], "w"))
    print("  server ok: prompt_n", t.get("prompt_n"), "pp", round(t.get("prompt_per_second", 0), 2), "tg", round(t.get("predicted_per_second", 0), 2))
except Exception as e:
    print("  server FAILED:", e)
PY
  kill $SRV 2>/dev/null; wait $SRV 2>/dev/null; kill $SMI 2>/dev/null; wait $SMI 2>/dev/null
  echo "  server peak VRAM $(peak "$OUT/vram_server_$name.csv") MiB" | tee -a "$OUT/run.log"
  python3 -c "import json,sys;[print('  bench', 'pp%d'%r['n_prompt'] if r['n_gen']==0 else 'tg%d'%r['n_gen'], round(r['avg_ts'],2)) for r in json.load(open(sys.argv[1]))]" "$OUT/bench_$name.json" 2>&1 | tee -a "$OUT/run.log"
done

[ "${SKIP_KVCHECK:-0}" = 1 ] && { log "done: $OUT"; exit 0; }
# quality sanity: greedy 64 tokens, short prompt, same ngl, f16 vs q8_0 vs q4_0 KV
for kv in f16 q8_0 q4_0; do
  "$BIN/llama-completion" -m "$M" -ngl 58 -fa 1 -ctk $kv -ctv $kv -b 2048 -ub 256 -c 4096 \
    -p "The three most important ideas in computer science are" -n 64 --temp 0 --seed 1 -no-cnv > "$OUT/kvcheck_$kv.txt" 2> "$OUT/kvcheck_$kv.log"
done
log "done: $OUT"
