#!/usr/bin/env bash
# Unattended remainder of the W0 feasibility spike. Writes docs/wsp-feasibility.md, commits and pushes.
cd ~/1060
R=.wsp/results; mkdir -p $R; L=.wsp/logs; PY=.venv-wsp/bin/python
B=backends/builds/d2db58ba90808f32/build/bin; Q=.wsp/build-cpu/bin
DOC=docs/wsp-feasibility.md; M=models/wsp
log(){ echo "$(date -u +%FT%TZ) $*" | tee -a $R/progress.log; }
gate(){ ac=$(cat /sys/class/power_supply/AC/online); pl=$(nvidia-smi -q -d POWER | awk -F: '/Current Power Limit/{print $2; exit}' | tr -d ' W')
  if [ "$ac" != 1 ] || [ "${pl%.*}" != 70 ]; then log "BLOCKER power gate failed (AC=$ac limit=$pl)"; echo "**Stopped: power gate failed before $1 (AC=$ac, GPU limit=$pl W).**" >> $DOC; finish; exit 1; fi; }
waitup(){ for i in $(seq 240); do curl -sf localhost:$1/health >/dev/null && return 0; sleep 2; done; return 1; }
finish(){ git add $DOC scripts/wsp && git commit -q -m "W0 weight-space programming feasibility results

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" ; for d in 2 4 8 16 0; do git push -q -u origin claude/project-thread-x6m1aj && break; sleep $d; done; log "DONE pushed=$?"; }
md(){ echo "$*" >> $DOC; }

log "start"
# ---- step 3b: 4B Q8_0
[ -f $M/qwen3.5-4b-q8_0.gguf ] || /usr/bin/time -v $Q/llama-quantize $M/qwen3.5-4b-f16.gguf $M/qwen3.5-4b-q8_0.gguf Q8_0 > $L/quant-4b.log 2>&1; log "quant4b exit=$?"
q4=done-earlier
c4=$(grep "Maximum resident" $L/conv-4b.log | awk '{printf "%.2f GB", $NF/1048576}'); ct4=$(grep "Elapsed" $L/conv-4b.log | awk '{print $NF}')
f4=$(ls -l $M/qwen3.5-4b-f16.gguf | awk '{printf "%.2f GB", $5/1e9}')


# ---- step 5: llama-server throughput
md ""; md "## 5. llama-server throughput (Q8_0, 256 generated tokens per request, ignore_eos, greedy)"; md ""
md "| model | -np | aggregate gen tok/s | per-slot tok/s | VRAM used (MiB) |"; md "|---|---|---|---|---|"
for m in 0.8b 4b; do for np in 1 2 4; do
  gate "throughput $m np=$np"
  $B/llama-server -m $M/qwen3.5-$m-q8_0.gguf -ngl 99 -fa on -np $np -c $((np*2048)) --port 8093 > $L/server-$m-$np.log 2>&1 & sp=$!
  if waitup 8093; then out=$($PY scripts/wsp/throughput.py 8093 $np 256 2>>$L/tp-err.log); vram=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
    echo "$out" >> $R/throughput.jsonl
    md "| $m | $np | $(echo "$out" | $PY -c 'import json,sys;d=json.load(sys.stdin);print(d["aggregate_tok_s"],"|",d["per_slot_tok_s"])' 2>/dev/null || echo 'error | -') | $vram |"
  else md "| $m | $np | server failed to start | - | - |"; fi
  kill $sp; wait $sp 2>/dev/null; sleep 3
done; done

# ---- step 6: transformers fp32 forward + hook
gate "transformers forward"
md ""; md "## 6. PyTorch on sm_61 + transformers hook"; md ""
md "torch 2.14.1+cu126 (pip, cu126 index): wheel arch list has sm_60 (not sm_61) and runs on the 1060: 2048² fp32 matmul max abs err vs CPU 4.6e-4, about 2.9 TFLOPS fp32."
md "Qwen3.5-0.8B fp32, 512-token forward, forward hook on \`model.layers[15].mlp\`:"; md ""
out=$($PY scripts/wsp/hook_forward.py models/hf/Qwen3.5-0.8B models/wikitext/wikitext-2-raw/wiki.test.raw 2> $L/hook.log | tail -1); log "hook: $out"
md '```'; md "${out:-FAILED, see .wsp/logs/hook.log: $(tail -3 $L/hook.log)}"; md '```'

# ---- step 7: GSM8K loop audit
md ""; md "## 7. Loop audit: GSM8K train[0:32], thinking on, T=0.6 top-p 0.95 top-k 20, max 8192 tokens"; md ""
md "Loop = some 12-word n-gram repeats >= 5 times in the last 3000 words of reasoning+answer. Accuracy is from the last \\\\boxed{} in the final answer."; md ""
for m in 0.8b 4b; do
  gate "loop audit $m"; np=$([ $m = 4b ] && echo 2 || echo 4)
  $B/llama-server -m $M/qwen3.5-$m-q8_0.gguf -ngl 99 -fa on -np $np -c $((np*9216)) --jinja --port 8094 > $L/audit-server-$m.log 2>&1 & sp=$!
  if waitup 8094; then t=$(date +%s)
    $PY scripts/wsp/loop_audit.py .wsp/data/gsm8k/main/train-00000-of-00001.parquet 8094 $R/audit-$m.jsonl --workers $np > $L/audit-$m.log 2>&1
    log "audit $m exit=$? $(tail -1 $L/audit-$m.log)"
    md "**$m** (-np $np, $(( ($(date +%s)-t)/60 )) min): \`$(tail -1 $L/audit-$m.log)\`"; md ""
  else md "**$m**: server failed to start"; fi
  kill $sp; wait $sp 2>/dev/null; sleep 3
done
md "Per-problem records: \`.wsp/results/audit-*.jsonl\` on the laptop (not committed)."
finish
