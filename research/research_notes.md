# Research notes

Written by the research agent after each experiment. Each entry links its experiment record; only
the numbers in those records are measurements. Everything else here is interpretation.

### 20261101-180123-d476: Enabling flash attention (flash_attn=1) on the CUDA backend at the pinned commit changes balanced performance on this SM
_Decision: **promote** (record experiments/records/20261101-180123-d476.json). Agent interpretation below; only the record's numbers are measurements._

### Experiment 20261101-180123-d476: flash attention on CUDA (FA 0 → 1, f16 KV)

**What was measured**

Single change: `runtime.flash_attn` 0 → 1 at commit 7bd18083 (b12021). Everything else was identical. There were 2 interleaved rounds × 5 repetitions (n=10 for throughput, n=2 for TTFT, VRAM, RSS and efficiency).

| Metric | Baseline | Candidate | Ratio | Noise | Significant |
|---|---|---|---|---|---|
| tg128 (t/s) | 41.139 | 43.678 | 1.0617 | 0.001 | yes |
| pp512 (t/s) | 771.30 | 769.64 | 0.9978 | 0.0039 | no |
| pp2048 (t/s) | 680.83 | 677.44 | 0.995 | 0.003 | no |
| TTFT (s) | 0.6878 | 0.6880 | 0.9997 | 0.0013 | no |
| VRAM peak (MiB) | 2525 | 2509 | 1.0064 | 0.0 | no |
| Host RSS peak (MB) | 2214.0 | 2214.15 | 0.9999 | 0.0001 | no |
| tokens/J | 0.6175 | 0.6677 | 1.0814 | 0.0067 | yes |

- **Decode:** the tg128 trial ranges do not overlap (baseline 41.075–41.180, candidate 43.626–43.697).
- **Quality:** KLD vs the q8_0 reference went from 0.020544 to 0.020753, and PPL from 11.877513 to 11.879274. There were no gate failures. Each arm had one quality run, so the record has no noise estimate for KLD or PPL.
- **Stability:** no errors and no safety events.
- **Conditions matched across arms:**
  - Mean SM clock was about 1521–1529 MHz in both arms.
  - Memory clock stayed at 3802 MHz throughout.
  - Max temperature was 65–67 °C in both.
  - Clock-event reason `0x4` appeared in 20–23 samples per run in both arms.
- **Power and energy:** mean power was 66.94 / 66.31 W for baseline and 65.46 / 65.35 W for candidate. Whole-run energy was 2511 / 2487 J for baseline and 2453 / 2449 J for candidate.
- **Decision:** promote, with balanced score 1.0293 against a threshold of 0.01.

**Verdict on the hypothesis**

- **Decode: supported, and larger than predicted.** I predicted "within noise or slightly higher". The result was +6.2%, about 60× the measured noise.
- **VRAM: direction supported, magnitude largely refuted.** Peak VRAM fell 16 MiB (0.6%), and the controller did not count this as significant. The expected saving from not materialising the KQ matrix mostly did not appear in peak VRAM.
- **Prompt processing: unresolved, with no significant change.** pp2048 was 0.5% lower, which is slightly larger than its noise figure of 0.003, but the decision rule did not flag it. If the trial arrays are in round order, the candidate was lower in both rounds: about −0.6% in round 1 and −0.3% in round 2. That is a possible small, consistent cost that this sample size cannot confirm.
- **Quality: passed the gate.** "Unchanged within measured noise" cannot be strictly confirmed because quality has n=1 per arm.
- **Efficiency: significant gain.** tokens/J rose 8.1%. This is larger than the tg gain (6.2%) and larger than the drop in whole-run energy (about 1.9%). The record does not show how tokens/J is derived, so I would check that before relying on its size.

**What it suggests about the bottleneck (hypotheses)**

- **Decode is not purely limited by weight bandwidth.** Clocks and power-cap behaviour were the same in both arms, so the 6% tg gain came from less work or overhead per token, not from different operating conditions. tg128 runs at very short KV lengths (≤128), so the KQ tensors are tiny. The saving is therefore more likely fewer kernel launches and memory round-trips per layer than reduced attention bandwidth. *Hypothesis:* a meaningful part of decode time on this GPU is fixed per-op overhead, and other op fusions or launch reductions could pay off.
- **The VRAM peak is probably set elsewhere.** *Hypothesis:* the compute-buffer peak comes from another stage of the graph, such as output-layer buffers with a ~128k vocabulary, not from attention. Removing KQ therefore barely moves it.
- **Prompt processing stays compute-bound.** It is insensitive to the FA path, and Pascal's tile FA kernel roughly matches the non-FA matmul path here.

**Follow-up experiments**

1. **H008: `cache_type_k = cache_type_v = q8_0` on top of the promoted FA=1 config.** This was the planned next step, and FA turned out neutral-to-positive on every metric, so the precondition is met. Watch VRAM, tg128 and KLD. *Expected, not measured:* the KV saving at these benchmark context lengths is on the order of 100 MiB. Short-context tg128 may also hide any dequantisation cost in decode.
2. **`ubatch_size` 512 → 1024 with FA=1, as a single variable.** This tests whether pp512/pp2048 can improve now that FA removes the ub×n_kv KQ buffer. The VRAM change will also help locate what actually sets the peak.

**Record note:** the promoted `experimental_config` still contains `baseline_quality` from the FA=0 run (argv `-fa 0`, KLD 0.020544), along with the old `validated_utc` and the "not yet measured" note. The real FA=1 quality result is only under `correctness.candidate` (KLD 0.020753). If the controller uses `baseline_quality` as the reference for the next experiment, it will be comparing against the wrong config. That metadata should be updated.

### 20261101-180922-4bd0: H008, first half: with flash_attn=1 already promoted, quantising the KV cache to q8_0 for both K and V (cache_type_k = c
_Decision: **discard** (record experiments/records/20261101-180922-4bd0.json). Agent interpretation below; only the record's numbers are measurements._

**Experiment 20261101-180922-4bd0: KV cache f16 → q8_0 (K and V), FA=1, Llama-3.2-3B Q4_K_M. Decision: discard** (balanced score 0.9878; wins only the `best_model_capacity` profile)

### What was measured

| Metric | Baseline (f16 KV) | Candidate (q8_0 KV) | Ratio | Noise | Significant |
|---|---|---|---|---|---|
| vram_peak_mib | 2509 | 2409 | 1.0415 | 0.0 | yes |
| tg128_ts | 43.689 | 42.847 | 0.9807 | 0.0014 | yes |
| pp512_ts | 771.40 | 755.18 | 0.979 | 0.0041 | yes |
| pp2048_ts | 678.30 | 666.55 | 0.9827 | 0.0027 | yes |
| ttft_s | 0.6867 | 0.7008 | 0.98 | 0.0023 | yes |
| tokens_per_joule | 0.6668 | 0.6588 | 0.988 | 0.0131 | no |
| host_rss_peak_mb | 2214.0 | 2214.05 | 1.0 | 0.0 | no |

Quality: the controller re-measured the baseline arm at `-fa 1` (KLD 0.020753, PPL 11.879274), so the stale FA=0 `baseline_quality` block did not affect the comparison. The candidate gave KLD 0.020697 and PPL 11.875227, a difference of −0.000056 KLD. There were no gate failures and no stability or safety events.

Clocks do not explain the throughput gap:
- Mean SM clock was 1531–1537 MHz in every round of both arms.
- The `0x4` (SW power cap) clock-event reason appeared at similar counts in both arms (18 and 20 vs 16 and 23).
- Mean power was 64–66 W in both arms.
- Max temperature was 65–67 °C in both arms.

### Verdict on H008 (first half): mixed

**(a) VRAM: supported.** Peak VRAM fell by exactly 100 MiB (4.0%), matching the ~100 MiB / ~4% estimate. That estimate gives about 105 MiB at a 2048-token context; the remaining few MiB are probably allocation granularity (hypothesis). The f16 KV cache is therefore close to the predicted ~224 MiB. That is real but small. Roughly 2.3 GB of the peak is something else: per the hypothesis, mostly the ~1.9 GB of weights plus compute buffers and CUDA context. The record does not decompose this.

**(b) Decode: refuted as stated.** The prediction allowed "slightly lower", but its rationale was that KV traffic is negligible, and on that rationale halving KV bytes should cost nothing or give a tiny gain. Instead tg128 lost a significant 1.9%. That is about 0.45 ms per token (22.89 → 23.34 ms), or roughly 16 µs per layer. Hypothesis: this is fixed per-token compute or kernel overhead, from q8_0 dequantisation in the FA vec kernel and/or quantising new K/V on write. It is not a bandwidth effect.

**(c) Prefill: direction supported, scaling refuted.**
- Both pp tests got slower, but pp2048 lost less in relative terms (−1.7%) than pp512 (−2.1%).
- In absolute time the added cost was about 14.3 ms for pp512 and about 53.2 ms for pp2048. That is 3.7× for 4× the tokens, about 27–28 µs per token in both cases.
- Converting the whole attended KV cache to f16 per ubatch would grow roughly with the sum of KV lengths, which is about 10× from pp512 to pp2048 at ub=512. The measured cost is near-linear in tokens. Hypothesis: the dominant extra cost is per-token work, such as writing or quantising K/V into the cache, not conversion of the existing cache.
- The 14.0 ms TTFT increase is consistent with the 14.3 ms pp512 increase.

**(d) Quality: refuted in the favourable direction.** The predicted ~1e-3 KLD rise did not appear. No quality cost was detected at c=512 over 20 chunks. The −0.000056 difference should not be read as an improvement. Quality at long context, where KV quantisation error could accumulate, was not measured.

### What it suggests about the bottleneck (hypothesis)
At benchmark context lengths (≤2048), decode on this card is bound by weight streaming plus fixed per-kernel cost, not by KV traffic. Shrinking KV bytes buys nothing for speed, and any added arithmetic shows up directly as lost throughput. Peak VRAM is dominated by weights, so further VRAM wins at these lengths must come from compute buffers (e.g. ubatch sizing) or the model itself, not the KV cache. The benefit of a q8_0 KV cache, about 50 KiB saved per token of context, only becomes material at contexts far beyond what this benchmark exercises.

### Follow-ups
1. **ubatch_size 512 → 1024**, single variable, on the promoted f16-KV config. This targets pp512, pp2048 and TTFT directly. Measuring the change in vram_peak_mib will also show how much of the non-KV peak is compute buffer.
2. **Keep q8_0/q8_0 KV as the recorded capacity-profile option rather than retesting variants now.** It costs about 2% throughput, has no detected quality cost, and saves 100 MiB at 2048 context. K-only q8_0 is deprioritised for three reasons:
   - The KLD cost it was meant to work around did not materialise.
   - It would save only about half the VRAM.
   - Mixed q8_0/f16 is not a default-compiled FA vec type pair (hypothesis: this needs `GGML_CUDA_FA_ALL_QUANTS=ON`, a second variable).

   For humans: this benchmark cannot show the capacity value of KV quantisation. Answering that needs a long-context workload, which is a benchmark change outside this agent's remit.
