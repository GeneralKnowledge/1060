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
