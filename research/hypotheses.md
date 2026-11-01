# Hypotheses

None of these are established unless the status says so. Known results live in `research/priors.md`;
experiments go to the open questions there.

| ID | Hypothesis | Test (milestone) | Status |
|---|---|---|---|
| H001 | GPU is a GTX 1060 Max-Q 6 GB (GP106) | M0 audit | confirmed by the audit (VBIOS 86.06.6F.00.46) |
| H002 | Installed RAM is a single 8 GB SODIMM (single-channel) | M0 audit with `--sudo` | open |
| H003 | Power limit is not adjustable via `nvidia-smi -pl` | M0 audit reads limits | open |
| H004 | Decode (tg) is memory-bandwidth bound; tok/s scales with model bytes, not FLOPs | M1 quant sweep | open |
| H005 | llama.cpp CUDA beats Vulkan on prompt processing; the decode gap is smaller | M1 backend comparison | open |
| H006 | Sustained runs throttle on thermal or power before 5 minutes | M1 telemetry | open |
| H007 | `GGML_CUDA_F16` is slower on Pascal (1/64-rate FP16) | M1/M2 build-flag experiment | known (hardware spec), moved to priors; not worth an experiment |
| H008 | KV cache q8_0 + flash attention extends usable context with negligible KLD cost | M2 | speed half measured: discarded at ctx 4096 (`20261101-180922-4bd0`); long-context capacity still open |
