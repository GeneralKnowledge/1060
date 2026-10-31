# Hypotheses

None of these are established. Each needs an experiment before it can move to research notes.

| ID | Hypothesis | Test (milestone) | Status |
|---|---|---|---|
| H001 | GPU is a GTX 1060 Max-Q 6 GB (GP106) | M0 audit | open |
| H002 | Installed RAM is a single 8 GB SODIMM (single-channel) | M0 audit with `--sudo` | open |
| H003 | Power limit is not adjustable via `nvidia-smi -pl` | M0 audit reads limits | open |
| H004 | Decode (tg) is memory-bandwidth bound; tok/s scales with model bytes, not FLOPs | M1 quant sweep | open |
| H005 | llama.cpp CUDA beats Vulkan on prompt processing; the decode gap is smaller | M1 backend comparison | open |
| H006 | Sustained runs throttle on thermal or power before 5 minutes | M1 telemetry | open |
| H007 | `GGML_CUDA_F16` is slower on Pascal (1/64-rate FP16) | M1/M2 build-flag experiment | open |
| H008 | KV cache q8_0 + flash attention extends usable context with negligible KLD cost | M2 | open |
