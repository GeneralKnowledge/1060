# Technical assessment and M0/M1 plan

_Written 2026-10-31, before the target laptop was available for inspection._

Every hardware statement below is tagged:

- **[verified]**: observed on the target machine (nothing yet; the M0 audit produces these).
- **[reported]**: vendor documentation, release notes or press reports. Not reproduced here.
- **[hypothesis]**: an inference to be tested.

Until `scripts/hardware_audit.sh` runs on the laptop, the profile in
`research/hardware_profile.json` does not exist and none of the hardware claims here are established.

## 1. What the machine can realistically do

**GPU identity is unconfirmed.** The Dell G3 17 (3779, 2018) shipped with GTX 1050, 1050 Ti or GTX 1060
**Max-Q 6 GB** options [reported]. If ours is the 1060 Max-Q: GP106, 10 SMs / 1280 CUDA cores, 6 GB GDDR5 on a
192-bit bus (about 192 GB/s), and a VBIOS power limit far below the desktop card's 120 W [reported, exact figure
unknown]. A 3 GB variant also existed on other laptops, so VRAM must come from the audit.

Properties of SM 6.1 that shape everything [reported, NVIDIA architecture docs]:

- **No tensor cores, and FP16 arithmetic runs at 1/64 of the FP32 rate on GP106.** FP16 compute paths are a
  trap. Useful math is FP32 or **INT8 via `DP4A`** (4-way int8 dot product, introduced in SM 6.1).
- **Token generation is memory-bandwidth bound.** An upper bound is `bandwidth / bytes read per token`. With
  ~192 GB/s, a 7-8B Q4_K_M model (~4.7 GB) caps near 40 tok/s, and real llama.cpp numbers are typically half that
  or less [hypothesis until measured]. A 3-4B Q4 model (~2-2.5 GB) leaves room for a usable KV cache.
- **Prompt processing is compute bound.** On Pascal it rides on llama.cpp's MMQ (DP4A-based quantised matmul)
  kernels, so this is where kernel work can pay off.
- **Laptop thermals and power limits will likely dominate sustained performance** [hypothesis]. A 2018 laptop
  probably has dried thermal paste and dusty fins; benchmarks must record clock-event reasons so throttling is
  seen, not mistaken for regressions.
- **8 GB host RAM** caps what can be memory-mapped, CPU-offloaded or compiled in parallel. If only one SODIMM is
  fitted (common in the base G3 config [reported]), memory is single-channel and CPU offload will be slow.

Realistic goal for the balanced profile: a 3-4B model fully on GPU at Q4-Q6 with 4-8k context, and a 7-8B Q4
model as the "capacity" profile with full or near-full GPU offload if VRAM is 6 GB.

## 2. Appropriate software stack

| Layer | Choice | Why |
|---|---|---|
| OS | Ubuntu Server LTS, as installed | Version unknown; recorded by the audit. |
| Driver | **NVIDIA R580 branch** (`nvidia-driver-580` / `-580-server`), pinned | The 590 branch dropped Maxwell/Pascal/Volta; R580 is the last branch supporting Pascal and receives security fixes as a long-lived branch [reported]. Never let the package manager move to 590+. |
| CUDA toolkit | **12.x (12.8 or 12.9)**, pinned | CUDA 13.0 removed compilation for SM 5.x-7.0 [reported]. The R580 driver itself runs CUDA 12.x binaries fine. |
| Inference engine | **llama.cpp**, pinned commit | Actively supports SM 6.1 with DP4A MMQ kernels, also has a Vulkan backend and a CPU backend for comparison. |
| Builds | three builds of the same commit: `GGML_CUDA=ON CMAKE_CUDA_ARCHITECTURES=61`, `GGML_VULKAN=ON`, CPU-only | Compare backends on identical code. |
| Not used | PyTorch-based engines (vLLM, exllamav2/v3, TensorRT-LLM) | Need SM 7.0+ or FP16 tensor paths; current PyTorch is dropping Pascal wheels [reported]. |
| Vulkan driver | NVIDIA proprietary ICD | Nouveau/NVK cannot reclock Pascal (no GSP), so it would run at boot clocks [reported]. |

The driver choice is the single biggest risk item: if the machine is on nouveau or on an unsupported driver, M1
needs a driver install, which requires your approval.

## 3. Experiments that are immediately possible (once M1 exists)

All are software-only and reversible:

1. Backend comparison: CUDA vs Vulkan vs CPU, same model and settings.
2. GPU layer split (`-ngl`) for models that do not fully fit.
3. MMQ vs cuBLAS paths (`GGML_CUDA_FORCE_MMQ` / `GGML_CUDA_FORCE_CUBLAS`) for prompt processing.
4. Flash attention on/off, and KV-cache types (`f16`, `q8_0`, `q4_0`) vs context length.
5. `-b` / `-ub` (batch / micro-batch) sweeps for prompt processing.
6. Quantisation sweep at fixed model: Q4_0, Q4_K_M, Q5_K_M, Q6_K, IQ4_XS, with KL-divergence vs a reference.
7. CPU threads and mmap/mlock behaviour under 8 GB RAM.
8. Build flags: CUDA graphs, `GGML_CUDA_F16` (expected to be slower on Pascal, worth confirming), `-O3`/LTO.

## 4. Optimisations needing further research

- Custom DP4A mat-vec kernels for the specific quant formats used, tuned for 10 SMs.
- Speculative decoding with a small draft model (VRAM-constrained; 0.5-1B draft).
- CPU/GPU hybrid placement of specific tensors (`--override-tensor`) beyond simple layer splits.
- Alternative forks (e.g. ik_llama.cpp) whose quant types may suit Pascal; must be benchmarked, not assumed.
- Ternary/BitNet-style models: interesting because they map to integer ops, but model quality and kernel support on
  Pascal are unproven.

## 5. Hardware restrictions that may be exploitable (all approval-gated, Mode 2/3)

- **Power limit**: on GeForce mobile parts `nvidia-smi -pl` is usually "not supported" [reported]; the audit reads
  min/default/max limits to confirm. If adjustable, it is a supported runtime interface (resets on reboot).
- **Clock locking** (`nvidia-smi -lgc/-lmc`): documented for Volta+; whether Pascal mobile accepts it is unknown.
  Worth one approved, reversible test, because a locked lower clock is the closest thing to undervolting on Linux.
- **Clock offsets** need `nvidia-settings` with Coolbits and an X server. Headless means a dummy X server, which is a
  persistent config change and requires approval.
- **Undervolting**: no supported V/F-curve interface for Pascal on Linux [reported]. Effective undervolt = clock
  cap at a lower P-state voltage; research only.
- **Fans** are controlled by the Dell embedded controller, possibly visible via `dell-smm-hwmon` [hypothesis]. Manual
  control on G-series is unreliable and could overheat the machine, so it stays approval-gated.
- **Physical**: repasting and cleaning may be the largest sustained-performance gain available. That is a human
  task, and the lab should measure before and after.
- VBIOS modification on this generation is signature-protected [reported]. Research only, never executed.

## 6. What remains unknown (the audit answers the first group)

Answered by M0: exact GPU SKU and PCI IDs, VRAM size, VBIOS version, installed driver and whether nouveau is active,
CUDA/Vulkan availability, PCIe link width/generation, power-limit readability, thermal thresholds, idle clocks and
temperatures, fan/pwm interfaces, CPU model and SIMD flags, RAM size and (with `--sudo`) DIMM/channel layout, storage
type, Ubuntu release, Optimus/hybrid graphics state.

Needs M1 load tests: sustained clocks under load, time-to-throttle, which throttle reason fires first, real memory
bandwidth, power draw under load, run-to-run variance.

## 7. Risks and dependencies

| Risk | Mitigation |
|---|---|
| Unattended upgrades move the driver to 590+ or break DKMS on a kernel update | Pin/hold driver and kernel packages (approval required). Audit records versions so drift is visible. |
| CUDA compile OOMs with 8 GB RAM | Build with `-j2`, CUDA arch 61 only. |
| Thermal state makes results noisy | Cool-down gate before each trial (GPU temp within N °C of idle), repeated trials, record clock-event reasons. |
| llama.cpp drops or regresses Pascal paths | Pin a commit; upgrades are explicit experiments. |
| Optimus: dGPU sleeps (runtime D3) and wakes per call | Audit records runtime power state; M1 warms up before timing. |
| Host RAM exhaustion from mmap + CPU offload | Safety limits in M2; benchmark records peak RSS. |

## 8. Initial benchmark workload (proposed for M1)

Fixed, hashed inputs; any change to them is a new benchmark version.

- **Primary model**: a ~3B instruct model at Q4_K_M (e.g. Llama-3.2-3B-Instruct or Qwen2.5-3B-Instruct, pinned by
  SHA-256). Fits fully in 6 GB with room for KV cache; also fits a 3 GB card at lower context.
- **Capacity model**: a 7-8B instruct model at Q4_K_M, partial or full offload depending on audited VRAM.
- **Metrics via `llama-bench -o json`**: `pp512`, `pp2048` (prompt), `tg128` (decode), at `-r 5` repetitions after
  one warm-up, context 4096, FA on and off recorded as separate profiles.
- **Time to first token**: `llama-server` with a fixed 1k-token prompt, measured client-side.
- **Quality / correctness**: `llama-perplexity` on a fixed 20-chunk wikitext-2 slice, plus `--kl-divergence`
  against reference logits from a Q8_0 build of the same model. A candidate fails if KLD rises above a set bound.
- **Telemetry**: 1 Hz nvidia-smi sampling (temperature, SM/mem clock, power, clock-event reasons) alongside each run.
- **Noise**: report median and spread; an improvement smaller than the spread from 3 baseline runs is "inconclusive".

## 9. Smallest viable M0/M1

**M0 (this PR)**: `scripts/hardware_audit.sh` → `lab/hardware_audit.py`, stdlib only, read-only. Produces
`research/hardware_profile.json` where every fact carries `status` and `source`, keeps raw evidence, checks the
spec's claims, and lists unknowns. Tests cover the parsers and the read-only command whitelist.

**M1 (next)**: about four files.

1. `scripts/build_llama.sh`: pinned llama.cpp commit, three builds into `backends/`, `-j2`.
2. `config/benchmark.yaml`: models with SHA-256, prompt sets, repetitions, benchmark version.
3. `benchmarks/benchmark.py`: runs `llama-bench`/`llama-perplexity`, samples telemetry, applies the cool-down gate,
   appends to `experiments/results.tsv`.
4. A first baseline run of CPU vs CUDA vs Vulkan recorded as verified research.

M1 prerequisite: a working R580 driver and CUDA 12.x toolkit on the laptop. If the audit shows they are missing,
installing them is your call.
