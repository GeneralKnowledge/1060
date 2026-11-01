# Qwen3.5-4B: compatibility assessment and setup

Status as of 2026-11-01. This assessment comes from the llama.cpp source at the pinned commit
`7bd18083082126c08d375cefaa6a5aa1c170ecde` (tag b12021) and the model's published `config.json`.
None of it has been run on the laptop yet. The smoke test in section 4 is the first hardware check.

## 1. Verdict

**The existing build is already Qwen3.5-compatible, so no llama.cpp upgrade is needed.** The pinned
b12021 has a `qwen35` architecture (`src/llama-arch.cpp`, `src/models/qwen35.cpp`). Every op the
graph uses has a CUDA implementation that targets SM 6.1 through the same CUDA 12.0 + g++-12
toolchain, and the driver (550) and CUDA toolkit stay as they are.

So the "separate Qwen-compatible backend" is the same build: `backends/builds/d2db58ba90808f32`,
key = backend + commit + cmake flags. Qwen experiments get their own build directory automatically
once they change the commit or flags. The Llama build is untouched either way.

I did not search for the *minimum* llama.cpp revision that supports Qwen3.5. Since b12021 works,
there is no reason to go older. If a newer commit is ever needed, it becomes an experiment that
changes `build.llama_cpp_commit`, in its own build directory.

| Item | Status at b12021 (source) |
|---|---|
| Architecture `qwen35` (hybrid DeltaNet + attention) | present |
| `GGML_OP_GATED_DELTA_NET` on CUDA | supported (`gated_delta_net.cu`, value head 16/32/64/128; Qwen3.5-4B uses 128) |
| Fused DeltaNet for prompt and decode | enabled automatically when the backend supports the op (`fused_gdn_ar`, `fused_gdn_ch`) |
| `SSM_CONV` (conv width 4) | supported, including fused conv+SiLU; needs `d_inner % 128 == 0` (8192 here) |
| `L2_NORM`, `SIGMOID`, `SOFTPLUS`, `CONCAT`, multi-section `ROPE` | supported |
| Unfused fallback ops (`CUMSUM`, `TRI`, `SOLVE_TRI`, `PAD`) | supported (normally unused) |
| Flash attention at head size 256 on Pascal | tile kernel (prompt) and vec kernel (decode) instantiated for D=256 |
| Quantised matmul | MMQ / MMVQ with DP4A, the same paths as Llama |

Architecture support does not mean the kernels are well tuned for this card. The DeltaNet kernel
has no architecture-specific tuning, and MMVQ uses GENERIC parameters on Pascal. Those are research
questions (see `research/workloads/qwen35-4b.md`), not blockers.

## 2. Model files

| Role | File | Size | Source |
|---|---|---|---|
| Workload | `models/Qwen3.5-4B-Q4_K_M.gguf` | 2.74 GB | `unsloth/Qwen3.5-4B-GGUF` |
| KLD reference | `models/Qwen3.5-4B-Q8_0.gguf` | 4.48 GB | same repo |
| Reference logits | `models/qwen3.5-4b-q8_0.kld` | about 2.5 GB (248k vocab, estimate) | written by `lab reference` |

Sizes are as listed on Hugging Face on 2026-11-01. The official `Qwen/Qwen3.5-4B-GGUF` repo was not
reachable, and bartowski's files are slightly larger.

**Pinning.** The sha256 of the GGUF pins the weights, and also the tokenizer and chat template,
because both are stored inside the GGUF. `lab smoke` records the tokenizer model and pre-tokenizer
names (`tokenizer.ggml.model`, `tokenizer.ggml.pre`) and whether a chat template is present. Pin
the Hugging Face commit you downloaded from in `source_revision`. The mmproj vision file is not
used.

## 3. What changed in the lab

- `config/benchmark.toml` (protected, needs re-approval):
  - New workload `qwen35-4b` and family `qwen3.5-4b`, with its own reference logits and its own
    kernel suites.
  - The Llama workload, its hashes and its reference file are unchanged.
- Separate histories:
  - Each workload has its own winners directory: `winners/` for Llama (as before) and
    `winners/qwen35-4b/` for Qwen.
  - The agent only sees records of the workload it is working on.
  - Every record's `execution_environment` carries the workload, family, model file and sha256,
    source revision, architecture, context, backend and llama.cpp commit.
- `--workload` option: `python3 -m lab --workload qwen35-4b <command>`. The default is `[session]
  workload` in `config/research.toml`, which stays `small` (Llama) until the Qwen baseline is
  validated.
- New commands:
  - `lab smoke` loads the model and generates text. It checks the architecture, the CUDA backend
    and that every layer is offloaded, and records VRAM, host RAM, temperature and clocks. It works
    before the hash is pinned, and prints the hash to pin.
  - `lab reference` writes the family's Q8_0 reference logits, once.
- `llama-completion` was added to the build targets, for the smoke test. Existing builds only get
  the missing target added; there is no full rebuild.
- Research: the workload brief is `research/workloads/qwen35-4b.md`. The agent prompt now says that
  results from another workload are priors, not measurements.

## 4. Running it on the laptop

```bash
cd ~/1060 && git pull
mkdir -p models
wget -P models https://huggingface.co/unsloth/Qwen3.5-4B-GGUF/resolve/main/Qwen3.5-4B-Q4_K_M.gguf
wget -P models https://huggingface.co/unsloth/Qwen3.5-4B-GGUF/resolve/main/Qwen3.5-4B-Q8_0.gguf

# 1. Smoke test (adds the llama-completion target to the existing build the first time)
python3 -m lab --workload qwen35-4b smoke
#    -> paste "pin_this_sha256" into [workloads.qwen35-4b] sha256

# 2. Reference logits from the Q8_0 model (lower --ngl if it runs out of VRAM)
python3 -m lab --workload qwen35-4b reference

# 3. Human: approve the changed benchmark, then the baseline
python3 -m lab approve-benchmark "Qwen3.5-4B workload, hashes and kernel suites"
python3 -m lab --workload qwen35-4b baseline
```

Before the baseline, check two things in its record:
- The kernel-check suites must report passed cases. A suite with 0 passed means an op name or
  shape filter matched nothing in test-backend-ops and needs fixing.
- The decode check must run in under 300 s.

To make Qwen the default for sessions and sweeps, set `[session] workload = "qwen35-4b"` in
`config/research.toml`.

## 5. Known limitations

- test-backend-ops tests fixed shapes, which do not include every shape the model uses (see the
  workload brief). The decode-path and KL-divergence checks are the end-to-end net.
- The KL-divergence reference is written with the Q8_0 model at full offload. The 248k vocabulary
  makes the logit buffers large, so if the reference run runs out of memory, use
  `--ngl` below 99. Partial offload changes only speed, not the reference values.
- Vulkan is not checked: there is no Vulkan SDK on the laptop, and the kernel check is CUDA-only.
- Nothing in this document has been measured on the laptop yet.
