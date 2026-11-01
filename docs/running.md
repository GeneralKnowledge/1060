# Running the lab on the laptop

Everything below runs as a normal user. Steps marked **(human)** change the machine or the
benchmark definition and are yours to run; the lab never does them itself.

## 0. Audit (M0)

```bash
scripts/hardware_audit.sh --sudo     # read-only; --sudo only adds `sudo -n dmidecode`
```

## 1. Toolchain (human, only if the audit shows it missing)

- An NVIDIA driver that supports Pascal (R580 is the last branch) and a **CUDA 12.x** toolkit
  (CUDA 13 cannot build for sm_61). Hold the driver package so upgrades do not move it to 590+.
  The lab laptop currently runs driver 550.163.01 with CUDA 12.0 and g++-12 as the host
  compiler; that combination built and validated the baseline, so no driver change is needed.
- `cmake`, `g++`, `git`; for the Vulkan backend also the Vulkan headers and `glslc`.
- Python 3.11+ (stdlib only). `pip install anthropic` if you use the Anthropic provider.

## 2. Build llama.cpp (M1)

```bash
scripts/build_llama.sh                    # backend from winners/best_config.json (cuda)
scripts/build_llama.sh --backend vulkan
scripts/build_llama.sh --backend cpu
```

Pinned to llama.cpp tag b12021 (`7bd1808`). Builds use `-j2` because of the 8 GB of RAM.

## 3. Benchmark inputs (human)

```bash
mkdir -p models
# Check these file names on Hugging Face before downloading.
wget -P models https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q4_K_M.gguf
wget -P models https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q8_0.gguf
# Quality text: wikitext-2 test set (llama.cpp ships scripts/get-wikitext-2.sh), first ~200 KB
head -c 200000 wiki.test.raw > benchmarks/workloads/wikitext-2-test-slice.txt
python3 -m lab hash models/Llama-3.2-3B-Instruct-Q4_K_M.gguf     # paste into config/benchmark.toml
python3 -m lab hash benchmarks/workloads/wikitext-2-test-slice.txt
# Reference logits for KL divergence, from the Q8_0 model (written once, then fixed):
BIN=$(ls -d backends/builds/*/build/bin | head -1)
$BIN/llama-perplexity -m models/Llama-3.2-3B-Instruct-Q8_0.gguf \
  -f benchmarks/workloads/wikitext-2-test-slice.txt --chunks 20 -c 512 -ngl 99 \
  --kl-divergence-base models/llama-3.2-3b-q8_0.kld
python3 -m lab approve-benchmark "pinned model + dataset hashes"   # re-locks the benchmark
```

Any edit to `benchmarks/` or `config/benchmark.toml` invalidates the lock, and the controller
refuses to run until a human re-approves it. This is what stops an experiment from changing the
benchmark and then using it to declare success.

## 4. Baseline (M2)

```bash
python3 -m lab baseline      # measures winners/best_config.json and marks it validated
python3 -m lab status
```

## 5. Experiments

Manual (M2):
```bash
scripts/run_experiment.sh experiments/configurations/example_flash_attn.json
```

Agent (M3), one step or unattended (M4):
```bash
export ANTHROPIC_API_KEY=...        # or set provider.kind in config/research.toml
python3 -m lab agent-step
tmux new -d -s research 'python3 -m lab session --max-hours 8 | tee -a experiments/session.log'
touch experiments/STOP              # ask a running session to stop after the current experiment
```

Each experiment runs in `experiments/worktrees/<id>` on branch `exp/<id>`. It produces
`experiments/records/<id>.json` (plus `.diff` when files changed) and a row in
`experiments/results.tsv`. A promotion merges the branch, archives the previous winner to
`winners/history/` and commits. Hardware or privileged proposals are written to
`experiments/proposals/` for you to review; nothing in that folder is executed.

`scripts/restore_best.sh` removes stray worktrees and branches and resets `winners/` to the last
commit. To roll back a promotion, use `git revert` on its merge commit.
