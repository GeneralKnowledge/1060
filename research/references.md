# References

Status of each source: **reported** (not reproduced on our hardware).

- NVIDIA R590 Linux driver drops Maxwell/Pascal/Volta; R580 is the last supporting branch.
  [KitGuru](https://www.kitguru.net/components/graphic-cards/joao-silva/nvidia-drops-maxwell-and-pascal-support-with-new-linux-590-branch-drivers/),
  [Igor'sLAB on Arch moving Pascal off the default driver](https://www.igorslab.de/en/arch-linux-is-phasing-out-pascal-gtx-10-cards-are-being-removed-from-the-default-path-with-nvidia-590/),
  [negativo17 R580 LTS repository](https://negativo17.org/nvidia-driver-580-lts-repository/),
  [Guru3D: 582.66 security driver for Maxwell/Pascal](https://guru3d.com/story/nvidia-releases-geforce-58266-security-driver-for-maxwell-and-pascal-gpus/)
- CUDA 13 drops offline compilation for SM 5.x-7.0; 12.x is the last toolkit line for Pascal.
  [Tom's Hardware](https://www.tomshardware.com/pc-components/gpus/nvidia-to-drop-cuda-support-for-maxwell-pascal-and-volta-gpus-with-the-next-major-toolkit-release),
  [CUDA features archive](https://docs.nvidia.com/cuda/cuda-features-archive/index.html)
- PyTorch dropping CUDA 12.6 wheels (and with them Maxwell/Pascal/Volta) from 2.15.
  [PyTorch dev-discuss](https://dev-discuss.pytorch.org/t/notice-cuda-12-6-wheels-will-no-longer-be-published-from-pytorch-2-15-drops-maxwell-pascal-volta/3432)
- Dell G3 17 3779 configurations (GTX 1050 / 1050 Ti / 1060 Max-Q 6 GB).
  [LaptopMedia](https://laptopmedia.com/laptop-specs/dell-g3-17-3779-2/specs/),
  [GTX 1060 Max-Q 6 GB specs](https://devicebeast.com/laptop-gpu/nvidia-geforce-gtx-1060-max-q-6gb-gddr5)
- llama.cpp (ggml-org/llama.cpp): CUDA backend with DP4A MMQ kernels, Vulkan backend, `llama-bench`,
  `llama-perplexity --kl-divergence`. https://github.com/ggml-org/llama.cpp
- Karpathy, autoresearch (project inspiration).

## Qwen3.5-4B workload
- Model card and `config.json`: https://huggingface.co/Qwen/Qwen3.5-4B
- GGUF files used: https://huggingface.co/unsloth/Qwen3.5-4B-GGUF (Unsloth guide: https://unsloth.ai/docs/models/qwen3.5)
- llama.cpp b12021 sources for the architecture: `src/models/qwen35.cpp`, `src/models/delta-net-base.cpp`,
  `ggml/src/ggml-cuda/gated_delta_net.cu`, `ggml/src/ggml-cuda/ssm-conv.cu`, `ggml/src/ggml-cuda/fattn.cu`
- Gated DeltaNet: Yang, Kautz, Hatamizadeh, "Gated Delta Networks: Improving Mamba2 with Delta Rule" (2024), arXiv:2412.06464
