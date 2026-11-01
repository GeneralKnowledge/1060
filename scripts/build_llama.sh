#!/usr/bin/env bash
# Build the pinned llama.cpp for the best config's backend (or --backend cuda|vulkan|cpu).
# Builds go to backends/builds/<hash>/ and are reused. Nothing is installed system-wide.
# Prerequisites (not installed by this script): cmake, a C++ compiler, and for CUDA a 12.x
# toolkit (CUDA 13 cannot target sm_61); for Vulkan the Vulkan headers and glslc.
set -euo pipefail
cd "$(dirname "$0")/.."
exec python3 -m lab build "$@"
