#!/bin/bash
# Build llama.cpp Turboquant fork inside a stock CUDA container.
# Binaries land in $REPO_DIR/build/bin/ on the host via bind-mount.

# --- EXACT REPO PATH ---
REPO_DIR="/home/artha-mukherjee/llama-cpp-turboquant"
# -----------------------

echo "Step 1: Checking TurboQuant repository..."
if [ -d "$REPO_DIR" ]; then
    cd "$REPO_DIR" || exit 1
    echo "Using existing repository at $REPO_DIR"
else
    git clone -b feature/turboquant-kv-cache https://github.com "$REPO_DIR"
    cd "$REPO_DIR" || exit 1
fi

echo "Step 2: Building llama-server inside CUDA container..."
docker run --rm --gpus all \
  -v "$REPO_DIR":/root/llama-turbo \
  -w /root/llama-turbo \
  nvidia/cuda:12.5.1-devel-ubuntu24.04 \
  bash -c "
    set -e
    apt-get update -qq && apt-get install -y -qq cmake build-essential libgomp1 > /dev/null 2>&1
    rm -rf build
    mkdir -p build
    cmake -B build -DGGML_CUDA=ON -DGGML_NCCL=OFF
    cmake --build build --config Release -j\$(nproc)
  "

echo "Build complete. Binaries in $REPO_DIR/build/bin/"
