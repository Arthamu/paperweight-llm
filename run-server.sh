#!/bin/bash
# Launch local Qwen3.6-35B-A3B (multimodal) on llama.cpp Turboquant fork
# Hardware: i7-9750H + GTX 1660 Ti 6GB + 32GB RAM
# Result: 25-27 tok/s generation, 128K context, vision-capable

set -e

echo "🚀 Launching local Qwen3.6-35B TurboQuant container..."

# Docker socket perms (only needed once per session)
sudo chmod 666 /var/run/docker.sock

# Stop any prior container using this image
docker ps -q --filter "ancestor=nvidia/cuda:12.5.1-devel-ubuntu24.04" | xargs -r docker stop

docker run -d \
  --name qwen-server \
  --rm \
  --gpus all \
  --ulimit memlock=-1 \
  --cap-add=IPC_LOCK \
  --network=host \
  -v /run/nvidia-persistenced/socket:/run/nvidia-persistenced/socket \
  -v /home/artha-mukherjee/llama-cpp-turboquant:/root/llama-turbo \
  -v /home/artha-mukherjee/.lmstudio:/root/.lmstudio \
  -e LD_LIBRARY_PATH=/root/llama-turbo/build/bin:/root/llama-turbo/build \
  -w /root/llama-turbo \
  nvidia/cuda:12.5.1-devel-ubuntu24.04 \
  ./build/bin/llama-server \
    -m /root/.lmstudio/models/lmstudio-community/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-Q4_K_M.gguf \
    --mmproj /root/.lmstudio/models/lmstudio-community/Qwen3.6-35B-A3B-GGUF/mmproj-Qwen3.6-35B-A3B-BF16.gguf \
    --no-mmproj-offload \
    --host 0.0.0.0 --port 8080 \
    --alias local \
    -ngl 99 \
    --n-cpu-moe 36 \
    -c 128000 \
    --cache-type-k turbo4 \
    --cache-type-v turbo3 \
    -t 6 -tb 6 \
    --mlock --no-mmap \
    --ui-mcp-proxy

echo "⏳ Waiting for model weights to load (~90s)..."
docker logs -f qwen-server | grep -m 1 "HTTP server listening"
echo "✅ llama-server is online at http://localhost:8080"
