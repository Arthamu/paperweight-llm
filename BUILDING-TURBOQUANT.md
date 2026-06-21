# Building llama.cpp Turboquant fork on the host

This is how I build the Turboquant fork on the host (Ubuntu 22.04/24.04) and
bind-mount the resulting binaries into a stock CUDA Docker container at
runtime. No custom image. No 8GB Docker layer for every CUDA bump.

**Target hardware:** NVIDIA GTX 1660 Ti → CUDA compute capability **`sm_75`** (Turing).

If you're on a different GPU, swap the arch. Reference:
- 1660 Ti / 2060 / 2070 / 2080 → `75`
- 3060 / 3070 / 3080 / 3090 → `86`
- 4060 / 4070 / 4080 / 4090 → `89`
- A100 → `80`, H100 → `90`

---

## 1. Prerequisites

    sudo apt update
    sudo apt install -y build-essential cmake git ccache pkg-config libcurl4-openssl-dev

CUDA toolkit 12.5 (matches the runtime container):

    # Pick the .run installer or apt repo from
    # https://developer.nvidia.com/cuda-12-5-0-download-archive
    nvcc --version   # confirm 12.5.x

NVIDIA driver: any version ≥ 555 supports CUDA 12.5 runtime.

    nvidia-smi   # confirm driver + GPU visible

---

## 2. Clone the fork

    cd ~
    git clone https://github.com/TheTom/llama-cpp-turboquant.git
    cd llama-cpp-turboquant

**Pin a commit.** This fork moves fast and can break.

    git log --oneline -5
    # Note the SHA you're building. Record it in your repo README.

---

## 3. Configure with CMake

    cmake -B build \
      -DGGML_CUDA=ON \
      -DCMAKE_CUDA_ARCHITECTURES=75 \
      -DCMAKE_BUILD_TYPE=Release \
      -DLLAMA_CURL=ON \
      -DGGML_CUDA_F16=ON \
      -DGGML_CUDA_FORCE_MMQ=ON \
      -DGGML_CUDA_FORCE_CUBLAS=OFF

Notes on the flags that matter for Turing:

| Flag | Why |
|---|---|
| `CMAKE_CUDA_ARCHITECTURES=75` | Build only for your GPU. Cuts compile time ~4×. |
| `GGML_CUDA_F16=ON` | Enables FP16 paths. Turing supports it natively. |
| `GGML_CUDA_FORCE_MMQ=ON` | Use the integer-mul kernels for quantized matmul. On Turing, MMQ is faster than cuBLAS for Q4_K_M. |
| `GGML_CUDA_FORCE_CUBLAS=OFF` | Don't fall back to cuBLAS for quantized matmul. |

If the fork ships extra Turboquant-specific options (KV codebook tuning, etc.), check the fork's `CMakeLists.txt` for new flags before building.

---

## 4. Build

    cmake --build build --config Release -j $(nproc)

On an i7-9750H this takes about 8–12 minutes the first time.
With `ccache`, incremental rebuilds drop to under a minute.

Verify the binaries:

    ls -la build/bin/
    # Should include: llama-server, llama-cli, llama-bench, ...

Smoke test:

    ./build/bin/llama-cli --version
    ./build/bin/llama-server --help | grep -E "n-cpu-moe|cache-type|ui-mcp"

You should see `--n-cpu-moe`, `--cache-type-k turbo4`, and `--ui-mcp-proxy`
in the help output. If any are missing, you're on the wrong fork or commit.

---

## 5. Why bind-mount instead of building inside Docker

The script in `run-server.sh` mounts `~/llama-cpp-turboquant` into a stock
`nvidia/cuda:12.5.1-devel-ubuntu24.04` container at `/root/llama-turbo`,
then runs the host-built binary. Three reasons:

1. **Iteration speed.** Rebuilding the fork doesn't rebuild a Docker image.
   Edit code → `cmake --build build` → restart container.
2. **Layer size.** A CUDA devel image plus llama.cpp build artifacts is ~10GB.
   Bind-mounting keeps the image immutable and the binary outside.
3. **CUDA driver compat.** The container only needs the CUDA *runtime*; the
   *driver* lives on the host. The stock NVIDIA CUDA image already includes
   everything else.

The trade: the binary is dynamically linked against host CUDA libs that must
match what's in the container. Using `nvidia/cuda:12.5.1-devel-ubuntu24.04`
matches a host CUDA 12.5 build cleanly. If your host CUDA is 12.6, switch
the container tag to `12.6.x-devel-ubuntu24.04`.

The `LD_LIBRARY_PATH=/root/llama-turbo/build/bin:/root/llama-turbo/build`
env var in `run-server.sh` ensures the container finds the build's bundled
shared libs first.

---

## 6. Rebuild workflow

When the fork updates:

    cd ~/llama-cpp-turboquant
    git fetch
    git log HEAD..origin/main --oneline   # review changes
    git checkout <new-commit-sha>
    cmake --build build --config Release -j $(nproc)
    docker stop qwen-server || true
    ./run-server.sh

No image rebuild, no model re-download. Restart is ~90s (model load) plus the
build time.

---

## 7. Common build problems

**`nvcc fatal: Unsupported gpu architecture 'compute_75'`**
Your CUDA toolkit is older than 10.0. Upgrade to 12.5.

**`undefined reference to cublasLtMatmul`**
You're linking against the wrong cuBLAS. Make sure `CUDA_HOME` points at
`/usr/local/cuda-12.5` and rerun `cmake -B build` from a clean state.

**Build OOMs on the laptop**
`-j $(nproc)` = 12 parallel jobs on a 9750H. nvcc compiles are heavy.
Drop to `-j 4` if you see the OOM killer trigger.

**Binary builds but `llama-server` crashes on startup with CUDA error**
Driver/runtime version skew. `nvidia-smi` shows the driver's max supported
CUDA. The container's CUDA runtime version must be ≤ that. Drop the
container tag to `12.4.x` if your driver is older.

**`turbo4` / `turbo3` not recognized**
You're on mainline llama.cpp, not the Turboquant fork. Check `git remote -v`.

---

## 8. Verifying the build is actually using the GPU correctly

Once the server is running, in another terminal:

    watch -n 0.5 nvidia-smi

Send a request. During generation you should see:
- GPU utilization: 60–90%
- Memory used: ~5.4 GB
- Power draw: ~70W
- Process listed: `llama-server`

If GPU utilization stays at 0% during generation, `-ngl 99` didn't take.
Check `docker logs qwen-server` for a line like:

    llm_load_tensors: offloaded 64/64 layers to GPU

If it says `0/64`, the binary wasn't built with CUDA. Start over from step 3.
