# Benchmark methodology

How the "25–27 tok/s" number was measured. If you publish numbers, publish
the method too — r/LocalLLaMA and HN will ask, and "trust me" is not an
answer.

## Hardware under test
- CPU: Intel Core i7-9750H (6 cores / 12 threads, base 2.6 GHz, Coffee Lake-H)
- GPU: NVIDIA GTX 1660 Ti, 6 GB VRAM (Turing, sm_75, no Tensor cores)
- RAM: 32 GB DDR4-2666
- Storage: NVMe SSD (model loaded from local disk)
- OS: Ubuntu 24.04, kernel 6.x
- NVIDIA driver: ≥ 555
- CUDA runtime: 12.5.1 (matched between host build and container)
- Power: AC plugged in, balanced perf governor
- Cooling: laptop on stand, room temperature ~22°C

## Software under test
- llama.cpp Turboquant fork — commit `4595fff0bbd15ee01663699b788eea70e7e1cd69`
- Built with: `GGML_CUDA=ON CMAKE_CUDA_ARCHITECTURES=75 GGML_CUDA_F16=ON GGML_CUDA_FORCE_MMQ=ON`
- Runtime: Docker, image `nvidia/cuda:12.5.1-devel-ubuntu24.04`, binary bind-mounted from host
- Server flags (full): see `run-server.sh`
- Key flags: `-ngl 99 --n-cpu-moe 36 -c 128000 --cache-type-k turbo4 --cache-type-v turbo3 -t 6 -tb 6 --mlock --no-mmap`

## Model
- Qwen3.6 35B-A3B (MoE, ~3B active params per token)
- Quant: Q4_K_M (~21 GB on disk)
- Vision projector: BF16 mmproj loaded but not exercised in this benchmark
- Source: `lmstudio-community/Qwen3.6-35B-A3B-GGUF` on Hugging Face
- File hash: `<sha256>` (fill in for reproducibility)

## Method

The `bench.sh` script in this repo:

1. Sends a fixed system prompt (~500 tokens, real prose) and a fixed user
   question via `POST /v1/chat/completions`.
2. Requests exactly **512 generated tokens, greedy (`temperature=0`)** so
   token count is deterministic across runs.
3. Reads `timings.prompt_ms`, `timings.predicted_ms`, `timings.prompt_n`,
   `timings.predicted_n` from the llama-server response. These are the
   server's own measurements — wall-clock would include network and JSON
   overhead.
4. Runs **10 iterations**, discards the first 2 as warmup (KV cache cold
   start, CUDA kernel JIT, page faults).
5. Reports min / median / mean / max for generation tok/s, prompt-eval
   tok/s, and time-to-first-token (TTFT, derived from `prompt_ms`).

### Why these choices

- **Greedy (`temp=0`)**: same output every run, so token counts are stable
  and any variance is purely runtime, not sampling.
- **512 generated tokens**: long enough to amortize startup costs, short
  enough that thermal throttling doesn't dominate.
- **Server-side timings, not wall-clock**: removes HTTP/JSON noise and
  matches what other llama.cpp benchmarks report.
- **Drop 2 warmup runs**: the first generation pays a CUDA graph capture
  cost and a one-time KV allocation. Reporting it would lie low.
- **Fresh server, but warm model**: `--mlock` keeps weights in RAM. After
  load, every run starts from the same memory state. No `drop_caches`
  between runs — that would simulate a fresh boot, not normal use.

### What the numbers do *not* include
- Cold model load (~90 s with `--mlock --no-mmap`). Reported separately.
- Long-context behavior. This benchmark uses ~600 prompt tokens. Throughput
  at 100K tokens of real context is lower; that's a different test.
- Vision (mmproj) latency. Image prefill adds ~200–400 ms per image and is
  out of scope here.
- Concurrent requests. llama-server is single-stream in this config.

## Results (measured)

    ===== Results (excluding 2 warmup runs) =====
    Generation tok/s       min=26.68  median=27.92  mean=27.65  max=28.04  (n=8)
    Prompt eval tok/s      min=42.20  median=43.20  mean=43.21  max=44.74  (n=8)
    TTFT (seconds)         min=0.09   median=0.09   mean=0.09   max=0.10   (n=8)

    Hardware snapshot during run:
    NVIDIA GeForce GTX 1660 Ti, 4771 MiB / 6144 MiB used, 48.55 W, 39% util

### What these numbers tell us

**Generation: ~28 tok/s, very stable.** Standard deviation across the 8
measured runs is under 0.6 tok/s. This is a reliable, repeatable number.

**TTFT: 90 ms.** Genuinely fast — at this prompt size the first token
arrives faster than most cloud APIs. Note this scales with prompt length;
TTFT at 50K input tokens will be ~20 minutes, not 90 ms.

**Prompt eval: 43 tok/s.** This is the honest weak spot. Filling a long
context is slow. 10K input tokens → ~4 minutes before generation starts.
For chat or short prompts this is fine. For long-document Q&A, prefer
shorter excerpts or pre-summarize.

**GPU at 39% utilization, 48 W.** The GPU is idle most of the cycle,
waiting on CPU MoE work. The bottleneck is CPU expert matmul, not GPU
compute. This is why throwing a faster GPU at the problem wouldn't help
much — but a faster CPU would.

**VRAM at 4.77 / 6.14 GB.** ~1.4 GB of headroom. Options:
- Lower `--n-cpu-moe` (e.g., 30 instead of 36) to push more experts back
  to GPU. Likely 10–15% generation speedup. Try it.
- Or raise context to 160K. Probably fits with current settings.

## Comparison: mainline vs Turboquant

Run the same `bench.sh` against mainline llama.cpp (same commit-day, same
flags except KV cache type forced to `q8_0`):

| Build | Gen tok/s (median) | PP tok/s (median) | VRAM | TTFT |
|---|---|---|---|---|
| Mainline llama.cpp `<SHA>`, q8_0 KV | _fill in_ | _fill in_ | _fill in_ | _fill in_ |
| **Turboquant fork `4595fff`, turbo4/turbo3 KV** | **27.92** | **43.20** | **4.77 GB** | **0.09 s** |

The Turboquant delta is the entire reason the article is interesting.
Make sure this table has real mainline numbers before publishing.

## Reproducing

1. Build the fork per `BUILDING-TURBOQUANT.md`.
2. Start the server: `./run-server.sh`.
3. Wait for `HTTP server listening` in logs (~90 s).
4. In another terminal: `./bench.sh`.
5. Paste output here. Open a GitHub issue if your numbers are >20% off and
   include `nvidia-smi`, `lscpu`, `free -h`, and the commit SHA.

## Honest caveats
- Turing is old. On Ampere or Ada, prompt-eval will be 3–5× faster and
  generation 2–3× faster on the same model.
- Q4_K_M is a quality compromise. Q5_K_M or Q6_K would be slightly better
  output and slightly slower; they don't fit cleanly on 6 GB with 128K
  context even with turbo3/4 KV.
- The MoE routing in Qwen3.6-A3B is well-behaved on this hardware.
  Different MoE models (different expert counts, different routing) won't
  necessarily hit the same numbers with `--n-cpu-moe`.
