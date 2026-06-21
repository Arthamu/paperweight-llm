# qwen-laptop-ai

Qwen3.6 35B-A3B (vision-capable, 128K context) running at **27.92 tok/s median** on a 2019
gaming laptop. GTX 1660 Ti, 6 GB VRAM. opencode pointed at it as a local backend. No cloud.

<video src="demo.mp4" controls muted autoplay loop playsinline width="900">
  <a href="demo.gif"><img src="demo.gif" alt="Qwen3.6 35B-A3B demo" /></a>
</video>

> 1:34 of demo: opencode generating an HTML page → the same model in a chat UI doing live
> web research via custom MCP tools → `bench.sh` for the receipts.

> **Side note on the chat-UI section:** the `google_search` and `deep_research` tool calls
> visible in the middle of the demo come from a small MCP server I wrote in a sibling repo —
> [`local-deep-research`](https://github.com/Arthamu/paperweight-llm/tree/main/local-deep-research) — that gives the local
> model internet access. It plugs into the chat UI through the same `--ui-mcp-proxy`
> mechanism the coding agent uses: local model, real-time sources, no cloud LLM in the loop.

## Hardware

| | |
|---|---|
| CPU | Intel Core i7-9750H (6 cores / 12 threads, 2019) |
| GPU | NVIDIA GTX 1660 Ti, 6 GB VRAM (Turing, sm_75) |
| RAM | 32 GB DDR4 |
| OS  | Ubuntu 24.04 |

## Stack

- **Engine:** llama.cpp Turboquant fork @
  [`4595fff`](https://github.com/TheTom/llama-cpp-turboquant/commit/4595fff0bbd15ee01663699b788eea70e7e1cd69)
- **Model:** Qwen3.6 35B-A3B, Q4_K_M (`lmstudio-community/Qwen3.6-35B-A3B-GGUF`)
- **Vision:** BF16 mmproj loaded, kept on CPU (`--no-mmproj-offload`)
- **Runtime:** Docker, `nvidia/cuda:12.5.1-devel-ubuntu24.04`, fork built on host and
  bind-mounted in
- **Client:** opencode via `--ui-mcp-proxy` at `http://localhost:8080`

## Quick start

**Prerequisite:** build the Turboquant fork. The included `build.sh` clones the fork and
compiles it inside a CUDA container — no host CUDA toolkit needed.

```bash
# 1. Build the fork (one-time; ~10 min on a 9750H).
./build.sh

# 2. Drop the GGUF in the path the script expects, then:
./run-server.sh

# 3. Wait ~90 s for "HTTP server listening" in the container logs.

# 4. In another terminal, run the benchmark:
./bench.sh

# 5. Or talk to it directly:
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"local","messages":[{"role":"user","content":"hi"}],"max_tokens":16}'
```

To rebuild and relaunch in one command, use [`build-and-run.sh`](build-and-run.sh) with the
`--build` flag. For host-side build details (CMake flags, Turing-specific options, common
errors), see [`BUILDING-TURBOQUANT.md`](BUILDING-TURBOQUANT.md).

## The flags that matter

| Flag | Why |
|---|---|
| `-ngl 99` | All layers on GPU |
| `--n-cpu-moe 36` | Push expert tensors to CPU RAM, keep attention/shared on GPU. **The unlock.** |
| `-c 128000` | Full 128K context |
| `--cache-type-k turbo4` | Keys quantized with Turboquant's tighter codebook |
| `--cache-type-v turbo3` | Values, slightly looser — keys tolerate less loss in this model |
| `-t 6 -tb 6` | Match physical cores; going higher contends with MoE work |
| `--mlock --no-mmap` | Pin in RAM, slow first load, instant after |
| `--ui-mcp-proxy` | Expose llama-server as an MCP-compatible endpoint |
| `--mmproj …` `--no-mmproj-offload` | BF16 vision projector on CPU |

## Measured numbers

10 runs, 2 warmup discarded, server-side timings:

| | |
|---|---|
| Generation tok/s | min 26.68 · **median 27.92** · mean 27.65 · max 28.04 |
| Prompt eval tok/s | min 42.20 · median 43.20 · max 44.74 |
| TTFT (≈500-token prompt) | 90 ms |
| VRAM | **4.77 / 6.14 GB** (1.4 GB headroom) |
| GPU power | 48.5 W |
| GPU utilization | **39 %** ← bottleneck is CPU MoE work, not GPU |

The GPU sitting at 39 % is the interesting number. The bottleneck on this hardware is the
6-core 2019 Intel chip, not the GPU. On a modern desktop CPU this same configuration would
likely clear 40 tok/s.

Methodology: [`BENCHMARK.md`](BENCHMARK.md).

## Honest weak spots

- **Prompt eval at 43 tok/s.** Filling 128K context takes minutes. For chat-style use it's
  invisible. For long-document Q&A, pre-summarize.
- **Q4_K_M is a quality compromise.** Q5/Q6 don't fit cleanly on 6 GB at 128K even with
  turbo3/4.
- **Turing is old.** On Ampere or Ada the same setup should be 2–3× faster on generation,
  3–5× on prompt eval.
- **The KV quants are fork-specific.** A GGUF saved with a turbo-quant KV won't load on
  mainline llama.cpp.

## Files

- [`build.sh`](build.sh) — clone the Turboquant fork and build inside a CUDA container, no launch
- [`run-server.sh`](run-server.sh) — Docker launch with the actual flags
- [`build-and-run.sh`](build-and-run.sh) — same launch but with an optional `--build` flag that fetches the fork and recompiles inside the CUDA container before starting
- [`bench.sh`](bench.sh) — 10-run greedy benchmark, server-side timings
- [`BENCHMARK.md`](BENCHMARK.md) — methodology and full results
- [`BUILDING-TURBOQUANT.md`](BUILDING-TURBOQUANT.md) — host-side build for sm_75
- [`coding-agent-config.md`](coding-agent-config.md) — wiring opencode (or Claude Code) at `localhost:8080`

## Reproducing

If you reproduce this on your own hardware, please open an issue with your numbers and
include `nvidia-smi`, `lscpu`, `free -h`, and the fork commit you built. Especially curious
how much the CPU bottleneck moves on a 13th-gen Intel or a Ryzen 7950X.

## License

MIT.
