# A GPU-Poor's Guide to Local LLM Inference in 2026

*MoE math and KV cache quants below q8_0, plus a worked example: a 35B Mixture-of-Experts on 6 GB of VRAM.*

---

What works on consumer hardware right now if your GPU is in the 4 to 12 GB VRAM band. Techniques first, then a worked example: a 35-billion-parameter Mixture-of-Experts model running at 28 tokens per second with the full 128K context window, on a 2019 gaming laptop with a GTX 1660 Ti and 6 GB of VRAM.

The gap between "you need a 4090" and "your old laptop will do" has narrowed in 2025 and 2026 in ways that aren't widely written about yet.

## What "GPU-poor" actually means in 2026

Any consumer setup with 4 to 12 GB of VRAM. That covers a 6 GB GTX 1660 Ti, an 8 or 12 GB RTX 3060, an 8 GB 4060, most M-series Macs with 8 to 16 GB of unified RAM, and most laptop GPUs from the last six years.

The conventional wisdom said you needed a 24 GB 3090 or a Mac Studio. Not anymore.

## What moved the line

### 1. MoE models with small active parameter counts

A Mixture-of-Experts model has many more *total* parameters than *active* parameters per token. Qwen3.6 35B-A3B: 35 billion total, ~3 billion active at any token (the "A3B" suffix means "Activated 3 Billion"). The router picks a small subset of experts; the rest stay quiet.

Compute load per token is closer to a 3B dense model than a 35B one, and inactive experts can live somewhere slower than VRAM as long as the active path stays fast.

A 35B dense model needed 24 GB of VRAM minimum a couple of years ago. A 35B-A3B with the right tensor placement fits on 6 GB.

### 2. Tensor placement flags that respect MoE structure

For dense models, the standard advice is "fit as many layers as you can on GPU, push the rest to CPU." That fails for MoE because every layer's attention path needs the GPU.

The flag for MoE in mainline llama.cpp is `--n-cpu-moe`. Keep every layer's attention and shared tensors on GPU, push the *expert* tensors out to system RAM.

```
-ngl 99
--n-cpu-moe 36
```

The numbers fall out nicely on a 35B-A3B Q4_K_M quant. Attention plus shared weights fits comfortably under 5 GB of VRAM. Expert weights pushed to CPU sit around 16 GB, easy on 32 GB system RAM. The GPU handles attention work, the CPU handles expert matmuls.

Granularity is per-layer-of-experts, not per-individual-expert: all routed experts of a selected layer go to CPU together. For per-expert placement, fall back to hand-written `-ot` regexes against `blk.<i>.ffn_*_exps`. Dense models still want `--tensor-split` and `-ts`.

### 3. KV cache quantization beyond q8_0

This is the one almost nobody writes about, and it's what makes long context viable on small VRAM.

At 128K, the KV cache is the dominant memory cost. Mainline llama.cpp does support sub-q8_0 KV cache (q5_1, q5_0, q4_1, q4_0, iq4_nl have all been available in the CUDA backend since [PR #7527](https://github.com/ggerganov/llama.cpp/pull/7527)), but in practice q8_0 is the quality floor most practitioners settle on. The most-cited public benchmark is Sam McLeod's writeup ([Bringing K/V Context Quantisation to Ollama](https://smcleod.net/2024/12/bringing-kv-context-quantisation-to-ollama/)): q8_0 KV is essentially indistinguishable from f16, q5_1 shows a small perplexity bump, and q4_0 visibly hurts code generation, especially on V with Flash Attention.

So the question for a 6 GB card at 128K isn't "does mainline support sub-q8_0?" It does. The real question is "can I push below q8_0 without the quality regression?" The Turboquant fork ([github.com/TheTom/llama-cpp-turboquant](https://github.com/TheTom/llama-cpp-turboquant)) ships two new KV cache formats, `turbo4` for keys and `turbo3` for values, with a codebook tuned to attention K/V tensor distributions instead of treating them as generic floats.

```
--cache-type-k turbo4
--cache-type-v turbo3
```

Asymmetric on purpose. The fork puts more bits on K than V on the claim that K and V tensors have different distributions, an intuition with academic backing in [KVQuant (Hooper et al., 2024)](https://arxiv.org/abs/2401.18079). I have not yet run a wikitext-2 perplexity comparison of `turbo4`/`turbo3` against `q8_0`/`q8_0` on this model. Worth flagging: McLeod's data suggests V is *more* sensitive to quant pressure than K, opposite to the fork's K-heavy split. Whether that's model-specific or a real gap in the defaults is open.

On a 6 GB card at 128K, q8_0 KV already fits but eats most of the spare VRAM. The asymmetric turbo4/turbo3 path frees ~1.4 GB of headroom that I now use for batch and graph buffers. The framing isn't "the fork makes 128K possible." It's "the fork makes 128K comfortable."

Caveat: these specific quants are fork-specific, and a KV cache saved with a turbo-quant format will not load on mainline. Pin a commit and read the diff before running it in production. I'm pinned at `4595fff`. Note that mainline already exposes independent `--cache-type-k` and `--cache-type-v` flags, so asymmetric K/V *selection* is mainline today; what's fork-specific in Turboquant is the K/V-tuned codebooks themselves.

Other forks do similar work. ik_llama.cpp by Iwan Kawrakow ([ikawrakow/ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp)) introduces the IQK quant family and CPU matmul kernels.

### 4. MCP-based tooling so the local model is actually useful

Model Context Protocol (MCP) became the common protocol for agent tooling in 2025; most major coding agents (Claude Code, Cursor, opencode) speak it. The Turboquant fork ships a `--ui-mcp-proxy` flag that exposes llama-server with an MCP-compatible web UI. Mainline llama-server has its own built-in chat UI, but the MCP-proxy wiring is fork-specific in the build I run; check `llama-server --help` for yours.

Point any MCP-aware coding agent at `http://localhost:8080`, and it transparently uses the local model. No cloud LLM in the loop.

You can stack MCP servers in front. I wrote a small one called `local-deep-research` ([github.com/Arthamu/local-deep-research](https://github.com/Arthamu/local-deep-research)) with `google_search` and `deep_research` tools. The local 35B-A3B does the reasoning while the MCP server fetches live sources, and the chat UI sees both as tools.

Net effect: your local model answers questions about live web results and edits code on your machine without sending data to a cloud provider.

## The worked example: 28 tok/s on a GTX 1660 Ti

Measured numbers from the cheapest hardware I had on a shelf.

**The hardware:**
- Intel Core i7-9750H (6 cores, 2019)
- NVIDIA GTX 1660 Ti, 6 GB VRAM (Turing TU116, sm_75, no Tensor cores)
- 32 GB DDR4
- Ubuntu 24.04, Docker, NVIDIA driver 555+

The GTX 1660 Ti is TU116 silicon, sharing Turing's compute capability 7.5 with the RTX 20-series but shipping *without* Tensor Cores ([GeForce 16 series](https://en.wikipedia.org/wiki/GeForce_16_series)). llama.cpp's CUDA backend falls back to dp4a INT8 dot-product kernels instead of mma tensor-core paths. This is the floor of what works.

**The model:** Qwen3.6 35B-A3B at Q4_K_M (~21 GB on disk, GGUF from `lmstudio-community/Qwen3.6-35B-A3B-GGUF`). BF16 vision projector loads via `--mmproj` since the model is multimodal.

**The engine:** llama.cpp Turboquant fork at commit `4595fff`, built with `GGML_CUDA=ON CMAKE_CUDA_ARCHITECTURES=75 GGML_CUDA_F16=ON GGML_CUDA_FORCE_MMQ=ON` on the host, bind-mounted into a stock `nvidia/cuda:12.5.1-devel-ubuntu24.04` container. `GGML_CUDA_FORCE_MMQ=ON` is non-optional on TU116: the CUDA backend's [build docs](https://github.com/ggerganov/llama.cpp/blob/master/docs/build.md) note the custom MMQ kernels were tuned for RTX 3000/4000, and on Turing without tensor cores the cuBLAS FP16 path is software-emulated and slower at long context. CUDA 12.5 supports sm_75 with NVIDIA driver 555+ ([release notes](https://docs.nvidia.com/cuda/archive/12.5.1/cuda-toolkit-release-notes/index.html)).

**The full launch flags:**

```
-ngl 99
--n-cpu-moe 36
-c 128000
--cache-type-k turbo4
--cache-type-v turbo3
-t 6 -tb 6
--mlock --no-mmap
--ui-mcp-proxy
--mmproj <path>
--no-mmproj-offload
--alias local
```

Of those, only `--cache-type-k turbo4` and `--cache-type-v turbo3` are fork-specific. The rest is mainline llama.cpp.

**Measured numbers (10 runs, 2 warmup discarded, server-side timings):**

| Metric | Value |
|---|---|
| Generation tok/s | min 26.68 · **median 27.92** · mean 27.65 ± 0.57 (n=8) · max 28.04 |
| Prompt eval tok/s | min 42.20 · median 43.20 · max 44.74 |
| TTFT, warm cache, ≈500-token prompt | 90 ms |
| VRAM used | **4.77 / 6.14 GB** (1.4 GB headroom) |
| GPU power | 48.5 W |
| GPU utilization | **39 %** |

A note on TTFT: bench.sh sends the same prompt 10 times back-to-back, so runs 3-10 are full prompt-cache hits with llama-server reusing prior KV state. The 90 ms is steady-state warm-cache TTFT. A genuinely cold 500-token prefill on this hardware works out to ~500 / 43.2 ≈ 11.6 s. Two orders of magnitude in the user-facing number depending on which condition you mean, so quote the right one.

Raw per-iteration generation values, in execution order: 26.68, 26.79, 27.89, 27.90, 27.94, 27.95, 28.01, 28.04 tok/s. Six values cluster between 27.89 and 28.04; the two slow ones are first iterations after warmup, suggesting the warmup is a run or two short.

### The most surprising number

The GPU sits at **39 % utilization** during generation. The bottleneck is not the GPU. It's the 2019 6-core Intel chip doing the expert matmuls on the CPU side, the path `--n-cpu-moe 36` selects.

A quick check this is genuinely CPU-compute and not PCIe contention. Under `--n-cpu-moe`, during decode, only the per-layer per-token hidden state crosses PCIe. With 36 CPU-resident MoE layers, two crossings per layer, fp16, and a hidden_dim around 4096, that's ~576 KB per token. At 28 tok/s it works out to ~16 MB/s, three orders of magnitude under PCIe 3.0 x16's ~16 GB/s ceiling. The other direction agrees: streaming ~1.65 GB of active experts per token to sustain 28 tok/s would demand ~46 GB/s, about 3× what PCIe 3.0 x16 carries. (Worth checking your link width with `nvidia-smi -q | grep -i pcie`; many laptops downshift to x8, which only strengthens the conclusion.)

This is the opposite of what most local-LLM articles assume. Upgrading the GPU would barely move the needle. Upgrading the CPU would. On a 13th-gen Intel or Ryzen 7950X with ~1.5-1.7× sustained DRAM bandwidth, this should land in the low 40s tok/s based on bandwidth scaling, untested by me on those chips.

Prompt eval at 43 tok/s is the honest weak spot. That number is from a ~500-token prompt fitting in a single 512-token microbatch, so matmul cost dominates. As context grows, attention work scales with cumulative K, so cumulative prompt-fill cost grows roughly as N²/2. On Turing without tensor cores the breakpoint where attention starts visibly degrading prompt-eval is somewhere in the 4K-8K range, and a naïve linear extrapolation to a near-full 128K window gives around 50 minutes; actual fill time is worse. I have not measured an 8K, 32K, or 128K datapoint yet; that's the next test on my list. For chat-style use this is invisible; for long-document Q&A, pre-summarize or excerpt.

## How to translate this to your hardware

Techniques generalize, your specific flags will not.

**4-6 GB VRAM (1660 Ti, 3050, base 4060):** Lean on MoE. Qwen3.6 35B-A3B at Q4_K_M is a sweet spot. Use `--n-cpu-moe` aggressively. 8K-32K context is comfortable on plain q8_0 KV; 128K needs turbo3/4, or you tolerate q4_0's quality regression.

**8 GB VRAM (3060, 3070 mobile, 4060):** A dense 7B-13B at Q4 fits without CPU offload. For MoE, lower `--n-cpu-moe` to 24-30 and reclaim some speed. Q5_K_M starts to fit at long context.

**12 GB VRAM (3060 12 GB, 4070):** A 14B dense at Q5 fits. A 35B-A3B with `--n-cpu-moe 24` should clear 35-40 tok/s. Long context with q8_0 KV is fine without the fork.

**16 GB Mac (M1/M2/M3 Pro):** Different story. Metal backend, not CUDA, so the Turboquant fork doesn't apply. Unified memory means no PCIe bottleneck, so `--n-cpu-moe` doesn't help. A 14B-24B dense at Q4 runs at 25-35 tok/s on mainline llama.cpp with Metal.

**24 GB+ VRAM (3090, 4090, 7900 XTX):** You aren't GPU-poor. The same MCP wiring works; you'll have headroom for bigger models and longer context.

The variables to adjust on any hardware: model size and quantization level, context window, KV cache quant format, `--n-cpu-moe` count for MoE, `--tensor-split` for dense, and thread count to physical cores.

## Wiring it into a workflow

Three ways I actually use it.

**As a coding agent backend.** opencode pointed at `http://localhost:8080`. I tried Claude Code first; it works mechanically, but agents designed against frontier-class APIs assume more total context budget than a 35B-A3B local model usefully has, and tool-use chains felt cut short. opencode's leaner default system prompt left more room. Worth a tokenizer pass yourself: count each agent's static system prompt and built-in tool definitions (the second is usually 3-10× the first) against the Qwen tokenizer to see what's gone before the first user message.

**As a chat UI with web research.** The Turboquant fork ships an open-source web frontend. Drop the `local-deep-research` MCP server in front and the local model gets `google_search` and `deep_research` tools.

**As a private long-context Q&A engine.** Documents I won't send to a vendor (internal RFCs, draft contracts) get loaded into the 128K window and queried locally.

Any one of these used to need a frontier API subscription, and the combination used to need a 24 GB rig. Now it's an old laptop in the corner.

## Honest weak spots

Local inference is not a complete substitute for frontier cloud models. It is a substitute for *most* of what you do with them.

A 35B-A3B at Q4 will fall behind on multi-file architectural reasoning or subtle bug hunts that need long chain-of-thought. The gap is real; it just isn't the gap that matters for everyday work.

Q4_K_M is a quality compromise. Q5_K_M and Q6_K are slightly better at slightly lower throughput, but neither fits cleanly on 6 GB at 128K even with turbo3/4 KV.

Prompt eval stays the visible weak spot. Mental model: this is a daily driver for ~70 % of tasks. Keep cloud frontier models for the hard 30 %.

## What I run, in case it helps

- **Engine:** llama.cpp Turboquant fork @ `4595fff`
- **Model:** Qwen3.6 35B-A3B at Q4_K_M (`lmstudio-community/Qwen3.6-35B-A3B-GGUF`) with BF16 mmproj
- **Runtime:** Docker on `nvidia/cuda:12.5.1-devel-ubuntu24.04`, fork built on host
- **Coding agent:** opencode wired to `localhost:8080`
- **Chat UI:** the fork's `--ui-mcp-proxy` web frontend
- **Web research:** `local-deep-research` MCP server
- **Bench:** ten runs of `bench.sh`, two warmup discarded, server-side timings

Two repos. Main project (Docker setup, run script, build script, benchmark, methodology): **[github.com/Arthamu/paperweight-llm](https://github.com/Arthamu/paperweight-llm)**. MCP server for live web research: **[github.com/Arthamu/local-deep-research](https://github.com/Arthamu/local-deep-research)**.

If you reproduce this on different hardware, please open an issue with your numbers, `nvidia-smi`, `lscpu`, `free -h`, and the fork commit. I'm curious how the CPU bottleneck moves on a modern desktop chip.
