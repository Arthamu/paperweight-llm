# Pointing a coding agent at your local llama-server

This guide wires a coding agent into the local Qwen3.6 35B-A3B server running on
`http://localhost:8080` (started by `run-server.sh`).

The llama-server in the Turboquant fork exposes an **OpenAI-compatible** API plus, with
`--ui-mcp-proxy`, an **MCP-compatible** endpoint. Most coding agents can consume one or both.

The demo video uses **opencode**. See "Why opencode and not Claude Code" below for the
honest answer.

---

## Why opencode and not Claude Code

Claude Code's default system prompt is large — by the time it loads, the usable context the
local 35B has for tool use, file reads, and reasoning is meaningfully smaller than what the
agent is designed for. The model doesn't fail; it just runs short.

opencode ships a leaner system prompt that leaves more room for the model to do work.
Empirically, on this hardware with this model, opencode produces longer, more useful tool-use
chains before the context budget runs out.

The wiring (the `--ui-mcp-proxy` endpoint, the OpenAI-compat path) is identical for both
clients. The choice is about prompt-size economics on a 35B-class local model, not capability.

If you have a frontier-class local model with a much larger effective context (a 70B+ at
Q5/Q6, or hardware that lets you run with `q8_0` KV at long context), Claude Code works fine.

---

## Path A — opencode (the demo client)

opencode reads from a project-local or user-level config. Point its model at the local
endpoint:

`~/.config/opencode/opencode.json` (path may vary by version):

    {
      "providers": {
        "local": {
          "type": "openai-compatible",
          "baseUrl": "http://localhost:8080/v1",
          "apiKey": "local-no-auth"
        }
      },
      "models": {
        "local": {
          "provider": "local",
          "model": "local"
        }
      },
      "defaultModel": "local"
    }

> The API key is required by the client but is ignored by llama-server. Any non-empty string
> works.

Verify the server is reachable first:

    curl http://localhost:8080/v1/models
    # → {"object":"list","data":[{"id":"local",...}]}

The `"id":"local"` matches the `--alias local` flag in `run-server.sh`.

Then start opencode in your repo and confirm tokens stream at ~25 tok/s with the GPU fan
spinning up.

---

## Path B — Claude Code (works, with the prompt-size caveat above)

Claude Code redirects to any OpenAI-compatible base URL via two env vars.

Add to your shell rc (`~/.bashrc`, `~/.zshrc`):

    export ANTHROPIC_BASE_URL="http://localhost:8080/v1"
    export ANTHROPIC_API_KEY="local-no-auth"
    export ANTHROPIC_MODEL="local"

Reload and run:

    source ~/.zshrc
    claude-code "write a python script that prints fibonacci numbers"

If tool-use chains feel cut short or the model loses track of file context, that's the
prompt-size issue, not a wiring problem. Switch to opencode.

---

## Path C — MCP proxy endpoint (`--ui-mcp-proxy`)

The `--ui-mcp-proxy` flag exposes a Model Context Protocol endpoint at:

    http://localhost:8080/mcp

Useful if your client consumes MCP servers natively via `mcp.json` or similar:

    {
      "mcpServers": {
        "local-qwen": {
          "transport": "http",
          "url": "http://localhost:8080/mcp",
          "description": "Local Qwen3.6 35B-A3B via llama-server"
        }
      }
    }

If your client is stdio-only MCP, run `mcp-proxy` (the npm package) in front to bridge.

---

## Tuning for a local backend

Local generation is fast but not infinite. A few defaults that help:

| Setting | Recommended | Why |
|---|---|---|
| `max_tokens` per response | 2048 | Long enough for real edits, short enough to stay snappy |
| Temperature | 0.2 | Deterministic enough for code |
| Streaming | **on** | You want every token visible immediately |
| Auto-context | trim aggressively | 128K is available but loading it costs prompt-eval time (~43 tok/s) |
| Tool-use loops | cap at 6–8 turns | Local model is less self-correcting than a frontier model |

---

## What works well, what doesn't

**Works well on this setup**
- Single-file edits, refactors, doc generation
- Reading screenshots / UI mockups (multimodal via `--mmproj`)
- Bash / Python / Go / TypeScript boilerplate
- Long-context Q&A on private repos (within the prompt-eval budget)

**Falls behind a frontier model on**
- Multi-file architectural reasoning across 50+ files
- Subtle bug hunts that need strong chain-of-thought
- Anything where you'd notice a frontier-model jump in IQ

The right mental model: this is your daily-driver for 70 % of tasks. Keep a frontier model
for the hard 30 %.

---

## Troubleshooting

**`curl /v1/models` works but the client hangs**
Some clients send an `anthropic-version` header. Some llama-server builds reject unknown
headers. Check `docker logs qwen-server`.

**Generation is slow (<10 tok/s)**
Check `nvidia-smi`. If GPU utilization is <30 % during generation, the model fell back to
CPU. Likely cause: `-ngl 99` didn't take. Re-check the startup log for "offloaded N/N
layers to GPU".

**Client complains about context length**
Tokenizer mismatch. Lower the auto-context ceiling to ~100K to leave headroom.

**MCP path returns 404**
The fork's `--ui-mcp-proxy` may mount under `/api/mcp` or `/v1/mcp` instead of `/mcp`. Run
`curl -s http://localhost:8080/ | grep -i mcp` to find the real path, or check the server
startup log.

**Tool-use chains feel short or the model "forgets" earlier files**
The agent's system prompt is consuming too much of the context budget. Try a leaner client
(opencode is the example above) or trim the agent's prompt.
