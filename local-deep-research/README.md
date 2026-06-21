# local-deep-research

MCP server that runs multi-step web research using **local LLMs** (llama.cpp).

Powers three MCP tools:
- `google_search` — search the web via Serper API
- `google_scrape` — scrape & clean webpage content (via crawl4ai)
- `deep_research` — orchestrate searches, scrapes, and LLM synthesis into a structured report

## Prerequisites

- Node.js >= 20
- [llama.cpp server](https://github.com/ggml-org/llama.cpp) running (e.g. `llama-server -m model.gguf --port 8080`)
- [Serper API key](https://serper.dev) (free tier: 2500 searches/month)
- [crawl4ai](https://github.com/unclecode/crawl4ai) (optional, for web scraping — falls back gracefully)

## Setup

```bash
cd local-deep-research

# Install deps
npm install

# Build TypeScript
npm run build

# Configure
cp .env.example .env
# Edit .env with your Serper API key and LLM endpoint
```

## Usage

### Option A: Direct MCP (stdio)

Add to your MCP client config:

```json
{
  "mcpServers": {
    "local-deep-research": {
      "command": "node",
      "args": ["/path/to/paperweight-llm/local-deep-research/dist/index.js"],
      "env": {
        "SERPER_API_KEY": "your-key",
        "LLAMA_ENDPOINT": "http://localhost:8080"
      }
    }
  }
}
```

### Option B: HTTP bridge via supergateway

```bash
# Start the HTTP/SSE bridge
SERPER_API_KEY=your-key LLAMA_ENDPOINT=http://localhost:8080 ./start-all.sh
```

This exposes the MCP server at `http://localhost:3001/sse`.

## Integration with Llama Chat UI

[Llama Chat UI](https://github.com/ggml-org/llama-cpp-ui) is a minimal chat interface for llama.cpp. To connect:

1. **Start your llama.cpp server** (if not already running):
   ```bash
   llama-server -m /path/to/model.gguf --port 8080 --ctx-size 8192
   ```

2. **Start the MCP HTTP bridge**:
   ```bash
   SERPER_API_KEY=your-key LLAMA_ENDPOINT=http://localhost:8080 ./start-all.sh
   ```

3. **Configure Llama Chat UI** to use the MCP tools:
   - Open Llama Chat UI settings
   - Under **MCP Servers**, add:
     ```
     Name: local-deep-research
     URL: http://localhost:3001/sse
     ```
   - The three tools (`google_search`, `google_scrape`, `deep_research`) will appear as available tools

4. **Or use any MCP-compatible client** (Open WebUI, Continue.dev, Claude Desktop, etc.) by pointing it to the SSE endpoint or stdio command.

## Environment Variables

| Variable | Default | Required | Description |
|---|---|---|---|
| `SERPER_API_KEY` | — | Yes | Serper.dev API key for Google Search |
| `LLAMA_ENDPOINT` | `http://localhost:8080` | No | llama.cpp server URL |
| `LLAMA_MODEL` | `""` | No | Optional model name to pass to the LLM |
| `CRAWL4AI_ENDPOINT` | `http://localhost:11235` | No | crawl4ai server URL |
