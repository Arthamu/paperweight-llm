#!/usr/bin/env bash
set -e

MCP_DIR="$(cd "$(dirname "$0")" && pwd)"

# Load .env if present
if [ -f "$MCP_DIR/.env" ]; then
  set -a
  source "$MCP_DIR/.env"
  set +a
fi

SERPER_API_KEY="${SERPER_API_KEY:?SERPER_API_KEY is required}"
LLAMA_ENDPOINT="${LLAMA_ENDPOINT:-http://localhost:8080}"
MCP_PORT="${MCP_PORT:-3001}"
MCP_HOST="${MCP_HOST:-localhost}"
MCP_URL="http://${MCP_HOST}:${MCP_PORT}"

echo "========================================="
echo " Starting local-deep-research MCP bridge"
echo "========================================="
echo "  LLM backend: ${LLAMA_ENDPOINT}"
echo "  MCP bridge:  ${MCP_URL}/sse"
echo ""

# Cleanup previous instance
pkill -f "supergateway.*dist/index" 2>/dev/null || true
sleep 1

# Start MCP bridge (stdio -> Streamable HTTP)
echo "Starting MCP bridge on port ${MCP_PORT}..."
npx -y supergateway \
  --stdio "node $MCP_DIR/dist/index.js" \
  --port "$MCP_PORT" \
  --baseUrl "$MCP_URL" \
  --outputTransport streamableHttp \
  --streamableHttpPath /sse \
  --stateful \
  --cors \
  > /tmp/local-deep-research.log 2>&1 &

echo "MCP bridge PID: $!"
echo "Logs: tail -f /tmp/local-deep-research.log"
echo ""
echo "Connect your MCP client to:"
echo "  URL: ${MCP_URL}/sse"
