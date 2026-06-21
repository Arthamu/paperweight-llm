#!/usr/bin/env bash
set -e

# Set your Serper API key here
export SERPER_API_KEY="${SERPER_API_KEY:-YOUR_SERPER_API_KEY}"
export LLAMA_ENDPOINT="${LLAMA_ENDPOINT:-http://localhost:8080}"

PORT="${PORT:-3001}"

echo "Starting local-deep-research bridge on port $PORT..."
echo "  SERPER_API_KEY: ${SERPER_API_KEY:0:8}..."
echo "  LLAMA_ENDPOINT: $LLAMA_ENDPOINT"
echo ""
echo "  SSE endpoint:  http://localhost:$PORT/sse"
echo "  Message URL:   http://localhost:$PORT/message"
echo ""

exec npx -y supergateway \
  --stdio "node $(dirname "$0")/dist/index.js" \
  --port "$PORT" \
  --baseUrl "http://localhost:$PORT" \
  --ssePath /sse \
  --messagePath /message
