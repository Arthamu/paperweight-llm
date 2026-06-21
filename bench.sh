#!/bin/bash
# bench.sh — honest benchmark for local Qwen3.6-35B-A3B on llama-server
#
# Measures: generation tok/s, prompt-eval tok/s, time-to-first-token (TTFT)
# Usage:    ./bench.sh [host:port]
# Default:  http://localhost:8080
#
# Requires: curl, jq, awk

set -euo pipefail

ENDPOINT="${1:-http://localhost:8080}"
MODEL="local"
RUNS=10
WARMUP=2
GEN_TOKENS=512

# 512-token-ish system prompt (deliberately a chunk of code+prose, not lorem)
SYS_PROMPT=$(cat <<'EOF'
You are a precise senior software engineer. Answer in plain prose only.
No markdown formatting, no bullet points, no code fences in your reply.
Keep responses focused. Do not repeat the question.

Context: the user is benchmarking a local LLM server. They want measured,
deterministic generation. You will be asked the same short question
multiple times. Answer it the same way each time, in roughly the same
length, so token-per-second measurements are stable across runs.

Your style: direct, technical, no filler, no apologies, no preamble.
Begin the answer immediately with the substantive content.
EOF
)

USER_Q="Explain in roughly 400 words what a Mixture-of-Experts model is, why active parameters differ from total parameters, and why this matters for inference on consumer GPUs. Be specific."

echo "==> Endpoint:    $ENDPOINT"
echo "==> Model alias: $MODEL"
echo "==> Runs:        $RUNS (first $WARMUP discarded as warmup)"
echo "==> Gen tokens:  $GEN_TOKENS, greedy (temp=0)"
echo

# Sanity check
if ! curl -sf "$ENDPOINT/v1/models" >/dev/null; then
  echo "ERROR: server not reachable at $ENDPOINT" >&2
  exit 1
fi

PAYLOAD=$(jq -n \
  --arg model "$MODEL" \
  --arg sys "$SYS_PROMPT" \
  --arg usr "$USER_Q" \
  --argjson maxtok "$GEN_TOKENS" \
  '{
    model: $model,
    messages: [
      {role: "system", content: $sys},
      {role: "user",   content: $usr}
    ],
    max_tokens: $maxtok,
    temperature: 0,
    stream: false
  }')

declare -a GEN_TPS
declare -a PP_TPS
declare -a TTFT

for i in $(seq 1 "$RUNS"); do
  printf "Run %2d/%2d  ... " "$i" "$RUNS"

  T0=$(date +%s.%N)
  RESP=$(curl -sf "$ENDPOINT/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d "$PAYLOAD")
  T1=$(date +%s.%N)

  # llama.cpp returns timings in `timings` (non-standard) — fall back to usage if absent
  PP_MS=$(echo "$RESP"     | jq -r '.timings.prompt_ms     // empty')
  GEN_MS=$(echo "$RESP"    | jq -r '.timings.predicted_ms  // empty')
  PP_TOK=$(echo "$RESP"    | jq -r '.timings.prompt_n      // .usage.prompt_tokens     // 0')
  GEN_TOK=$(echo "$RESP"   | jq -r '.timings.predicted_n   // .usage.completion_tokens // 0')

  if [[ -z "$PP_MS" || -z "$GEN_MS" ]]; then
    # Fall back to wall-clock approximation
    WALL=$(awk -v a="$T0" -v b="$T1" 'BEGIN{printf "%.3f", b-a}')
    GEN_TPS_I=$(awk -v t="$GEN_TOK" -v s="$WALL" 'BEGIN{printf "%.2f", t/s}')
    PP_TPS_I="n/a"
    TTFT_I="n/a"
  else
    GEN_TPS_I=$(awk -v t="$GEN_TOK" -v ms="$GEN_MS" 'BEGIN{printf "%.2f", t*1000/ms}')
    PP_TPS_I=$(awk  -v t="$PP_TOK"  -v ms="$PP_MS"  'BEGIN{printf "%.2f", t*1000/ms}')
    TTFT_I=$(awk    -v ms="$PP_MS"                  'BEGIN{printf "%.3f", ms/1000}')
  fi

  printf "gen=%6s tok/s   pp=%7s tok/s   ttft=%6ss   gen_tok=%d\n" \
    "$GEN_TPS_I" "$PP_TPS_I" "$TTFT_I" "$GEN_TOK"

  if (( i > WARMUP )); then
    GEN_TPS+=("$GEN_TPS_I")
    [[ "$PP_TPS_I" != "n/a" ]] && PP_TPS+=("$PP_TPS_I")
    [[ "$TTFT_I"   != "n/a" ]] && TTFT+=("$TTFT_I")
  fi
done

echo
echo "===== Results (excluding $WARMUP warmup runs) ====="

stat() {
  local label=$1; shift
  local vals=("$@")
  [[ ${#vals[@]} -eq 0 ]] && { printf "%-22s n/a\n" "$label"; return; }
  printf '%s\n' "${vals[@]}" | awk -v label="$label" '
    { v[NR]=$1; sum+=$1; if(NR==1||$1<min)min=$1; if(NR==1||$1>max)max=$1 }
    END {
      n=NR; mean=sum/n
      asort(v)
      median = (n%2) ? v[(n+1)/2] : (v[n/2]+v[n/2+1])/2
      printf "%-22s min=%.2f  median=%.2f  mean=%.2f  max=%.2f  (n=%d)\n",
        label, min, median, mean, max, n
    }'
}

stat "Generation tok/s"  "${GEN_TPS[@]}"
stat "Prompt eval tok/s" "${PP_TPS[@]}"
stat "TTFT (seconds)"    "${TTFT[@]}"

echo
echo "Hardware snapshot:"
nvidia-smi --query-gpu=name,memory.used,memory.total,power.draw,utilization.gpu \
  --format=csv,noheader 2>/dev/null || echo "  nvidia-smi unavailable"
echo
echo "Done."
