#!/usr/bin/env bash
# launches the qwen3-14b baseline vllm server that llm.py talks to
# refuses to start if the port is taken or the gpu doesn't have room, so it never disturbs another server
set -euo pipefail

VLLM_BIN=${VLLM_BIN:-$HOME/llm-behavior-ci/.venv-vllm/bin/vllm}
LLM_SERVE_MODEL=${LLM_SERVE_MODEL:-Qwen/Qwen3-14B-AWQ}
LLM_PORT=${LLM_PORT:-8001}
LLM_MAX_CONCURRENCY=${LLM_MAX_CONCURRENCY:-8}
LLM_MAX_MODEL_LEN=${LLM_MAX_MODEL_LEN:-4096}
LLM_GPU_UTIL=${LLM_GPU_UTIL:-0.80}
# extra vllm flags, e.g. compilation workarounds
LLM_EXTRA_ARGS=${LLM_EXTRA_ARGS:-}

# blackwell gpus segfault on the first forward pass without these
export VLLM_USE_FLASHINFER_SAMPLER=${VLLM_USE_FLASHINFER_SAMPLER:-0}
export TRITON_PTXAS_BLACKWELL_PATH=${TRITON_PTXAS_BLACKWELL_PATH:-$(dirname "$(dirname "$VLLM_BIN")")/lib/python3.12/site-packages/nvidia/cuda_nvcc/bin/ptxas}

if ss -ltn | grep -q ":$LLM_PORT "; then
    echo "port $LLM_PORT is already in use"
    exit 1
fi

read -r total free <<< "$(nvidia-smi --query-gpu=memory.total,memory.free --format=csv,noheader,nounits | head -1 | tr -d ',')"
needed=$(awk "BEGIN {printf \"%d\", $total * $LLM_GPU_UTIL}")
if (( free < needed )); then
    echo "not enough free gpu memory: ${free}MiB free, ${needed}MiB needed for --gpu-memory-utilization $LLM_GPU_UTIL"
    exit 1
fi

exec "$VLLM_BIN" serve "$LLM_SERVE_MODEL" \
    --host 127.0.0.1 \
    --port "$LLM_PORT" \
    --max-model-len "$LLM_MAX_MODEL_LEN" \
    --max-num-seqs "$LLM_MAX_CONCURRENCY" \
    --gpu-memory-utilization "$LLM_GPU_UTIL" \
    --enable-prefix-caching \
    $LLM_EXTRA_ARGS
