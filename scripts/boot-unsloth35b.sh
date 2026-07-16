#!/usr/bin/env bash
# Boot unsloth 35B Fast (NVFP4). Default image: eugr/spark-vllm (vLLM main, has b12x kernels).
# Usage: boot-unsloth35b-v025.sh [gpu_util] [max_seqs] [moe_backend]   defaults: 0.65 8 auto
#
# Env hooks:
#   IMAGE            container image (default eugr/spark-vllm:latest)
#   NAME             container name (default unsloth35b-v025)
#   LINEAR_BACKEND   adds --linear-backend <val>
#   SPEC_JSON        full --speculative-config JSON override
#   SPEC_K           MTP speculative tokens (default 3; ignored if SPEC_JSON set)
#   SPEC_MOE         draft-head MoE backend (e.g. triton) — REQUIRED when MOE=flashinfer_b12x,
#                    because the MTP draft head is UNQUANTIZED and b12x has no kernel for it
#                    ("moe_backend='flashinfer_b12x' is not supported for unquantized MoE.
#                     Expected one of ['triton','flashinfer_trtllm','flashinfer_cutlass','aiter']")
#   SPEC_ATTN        draft-head attention backend (e.g. TRITON_ATTN)
#   NO_SPEC=1        disable speculative decoding entirely
#   U35B_MAX_PIXELS  lower per-image resolution (multi-image mode)
#   U35B_IMG_LIMIT   images per prompt (default 8)
#   EXTRA_ENV        extra "-e K=V" docker args
set -e
UTIL="${1:-0.65}"
SEQS="${2:-8}"
MOE="${3:-auto}"
IMAGE="${IMAGE:-eugr/spark-vllm:latest}"
NAME="${NAME:-unsloth35b-v025}"
IMG_LIMIT="${U35B_IMG_LIMIT:-8}"

# ---- speculative config -------------------------------------------------
SPEC_ARGS=()
if [ -z "${NO_SPEC:-}" ]; then
  if [ -n "${SPEC_JSON:-}" ]; then
    SPEC="$SPEC_JSON"
  else
    SPEC="{\"method\":\"mtp\",\"num_speculative_tokens\":${SPEC_K:-3}"
    [ -n "${SPEC_MOE:-}" ]  && SPEC="${SPEC},\"moe_backend\":\"${SPEC_MOE}\""
    [ -n "${SPEC_ATTN:-}" ] && SPEC="${SPEC},\"attention_backend\":\"${SPEC_ATTN}\""
    SPEC="${SPEC}}"
  fi
  SPEC_ARGS=(--speculative-config "$SPEC")
  echo "Speculative config: $SPEC"
else
  echo "Speculative decoding DISABLED (NO_SPEC=1)"
fi

# ---- multimodal ---------------------------------------------------------
MM_ARGS=(--limit-mm-per-prompt "{\"image\":${IMG_LIMIT},\"video\":2}")
if [ -n "${U35B_MAX_PIXELS:-}" ]; then
  MM_ARGS+=(--mm-processor-kwargs "{\"max_pixels\":${U35B_MAX_PIXELS}}")
  echo "Multi-image mode: max_pixels=${U35B_MAX_PIXELS} image_limit=${IMG_LIMIT}"
fi

EXTRA_ARGS=()
if [ -n "${LINEAR_BACKEND:-}" ]; then
  EXTRA_ARGS+=(--linear-backend "$LINEAR_BACKEND")
  echo "Linear backend forced: ${LINEAR_BACKEND}"
fi

docker update --restart=no "$NAME" 2>/dev/null || true
docker stop -t 30 "$NAME" 2>/dev/null || true
docker rm -f "$NAME" 2>/dev/null || true
for i in $(seq 1 24); do
  avail=$(awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo)
  [ "$avail" -ge 85 ] && break
  sleep 5
done
echo "MemAvailable before boot: ${avail} GiB"
if [ "$avail" -lt 60 ]; then
  echo "ABORT: unified memory not released (${avail} GiB) — run clear-ram, retry, then reboot."
  exit 2
fi

docker run -d --name "$NAME" --gpus all --ipc=host --net=host \
  -e CUTE_DSL_ARCH=sm_121a \
  ${EXTRA_ENV:-} \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -v ~/vllm-cache:/root/.cache/vllm \
  --entrypoint vllm "$IMAGE" \
  serve unsloth/Qwen3.6-35B-A3B-NVFP4-Fast \
  --served-model-name unsloth35b aeon-fast aeon-deep \
  --host 0.0.0.0 --port 8888 \
  --trust-remote-code \
  --kv-cache-dtype fp8 \
  --moe-backend "$MOE" \
  --gpu-memory-utilization "$UTIL" \
  --max-model-len 262144 \
  --max-num-seqs "$SEQS" \
  --max-num-batched-tokens 32768 \
  --enable-chunked-prefill \
  --async-scheduling \
  --enable-prefix-caching \
  "${SPEC_ARGS[@]}" \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen3_xml \
  --enable-auto-tool-choice \
  "${EXTRA_ARGS[@]}" \
  "${MM_ARGS[@]}"

echo "$NAME starting (image=$IMAGE util=$UTIL seqs=$SEQS moe=$MOE imgs=$IMG_LIMIT); waiting for health"
for i in $(seq 1 180); do
  curl -sf -m 3 http://127.0.0.1:8888/health >/dev/null 2>&1 && { echo "LLM ready"; exit 0; }
  docker ps --format '{{.Names}}' | grep -qx "$NAME" || {
    echo "CONTAINER DIED during startup — last logs:"; docker logs --tail 25 "$NAME" 2>&1; exit 3; }
  sleep 10
done
echo "LLM health check timed out"; exit 1
