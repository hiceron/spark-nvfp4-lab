# Deploy — copy-paste recipes for a DGX Spark (GB10)

Everything here runs on a stock Spark with Docker. No custom builds, no patches. Weights
download from HuggingFace on first boot (20–25 GB per model — the `-v` cache mounts make
that a one-time cost).

**Universal rules first** (each learned the hard way — see [TROUBLESHOOTING.md](TROUBLESHOOTING.md)):

1. **Never set `--gpu-memory-utilization` above 0.80.** The OS lives in the same 121 GiB.
2. **Between model switches:** stop the old container, then `sync; echo 3 | sudo tee /proc/sys/vm/drop_caches`, then boot. Skipping this = "unified memory not released" or worse.
3. **After every boot, send one real chat request.** `/health` can pass over a dead engine.
4. **If speculative decoding is on, `util` × `max-num-seqs` is a joint budget** — the drafter's per-conversation scratch memory is invisible to `gpu-memory-utilization`. U=0.80 + N=4 DFlash took our whole machine down.

---

## Stack A — unsloth NVFP4 (censored) on vLLM main

### 35B-A3B daily driver — 89–95 tok/s single, 926 aggregate @ c=64, 10.65× 262k KV

```bash
docker pull eugr/spark-vllm:latest
mkdir -p ~/vllm-cache

docker run -d --name unsloth35b --gpus all --ipc=host --net=host \
  -e CUTE_DSL_ARCH=sm_121a \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -v ~/vllm-cache:/root/.cache/vllm \
  --entrypoint vllm eugr/spark-vllm:latest \
  serve unsloth/Qwen3.6-35B-A3B-NVFP4-Fast \
  --served-model-name unsloth35b \
  --host 0.0.0.0 --port 8888 --trust-remote-code \
  --kv-cache-dtype fp8 --moe-backend auto \
  --gpu-memory-utilization 0.55 --max-model-len 262144 \
  --max-num-seqs 8 --max-num-batched-tokens 32768 \
  --enable-chunked-prefill --async-scheduling --enable-prefix-caching \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
  --reasoning-parser qwen3 --tool-call-parser qwen3_xml --enable-auto-tool-choice
```

Tuning: `--max-num-seqs` is the parallel-request ceiling (we validated up to 64; per-request
speed at 64 is ~15 tok/s). Raise util to 0.65 for ~21× KV; decode speed does not change.
**Leave `--moe-backend auto`** — we measured the b12x path at −11% decode and a third of the
KV on this box (see [FINDINGS.md](FINDINGS.md) §1; the b12x+MTP recipe is there if you want
to reproduce the measurement).

### 27B dense coder — 33.5–35.5 tok/s via DFlash drafter graft

Uses AEON's DFlash drafter (hidden size and vocab match unsloth's 27B). Clone the drafter
once from [AEON-7/Qwen3.6-27B-AEON-Ultimate-Uncensored-DFlash](https://github.com/AEON-7/Qwen3.6-27B-AEON-Ultimate-Uncensored-DFlash)
(the `models/dflash-drafter` directory), then:

```bash
docker run -d --name unsloth27b --gpus all --ipc=host --net=host \
  -e CUTE_DSL_ARCH=sm_121a \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -v ~/vllm-cache:/root/.cache/vllm \
  -v /path/to/dflash-drafter:/models/dflash-drafter:ro \
  --entrypoint vllm eugr/spark-vllm:latest \
  serve unsloth/Qwen3.6-27B-NVFP4 \
  --served-model-name unsloth27b \
  --host 0.0.0.0 --port 8888 --trust-remote-code \
  --kv-cache-dtype fp8 \
  --gpu-memory-utilization 0.74 --max-model-len 262144 \
  --max-num-seqs 4 --max-num-batched-tokens 32768 \
  --enable-chunked-prefill --async-scheduling --enable-prefix-caching \
  --speculative-config '{"method":"dflash","model":"/models/dflash-drafter","num_speculative_tokens":10}' \
  --reasoning-parser qwen3 --tool-call-parser qwen3_xml --enable-auto-tool-choice
```

U=0.74 / N=4 gives 4.24× full-262k KV. Know the prefill bill: ~30 s to first token on a 32k
prompt (dense + kernel-bound). We keep it for coding quality; everything else runs the 35B.

---

## Stack B — AEON uncensored on aeon-vllm-ultimate

Use image `ghcr.io/aeon-7/aeon-vllm-ultimate:2026-07-14-v0.25.0` — we A/B'd it against
0.24 on all three checkpoints: +22% decode on the 27B MTP, +8.5% on heretic, neutral on
ornith ([FINDINGS.md](FINDINGS.md) §4).

Non-negotiables for this stack:
- **`--kv-cache-dtype auto` (BF16).** fp8 KV boots, passes health, and dies on the first
  request — these checkpoints ship no fp8 calibration scales.
- **Leave kernel selection alone.** It picks Marlin; forcing Cutlass is rejected (their
  quant scheme is Marlin-only — measured, not assumed).
- **Restart nightly.** DFlash acceptance decays over ~20 h; the engine dies ~23 h. The
  restart must be stop → `drop_caches` → start (never `docker restart` — ghost memory).

### Ornith / heretic 35B (U=0.55, N=4 → ~2.7–2.9× 262k KV)

```bash
docker run -d --name heretic --gpus all --ipc=host --net=host \
  -e CUTE_DSL_ARCH=sm_121a \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -v ~/vllm-cache:/root/.cache/vllm \
  --entrypoint vllm ghcr.io/aeon-7/aeon-vllm-ultimate:2026-07-14-v0.25.0 \
  serve AEON-7/Qwen3.6-35B-A3B-heretic-NVFP4 \
  --host 0.0.0.0 --port 8888 --trust-remote-code \
  --quantization compressed-tensors --kv-cache-dtype auto \
  --gpu-memory-utilization 0.55 --max-model-len 262144 \
  --max-num-seqs 4 --max-num-batched-tokens 16384 \
  --enable-chunked-prefill --enable-prefix-caching \
  --speculative-config '{"method":"dflash","model":"AEON-7/AEON-DFlash-Qwen3.6-35B-A3B","num_speculative_tokens":6}' \
  --reasoning-parser qwen3 --tool-call-parser qwen3_coder --enable-auto-tool-choice
```

(For Ornith swap the model to `AEON-7/Ornith-1.0-35B-AEON-Ultimate-Uncensored-NVFP4` and
add `--mamba-cache-dtype float32`.)

### AEON 27B MTP — ⚠️ sizing matters, this one bites

Needs **27.14 GiB KV per full-262k conversation** (~5× the 35Bs). The only safe recipe we
validated: **U=0.70, N=2** → 1.47× KV, 30.4 tok/s. U=0.55 refuses to boot; **U=0.80 + N=4
hard-locked the machine** (uncounted DFlash buffers). See rule 4 at the top.

---

## Benchmark it yourself

```bash
python3 scripts/concurrency_bench.py <served-model-name> --levels 1,8 --max-tokens 256 --reps 3
python3 scripts/concurrency_bench.py <served-model-name> --prompt-tokens 32768 --max-tokens 1 --levels 1 --reps 2
```

Read [METHODOLOGY.md](METHODOLOGY.md) first — this tool streams and reads ~15–25% lower than
non-streaming benches. Compare only against itself.
