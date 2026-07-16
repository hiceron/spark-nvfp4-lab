# Methodology — and the mistakes

Benchmark numbers on the internet are mostly unfalsifiable: one config, one run, no control,
no failure reported. This documents how these were taken, including where the method failed.

---

## Rules

**One variable.** Same boot script, same bench, same box, same session. Image A/Bs swap only
`IMAGE=`. If the script differs, the comparison is void.

**Never trust `/health`.** It's a different process from the engine (see
[TROUBLESHOOTING.md](TROUBLESHOOTING.md#3-zombie-boots-health-lies)). Every number here comes
from an engine that answered a real completion first.

**Reps until the spread is smaller than the claim.** Two reps of ornith disagreed by 15%
(70.8 vs 61.7 tok/s) — that can't resolve a 4% delta. Where a difference is claimed here, the
ranges **don't overlap**.

**Prefill and decode are different questions.** Decode is bandwidth-bound; prefill is
compute-bound. A decode-only bench cannot tell you whether a kernel change helped prefill.
We nearly published exactly that error.

**Defeat the prefix cache.** Every request gets a unique random leading tag. Agent workloads
hit 86% prefix-cache hits in the wild — great for production, fatal for a prefill benchmark.

---

## What's measured

| Metric | How |
|---|---|
| Prefill (pp8192 / pp32768) | Filler prompt calibrated against `usage.prompt_tokens`, `max_tokens=1`. Σprompt_tokens ÷ wall |
| Decode | ~50-token prompt, `max_tokens=256`, `ignore_eos=True`. Σcompletion_tokens ÷ wall |
| TTFT | Streamed; first content-bearing chunk |
| KV pool | vLLM's own `Maximum concurrency for 262,144 tokens per request` boot line |
| Spec acceptance | `vllm:spec_decode_num_accepted_tokens_total ÷ ..._draft_tokens_total` |

`scripts/concurrency_bench.py <model> [--prompt-tokens N] [--max-tokens N] [--levels 1,8] [--reps 3]`

---

## ⚠️ These numbers are not comparable to other people's

Our bench **streams** tokens; most don't. Streaming costs per-token overhead.

**Proof, on one unchanged image:** ornith measures **68.6–71.9 tok/s** with this script and
**92.7** with our older non-streaming script. Same image, same model, same box, same day.
A ~25% method gap.

So: same-script comparisons are valid; cross-tool ones are not. If you reproduce these,
reproduce the *script*, not the number. This is also why we don't claim to beat anyone's
published figures — we can't, from here.

---

## Two conclusions we got wrong

Both were caught by controls. Both would have been published as fact.

### "The new AEON image is 15–20% slower"

Ran the 35B on AEON 0.25 with the new streaming bench: 75.9/78.5 tok/s. Compared it against
the documented eugr baseline of 90–98. Conclusion: new image is slower. **Wrong** — those
baselines came from a *different script*. Running eugr through the *same* bench: 75.2/83.3.
**The images were identical.** The 20% was entirely measurement method.

*Lesson: never compare a fresh measurement to a stored number taken by different tooling.*

### "b12x is dead on GB10 — the sm_121 gate blocks it"

Two failures said so:
- `--linear-backend flashinfer_b12x` → *no b12x kernel exists for this layer type*
- `--moe-backend flashinfer_cutedsl` → *kernel does not support current device cuda*

Concluded the gate was unfixable — and even blamed the vLLM version string
(`0.23.1rc1.dev1043`, "too old for the ≥0.25.0 unsloth requires"). **Wrong on both counts.**
vLLM's own probe returned `b12x gemm True | b12x moe True` on that exact image; the version
string is a stale-git-tag artifact and says nothing about capability. The real blocker was
the **unquantized MTP draft head**, and the fix was `SpeculativeConfig.moe_backend` — which
[r0b0tlab](https://github.com/r0b0tlab/qwen36-35b-a3b-nvfp4-sm121-vllm) had implied all along
with "Triton draft MoE".

*Lesson: probe capability, don't read version strings. And when someone's config disagrees
with your conclusion, they're probably right.*

---

## Replicating

```bash
# 1. Does your box even have b12x?
python -c "import torch; from vllm.utils.flashinfer import has_flashinfer_b12x_gemm as g, has_flashinfer_b12x_moe as m
cap = torch.cuda.get_device_capability(); print('cap', cap, '| b12x gemm', g(), '| b12x moe', m())"

# 2. Baseline: auto + MTP k=3
IMAGE=eugr/spark-vllm:latest bash scripts/boot-unsloth35b.sh 0.55 8 auto
python3 scripts/concurrency_bench.py unsloth35b --prompt-tokens 8192  --max-tokens 1 --levels 1 --reps 2
python3 scripts/concurrency_bench.py unsloth35b --prompt-tokens 32768 --max-tokens 1 --levels 1 --reps 2
python3 scripts/concurrency_bench.py unsloth35b --levels 1,8 --max-tokens 256 --reps 3

# 3. b12x + Triton draft MoE — same everything else
SPEC_K=2 SPEC_MOE=triton bash scripts/boot-unsloth35b.sh 0.55 8 flashinfer_b12x
# ...same three bench lines...
```

Between every config: **stop the running engine**, `drop_caches`, confirm ≥100 GiB free.
Skip that and you're benchmarking memory pressure, or hard-locking the box.

## Environment

GB10 (sm_121) · 121 GiB unified LPDDR5X · ~273 GB/s · driver 595.71.05 ·
`eugr/spark-vllm:latest` (vLLM main @ 2026-07-11, FlashInfer 0.6.15, nvidia-cutlass-dsl 4.5.2)
· `ghcr.io/aeon-7/aeon-vllm-ultimate:2026-07-01-v0.24.0` and `:2026-07-14-v0.25.0`.
