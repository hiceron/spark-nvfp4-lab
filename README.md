# DGX Spark (GB10 / sm_121) — NVFP4 serving: what we actually measured

One DGX Spark. Two weeks of chasing a **1.79×** speedup claim. This repo is the receipts:
the recipe that finally made the fast kernels engage, the measurements showing they don't
help *on this hardware*, the prefill curves nobody publishes, and the operational failure
modes that took the box down three times.

**Everything here is measured on one machine** — GB10, 121 GiB unified memory, driver
595.71.05 — with the same script, the same bench, one variable changed at a time.

---

## Standing on other people's work

This repo exists because of four people who published first. Where our numbers disagree with
theirs, it's hardware/model context — not a claim that they're wrong.

| Source | What we took from it |
|---|---|
| **[unsloth](https://unsloth.ai/docs/models/qwen3.6#nvfp4)** | The NVFP4 checkpoints themselves; `CUTE_DSL_ARCH=sm_121a`; `--moe-backend flashinfer_b12x` for Spark; the b12x availability probe; the Marlin warning; published MMLU-Pro accuracy for both models |
| **[r0b0tlab/qwen36-35b-a3b-nvfp4-sm121-vllm](https://github.com/r0b0tlab/qwen36-35b-a3b-nvfp4-sm121-vllm)** | **The key insight**: "B12X routed experts + MTP k=2, **Triton draft MoE**". That one phrase is what unlocked b12x+MTP for us |
| **[walterra.dev](https://walterra.dev/blog/2026-07-15-vllm-ultimate-gx10-aeon-snake)** | The persistent `/root/.cache/vllm` mount (cold 13–16 min → warm ~2 min); `TRITON_ATTN` with fp8 KV + DFlash; the fork-starvation warning at high util on unified memory |
| **[eugr/spark-vllm-docker](https://github.com/eugr/spark-vllm-docker)** | The image every unsloth model here runs on; `--earlyoom`; the `drop_caches` workaround; and the honest note that "NVFP4 performance on Spark [is] not fully optimized in vLLM (any build)" — which our data independently confirms |
| **[AEON-7/vllm-ultimate-dgx-spark](https://github.com/AEON-7/vllm-ultimate-dgx-spark)** | The AEON images, the DFlash drafters, and the uncensored checkpoints |

---

## The four findings

### 1. b12x + MTP *can* run together — here's the command

Unsloth documents `--moe-backend flashinfer_b12x` for DGX Spark. It dies the moment you add
MTP:

```
ValueError: moe_backend='flashinfer_b12x' is not supported for unquantized MoE.
Expected one of ['triton', 'flashinfer_trtllm', 'flashinfer_cutlass', 'aiter'].
```

The MTP **draft head ships unquantized**, and `--moe-backend` is global. The fix is that
`SpeculativeConfig` has **its own** `moe_backend` field:

```bash
export CUTE_DSL_ARCH=sm_121a
vllm serve unsloth/Qwen3.6-35B-A3B-NVFP4-Fast \
  --moe-backend flashinfer_b12x \
  --speculative-config '{"method":"mtp","num_speculative_tokens":2,"moe_backend":"triton"}'
```

Log confirms it: `Using 'FLASHINFER_B12X' NvFp4 MoE backend`. (Note `FLASHINFER_B12X` does
**not** appear in vLLM's "potential backends" list — that list misleads.)

Check availability first (unsloth's probe — passes on eugr's image):

```bash
python -c "import torch; from vllm.utils.flashinfer import has_flashinfer_b12x_gemm as g, has_flashinfer_b12x_moe as m
cap = torch.cuda.get_device_capability(); print('cap', cap, '| b12x gemm', g(), '| b12x moe', m())"
# cap (12, 1) | b12x gemm True | b12x moe True
```

### 2. …and on GB10 it makes things worse

Same box, same bench, only the backend changed — Qwen3.6-35B-A3B-NVFP4-Fast, U=0.55, N=8:

| | `auto` (Cutlass) + MTP k=3 | **b12x** + Triton draft + MTP k=2 |
|---|---|---|
| Prefill pp8192 | 6,694 / 6,488 tok/s | 6,645 / 6,775 |
| Prefill pp32768 | 4,824 / 4,850 | 4,968 / 4,993 |
| Decode c=1 | **86.0 / 89.6** | 75.2 / 81.8 |
| Decode c=8 | 322 / **336** | 184* / 337 |
| **KV @262k** | **10.65×** | **3.62×** |

\* first rep an outlier (TTFT 4.3 s)

**Prefill flat, decode ~11% down, KV pool cut to a third.**

**Why** — visible in the boot log: the 35B-A3B is a **GDN hybrid**. Prefill runs through
`Using Triton/FLA GDN prefill kernel`, and decode on GB10 is memory-bandwidth-bound
(273 GB/s). b12x accelerates **MoE GEMMs** — a stage that isn't the bottleneck here.
Unsloth's 1.79× is real; it's measured where the MoE GEMM *is* the critical path. On a
GDN-hybrid A3B on GB10, it isn't.

One caveat that keeps us honest: unsloth warn that without b12x you fall back to
**Marlin W4A16** (their measurement: 105.6 vs 125.9 tok/s). We never hit that — our `auto`
picks **Cutlass**, not Marlin. Their advice is right for the case they describe; it didn't
apply to us.

### 3. Prefill collapses with context — publish the curve, not one number

Everyone quotes a single prefill number. It's meaningless without the context length.

| Prompt | 35B-A3B (fp8 KV) | 27B dense (fp8 KV) |
|---|---|---|
| 8k | 6,600 tok/s · TTFT **1.2 s** | 1,600 tok/s · TTFT 5.1 s |
| 32k | 4,850 tok/s · TTFT **6.8 s** | 1,085 tok/s · TTFT **30.0 s** |

A 32k prompt costs **30 seconds before the first token** on the dense 27B. Our own docs had
"~7,000 tok/s prefill" recorded as fact — it was an 8k-only measurement, and real 30k+ use
had been landing at 2,000–2,400 the whole time.

### 4. The dense 27B costs 3–4× speed for ~0.4 MMLU-Pro

| | 35B-A3B | 27B dense |
|---|---|---|
| Prefill 8k / 32k | 6,600 / 4,850 | 1,600 / 1,085 |
| Decode c=1 / c=8 | 87.8 / 329 | 30.5 / 137 |
| KV @262k | 10.65× | 1.89× |
| **MMLU-Pro** (unsloth's published figures) | **85.85** | **86.25** |

Prefill 4.1–4.5× faster, decode 2.4–2.9× faster, KV 5.6× larger — for **0.4 MMLU-Pro
points**. Worth knowing before you pick the "smarter" dense model as your coding driver.

---

## Method (why these numbers are worth anything)

- **One variable at a time.** Same boot script, same bench, same box, same hour. Every image
  A/B swaps only `IMAGE=`.
- **Zombie rule.** vLLM's `/health` **lies**. It reports ready while the engine dies on the
  first real request. Every boot here sends one real completion before any number is taken.
- **3 reps once 2 wasn't enough.** A 15% spread between two runs can't resolve a 4% delta.
  Where we claim a difference, the ranges don't overlap.
- **Controls catch you.** Twice we reached a confident wrong conclusion that only a control
  run reversed. See [METHODOLOGY.md](METHODOLOGY.md#two-conclusions-we-got-wrong).

⚠️ **Our bench streams tokens and reads ~15–25% lower than non-streaming tooling.** Ornith on
one unchanged image measures 68.6–71.9 here and 92.7 with our older non-streaming script.
**Do not compare these numbers against other people's** — only same-script numbers are valid.

---

## Contents

| File | |
|---|---|
| [DEPLOY.md](DEPLOY.md) | **Start here** — copy-paste recipes for both stacks, with the safety rules inline |
| [FINDINGS.md](FINDINGS.md) | Every measurement, including the AEON image A/B |
| [TROUBLESHOOTING.md](TROUBLESHOOTING.md) | The five failure modes that cost us three outages |
| [METHODOLOGY.md](METHODOLOGY.md) | How to replicate, and the mistakes we made |
| [scripts/](scripts/) | The bench and boot scripts, as run |

## Hardware

DGX Spark · GB10 (Grace-Blackwell, **sm_121**) · 121 GiB unified LPDDR5X · ~273 GB/s ·
driver 595.71.05 · `eugr/spark-vllm:latest` (vLLM main) and
`ghcr.io/aeon-7/aeon-vllm-ultimate` 0.24 / 0.25.0.

## License

MIT for the scripts. Measurements are facts — use them, and please say where they came from.
