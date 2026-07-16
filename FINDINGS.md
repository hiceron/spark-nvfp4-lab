# Findings — every measurement

All figures from one GB10 (121 GiB unified, ~273 GB/s, driver 595.71.05). Same boot script,
same bench, one variable per comparison. Every boot verified with a real completion before
any number was taken. **See [METHODOLOGY.md](METHODOLOGY.md) before comparing anything here
to numbers from other tooling — our bench streams and reads ~15–25% lower.**

---

## 1. b12x on GB10 — engages, then loses

`unsloth/Qwen3.6-35B-A3B-NVFP4-Fast` · eugr image · U=0.55 · N=8 · fp8 KV

| Metric | `auto` (→Cutlass) + MTP k=3 | b12x + Triton draft + MTP k=2 |
|---|---|---|
| Prefill pp8192 | 6,694 / 6,488 | 6,645 / 6,775 |
| Prefill pp32768 | 4,824 / 4,850 | 4,968 / 4,993 |
| Decode c=1 | **86.0 / 89.6** | 75.2 / 81.8 |
| Decode c=8 | 322 / 336 | 184* / 337 |
| KV @262k | **10.65×** | **3.62×** |
| MTP acceptance | 67.8% | 75.5% |

\* outlier, TTFT 4.3 s

**Prefill flat · decode −11% · KV pool cut to a third.** Cause is structural: the 35B-A3B is a
GDN hybrid (`Using Triton/FLA GDN prefill kernel`), so prefill never touches the MoE GEMMs
b12x accelerates, and decode is bandwidth-bound at 273 GB/s.

Rejected outright:

| Attempt | Error |
|---|---|
| `--linear-backend flashinfer_b12x` | `no 'flashinfer_b12x' kernel exists for this layer type` |
| `--moe-backend flashinfer_cutedsl` | `NvFp4 MoE backend 'FLASHINFER_CUTEDSL' does not support ... kernel does not support current device cuda` |
| `--moe-backend flashinfer_b12x` **with MTP, no drafter override** | `not supported for unquantized MoE. Expected one of ['triton', ...]` |
| b12x on the **dense 27B** | `no 'flashinfer_b12x' kernel exists for this layer type` — dense, no routed-expert MoE |

---

## 2. Prefill vs context

Single request, prefix cache defeated.

| Prompt | 35B-A3B (fp8 KV) | 27B dense (fp8 KV) |
|---|---|---|
| 8k | 6,600 tok/s · TTFT 1.2 s | 1,600 tok/s · TTFT 5.1 s |
| 32k | 4,850 tok/s · TTFT 6.8 s | 1,085 tok/s · **TTFT 30.0 s** |

MoE loses 27% from 8k→32k; dense loses 32% from a 4× lower start. **Quoting a single prefill
number without its context length is meaningless.**

---

## 3. Dense 27B vs MoE 35B — the price of "smarter"

| | 35B-A3B | 27B dense | ratio |
|---|---|---|---|
| Prefill 8k / 32k | 6,600 / 4,850 | 1,600 / 1,085 | 4.1× / 4.5× |
| Decode c=1 / c=8 | 87.8 / 329 | 30.5 / 137 | 2.9× / 2.4× |
| KV @262k | 10.65× | 1.89× | 5.6× |
| MMLU-Pro *(unsloth's published figures, not ours)* | 85.85 | 86.25 | **−0.4** |

---

## 4. AEON image 0.24 vs 0.25.0 — test every checkpoint

Same boot script, same bench, only `IMAGE=` swapped. U=0.55/N=4 for the 35B checkpoints;
U=0.70/N=2 for the 27B (it needs 27.14 GiB KV per 262k seq — 5× ornith).

| | Ornith 0.24 → 0.25 | Heretic 0.24 → 0.25 | AEON 27B MTP 0.24 → 0.25 |
|---|---|---|---|
| pp8192 | 4,578 → 4,761 | 5,116 → 5,186 | 1,757 → **1,885** (+7.4%) |
| pp32768 | 4,207 → 4,286 | 4,418 → **4,585** (+3.8%) | 1,322 → **1,456** (+9%) |
| Decode c=1 (mean) | 66.3 → 63.4 | 70.8 → **76.8** (+8.5%) | 25.0 → **30.4** (+22%) |
| KV @262k | 2.68 → 2.69× | 2.69 → **2.88×** (+7%) | 1.47 → 1.47× |
| Spec acceptance | 44.1 → 40.5% | 45.2 → 45.4% | 27.8 → **31.3%** |
| **Verdict** | dead heat | **0.25 wins** | **0.25 wins big** |

Where a win is claimed, the three-rep ranges **don't overlap** (heretic 68.1–72.6 vs
75.7–78.3; 27B 23.1–26.5 vs 29.7–31.3). Ornith's own runs disagreed by 15%, which is why its
result is "dead heat" and not a 4% claim.

> **The lesson that nearly cost us the finding:** our first pass tested **ornith only, decode
> only**, concluded "dead heat — keep the old image", and stopped. That was one checkpoint
> away from discarding a **+22% decode gain**. One model is not a matrix; decode is not prefill.

All three AEON checkpoints auto-select **Marlin** on both images — which unsloth flag as the
slow path for W4A4 (their figures: 105.6 vs 125.9 tok/s). **Tested 2026-07-16: Marlin is not
a mistake here.** Forcing `--moe-backend flashinfer_cutlass` on ornith and heretic both
rejected at boot:

```
ValueError: NvFp4 MoE backend 'FLASHINFER_CUTLASS' does not support the deployment
configuration since kernel does not support quantization scheme
QuantKey(u8, scale(f8e4m3fn, static, GroupShape(row=1, col=16)), scale2(f32, static, per_tensor), symmetric)
```

The AEON compressed-tensors scheme (u8 weights + fp8 group scales) is only implemented by
Marlin in this build. The unsloth warning concerns *their* W4A4 scheme; no speed was left on
the table for AEON checkpoints.

---

## 5. Image ↔ checkpoint matrix

| Checkpoint | Image | Why |
|---|---|---|
| unsloth 35B-A3B, unsloth 27B | **eugr** (vLLM main) | KV 10.38× vs 9.39× on AEON-0.25; decode identical. The 27B **zombie-crashes** on AEON images (`Window left is not the same for all layers`) |
| Ornith, heretic, AEON 27B MTP | **AEON 0.25.0** | 0.24 deleted after the table above |

fp8 KV works on unsloth checkpoints (they ship calibration scales) and **zombie-boots** on
AEON ones (they don't) — health passes, engine dies on the first token.

---

## 6. Speculative-decode speed is a category property (and k doesn't transfer)

Same 27B (unsloth weights + AEON DFlash drafter graft), same config, non-streaming,
200-token outputs — only the prompt type changed:

| Category | k=10 | k=15 (AEON's headline k) |
|---|---|---|
| math | **44.6** | 33.5 (spread 21.8–43.3) |
| json | 36.7 | **41.6** |
| coding | 32.6 | 35.7 |
| prose | 27.3 | 27.6 |
| KV @262k | **4.24×** | 3.77× |

Two conclusions. **First:** "median/peak tok/s" figures for spec-decode models are category
statistics — AEON's published "median 38.5 / peak 71.3" for his 27B decodes as
across-category median / best category at 200-token outputs, k=15, single stream. Measured
per-category, my numbers land in the same territory (his own docs show a 34–56 range). Ask
"which category?" whenever you see a spec-decode peak. **Second:** the optimal draft length
belongs to the checkpoint+drafter pair, not the method — k=15 is his tuned headline, and on
my grafted drafter it's net-flat with worse variance and a smaller KV pool. k=10 stays.

## Raw logs

`results/` holds the unedited output of every matrix, failures included.
