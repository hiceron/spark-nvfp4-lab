# X.com drafts + posting playbook

Blog: https://hiceron.github.io/spark-nvfp4-lab/ · Repo: https://github.com/hiceron/spark-nvfp4-lab  
X: Dawid [@Hiceron2](https://x.com/Hiceron2)

Voice: first person ("I"), with the credit line — this work was done **with Claude
(Anthropic) and Rem, my personal AI agent**. Attribution to
**@UnslothAI**, **r0b0tlab**, **@walterra** (walterra.dev), **eugr**, **AEON-7** is
mandatory — verify handles before tagging.

Rule that survived editing: **stacks compare only against themselves.** unsloth numbers vs
unsloth claims, AEON numbers vs AEON's published Spark figures. No cross-stack "beats" rows.

---

## Option A — the thread (recommended)

**1/**
> **926 tok/s aggregate from one DGX Spark.**
>
> 64 parallel streams, 262k context, unsloth's 35B NVFP4, GPU util at just 0.55 — and the
> throughput was still climbing when I ran out of test ceiling.
>
> Four days of benchmarking + optimizing the new NVFP4 drops, with Claude + my AI agent Rem.
> Recipes, failures, one crashed machine 🧵

**2/**
> What one Spark delivers with the unsloth stack:
>
> • 35B-A3B: 89–95 tok/s single-stream · **10.65× full-262k KV** at U=0.55 (21.46× at 0.65)
> • 27B dense: **33.5–35.5 tok/s** — an AEON DFlash drafter grafted onto unsloth weights (+35%)
>
> The full serve command is at the top of the blog. Three commands, no custom builds.

**3/**
> The recipe nobody documents: b12x + MTP together.
>
> unsloth's Spark advice (`--moe-backend flashinfer_b12x`) dies the moment you add MTP:
> `ValueError: ... not supported for unquantized MoE`
>
> The MTP **draft head ships unquantized** and the flag is global.

**4/**
> The fix — SpeculativeConfig has its OWN moe_backend:
>
> ```
> export CUTE_DSL_ARCH=sm_121a
> vllm serve unsloth/Qwen3.6-35B-A3B-NVFP4-Fast \
>   --moe-backend flashinfer_b12x \
>   --speculative-config '{"method":"mtp","num_speculative_tokens":2,"moe_backend":"triton"}'
> ```
>
> h/t r0b0tlab — his config implied this all along.

**5/**
> Then I measured it, and here's the plot twist: **on GB10, the "fast path" loses.**
>
> Prefill: flat · Decode: **−11%** · KV pool: **10.65× → 3.62×**
>
> Why: this model is a GDN hybrid — prefill runs on a Triton kernel, decode is pinned by
> 273 GB/s bandwidth. b12x accelerates the one stage that isn't the bottleneck on a Spark.
> unsloth's 1.79× is real — on hardware where MoE compute IS the bottleneck.

**6/**
> What `gpu-memory-utilization` actually buys on unified memory: KV pool, NOT speed.
>
> 35B: 0.55 → 10.65× of 262k · 0.65 → 21.46× — decode identical
> 27B: 0.55 → 1.9× · 0.74 → 4.24× — decode identical
>
> Hard ceiling is real: spec-decode buffers scale with max-num-seqs OUTSIDE that
> accounting. U=0.80 + 4 DFlash seqs hard-locked my box. Power button.

**7/**
> Prefill is a curve, not a number:
>
> 35B-A3B: 6,600 tok/s @8k → 4,850 @32k
> 27B dense: 1,600 @8k → 1,085 @32k
>
> A 32k prompt = 6.8 s to first token on the MoE, **30 s** on the dense.
> (Benchmarks say they're 0.4 MMLU-Pro apart. My real coding use still prefers the dense
> one for quality — pick per workload.)

**8/**
> Speculative decoding speed is a function of OUTPUT PREDICTABILITY — nobody tells you this:
>
> Same 27B, same config, only the prompt type changed:
> math 44.6 · JSON 36.7 · coding 32.6 · prose 27.3 tok/s
>
> When you see a "peak" tok/s for a spec-decode model, ask which category it was.

**9/**
> The AEON (uncensored) stack got its own week: image 0.24 → 0.25.0, A/B on all three
> checkpoints:
>
> Ornith: dead heat · Heretic: **+8.5%** · 27B MTP: **+22% decode**
>
> I almost called it after testing one model — the ONE of three that showed nothing.
> Test everything. Decode is not prefill. Three reps, not two.

**10/**
> Also settled by measurement: AEON checkpoints auto-pick Marlin kernels. Forcing the
> "faster" Cutlass path:
>
> `FLASHINFER_CUTLASS does not support ... QuantKey(u8, scale(f8e4m3fn...))`
>
> Marlin is the ONLY backend for that quant format. No speed left on the table.

**11/**
> Credit where due — most of this started as other people's work:
>
> • r0b0tlab: the b12x+MTP key phrase
> • @UnslothAI: the checkpoints, probe, accuracy data
> • AEON-7: the images, drafters (one powers my 27B), and properly Spark-measured cards
> • @walterra: the vllm-cache mount + fork-starvation warning
> • eugr: the nightly image everything runs on

**12/**
> I got 2 conclusions WRONG before controls caught them, and nearly a 3rd:
>
> "New image is 20% slower" — my own bench change.
> "b12x is dead on sm_121" — the probe said available all along.
> Near-kept the old AEON image off a one-model test — would've cost the +22%.
>
> Full writeup, recipes, raw logs, failure catalog:
> https://hiceron.github.io/spark-nvfp4-lab/ · https://github.com/hiceron/spark-nvfp4-lab

---

## Option B — single post

> 926 tok/s aggregate from one DGX Spark — unsloth's 35B NVFP4, 64 streams, 262k context,
> GPU util 0.55. Four days of benchmarking + optimizing the new NVFP4 drops with Claude and
> my AI agent Rem.
>
> Also in the writeup: the undocumented b12x+MTP recipe (then measured as a LOSS on GB10 —
> the model's prefill never touches the kernels b12x accelerates), why spec-decode speed
> depends on prompt category (math 44.6 vs prose 27.3 on the same model), the AEON image
> A/B that turned up +22%, and the config that hard-locked my machine.
>
> https://hiceron.github.io/spark-nvfp4-lab/
>
> h/t r0b0tlab, @UnslothAI, @walterra, eugr, AEON-7 — most of this started as their work.

---

## Posting playbook

**Where to host the blog.** Recommended: **GitHub Pages from the same repo** — put the
blog HTML at `docs/index.html`, enable Pages (Settings → Pages → main /docs). Free, no
account setup, and the dev audience trusts a github.io link next to a repo. Firebase
Hosting (your free account) works too and is fine — but it splits the ecosystem across two
platforms for no gain. The claude.ai artifact link works if shared, but a custom-domain-less
claude.ai URL reads less "yours" to strangers. Verdict: **repo + Pages, one link ecosystem**;
keep the artifact as your private working copy.

**Images (attach 2–4 to the thread — posts with images get ~2× engagement):**
1. Tweet 1: screenshot of the blog hero (title + the four verdict tiles) — dark theme.
2. Tweet 5: the auto-vs-b12x table (the −11% / 3.62× row visible).
3. Tweet 7: the prefill bar chart (it's the most shareable visual).
4. Tweet 9: the three per-model A/B mini-tables side by side.
Screenshot at ~1200×675 (16:9) or 1600×900 for crispness; X compresses hard.

**Mechanics:** post the thread in one sitting (drafts ready), pin tweet 1 to your profile,
reply to your own last tweet with the repo link again (threads get unrolled — the last
tweet is the second-most-read). Best windows for dev content: Tue–Thu, morning US time.
Reply to questions fast in the first 2 hours; that's when the algorithm decides.

**If someone posts better numbers:** link + thank-you, not defense. The methodology
sections exist precisely so the numbers can be checked — that's the brand.
