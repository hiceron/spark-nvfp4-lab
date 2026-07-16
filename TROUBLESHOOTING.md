# Failure modes on GB10 / DGX Spark

Five ways this box breaks. Each one cost us real downtime. All are specific to **unified
memory** — none of them happen the same way on a discrete GPU.

---

## 1. Ghost memory: the engine is gone, the RAM isn't

**Symptom.** No LLM container running, yet `free` shows ~94 GiB *used*. The next boot dies:

```
ABORT: unified memory not released (35 GiB)
# or, from vLLM itself:
RuntimeError: Engine core initialization failed
```

**Diagnostic that identifies it in one shot** — if all three are true, it's ghost memory:

```bash
free -g                                    # ~94 GiB "used"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv   # ...but NO GPU processes
ps aux --sort=-rss | head -3               # ...and no large RSS anywhere
```

Memory the driver still holds for an engine that no longer exists. It is **not** page cache,
but `drop_caches` releases it anyway.

**Fix.** Verified repeatedly: 94 GiB used → 117 GiB available in ~4 seconds.

```bash
sync; echo 3 | sudo tee /proc/sys/vm/drop_caches
```

**Reboot only if that fails.** We wasted a reboot before learning this.

---

## 2. `docker restart` can never work here

We ran a nightly restart at 07:00 (DFlash acceptance decays after ~20 h; the engine crashes
around 23 h). The script did `docker restart`. **It failed every single night**, silently:

```
07:00:01 restarting unsloth27b-v025
→ RuntimeError: Engine core initialization failed
# box dead until a human noticed at 11:16
```

`docker restart` brings the container straight back into the ghost memory its own shutdown
just created. It finds ~27 GiB, needs ~90 GiB, dies.

**Correct sequence** (mirrors what a model switch does):

```
detect running LLM → pause voice/ASR/TTS → stop LLM → drop_caches
→ docker start → wait health → SEND A REAL REQUEST → restore voice
```

Two details that matter:
- **Pause the voice containers.** They hold unified memory the LLM needs to reclaim.
- **Use `docker start`, not the boot script.** The container name doesn't encode its config
  (`unsloth27b-v025` might be U=0.55/N=8 *or* U=0.74/N=4). `docker start` replays the exact
  original args; re-deriving them guesses, and guesses wrong.

---

## 3. Zombie boots: `/health` lies

The single most valuable rule here.

```
LLM ready                      ← health check passed
BOOT_RC=0                      ← boot script "succeeded"
{"error":{"message":"EngineCore encountered an issue..."}}   ← first real request
→ engine dead, every subsequent request: connection refused
```

**`/health` is served by the API process. The engine is a different process.** It will
happily report healthy while the engine is dead or about to die on first token.

**Rule: after every boot, send one real completion before trusting anything.** It has caught:
- fp8 KV on AEON checkpoints (they ship no k/v scales → dies on first token)
- DFlash + FlashInfer attention: `ValueError: Window left is not the same for all layers`

Also make your boot loop detect a **dead container**, or you'll wait out the full health
timeout on a corpse (we burned 21 minutes doing exactly that):

```bash
docker ps --format '{{.Names}}' | grep -qx "$NAME" || { echo "CONTAINER DIED"; docker logs --tail 20 "$NAME"; exit 3; }
```

---

## 4. DFlash buffers are invisible to `gpu-memory-utilization` — and they OOM the box

**This one hard-locked the machine.** Not a crash, not a hang: no SSH, no ping, dead.
Physical power-cycle required.

```
DFLASH=1 DFLASH_K=10 boot-aeon27b.sh 0.80 4     # ← U=0.80, N=4
→ box gone in ~2 minutes
```

**Why.** `--gpu-memory-utilization 0.80` reserves ~97 GiB of 121 GiB. Then **DFlash
per-sequence draft+verify buffers allocate on top of that**, they are **not counted** by
`gpu-memory-utilization`, and they **scale with `max-num-seqs`**. Four sequences × k=10
drafts overflowed the remaining ~24 GiB → NVRM cascade → whole box.

**Rules:**
- `util` and `max-num-seqs` are **not independent** when DFlash/spec-decode is on. Every
  `MemAvailable`-before-boot check will pass, and the box will still die *after* boot.
- Budget backwards: total − (util × total) must cover the OS **and** N × drafter buffers.
- Same config at **U=0.70 / N=2** boots fine (KV 1.47× @262k, ~36 GiB headroom).
- Independent confirmation: [walterra.dev](https://walterra.dev/blog/2026-07-15-vllm-ultimate-gx10-aeon-snake)
  hit "fork starvation" at U=0.85 — kernel alive, userspace can't spawn, **SSH hangs at
  banner**. That's the milder version of the same wall.

---

## 5. KV per sequence varies 5× between models — don't reuse a util

Same box, same 262k target, wildly different requirements:

| Model | KV per 262k seq | Consequence |
|---|---|---|
| Ornith 35B (BF16) | 13.24 GiB | fine at U=0.55 |
| 35B-A3B (fp8) | ~3.25 GiB | 10.65× at U=0.55 |
| **AEON 27B MTP (BF16)** | **27.14 GiB** | **cannot serve 262k below ~U=0.65** |

Reusing U=0.55 from a 35B on the AEON 27B produces:

```
ValueError: To serve at least one request with the model's max seq len (262144),
27.14 GiB KV cache is needed, which is larger than the available KV cache memory (21.51 GiB).
```

Deriving the right util from that error is easy and worth doing:
`non-KV overhead = util × 121.63 − reported_available_KV` → here ~45 GiB → so
U=0.70 → 85.1 − 45 ≈ 40 GiB KV ≈ **1.46×** at 262k. Measured: **1.47×**.

---

## Quick reference

| Symptom | Cause | Fix |
|---|---|---|
| Boot aborts, "unified memory not released" | ghost memory | `drop_caches`, retry, **then** reboot |
| Health OK → first request errors | zombie boot | real-completion check; check kv scales / attention backend |
| Box unreachable, no ping | DFlash buffers OOM | power-cycle; lower `util` **and** `max-num-seqs` |
| `no 'flashinfer_b12x' kernel exists for this layer type` | global flag hit an unquantized layer | put `moe_backend` in `--speculative-config` |
| `kernel does not support current device cuda` | sm_121 vs sm_10x gate | not fixable from config |
| Boot "fails" but engine is fine | health timeout too short | AEON image needs ~15 min cold (`init engine took 670 s`) |
| Every switch recompiles | `/root/.cache/vllm` is inside the container | mount `~/vllm-cache:/root/.cache/vllm` |
