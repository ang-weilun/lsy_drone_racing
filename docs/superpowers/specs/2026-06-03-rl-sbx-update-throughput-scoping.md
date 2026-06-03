# rl_sbx PPO update throughput — scoping

**Date:** 2026-06-03
**Status:** scoping (no implementation). Investigated via 4-agent code sweep while an L2 run trained.

## The bottleneck (measured)

Profiled `JitScanPPO` on a 5090, L2 recipe, 512-wide, `diag_every=20`:

```
prof_scan_s             0.34 s   rollout collection (jax.lax.scan — ~3M steps/s)
prof_update_plus_log_s  9.7  s   PPO update loop  ← 96% of the time
prof_host_s             0.85 s
→ ~91k SPS
```

The update is **host-bound, not compute-bound** (GPU ~1% during it). The scan rollout lands on device, gets copied to SB3's **host numpy `RolloutBuffer`**, and the `for epoch: for minibatch:` loop slices each of ~768 minibatches back to the GPU via `.numpy()` (`jit_scan_ppo.py:649,654,662-667`). The GPU starves on host data movement + Python dispatch.

## Correction: `rl_song` does NOT fuse the update

I had assumed the path to ~500K was "port rl_song's fused update." **It isn't — rl_song has no fused update.** Its update is the *same* host-side double-loop we have (`rl_song/train.py:869-885`, per-minibatch jitted `_train_minibatch` at `train.py:902`). Only the **rollout** is scanned (`scan_rollout`, `rollout.py:412`). rl_song reaches ~700k SPS purely by **rollout scale**: `n_envs=16384 × n_steps=250 = 4.1M transitions/rollout` (`config.py:112,130`), so the fixed host-loop overhead is amortized over a giant batch — and its net is only 256-wide (`policy.py:36`). So there are two distinct levers, not one.

## Lever A (cheap, proven) — bigger rollouts

Scale `n_envs`/`n_steps` so the fixed ~10 s host overhead is amortized over more samples. This is *literally what rl_song does* — no rewrite, just config. Near-linear SPS until limited by:
- **GPU memory:** our config is **512-wide + 78-d obs** (vs rl_song's 256-wide); 16384 envs may not fit 32 GB — must measure. A 4096→8192 step is the safe first move.
- **Compute-bound crossover:** at 512-wide the per-sample backward is ~4× rl_song's, so gains go sub-linear sooner than rl_song sees.
- **PPO batch dynamics:** more envs = fewer updates per env-step + different gradient noise; the relB recipe was tuned at 4096, so a bigger batch needs a quick convergence sanity-check before the numbers are trusted.

**Verdict:** the right first lever. Low risk, no code, partial gain (~+30–100%).

## Lever B (net-new, biggest ceiling) — fuse the update on-device

Replace the host minibatch loop with a single jitted `jax.lax.scan` over (epoch × minibatch), keeping the rollout on device. Expected: update 9.7 s → ~1–2 s ⇒ **~91k → ~300–500k SPS**.

### Feasibility — yes, with caveats
- `_one_update_clipped_vf` is **already a pure `@jax.jit` function** suitable as the scan body (`jit_scan_ppo.py:464-484`); grads applied functionally via `TrainState.apply_gradients` (`:569,582`).
- Optimizer is **pure optax** (`optax.chain(clip_by_global_norm, inject_hyperparams(adam))`, `policy.py:367-381`) — jittable, composes inside a scan.
- With **`target_kl=None`** (our recipe), LR is set **once per rollout** and held constant (`jit_scan_ppo.py:613-617`), so it bakes into the opt_state before the scan — no per-step LR mutation. The KL-adaptive-LR branch is dead (`:644,685-691`).
- Per-minibatch diagnostics already collected as device arrays and reduced once (`:645-646,678-683,695-696`) → become scan outputs.
- **Carry:** `(actor_state, vf_state, rng_key)`. Advantage norm is per-minibatch (no running stats needed).

### The net-new code (3 blockers)
1. **In-JAX minibatch path:** replace `RolloutBuffer.get()` (host numpy permute + `.numpy()`) with an in-JAX `jax.random.permutation` + reshape of device arrays into `(n_epochs, n_minibatches, batch, …)`.
2. **In-JAX GAE/returns:** advantages/returns are currently computed by SB3's host buffer (`compute_returns_and_advantage`, `jit_scan_ppo.py:294-296`). Must reimplement GAE in JAX — the **largest, most correctness-sensitive piece**. (rl_song's `_compute_gae` is already a `lax.scan`, `train.py:757,793` — reusable as a model.)
3. **RNG in carry:** a per-epoch shuffle key joins the carry (today the update has no RNG; SB3 shuffles host-side).

### Parity bar (what must match — divergence here is silent)
A fused update is a valid drop-in only if it reproduces the host loop within tol:
- **Pessimistic clipped value loss with `VALUE_LOSS_SCALE=0.5`** — NOT stock SBX MSE (`jit_scan_ppo.py:571-579,101`). Copying stock SBX silently diverges here.
- **Per-minibatch** advantage norm `(adv-mean)/(std+1e-8)`, pop-std, gated `len>1` (`:549-550`) — not per-batch.
- **Exact shuffle:** fresh `np.random.permutation` per epoch off the **global numpy RNG** (`stable_baselines3/buffers.py:483`), sequential slices, on the `swap_and_flatten`'d `(n_steps·n_envs,…)` order. A JAX PRNG or one-permutation-for-all-epochs pairs different samples → diverges the Adam path.
- **Ragged last minibatch:** SB3 has no drop-last (`buffers.py:504-506`); a scan reshaping into equal tiles drops/pads it. **Enforce `N % batch_size == 0`** (drop-last or pad) or replicate the tail.
- **Sequential Adam order**, actor-then-critic, two separate opt_states (`:566-569,581-582`). A vmap-over-minibatch formulation is mathematically different → diverges.
- Logged scalars: last-minibatch losses (`:705-710`), rollout-mean KL/clip (`:695-696`), `_n_updates += n_epochs` (`:697`).

### Parity harness (mirror `check_obs_encoder_parity.py`)
Pin a fixed numpy+JAX seed; clone initial `actor_state.params`, `vf_state.params`, and **both opt_states**; build one fixed rollout batch (choose a `(batch_size, N)` where `N % batch_size != 0` to exercise the ragged tail). Run (1) the current host loop and (2) the fused scan **from the same cloned init and the same consumed permutation stream**. Assert `tree_map(assert_allclose, …)` over actor params, vf params, both opt_states, plus the 5 loss scalars and rollout-mean KL/clip, at `atol/rtol≈1e-5` (loosen to ~1e-4 if tensorcore matmul; or force `jax_default_matmul_precision='highest'`).

### Ranked risks
1. Shuffle RNG mismatch (silent, looks like noise). 2. Adv-norm scope (per-mb vs per-batch). 3. Ragged tail. 4. float32 vs bf16/tf32 precision over many Adam steps. 5. Optimizer accumulation order. 6. Constant-LR baking.

## Recommendation

1. **Now / cheap:** pull Lever A (bump `n_envs` 4096→8192, keep minibatch *count* fixed) and measure SPS + a short convergence sanity-check. Likely ~+50–100% for free, modulo 512-wide memory.
2. **Separate workstream:** Lever B (fuse) only if A doesn't get us far enough. It's ~1–2 focused days (in-JAX GAE + permute + scan + parity harness), net-new (no rl_song reference), with real divergence risk — but it's the only path to the ~500K ceiling and would 5–10× *every* future campaign run. Gate it behind the parity harness above.
3. **Not worth it for the current diagnostic run** (we're at 91k → L2 ~55 min, L3 stage-5 faster). This is campaign infrastructure, not blocking.

**Open unknowns:** exact `(n_steps, n_envs, batch_size, n_epochs)` and `N % batch_size` for the production recipe (verify divisibility); whether 16384 envs fit 32 GB at 512-wide; SBX runtime version vs the `/tmp/sbx-src` source the sweep read.
