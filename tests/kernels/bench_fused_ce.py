#!/usr/bin/env python3
"""Head-to-head benchmark: bmgpt Pallas fused CE vs Marin XLA streaming vs naive reference.

Run on TPU v4-64 (replicated, all workers):
    gcloud compute tpus tpu-vm ssh "$TPU_NAME" --worker=all \
      --command="cd /path/to/repo && uv run python tests/bench_fused_ce.py"

Defaults match production DCLM 740M config:
    batch=16, pos=2048, embed=1536, vocab=129024
"""
# ────────────────────────────────────────────────────────────────────
# Vendored Marin/Levanter XLA streaming cross-entropy (Apache-2.0)
#
# Source: https://github.com/stanford-crfm/levanter  (The Levanter Authors)
# Files: reference.py (streaming fwd), xla.py (custom VJP bwd)
# SPDX-License-Identifier: Apache-2.0
# ────────────────────────────────────────────────────────────────────

from __future__ import annotations

import argparse
import time
from functools import partial
from typing import Optional

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int

# ── Marin streaming forward (fori_loop, no full logit materialization) ───────


def _apply_logit_soft_cap(
  logits: Float[Array, "B V"], logit_soft_cap: Optional[float]
) -> Float[Array, "B V"]:
  if logit_soft_cap is None:
    return logits
  return jnp.tanh(logits / logit_soft_cap) * logit_soft_cap


def linear_softmax_cross_entropy_loss_streaming(
  x: Float[Array, "B H"],
  labels: Int[Array, "B"],
  w: Float[Array, "H V"],
  *,
  block_size: int,
  dtype: Optional[jnp.dtype] = jnp.float32,
  logit_soft_cap: Optional[float] = None,
  precision: jax.lax.PrecisionLike = None,
) -> tuple[Float[Array, "B"], Float[Array, "B"]]:
  """Streaming reference loss + logsumexp without materializing logits."""
  if block_size <= 0:
    raise ValueError(f"block_size must be positive, got {block_size}.")

  b_dim = x.shape[0]
  v_dim = w.shape[1]
  out_dtype = jnp.dtype(dtype) if dtype is not None else x.dtype

  pad = (-v_dim) % block_size
  if pad:
    w = jnp.pad(w, ((0, 0), (0, pad)), mode="constant", constant_values=0)
  v_padded = v_dim + pad
  num_blocks = v_padded // block_size

  logsumexp_init = jnp.full_like(x.sum(-1), -jnp.inf, dtype=out_dtype)
  label_logit_init = jnp.full_like(x.sum(-1), -jnp.inf, dtype=out_dtype)

  def body(block_idx, state):
    logsumexp, label_logit = state
    start = block_idx * block_size

    w_block = jax.lax.dynamic_slice(w, (0, start), (w.shape[0], block_size))
    logits = jax.lax.dot_general(
      x,
      w_block,
      (((1,), (0,)), ((), ())),
      precision=precision,
      preferred_element_type=jnp.float32,
    )
    if dtype is not None:
      logits = logits.astype(dtype)
    logits = _apply_logit_soft_cap(logits, logit_soft_cap)

    valid = (start + jnp.arange(block_size)) < v_dim
    logits = jnp.where(valid, logits, -jnp.inf)

    block_lse = jax.nn.logsumexp(logits, axis=-1)
    logsumexp = jnp.logaddexp(logsumexp, block_lse)

    in_block = (labels >= start) & (labels < start + block_size)
    label_idx = labels - start
    safe_idx = jnp.where(in_block, label_idx, 0)
    block_label_logit = logits[jnp.arange(b_dim), safe_idx]
    label_logit = jnp.where(in_block, block_label_logit, label_logit)
    return logsumexp, label_logit

  logsumexp, label_logit = jax.lax.fori_loop(
    0, num_blocks, body, (logsumexp_init, label_logit_init)
  )
  loss = logsumexp - label_logit
  return loss, logsumexp


# ── Marin streaming backward (fori_loop custom VJP) ─────────────────────────


def _materialize_cotangent(
  cotangent: jax.Array | jax.custom_derivatives.SymbolicZero,
  reference: jax.Array,
) -> jax.Array:
  if isinstance(cotangent, jax.custom_derivatives.SymbolicZero):
    return jnp.zeros_like(reference)
  return jnp.asarray(cotangent, dtype=reference.dtype)


def _linear_softmax_cross_entropy_loss_streaming_bwd(
  x: Float[Array, "B H"],
  labels: Int[Array, "B"],
  w: Float[Array, "H V"],
  lse: Float[Array, "B"],
  dout_loss: Float[Array, "B"],
  dout_lse: Float[Array, "B"],
  *,
  block_size: int,
  dtype: Optional[jnp.dtype],
  logit_soft_cap: Optional[float],
  precision: jax.lax.PrecisionLike,
) -> tuple[Float[Array, "B H"], Float[Array, "H V"]]:
  if block_size <= 0:
    raise ValueError(f"block_size must be positive, got {block_size}.")

  b_dim, h_dim = x.shape
  v_dim = w.shape[1]
  row_indices = jnp.arange(b_dim, dtype=labels.dtype)

  pad = (-v_dim) % block_size
  if pad:
    w_padded = jnp.pad(w, ((0, 0), (0, pad)), mode="constant", constant_values=0)
  else:
    w_padded = w

  v_padded = v_dim + pad
  num_blocks = v_padded // block_size
  lse_dtype = lse.dtype
  dout_loss = dout_loss.astype(lse_dtype)
  dout_lse = dout_lse.astype(lse_dtype)
  gx_init = jnp.zeros_like(x)
  gw_init = jnp.zeros((h_dim, v_padded), dtype=w.dtype)

  def body(block_idx, state):
    gx, gw = state
    start = block_idx * block_size

    w_block = jax.lax.dynamic_slice(w_padded, (0, start), (h_dim, block_size))
    logits = jax.lax.dot_general(
      x,
      w_block,
      (((1,), (0,)), ((), ())),
      precision=precision,
      preferred_element_type=jnp.float32,
    )
    if dtype is not None:
      logits = logits.astype(dtype)

    cap_deriv = jnp.asarray(1.0, dtype=logits.dtype)
    if logit_soft_cap is not None:
      tanh_arg = logits / logit_soft_cap
      tanh_val = jnp.tanh(tanh_arg)
      logits = tanh_val * logit_soft_cap
      cap_deriv = (1.0 - tanh_val**2).astype(logits.dtype)

    valid = (start + jnp.arange(block_size, dtype=labels.dtype)) < v_dim
    logits = jnp.where(valid[None, :], logits, -jnp.inf)

    probs = jnp.exp(logits - lse[:, None].astype(logits.dtype))
    delta = (
      dout_loss[:, None].astype(logits.dtype) + dout_lse[:, None].astype(logits.dtype)
    ) * probs

    in_block = (labels >= start) & (labels < start + block_size)
    label_idx = labels - start
    safe_idx = jnp.where(in_block, label_idx, 0)
    delta = delta.at[row_indices, safe_idx].add(
      jnp.where(in_block, -dout_loss.astype(logits.dtype), 0.0)
    )
    delta = (delta * cap_deriv).astype(logits.dtype)

    gx_block = jax.lax.dot_general(
      delta,
      w_block,
      (((1,), (1,)), ((), ())),
      precision=precision,
      preferred_element_type=jnp.float32,
    ).astype(gx.dtype)
    gw_block = jax.lax.dot_general(
      x,
      delta,
      (((0,), (0,)), ((), ())),
      precision=precision,
      preferred_element_type=jnp.float32,
    ).astype(gw.dtype)
    gx = gx + gx_block
    gw = jax.lax.dynamic_update_slice(gw, gw_block, (0, start))
    return gx, gw

  gx, gw = jax.lax.fori_loop(0, num_blocks, body, (gx_init, gw_init))
  return gx, gw[:, :v_dim]


@partial(jax.custom_vjp, nondiff_argnums=(0, 1, 2, 3))
def _linear_softmax_cross_entropy_loss_streaming_custom_vjp(
  block_size: int,
  dtype: Optional[jnp.dtype],
  logit_soft_cap: Optional[float],
  precision: jax.lax.PrecisionLike,
  x: Float[Array, "B H"],
  labels: Int[Array, "B"],
  w: Float[Array, "H V"],
) -> tuple[Float[Array, "B"], Float[Array, "B"]]:
  return linear_softmax_cross_entropy_loss_streaming(
    x,
    labels,
    w,
    block_size=block_size,
    dtype=dtype,
    logit_soft_cap=logit_soft_cap,
    precision=precision,
  )


def _custom_vjp_fwd(
  block_size,
  dtype,
  logit_soft_cap,
  precision,
  x,
  labels,
  w,
):
  loss, lse = linear_softmax_cross_entropy_loss_streaming(
    x,
    labels,
    w,
    block_size=block_size,
    dtype=dtype,
    logit_soft_cap=logit_soft_cap,
    precision=precision,
  )
  return (loss, lse), (x, labels, w, lse)


def _custom_vjp_bwd(
  block_size,
  dtype,
  logit_soft_cap,
  precision,
  residuals,
  cotangents,
):
  x, labels, w, lse = residuals
  dout_loss, dout_lse = cotangents
  dout_loss_arr = _materialize_cotangent(dout_loss, lse)
  dout_lse_arr = _materialize_cotangent(dout_lse, lse)
  gx, gw = _linear_softmax_cross_entropy_loss_streaming_bwd(
    x,
    labels,
    w,
    lse,
    dout_loss_arr,
    dout_lse_arr,
    block_size=block_size,
    dtype=dtype,
    logit_soft_cap=logit_soft_cap,
    precision=precision,
  )
  return gx, None, gw


_linear_softmax_cross_entropy_loss_streaming_custom_vjp.defvjp(
  _custom_vjp_fwd,
  _custom_vjp_bwd,
)


def linear_softmax_cross_entropy_loss_xla(
  x: Float[Array, "B H"],
  labels: Int[Array, "B"],
  w: Float[Array, "H V"],
  *,
  v_block_size: int = 4096,
  dtype: Optional[jnp.dtype] = jnp.float32,
  logit_soft_cap: Optional[float] = None,
  precision: jax.lax.PrecisionLike = None,
) -> tuple[Float[Array, "B"], Float[Array, "B"]]:
  """Marin XLA streaming cross-entropy (custom VJP, fori_loop fwd+bwd)."""
  return _linear_softmax_cross_entropy_loss_streaming_custom_vjp(
    v_block_size,
    dtype,
    logit_soft_cap,
    precision,
    x,
    labels,
    w,
  )


# ── End vendored Marin code ─────────────────────────────────────────────────


# ── Levanter Pallas kernel (optional) ───────────────────────────────────────

HAS_LEVANTER = False
try:
  from levanter.kernels.pallas.fused_cross_entropy_loss import (
    fused_cross_entropy_loss_and_logsumexp_penalty,
  )

  HAS_LEVANTER = True
except ImportError:
  pass


# ── Timing harness (adapted from Marin's bench scripts) ─────────────────────


def time_fn(fn, *args, warmup: int = 3, steps: int = 5):
  t0 = time.perf_counter()
  out = fn(*args)
  jax.block_until_ready(out)
  compile_time = time.perf_counter() - t0

  for _ in range(warmup - 1):
    out = fn(*args)
    jax.block_until_ready(out)

  run_times = []
  for _ in range(steps):
    t0 = time.perf_counter()
    out = fn(*args)
    jax.block_until_ready(out)
    run_times.append(time.perf_counter() - t0)

  return compile_time, run_times


def report(
  name: str, compile_time: float, run_times: list[float], num_tokens: int, mode: str
):
  avg = sum(run_times) / len(run_times)
  tok_s = num_tokens / avg
  print(
    f"  [{mode:7s}] {name:30s}  compile={compile_time:7.3f}s  "
    f"avg={avg * 1e3:8.2f}ms  tok/s={tok_s:12,.0f}"
  )


# ── Roofline estimates for TPU v4 (per chip) ────────────────────────────────
# From Marin bench scripts: 275 TFLOPS bf16, 1.2 TB/s HBM bandwidth
V4_BF16_TFLOPS = 275.0
V4_HBM_BW_TBS = 1.2


def roofline_estimate(batch: int, pos: int, embed: int, vocab: int):
  n = batch * pos
  # Forward: x @ w^T → 2*N*D*V FLOPs, read N*D + D*V + write N*V elements
  fwd_flops = 2 * n * embed * vocab
  fwd_bytes = (n * embed + embed * vocab + n * vocab) * 2  # bf16
  fwd_compute_s = fwd_flops / (V4_BF16_TFLOPS * 1e12)
  fwd_memory_s = fwd_bytes / (V4_HBM_BW_TBS * 1e12)

  # Backward: ~4x forward FLOPs (dL/dx and dL/dw matmuls)
  bwd_flops = 4 * n * embed * vocab
  bwd_bytes = 2 * fwd_bytes
  bwd_compute_s = bwd_flops / (V4_BF16_TFLOPS * 1e12)
  bwd_memory_s = bwd_bytes / (V4_HBM_BW_TBS * 1e12)

  print(f"\n  Roofline (single v4 chip, N={n}, D={embed}, V={vocab}):")
  print(
    f"    FWD: compute={fwd_compute_s * 1e3:.3f}ms  "
    f"memory={fwd_memory_s * 1e3:.3f}ms  "
    f"({'compute' if fwd_compute_s > fwd_memory_s else 'memory'}-bound)"
  )
  print(
    f"    BWD: compute={bwd_compute_s * 1e3:.3f}ms  "
    f"memory={bwd_memory_s * 1e3:.3f}ms  "
    f"({'compute' if bwd_compute_s > bwd_memory_s else 'memory'}-bound)"
  )


# ── Loss functions ───────────────────────────────────────────────────────────


def ref_loss(x_flat, w_hv, labels_flat):
  logits = x_flat @ w_hv
  lse = jax.nn.logsumexp(logits, axis=-1)
  label_logits = logits[jnp.arange(x_flat.shape[0]), labels_flat]
  return (lse - label_logits).mean()


def marin_xla_loss(x_flat, w_hv, labels_flat, *, v_block_size: int):
  loss, _lse = linear_softmax_cross_entropy_loss_xla(
    x_flat,
    labels_flat,
    w_hv,
    v_block_size=v_block_size,
  )
  return loss.mean()


def make_bmgpt_loss_fn(batch, pos, vocab, block_q, block_kv, block_kv_compute):
  from jax.experimental.pallas.ops.tpu.splash_attention import (
    BlockSizes as SplashBlockSizes,
  )
  from jax.experimental.pallas.ops.tpu.splash_attention.splash_attention_mask import (
    MultiHeadMask,
  )

  from bmgpt.kernels.lse_kernel import (
    BlockSizes,
    make_lse_fused_kernel,
  )
  from bmgpt.splash_helpers import VocabMask

  max_valid_id = vocab - 2  # workaround for JAX masklib bug

  mask = MultiHeadMask(
    [VocabMask((pos, vocab), max_valid_id=max_valid_id) for _ in range(batch)]
  )

  splash_block_sizes = SplashBlockSizes(
    block_q=block_q,
    block_kv=block_kv,
    block_kv_compute=block_kv_compute,
  )
  dk_block_sizes = BlockSizes(
    block_q=block_q,
    block_kv=block_kv,
    block_kv_compute=block_kv_compute,
  )

  kernel = make_lse_fused_kernel(
    mask,
    splash_block_sizes=splash_block_sizes,
    dk_block_sizes=dk_block_sizes,
    is_mqa=True,
  )

  def bmgpt_loss(x_3d, w_vd, labels_2d):
    lse = kernel(x_3d, w_vd)  # [B, S]
    label_logits = jnp.einsum(
      "bsd,bsd->bs",
      x_3d,
      w_vd[labels_2d],
      preferred_element_type=jnp.float32,
    )
    return (lse - label_logits).mean()

  return bmgpt_loss, kernel


def levanter_loss(x_flat, w_hv, labels_flat):
  return fused_cross_entropy_loss_and_logsumexp_penalty(
    x_flat,
    labels_flat,
    w_hv,
    reduction="mean",
    logsumexp_weight=0.0,
  )


# ── Correctness check ───────────────────────────────────────────────────────


def check_correctness(implementations: dict):
  print("\n  Correctness check:")
  values = {}
  grads = {}

  for name, (fn, args) in implementations.items():
    jit_fn = jax.jit(fn)
    val = jit_fn(*args)
    jax.block_until_ready(val)
    values[name] = float(val)

    grad_fn = jax.jit(jax.grad(fn, argnums=0))
    g = grad_fn(*args)
    jax.block_until_ready(g)
    grads[name] = g

  ref_name = "reference" if "reference" in values else list(values.keys())[0]
  ref_val = values[ref_name]

  for name, val in values.items():
    diff = abs(val - ref_val)
    ok = "OK" if diff < 1e-2 else "MISMATCH"
    print(f"    {name:30s}  loss={val:.6f}  diff={diff:.2e}  [{ok}]")

  if len(grads) > 1 and ref_name in grads:
    ref_g = grads[ref_name]
    for name, g in grads.items():
      if name == ref_name:
        continue
      ref_norm = float(jnp.linalg.norm(ref_g))
      g_norm = float(jnp.linalg.norm(g))
      cos_sim = float(jnp.sum(ref_g * g)) / (ref_norm * g_norm + 1e-12)
      rel_diff = abs(ref_norm - g_norm) / (ref_norm + 1e-12)
      ok = "OK" if cos_sim > 0.98 and rel_diff < 0.05 else "CHECK"
      print(
        f"    grad {name:26s}  cos_sim={cos_sim:.4f}  "
        f"rel_norm_diff={rel_diff:.4f}  [{ok}]"
      )


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
  parser = argparse.ArgumentParser(
    description="Benchmark fused cross-entropy loss implementations"
  )
  parser.add_argument("--batch", type=int, default=16)
  parser.add_argument("--pos", type=int, default=2048)
  parser.add_argument("--embed", type=int, default=1536)
  parser.add_argument("--vocab", type=int, default=129024)
  parser.add_argument(
    "--input-dtype", choices=["bfloat16", "float32"], default="bfloat16"
  )
  parser.add_argument(
    "--accum-dtype", choices=["float32", "bfloat16"], default="float32"
  )
  parser.add_argument(
    "--implementation",
    choices=["bmgpt", "marin_xla", "reference", "levanter", "all"],
    default="all",
  )
  parser.add_argument(
    "--block-q",
    type=int,
    default=512,
    help="bmgpt kernel block_q (fused_xent_block_size_T)",
  )
  parser.add_argument(
    "--block-kv",
    type=int,
    default=256,
    help="bmgpt kernel block_kv (fused_xent_block_size_V)",
  )
  parser.add_argument(
    "--block-kv-compute", type=int, default=256, help="bmgpt kernel block_kv_compute"
  )
  parser.add_argument(
    "--v-block-size", type=int, default=0, help="Marin XLA v_block_size (0=auto 4096)"
  )
  parser.add_argument("--sweep", action="store_true", help="Sweep bmgpt block sizes")
  parser.add_argument("--steps", type=int, default=5)
  parser.add_argument("--warmup", type=int, default=3)
  args = parser.parse_args()

  jax.distributed.initialize()

  input_dtype = jnp.bfloat16 if args.input_dtype == "bfloat16" else jnp.float32
  num_tokens = args.batch * args.pos

  print("Fused CE Loss Benchmark")
  print(f"  devices: {jax.device_count()} ({jax.local_device_count()} local)")
  print(
    f"  shape: batch={args.batch}, pos={args.pos}, embed={args.embed}, "
    f"vocab={args.vocab}"
  )
  print(f"  dtype: input={args.input_dtype}, accum={args.accum_dtype}")
  print(f"  tokens: {num_tokens:,}")

  roofline_estimate(args.batch, args.pos, args.embed, args.vocab)

  key = jax.random.PRNGKey(42)
  k1, k2, k3 = jax.random.split(key, 3)

  x_3d = jax.random.normal(k1, (args.batch, args.pos, args.embed), dtype=input_dtype)
  w_vd = jax.random.normal(k2, (args.vocab, args.embed), dtype=input_dtype)
  labels_2d = jax.random.randint(
    k3, (args.batch, args.pos), minval=0, maxval=args.vocab - 2
  )

  x_flat = x_3d.reshape(args.batch * args.pos, args.embed)
  w_hv = w_vd.T  # [embed, vocab] — Marin convention is [H, V]
  labels_flat = labels_2d.ravel()

  v_block_size = args.v_block_size if args.v_block_size > 0 else 4096
  run_all = args.implementation == "all"

  impls_to_run = []
  correctness_impls = {}

  # ── Reference ────────────────────────────────────────────────────────
  if run_all or args.implementation == "reference":
    impls_to_run.append(("reference", ref_loss, (x_flat, w_hv, labels_flat)))
    correctness_impls["reference"] = (ref_loss, (x_flat, w_hv, labels_flat))

  # ── Marin XLA ────────────────────────────────────────────────────────
  if run_all or args.implementation == "marin_xla":
    marin_fn = partial(marin_xla_loss, v_block_size=v_block_size)
    impls_to_run.append(("marin_xla", marin_fn, (x_flat, w_hv, labels_flat)))
    correctness_impls["marin_xla"] = (marin_fn, (x_flat, w_hv, labels_flat))

  # ── bmgpt Pallas ─────────────────────────────────────────────────────
  if run_all or args.implementation == "bmgpt":
    if not args.sweep:
      bmgpt_fn, _kernel = make_bmgpt_loss_fn(
        args.batch,
        args.pos,
        args.vocab,
        args.block_q,
        args.block_kv,
        args.block_kv_compute,
      )
      label = f"bmgpt(q={args.block_q},kv={args.block_kv},kvc={args.block_kv_compute})"
      impls_to_run.append((label, bmgpt_fn, (x_3d, w_vd, labels_2d)))
      correctness_impls[label] = (bmgpt_fn, (x_3d, w_vd, labels_2d))

  # ── Levanter Pallas (optional) ───────────────────────────────────────
  if (run_all or args.implementation == "levanter") and HAS_LEVANTER:
    impls_to_run.append(("levanter_pallas", levanter_loss, (x_flat, w_hv, labels_flat)))
    correctness_impls["levanter_pallas"] = (levanter_loss, (x_flat, w_hv, labels_flat))
  elif args.implementation == "levanter" and not HAS_LEVANTER:
    print("\n  WARNING: levanter not installed, skipping levanter_pallas")

  # ── Correctness ──────────────────────────────────────────────────────
  if len(correctness_impls) > 1:
    check_correctness(correctness_impls)

  # ── Benchmark ────────────────────────────────────────────────────────
  print(f"\n  Benchmark ({args.warmup} warmup, {args.steps} steps):")

  for name, fn, fn_args in impls_to_run:
    jit_fwd = jax.jit(fn)
    ct, rts = time_fn(jit_fwd, *fn_args, warmup=args.warmup, steps=args.steps)
    report(name, ct, rts, num_tokens, "fwd")

    jit_vg = jax.jit(jax.value_and_grad(fn, argnums=0))
    ct, rts = time_fn(jit_vg, *fn_args, warmup=args.warmup, steps=args.steps)
    report(name, ct, rts, num_tokens, "fwd+bwd")

  # ── Block size sweep mode ────────────────────────────────────────────
  if args.sweep and (run_all or args.implementation == "bmgpt"):
    print("\n  Block size sweep (bmgpt Pallas):")
    print(
      f"  {'block_q':>8s} {'block_kv':>8s} {'block_kvc':>9s} "
      f"{'fwd_ms':>8s} {'fwd+bwd_ms':>10s} {'fwd_tok/s':>12s} "
      f"{'bwd_tok/s':>12s}"
    )
    print("  " + "-" * 80)

    block_q_opts = [128, 256, 512]
    block_kv_opts = [128, 256, 384, 512]

    for bq in block_q_opts:
      for bkv in block_kv_opts:
        if (args.batch * args.pos) % bq != 0:
          continue
        if args.vocab % bkv != 0:
          continue

        bkvc = bkv  # sweep compute block same as memory block
        try:
          sweep_fn, _ = make_bmgpt_loss_fn(
            args.batch,
            args.pos,
            args.vocab,
            bq,
            bkv,
            bkvc,
          )
        except Exception as e:
          print(f"  {bq:>8d} {bkv:>8d} {bkvc:>9d}  SKIP: {e}")
          continue

        jit_fwd = jax.jit(sweep_fn)
        ct_fwd, rts_fwd = time_fn(
          jit_fwd,
          x_3d,
          w_vd,
          labels_2d,
          warmup=args.warmup,
          steps=args.steps,
        )
        avg_fwd = sum(rts_fwd) / len(rts_fwd)

        jit_vg = jax.jit(jax.value_and_grad(sweep_fn, argnums=0))
        ct_bwd, rts_bwd = time_fn(
          jit_vg,
          x_3d,
          w_vd,
          labels_2d,
          warmup=args.warmup,
          steps=args.steps,
        )
        avg_bwd = sum(rts_bwd) / len(rts_bwd)

        print(
          f"  {bq:>8d} {bkv:>8d} {bkvc:>9d} "
          f"{avg_fwd * 1e3:>8.2f} {avg_bwd * 1e3:>10.2f} "
          f"{num_tokens / avg_fwd:>12,.0f} "
          f"{num_tokens / avg_bwd:>12,.0f}"
        )

    print(f"\n  Marin XLA baseline (v_block_size={v_block_size}):")
    marin_fn = partial(marin_xla_loss, v_block_size=v_block_size)
    jit_fwd = jax.jit(marin_fn)
    ct, rts = time_fn(
      jit_fwd, x_flat, w_hv, labels_flat, warmup=args.warmup, steps=args.steps
    )
    report("marin_xla", ct, rts, num_tokens, "fwd")
    jit_vg = jax.jit(jax.value_and_grad(marin_fn, argnums=0))
    ct, rts = time_fn(
      jit_vg, x_flat, w_hv, labels_flat, warmup=args.warmup, steps=args.steps
    )
    report("marin_xla", ct, rts, num_tokens, "fwd+bwd")

  print("\nDone.")


if __name__ == "__main__":
  main()
