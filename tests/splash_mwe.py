"""
Minimal Working Example: JAX 0.8.1 shard_map + vmap interaction bug with splash attention

This MWE demonstrates a regression in JAX 0.8.1 where using shard_map inside a vmapped
function (with value_and_grad) causes sharding assertion failures in the backward pass.

The pattern that breaks:
    value_and_grad(vmap(fn_that_internally_calls_shard_map))

The error manifests as a sharding mismatch like:
    AssertionError: (ShapedArray(bfloat16[B,3,N,S,H]), ShapedArray(bfloat16[B@dp,3,N,S,H]))

This worked in JAX 0.7.2 but fails in JAX 0.8.1.

Environment:
- JAX 0.8.1
- Single-host TPU v4 with 4 chips, or multi-host (e.g., v4-16 with 4 hosts)

Usage:
    # Single-host:
    python tests/splash_mwe.py

    # Multi-host (via gcloud):
    ./deploy/run_mwe.sh <TPU_NAME>
"""

from functools import partial

import jax
import jax.numpy as jnp
from jax.experimental.pallas.ops.tpu.splash_attention import (
  BlockSizes,
  CausalMask,
  MultiHeadMask,
  SegmentIds,
  make_splash_mha,
)

# Configuration - BATCH_SIZE will be set dynamically based on device count
NUM_HEADS = 4
SEQ_LEN = 256
HEAD_DIM = 64
DTYPE = jnp.bfloat16


def make_splash_kernel_with_shard_map(mesh):
  mask = MultiHeadMask([CausalMask(shape=(SEQ_LEN, SEQ_LEN)) for _ in range(NUM_HEADS)])
  block_sizes = BlockSizes(
    block_q=128,
    block_kv=128,
    block_kv_compute=128,
    block_q_dkv=128,
    block_kv_dkv=128,
    block_kv_dkv_compute=128,
    block_q_dq=128,
    block_kv_dq=128,
  )

  # Sharding spec for splash attention (heads, seq dimensions only - no batch)
  splash_spec = jax.sharding.PartitionSpec(None, None)
  splash_sharding = jax.sharding.NamedSharding(mesh, splash_spec)

  kernel = make_splash_mha(
    mask,
    head_shards=1,
    q_seq_shards=1,
    block_sizes=block_sizes,
  )
  kernel_spec = kernel.manual_sharding_spec(splash_sharding)

  @partial(
    jax.shard_map,
    mesh=mesh,
    in_specs=(
      kernel_spec,
      splash_spec,
      splash_spec,
      splash_spec,
      jax.sharding.PartitionSpec(),
    ),
    out_specs=splash_spec,
    check_vma=False,
  )
  def splash_sharded(kernel, q, k, v, segment_ids):
    return kernel(q, k, v, segment_ids=segment_ids)

  return splash_sharded, kernel


def attention_fn_with_internal_shard_map(splash_sharded, kernel, key, x_seq):
  s = x_seq.shape[0]

  q, k, v = jnp.ones((NUM_HEADS, s, HEAD_DIM), out_sharding=jax.P())
  segment_ids = SegmentIds(q=jnp.zeros((s,)), kv=jnp.zeros((s,)))

  # Scale Q and K as splash attention expects
  scale = HEAD_DIM**-0.25
  out = splash_sharded(kernel, q * scale, k * scale, v, segment_ids)
  return out


def test_case_fails_vmap_outside_shard_map(mesh, batch_size):
  print("\n" + "=" * 70)
  print("CASE B: vmap OUTSIDE shard_map (expected to FAIL in JAX 0.8.1)")
  print("=" * 70)

  splash_sharded, kernel = make_splash_kernel_with_shard_map(mesh)

  # Create batched inputs matching the actual code structure
  # x_seq shape: (batch, seq_len, d_model)
  # w_qkv shape: (d_model, 3, num_heads, head_dim)
  d_model = NUM_HEADS * HEAD_DIM

  key = jax.random.key(0)
  k1, k2 = jax.random.split(key, 2)

  # Input sequence sharding: batch dimension sharded
  input_sharding = jax.sharding.NamedSharding(
    mesh, jax.sharding.PartitionSpec("dp", None, None)
  )

  # Weight sharding: replicated (matching typical parameter sharding)
  weight_sharding = jax.sharding.NamedSharding(
    mesh, jax.sharding.PartitionSpec(None, None, None, None)
  )

  # Shape: (batch, seq_len, d_model)
  x_seq = jax.random.normal(k1, (batch_size, SEQ_LEN, d_model), dtype=DTYPE)
  x_seq = jax.device_put(x_seq, input_sharding)

  @jax.jit
  def step(x_seq):
    def loss_fn(x_seq):
      # vmap over batch dimension, calling shard_map-wrapped kernel inside
      # The einsum happens inside each vmapped call
      attn_fn = partial(
        attention_fn_with_internal_shard_map, splash_sharded, kernel, k2
      )
      out = jax.vmap(attn_fn)(x_seq)  # vmap adds batch dim
      return out.sum()

    loss, grads = jax.value_and_grad(loss_fn, argnums=(0, 1))(w_qkv, x_seq)
    return loss, grads

  print(f"Input shapes: x_seq={x_seq.shape}, w_qkv={w_qkv.shape}")
  print(f"Input sharding: x_seq={x_seq.sharding}, w_qkv={w_qkv.sharding}")
  print("Running value_and_grad(vmap(fn_with_shard_map_inside))...")

  try:
    with jax.set_mesh(mesh):
      loss, grads = step(w_qkv, x_seq)
    print(f"SUCCESS: loss={loss}, grad shapes={[g.shape for g in grads]}")
    return True
  except AssertionError as e:
    print(f"FAILED with AssertionError: {e}")
    return False
  except Exception as e:
    print(f"FAILED with {type(e).__name__}: {e}")
    return False


def main():
  jax.distributed.initialize()

  print("=" * 70)
  print("JAX Splash Attention + shard_map + vmap Bug MWE")
  print("=" * 70)
  print(f"JAX version: {jax.__version__}")
  print(f"Process index: {jax.process_index()} / {jax.process_count()}")
  print(f"Local devices: {jax.local_devices()}")
  print(f"All devices: {jax.devices()}")
  print(f"Global device count: {jax.device_count()}")

  # Create a simple 1D data-parallel mesh across all devices
  devices = jax.devices()
  mesh = jax.sharding.Mesh(devices, ("dp",), (jax.sharding.AxisType.Explicit,))
  print(f"Mesh: {mesh}")

  # Batch size = number of devices (one example per device)
  batch_size = jax.device_count()

  print(f"\nTest configuration:")
  print(f"  batch_size={batch_size} (1 per device), NUM_HEADS={NUM_HEADS}")
  print(f"  SEQ_LEN={SEQ_LEN}, HEAD_DIM={HEAD_DIM}")

  results = {}

  # Test Case B (vmap outside shard_map - expected to fail in 0.8.1)
  results["case_b_vmap_outside"] = test_case_fails_vmap_outside_shard_map(
    mesh, batch_size
  )

  # Summary
  print("\n" + "=" * 70)
  print("SUMMARY")
  print("=" * 70)
  for name, passed in results.items():
    status = "PASSED ✓" if passed else "FAILED ✗"
    print(f"  {name}: {status}")


if __name__ == "__main__":
  main()
