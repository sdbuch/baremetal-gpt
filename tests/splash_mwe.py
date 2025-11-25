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
- Single-host TPU v4 with 4 chips (can also reproduce on multi-host)

Usage:
    python splash_mwe.py
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

# Configuration
BATCH_SIZE = 4
NUM_HEADS = 4
SEQ_LEN = 256
HEAD_DIM = 64
DTYPE = jnp.bfloat16


def make_splash_kernel_with_shard_map(mesh):
  """
  Creates a splash attention kernel wrapped with shard_map.

  This mirrors the pattern in typical splash attention usage where
  the kernel is wrapped with shard_map for TPU execution.

  The kernel expects inputs of shape (num_heads, seq_len, head_dim).
  """
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
    check_rep=False,
  )
  def splash_sharded(kernel, q, k, v, segment_ids):
    return kernel(q, k, v, segment_ids=segment_ids)

  return splash_sharded, kernel


def attention_fn_with_internal_shard_map(splash_sharded, kernel, q, k, v):
  """
  Attention function that internally calls the shard_map-wrapped kernel.

  Input shapes: (num_heads, seq_len, head_dim)
  This is designed to be vmapped over a batch dimension.
  """
  s = q.shape[1]
  segment_ids = SegmentIds(q=jnp.zeros((s,)), kv=jnp.zeros((s,)))

  # Scale Q and K as splash attention expects
  scale = HEAD_DIM**-0.25
  out = splash_sharded(kernel, q * scale, k * scale, v, segment_ids)
  return out


def test_case_fails_vmap_outside_shard_map(mesh):
  """
  CASE B: EXPECTED TO FAIL in JAX 0.8.1

  Pattern: value_and_grad(vmap(fn_that_calls_shard_map))

  The shard_map is defined with specs for (heads, seq) dimensions only.
  vmap adds a batch dimension externally.
  In the backward pass, sharding assertions fail.
  """
  print("\n" + "=" * 70)
  print("CASE B: vmap OUTSIDE shard_map (expected to FAIL in JAX 0.8.1)")
  print("=" * 70)

  splash_sharded, kernel = make_splash_kernel_with_shard_map(mesh)

  # Create batched inputs with data-parallel sharding on batch dimension
  key = jax.random.key(0)
  k1, k2, k3, k4 = jax.random.split(key, 4)

  data_sharding = jax.sharding.NamedSharding(
    mesh, jax.sharding.PartitionSpec("dp", None, None, None)
  )

  # Shape: (batch, num_heads, seq_len, head_dim)
  q = jax.random.normal(k1, (BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM), dtype=DTYPE)
  k = jax.random.normal(k2, (BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM), dtype=DTYPE)
  v = jax.random.normal(k3, (BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM), dtype=DTYPE)

  # Shard on batch dimension
  q = jax.device_put(q, data_sharding)
  k = jax.device_put(k, data_sharding)
  v = jax.device_put(v, data_sharding)

  def loss_fn(q, k, v):
    # vmap over batch dimension, calling shard_map-wrapped kernel inside
    attn_fn = partial(attention_fn_with_internal_shard_map, splash_sharded, kernel)
    out = jax.vmap(attn_fn)(q, k, v)  # vmap adds batch dim
    return out.sum()

  print(f"Input shapes: q={q.shape}, k={k.shape}, v={v.shape}")
  print(f"Input sharding: {q.sharding}")
  print("Running value_and_grad(vmap(fn_with_shard_map_inside))...")

  try:
    loss, grads = jax.value_and_grad(loss_fn, argnums=(0, 1, 2))(q, k, v)
    print(f"SUCCESS: loss={loss}, grad shapes={[g.shape for g in grads]}")
    return True
  except AssertionError as e:
    print(f"FAILED with AssertionError: {e}")
    return False
  except Exception as e:
    print(f"FAILED with {type(e).__name__}: {e}")
    return False


def test_case_works_batch_inside(mesh):
  """
  CASE A: EXPECTED TO WORK

  Pattern: Handle batch dimension inside the function, no external vmap over shard_map.

  This approach processes the full batch at once inside the shard_map,
  rather than vmapping over individual examples.
  """
  print("\n" + "=" * 70)
  print("CASE A: Batch handled INSIDE (no vmap over shard_map) - expected to WORK")
  print("=" * 70)

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

  # Sharding spec includes batch dimension
  # (batch, heads, seq) - batch sharded on "dp"
  batched_splash_spec = jax.sharding.PartitionSpec("dp", None, None)
  batched_splash_sharding = jax.sharding.NamedSharding(mesh, batched_splash_spec)

  kernel = make_splash_mha(
    mask,
    head_shards=1,
    q_seq_shards=1,
    block_sizes=block_sizes,
  )

  # Get kernel spec for non-batched inputs, then we'll vmap inside shard_map
  unbatched_sharding = jax.sharding.NamedSharding(
    mesh, jax.sharding.PartitionSpec(None, None)
  )
  kernel_spec = kernel.manual_sharding_spec(unbatched_sharding)

  @partial(
    jax.shard_map,
    mesh=mesh,
    in_specs=(
      kernel_spec,
      batched_splash_spec,
      batched_splash_spec,
      batched_splash_spec,
      jax.sharding.PartitionSpec("dp"),  # segment_ids batched
    ),
    out_specs=batched_splash_spec,
    check_rep=False,
  )
  def splash_batched_sharded(kernel, q, k, v, segment_ids):
    # vmap INSIDE shard_map over batch dimension
    def single_example(q_i, k_i, v_i, seg_i):
      return kernel(q_i, k_i, v_i, segment_ids=seg_i)

    return jax.vmap(single_example)(q, k, v, segment_ids)

  # Create batched inputs
  key = jax.random.key(0)
  k1, k2, k3, k4 = jax.random.split(key, 4)

  data_sharding = jax.sharding.NamedSharding(
    mesh, jax.sharding.PartitionSpec("dp", None, None, None)
  )

  # Shape: (batch, num_heads, seq_len, head_dim)
  q = jax.random.normal(k1, (BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM), dtype=DTYPE)
  k = jax.random.normal(k2, (BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM), dtype=DTYPE)
  v = jax.random.normal(k3, (BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM), dtype=DTYPE)

  q = jax.device_put(q, data_sharding)
  k = jax.device_put(k, data_sharding)
  v = jax.device_put(v, data_sharding)

  def loss_fn(q, k, v):
    s = q.shape[2]
    # Batched segment IDs
    segment_ids = SegmentIds(
      q=jnp.zeros((BATCH_SIZE, s)),
      kv=jnp.zeros((BATCH_SIZE, s)),
    )
    scale = HEAD_DIM**-0.25
    out = splash_batched_sharded(kernel, q * scale, k * scale, v, segment_ids)
    return out.sum()

  print(f"Input shapes: q={q.shape}, k={k.shape}, v={v.shape}")
  print(f"Input sharding: {q.sharding}")
  print("Running value_and_grad with batch handled inside shard_map...")

  try:
    loss, grads = jax.value_and_grad(loss_fn, argnums=(0, 1, 2))(q, k, v)
    print(f"SUCCESS: loss={loss}, grad shapes={[g.shape for g in grads]}")
    return True
  except AssertionError as e:
    print(f"FAILED with AssertionError: {e}")
    return False
  except Exception as e:
    print(f"FAILED with {type(e).__name__}: {e}")
    return False


def test_case_works_no_shard_map(mesh):
  """
  CASE C: Reference - no shard_map at all (pure JAX attention)

  This confirms the issue is specifically with shard_map + vmap interaction,
  not with splash attention or vmap in general.
  """
  print("\n" + "=" * 70)
  print("CASE C: Pure JAX attention (no shard_map) - expected to WORK")
  print("=" * 70)

  key = jax.random.key(0)
  k1, k2, k3, k4 = jax.random.split(key, 4)

  data_sharding = jax.sharding.NamedSharding(
    mesh, jax.sharding.PartitionSpec("dp", None, None, None)
  )

  q = jax.random.normal(k1, (BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM), dtype=DTYPE)
  k = jax.random.normal(k2, (BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM), dtype=DTYPE)
  v = jax.random.normal(k3, (BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM), dtype=DTYPE)

  q = jax.device_put(q, data_sharding)
  k = jax.device_put(k, data_sharding)
  v = jax.device_put(v, data_sharding)

  def pure_attention(q, k, v):
    """Standard scaled dot-product attention."""
    scale = HEAD_DIM**-0.5
    logits = jnp.einsum("nsh,nth->nst", q, k) * scale
    # Causal mask
    mask = jnp.tril(jnp.ones((SEQ_LEN, SEQ_LEN), dtype=bool))
    logits = jnp.where(mask, logits, -jnp.inf)
    probs = jax.nn.softmax(logits, axis=-1)
    return jnp.einsum("nst,nth->nsh", probs, v)

  def loss_fn(q, k, v):
    out = jax.vmap(pure_attention)(q, k, v)
    return out.sum()

  print(f"Input shapes: q={q.shape}, k={k.shape}, v={v.shape}")
  print(f"Input sharding: {q.sharding}")
  print("Running value_and_grad(vmap(pure_attention))...")

  try:
    loss, grads = jax.value_and_grad(loss_fn, argnums=(0, 1, 2))(q, k, v)
    print(f"SUCCESS: loss={loss}, grad shapes={[g.shape for g in grads]}")
    return True
  except AssertionError as e:
    print(f"FAILED with AssertionError: {e}")
    return False
  except Exception as e:
    print(f"FAILED with {type(e).__name__}: {e}")
    return False


def main():
  print("=" * 70)
  print("JAX Splash Attention + shard_map + vmap Bug MWE")
  print("=" * 70)
  print(f"JAX version: {jax.__version__}")
  print(f"Devices: {jax.devices()}")
  print(f"Device count: {jax.device_count()}")

  # Create a simple 1D data-parallel mesh
  devices = jax.devices()
  mesh = jax.sharding.Mesh(devices, ("dp",))
  print(f"Mesh: {mesh}")
  print(f"\nTest configuration:")
  print(f"  BATCH_SIZE={BATCH_SIZE}, NUM_HEADS={NUM_HEADS}")
  print(f"  SEQ_LEN={SEQ_LEN}, HEAD_DIM={HEAD_DIM}")

  results = {}

  # Test Case C first (baseline - should always work)
  results["case_c_no_shard_map"] = test_case_works_no_shard_map(mesh)

  # Test Case A (batch inside shard_map - should work)
  results["case_a_batch_inside"] = test_case_works_batch_inside(mesh)

  # Test Case B (vmap outside shard_map - expected to fail in 0.8.1)
  results["case_b_vmap_outside"] = test_case_fails_vmap_outside_shard_map(mesh)

  # Summary
  print("\n" + "=" * 70)
  print("SUMMARY")
  print("=" * 70)
  for name, passed in results.items():
    status = "PASSED ✓" if passed else "FAILED ✗"
    print(f"  {name}: {status}")

  print("\n" + "=" * 70)
  if not results["case_b_vmap_outside"] and results["case_a_batch_inside"]:
    print("BUG CONFIRMED: vmap outside shard_map fails while batch-inside works.")
    print("This is a regression from JAX 0.7.2 -> 0.8.1")
  elif all(results.values()):
    print("All tests passed - bug may not reproduce in this configuration.")
  else:
    print("Unexpected failure pattern - investigate further.")
  print("=" * 70)


if __name__ == "__main__":
  main()
