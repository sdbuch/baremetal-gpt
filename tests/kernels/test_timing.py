"""Timing tests for fused cross entropy kernel vs reference implementations."""

import time
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from bmgpt.losses import fused_softmax_cross_entropy
from bmgpt.model import LMHead
from bmgpt.splash_helpers import make_lse_kernel_sharded

from .conftest import (
  BFLOAT16_ATOL,
  BFLOAT16_RTOL,
  DEFAULT_BLOCK_SIZE,
  FORWARD_ATOL,
  FORWARD_RTOL,
  SMALL_CONFIG_IDS,
  SMALL_CONFIGS,
  make_test_config,
  requires_tpu,
)

# =============================================================================
# Reference Implementations for Timing Comparison
# =============================================================================


def _scanned_logsumexp_single_block(
  outputs_block: jax.Array,
  w_unemb_padded: jax.Array,
  block_size_v: int,
  vocab_size: int,
  num_blocks_v: int,
) -> jax.Array:
  """Compute logsumexp for a single block of tokens over all vocab blocks.

  Args:
    outputs_block: Input activations of shape (block_size_n, D)
    w_unemb_padded: Padded unembedding weights of shape (V_padded, D)
    block_size_v: Block size for vocabulary dimension
    vocab_size: Original vocabulary size (for masking)
    num_blocks_v: Number of vocabulary blocks

  Returns:
    logsumexp values of shape (block_size_n,)
  """
  block_size_n, d = outputs_block.shape

  def scan_body(carry, block_idx):
    """Process one block of vocabulary."""
    m_prev, l_prev = carry

    start_idx = block_idx * block_size_v
    w_block = jax.lax.dynamic_slice(w_unemb_padded, (start_idx, 0), (block_size_v, d))
    logits_block = jnp.dot(outputs_block, w_block.T)

    # Mask out padding positions
    vocab_indices = start_idx + jnp.arange(block_size_v)
    valid_mask = vocab_indices < vocab_size
    logits_block = jnp.where(valid_mask[None, :], logits_block, -jnp.inf)

    # Online logsumexp update
    m_curr = logits_block.max(axis=-1)
    m_next = jnp.maximum(m_prev, m_curr)
    s_curr = jnp.exp(logits_block - m_next[:, None]).sum(axis=-1)
    alpha = jnp.exp(m_prev - m_next)
    l_next = s_curr + alpha * l_prev

    return (m_next, l_next), None

  m_init = -jnp.inf * jnp.ones((block_size_n,), dtype=jnp.float32, out_sharding=jax.P())
  l_init = jnp.zeros((block_size_n,), dtype=jnp.float32, out_sharding=jax.P())

  (m_final, l_final), _ = jax.lax.scan(
    scan_body, (m_init, l_init), jnp.arange(num_blocks_v)
  )

  return jnp.log(l_final) + m_final


def scanned_logsumexp_blocked(
  outputs: jax.Array,
  w_unemb: jax.Array,
  block_size_v: int = DEFAULT_BLOCK_SIZE,
  block_size_n: int | None = None,
  data_sharding: list[str | None] | None = None,
) -> jax.Array:
  """Compute logsumexp(outputs @ w_unemb.T) using a scan over vocab blocks.

  This reference implementation avoids materializing the full (N, V) logits matrix
  by scanning over blocks of the vocabulary dimension, maintaining running max and
  sum-of-exp accumulators (online logsumexp).

  Args:
    outputs: Input activations of shape (N, D)
    w_unemb: Unembedding weights of shape (V, D)
    block_size_v: Block size for scanning over vocabulary dimension
    block_size_n: Block size for token dimension. If None, processes all tokens
                  together. If specified, vmaps over token blocks for better
                  comparison with fused kernels.
    data_sharding: Optional sharding spec for data dimension. When block_size_n
                   is specified, the leading (num_blocks_n) axis will be sharded.

  Returns:
    logsumexp values of shape (N,)
  """
  n, d = outputs.shape
  v, _ = w_unemb.shape

  # Pad w_unemb to be a multiple of block_size_v
  v_padded = ((v + block_size_v - 1) // block_size_v) * block_size_v
  pad_amount_v = v_padded - v
  if pad_amount_v > 0:
    w_unemb_padded = jnp.pad(w_unemb, ((0, pad_amount_v), (0, 0)), mode="constant")
  else:
    w_unemb_padded = w_unemb

  num_blocks_v = v_padded // block_size_v

  if block_size_n is None:
    # Original behavior: process all tokens together
    return _scanned_logsumexp_single_block(
      outputs, w_unemb_padded, block_size_v, v, num_blocks_v
    )

  # Block over token dimension: pad N, reshape, vmap, reshape back
  n_padded = ((n + block_size_n - 1) // block_size_n) * block_size_n
  pad_amount_n = n_padded - n
  if pad_amount_n > 0:
    outputs_padded = jnp.pad(outputs, ((0, pad_amount_n), (0, 0)), mode="constant")
  else:
    outputs_padded = outputs

  num_blocks_n = n_padded // block_size_n

  # Reshape to (num_blocks_n, block_size_n, D)
  # When sharded, the N axis is sharded across data_sharding, so after reshape
  # the leading num_blocks_n axis should be sharded
  if data_sharding:
    outputs_blocked = outputs_padded.reshape(
      num_blocks_n, block_size_n, d, out_sharding=jax.P(*data_sharding, None, None)
    )
  else:
    outputs_blocked = outputs_padded.reshape(num_blocks_n, block_size_n, d)

  print(jax.typeof(outputs_blocked))
  print(jax.typeof(w_unemb_padded))
  # vmap over token blocks
  vmapped_lse = jax.vmap(
    lambda x: _scanned_logsumexp_single_block(
      x, w_unemb_padded, block_size_v, v, num_blocks_v
    )
  )(outputs_blocked)

  # Reshape back to (N_padded,) and slice to (N,)
  if data_sharding:
    lse_flat = vmapped_lse.reshape(n_padded, out_sharding=jax.P(*data_sharding))
  else:
    lse_flat = vmapped_lse.reshape(n_padded)
  return lse_flat[:n]


def scanned_cross_entropy(
  outputs: jax.Array,
  w_unemb: jax.Array,
  targets: jax.Array,
  block_size_v: int = DEFAULT_BLOCK_SIZE,
  block_size_n: int | None = None,
  data_sharding: list[str | None] | None = None,
) -> jax.Array:
  """Cross entropy using scanned logsumexp reference.

  Args:
    outputs: (B, S, D) or (N, D) input activations
    w_unemb: (V, D) unembedding weights
    targets: (B, S) or (N,) target token indices
    block_size_v: Block size for vocab dimension scan
    block_size_n: Block size for token dimension. If None, processes all tokens
                  together. If specified, vmaps over token blocks for better
                  comparison with fused kernels (equivalent to block_q).
    data_sharding: Optional sharding spec for data dimension

  Returns:
    Scalar mean cross entropy loss
  """
  # Only reshard if we have an active mesh context (e.g., on TPU)
  if data_sharding is not None:
    w_unemb = jax.sharding.reshard(w_unemb, out_shardings=jax.P())

  # Flatten if needed
  if outputs.ndim == 3:
    b, s, d = outputs.shape
    if data_sharding:
      outputs_flat = outputs.reshape(b * s, d, out_sharding=jax.P(*data_sharding, None))
      targets_flat = targets.ravel(out_sharding=jax.P(*data_sharding))
    else:
      outputs_flat = outputs.reshape(b * s, d)
      targets_flat = targets.ravel()
  else:
    outputs_flat = outputs
    targets_flat = targets
  print(jax.typeof(outputs))
  print(jax.typeof(w_unemb))

  lse = scanned_logsumexp_blocked(
    outputs_flat, w_unemb, block_size_v, block_size_n, data_sharding
  )

  if data_sharding:
    per_token_unembs = w_unemb.at[targets_flat].get(
      out_sharding=jax.P(*data_sharding, None)
    )
  else:
    per_token_unembs = w_unemb[targets_flat]
  label_logits = jnp.sum(outputs_flat * per_token_unembs, axis=-1)

  return (lse - label_logits).mean()


def ref_cross_entropy_materialized(
  outputs: jax.Array,
  w_unemb: jax.Array,
  targets: jax.Array,
  data_sharding: list[str | None] | None = None,
) -> jax.Array:
  """Reference cross-entropy that materializes the full logits matrix.

  Args:
    outputs: (B, S, D) or (N, D) input activations
    w_unemb: (V, D) unembedding weights
    targets: (B, S) or (N,) target token indices
    data_sharding: Optional sharding spec for data dimension

  Returns:
    Scalar mean cross entropy loss
  """
  # Only reshard if we have an active mesh context (e.g., on TPU)
  if data_sharding is not None:
    w_unemb = jax.sharding.reshard(w_unemb, out_shardings=jax.P())

  if outputs.ndim == 3:
    b, s, d = outputs.shape
    if data_sharding:
      outputs_flat = outputs.reshape(b * s, d, out_sharding=jax.P(*data_sharding, None))
      targets_flat = targets.ravel(out_sharding=jax.P(*data_sharding))
    else:
      outputs_flat = outputs.reshape(b * s, d)
      targets_flat = targets.ravel()
  else:
    outputs_flat = outputs
    targets_flat = targets

  logits = jnp.dot(outputs_flat, w_unemb.T)
  label_logits = jnp.take_along_axis(logits, targets_flat[:, None], axis=-1).squeeze(-1)
  lse = jax.nn.logsumexp(logits, axis=-1)
  return (lse - label_logits).mean()


# =============================================================================
# Timing Utilities
# =============================================================================


def time_fn(fn, *args, warmup_iters: int = 3, time_iters: int = 10):
  """Time a function with warmup iterations."""
  for _ in range(warmup_iters):
    result = fn(*args)
    jax.block_until_ready(result)

  start = time.perf_counter()
  for _ in range(time_iters):
    result = fn(*args)
    jax.block_until_ready(result)
  end = time.perf_counter()

  return (end - start) / time_iters, result


# =============================================================================
# Correctness Tests - verify scanned reference matches materialized reference
# =============================================================================


@pytest.mark.parametrize(
  "batch_size,seq_len,d_model,vocab_size", SMALL_CONFIGS, ids=SMALL_CONFIG_IDS
)
@pytest.mark.parametrize(
  "block_size_v", [128, 256, 512], ids=["blk128", "blk256", "blk512"]
)
def test_scanned_reference_correctness(
  batch_size, seq_len, d_model, vocab_size, block_size_v
):
  """Verify scanned logsumexp reference matches materialized logits reference."""
  key = jax.random.PRNGKey(42)
  key_out, key_w, key_tgt = jax.random.split(key, 3)

  outputs = jax.random.normal(
    key_out, (batch_size, seq_len, d_model), dtype=jnp.float32
  )
  w_unemb = jax.random.normal(key_w, (vocab_size, d_model), dtype=jnp.float32)
  targets = jax.random.randint(key_tgt, (batch_size, seq_len), 0, vocab_size)

  loss_ref = jax.jit(ref_cross_entropy_materialized)(outputs, w_unemb, targets)
  loss_scanned = jax.jit(partial(scanned_cross_entropy, block_size_v=block_size_v))(
    outputs, w_unemb, targets
  )

  np.testing.assert_allclose(
    loss_scanned,
    loss_ref,
    rtol=FORWARD_RTOL,
    atol=FORWARD_ATOL,
    err_msg=f"Scanned (block_size={block_size_v}) does not match materialized reference",
  )


@pytest.mark.parametrize("block_size_v", [128, 256, 512, 1024])
def test_scanned_logsumexp_correctness(block_size_v):
  """Verify scanned logsumexp matches jax.nn.logsumexp on raw logits."""
  key = jax.random.PRNGKey(123)
  key_out, key_w = jax.random.split(key, 2)

  n, d, v = 256, 128, 2048
  outputs = jax.random.normal(key_out, (n, d), dtype=jnp.float32)
  w_unemb = jax.random.normal(key_w, (v, d), dtype=jnp.float32)

  @jax.jit
  def ref_lse(outputs, w_unemb):
    logits = jnp.dot(outputs, w_unemb.T)
    return jax.nn.logsumexp(logits, axis=-1)

  lse_ref = ref_lse(outputs, w_unemb)
  lse_scanned = jax.jit(partial(scanned_logsumexp_blocked, block_size_v=block_size_v))(
    outputs, w_unemb
  )

  np.testing.assert_allclose(
    lse_scanned,
    lse_ref,
    rtol=FORWARD_RTOL,
    atol=FORWARD_ATOL,
    err_msg=f"Scanned logsumexp (block_size={block_size_v}) does not match reference",
  )


@pytest.mark.parametrize("block_size_v", [128, 256, 512])
def test_scanned_reference_partial_blocks(block_size_v):
  """Test scanned reference handles partial final blocks correctly."""
  key = jax.random.PRNGKey(456)
  key_out, key_w, key_tgt = jax.random.split(key, 3)

  batch_size, seq_len, d_model = 2, 64, 128
  vocab_size = block_size_v * 3 + block_size_v // 2

  outputs = jax.random.normal(
    key_out, (batch_size, seq_len, d_model), dtype=jnp.float32
  )
  w_unemb = jax.random.normal(key_w, (vocab_size, d_model), dtype=jnp.float32)
  targets = jax.random.randint(key_tgt, (batch_size, seq_len), 0, vocab_size)

  loss_ref = jax.jit(ref_cross_entropy_materialized)(outputs, w_unemb, targets)
  loss_scanned = jax.jit(partial(scanned_cross_entropy, block_size_v=block_size_v))(
    outputs, w_unemb, targets
  )

  np.testing.assert_allclose(
    loss_scanned,
    loss_ref,
    rtol=FORWARD_RTOL,
    atol=FORWARD_ATOL,
    err_msg=f"Partial block handling failed for vocab={vocab_size}, block={block_size_v}",
  )


@pytest.mark.parametrize(
  "block_size_n,block_size_v",
  [(64, 128), (128, 128), (128, 256), (256, 512)],
  ids=["n64_v128", "n128_v128", "n128_v256", "n256_v512"],
)
def test_scanned_reference_with_block_n(block_size_n, block_size_v):
  """Test scanned reference with blocking over both token and vocab dimensions."""
  key = jax.random.PRNGKey(789)
  key_out, key_w, key_tgt = jax.random.split(key, 3)

  batch_size, seq_len, d_model = 4, 256, 128
  vocab_size = 2048

  outputs = jax.random.normal(
    key_out, (batch_size, seq_len, d_model), dtype=jnp.float32
  )
  w_unemb = jax.random.normal(key_w, (vocab_size, d_model), dtype=jnp.float32)
  targets = jax.random.randint(key_tgt, (batch_size, seq_len), 0, vocab_size)

  loss_ref = jax.jit(ref_cross_entropy_materialized)(outputs, w_unemb, targets)

  # Test with block_size_n
  loss_scanned_blocked = jax.jit(
    partial(scanned_cross_entropy, block_size_v=block_size_v, block_size_n=block_size_n)
  )(outputs, w_unemb, targets)

  np.testing.assert_allclose(
    loss_scanned_blocked,
    loss_ref,
    rtol=FORWARD_RTOL,
    atol=FORWARD_ATOL,
    err_msg=f"Token-blocked (n={block_size_n}, v={block_size_v}) does not match reference",
  )


@pytest.mark.parametrize("block_size_n", [64, 128, 256])
def test_scanned_logsumexp_with_block_n(block_size_n):
  """Verify scanned logsumexp with token blocking matches reference."""
  key = jax.random.PRNGKey(321)
  key_out, key_w = jax.random.split(key, 2)

  n, d, v = 512, 128, 2048
  block_size_v = 256
  outputs = jax.random.normal(key_out, (n, d), dtype=jnp.float32)
  w_unemb = jax.random.normal(key_w, (v, d), dtype=jnp.float32)

  @jax.jit
  def ref_lse(outputs, w_unemb):
    logits = jnp.dot(outputs, w_unemb.T)
    return jax.nn.logsumexp(logits, axis=-1)

  lse_ref = ref_lse(outputs, w_unemb)
  lse_scanned = jax.jit(
    partial(
      scanned_logsumexp_blocked, block_size_v=block_size_v, block_size_n=block_size_n
    )
  )(outputs, w_unemb)

  np.testing.assert_allclose(
    lse_scanned,
    lse_ref,
    rtol=FORWARD_RTOL,
    atol=FORWARD_ATOL,
    err_msg=f"Token-blocked logsumexp (n={block_size_n}) does not match reference",
  )


@pytest.mark.parametrize("block_size_n", [64, 128])
def test_scanned_reference_partial_block_n(block_size_n):
  """Test scanned reference handles partial token blocks correctly."""
  key = jax.random.PRNGKey(654)
  key_out, key_w, key_tgt = jax.random.split(key, 3)

  # Use token count that's not divisible by block_size_n
  batch_size, seq_len, d_model = 3, 70, 128
  vocab_size = 1024
  block_size_v = 128

  outputs = jax.random.normal(
    key_out, (batch_size, seq_len, d_model), dtype=jnp.float32
  )
  w_unemb = jax.random.normal(key_w, (vocab_size, d_model), dtype=jnp.float32)
  targets = jax.random.randint(key_tgt, (batch_size, seq_len), 0, vocab_size)

  loss_ref = jax.jit(ref_cross_entropy_materialized)(outputs, w_unemb, targets)
  loss_scanned = jax.jit(
    partial(scanned_cross_entropy, block_size_v=block_size_v, block_size_n=block_size_n)
  )(outputs, w_unemb, targets)

  np.testing.assert_allclose(
    loss_scanned,
    loss_ref,
    rtol=FORWARD_RTOL,
    atol=FORWARD_ATOL,
    err_msg=f"Partial token block (n={block_size_n}) handling failed",
  )


# =============================================================================
# Timing Tests - compare fused kernel vs scanned reference
# =============================================================================

# Block size configurations for timing tests
# Each tuple is (block_q, block_kv) matching fused kernel's BlockSizes
TIMING_BLOCK_CONFIGS = [
  (128, 128),
  (128, 256),
  (256, 256),
  (256, 512),
  (512, 512),
]
TIMING_BLOCK_IDS = [f"q{q}_kv{kv}" for q, kv in TIMING_BLOCK_CONFIGS]


@requires_tpu
@pytest.mark.parametrize("block_q,block_kv", TIMING_BLOCK_CONFIGS, ids=TIMING_BLOCK_IDS)
def test_timing_forward_pass(block_q: int, block_kv: int):
  """Compare forward pass timing: fused kernel vs scanned logsumexp reference."""
  num_devices = jax.device_count()
  seq_len = 256
  batch_size = 32 * 8
  d_model = 768
  vocab_size = 2 * 4096
  max_valid_id = vocab_size - 256

  config = make_test_config(
    batch_size,
    seq_len,
    d_model,
    vocab_size,
    max_valid_id,
    data_sharding=["fsdp"],
    mesh_shape=[num_devices],
    mesh_axis_names=["fsdp"],
  )
  mesh = jax.make_mesh([num_devices], ["fsdp"], (jax.sharding.AxisType.Explicit,))

  key = jax.random.PRNGKey(42)
  key_out, key_w, key_tgt = jax.random.split(key, 3)

  outputs = jax.random.normal(
    key_out, (batch_size, seq_len, d_model), dtype=jnp.bfloat16
  )
  w_unemb = 0.02 * jax.random.normal(key_w, (vocab_size, d_model), dtype=jnp.bfloat16)
  bias = jnp.zeros(vocab_size, dtype=jnp.bfloat16)
  targets = jax.random.randint(key_tgt, (batch_size, seq_len), 0, max_valid_id)

  # Pre-shard inputs
  batch_sharding = jax.NamedSharding(mesh, jax.P("fsdp", None, None))
  targets_sharding = jax.NamedSharding(mesh, jax.P("fsdp", None))
  weight_sharding = jax.NamedSharding(mesh, jax.P(None, "fsdp"))
  outputs = jax.device_put(outputs, batch_sharding)
  targets = jax.device_put(targets, targets_sharding)
  w_unemb = jax.device_put(w_unemb, weight_sharding)
  print(jax.typeof(outputs))
  print(jax.typeof(w_unemb))

  unemb = LMHead(w=w_unemb, bias=bias)

  num_tokens = batch_size * seq_len
  shard_mapped__kernel = make_lse_kernel_sharded(
    q_seq_len=num_tokens,
    k_seq_len=vocab_size,
    mesh=mesh,
    data_sharding=config.sharding.data,
    q_seq_shards=num_devices,
    block_size=block_q,
    max_valid_id=max_valid_id,
  )

  loss_fn_fused = jax.jit(
    partial(
      fused_softmax_cross_entropy, config, shard_mapped__kernel=shard_mapped__kernel
    )
  )
  # loss_fn_scanned = jax.jit(
  loss_fn_scanned = partial(
      scanned_cross_entropy,
      block_size_v=block_kv,
      block_size_n=block_q,
      data_sharding=["fsdp"],
  )
  # )

  with jax.set_mesh(mesh):
    time_fused, loss_fused = time_fn(
      loss_fn_fused, unemb, outputs, targets, warmup_iters=5, time_iters=20
    )
    time_scanned, loss_scanned = time_fn(
      loss_fn_scanned, outputs, w_unemb, targets, warmup_iters=5, time_iters=20
    )

  print(f"\n{'=' * 60}")
  print(f"Timing Results (block_q={block_q}, block_kv={block_kv})")
  print(f"{'=' * 60}")
  print(f"Fused kernel:      {time_fused * 1000:.3f} ms")
  print(f"Scanned reference: {time_scanned * 1000:.3f} ms")
  print(f"Speedup:           {time_scanned / time_fused:.2f}x")
  print(f"{'=' * 60}")

  np.testing.assert_allclose(
    loss_fused,
    loss_scanned,
    rtol=BFLOAT16_RTOL,
    atol=BFLOAT16_ATOL,
    err_msg="Fused kernel and scanned reference produce different losses",
  )


@requires_tpu
def test_timing_sweep_vocab_sizes():
  """Sweep over vocabulary sizes to see scaling behavior at production scale."""
  num_devices = jax.device_count()
  block_q = 512
  block_kv = 512
  seq_len = 2048
  batch_size = 512
  d_model = 768

  vocab_sizes = [8192, 16384, 32768, 65536, 131072, 262144]

  print(f"\n{'=' * 70}")
  print("Vocabulary Size Sweep (Forward Only)")
  print(f"{'=' * 70}")
  print(
    f"{'Vocab Size':>12} | {'Fused (ms)':>12} | {'Scanned (ms)':>14} | {'Speedup':>8}"
  )
  print(f"{'-' * 70}", flush=True)

  for vocab_size in vocab_sizes:
    print(f"  [Compiling vocab_size={vocab_size}]...", end="", flush=True)
    max_valid_id = vocab_size - 128

    config = make_test_config(
      batch_size,
      seq_len,
      d_model,
      vocab_size,
      max_valid_id,
      data_sharding=["fsdp"],
      mesh_shape=[num_devices],
      mesh_axis_names=["fsdp"],
    )
    mesh = jax.make_mesh([num_devices], ["fsdp"], (jax.sharding.AxisType.Explicit,))

    key = jax.random.PRNGKey(42)
    key_out, key_w, key_tgt = jax.random.split(key, 3)

    outputs = jax.random.normal(
      key_out, (batch_size, seq_len, d_model), dtype=jnp.bfloat16
    )
    w_unemb = 0.02 * jax.random.normal(key_w, (vocab_size, d_model), dtype=jnp.bfloat16)
    bias = jnp.zeros(vocab_size, dtype=jnp.bfloat16)
    targets = jax.random.randint(key_tgt, (batch_size, seq_len), 0, max_valid_id)

    batch_sharding = jax.NamedSharding(mesh, jax.P("fsdp", None, None))
    targets_sharding = jax.NamedSharding(mesh, jax.P("fsdp", None))
    weight_sharding = jax.NamedSharding(mesh, jax.P(None, "fsdp"))
    outputs = jax.device_put(outputs, batch_sharding)
    targets = jax.device_put(targets, targets_sharding)
    w_unemb = jax.device_put(w_unemb, weight_sharding)

    unemb = LMHead(w=w_unemb, bias=bias)

    num_tokens = batch_size * seq_len
    shard_mapped__kernel = make_lse_kernel_sharded(
      q_seq_len=num_tokens,
      k_seq_len=vocab_size,
      mesh=mesh,
      data_sharding=config.sharding.data,
      q_seq_shards=num_devices,
      block_size=block_q,
      max_valid_id=max_valid_id,
    )

    loss_fn_fused = jax.jit(
      partial(
        fused_softmax_cross_entropy, config, shard_mapped__kernel=shard_mapped__kernel
      )
    )
    loss_fn_scanned = jax.jit(
      partial(
        scanned_cross_entropy,
        block_size_v=block_kv,
        block_size_n=block_q,
        data_sharding=["fsdp"],
      )
    )

    with jax.set_mesh(mesh):
      time_fused, _ = time_fn(
        loss_fn_fused, unemb, outputs, targets, warmup_iters=1, time_iters=2
      )
      print(" fused...", end="", flush=True)
      time_scanned, _ = time_fn(
        loss_fn_scanned, outputs, w_unemb, targets, warmup_iters=1, time_iters=2
      )
      print(" scanned done")

    speedup = time_scanned / time_fused
    print(
      f"{vocab_size:>12} | {time_fused * 1000:>12.3f} | "
      f"{time_scanned * 1000:>14.3f} | {speedup:>7.2f}x"
    )

  print(f"{'=' * 70}")


@requires_tpu
def test_timing_sweep_batch_sizes():
  """Sweep over batch sizes to see scaling behavior at production scale."""
  num_devices = jax.device_count()
  block_q = 512
  block_kv = 512
  d_model = 768
  vocab_size = 32768
  max_valid_id = vocab_size - 128

  batch_seq_configs = [
    (128, 1024),
    (256, 1024),
    (256, 2048),
    (512, 1024),
    (512, 2048),
    (1024, 1024),
    (1024, 2048),
  ]

  print(f"\n{'=' * 85}")
  print("Batch/Sequence Size Sweep (Production Scale)")
  print(f"{'=' * 85}")
  print(
    f"{'(B, S)':>14} | {'N':>10} | {'Fused (ms)':>12} | "
    f"{'Scanned (ms)':>14} | {'Speedup':>8}"
  )
  print(f"{'-' * 85}", flush=True)

  for batch_size, seq_len in batch_seq_configs:
    print(f"  [Compiling B={batch_size}, S={seq_len}]...", end="", flush=True)
    n_tokens = batch_size * seq_len

    config = make_test_config(
      batch_size,
      seq_len,
      d_model,
      vocab_size,
      max_valid_id,
      data_sharding=["fsdp"],
      mesh_shape=[num_devices],
      mesh_axis_names=["fsdp"],
    )
    mesh = jax.make_mesh([num_devices], ["fsdp"], (jax.sharding.AxisType.Explicit,))

    key = jax.random.PRNGKey(42)
    key_out, key_w, key_tgt = jax.random.split(key, 3)

    outputs = jax.random.normal(
      key_out, (batch_size, seq_len, d_model), dtype=jnp.bfloat16
    )
    w_unemb = 0.02 * jax.random.normal(key_w, (vocab_size, d_model), dtype=jnp.bfloat16)
    bias = jnp.zeros(vocab_size, dtype=jnp.bfloat16)
    targets = jax.random.randint(key_tgt, (batch_size, seq_len), 0, max_valid_id)

    batch_sharding = jax.NamedSharding(mesh, jax.P("fsdp", None, None))
    targets_sharding = jax.NamedSharding(mesh, jax.P("fsdp", None))
    weight_sharding = jax.NamedSharding(mesh, jax.P(None, "fsdp"))
    outputs = jax.device_put(outputs, batch_sharding)
    targets = jax.device_put(targets, targets_sharding)
    w_unemb = jax.device_put(w_unemb, weight_sharding)

    unemb = LMHead(w=w_unemb, bias=bias)

    shard_mapped__kernel = make_lse_kernel_sharded(
      q_seq_len=n_tokens,
      k_seq_len=vocab_size,
      mesh=mesh,
      data_sharding=config.sharding.data,
      q_seq_shards=num_devices,
      block_size=block_q,
      max_valid_id=max_valid_id,
    )

    loss_fn_fused = jax.jit(
      partial(
        fused_softmax_cross_entropy, config, shard_mapped__kernel=shard_mapped__kernel
      )
    )
    loss_fn_scanned = jax.jit(
      partial(
        scanned_cross_entropy,
        block_size_v=block_kv,
        block_size_n=block_q,
        data_sharding=["fsdp"],
      )
    )

    with jax.set_mesh(mesh):
      time_fused, _ = time_fn(
        loss_fn_fused, unemb, outputs, targets, warmup_iters=1, time_iters=2
      )
      print(" fused done...", end="", flush=True)
      time_scanned, _ = time_fn(
        loss_fn_scanned, outputs, w_unemb, targets, warmup_iters=1, time_iters=2
      )
      print(" scanned done")

    speedup = time_scanned / time_fused
    print(
      f"({batch_size:>5}, {seq_len:>4}) | {n_tokens:>10} | "
      f"{time_fused * 1000:>12.3f} | {time_scanned * 1000:>14.3f} | {speedup:>7.2f}x"
    )

  print(f"{'=' * 85}")


# =============================================================================
# Forward-Backward Timing Tests
# =============================================================================


@requires_tpu
def test_timing_fwd_bwd_sweep_vocab_sizes():
  """Sweep vocab sizes comparing forward-backward passes."""
  num_devices = jax.device_count()
  block_q = 512
  block_kv = 512
  seq_len = 2048
  batch_size = 512
  d_model = 768

  vocab_sizes = [8192, 16384, 32768, 65536, 131072]

  print(f"\n{'=' * 70}")
  print("Forward-Backward Timing: Vocabulary Size Sweep")
  print(f"{'=' * 70}")
  print(
    f"{'Vocab Size':>12} | {'Fused (ms)':>12} | {'Scanned (ms)':>14} | {'Speedup':>8}"
  )
  print(f"{'-' * 70}", flush=True)

  for vocab_size in vocab_sizes:
    print(f"  [Compiling vocab_size={vocab_size}]...", end="", flush=True)
    max_valid_id = vocab_size - 128

    config = make_test_config(
      batch_size,
      seq_len,
      d_model,
      vocab_size,
      max_valid_id,
      data_sharding=["fsdp"],
      mesh_shape=[num_devices],
      mesh_axis_names=["fsdp"],
    )
    mesh = jax.make_mesh([num_devices], ["fsdp"], (jax.sharding.AxisType.Explicit,))

    key = jax.random.PRNGKey(42)
    key_out, key_w, key_tgt = jax.random.split(key, 3)

    outputs = jax.random.normal(
      key_out, (batch_size, seq_len, d_model), dtype=jnp.bfloat16
    )
    w_unemb = 0.02 * jax.random.normal(key_w, (vocab_size, d_model), dtype=jnp.bfloat16)
    bias = jnp.zeros(vocab_size, dtype=jnp.bfloat16)
    targets = jax.random.randint(key_tgt, (batch_size, seq_len), 0, max_valid_id)

    batch_sharding = jax.NamedSharding(mesh, jax.P("fsdp", None, None))
    targets_sharding = jax.NamedSharding(mesh, jax.P("fsdp", None))
    weight_sharding = jax.NamedSharding(mesh, jax.P(None, "fsdp"))
    outputs = jax.device_put(outputs, batch_sharding)
    targets = jax.device_put(targets, targets_sharding)
    w_unemb = jax.device_put(w_unemb, weight_sharding)

    unemb = LMHead(w=w_unemb, bias=bias)

    num_tokens = batch_size * seq_len
    shard_mapped__kernel = make_lse_kernel_sharded(
      q_seq_len=num_tokens,
      k_seq_len=vocab_size,
      mesh=mesh,
      data_sharding=config.sharding.data,
      q_seq_shards=num_devices,
      block_size=block_q,
      max_valid_id=max_valid_id,
    )

    def fused_loss(outputs, unemb, targets):
      return fused_softmax_cross_entropy(
        config, unemb, outputs, targets, shard_mapped__kernel=shard_mapped__kernel
      )

    def scanned_loss(outputs, w_unemb, targets):
      return scanned_cross_entropy(
        outputs,
        w_unemb,
        targets,
        block_size_v=block_kv,
        block_size_n=block_q,
        data_sharding=["fsdp"],
      )

    fused_fwd_bwd = jax.jit(jax.value_and_grad(fused_loss))
    scanned_fwd_bwd = jax.jit(jax.value_and_grad(scanned_loss))

    with jax.set_mesh(mesh):
      time_fused, _ = time_fn(
        fused_fwd_bwd, outputs, unemb, targets, warmup_iters=1, time_iters=2
      )
      print(" fused...", end="", flush=True)
      time_scanned, _ = time_fn(
        scanned_fwd_bwd, outputs, w_unemb, targets, warmup_iters=1, time_iters=2
      )
      print(" scanned done")

    speedup = time_scanned / time_fused
    print(
      f"{vocab_size:>12} | {time_fused * 1000:>12.3f} | "
      f"{time_scanned * 1000:>14.3f} | {speedup:>7.2f}x"
    )

  print(f"{'=' * 70}")


@requires_tpu
def test_timing_fwd_bwd_sweep_batch_sizes():
  """Sweep batch sizes comparing forward-backward passes."""
  num_devices = jax.device_count()
  block_q = 512
  block_kv = 512
  d_model = 768
  vocab_size = 32768
  max_valid_id = vocab_size - 128

  batch_seq_configs = [
    (128, 1024),
    (256, 1024),
    (256, 2048),
    (512, 1024),
    (512, 2048),
    (1024, 1024),
  ]

  print(f"\n{'=' * 90}")
  print("Forward-Backward Timing: Batch/Sequence Size Sweep")
  print(f"{'=' * 90}")
  print(
    f"{'(B, S)':>14} | {'N':>10} | {'Fused (ms)':>12} | "
    f"{'Scanned (ms)':>14} | {'Speedup':>8}"
  )
  print(f"{'-' * 90}", flush=True)

  for batch_size, seq_len in batch_seq_configs:
    print(f"  [Compiling B={batch_size}, S={seq_len}]...", end="", flush=True)
    n_tokens = batch_size * seq_len

    config = make_test_config(
      batch_size,
      seq_len,
      d_model,
      vocab_size,
      max_valid_id,
      data_sharding=["fsdp"],
      mesh_shape=[num_devices],
      mesh_axis_names=["fsdp"],
    )
    mesh = jax.make_mesh([num_devices], ["fsdp"], (jax.sharding.AxisType.Explicit,))

    key = jax.random.PRNGKey(42)
    key_out, key_w, key_tgt = jax.random.split(key, 3)

    outputs = jax.random.normal(
      key_out, (batch_size, seq_len, d_model), dtype=jnp.bfloat16
    )
    w_unemb = 0.02 * jax.random.normal(key_w, (vocab_size, d_model), dtype=jnp.bfloat16)
    bias = jnp.zeros(vocab_size, dtype=jnp.bfloat16)
    targets = jax.random.randint(key_tgt, (batch_size, seq_len), 0, max_valid_id)

    batch_sharding = jax.NamedSharding(mesh, jax.P("fsdp", None, None))
    targets_sharding = jax.NamedSharding(mesh, jax.P("fsdp", None))
    weight_sharding = jax.NamedSharding(mesh, jax.P(None, "fsdp"))
    outputs = jax.device_put(outputs, batch_sharding)
    targets = jax.device_put(targets, targets_sharding)
    w_unemb = jax.device_put(w_unemb, weight_sharding)

    unemb = LMHead(w=w_unemb, bias=bias)

    shard_mapped__kernel = make_lse_kernel_sharded(
      q_seq_len=n_tokens,
      k_seq_len=vocab_size,
      mesh=mesh,
      data_sharding=config.sharding.data,
      q_seq_shards=num_devices,
      block_size=block_q,
      max_valid_id=max_valid_id,
    )

    def fused_loss(outputs, unemb, targets):
      return fused_softmax_cross_entropy(
        config, unemb, outputs, targets, shard_mapped__kernel=shard_mapped__kernel
      )

    def scanned_loss(outputs, w_unemb, targets):
      return scanned_cross_entropy(
        outputs,
        w_unemb,
        targets,
        block_size_v=block_kv,
        block_size_n=block_q,
        data_sharding=["fsdp"],
      )

    fused_fwd_bwd = jax.jit(jax.value_and_grad(fused_loss))
    scanned_fwd_bwd = jax.jit(jax.value_and_grad(scanned_loss))

    with jax.set_mesh(mesh):
      time_fused, _ = time_fn(
        fused_fwd_bwd, outputs, unemb, targets, warmup_iters=1, time_iters=2
      )
      print(" fused done...", end="", flush=True)
      time_scanned, _ = time_fn(
        scanned_fwd_bwd, outputs, w_unemb, targets, warmup_iters=1, time_iters=2
      )
      print(" scanned done")

    speedup = time_scanned / time_fused
    print(
      f"({batch_size:>5}, {seq_len:>4}) | {n_tokens:>10} | "
      f"{time_fused * 1000:>12.3f} | {time_scanned * 1000:>14.3f} | {speedup:>7.2f}x"
    )

  print(f"{'=' * 90}")
