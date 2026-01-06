"""Timing tests for fused cross entropy kernel vs reference implementations."""

import time
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from bmgpt.config import (
  Config,
  DatasetConfig,
  DatasetName,
  ModelConfig,
  ShardingConfig,
  TransformerType,
)
from bmgpt.losses import fused_softmax_cross_entropy
from bmgpt.model import LMHead
from bmgpt.splash_helpers import make_lse_kernel_sharded


def scanned_logsumexp_blocked(
  outputs: jax.Array,
  w_unemb: jax.Array,
  block_size_v: int = 128,
) -> jax.Array:
  """Compute logsumexp(outputs @ w_unemb.T) using a scan over vocab blocks.

  This reference implementation avoids materializing the full (N, V) logits matrix
  by scanning over blocks of the vocabulary dimension, maintaining running max and
  sum-of-exp accumulators (online logsumexp).

  Args:
    outputs: Input activations of shape (N, D)
    w_unemb: Unembedding weights of shape (V, D)
    block_size_v: Block size for scanning over vocabulary dimension

  Returns:
    logsumexp values of shape (N,)
  """
  n, d = outputs.shape
  v, _ = w_unemb.shape

  # Pad w_unemb to be a multiple of block_size_v
  # Use -inf padding so exp(-inf) = 0, contributing nothing to logsumexp
  v_padded = ((v + block_size_v - 1) // block_size_v) * block_size_v
  pad_amount = v_padded - v
  if pad_amount > 0:
    # Pad with zeros in weight space - these will produce logits that we'll mask
    w_unemb_padded = jnp.pad(w_unemb, ((0, pad_amount), (0, 0)), mode="constant")
  else:
    w_unemb_padded = w_unemb

  num_blocks = v_padded // block_size_v

  def scan_body(carry, block_idx):
    """Process one block of vocabulary."""
    m_prev, l_prev = carry

    # Compute block boundaries (in original vocab space for masking)
    start_idx = block_idx * block_size_v

    # Get weight block from padded array
    w_block = jax.lax.dynamic_slice(w_unemb_padded, (start_idx, 0), (block_size_v, d))
    # logits_block: (N, block_size_v)
    logits_block = jnp.dot(outputs, w_block.T)

    # Mask out padding positions (indices >= original vocab size)
    vocab_indices = start_idx + jnp.arange(block_size_v)
    valid_mask = vocab_indices < v
    logits_block = jnp.where(valid_mask[None, :], logits_block, -jnp.inf)

    # Online logsumexp update
    m_curr = logits_block.max(axis=-1)
    m_next = jnp.maximum(m_prev, m_curr)

    # Compute sum of exp for this block, shifted by new max
    s_curr = jnp.exp(logits_block - m_next[:, None]).sum(axis=-1)

    # Update running sum with correction for max shift
    alpha = jnp.exp(m_prev - m_next)
    l_next = s_curr + alpha * l_prev

    return (m_next, l_next), None

  # Initialize accumulators
  m_init = jnp.full((n,), -jnp.inf, dtype=jnp.float32)
  l_init = jnp.zeros((n,), dtype=jnp.float32)

  # Scan over blocks
  (m_final, l_final), _ = jax.lax.scan(
    scan_body,
    (m_init, l_init),
    jnp.arange(num_blocks),
  )

  # Final logsumexp = log(sum) + max
  lse = jnp.log(l_final) + m_final
  return lse


def scanned_cross_entropy(
  outputs: jax.Array,
  w_unemb: jax.Array,
  targets: jax.Array,
  block_size_v: int = 128,
  data_sharding: list[str | None] | None = None,
) -> jax.Array:
  """Cross entropy using scanned logsumexp reference.

  Args:
    outputs: (B, S, D) or (N, D) input activations
    w_unemb: (V, D) unembedding weights (may be sharded, will be replicated internally)
    targets: (B, S) or (N,) target token indices
    block_size_v: Block size for vocab dimension scan
    data_sharding: Optional sharding spec for data dimension (e.g., ["fsdp"])

  Returns:
    Scalar mean cross entropy loss
  """
  # Replicate weights (matches fused_softmax_cross_entropy behavior)
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

  # Compute LSE via blocked scan
  lse = scanned_logsumexp_blocked(outputs_flat, w_unemb, block_size_v)

  # Compute label logits - use explicit out_sharding for gather when sharded
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

  This is the standard implementation used in test_losses.py for correctness
  verification. It explicitly computes the full (N, V) logits matrix.

  Args:
    outputs: (B, S, D) or (N, D) input activations
    w_unemb: (V, D) unembedding weights (may be sharded, will be replicated internally)
    targets: (B, S) or (N,) target token indices
    data_sharding: Optional sharding spec for data dimension (e.g., ["fsdp"])

  Returns:
    Scalar mean cross entropy loss
  """
  # Replicate weights (matches fused_softmax_cross_entropy behavior)
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

  # Materialize full logits matrix: (N, V)
  logits = jnp.dot(outputs_flat, w_unemb.T)

  # Compute cross entropy
  label_logits = jnp.take_along_axis(logits, targets_flat[:, None], axis=-1).squeeze(-1)
  lse = jax.nn.logsumexp(logits, axis=-1)
  return (lse - label_logits).mean()


# =============================================================================
# Correctness Tests - verify scanned reference matches materialized reference
# =============================================================================

CORRECTNESS_CONFIGS = [
  (1, 128, 64, 256),
  (2, 128, 128, 512),
  (4, 128, 128, 1024),
  (4, 256, 256, 2048),
  (2, 256, 128, 4096),
]
CORRECTNESS_CONFIG_IDS = [f"B{b}_S{s}_D{d}_V{v}" for b, s, d, v in CORRECTNESS_CONFIGS]


@pytest.mark.parametrize(
  "batch_size,seq_len,d_model,vocab_size",
  CORRECTNESS_CONFIGS,
  ids=CORRECTNESS_CONFIG_IDS,
)
@pytest.mark.parametrize(
  "block_size_v", [128, 256, 512], ids=["blk128", "blk256", "blk512"]
)
def test_scanned_reference_correctness(
  batch_size, seq_len, d_model, vocab_size, block_size_v
):
  """Verify scanned logsumexp reference matches materialized logits reference.

  This test runs on any platform (CPU/GPU/TPU) and verifies that the blocked
  scan implementation produces the same results as the standard implementation
  that materializes the full logits matrix.
  """
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

  # Match tolerance from test_losses.py
  np.testing.assert_allclose(
    loss_scanned,
    loss_ref,
    rtol=1e-4,
    atol=1e-4,
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

  # Reference: materialize logits and use jax.nn.logsumexp
  @jax.jit
  def ref_lse(outputs, w_unemb):
    logits = jnp.dot(outputs, w_unemb.T)
    return jax.nn.logsumexp(logits, axis=-1)

  lse_ref = ref_lse(outputs, w_unemb)

  # Scanned implementation
  lse_scanned = jax.jit(partial(scanned_logsumexp_blocked, block_size_v=block_size_v))(
    outputs, w_unemb
  )

  # Match tolerance from test_losses.py
  np.testing.assert_allclose(
    lse_scanned,
    lse_ref,
    rtol=1e-4,
    atol=1e-4,
    err_msg=f"Scanned logsumexp (block_size={block_size_v}) does not match reference",
  )


@pytest.mark.parametrize("block_size_v", [128, 256, 512])
def test_scanned_reference_partial_blocks(block_size_v):
  """Test scanned reference handles partial final blocks correctly.

  When vocab_size is not divisible by block_size_v, the final block
  needs special handling to avoid including garbage values.
  """
  key = jax.random.PRNGKey(456)
  key_out, key_w, key_tgt = jax.random.split(key, 3)

  batch_size, seq_len, d_model = 2, 64, 128
  # Choose vocab sizes that don't divide evenly by block_size_v
  vocab_size = block_size_v * 3 + block_size_v // 2  # e.g., 128*3 + 64 = 448

  outputs = jax.random.normal(
    key_out, (batch_size, seq_len, d_model), dtype=jnp.float32
  )
  w_unemb = jax.random.normal(key_w, (vocab_size, d_model), dtype=jnp.float32)
  targets = jax.random.randint(key_tgt, (batch_size, seq_len), 0, vocab_size)

  loss_ref = jax.jit(ref_cross_entropy_materialized)(outputs, w_unemb, targets)
  loss_scanned = jax.jit(partial(scanned_cross_entropy, block_size_v=block_size_v))(
    outputs, w_unemb, targets
  )

  # Match tolerance from test_losses.py
  np.testing.assert_allclose(
    loss_scanned,
    loss_ref,
    rtol=1e-4,
    atol=1e-4,
    err_msg=f"Partial block handling failed for vocab_size={vocab_size}, block_size={block_size_v}",
  )


# =============================================================================
# Timing Tests - compare fused kernel vs scanned reference
# =============================================================================


def make_test_config(
  batch_size: int,
  seq_len: int,
  d_model: int,
  vocab_size: int,
  max_valid_id: int,
  data_sharding: list[str | None],
  mesh_shape: list[int],
  mesh_axis_names: list[str],
) -> Config:
  """Create a Config for testing."""
  wunemb_sharding = [None, mesh_axis_names[0]] if mesh_axis_names else [None, None]
  return Config(
    seed=42,
    used_fused_xent_loss=True,
    train_dataset=DatasetConfig(
      name=DatasetName.SHAKESPEARE,
      path="",
      seq_len=seq_len,
      max_valid_token_id=max_valid_id,
      global_batch_size=batch_size,
    ),
    model=ModelConfig(
      transformer_type=TransformerType.DISCRETE,
      d_model=d_model,
      num_heads=4,
      d_head=d_model // 4,
      num_layers=1,
      num_vocab=vocab_size,
      num_classes=vocab_size,
    ),
    sharding=ShardingConfig(
      mesh_shape=mesh_shape,
      mesh_axis_names=mesh_axis_names,
      wqkv=[None, None, None, None],
      wo=[None, None, None],
      wup=[None, None],
      wdown=[None, None],
      wemb=[None, None],
      wunemb=wunemb_sharding,
      data=data_sharding,
      mlp_hidden=[None],
      res_stream=[None],
      att_qkv=[None, None, None, None],
    ),
  )


def time_fn(fn, *args, warmup_iters: int = 3, time_iters: int = 10):
  """Time a function with warmup iterations."""
  # Warmup
  for _ in range(warmup_iters):
    result = fn(*args)
    jax.block_until_ready(result)

  # Timed iterations
  start = time.perf_counter()
  for _ in range(time_iters):
    result = fn(*args)
    jax.block_until_ready(result)
  end = time.perf_counter()

  return (end - start) / time_iters, result


# Block sizes to test for the scanned reference
BLOCK_SIZES_V = [128, 256, 512, 1024, 2048]
BLOCK_SIZE_IDS = [f"block_v_{bs}" for bs in BLOCK_SIZES_V]


@pytest.mark.skipif(
  not jax.devices()[0].platform == "tpu",
  reason="Timing tests use production kernels (no interpret mode)",
)
@pytest.mark.parametrize("block_size_v", BLOCK_SIZES_V, ids=BLOCK_SIZE_IDS)
def test_timing_forward_pass(block_size_v: int):
  """Compare forward pass timing: fused kernel vs scanned logsumexp reference.

  This test compares:
  1. The fused LSE kernel (production implementation)
  2. A reference implementation using jax.lax.scan over vocab blocks

  Both compute logsumexp(inputs @ weights.T) but the kernel uses Pallas TPU
  optimizations while the reference uses standard JAX ops with blocking.
  """
  num_devices = jax.device_count()
  kernel_block_size = 128
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

  unemb = LMHead(w=w_unemb, bias=bias)

  num_tokens = batch_size * seq_len
  shard_mapped__kernel = make_lse_kernel_sharded(
    q_seq_len=num_tokens,
    k_seq_len=vocab_size,
    mesh=mesh,
    data_sharding=config.sharding.data,
    q_seq_shards=num_devices,
    block_size=kernel_block_size,
    max_valid_id=max_valid_id,
  )

  # JIT compile both implementations
  loss_fn_fused = jax.jit(
    partial(
      fused_softmax_cross_entropy, config, shard_mapped__kernel=shard_mapped__kernel
    )
  )

  loss_fn_scanned = jax.jit(
    partial(scanned_cross_entropy, block_size_v=block_size_v, data_sharding=["fsdp"])
  )

  with jax.set_mesh(mesh):
    # Time fused kernel
    time_fused, loss_fused = time_fn(
      loss_fn_fused,
      unemb,
      outputs,
      targets,
      warmup_iters=5,
      time_iters=20,
    )

    # Time scanned reference (reshards weights internally to match fused)
    time_scanned, loss_scanned = time_fn(
      loss_fn_scanned,
      outputs,
      w_unemb,
      targets,
      warmup_iters=5,
      time_iters=20,
    )

  # Print timing results
  print(f"\n{'=' * 60}")
  print(f"Timing Results (block_size_v={block_size_v})")
  print(f"{'=' * 60}")
  print(f"Fused kernel:      {time_fused * 1000:.3f} ms")
  print(f"Scanned reference: {time_scanned * 1000:.3f} ms")
  print(f"Speedup:           {time_scanned / time_fused:.2f}x")
  print(f"Loss fused:        {float(loss_fused):.6f}")
  print(f"Loss scanned:      {float(loss_scanned):.6f}")
  print(f"Loss diff:         {float(abs(loss_fused - loss_scanned)):.2e}")
  print(f"{'=' * 60}")

  # Verify correctness (loose tolerance for bfloat16)
  np.testing.assert_allclose(
    loss_fused,
    loss_scanned,
    rtol=1e-2,
    atol=1e-2,
    err_msg="Fused kernel and scanned reference produce different losses",
  )


@pytest.mark.skipif(
  not jax.devices()[0].platform == "tpu",
  reason="Timing tests use production kernels (no interpret mode)",
)
def test_timing_sweep_vocab_sizes():
  """Sweep over vocabulary sizes to see scaling behavior at production scale."""
  num_devices = jax.device_count()
  kernel_block_size = 512
  seq_len = 2048
  batch_size = 512
  d_model = 768
  block_size_v = 512  # Fixed scan block size

  # Production-scale vocab sizes up to 128K (and 256K to test OOM boundary)
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

    # Pre-shard inputs
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
      block_size=kernel_block_size,
      max_valid_id=max_valid_id,
    )

    loss_fn_fused = jax.jit(
      partial(
        fused_softmax_cross_entropy, config, shard_mapped__kernel=shard_mapped__kernel
      )
    )
    loss_fn_scanned = jax.jit(
      partial(scanned_cross_entropy, block_size_v=block_size_v, data_sharding=["fsdp"])
    )

    with jax.set_mesh(mesh):
      time_fused, _ = time_fn(
        loss_fn_fused,
        unemb,
        outputs,
        targets,
        warmup_iters=1,
        time_iters=2,
      )
      print(" fused...", end="", flush=True)
      time_scanned, _ = time_fn(
        loss_fn_scanned,
        outputs,
        w_unemb,
        targets,
        warmup_iters=1,
        time_iters=2,
      )
      print(" scanned done")

    speedup = time_scanned / time_fused
    print(
      f"{vocab_size:>12} | {time_fused * 1000:>12.3f} | {time_scanned * 1000:>14.3f} | {speedup:>7.2f}x"
    )

  print(f"{'=' * 70}")


@pytest.mark.skipif(
  not jax.devices()[0].platform == "tpu",
  reason="Timing tests use production kernels (no interpret mode)",
)
def test_timing_sweep_batch_sizes():
  """Sweep over batch sizes (N = batch * seq_len) to see scaling behavior at production scale."""
  num_devices = jax.device_count()
  kernel_block_size = 512
  d_model = 768
  # Proportionally smaller vocab (32K vs 128K in vocab sweep) to allow larger batch/seq
  vocab_size = 32768
  max_valid_id = vocab_size - 128
  block_size_v = 512

  # Production-scale (batch, seq_len) combinations
  # Target: 512 batch, 2K seq (from vocab sweep) - vary around this
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
    f"{'(B, S)':>14} | {'N':>10} | {'Fused (ms)':>12} | {'Scanned (ms)':>14} | {'Speedup':>8}"
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

    # Pre-shard inputs
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
      block_size=kernel_block_size,
      max_valid_id=max_valid_id,
    )

    loss_fn_fused = jax.jit(
      partial(
        fused_softmax_cross_entropy, config, shard_mapped__kernel=shard_mapped__kernel
      )
    )
    loss_fn_scanned = jax.jit(
      partial(scanned_cross_entropy, block_size_v=block_size_v, data_sharding=["fsdp"])
    )

    with jax.set_mesh(mesh):
      time_fused, _ = time_fn(
        loss_fn_fused,
        unemb,
        outputs,
        targets,
        warmup_iters=1,
        time_iters=2,
      )
      print(" fused done...", end="", flush=True)
      time_scanned, _ = time_fn(
        loss_fn_scanned,
        outputs,
        w_unemb,
        targets,
        warmup_iters=1,
        time_iters=2,
      )
      print(" scanned done")

    speedup = time_scanned / time_fused
    print(
      f"({batch_size:>5}, {seq_len:>4}) | {n_tokens:>10} | {time_fused * 1000:>12.3f} | {time_scanned * 1000:>14.3f} | {speedup:>7.2f}x"
    )

  print(f"{'=' * 85}")


# =============================================================================
# Forward-Backward Timing Tests
# =============================================================================


@pytest.mark.skipif(
  not jax.devices()[0].platform == "tpu",
  reason="Timing tests use production kernels (no interpret mode)",
)
def test_timing_fwd_bwd_sweep_vocab_sizes():
  """Sweep vocab sizes comparing forward-backward passes."""
  num_devices = jax.device_count()
  kernel_block_size = 512
  seq_len = 2048
  batch_size = 512
  d_model = 768
  block_size_v = 512

  # Test at production scale
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

    # Pre-shard inputs
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
      block_size=kernel_block_size,
      max_valid_id=max_valid_id,
    )

    # Create grad functions (gradient w.r.t. outputs)
    def fused_loss(outputs, unemb, targets):
      return fused_softmax_cross_entropy(
        config, unemb, outputs, targets, shard_mapped__kernel=shard_mapped__kernel
      )

    def scanned_loss(outputs, w_unemb, targets):
      return scanned_cross_entropy(
        outputs, w_unemb, targets, block_size_v=block_size_v, data_sharding=["fsdp"]
      )

    # JIT the value_and_grad functions
    fused_fwd_bwd = jax.jit(jax.value_and_grad(fused_loss))
    scanned_fwd_bwd = jax.jit(jax.value_and_grad(scanned_loss))

    with jax.set_mesh(mesh):
      time_fused, _ = time_fn(
        fused_fwd_bwd,
        outputs,
        unemb,
        targets,
        warmup_iters=1,
        time_iters=2,
      )
      print(" fused...", end="", flush=True)
      time_scanned, _ = time_fn(
        scanned_fwd_bwd,
        outputs,
        w_unemb,
        targets,
        warmup_iters=1,
        time_iters=2,
      )
      print(" scanned done")

    speedup = time_scanned / time_fused
    print(
      f"{vocab_size:>12} | {time_fused * 1000:>12.3f} | {time_scanned * 1000:>14.3f} | {speedup:>7.2f}x"
    )

  print(f"{'=' * 70}")


@pytest.mark.skipif(
  not jax.devices()[0].platform == "tpu",
  reason="Timing tests use production kernels (no interpret mode)",
)
def test_timing_fwd_bwd_sweep_batch_sizes():
  """Sweep batch sizes comparing forward-backward passes."""
  num_devices = jax.device_count()
  kernel_block_size = 512
  d_model = 768
  vocab_size = 32768
  max_valid_id = vocab_size - 128
  block_size_v = 512

  # Production-scale batch/seq combinations
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
    f"{'(B, S)':>14} | {'N':>10} | {'Fused (ms)':>12} | {'Scanned (ms)':>14} | {'Speedup':>8}"
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

    # Pre-shard inputs
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
      block_size=kernel_block_size,
      max_valid_id=max_valid_id,
    )

    def fused_loss(outputs, unemb, targets):
      return fused_softmax_cross_entropy(
        config, unemb, outputs, targets, shard_mapped__kernel=shard_mapped__kernel
      )

    def scanned_loss(outputs, w_unemb, targets):
      return scanned_cross_entropy(
        outputs, w_unemb, targets, block_size_v=block_size_v, data_sharding=["fsdp"]
      )

    fused_fwd_bwd = jax.jit(jax.value_and_grad(fused_loss))
    scanned_fwd_bwd = jax.jit(jax.value_and_grad(scanned_loss))

    with jax.set_mesh(mesh):
      time_fused, _ = time_fn(
        fused_fwd_bwd,
        outputs,
        unemb,
        targets,
        warmup_iters=1,
        time_iters=2,
      )
      print(" fused done...", end="", flush=True)
      time_scanned, _ = time_fn(
        scanned_fwd_bwd,
        outputs,
        w_unemb,
        targets,
        warmup_iters=1,
        time_iters=2,
      )
      print(" scanned done")

    speedup = time_scanned / time_fused
    print(
      f"({batch_size:>5}, {seq_len:>4}) | {n_tokens:>10} | {time_fused * 1000:>12.3f} | {time_scanned * 1000:>14.3f} | {speedup:>7.2f}x"
    )

  print(f"{'=' * 90}")
