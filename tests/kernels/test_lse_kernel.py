"""Tests for the LSE forward kernel (fused logsumexp computation)."""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import multihost_utils
from jax.experimental.pallas.ops.tpu.splash_attention.splash_attention_mask import (
  MultiHeadMask,
)
from jax.experimental.pallas.ops.tpu.splash_attention.splash_attention_mask_info import (
  _process_mask,
)

from bmgpt.kernels.lse_kernel import (
  BlockSizes,
  QKVLayout,
  _lse_backward_dk,
  _lse_backward_dq,
  _lse_custom,
  _lse_forward,
  make_lse_kernel,
)
from bmgpt.kernels.lse_mask import VocabMask

from .conftest import (
  BACKWARD_ATOL,
  BACKWARD_RTOL,
  DEFAULT_BLOCK_SIZE,
  DEFAULT_MASK_VALUE,
  FORWARD_ATOL,
  FORWARD_RTOL,
  ref_backward_dk,
  ref_backward_dq,
  ref_lse_forward,
  requires_tpu,
)

# =============================================================================
# Test Configuration
# =============================================================================

# LSE forward test configurations: (seed, num_tokens, vocab_size, max_valid_id, scale)
LSE_FORWARD_CONFIGS = [
  pytest.param(42, 512, 1024, 1000, 1.0, id="basic"),
  pytest.param(123, 256, 512, 511, 1.0, id="no_padding"),
  pytest.param(999, 256, 512, 500, 10.0, id="numerical_stability"),
  pytest.param(456, 256, 2048, 1999, 1.0, id="larger_vocab"),
]

# LSE backward test configurations: (seed, num_tokens, vocab_size, max_valid_id, scale)
LSE_BACKWARD_CONFIGS = [
  pytest.param(42, 512, 1024, 1000, 1.0, id="basic"),
  pytest.param(123, 256, 512, 511, 1.0, id="no_padding"),
  pytest.param(999, 256, 512, 500, 10.0, id="numerical_stability"),
]


# =============================================================================
# Helper Functions
# =============================================================================


def make_kernel_inputs(seed: int, num_tokens: int, vocab_size: int, scale: float = 1.0):
  """Generate random inputs for kernel tests."""
  key = jax.random.PRNGKey(seed)
  key_q, key_k = jax.random.split(key)
  num_heads = 1
  head_dim = 128
  q = scale * jax.random.normal(
    key_q, (num_heads, num_tokens, head_dim), dtype=jnp.float32
  )
  k = scale * jax.random.normal(
    key_k, (num_heads, vocab_size, head_dim), dtype=jnp.float32
  )
  return q, k


def make_kernel_inputs_with_upstream(
  seed: int, num_tokens: int, vocab_size: int, scale: float = 1.0
):
  """Generate random inputs including upstream gradient for backward tests."""
  key = jax.random.PRNGKey(seed)
  key_q, key_k, key_do = jax.random.split(key, 3)
  num_heads = 1
  head_dim = 128
  q = scale * jax.random.normal(
    key_q, (num_heads, num_tokens, head_dim), dtype=jnp.float32
  )
  k = scale * jax.random.normal(
    key_k, (num_heads, vocab_size, head_dim), dtype=jnp.float32
  )
  do = jax.random.normal(key_do, (num_heads, num_tokens), dtype=jnp.float32)
  return q, k, do


def setup_kernel_masks(num_tokens: int, vocab_size: int, max_valid_id: int):
  """Create mask objects and block sizes for kernel tests."""
  mask_obj = MultiHeadMask([VocabMask((num_tokens, vocab_size), max_valid_id)])
  block_sizes = BlockSizes(
    block_q=DEFAULT_BLOCK_SIZE,
    block_kv=DEFAULT_BLOCK_SIZE,
    block_kv_compute=DEFAULT_BLOCK_SIZE,
  )
  return mask_obj, block_sizes


# =============================================================================
# LSE Forward Tests
# =============================================================================


@pytest.mark.parametrize(
  "seed,num_tokens,vocab_size,max_valid_id,scale", LSE_FORWARD_CONFIGS
)
def test_lse_forward(seed, num_tokens, vocab_size, max_valid_id, scale):
  """Test LSE forward kernel computes correct LSE compared to reference."""
  q, k = make_kernel_inputs(seed, num_tokens, vocab_size, scale)
  mask_obj, block_sizes = setup_kernel_masks(num_tokens, vocab_size, max_valid_id)

  # Reference computation
  lse_ref = ref_lse_forward(q, k, max_valid_id, vocab_size)

  # Kernel computation
  mask_info, mask_fn = _process_mask(
    mask_obj, (DEFAULT_BLOCK_SIZE, DEFAULT_BLOCK_SIZE), is_dkv=False
  )
  lse_kernel = _lse_forward(
    fwd_mask_info=mask_info,
    q=q,
    k=k,
    segment_ids=None,
    sinks=None,
    mask_value=DEFAULT_MASK_VALUE,
    is_mqa=False,
    block_sizes=block_sizes,
    mask_function=mask_fn,
    attn_logits_soft_cap=None,
    interpret=True,
  )

  # Numerical stability check for scaled inputs
  if scale > 1.0:
    assert jnp.isfinite(lse_kernel).all(), "Kernel produced non-finite values"
    assert jnp.isfinite(lse_ref).all(), "Reference produced non-finite values"

  np.testing.assert_allclose(lse_kernel, lse_ref, rtol=FORWARD_RTOL, atol=FORWARD_ATOL)


@requires_tpu
def test_lse_forward_on_tpu():
  """Test LSE forward kernel on TPU without interpret mode."""
  seed, num_tokens, vocab_size, max_valid_id = 42, 512, 1024, 1000
  q, k = make_kernel_inputs(seed, num_tokens, vocab_size)
  mask_obj, block_sizes = setup_kernel_masks(num_tokens, vocab_size, max_valid_id)

  lse_ref = ref_lse_forward(q, k, max_valid_id, vocab_size)

  mask_info, mask_fn = _process_mask(
    mask_obj, (DEFAULT_BLOCK_SIZE, DEFAULT_BLOCK_SIZE), is_dkv=False
  )
  lse_kernel = _lse_forward(
    fwd_mask_info=mask_info,
    q=q,
    k=k,
    segment_ids=None,
    sinks=None,
    mask_value=DEFAULT_MASK_VALUE,
    is_mqa=False,
    block_sizes=block_sizes,
    mask_function=mask_fn,
    attn_logits_soft_cap=None,
    interpret=False,
  )

  np.testing.assert_allclose(lse_kernel, lse_ref, rtol=FORWARD_RTOL, atol=FORWARD_ATOL)


# =============================================================================
# LSE Backward dQ Tests (S @ K computation)
# =============================================================================


@pytest.mark.parametrize(
  "seed,num_tokens,vocab_size,max_valid_id,scale", LSE_BACKWARD_CONFIGS
)
def test_lse_backward_dq(seed, num_tokens, vocab_size, max_valid_id, scale):
  """Test LSE backward dQ kernel computes correct (do * S) @ K."""
  q, k, do = make_kernel_inputs_with_upstream(seed, num_tokens, vocab_size, scale)
  mask_obj, block_sizes = setup_kernel_masks(num_tokens, vocab_size, max_valid_id)

  # Reference computation
  dq_ref = ref_backward_dq(q, k, do, max_valid_id, vocab_size)

  # Kernel computation
  mask_info, mask_fn = _process_mask(
    mask_obj, (DEFAULT_BLOCK_SIZE, DEFAULT_BLOCK_SIZE), is_dkv=False
  )

  lse_kernel = _lse_forward(
    fwd_mask_info=mask_info,
    q=q,
    k=k,
    segment_ids=None,
    sinks=None,
    mask_value=DEFAULT_MASK_VALUE,
    is_mqa=False,
    block_sizes=block_sizes,
    mask_function=mask_fn,
    attn_logits_soft_cap=None,
    interpret=True,
  )

  dq_kernel = _lse_backward_dq(
    q=q,
    k=k,
    d_lse=do,
    segment_ids=None,
    sinks=None,
    logsumexp=lse_kernel,
    bq=DEFAULT_BLOCK_SIZE,
    bkv=DEFAULT_BLOCK_SIZE,
    is_mqa=False,
    mask_info=mask_info,
    mask_value=DEFAULT_MASK_VALUE,
    attn_logits_soft_cap=None,
    q_layout=QKVLayout.HEAD_DIM_MINOR,
    k_layout=QKVLayout.HEAD_DIM_MINOR,
    mask_function=mask_fn,
    interpret=True,
  )

  if scale > 1.0:
    assert jnp.isfinite(dq_kernel).all(), "Kernel produced non-finite values"
    assert jnp.isfinite(dq_ref).all(), "Reference produced non-finite values"

  np.testing.assert_allclose(dq_kernel, dq_ref, rtol=BACKWARD_RTOL, atol=BACKWARD_ATOL)


@requires_tpu
def test_lse_backward_dq_on_tpu():
  """Test LSE backward dQ kernel on TPU without interpret mode."""
  seed, num_tokens, vocab_size, max_valid_id = 42, 512, 1024, 1000
  q, k, do = make_kernel_inputs_with_upstream(seed, num_tokens, vocab_size)
  mask_obj, block_sizes = setup_kernel_masks(num_tokens, vocab_size, max_valid_id)

  dq_ref = ref_backward_dq(q, k, do, max_valid_id, vocab_size)

  mask_info, mask_fn = _process_mask(
    mask_obj, (DEFAULT_BLOCK_SIZE, DEFAULT_BLOCK_SIZE), is_dkv=False
  )

  lse_kernel = _lse_forward(
    fwd_mask_info=mask_info,
    q=q,
    k=k,
    segment_ids=None,
    sinks=None,
    mask_value=DEFAULT_MASK_VALUE,
    is_mqa=False,
    block_sizes=block_sizes,
    mask_function=mask_fn,
    attn_logits_soft_cap=None,
    interpret=False,
  )

  dq_kernel = _lse_backward_dq(
    q=q,
    k=k,
    d_lse=do,
    segment_ids=None,
    sinks=None,
    logsumexp=lse_kernel,
    bq=DEFAULT_BLOCK_SIZE,
    bkv=DEFAULT_BLOCK_SIZE,
    is_mqa=False,
    mask_info=mask_info,
    mask_value=DEFAULT_MASK_VALUE,
    attn_logits_soft_cap=None,
    q_layout=QKVLayout.HEAD_DIM_MINOR,
    k_layout=QKVLayout.HEAD_DIM_MINOR,
    mask_function=mask_fn,
    interpret=False,
  )

  np.testing.assert_allclose(dq_kernel, dq_ref, rtol=BACKWARD_RTOL, atol=BACKWARD_ATOL)


# =============================================================================
# LSE Backward dK Tests (S^T @ Q computation)
# =============================================================================


@pytest.mark.parametrize(
  "seed,num_tokens,vocab_size,max_valid_id,scale", LSE_BACKWARD_CONFIGS
)
def test_lse_backward_dk(seed, num_tokens, vocab_size, max_valid_id, scale):
  """Test LSE backward dK kernel computes correct S^T @ d_lse_q."""
  q, k, d_lse = make_kernel_inputs_with_upstream(seed, num_tokens, vocab_size, scale)
  mask_obj, block_sizes = setup_kernel_masks(num_tokens, vocab_size, max_valid_id)
  d_lse_q = d_lse[:, :, None] * q

  # Reference computation
  dk_ref = ref_backward_dk(q, k, d_lse, max_valid_id, vocab_size)

  # Kernel computation - need different mask processing for dK
  mask_info, mask_fn = _process_mask(
    mask_obj, (DEFAULT_BLOCK_SIZE, DEFAULT_BLOCK_SIZE), is_dkv=True
  )
  fwd_mask_info, fwd_mask_fn = _process_mask(
    mask_obj, (DEFAULT_BLOCK_SIZE, DEFAULT_BLOCK_SIZE), is_dkv=False
  )

  lse_kernel = _lse_forward(
    fwd_mask_info=fwd_mask_info,
    q=q,
    k=k,
    segment_ids=None,
    sinks=None,
    mask_value=DEFAULT_MASK_VALUE,
    is_mqa=False,
    block_sizes=block_sizes,
    mask_function=fwd_mask_fn,
    attn_logits_soft_cap=None,
    interpret=True,
  )

  dk_kernel = _lse_backward_dk(
    q=q,
    k=k,
    d_lse_q=d_lse_q,
    segment_ids=None,
    sinks=None,
    logsumexp=lse_kernel,
    bq=DEFAULT_BLOCK_SIZE,
    bkv=DEFAULT_BLOCK_SIZE,
    bkv_compute=DEFAULT_BLOCK_SIZE,
    is_mqa=False,
    mask_info=mask_info,
    mask_value=DEFAULT_MASK_VALUE,
    attn_logits_soft_cap=None,
    q_layout=QKVLayout.HEAD_DIM_MINOR,
    k_layout=QKVLayout.HEAD_DIM_MINOR,
    mask_function=mask_fn,
    interpret=True,
  )

  if scale > 1.0:
    assert jnp.isfinite(dk_kernel).all(), "Kernel produced non-finite values"
    assert jnp.isfinite(dk_ref).all(), "Reference produced non-finite values"

  np.testing.assert_allclose(dk_kernel, dk_ref, rtol=BACKWARD_RTOL, atol=BACKWARD_ATOL)


@requires_tpu
def test_lse_backward_dk_on_tpu():
  """Test LSE backward dK kernel on TPU without interpret mode."""
  seed, num_tokens, vocab_size, max_valid_id = 42, 512, 1024, 1000
  q, k, d_lse = make_kernel_inputs_with_upstream(seed, num_tokens, vocab_size)
  mask_obj, block_sizes = setup_kernel_masks(num_tokens, vocab_size, max_valid_id)
  d_lse_q = d_lse[:, :, None] * q

  dk_ref = ref_backward_dk(q, k, d_lse, max_valid_id, vocab_size)

  mask_info, mask_fn = _process_mask(
    mask_obj, (DEFAULT_BLOCK_SIZE, DEFAULT_BLOCK_SIZE), is_dkv=True
  )
  fwd_mask_info, fwd_mask_fn = _process_mask(
    mask_obj, (DEFAULT_BLOCK_SIZE, DEFAULT_BLOCK_SIZE), is_dkv=False
  )

  lse_kernel = _lse_forward(
    fwd_mask_info=fwd_mask_info,
    q=q,
    k=k,
    segment_ids=None,
    sinks=None,
    mask_value=DEFAULT_MASK_VALUE,
    is_mqa=False,
    block_sizes=block_sizes,
    mask_function=fwd_mask_fn,
    attn_logits_soft_cap=None,
    interpret=False,
  )

  dk_kernel = _lse_backward_dk(
    q=q,
    k=k,
    d_lse_q=d_lse_q,
    segment_ids=None,
    sinks=None,
    logsumexp=lse_kernel,
    bq=DEFAULT_BLOCK_SIZE,
    bkv=DEFAULT_BLOCK_SIZE,
    bkv_compute=DEFAULT_BLOCK_SIZE,
    is_mqa=False,
    mask_info=mask_info,
    mask_value=DEFAULT_MASK_VALUE,
    attn_logits_soft_cap=None,
    q_layout=QKVLayout.HEAD_DIM_MINOR,
    k_layout=QKVLayout.HEAD_DIM_MINOR,
    mask_function=mask_fn,
    interpret=False,
  )

  np.testing.assert_allclose(dk_kernel, dk_ref, rtol=BACKWARD_RTOL, atol=BACKWARD_ATOL)


# =============================================================================
# LSE Custom VJP Tests (full differentiable wrapper)
# =============================================================================


@pytest.mark.parametrize(
  "seed,num_tokens,vocab_size,max_valid_id",
  [
    pytest.param(42, 256, 512, 500, id="forward"),
    pytest.param(123, 256, 512, 500, id="gradients"),
  ],
)
def test_lse_custom(seed, num_tokens, vocab_size, max_valid_id):
  """Test _lse_custom forward and gradient computation."""
  q, k = make_kernel_inputs(seed, num_tokens, vocab_size)
  mask_obj, block_sizes = setup_kernel_masks(num_tokens, vocab_size, max_valid_id)

  # Reference
  def ref_loss(q, k):
    lse = ref_lse_forward(q, k, max_valid_id, vocab_size)
    return lse.sum()

  loss_ref = ref_loss(q, k)
  dq_ref, dk_ref = jax.grad(ref_loss, argnums=(0, 1))(q, k)

  # Kernel via custom_vjp wrapper
  fwd_mask_info, mask_fn = _process_mask(
    mask_obj, (DEFAULT_BLOCK_SIZE, DEFAULT_BLOCK_SIZE), is_dkv=False
  )
  dq_mask_info, _ = _process_mask(
    mask_obj, (DEFAULT_BLOCK_SIZE, DEFAULT_BLOCK_SIZE), is_dkv=False
  )
  dk_mask_info, _ = _process_mask(
    mask_obj, (DEFAULT_BLOCK_SIZE, DEFAULT_BLOCK_SIZE), is_dkv=True
  )

  def kernel_loss(q, k):
    lse = _lse_custom(
      fwd_mask_info=fwd_mask_info,
      dq_mask_info=dq_mask_info,
      dk_mask_info=dk_mask_info,
      q=q,
      k=k,
      segment_ids=None,
      sinks=None,
      mask_value=DEFAULT_MASK_VALUE,
      is_mqa=False,
      block_sizes=block_sizes,
      mask_function=mask_fn,
      attn_logits_soft_cap=None,
      interpret=True,
    )
    return lse.sum()

  loss_kernel = kernel_loss(q, k)
  dq_kernel, dk_kernel = jax.grad(kernel_loss, argnums=(0, 1))(q, k)

  np.testing.assert_allclose(
    loss_kernel, loss_ref, rtol=FORWARD_RTOL, atol=FORWARD_ATOL
  )
  np.testing.assert_allclose(dq_kernel, dq_ref, rtol=BACKWARD_RTOL, atol=BACKWARD_ATOL)
  np.testing.assert_allclose(dk_kernel, dk_ref, rtol=BACKWARD_RTOL, atol=BACKWARD_ATOL)


@requires_tpu
def test_lse_custom_on_tpu():
  """Test _lse_custom gradients on TPU without interpret mode."""
  seed, num_tokens, vocab_size, max_valid_id = 456, 512, 1024, 1000
  q, k = make_kernel_inputs(seed, num_tokens, vocab_size)
  mask_obj, block_sizes = setup_kernel_masks(num_tokens, vocab_size, max_valid_id)

  def ref_loss(q, k):
    lse = ref_lse_forward(q, k, max_valid_id, vocab_size)
    return lse.sum()

  loss_ref = ref_loss(q, k)
  dq_ref, dk_ref = jax.grad(ref_loss, argnums=(0, 1))(q, k)

  fwd_mask_info, mask_fn = _process_mask(
    mask_obj, (DEFAULT_BLOCK_SIZE, DEFAULT_BLOCK_SIZE), is_dkv=False
  )
  dq_mask_info, _ = _process_mask(
    mask_obj, (DEFAULT_BLOCK_SIZE, DEFAULT_BLOCK_SIZE), is_dkv=False
  )
  dk_mask_info, _ = _process_mask(
    mask_obj, (DEFAULT_BLOCK_SIZE, DEFAULT_BLOCK_SIZE), is_dkv=True
  )

  def kernel_loss(q, k):
    lse = _lse_custom(
      fwd_mask_info=fwd_mask_info,
      dq_mask_info=dq_mask_info,
      dk_mask_info=dk_mask_info,
      q=q,
      k=k,
      segment_ids=None,
      sinks=None,
      mask_value=DEFAULT_MASK_VALUE,
      is_mqa=False,
      block_sizes=block_sizes,
      mask_function=mask_fn,
      attn_logits_soft_cap=None,
      interpret=False,
    )
    return lse.sum()

  loss_kernel = kernel_loss(q, k)
  dq_kernel, dk_kernel = jax.grad(kernel_loss, argnums=(0, 1))(q, k)

  np.testing.assert_allclose(
    loss_kernel, loss_ref, rtol=FORWARD_RTOL, atol=FORWARD_ATOL
  )
  np.testing.assert_allclose(dq_kernel, dq_ref, rtol=BACKWARD_RTOL, atol=BACKWARD_ATOL)
  np.testing.assert_allclose(dk_kernel, dk_ref, rtol=BACKWARD_RTOL, atol=BACKWARD_ATOL)


# =============================================================================
# LSEKernel and make_lse_kernel Tests
# =============================================================================


@requires_tpu
def test_lse_kernel_sharded_q_seq():
  """Test LSEKernel with q_seq sharding via shard_map."""
  num_heads = 1
  vocab_size = 2048
  head_dim = 128
  max_valid_id = 2000

  q_seq_shards = len(jax.devices())
  num_tokens = q_seq_shards * DEFAULT_BLOCK_SIZE

  key = jax.random.PRNGKey(789)
  key_q, key_k = jax.random.split(key)
  q = jax.random.normal(key_q, (num_heads, num_tokens, head_dim), dtype=jnp.float32)
  k = jax.random.normal(key_k, (num_heads, vocab_size, head_dim), dtype=jnp.float32)

  def ref_loss(q, k):
    lse = ref_lse_forward(q, k, max_valid_id, vocab_size)
    return lse.sum()

  loss_ref = ref_loss(q, k)
  dq_ref, dk_ref = jax.grad(ref_loss, argnums=(0, 1))(q, k)

  mask_obj = MultiHeadMask([VocabMask((num_tokens, vocab_size), max_valid_id)])
  block_sizes = BlockSizes(
    block_q=DEFAULT_BLOCK_SIZE,
    block_kv=DEFAULT_BLOCK_SIZE,
    block_kv_compute=DEFAULT_BLOCK_SIZE,
  )
  kernel = make_lse_kernel(
    mask_obj,
    block_sizes=block_sizes,
    is_mqa=False,
    head_shards=1,
    q_seq_shards=q_seq_shards,
    interpret=False,
  )

  mesh = jax.make_mesh((q_seq_shards,), ("q_seq",), (jax.sharding.AxisType.Explicit,))
  q_spec = jax.P(None, "q_seq", None)
  k_spec = jax.P(None, None, None)
  lse_spec = jax.P(None, "q_seq")

  kernel_sharding = jax.sharding.NamedSharding(mesh, jax.P(None, "q_seq"))
  kernel_spec = kernel.manual_sharding_spec(kernel_sharding)

  @functools.partial(
    jax.shard_map,
    mesh=mesh,
    in_specs=(kernel_spec, q_spec, k_spec),
    out_specs=lse_spec,
    check_vma=False,
  )
  def sharded_kernel(kernel, q, k):
    return kernel(q, k)

  def sharded_loss(q, k):
    lse = sharded_kernel(kernel, q, k)
    return lse.sum()

  with jax.set_mesh(mesh):
    loss_kernel = sharded_loss(q, k)
    dq_kernel, dk_kernel = jax.grad(sharded_loss, argnums=(0, 1))(q, k)

  dq_kernel = multihost_utils.process_allgather(dq_kernel, tiled=True)
  dk_kernel = multihost_utils.process_allgather(dk_kernel, tiled=True)

  np.testing.assert_allclose(
    loss_kernel, loss_ref, rtol=FORWARD_RTOL, atol=FORWARD_ATOL
  )
  np.testing.assert_allclose(dq_kernel, dq_ref, rtol=BACKWARD_RTOL, atol=BACKWARD_ATOL)
  np.testing.assert_allclose(dk_kernel, dk_ref, rtol=BACKWARD_RTOL, atol=BACKWARD_ATOL)


@requires_tpu
@pytest.mark.slow
def test_lse_kernel_memory_pressure():
  """Test LSE kernel under memory pressure with large batch and vocab.

  Uses BS=8M (2^23) tokens and V=128K (2^17) vocab size in q_seq sharded mode.
  This stresses memory bandwidth and tests the kernel at production-like scale.
  """
  num_heads = 1
  head_dim = 128
  block_size = 512

  num_tokens = 2**23  # 8M = 8,388,608
  vocab_size = 2**17  # 128K = 131,072
  max_valid_id = vocab_size - 1024

  q_seq_shards = len(jax.devices())

  key = jax.random.PRNGKey(999)
  key_q, key_k = jax.random.split(key)
  q = jax.random.normal(key_q, (num_heads, num_tokens, head_dim), dtype=jnp.bfloat16)
  k = jax.random.normal(key_k, (num_heads, vocab_size, head_dim), dtype=jnp.bfloat16)

  # Verify naive implementation OOMs
  @jax.jit
  def naive_lse(q, k):
    logits = jnp.einsum("hsd,htd->hst", q, k)
    return jax.nn.logsumexp(logits, axis=-1).sum()

  try:
    _ = naive_lse(q, k).block_until_ready()
    naive_oom = False
  except Exception:
    naive_oom = True

  assert naive_oom, "Naive LSE should OOM at 8M x 128K scale but didn't!"

  # Test kernel handles this scale
  assert num_tokens % (q_seq_shards * block_size) == 0

  mask_obj = MultiHeadMask([VocabMask((num_tokens, vocab_size), max_valid_id)])
  block_sizes = BlockSizes(
    block_q=block_size,
    block_kv=block_size,
    block_kv_compute=block_size,
  )
  kernel = make_lse_kernel(
    mask_obj,
    block_sizes=block_sizes,
    is_mqa=False,
    head_shards=1,
    q_seq_shards=q_seq_shards,
    interpret=False,
  )

  mesh = jax.make_mesh((q_seq_shards,), ("q_seq",), (jax.sharding.AxisType.Explicit,))
  q_spec = jax.P(None, "q_seq", None)
  k_spec = jax.P(None, None, None)
  lse_spec = jax.P(None, "q_seq")

  kernel_sharding = jax.sharding.NamedSharding(mesh, jax.P(None, "q_seq"))
  kernel_spec = kernel.manual_sharding_spec(kernel_sharding)

  @functools.partial(
    jax.shard_map,
    mesh=mesh,
    in_specs=(kernel_spec, q_spec, k_spec),
    out_specs=lse_spec,
    check_vma=False,
  )
  def sharded_kernel(kernel, q, k):
    return kernel(q, k)

  def sharded_loss(q, k):
    lse = sharded_kernel(kernel, q, k)
    return lse.sum()

  with jax.set_mesh(mesh):
    loss = sharded_loss(q, k)
    dq, dk = jax.grad(sharded_loss, argnums=(0, 1))(q, k)

  dq = multihost_utils.process_allgather(dq, tiled=True)
  dk = multihost_utils.process_allgather(dk, tiled=True)

  assert jnp.isfinite(loss), f"Loss is not finite: {loss}"
  assert jnp.all(jnp.isfinite(dq)), "dQ contains non-finite values"
  assert jnp.all(jnp.isfinite(dk)), "dK contains non-finite values"
  assert dq.shape == q.shape
  assert dk.shape == k.shape


@pytest.mark.parametrize(
  "seed,num_tokens,vocab_size,max_valid_id",
  [
    pytest.param(42, 256, 512, 500, id="forward"),
    pytest.param(123, 256, 512, 500, id="gradients"),
  ],
)
def test_make_lse_kernel(seed, num_tokens, vocab_size, max_valid_id):
  """Test make_lse_kernel factory forward and gradient computation."""
  q, k = make_kernel_inputs(seed, num_tokens, vocab_size)

  def ref_loss(q, k):
    lse = ref_lse_forward(q, k, max_valid_id, vocab_size)
    return lse.sum()

  loss_ref = ref_loss(q, k)
  dq_ref, dk_ref = jax.grad(ref_loss, argnums=(0, 1))(q, k)

  mask_obj = MultiHeadMask([VocabMask((num_tokens, vocab_size), max_valid_id)])
  kernel = make_lse_kernel(mask_obj, is_mqa=False, interpret=True)

  def kernel_loss(q, k):
    lse = kernel(q, k)
    return lse.sum()

  loss_kernel = kernel_loss(q, k)
  dq_kernel, dk_kernel = jax.grad(kernel_loss, argnums=(0, 1))(q, k)

  np.testing.assert_allclose(
    loss_kernel, loss_ref, rtol=FORWARD_RTOL, atol=FORWARD_ATOL
  )
  np.testing.assert_allclose(dq_kernel, dq_ref, rtol=BACKWARD_RTOL, atol=BACKWARD_ATOL)
  np.testing.assert_allclose(dk_kernel, dk_ref, rtol=BACKWARD_RTOL, atol=BACKWARD_ATOL)


@requires_tpu
def test_make_lse_kernel_on_tpu():
  """Test make_lse_kernel on TPU without interpret mode."""
  seed, num_tokens, vocab_size, max_valid_id = 456, 512, 1024, 1000
  q, k = make_kernel_inputs(seed, num_tokens, vocab_size)

  def ref_loss(q, k):
    lse = ref_lse_forward(q, k, max_valid_id, vocab_size)
    return lse.sum()

  loss_ref = ref_loss(q, k)
  dq_ref, dk_ref = jax.grad(ref_loss, argnums=(0, 1))(q, k)

  mask_obj = MultiHeadMask([VocabMask((num_tokens, vocab_size), max_valid_id)])
  kernel = make_lse_kernel(mask_obj, is_mqa=False, interpret=False)

  def kernel_loss(q, k):
    lse = kernel(q, k)
    return lse.sum()

  loss_kernel = kernel_loss(q, k)
  dq_kernel, dk_kernel = jax.grad(kernel_loss, argnums=(0, 1))(q, k)

  np.testing.assert_allclose(
    loss_kernel, loss_ref, rtol=FORWARD_RTOL, atol=FORWARD_ATOL
  )
  np.testing.assert_allclose(dq_kernel, dq_ref, rtol=BACKWARD_RTOL, atol=BACKWARD_ATOL)
  np.testing.assert_allclose(dk_kernel, dk_ref, rtol=BACKWARD_RTOL, atol=BACKWARD_ATOL)


# =============================================================================
# Fused Cross-Entropy Loss Tests
# =============================================================================


def test_fused_cross_entropy_matches_naive():
  """Test fused cross-entropy loss matches naive implementation."""
  batch_size = 4
  seq_len = 32
  d_model = 128
  vocab_size = 512
  max_valid_id = vocab_size - 1

  key = jax.random.PRNGKey(42)
  key_out, key_w, key_tgt = jax.random.split(key, 3)

  outputs = jax.random.normal(
    key_out, (batch_size, seq_len, d_model), dtype=jnp.float32
  )
  w_unemb = jax.random.normal(key_w, (vocab_size, d_model), dtype=jnp.float32)
  targets = jax.random.randint(key_tgt, (batch_size, seq_len), 0, vocab_size)

  def naive_cross_entropy(outputs, w_unemb, targets):
    logits = jnp.einsum("bsd,vd->bsv", outputs, w_unemb)
    label_logits = jnp.take_along_axis(logits, targets[..., None], axis=-1).squeeze(-1)
    lse = jax.nn.logsumexp(logits, axis=-1)
    return (lse - label_logits).mean()

  def fused_cross_entropy(outputs, w_unemb, targets):
    b, s = outputs.shape[:2]
    outputs_flat = outputs.reshape(b * s, -1)
    targets_flat = targets.ravel()

    q = outputs_flat[None]
    k = w_unemb[None]

    mask_obj = MultiHeadMask([VocabMask((b * s, vocab_size), max_valid_id)])
    block_sizes = BlockSizes(
      block_q=DEFAULT_BLOCK_SIZE,
      block_kv=DEFAULT_BLOCK_SIZE,
      block_kv_compute=DEFAULT_BLOCK_SIZE,
    )
    kernel = make_lse_kernel(
      mask_obj, block_sizes=block_sizes, is_mqa=False, interpret=True
    )
    lse = kernel(q, k).squeeze(0)

    per_token_unembs = w_unemb[targets_flat]
    label_logits = jnp.sum(outputs_flat * per_token_unembs, axis=-1)

    return (lse - label_logits).mean()

  loss_naive = naive_cross_entropy(outputs, w_unemb, targets)
  loss_fused = fused_cross_entropy(outputs, w_unemb, targets)

  np.testing.assert_allclose(
    loss_fused, loss_naive, rtol=FORWARD_RTOL, atol=FORWARD_ATOL
  )

  grad_naive = jax.grad(naive_cross_entropy, argnums=(0, 1))(outputs, w_unemb, targets)
  grad_fused = jax.grad(fused_cross_entropy, argnums=(0, 1))(outputs, w_unemb, targets)

  np.testing.assert_allclose(
    grad_fused[0], grad_naive[0], rtol=BACKWARD_RTOL, atol=BACKWARD_ATOL
  )
  np.testing.assert_allclose(
    grad_fused[1], grad_naive[1], rtol=BACKWARD_RTOL, atol=BACKWARD_ATOL
  )


@requires_tpu
def test_fused_cross_entropy_on_tpu():
  """Test fused cross-entropy on TPU at larger scale."""
  batch_size = 8
  seq_len = 128
  d_model = 128
  vocab_size = 2048
  max_valid_id = vocab_size - 128

  key = jax.random.PRNGKey(123)
  key_out, key_w, key_tgt = jax.random.split(key, 3)

  outputs = jax.random.normal(
    key_out, (batch_size, seq_len, d_model), dtype=jnp.float32
  )
  w_unemb = jax.random.normal(key_w, (vocab_size, d_model), dtype=jnp.float32)
  targets = jax.random.randint(key_tgt, (batch_size, seq_len), 0, max_valid_id)

  def naive_cross_entropy(outputs, w_unemb, targets):
    logits = jnp.einsum("bsd,vd->bsv", outputs, w_unemb)
    vocab_ids = jnp.arange(vocab_size)
    mask = vocab_ids <= max_valid_id
    logits = jnp.where(mask[None, None, :], logits, DEFAULT_MASK_VALUE)
    label_logits = jnp.take_along_axis(logits, targets[..., None], axis=-1).squeeze(-1)
    lse = jax.nn.logsumexp(logits, axis=-1)
    return (lse - label_logits).mean()

  def fused_cross_entropy(outputs, w_unemb, targets):
    b, s = outputs.shape[:2]
    outputs_flat = outputs.reshape(b * s, -1)
    targets_flat = targets.ravel()

    q = outputs_flat[None]
    k = w_unemb[None]

    mask_obj = MultiHeadMask([VocabMask((b * s, vocab_size), max_valid_id)])
    block_sizes = BlockSizes(
      block_q=DEFAULT_BLOCK_SIZE,
      block_kv=DEFAULT_BLOCK_SIZE,
      block_kv_compute=DEFAULT_BLOCK_SIZE,
    )
    kernel = make_lse_kernel(
      mask_obj, block_sizes=block_sizes, is_mqa=False, interpret=False
    )
    lse = kernel(q, k).squeeze(0)

    per_token_unembs = w_unemb[targets_flat]
    label_logits = jnp.sum(outputs_flat * per_token_unembs, axis=-1)

    return (lse - label_logits).mean()

  loss_naive = naive_cross_entropy(outputs, w_unemb, targets)
  loss_fused = fused_cross_entropy(outputs, w_unemb, targets)

  np.testing.assert_allclose(
    loss_fused, loss_naive, rtol=FORWARD_RTOL, atol=FORWARD_ATOL
  )

  grad_naive = jax.grad(naive_cross_entropy, argnums=(0, 1))(outputs, w_unemb, targets)
  grad_fused = jax.grad(fused_cross_entropy, argnums=(0, 1))(outputs, w_unemb, targets)

  np.testing.assert_allclose(
    grad_fused[0], grad_naive[0], rtol=BACKWARD_RTOL, atol=BACKWARD_ATOL
  )
  np.testing.assert_allclose(
    grad_fused[1], grad_naive[1], rtol=BACKWARD_RTOL, atol=BACKWARD_ATOL
  )
