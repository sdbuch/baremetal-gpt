"""Tests for the fused LSE kernel (splash-forward-based cross-entropy)."""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import multihost_utils
from jax.experimental.pallas.ops.tpu.splash_attention import (
  BlockSizes as SplashBlockSizes,
)
from jax.experimental.pallas.ops.tpu.splash_attention.splash_attention_mask import (
  MultiHeadMask,
)

from bmgpt.kernels.lse_kernel import (
  BlockSizes,
  make_lse_fused_kernel,
  make_lse_kernel,
)
from bmgpt.splash_helpers import VocabMask

from .conftest import (
  BACKWARD_ATOL,
  BACKWARD_RTOL,
  DEFAULT_BLOCK_SIZE,
  FORWARD_ATOL,
  FORWARD_RTOL,
  is_tpu,
  ref_lse_forward,
  requires_tpu,
)

requires_multi_device = pytest.mark.skipif(
  len(jax.devices()) < 2,
  reason="Requires multiple devices (set JAX_NUM_CPU_DEVICES=N or use TPU)",
)

# =============================================================================
# Helper Functions
# =============================================================================


def make_kernels(num_tokens, vocab_size, max_valid_id, interpret=True):
  """Create both original and fused LSE kernels with matching configuration."""
  mask = MultiHeadMask([VocabMask((num_tokens, vocab_size), max_valid_id)])

  block_sizes = BlockSizes(
    block_q=DEFAULT_BLOCK_SIZE,
    block_kv=DEFAULT_BLOCK_SIZE,
    block_kv_compute=DEFAULT_BLOCK_SIZE,
  )
  splash_block_sizes = SplashBlockSizes(
    block_q=DEFAULT_BLOCK_SIZE,
    block_kv=DEFAULT_BLOCK_SIZE,
    block_kv_compute=DEFAULT_BLOCK_SIZE,
  )

  orig = make_lse_kernel(mask, block_sizes=block_sizes, interpret=interpret)
  fused = make_lse_fused_kernel(
    mask,
    splash_block_sizes=splash_block_sizes,
    dk_block_sizes=block_sizes,
    interpret=interpret,
  )
  return orig, fused


def make_inputs(seed, num_tokens, vocab_size, scale=1.0):
  """Generate random q, k inputs."""
  key = jax.random.PRNGKey(seed)
  k1, k2 = jax.random.split(key)
  num_heads = 1
  head_dim = 128
  q = scale * jax.random.normal(
    k1, (num_heads, num_tokens, head_dim), dtype=jnp.float32
  )
  k = scale * jax.random.normal(
    k2, (num_heads, vocab_size, head_dim), dtype=jnp.float32
  )
  return q, k


# =============================================================================
# Forward Tests
# =============================================================================

LSE_FUSED_CONFIGS = [
  pytest.param(42, 256, 512, 500, 1.0, id="basic"),
  pytest.param(123, 256, 512, 511, 1.0, id="no_padding"),
  pytest.param(999, 256, 512, 500, 10.0, id="numerical_stability"),
]


@pytest.mark.parametrize(
  "seed,num_tokens,vocab_size,max_valid_id,scale", LSE_FUSED_CONFIGS
)
def test_fused_forward_vs_original(seed, num_tokens, vocab_size, max_valid_id, scale):
  """Fused LSE kernel forward should match original LSE kernel."""
  q, k = make_inputs(seed, num_tokens, vocab_size, scale)
  orig_kernel, fused_kernel = make_kernels(num_tokens, vocab_size, max_valid_id)

  lse_orig = orig_kernel(q, k)
  lse_fused = fused_kernel(q, k)

  if scale > 1.0:
    assert jnp.isfinite(lse_orig).all(), "Original produced non-finite values"
    assert jnp.isfinite(lse_fused).all(), "Fused produced non-finite values"

  np.testing.assert_allclose(lse_fused, lse_orig, rtol=FORWARD_RTOL, atol=FORWARD_ATOL)


@pytest.mark.parametrize(
  "seed,num_tokens,vocab_size,max_valid_id,scale", LSE_FUSED_CONFIGS
)
def test_fused_forward_vs_reference(seed, num_tokens, vocab_size, max_valid_id, scale):
  """Fused LSE kernel forward should match reference implementation."""
  q, k = make_inputs(seed, num_tokens, vocab_size, scale)
  _, fused_kernel = make_kernels(num_tokens, vocab_size, max_valid_id)

  lse_ref = ref_lse_forward(q, k, max_valid_id, vocab_size)
  lse_fused = fused_kernel(q, k)

  np.testing.assert_allclose(lse_fused, lse_ref, rtol=FORWARD_RTOL, atol=FORWARD_ATOL)


# =============================================================================
# Backward Tests
# =============================================================================

LSE_FUSED_BACKWARD_CONFIGS = [
  pytest.param(42, 256, 512, 500, 1.0, id="basic"),
  pytest.param(123, 256, 512, 511, 1.0, id="no_padding"),
]


@pytest.mark.parametrize(
  "seed,num_tokens,vocab_size,max_valid_id,scale", LSE_FUSED_BACKWARD_CONFIGS
)
def test_fused_backward_vs_original(seed, num_tokens, vocab_size, max_valid_id, scale):
  """Fused LSE kernel gradients should match original LSE kernel."""
  q, k = make_inputs(seed, num_tokens, vocab_size, scale)
  orig_kernel, fused_kernel = make_kernels(num_tokens, vocab_size, max_valid_id)

  def orig_loss(q, k):
    return orig_kernel(q, k).sum()

  def fused_loss(q, k):
    return fused_kernel(q, k).sum()

  dq_orig, dk_orig = jax.grad(orig_loss, argnums=(0, 1))(q, k)
  dq_fused, dk_fused = jax.grad(fused_loss, argnums=(0, 1))(q, k)

  np.testing.assert_allclose(dq_fused, dq_orig, rtol=BACKWARD_RTOL, atol=BACKWARD_ATOL)
  np.testing.assert_allclose(dk_fused, dk_orig, rtol=BACKWARD_RTOL, atol=BACKWARD_ATOL)


@pytest.mark.parametrize(
  "seed,num_tokens,vocab_size,max_valid_id,scale", LSE_FUSED_BACKWARD_CONFIGS
)
def test_fused_backward_vs_reference(seed, num_tokens, vocab_size, max_valid_id, scale):
  """Fused LSE kernel gradients should match reference."""
  q, k = make_inputs(seed, num_tokens, vocab_size, scale)
  _, fused_kernel = make_kernels(num_tokens, vocab_size, max_valid_id)

  def ref_loss(q, k):
    return ref_lse_forward(q, k, max_valid_id, vocab_size).sum()

  def fused_loss(q, k):
    return fused_kernel(q, k).sum()

  dq_ref, dk_ref = jax.grad(ref_loss, argnums=(0, 1))(q, k)
  dq_fused, dk_fused = jax.grad(fused_loss, argnums=(0, 1))(q, k)

  np.testing.assert_allclose(dq_fused, dq_ref, rtol=BACKWARD_RTOL, atol=BACKWARD_ATOL)
  np.testing.assert_allclose(dk_fused, dk_ref, rtol=BACKWARD_RTOL, atol=BACKWARD_ATOL)


# =============================================================================
# End-to-End Cross-Entropy Tests
# =============================================================================


def test_fused_cross_entropy_matches_original():
  """Full cross-entropy loss (lse - label_logits) should match between kernels."""
  batch_size = 4
  seq_len = 32
  d_model = 128
  vocab_size = 512
  max_valid_id = vocab_size - 1
  num_tokens = batch_size * seq_len

  key = jax.random.PRNGKey(42)
  key_out, key_w, key_tgt = jax.random.split(key, 3)

  outputs = jax.random.normal(key_out, (num_tokens, d_model), dtype=jnp.float32)
  w_unemb = jax.random.normal(key_w, (vocab_size, d_model), dtype=jnp.float32)
  targets = jax.random.randint(key_tgt, (num_tokens,), 0, max_valid_id)

  orig_kernel, fused_kernel = make_kernels(num_tokens, vocab_size, max_valid_id)

  def cross_entropy(kernel, outputs, w_unemb, targets):
    q = outputs[None]
    k = w_unemb[None]
    lse = kernel(q, k).squeeze(0)
    target_unembs = w_unemb[targets]
    label_logits = jnp.sum(outputs * target_unembs, axis=-1)
    return (lse - label_logits).mean()

  loss_orig = cross_entropy(orig_kernel, outputs, w_unemb, targets)
  loss_fused = cross_entropy(fused_kernel, outputs, w_unemb, targets)

  np.testing.assert_allclose(
    loss_fused, loss_orig, rtol=FORWARD_RTOL, atol=FORWARD_ATOL
  )

  grad_orig = jax.grad(functools.partial(cross_entropy, orig_kernel), argnums=(0, 1))(
    outputs, w_unemb, targets
  )
  grad_fused = jax.grad(functools.partial(cross_entropy, fused_kernel), argnums=(0, 1))(
    outputs, w_unemb, targets
  )

  np.testing.assert_allclose(
    grad_fused[0], grad_orig[0], rtol=BACKWARD_RTOL, atol=BACKWARD_ATOL
  )
  np.testing.assert_allclose(
    grad_fused[1], grad_orig[1], rtol=BACKWARD_RTOL, atol=BACKWARD_ATOL
  )


# =============================================================================
# TPU Tests
# =============================================================================


@requires_tpu
def test_fused_forward_on_tpu():
  """Test fused LSE kernel forward on TPU without interpret mode."""
  seed, num_tokens, vocab_size, max_valid_id = 42, 512, 1024, 1000
  q, k = make_inputs(seed, num_tokens, vocab_size)
  orig_kernel, fused_kernel = make_kernels(
    num_tokens, vocab_size, max_valid_id, interpret=False
  )

  lse_orig = orig_kernel(q, k)
  lse_fused = fused_kernel(q, k)

  np.testing.assert_allclose(lse_fused, lse_orig, rtol=FORWARD_RTOL, atol=FORWARD_ATOL)


@requires_tpu
def test_fused_backward_on_tpu():
  """Test fused LSE kernel gradients on TPU without interpret mode."""
  seed, num_tokens, vocab_size, max_valid_id = 42, 512, 1024, 1000
  q, k = make_inputs(seed, num_tokens, vocab_size)
  orig_kernel, fused_kernel = make_kernels(
    num_tokens, vocab_size, max_valid_id, interpret=False
  )

  def orig_loss(q, k):
    return orig_kernel(q, k).sum()

  def fused_loss(q, k):
    return fused_kernel(q, k).sum()

  dq_orig, dk_orig = jax.grad(orig_loss, argnums=(0, 1))(q, k)
  dq_fused, dk_fused = jax.grad(fused_loss, argnums=(0, 1))(q, k)

  np.testing.assert_allclose(dq_fused, dq_orig, rtol=BACKWARD_RTOL, atol=BACKWARD_ATOL)
  np.testing.assert_allclose(dk_fused, dk_orig, rtol=BACKWARD_RTOL, atol=BACKWARD_ATOL)


# =============================================================================
# Sharded Tests (q_seq and head sharding)
# =============================================================================


@requires_multi_device
def test_fused_sharded_q_seq():
  """Test fused LSE kernel with q_seq sharding via shard_map."""
  num_heads = 1
  vocab_size = 2048
  head_dim = 128
  max_valid_id = 2000
  interpret = not is_tpu()

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
  splash_block_sizes = SplashBlockSizes(
    block_q=DEFAULT_BLOCK_SIZE,
    block_kv=DEFAULT_BLOCK_SIZE,
    block_kv_compute=DEFAULT_BLOCK_SIZE,
  )
  kernel = make_lse_fused_kernel(
    mask_obj,
    splash_block_sizes=splash_block_sizes,
    dk_block_sizes=block_sizes,
    is_mqa=False,
    head_shards=1,
    q_seq_shards=q_seq_shards,
    interpret=interpret,
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


@requires_multi_device
def test_fused_sharded_heads():
  """Test fused LSE kernel with head sharding via shard_map."""
  vocab_size = 1024
  head_dim = 128
  num_tokens = 256
  max_valid_id = 1000
  interpret = not is_tpu()

  num_heads = len(jax.devices())

  key = jax.random.PRNGKey(456)
  key_q, key_k = jax.random.split(key)
  q = jax.random.normal(key_q, (num_heads, num_tokens, head_dim), dtype=jnp.float32)
  k = jax.random.normal(key_k, (num_heads, vocab_size, head_dim), dtype=jnp.float32)

  def ref_loss(q, k):
    lse = ref_lse_forward(q, k, max_valid_id, vocab_size)
    return lse.sum()

  loss_ref = ref_loss(q, k)
  dq_ref, dk_ref = jax.grad(ref_loss, argnums=(0, 1))(q, k)

  mask_obj = MultiHeadMask(
    [VocabMask((num_tokens, vocab_size), max_valid_id) for _ in range(num_heads)]
  )
  block_sizes = BlockSizes(
    block_q=DEFAULT_BLOCK_SIZE,
    block_kv=DEFAULT_BLOCK_SIZE,
    block_kv_compute=DEFAULT_BLOCK_SIZE,
  )
  splash_block_sizes = SplashBlockSizes(
    block_q=DEFAULT_BLOCK_SIZE,
    block_kv=DEFAULT_BLOCK_SIZE,
    block_kv_compute=DEFAULT_BLOCK_SIZE,
  )
  kernel = make_lse_fused_kernel(
    mask_obj,
    splash_block_sizes=splash_block_sizes,
    dk_block_sizes=block_sizes,
    is_mqa=False,
    head_shards=num_heads,
    q_seq_shards=1,
    interpret=interpret,
  )

  mesh = jax.make_mesh((num_heads,), ("heads",), (jax.sharding.AxisType.Explicit,))
  q_spec = jax.P("heads", None, None)
  k_spec = jax.P("heads", None, None)
  lse_spec = jax.P("heads", None)

  kernel_sharding = jax.sharding.NamedSharding(mesh, jax.P("heads", None))
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
