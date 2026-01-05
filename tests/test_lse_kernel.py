"""Tests for the LSE forward kernel (fused logsumexp computation)."""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import multihost_utils
from jax import shard_map
from jax.sharding import AxisType

from bmgpt.kernels.lse_kernel import (
  BlockSizes,
  LSEKernel,
  QKVLayout,
  _lse_backward_dk,
  _lse_backward_dq,
  _lse_custom,
  _lse_forward,
  make_lse_kernel,
)
from bmgpt.kernels.lse_mask import VocabMask
from jax.experimental.pallas.ops.tpu.splash_attention.splash_attention_mask import (
  MultiHeadMask,
)
from jax.experimental.pallas.ops.tpu.splash_attention.splash_attention_mask_info import (
  _process_mask,
)


def test_lse_forward_lse_basic():
  """Test LSE forward kernel computes correct LSE compared to reference."""
  # Config - use power-of-2 sizes, 128-aligned for TPU
  num_heads = 1
  num_tokens = 512
  vocab_size = 1024
  head_dim = 128
  max_valid_id = 1000  # Test with padding (tokens 1001-1023 masked)
  block_size = 128

  # Generate random inputs
  key = jax.random.PRNGKey(42)
  key_q, key_k = jax.random.split(key)
  q = jax.random.normal(key_q, (num_heads, num_tokens, head_dim), dtype=jnp.float32)
  k = jax.random.normal(key_k, (num_heads, vocab_size, head_dim), dtype=jnp.float32)

  # Reference computation (mask out padding tokens)
  logits = jnp.einsum("hsd,htd->hst", q, k)  # (heads, tokens, vocab)
  # Mask: tokens with id > max_valid_id should be -inf
  vocab_ids = jnp.arange(vocab_size)
  mask = vocab_ids <= max_valid_id  # (vocab_size,)
  logits_masked = jnp.where(mask[None, None, :], logits, -1e10)
  lse_ref = jax.nn.logsumexp(logits_masked, axis=-1)  # (heads, tokens)

  # Kernel computation
  mask_obj = MultiHeadMask([VocabMask((num_tokens, vocab_size), max_valid_id)])
  block_sizes = BlockSizes(
    block_q=block_size,
    block_kv=block_size,
    block_kv_compute=block_size,
  )
  mask_info, mask_fn = _process_mask(mask_obj, (block_size, block_size), is_dkv=False)
  lse_kernel = _lse_forward(
    fwd_mask_info=mask_info,
    q=q,
    k=k,
    segment_ids=None,
    sinks=None,
    mask_value=-1e10,
    is_mqa=False,
    block_sizes=block_sizes,
    mask_function=mask_fn,
    attn_logits_soft_cap=None,
    interpret=True,  # Use interpret mode for CPU testing
  )

  np.testing.assert_allclose(lse_kernel, lse_ref, rtol=1e-4, atol=1e-4)


def test_lse_forward_lse_no_padding():
  """Test LSE forward with all vocab tokens valid (edge case)."""
  num_heads = 1
  num_tokens = 256
  vocab_size = 512
  head_dim = 128
  # All tokens valid - max_valid_id is last token
  max_valid_id = vocab_size - 1
  block_size = 128

  key = jax.random.PRNGKey(123)
  key_q, key_k = jax.random.split(key)
  q = jax.random.normal(key_q, (num_heads, num_tokens, head_dim), dtype=jnp.float32)
  k = jax.random.normal(key_k, (num_heads, vocab_size, head_dim), dtype=jnp.float32)

  # Reference: no masking needed
  logits = jnp.einsum("hsd,htd->hst", q, k)
  lse_ref = jax.nn.logsumexp(logits, axis=-1)

  # Kernel
  mask_obj = MultiHeadMask([VocabMask((num_tokens, vocab_size), max_valid_id)])
  block_sizes = BlockSizes(
    block_q=block_size,
    block_kv=block_size,
    block_kv_compute=block_size,
  )
  mask_info, mask_fn = _process_mask(mask_obj, (block_size, block_size), is_dkv=False)
  lse_kernel = _lse_forward(
    fwd_mask_info=mask_info,
    q=q,
    k=k,
    segment_ids=None,
    sinks=None,
    mask_value=-1e10,
    is_mqa=False,
    block_sizes=block_sizes,
    mask_function=mask_fn,
    attn_logits_soft_cap=None,
    interpret=True,
  )

  np.testing.assert_allclose(lse_kernel, lse_ref, rtol=1e-4, atol=1e-4)


def test_lse_forward_numerical_stability():
  """Test that kernel is numerically stable with large values."""
  num_heads = 1
  num_tokens = 256
  vocab_size = 512
  head_dim = 128
  max_valid_id = 500
  block_size = 128

  key = jax.random.PRNGKey(999)
  key_q, key_k = jax.random.split(key)
  # Large values that would cause naive softmax to overflow
  q = (
    jax.random.normal(key_q, (num_heads, num_tokens, head_dim), dtype=jnp.float32)
    * 10.0
  )
  k = (
    jax.random.normal(key_k, (num_heads, vocab_size, head_dim), dtype=jnp.float32)
    * 10.0
  )

  # Reference
  logits = jnp.einsum("hsd,htd->hst", q, k)
  vocab_ids = jnp.arange(vocab_size)
  mask = vocab_ids <= max_valid_id
  logits_masked = jnp.where(mask[None, None, :], logits, -1e10)
  lse_ref = jax.nn.logsumexp(logits_masked, axis=-1)

  # Kernel
  mask_obj = MultiHeadMask([VocabMask((num_tokens, vocab_size), max_valid_id)])
  block_sizes = BlockSizes(
    block_q=block_size,
    block_kv=block_size,
    block_kv_compute=block_size,
  )
  mask_info, mask_fn = _process_mask(mask_obj, (block_size, block_size), is_dkv=False)
  lse_kernel = _lse_forward(
    fwd_mask_info=mask_info,
    q=q,
    k=k,
    segment_ids=None,
    sinks=None,
    mask_value=-1e10,
    is_mqa=False,
    block_sizes=block_sizes,
    mask_function=mask_fn,
    attn_logits_soft_cap=None,
    interpret=True,
  )

  # Should not produce NaN/Inf
  assert jnp.isfinite(lse_kernel).all(), "Kernel produced non-finite values"
  assert jnp.isfinite(lse_ref).all(), "Reference produced non-finite values"
  np.testing.assert_allclose(lse_kernel, lse_ref, rtol=1e-3, atol=1e-3)


def test_lse_forward_larger_vocab():
  """Test with larger vocabulary size (more realistic scenario)."""
  num_heads = 1
  num_tokens = 256
  vocab_size = 2048
  head_dim = 128
  # Simulate padded vocab (e.g., 2000 real tokens, padded to 2048)
  max_valid_id = 1999
  block_size = 128

  key = jax.random.PRNGKey(456)
  key_q, key_k = jax.random.split(key)
  q = jax.random.normal(key_q, (num_heads, num_tokens, head_dim), dtype=jnp.float32)
  k = jax.random.normal(key_k, (num_heads, vocab_size, head_dim), dtype=jnp.float32)

  # Reference
  logits = jnp.einsum("hsd,htd->hst", q, k)
  vocab_ids = jnp.arange(vocab_size)
  mask = vocab_ids <= max_valid_id
  logits_masked = jnp.where(mask[None, None, :], logits, -1e10)
  lse_ref = jax.nn.logsumexp(logits_masked, axis=-1)

  # Kernel
  mask_obj = MultiHeadMask([VocabMask((num_tokens, vocab_size), max_valid_id)])
  block_sizes = BlockSizes(
    block_q=block_size,
    block_kv=block_size,
    block_kv_compute=block_size,
  )
  mask_info, mask_fn = _process_mask(mask_obj, (block_size, block_size), is_dkv=False)
  lse_kernel = _lse_forward(
    fwd_mask_info=mask_info,
    q=q,
    k=k,
    segment_ids=None,
    sinks=None,
    mask_value=-1e10,
    is_mqa=False,
    block_sizes=block_sizes,
    mask_function=mask_fn,
    attn_logits_soft_cap=None,
    interpret=True,
  )

  np.testing.assert_allclose(lse_kernel, lse_ref, rtol=1e-4, atol=1e-4)


@pytest.mark.skipif(
  not jax.devices()[0].platform == "tpu",
  reason="TPU test - skip on non-TPU platforms",
)
def test_lse_forward_on_tpu():
  """Test LSE forward kernel on TPU without interpret mode."""
  num_heads = 1
  num_tokens = 512
  vocab_size = 1024
  head_dim = 128
  max_valid_id = 1000
  block_size = 128

  key = jax.random.PRNGKey(42)
  key_q, key_k = jax.random.split(key)
  q = jax.random.normal(key_q, (num_heads, num_tokens, head_dim), dtype=jnp.float32)
  k = jax.random.normal(key_k, (num_heads, vocab_size, head_dim), dtype=jnp.float32)

  # Reference
  logits = jnp.einsum("hsd,htd->hst", q, k)
  vocab_ids = jnp.arange(vocab_size)
  mask = vocab_ids <= max_valid_id
  logits_masked = jnp.where(mask[None, None, :], logits, -1e10)
  lse_ref = jax.nn.logsumexp(logits_masked, axis=-1)

  # Kernel on TPU (interpret=False)
  mask_obj = MultiHeadMask([VocabMask((num_tokens, vocab_size), max_valid_id)])
  block_sizes = BlockSizes(
    block_q=block_size,
    block_kv=block_size,
    block_kv_compute=block_size,
  )
  mask_info, mask_fn = _process_mask(mask_obj, (block_size, block_size), is_dkv=False)
  lse_kernel = _lse_forward(
    fwd_mask_info=mask_info,
    q=q,
    k=k,
    segment_ids=None,
    sinks=None,
    mask_value=-1e10,
    is_mqa=False,
    block_sizes=block_sizes,
    mask_function=mask_fn,
    attn_logits_soft_cap=None,
    interpret=False,  # Real TPU execution
  )

  np.testing.assert_allclose(lse_kernel, lse_ref, rtol=1e-4, atol=1e-4)


# =============================================================================
# LSE Backward dQ Tests (S @ K computation)
# =============================================================================


def test_lse_backward_dq_basic():
  """Test LSE backward dQ kernel computes correct (do * S) @ K compared to reference."""
  num_heads = 1
  num_tokens = 512
  vocab_size = 1024
  head_dim = 128
  max_valid_id = 1000
  block_size = 128

  key = jax.random.PRNGKey(42)
  key_q, key_k, key_do = jax.random.split(key, 3)
  q = jax.random.normal(key_q, (num_heads, num_tokens, head_dim), dtype=jnp.float32)
  k = jax.random.normal(key_k, (num_heads, vocab_size, head_dim), dtype=jnp.float32)
  # Random do (g_lse) to test fused scaling
  do = jax.random.normal(key_do, (num_heads, num_tokens), dtype=jnp.float32)

  # Reference computation
  logits = jnp.einsum("hsd,htd->hst", q, k)  # (heads, tokens, vocab)
  vocab_ids = jnp.arange(vocab_size)
  mask = vocab_ids <= max_valid_id
  logits_masked = jnp.where(mask[None, None, :], logits, -1e10)
  # S = softmax(logits_masked)
  s = jax.nn.softmax(logits_masked, axis=-1)  # (heads, tokens, vocab)
  # (do * S) @ K reference: scale each row of S by do
  ds = do[:, :, None] * s  # (heads, tokens, vocab)
  dq_ref = jnp.einsum("hst,htd->hsd", ds, k)  # (heads, tokens, head_dim)

  # Kernel computation
  mask_obj = MultiHeadMask([VocabMask((num_tokens, vocab_size), max_valid_id)])
  block_sizes = BlockSizes(
    block_q=block_size,
    block_kv=block_size,
    block_kv_compute=block_size,
  )
  mask_info, mask_fn = _process_mask(mask_obj, (block_size, block_size), is_dkv=False)

  # First get LSE from forward kernel
  lse_kernel = _lse_forward(
    fwd_mask_info=mask_info,
    q=q,
    k=k,
    segment_ids=None,
    sinks=None,
    mask_value=-1e10,
    is_mqa=False,
    block_sizes=block_sizes,
    mask_function=mask_fn,
    attn_logits_soft_cap=None,
    interpret=True,
  )

  # Then compute (d_lse * S) @ K using backward dQ kernel
  dq_kernel = _lse_backward_dq(
    q=q,
    k=k,
    d_lse=do,
    segment_ids=None,
    sinks=None,
    logsumexp=lse_kernel,
    bq=block_size,
    bkv=block_size,
    is_mqa=False,
    mask_info=mask_info,
    mask_value=-1e10,
    attn_logits_soft_cap=None,
    q_layout=QKVLayout.HEAD_DIM_MINOR,
    k_layout=QKVLayout.HEAD_DIM_MINOR,
    mask_function=mask_fn,
    interpret=True,
  )

  # Backward kernel has more numerical accumulation than forward.
  # Note: 2e-2 tolerance is consistent with splash attention's precision on TPU.
  # Testing shows splash attention forward (S @ K) also requires ~1e-2 tolerance
  # due to block-wise accumulation on TPU MXUs. See test_splash_precision.py.
  np.testing.assert_allclose(dq_kernel, dq_ref, rtol=2e-2, atol=2e-2)


def test_lse_backward_dq_no_padding():
  """Test LSE backward dQ with all vocab tokens valid."""
  num_heads = 1
  num_tokens = 256
  vocab_size = 512
  head_dim = 128
  max_valid_id = vocab_size - 1
  block_size = 128

  key = jax.random.PRNGKey(123)
  key_q, key_k, key_do = jax.random.split(key, 3)
  q = jax.random.normal(key_q, (num_heads, num_tokens, head_dim), dtype=jnp.float32)
  k = jax.random.normal(key_k, (num_heads, vocab_size, head_dim), dtype=jnp.float32)
  do = jax.random.normal(key_do, (num_heads, num_tokens), dtype=jnp.float32)

  # Reference: no masking
  logits = jnp.einsum("hsd,htd->hst", q, k)
  s = jax.nn.softmax(logits, axis=-1)
  ds = do[:, :, None] * s
  dq_ref = jnp.einsum("hst,htd->hsd", ds, k)

  # Kernel
  mask_obj = MultiHeadMask([VocabMask((num_tokens, vocab_size), max_valid_id)])
  block_sizes = BlockSizes(
    block_q=block_size,
    block_kv=block_size,
    block_kv_compute=block_size,
  )
  mask_info, mask_fn = _process_mask(mask_obj, (block_size, block_size), is_dkv=False)

  lse_kernel = _lse_forward(
    fwd_mask_info=mask_info,
    q=q,
    k=k,
    segment_ids=None,
    sinks=None,
    mask_value=-1e10,
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
    bq=block_size,
    bkv=block_size,
    is_mqa=False,
    mask_info=mask_info,
    mask_value=-1e10,
    attn_logits_soft_cap=None,
    q_layout=QKVLayout.HEAD_DIM_MINOR,
    k_layout=QKVLayout.HEAD_DIM_MINOR,
    mask_function=mask_fn,
    interpret=True,
  )

  # 2e-2 tolerance consistent with splash attention precision on TPU MXUs
  np.testing.assert_allclose(dq_kernel, dq_ref, rtol=2e-2, atol=2e-2)


def test_lse_backward_dq_numerical_stability():
  """Test LSE backward dQ is numerically stable with large values."""
  num_heads = 1
  num_tokens = 256
  vocab_size = 512
  head_dim = 128
  max_valid_id = 500
  block_size = 128

  key = jax.random.PRNGKey(999)
  key_q, key_k, key_do = jax.random.split(key, 3)
  q = (
    jax.random.normal(key_q, (num_heads, num_tokens, head_dim), dtype=jnp.float32)
    * 10.0
  )
  k = (
    jax.random.normal(key_k, (num_heads, vocab_size, head_dim), dtype=jnp.float32)
    * 10.0
  )
  do = jax.random.normal(key_do, (num_heads, num_tokens), dtype=jnp.float32)

  # Reference
  logits = jnp.einsum("hsd,htd->hst", q, k)
  vocab_ids = jnp.arange(vocab_size)
  mask = vocab_ids <= max_valid_id
  logits_masked = jnp.where(mask[None, None, :], logits, -1e10)
  s = jax.nn.softmax(logits_masked, axis=-1)
  ds = do[:, :, None] * s
  dq_ref = jnp.einsum("hst,htd->hsd", ds, k)

  # Kernel
  mask_obj = MultiHeadMask([VocabMask((num_tokens, vocab_size), max_valid_id)])
  block_sizes = BlockSizes(
    block_q=block_size,
    block_kv=block_size,
    block_kv_compute=block_size,
  )
  mask_info, mask_fn = _process_mask(mask_obj, (block_size, block_size), is_dkv=False)

  lse_kernel = _lse_forward(
    fwd_mask_info=mask_info,
    q=q,
    k=k,
    segment_ids=None,
    sinks=None,
    mask_value=-1e10,
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
    bq=block_size,
    bkv=block_size,
    is_mqa=False,
    mask_info=mask_info,
    mask_value=-1e10,
    attn_logits_soft_cap=None,
    q_layout=QKVLayout.HEAD_DIM_MINOR,
    k_layout=QKVLayout.HEAD_DIM_MINOR,
    mask_function=mask_fn,
    interpret=True,
  )

  assert jnp.isfinite(dq_kernel).all(), "Kernel produced non-finite values"
  assert jnp.isfinite(dq_ref).all(), "Reference produced non-finite values"
  # 2e-2 tolerance consistent with splash attention precision on TPU MXUs
  np.testing.assert_allclose(dq_kernel, dq_ref, rtol=2e-2, atol=2e-2)


@pytest.mark.skipif(
  not jax.devices()[0].platform == "tpu",
  reason="TPU test - skip on non-TPU platforms",
)
def test_lse_backward_dq_on_tpu():
  """Test LSE backward dQ kernel on TPU without interpret mode."""
  num_heads = 1
  num_tokens = 512
  vocab_size = 1024
  head_dim = 128
  max_valid_id = 1000
  block_size = 128

  key = jax.random.PRNGKey(42)
  key_q, key_k, key_do = jax.random.split(key, 3)
  q = jax.random.normal(key_q, (num_heads, num_tokens, head_dim), dtype=jnp.float32)
  k = jax.random.normal(key_k, (num_heads, vocab_size, head_dim), dtype=jnp.float32)
  do = jax.random.normal(key_do, (num_heads, num_tokens), dtype=jnp.float32)

  # Reference
  logits = jnp.einsum("hsd,htd->hst", q, k)
  vocab_ids = jnp.arange(vocab_size)
  mask = vocab_ids <= max_valid_id
  logits_masked = jnp.where(mask[None, None, :], logits, -1e10)
  s = jax.nn.softmax(logits_masked, axis=-1)
  ds = do[:, :, None] * s
  dq_ref = jnp.einsum("hst,htd->hsd", ds, k)

  # Kernel on TPU
  mask_obj = MultiHeadMask([VocabMask((num_tokens, vocab_size), max_valid_id)])
  block_sizes = BlockSizes(
    block_q=block_size,
    block_kv=block_size,
    block_kv_compute=block_size,
  )
  mask_info, mask_fn = _process_mask(mask_obj, (block_size, block_size), is_dkv=False)

  lse_kernel = _lse_forward(
    fwd_mask_info=mask_info,
    q=q,
    k=k,
    segment_ids=None,
    sinks=None,
    mask_value=-1e10,
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
    bq=block_size,
    bkv=block_size,
    is_mqa=False,
    mask_info=mask_info,
    mask_value=-1e10,
    attn_logits_soft_cap=None,
    q_layout=QKVLayout.HEAD_DIM_MINOR,
    k_layout=QKVLayout.HEAD_DIM_MINOR,
    mask_function=mask_fn,
    interpret=False,
  )

  # 2e-2 tolerance consistent with splash attention precision on TPU MXUs
  np.testing.assert_allclose(dq_kernel, dq_ref, rtol=2e-2, atol=2e-2)


# =============================================================================
# LSE Backward dK Tests (S^T @ Q computation)
# =============================================================================


def test_lse_backward_dk_basic():
  """Test LSE backward dK kernel computes correct S^T @ d_lse_q compared to reference."""
  num_heads = 1
  num_tokens = 512
  vocab_size = 1024
  head_dim = 128
  max_valid_id = 1000
  block_size = 128

  key = jax.random.PRNGKey(42)
  key_q, key_k, key_d_lse = jax.random.split(key, 3)
  q = jax.random.normal(key_q, (num_heads, num_tokens, head_dim), dtype=jnp.float32)
  k = jax.random.normal(key_k, (num_heads, vocab_size, head_dim), dtype=jnp.float32)
  # Random d_lse (upstream gradient on logsumexp)
  d_lse = jax.random.normal(key_d_lse, (num_heads, num_tokens), dtype=jnp.float32)
  # Pre-compute d_lse_q = d_lse[..., None] * q for the kernel
  d_lse_q = d_lse[:, :, None] * q  # (heads, tokens, head_dim)

  # Reference computation
  logits = jnp.einsum("hsd,htd->hst", q, k)  # (heads, tokens, vocab)
  vocab_ids = jnp.arange(vocab_size)
  mask = vocab_ids <= max_valid_id
  logits_masked = jnp.where(mask[None, None, :], logits, -1e10)
  # S = softmax(logits_masked)
  s = jax.nn.softmax(logits_masked, axis=-1)  # (heads, tokens, vocab)
  # S^T @ d_lse_q reference
  dk_ref = jnp.einsum("hst,hsd->htd", s, d_lse_q)  # (heads, vocab, head_dim)

  # Kernel computation
  mask_obj = MultiHeadMask([VocabMask((num_tokens, vocab_size), max_valid_id)])
  block_sizes = BlockSizes(
    block_q=block_size,
    block_kv=block_size,
    block_kv_compute=block_size,
  )
  # For dK we need is_dkv=True for mask processing
  mask_info, mask_fn = _process_mask(mask_obj, (block_size, block_size), is_dkv=True)

  # First get LSE from forward kernel
  fwd_mask_info, fwd_mask_fn = _process_mask(
    mask_obj, (block_size, block_size), is_dkv=False
  )
  lse_kernel = _lse_forward(
    fwd_mask_info=fwd_mask_info,
    q=q,
    k=k,
    segment_ids=None,
    sinks=None,
    mask_value=-1e10,
    is_mqa=False,
    block_sizes=block_sizes,
    mask_function=fwd_mask_fn,
    attn_logits_soft_cap=None,
    interpret=True,
  )

  # Then compute S^T @ d_lse_q using backward dK kernel
  dk_kernel = _lse_backward_dk(
    q=q,
    k=k,
    d_lse_q=d_lse_q,
    segment_ids=None,
    sinks=None,
    logsumexp=lse_kernel,
    bq=block_size,
    bkv=block_size,
    bkv_compute=block_size,
    is_mqa=False,
    mask_info=mask_info,
    mask_value=-1e10,
    attn_logits_soft_cap=None,
    q_layout=QKVLayout.HEAD_DIM_MINOR,
    k_layout=QKVLayout.HEAD_DIM_MINOR,
    mask_function=mask_fn,
    interpret=True,
  )

  # Backward kernel has more numerical accumulation than forward.
  # 2e-2 tolerance is consistent with splash attention's precision on TPU MXUs.
  np.testing.assert_allclose(dk_kernel, dk_ref, rtol=2e-2, atol=2e-2)


def test_lse_backward_dk_no_padding():
  """Test LSE backward dK with all vocab tokens valid."""
  num_heads = 1
  num_tokens = 256
  vocab_size = 512
  head_dim = 128
  max_valid_id = vocab_size - 1
  block_size = 128

  key = jax.random.PRNGKey(123)
  key_q, key_k, key_d_lse = jax.random.split(key, 3)
  q = jax.random.normal(key_q, (num_heads, num_tokens, head_dim), dtype=jnp.float32)
  k = jax.random.normal(key_k, (num_heads, vocab_size, head_dim), dtype=jnp.float32)
  d_lse = jax.random.normal(key_d_lse, (num_heads, num_tokens), dtype=jnp.float32)
  d_lse_q = d_lse[:, :, None] * q

  # Reference: no masking
  logits = jnp.einsum("hsd,htd->hst", q, k)
  s = jax.nn.softmax(logits, axis=-1)
  dk_ref = jnp.einsum("hst,hsd->htd", s, d_lse_q)

  # Kernel
  mask_obj = MultiHeadMask([VocabMask((num_tokens, vocab_size), max_valid_id)])
  block_sizes = BlockSizes(
    block_q=block_size,
    block_kv=block_size,
    block_kv_compute=block_size,
  )
  mask_info, mask_fn = _process_mask(mask_obj, (block_size, block_size), is_dkv=True)

  fwd_mask_info, fwd_mask_fn = _process_mask(
    mask_obj, (block_size, block_size), is_dkv=False
  )
  lse_kernel = _lse_forward(
    fwd_mask_info=fwd_mask_info,
    q=q,
    k=k,
    segment_ids=None,
    sinks=None,
    mask_value=-1e10,
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
    bq=block_size,
    bkv=block_size,
    bkv_compute=block_size,
    is_mqa=False,
    mask_info=mask_info,
    mask_value=-1e10,
    attn_logits_soft_cap=None,
    q_layout=QKVLayout.HEAD_DIM_MINOR,
    k_layout=QKVLayout.HEAD_DIM_MINOR,
    mask_function=mask_fn,
    interpret=True,
  )

  # 2e-2 tolerance consistent with splash attention precision on TPU MXUs
  np.testing.assert_allclose(dk_kernel, dk_ref, rtol=2e-2, atol=2e-2)


def test_lse_backward_dk_numerical_stability():
  """Test LSE backward dK is numerically stable with large values."""
  num_heads = 1
  num_tokens = 256
  vocab_size = 512
  head_dim = 128
  max_valid_id = 500
  block_size = 128

  key = jax.random.PRNGKey(999)
  key_q, key_k, key_d_lse = jax.random.split(key, 3)
  q = (
    jax.random.normal(key_q, (num_heads, num_tokens, head_dim), dtype=jnp.float32)
    * 10.0
  )
  k = (
    jax.random.normal(key_k, (num_heads, vocab_size, head_dim), dtype=jnp.float32)
    * 10.0
  )
  d_lse = jax.random.normal(key_d_lse, (num_heads, num_tokens), dtype=jnp.float32)
  d_lse_q = d_lse[:, :, None] * q

  # Reference
  logits = jnp.einsum("hsd,htd->hst", q, k)
  vocab_ids = jnp.arange(vocab_size)
  mask = vocab_ids <= max_valid_id
  logits_masked = jnp.where(mask[None, None, :], logits, -1e10)
  s = jax.nn.softmax(logits_masked, axis=-1)
  dk_ref = jnp.einsum("hst,hsd->htd", s, d_lse_q)

  # Kernel
  mask_obj = MultiHeadMask([VocabMask((num_tokens, vocab_size), max_valid_id)])
  block_sizes = BlockSizes(
    block_q=block_size,
    block_kv=block_size,
    block_kv_compute=block_size,
  )
  mask_info, mask_fn = _process_mask(mask_obj, (block_size, block_size), is_dkv=True)

  fwd_mask_info, fwd_mask_fn = _process_mask(
    mask_obj, (block_size, block_size), is_dkv=False
  )
  lse_kernel = _lse_forward(
    fwd_mask_info=fwd_mask_info,
    q=q,
    k=k,
    segment_ids=None,
    sinks=None,
    mask_value=-1e10,
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
    bq=block_size,
    bkv=block_size,
    bkv_compute=block_size,
    is_mqa=False,
    mask_info=mask_info,
    mask_value=-1e10,
    attn_logits_soft_cap=None,
    q_layout=QKVLayout.HEAD_DIM_MINOR,
    k_layout=QKVLayout.HEAD_DIM_MINOR,
    mask_function=mask_fn,
    interpret=True,
  )

  assert jnp.isfinite(dk_kernel).all(), "Kernel produced non-finite values"
  assert jnp.isfinite(dk_ref).all(), "Reference produced non-finite values"
  # 2e-2 tolerance consistent with splash attention precision on TPU MXUs
  np.testing.assert_allclose(dk_kernel, dk_ref, rtol=2e-2, atol=2e-2)


@pytest.mark.skipif(
  not jax.devices()[0].platform == "tpu",
  reason="TPU test - skip on non-TPU platforms",
)
def test_lse_backward_dk_on_tpu():
  """Test LSE backward dK kernel on TPU without interpret mode."""
  num_heads = 1
  num_tokens = 512
  vocab_size = 1024
  head_dim = 128
  max_valid_id = 1000
  block_size = 128

  key = jax.random.PRNGKey(42)
  key_q, key_k, key_d_lse = jax.random.split(key, 3)
  q = jax.random.normal(key_q, (num_heads, num_tokens, head_dim), dtype=jnp.float32)
  k = jax.random.normal(key_k, (num_heads, vocab_size, head_dim), dtype=jnp.float32)
  d_lse = jax.random.normal(key_d_lse, (num_heads, num_tokens), dtype=jnp.float32)
  d_lse_q = d_lse[:, :, None] * q

  # Reference
  logits = jnp.einsum("hsd,htd->hst", q, k)
  vocab_ids = jnp.arange(vocab_size)
  mask = vocab_ids <= max_valid_id
  logits_masked = jnp.where(mask[None, None, :], logits, -1e10)
  s = jax.nn.softmax(logits_masked, axis=-1)
  dk_ref = jnp.einsum("hst,hsd->htd", s, d_lse_q)

  # Kernel on TPU
  mask_obj = MultiHeadMask([VocabMask((num_tokens, vocab_size), max_valid_id)])
  block_sizes = BlockSizes(
    block_q=block_size,
    block_kv=block_size,
    block_kv_compute=block_size,
  )
  mask_info, mask_fn = _process_mask(mask_obj, (block_size, block_size), is_dkv=True)

  fwd_mask_info, fwd_mask_fn = _process_mask(
    mask_obj, (block_size, block_size), is_dkv=False
  )
  lse_kernel = _lse_forward(
    fwd_mask_info=fwd_mask_info,
    q=q,
    k=k,
    segment_ids=None,
    sinks=None,
    mask_value=-1e10,
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
    bq=block_size,
    bkv=block_size,
    bkv_compute=block_size,
    is_mqa=False,
    mask_info=mask_info,
    mask_value=-1e10,
    attn_logits_soft_cap=None,
    q_layout=QKVLayout.HEAD_DIM_MINOR,
    k_layout=QKVLayout.HEAD_DIM_MINOR,
    mask_function=mask_fn,
    interpret=False,
  )

  # 2e-2 tolerance consistent with splash attention precision on TPU MXUs
  np.testing.assert_allclose(dk_kernel, dk_ref, rtol=2e-2, atol=2e-2)


# =============================================================================
# LSE Custom VJP Tests (full differentiable wrapper)
# =============================================================================


def test_lse_custom_forward():
  """Test _lse_custom forward pass computes correct LSE."""
  num_heads = 1
  num_tokens = 256
  vocab_size = 512
  head_dim = 128
  max_valid_id = 500
  block_size = 128

  key = jax.random.PRNGKey(42)
  key_q, key_k = jax.random.split(key)
  q = jax.random.normal(key_q, (num_heads, num_tokens, head_dim), dtype=jnp.float32)
  k = jax.random.normal(key_k, (num_heads, vocab_size, head_dim), dtype=jnp.float32)

  # Reference
  logits = jnp.einsum("hsd,htd->hst", q, k)
  vocab_ids = jnp.arange(vocab_size)
  mask = vocab_ids <= max_valid_id
  logits_masked = jnp.where(mask[None, None, :], logits, -1e10)
  lse_ref = jax.nn.logsumexp(logits_masked, axis=-1)

  # Kernel via custom_vjp wrapper
  mask_obj = MultiHeadMask([VocabMask((num_tokens, vocab_size), max_valid_id)])
  block_sizes = BlockSizes(
    block_q=block_size,
    block_kv=block_size,
    block_kv_compute=block_size,
  )
  fwd_mask_info, mask_fn = _process_mask(
    mask_obj, (block_size, block_size), is_dkv=False
  )
  dq_mask_info, _ = _process_mask(mask_obj, (block_size, block_size), is_dkv=False)
  dk_mask_info, _ = _process_mask(mask_obj, (block_size, block_size), is_dkv=True)

  lse_kernel = _lse_custom(
    fwd_mask_info=fwd_mask_info,
    dq_mask_info=dq_mask_info,
    dk_mask_info=dk_mask_info,
    q=q,
    k=k,
    segment_ids=None,
    sinks=None,
    mask_value=-1e10,
    is_mqa=False,
    block_sizes=block_sizes,
    mask_function=mask_fn,
    attn_logits_soft_cap=None,
    interpret=True,
  )

  np.testing.assert_allclose(lse_kernel, lse_ref, rtol=1e-4, atol=1e-4)


def test_lse_custom_gradients():
  """Test _lse_custom gradients via jax.grad match reference."""
  num_heads = 1
  num_tokens = 256
  vocab_size = 512
  head_dim = 128
  max_valid_id = 500
  block_size = 128

  key = jax.random.PRNGKey(123)
  key_q, key_k = jax.random.split(key)
  q = jax.random.normal(key_q, (num_heads, num_tokens, head_dim), dtype=jnp.float32)
  k = jax.random.normal(key_k, (num_heads, vocab_size, head_dim), dtype=jnp.float32)

  # Reference loss and gradients
  def ref_loss(q, k):
    logits = jnp.einsum("hsd,htd->hst", q, k)
    vocab_ids = jnp.arange(vocab_size)
    mask = vocab_ids <= max_valid_id
    logits_masked = jnp.where(mask[None, None, :], logits, -1e10)
    lse = jax.nn.logsumexp(logits_masked, axis=-1)
    return lse.sum()

  loss_ref = ref_loss(q, k)
  dq_ref, dk_ref = jax.grad(ref_loss, argnums=(0, 1))(q, k)

  # Kernel loss and gradients
  mask_obj = MultiHeadMask([VocabMask((num_tokens, vocab_size), max_valid_id)])
  block_sizes = BlockSizes(
    block_q=block_size,
    block_kv=block_size,
    block_kv_compute=block_size,
  )
  fwd_mask_info, mask_fn = _process_mask(
    mask_obj, (block_size, block_size), is_dkv=False
  )
  dq_mask_info, _ = _process_mask(mask_obj, (block_size, block_size), is_dkv=False)
  dk_mask_info, _ = _process_mask(mask_obj, (block_size, block_size), is_dkv=True)

  def kernel_loss(q, k):
    lse = _lse_custom(
      fwd_mask_info=fwd_mask_info,
      dq_mask_info=dq_mask_info,
      dk_mask_info=dk_mask_info,
      q=q,
      k=k,
      segment_ids=None,
      sinks=None,
      mask_value=-1e10,
      is_mqa=False,
      block_sizes=block_sizes,
      mask_function=mask_fn,
      attn_logits_soft_cap=None,
      interpret=True,
    )
    return lse.sum()

  loss_kernel = kernel_loss(q, k)
  dq_kernel, dk_kernel = jax.grad(kernel_loss, argnums=(0, 1))(q, k)

  # Forward pass should match
  np.testing.assert_allclose(loss_kernel, loss_ref, rtol=1e-4, atol=1e-4)

  # Gradients should match (2e-2 tolerance for kernel precision)
  np.testing.assert_allclose(dq_kernel, dq_ref, rtol=2e-2, atol=2e-2)
  np.testing.assert_allclose(dk_kernel, dk_ref, rtol=2e-2, atol=2e-2)


@pytest.mark.skipif(
  not jax.devices()[0].platform == "tpu",
  reason="TPU test - skip on non-TPU platforms",
)
def test_lse_custom_gradients_on_tpu():
  """Test _lse_custom gradients on TPU without interpret mode."""
  num_heads = 1
  num_tokens = 512
  vocab_size = 1024
  head_dim = 128
  max_valid_id = 1000
  block_size = 128

  key = jax.random.PRNGKey(456)
  key_q, key_k = jax.random.split(key)
  q = jax.random.normal(key_q, (num_heads, num_tokens, head_dim), dtype=jnp.float32)
  k = jax.random.normal(key_k, (num_heads, vocab_size, head_dim), dtype=jnp.float32)

  # Reference
  def ref_loss(q, k):
    logits = jnp.einsum("hsd,htd->hst", q, k)
    vocab_ids = jnp.arange(vocab_size)
    mask = vocab_ids <= max_valid_id
    logits_masked = jnp.where(mask[None, None, :], logits, -1e10)
    lse = jax.nn.logsumexp(logits_masked, axis=-1)
    return lse.sum()

  loss_ref = ref_loss(q, k)
  dq_ref, dk_ref = jax.grad(ref_loss, argnums=(0, 1))(q, k)

  # Kernel on TPU
  mask_obj = MultiHeadMask([VocabMask((num_tokens, vocab_size), max_valid_id)])
  block_sizes = BlockSizes(
    block_q=block_size,
    block_kv=block_size,
    block_kv_compute=block_size,
  )
  fwd_mask_info, mask_fn = _process_mask(
    mask_obj, (block_size, block_size), is_dkv=False
  )
  dq_mask_info, _ = _process_mask(mask_obj, (block_size, block_size), is_dkv=False)
  dk_mask_info, _ = _process_mask(mask_obj, (block_size, block_size), is_dkv=True)

  def kernel_loss(q, k):
    lse = _lse_custom(
      fwd_mask_info=fwd_mask_info,
      dq_mask_info=dq_mask_info,
      dk_mask_info=dk_mask_info,
      q=q,
      k=k,
      segment_ids=None,
      sinks=None,
      mask_value=-1e10,
      is_mqa=False,
      block_sizes=block_sizes,
      mask_function=mask_fn,
      attn_logits_soft_cap=None,
      interpret=False,  # Real TPU execution
    )
    return lse.sum()

  loss_kernel = kernel_loss(q, k)
  dq_kernel, dk_kernel = jax.grad(kernel_loss, argnums=(0, 1))(q, k)

  np.testing.assert_allclose(loss_kernel, loss_ref, rtol=1e-4, atol=1e-4)
  np.testing.assert_allclose(dq_kernel, dq_ref, rtol=2e-2, atol=2e-2)
  np.testing.assert_allclose(dk_kernel, dk_ref, rtol=2e-2, atol=2e-2)


# =============================================================================
# LSEKernel and make_lse_kernel Tests
# =============================================================================


@pytest.mark.skipif(
  not jax.devices()[0].platform == "tpu",
  reason="TPU test - skip on non-TPU platforms",
)
def test_lse_kernel_sharded_q_seq():
  """Test LSEKernel with q_seq sharding via shard_map."""

  num_heads = 1
  vocab_size = 2048
  head_dim = 128
  max_valid_id = 2000
  block_size = 128

  # Get number of devices for sharding
  q_seq_shards = len(jax.devices())

  # num_tokens must be divisible by q_seq_shards * block_size
  # so each shard gets at least one full block
  num_tokens = q_seq_shards * block_size

  key = jax.random.PRNGKey(789)
  key_q, key_k = jax.random.split(key)
  q = jax.random.normal(key_q, (num_heads, num_tokens, head_dim), dtype=jnp.float32)
  k = jax.random.normal(key_k, (num_heads, vocab_size, head_dim), dtype=jnp.float32)

  # Reference (unsharded)
  def ref_loss(q, k):
    logits = jnp.einsum("hsd,htd->hst", q, k)
    vocab_ids = jnp.arange(vocab_size)
    mask = vocab_ids <= max_valid_id
    logits_masked = jnp.where(mask[None, None, :], logits, -1e10)
    lse = jax.nn.logsumexp(logits_masked, axis=-1)
    return lse.sum()

  loss_ref = ref_loss(q, k)
  dq_ref, dk_ref = jax.grad(ref_loss, argnums=(0, 1))(q, k)

  # Sharded kernel
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

  # Create mesh and sharding (match config.py pattern with explicit axis types)
  mesh = jax.make_mesh((q_seq_shards,), ("q_seq",), (jax.sharding.AxisType.Explicit,))
  q_spec = jax.P(None, "q_seq", None)  # (heads, tokens, head_dim)
  k_spec = jax.P(None, None, None)  # (heads, vocab, head_dim) - replicated
  lse_spec = jax.P(None, "q_seq")  # (heads, tokens)

  # Get kernel sharding spec
  kernel_sharding = jax.sharding.NamedSharding(mesh, jax.P(None, "q_seq"))
  kernel_spec = kernel.manual_sharding_spec(kernel_sharding)

  # shard_mapped kernel (following splash_helpers pattern)
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

  # Use jax.set_mesh context (match train.py pattern)
  with jax.set_mesh(mesh):
    loss_kernel = sharded_loss(q, k)
    dq_kernel, dk_kernel = jax.grad(sharded_loss, argnums=(0, 1))(q, k)

  # Use process_allgather to collect sharded arrays in multi-host setup
  dq_kernel = multihost_utils.process_allgather(dq_kernel, tiled=True)
  dk_kernel = multihost_utils.process_allgather(dk_kernel, tiled=True)

  np.testing.assert_allclose(loss_kernel, loss_ref, rtol=1e-4, atol=1e-4)
  np.testing.assert_allclose(dq_kernel, dq_ref, rtol=2e-2, atol=2e-2)
  np.testing.assert_allclose(dk_kernel, dk_ref, rtol=2e-2, atol=2e-2)


@pytest.mark.skipif(
  not jax.devices()[0].platform == "tpu",
  reason="TPU test - skip on non-TPU platforms",
)
def test_lse_kernel_memory_pressure():
  """Test LSE kernel under memory pressure with large batch and vocab.

  Uses BS=8M (2^23) tokens and V=128K (2^17) vocab size in q_seq sharded mode.
  This stresses memory bandwidth and tests the kernel at production-like scale.
  The full logits matrix would be 8M × 128K = 1T elements (4TB in fp32), but
  the kernel computes LSE via online softmax without materializing it.
  """
  num_heads = 1
  head_dim = 128
  block_size = 512  # larger blocks for efficiency at scale

  # Large scale: 8M tokens, 128K vocab
  num_tokens = 2**23  # 8M = 8,388,608
  vocab_size = 2**17  # 128K = 131,072
  max_valid_id = vocab_size - 1024  # leave some padding tokens

  # Get number of devices for sharding
  q_seq_shards = len(jax.devices())

  # First, verify naive JAX implementation OOMs at this scale
  key = jax.random.PRNGKey(999)
  key_q, key_k = jax.random.split(key)
  q = jax.random.normal(key_q, (num_heads, num_tokens, head_dim), dtype=jnp.bfloat16)
  k = jax.random.normal(key_k, (num_heads, vocab_size, head_dim), dtype=jnp.bfloat16)

  @jax.jit
  def naive_lse(q, k):
    """Naive LSE that materializes full logits matrix - should OOM."""
    logits = jnp.einsum("hsd,htd->hst", q, k)  # 8M x 128K = 1T elements!
    return jax.nn.logsumexp(logits, axis=-1).sum()

  try:
    _ = naive_lse(q, k).block_until_ready()
    naive_oom = False
  except Exception as e:
    naive_oom = True
    print(f"\nNaive LSE OOM as expected: {type(e).__name__}")

  assert naive_oom, "Naive LSE should OOM at 8M x 128K scale but didn't!"

  # Verify num_tokens is divisible by shards * block_size
  assert num_tokens % (q_seq_shards * block_size) == 0, (
    f"num_tokens={num_tokens} must be divisible by "
    f"q_seq_shards={q_seq_shards} * block_size={block_size}"
  )

  # Now test that our kernel handles this scale (reuse q, k from above)
  # Sharded kernel setup
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

  # Create mesh and sharding
  mesh = jax.make_mesh((q_seq_shards,), ("q_seq",), (jax.sharding.AxisType.Explicit,))
  q_spec = jax.P(None, "q_seq", None)  # (heads, tokens, head_dim)
  k_spec = jax.P(None, None, None)  # (heads, vocab, head_dim) - replicated
  lse_spec = jax.P(None, "q_seq")  # (heads, tokens)

  # Get kernel sharding spec
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

  # Run forward and backward under memory pressure
  with jax.set_mesh(mesh):
    loss = sharded_loss(q, k)
    dq, dk = jax.grad(sharded_loss, argnums=(0, 1))(q, k)

  # Collect sharded arrays
  dq = multihost_utils.process_allgather(dq, tiled=True)
  dk = multihost_utils.process_allgather(dk, tiled=True)

  # Basic sanity checks (no reference computation due to memory constraints)
  assert jnp.isfinite(loss), f"Loss is not finite: {loss}"
  assert jnp.all(jnp.isfinite(dq)), "dQ contains non-finite values"
  assert jnp.all(jnp.isfinite(dk)), "dK contains non-finite values"

  # Check shapes
  assert dq.shape == q.shape, f"dQ shape mismatch: {dq.shape} vs {q.shape}"
  assert dk.shape == k.shape, f"dK shape mismatch: {dk.shape} vs {k.shape}"

  # Log memory stats for debugging
  print("\nMemory pressure test completed:")
  print(f"  num_tokens: {num_tokens:,} (2^{num_tokens.bit_length() - 1})")
  print(f"  vocab_size: {vocab_size:,} (2^{vocab_size.bit_length() - 1})")
  print(f"  q_seq_shards: {q_seq_shards}")
  print(f"  loss: {float(loss):.6f}")


def test_make_lse_kernel_forward():
  """Test make_lse_kernel factory and LSEKernel forward pass."""
  num_heads = 1
  num_tokens = 256
  vocab_size = 512
  head_dim = 128
  max_valid_id = 500

  key = jax.random.PRNGKey(42)
  key_q, key_k = jax.random.split(key)
  q = jax.random.normal(key_q, (num_heads, num_tokens, head_dim), dtype=jnp.float32)
  k = jax.random.normal(key_k, (num_heads, vocab_size, head_dim), dtype=jnp.float32)

  # Reference
  logits = jnp.einsum("hsd,htd->hst", q, k)
  vocab_ids = jnp.arange(vocab_size)
  mask = vocab_ids <= max_valid_id
  logits_masked = jnp.where(mask[None, None, :], logits, -1e10)
  lse_ref = jax.nn.logsumexp(logits_masked, axis=-1)

  # Use make_lse_kernel factory
  mask_obj = MultiHeadMask([VocabMask((num_tokens, vocab_size), max_valid_id)])
  kernel = make_lse_kernel(
    mask_obj,
    is_mqa=False,
    interpret=True,
  )

  lse_kernel = kernel(q, k)

  np.testing.assert_allclose(lse_kernel, lse_ref, rtol=1e-4, atol=1e-4)


def test_make_lse_kernel_gradients():
  """Test make_lse_kernel factory with gradients via jax.grad."""
  num_heads = 1
  num_tokens = 256
  vocab_size = 512
  head_dim = 128
  max_valid_id = 500

  key = jax.random.PRNGKey(123)
  key_q, key_k = jax.random.split(key)
  q = jax.random.normal(key_q, (num_heads, num_tokens, head_dim), dtype=jnp.float32)
  k = jax.random.normal(key_k, (num_heads, vocab_size, head_dim), dtype=jnp.float32)

  # Reference
  def ref_loss(q, k):
    logits = jnp.einsum("hsd,htd->hst", q, k)
    vocab_ids = jnp.arange(vocab_size)
    mask = vocab_ids <= max_valid_id
    logits_masked = jnp.where(mask[None, None, :], logits, -1e10)
    lse = jax.nn.logsumexp(logits_masked, axis=-1)
    return lse.sum()

  loss_ref = ref_loss(q, k)
  dq_ref, dk_ref = jax.grad(ref_loss, argnums=(0, 1))(q, k)

  # Use make_lse_kernel factory
  mask_obj = MultiHeadMask([VocabMask((num_tokens, vocab_size), max_valid_id)])
  kernel = make_lse_kernel(
    mask_obj,
    is_mqa=False,
    interpret=True,
  )

  def kernel_loss(q, k):
    lse = kernel(q, k)
    return lse.sum()

  loss_kernel = kernel_loss(q, k)
  dq_kernel, dk_kernel = jax.grad(kernel_loss, argnums=(0, 1))(q, k)

  np.testing.assert_allclose(loss_kernel, loss_ref, rtol=1e-4, atol=1e-4)
  np.testing.assert_allclose(dq_kernel, dq_ref, rtol=2e-2, atol=2e-2)
  np.testing.assert_allclose(dk_kernel, dk_ref, rtol=2e-2, atol=2e-2)


@pytest.mark.skipif(
  not jax.devices()[0].platform == "tpu",
  reason="TPU test - skip on non-TPU platforms",
)
def test_make_lse_kernel_on_tpu():
  """Test make_lse_kernel on TPU without interpret mode."""
  num_heads = 1
  num_tokens = 512
  vocab_size = 1024
  head_dim = 128
  max_valid_id = 1000

  key = jax.random.PRNGKey(456)
  key_q, key_k = jax.random.split(key)
  q = jax.random.normal(key_q, (num_heads, num_tokens, head_dim), dtype=jnp.float32)
  k = jax.random.normal(key_k, (num_heads, vocab_size, head_dim), dtype=jnp.float32)

  # Reference
  def ref_loss(q, k):
    logits = jnp.einsum("hsd,htd->hst", q, k)
    vocab_ids = jnp.arange(vocab_size)
    mask = vocab_ids <= max_valid_id
    logits_masked = jnp.where(mask[None, None, :], logits, -1e10)
    lse = jax.nn.logsumexp(logits_masked, axis=-1)
    return lse.sum()

  loss_ref = ref_loss(q, k)
  dq_ref, dk_ref = jax.grad(ref_loss, argnums=(0, 1))(q, k)

  # Kernel on TPU
  mask_obj = MultiHeadMask([VocabMask((num_tokens, vocab_size), max_valid_id)])
  kernel = make_lse_kernel(
    mask_obj,
    is_mqa=False,
    interpret=False,  # Real TPU execution
  )

  def kernel_loss(q, k):
    lse = kernel(q, k)
    return lse.sum()

  loss_kernel = kernel_loss(q, k)
  dq_kernel, dk_kernel = jax.grad(kernel_loss, argnums=(0, 1))(q, k)

  np.testing.assert_allclose(loss_kernel, loss_ref, rtol=1e-4, atol=1e-4)
  np.testing.assert_allclose(dq_kernel, dq_ref, rtol=2e-2, atol=2e-2)
  np.testing.assert_allclose(dk_kernel, dk_ref, rtol=2e-2, atol=2e-2)
