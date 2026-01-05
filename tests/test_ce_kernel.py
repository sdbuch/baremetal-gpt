"""Tests for the CE forward kernel (fused cross-entropy LSE computation)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from bmgpt.ce_kernel import BlockSizes, _ce_forward
from bmgpt.ce_mask import MultiHeadMask, VocabMask
from bmgpt.ce_mask_info import _process_mask


def test_ce_forward_lse_basic():
  """Test CE forward kernel computes correct LSE compared to reference."""
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
  lse_kernel = _ce_forward(
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


def test_ce_forward_lse_no_padding():
  """Test CE forward with all vocab tokens valid (edge case)."""
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
  lse_kernel = _ce_forward(
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


def test_ce_forward_numerical_stability():
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
  lse_kernel = _ce_forward(
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


def test_ce_forward_larger_vocab():
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
  lse_kernel = _ce_forward(
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
def test_ce_forward_on_tpu():
  """Test CE forward kernel on TPU without interpret mode."""
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
  lse_kernel = _ce_forward(
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
