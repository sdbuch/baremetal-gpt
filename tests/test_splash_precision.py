"""Test splash attention precision vs XLA reference.

This test checks whether splash attention's backward pass has similar
precision characteristics to our CE kernel backward.
"""

import jax
import jax.numpy as jnp
import numpy as np

# Import splash attention from JAX
from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_kernel
from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_mask
from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_mask_info


def reference_attention_forward(q, k, v, mask=None):
  """Reference attention forward pass using XLA."""
  # q: (heads, seq_q, dim)
  # k: (heads, seq_kv, dim)
  # v: (heads, seq_kv, dim_v)
  logits = jnp.einsum("hsd,htd->hst", q, k)
  if mask is not None:
    logits = jnp.where(mask, logits, -1e10)
  s = jax.nn.softmax(logits, axis=-1)
  o = jnp.einsum("hst,htd->hsd", s, v)
  return o


def reference_attention_backward_dq(q, k, v, do, mask=None):
  """Reference dQ computation using XLA.

  dQ = ds @ K where ds = (dp - di) * p
  dp = dO @ V^T
  di = sum(O * dO, axis=-1)
  p = softmax(Q @ K^T)
  """
  # Forward pass to get intermediates
  logits = jnp.einsum("hsd,htd->hst", q, k)
  if mask is not None:
    logits = jnp.where(mask, logits, -1e10)
  p = jax.nn.softmax(logits, axis=-1)
  o = jnp.einsum("hst,htd->hsd", p, v)

  # Backward pass
  dp = jnp.einsum("hsd,htd->hst", do, v)  # dO @ V^T
  di = jnp.sum(o * do, axis=-1, keepdims=True)  # sum(O * dO)
  ds = (dp - di) * p  # Jacobian correction
  dq = jnp.einsum("hst,htd->hsd", ds, k)  # ds @ K

  return dq


def test_splash_attention_dq_precision():
  """Compare splash attention dQ to XLA reference."""
  num_heads = 1
  seq_q = 512
  seq_kv = 1024
  head_dim = 128
  block_size = 128

  key = jax.random.PRNGKey(42)
  keys = jax.random.split(key, 4)
  q = jax.random.normal(keys[0], (num_heads, seq_q, head_dim), dtype=jnp.float32)
  k = jax.random.normal(keys[1], (num_heads, seq_kv, head_dim), dtype=jnp.float32)
  v = jax.random.normal(keys[2], (num_heads, seq_kv, head_dim), dtype=jnp.float32)
  do = jax.random.normal(keys[3], (num_heads, seq_q, head_dim), dtype=jnp.float32)

  # Reference computation
  dq_ref = reference_attention_backward_dq(q, k, v, do)

  # Splash attention computation via autodiff
  # Use FullMask (no masking) for simplicity
  mask = splash_attention_mask.FullMask((seq_q, seq_kv))
  multi_head_mask = splash_attention_mask.MultiHeadMask([mask])

  block_sizes = splash_attention_kernel.BlockSizes(
    block_q=block_size,
    block_kv=block_size,
    block_kv_compute=block_size,
  )

  # Process masks for forward and backward
  fwd_mask_info, mask_fn = splash_attention_mask_info._process_mask(
    multi_head_mask, (block_size, block_size), is_dkv=False
  )
  dkv_mask_info, _ = splash_attention_mask_info._process_mask(
    multi_head_mask, (block_size, block_size), is_dkv=True
  )

  # Check if we're on TPU
  is_tpu = jax.devices()[0].platform == "tpu"
  interpret = not is_tpu

  def splash_forward(q, k, v):
    result = splash_attention_kernel._splash_attention(
      fwd_mask_info=fwd_mask_info,
      dq_mask_info=fwd_mask_info,
      dkv_mask_info=dkv_mask_info,
      q=q,
      k=k,
      v=v,
      segment_ids=None,
      sinks=None,
      is_mqa=False,
      block_sizes=block_sizes,
      save_residuals=False,
      mask_value=-1e10,
      attn_logits_soft_cap=None,
      residual_checkpoint_name=None,
      mask_function=mask_fn,
      interpret=interpret,
    )
    return result.o

  # Compute gradient w.r.t. q
  def loss_fn(q):
    o = splash_forward(q, k, v)
    return jnp.sum(o * do)

  dq_splash = jax.grad(loss_fn)(q)

  # Compare
  max_abs_diff = jnp.max(jnp.abs(dq_splash - dq_ref))
  max_rel_diff = jnp.max(jnp.abs(dq_splash - dq_ref) / (jnp.abs(dq_ref) + 1e-8))

  print(f"Platform: {jax.devices()[0].platform}")
  print(f"Max absolute difference: {max_abs_diff}")
  print(f"Max relative difference: {max_rel_diff}")
  print(f"Reference dQ range: [{jnp.min(dq_ref)}, {jnp.max(dq_ref)}]")
  print(f"Splash dQ range: [{jnp.min(dq_splash)}, {jnp.max(dq_splash)}]")

  # Test with different tolerances
  for tol in [1e-4, 1e-3, 1e-2, 2e-2, 5e-2]:
    try:
      np.testing.assert_allclose(dq_splash, dq_ref, rtol=tol, atol=tol)
      print(f"PASS at tolerance {tol}")
      break
    except AssertionError:
      print(f"FAIL at tolerance {tol}")


def test_splash_attention_s_times_k_precision():
  """Compare S @ K computation (like our CE backward dQ, but simpler).

  This tests just the S @ K part without the Jacobian correction,
  which is what our CE kernel computes.
  """
  num_heads = 1
  seq_q = 512
  seq_kv = 1024
  head_dim = 128
  block_size = 128

  key = jax.random.PRNGKey(42)
  keys = jax.random.split(key, 2)
  q = jax.random.normal(keys[0], (num_heads, seq_q, head_dim), dtype=jnp.float32)
  k = jax.random.normal(keys[1], (num_heads, seq_kv, head_dim), dtype=jnp.float32)

  # Reference: S @ K where S = softmax(Q @ K^T)
  logits = jnp.einsum("hsd,htd->hst", q, k)
  s = jax.nn.softmax(logits, axis=-1)
  s_times_k_ref = jnp.einsum("hst,htd->hsd", s, k)

  # Splash attention forward with V=K gives us S @ K
  mask = splash_attention_mask.FullMask((seq_q, seq_kv))
  multi_head_mask = splash_attention_mask.MultiHeadMask([mask])

  block_sizes = splash_attention_kernel.BlockSizes(
    block_q=block_size,
    block_kv=block_size,
    block_kv_compute=block_size,
  )

  fwd_mask_info, mask_fn = splash_attention_mask_info._process_mask(
    multi_head_mask, (block_size, block_size), is_dkv=False
  )

  is_tpu = jax.devices()[0].platform == "tpu"
  interpret = not is_tpu

  # Use splash attention forward with V=K to compute S @ K
  # Signature: (fwd_mask_info, q, k, v, segment_ids, sinks, mask_value, is_mqa,
  #             block_sizes, residual_checkpoint_name, save_residuals,
  #             mask_function, attn_logits_soft_cap, interpret)
  # When save_residuals=True, return is (out, (logsumexp,))
  # When save_residuals=False, return is just out
  result = splash_attention_kernel._splash_attention_forward(  # type: ignore[call-overload]
    fwd_mask_info,
    q,
    k,
    k,  # V = K to get S @ K
    None,  # segment_ids
    None,  # sinks
    -1e10,  # mask_value
    False,  # is_mqa
    block_sizes,
    None,  # residual_checkpoint_name
    False,  # save_residuals=False returns just the output
    mask_fn,
    None,  # attn_logits_soft_cap
    interpret,
  )
  s_times_k_splash = (
    result  # When save_residuals=False, result is just the output array
  )

  # Compare
  max_abs_diff = jnp.max(jnp.abs(s_times_k_splash - s_times_k_ref))
  max_rel_diff = jnp.max(
    jnp.abs(s_times_k_splash - s_times_k_ref) / (jnp.abs(s_times_k_ref) + 1e-8)
  )

  print("\n--- S @ K Comparison (like CE backward dQ) ---")
  print(f"Platform: {jax.devices()[0].platform}")
  print(f"Max absolute difference: {max_abs_diff}")
  print(f"Max relative difference: {max_rel_diff}")
  print(f"Reference range: [{jnp.min(s_times_k_ref)}, {jnp.max(s_times_k_ref)}]")
  print(f"Splash range: [{jnp.min(s_times_k_splash)}, {jnp.max(s_times_k_splash)}]")

  for tol in [1e-4, 1e-3, 1e-2, 2e-2, 5e-2]:
    try:
      np.testing.assert_allclose(s_times_k_splash, s_times_k_ref, rtol=tol, atol=tol)
      print(f"PASS at tolerance {tol}")
      break
    except AssertionError:
      print(f"FAIL at tolerance {tol}")


if __name__ == "__main__":
  # Skip the dQ test for now - splash attention's custom_vjp doesn't play well
  # with jax.grad on named tuple fields. The S @ K test is what we care about
  # for CE kernel comparison anyway.
  #
  # print("=" * 60)
  # print("Testing splash attention dQ precision")
  # print("=" * 60)
  # test_splash_attention_dq_precision()

  print("=" * 60)
  print("Testing S @ K precision (CE-style computation)")
  print("=" * 60)
  test_splash_attention_s_times_k_precision()
