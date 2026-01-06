"""Tests for cross-entropy loss functions with the fused LSE kernel."""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import multihost_utils
from jax.experimental.pallas.ops.tpu.splash_attention.splash_attention_mask import (
  MultiHeadMask,
)

from bmgpt.kernels.lse_kernel import BlockSizes, make_lse_kernel
from bmgpt.kernels.lse_mask import VocabMask
from bmgpt.losses import fused_softmax_cross_entropy, softmax_cross_entropy
from bmgpt.model import LMHead
from bmgpt.splash_helpers import make_lse_kernel_sharded

from .conftest import (
  DEFAULT_BLOCK_SIZE,
  FORWARD_ATOL,
  FORWARD_RTOL,
  SMALL_CONFIG_IDS,
  SMALL_CONFIGS,
  TPU_CONFIG_IDS,
  TPU_CONFIGS,
  make_test_config,
  ref_cross_entropy,
  requires_tpu,
)


def fused_cross_entropy(
  outputs: jax.Array,
  w_unemb: jax.Array,
  targets: jax.Array,
  max_valid_id: int,
  block_size: int = DEFAULT_BLOCK_SIZE,
  interpret: bool = True,
) -> jax.Array:
  """Fused cross-entropy using LSE kernel directly (for kernel testing)."""
  b, s = outputs.shape[:2]
  vocab_size = w_unemb.shape[0]
  outputs_flat = outputs.reshape(b * s, -1)
  targets_flat = targets.ravel()

  q = outputs_flat[None]
  k = w_unemb[None]

  mask_obj = MultiHeadMask([VocabMask((b * s, vocab_size), max_valid_id)])
  block_sizes = BlockSizes(
    block_q=block_size,
    block_kv=block_size,
    block_kv_compute=block_size,
  )
  kernel = make_lse_kernel(
    mask_obj,
    block_sizes=block_sizes,
    is_mqa=False,
    interpret=interpret,
  )
  lse = kernel(q, k).squeeze(0)

  per_token_unembs = w_unemb[targets_flat]
  label_logits = jnp.sum(outputs_flat * per_token_unembs, axis=-1)

  return (lse - label_logits).mean()


# =============================================================================
# Kernel Internals Tests - test the LSE kernel directly
# =============================================================================


@pytest.mark.parametrize(
  "batch_size,seq_len,d_model,vocab_size", SMALL_CONFIGS, ids=SMALL_CONFIG_IDS
)
def test_fused_cross_entropy_forward(batch_size, seq_len, d_model, vocab_size):
  """Test fused cross-entropy forward pass matches reference."""
  key = jax.random.PRNGKey(42)
  key_out, key_w, key_tgt = jax.random.split(key, 3)

  outputs = jax.random.normal(
    key_out, (batch_size, seq_len, d_model), dtype=jnp.float32
  )
  w_unemb = jax.random.normal(key_w, (vocab_size, d_model), dtype=jnp.float32)
  max_valid_id = vocab_size - 1
  targets = jax.random.randint(key_tgt, (batch_size, seq_len), 0, max_valid_id)

  loss_ref = jax.jit(partial(ref_cross_entropy, max_valid_id=max_valid_id))(
    outputs, w_unemb, targets
  )
  loss_fused = jax.jit(
    partial(
      fused_cross_entropy, max_valid_id=max_valid_id, block_size=DEFAULT_BLOCK_SIZE
    )
  )(outputs, w_unemb, targets)

  np.testing.assert_allclose(loss_fused, loss_ref, rtol=FORWARD_RTOL, atol=FORWARD_ATOL)


@pytest.mark.parametrize(
  "batch_size,seq_len,d_model,vocab_size", SMALL_CONFIGS, ids=SMALL_CONFIG_IDS
)
def test_fused_cross_entropy_backward_outputs(batch_size, seq_len, d_model, vocab_size):
  """Test fused cross-entropy gradient w.r.t. outputs matches reference."""
  key = jax.random.PRNGKey(123)
  key_out, key_w, key_tgt = jax.random.split(key, 3)

  outputs = jax.random.normal(
    key_out, (batch_size, seq_len, d_model), dtype=jnp.float32
  )
  w_unemb = jax.random.normal(key_w, (vocab_size, d_model), dtype=jnp.float32)
  max_valid_id = vocab_size - 1
  targets = jax.random.randint(key_tgt, (batch_size, seq_len), 0, max_valid_id)

  grad_ref = jax.jit(
    jax.grad(lambda o: ref_cross_entropy(o, w_unemb, targets, max_valid_id))
  )(outputs)
  grad_fused = jax.jit(
    jax.grad(
      lambda o: fused_cross_entropy(
        o, w_unemb, targets, max_valid_id, DEFAULT_BLOCK_SIZE
      )
    )
  )(outputs)

  np.testing.assert_allclose(grad_fused, grad_ref, rtol=FORWARD_RTOL, atol=FORWARD_ATOL)


@pytest.mark.parametrize(
  "batch_size,seq_len,d_model,vocab_size", SMALL_CONFIGS, ids=SMALL_CONFIG_IDS
)
def test_fused_cross_entropy_backward_weights(batch_size, seq_len, d_model, vocab_size):
  """Test fused cross-entropy gradient w.r.t. weights matches reference."""
  key = jax.random.PRNGKey(456)
  key_out, key_w, key_tgt = jax.random.split(key, 3)

  outputs = jax.random.normal(
    key_out, (batch_size, seq_len, d_model), dtype=jnp.float32
  )
  w_unemb = jax.random.normal(key_w, (vocab_size, d_model), dtype=jnp.float32)
  max_valid_id = vocab_size - 1
  targets = jax.random.randint(key_tgt, (batch_size, seq_len), 0, max_valid_id)

  grad_ref = jax.jit(
    jax.grad(lambda w: ref_cross_entropy(outputs, w, targets, max_valid_id), argnums=0)
  )(w_unemb)
  grad_fused = jax.jit(
    jax.grad(
      lambda w: fused_cross_entropy(
        outputs, w, targets, max_valid_id, DEFAULT_BLOCK_SIZE
      ),
      argnums=0,
    )
  )(w_unemb)

  np.testing.assert_allclose(grad_fused, grad_ref, rtol=FORWARD_RTOL, atol=FORWARD_ATOL)


# =============================================================================
# Edge Cases
# =============================================================================


@pytest.mark.parametrize(
  "name,batch_size,seq_len,d_model,vocab_size,max_valid_id,scale",
  [
    pytest.param("single_block", 1, 128, 128, 256, 255, 1.0, id="single_block"),
    pytest.param("large_logits", 2, 128, 128, 512, 511, 5.0, id="large_logits"),
    pytest.param("with_padding", 2, 128, 128, 1024, 900, 1.0, id="with_padding"),
  ],
)
def test_fused_cross_entropy_edge_cases(
  name, batch_size, seq_len, d_model, vocab_size, max_valid_id, scale
):
  """Test edge cases: single block, large logits, vocab masking."""
  key = jax.random.PRNGKey(789)
  key_out, key_w, key_tgt = jax.random.split(key, 3)

  outputs = jax.random.normal(
    key_out, (batch_size, seq_len, d_model), dtype=jnp.float32
  )
  outputs = outputs * scale
  w_unemb = jax.random.normal(key_w, (vocab_size, d_model), dtype=jnp.float32)
  w_unemb = w_unemb * scale
  targets = jax.random.randint(key_tgt, (batch_size, seq_len), 0, max_valid_id)

  loss_ref = jax.jit(partial(ref_cross_entropy, max_valid_id=max_valid_id))(
    outputs, w_unemb, targets
  )
  loss_fused = jax.jit(
    partial(
      fused_cross_entropy, max_valid_id=max_valid_id, block_size=DEFAULT_BLOCK_SIZE
    )
  )(outputs, w_unemb, targets)

  if scale > 1.0:
    assert jnp.isfinite(loss_ref)
    assert jnp.isfinite(loss_fused)

  np.testing.assert_allclose(loss_fused, loss_ref, rtol=FORWARD_RTOL, atol=FORWARD_ATOL)


# =============================================================================
# TPU Tests
# =============================================================================


@requires_tpu
@pytest.mark.parametrize(
  "batch_size,seq_len,d_model,vocab_size", TPU_CONFIGS, ids=TPU_CONFIG_IDS
)
def test_fused_cross_entropy_on_tpu(batch_size, seq_len, d_model, vocab_size):
  """Test kernel on TPU without interpret mode."""
  key = jax.random.PRNGKey(123)
  key_out, key_w, key_tgt = jax.random.split(key, 3)

  outputs = jax.random.normal(
    key_out, (batch_size, seq_len, d_model), dtype=jnp.float32
  )
  w_unemb = jax.random.normal(key_w, (vocab_size, d_model), dtype=jnp.float32)
  max_valid_id = vocab_size - 128
  targets = jax.random.randint(key_tgt, (batch_size, seq_len), 0, max_valid_id)

  loss_ref = jax.jit(partial(ref_cross_entropy, max_valid_id=max_valid_id))(
    outputs, w_unemb, targets
  )
  loss_fused = jax.jit(
    partial(
      fused_cross_entropy,
      max_valid_id=max_valid_id,
      block_size=DEFAULT_BLOCK_SIZE,
      interpret=False,
    )
  )(outputs, w_unemb, targets)

  np.testing.assert_allclose(loss_fused, loss_ref, rtol=FORWARD_RTOL, atol=FORWARD_ATOL)

  # Test gradients
  grad_ref = jax.jit(
    jax.grad(
      lambda o, w: ref_cross_entropy(o, w, targets, max_valid_id), argnums=(0, 1)
    )
  )(outputs, w_unemb)

  grad_fused = jax.jit(
    jax.grad(
      lambda o, w: fused_cross_entropy(
        o, w, targets, max_valid_id, DEFAULT_BLOCK_SIZE, False
      ),
      argnums=(0, 1),
    )
  )(outputs, w_unemb)

  np.testing.assert_allclose(
    grad_fused[0], grad_ref[0], rtol=FORWARD_RTOL, atol=FORWARD_ATOL
  )
  np.testing.assert_allclose(
    grad_fused[1], grad_ref[1], rtol=FORWARD_RTOL, atol=FORWARD_ATOL
  )


# =============================================================================
# Integration Tests - test production losses.py functions
# =============================================================================


@requires_tpu
def test_losses_integration_forward():
  """Test production loss functions match on forward pass."""
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

  unemb = LMHead(w=w_unemb, bias=bias)

  num_tokens = batch_size * seq_len
  shard_mapped__kernel = make_lse_kernel_sharded(
    q_seq_len=num_tokens,
    k_seq_len=vocab_size,
    mesh=mesh,
    data_sharding=config.sharding.data,
    q_seq_shards=num_devices,
    block_size=DEFAULT_BLOCK_SIZE,
    max_valid_id=max_valid_id,
  )

  loss_fn_ref = jax.jit(partial(softmax_cross_entropy, config))
  loss_fn_fused = jax.jit(
    partial(
      fused_softmax_cross_entropy, config, shard_mapped__kernel=shard_mapped__kernel
    )
  )

  with jax.set_mesh(mesh):
    loss_ref = loss_fn_ref(unemb, outputs, targets)
    loss_fused = loss_fn_fused(unemb, outputs, targets)

  np.testing.assert_allclose(loss_fused, loss_ref, rtol=FORWARD_RTOL, atol=FORWARD_ATOL)


@requires_tpu
def test_losses_integration_backward():
  """Test production loss functions match on backward pass."""
  num_devices = jax.device_count()
  seq_len = 1024
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

  key = jax.random.PRNGKey(123)
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

  num_tokens = batch_size * seq_len
  shard_mapped__kernel = make_lse_kernel_sharded(
    q_seq_len=num_tokens,
    k_seq_len=vocab_size,
    mesh=mesh,
    data_sharding=config.sharding.data,
    q_seq_shards=num_devices,
    block_size=DEFAULT_BLOCK_SIZE,
    max_valid_id=max_valid_id,
  )

  @jax.jit
  def evaluate_step(outputs, w_unemb, targets):
    def ref_loss(o, w):
      return softmax_cross_entropy(config, LMHead(w=w, bias=bias), o, targets)

    def fused_loss(o, w):
      return fused_softmax_cross_entropy(
        config, LMHead(w=w, bias=bias), o, targets, shard_mapped__kernel
      )

    grad_ref = jax.grad(ref_loss, argnums=(0, 1))(outputs, w_unemb)
    grad_fused = jax.grad(fused_loss, argnums=(0, 1))(outputs, w_unemb)
    return grad_ref, grad_fused

  with jax.set_mesh(mesh):
    grad_ref, grad_fused = evaluate_step(outputs, w_unemb, targets)

  # Allgather sharded gradients before comparison
  grad_ref = jax.tree.map(
    lambda x: multihost_utils.process_allgather(x, tiled=True), grad_ref
  )
  grad_fused = jax.tree.map(
    lambda x: multihost_utils.process_allgather(x, tiled=True), grad_fused
  )

  np.testing.assert_allclose(
    grad_fused[0], grad_ref[0], rtol=FORWARD_RTOL, atol=FORWARD_ATOL
  )
  np.testing.assert_allclose(
    grad_fused[1], grad_ref[1], rtol=FORWARD_RTOL, atol=FORWARD_ATOL
  )
