import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import given, settings, strategies as st, HealthCheck, Phase
from jax import shard_map
from jax.experimental import multihost_utils

from bmgpt.kernels.lse_kernel import BlockSizes, make_lse_kernel
from bmgpt.kernels.lse_mask import VocabMask
from jax.experimental.pallas.ops.tpu.splash_attention.splash_attention_mask import (
  MultiHeadMask,
)

# block_size must be >= 128 (TPU NUM_LANES requirement) and divide (batch_size * seq_len)
CONFIGS = [
  (1, 128, 64, 256),
  (2, 128, 128, 512),
  (4, 128, 128, 1024),
  (4, 256, 256, 2048),
  (2, 256, 128, 4096),
]
CONFIG_IDS = [f"B{b}_S{s}_D{d}_V{v}" for b, s, d, v in CONFIGS]


def naive_cross_entropy(
  outputs: jax.Array,
  w_unemb: jax.Array,
  targets: jax.Array,
  max_valid_id: int | None = None,
) -> jax.Array:
  """Naive cross-entropy that materializes full logits matrix."""
  logits = jnp.einsum("bsd,vd->bsv", outputs, w_unemb)

  if max_valid_id is not None:
    vocab_ids = jnp.arange(logits.shape[-1])
    mask = vocab_ids <= max_valid_id
    logits = jnp.where(mask[None, None, :], logits, -1e10)

  label_logits = jnp.take_along_axis(logits, targets[..., None], axis=-1).squeeze(-1)
  lse = jax.nn.logsumexp(logits, axis=-1)
  return (lse - label_logits).mean()


def fused_cross_entropy(
  outputs: jax.Array,
  w_unemb: jax.Array,
  targets: jax.Array,
  max_valid_id: int,
  block_size: int = 128,
  interpret: bool = True,
) -> jax.Array:
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


@pytest.mark.parametrize(
  "batch_size,seq_len,d_model,vocab_size", CONFIGS, ids=CONFIG_IDS
)
def test_fused_cross_entropy_forward(batch_size, seq_len, d_model, vocab_size):
  key = jax.random.PRNGKey(42)
  key_out, key_w, key_tgt = jax.random.split(key, 3)

  outputs = jax.random.normal(
    key_out, (batch_size, seq_len, d_model), dtype=jnp.float32
  )
  w_unemb = jax.random.normal(key_w, (vocab_size, d_model), dtype=jnp.float32)
  max_valid_id = vocab_size - 1
  targets = jax.random.randint(key_tgt, (batch_size, seq_len), 0, max_valid_id)

  loss_naive = naive_cross_entropy(outputs, w_unemb, targets, max_valid_id)
  loss_fused = fused_cross_entropy(outputs, w_unemb, targets, max_valid_id, 128)

  np.testing.assert_allclose(loss_fused, loss_naive, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize(
  "batch_size,seq_len,d_model,vocab_size", CONFIGS, ids=CONFIG_IDS
)
def test_fused_cross_entropy_backward_outputs(batch_size, seq_len, d_model, vocab_size):
  key = jax.random.PRNGKey(123)
  key_out, key_w, key_tgt = jax.random.split(key, 3)

  outputs = jax.random.normal(
    key_out, (batch_size, seq_len, d_model), dtype=jnp.float32
  )
  w_unemb = jax.random.normal(key_w, (vocab_size, d_model), dtype=jnp.float32)
  max_valid_id = vocab_size - 1
  targets = jax.random.randint(key_tgt, (batch_size, seq_len), 0, max_valid_id)

  grad_naive = jax.grad(
    lambda o: naive_cross_entropy(o, w_unemb, targets, max_valid_id)
  )(outputs)
  grad_fused = jax.grad(
    lambda o: fused_cross_entropy(o, w_unemb, targets, max_valid_id, 128)
  )(outputs)

  np.testing.assert_allclose(grad_fused, grad_naive, rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize(
  "batch_size,seq_len,d_model,vocab_size", CONFIGS, ids=CONFIG_IDS
)
def test_fused_cross_entropy_backward_weights(batch_size, seq_len, d_model, vocab_size):
  key = jax.random.PRNGKey(456)
  key_out, key_w, key_tgt = jax.random.split(key, 3)

  outputs = jax.random.normal(
    key_out, (batch_size, seq_len, d_model), dtype=jnp.float32
  )
  w_unemb = jax.random.normal(key_w, (vocab_size, d_model), dtype=jnp.float32)
  max_valid_id = vocab_size - 1
  targets = jax.random.randint(key_tgt, (batch_size, seq_len), 0, max_valid_id)

  grad_naive = jax.grad(
    lambda w: naive_cross_entropy(outputs, w, targets, max_valid_id), argnums=0
  )(w_unemb)
  grad_fused = jax.grad(
    lambda w: fused_cross_entropy(outputs, w, targets, max_valid_id, 128), argnums=0
  )(w_unemb)

  np.testing.assert_allclose(grad_fused, grad_naive, rtol=2e-2, atol=2e-2)


batch_sizes = st.sampled_from([1, 2, 4])
seq_lens = st.sampled_from([128, 256])
d_models = st.sampled_from([64, 128, 256])
vocab_sizes = st.sampled_from([256, 512, 1024, 2048])
seeds = st.integers(min_value=0, max_value=2**31 - 1)


@given(
  batch_size=batch_sizes,
  seq_len=seq_lens,
  d_model=d_models,
  vocab_size=vocab_sizes,
  seed=seeds,
)
@settings(
  max_examples=20,
  deadline=None,
  suppress_health_check=[HealthCheck.too_slow],
  phases=[Phase.generate, Phase.target, Phase.shrink],
)
def test_fused_cross_entropy_hypothesis_forward(
  batch_size: int, seq_len: int, d_model: int, vocab_size: int, seed: int
):
  key = jax.random.PRNGKey(seed)
  key_out, key_w, key_tgt = jax.random.split(key, 3)

  outputs = jax.random.normal(
    key_out, (batch_size, seq_len, d_model), dtype=jnp.float32
  )
  w_unemb = jax.random.normal(key_w, (vocab_size, d_model), dtype=jnp.float32)
  max_valid_id = vocab_size - 1
  targets = jax.random.randint(key_tgt, (batch_size, seq_len), 0, max_valid_id)

  loss_naive = naive_cross_entropy(outputs, w_unemb, targets, max_valid_id)
  loss_fused = fused_cross_entropy(outputs, w_unemb, targets, max_valid_id, 128)

  np.testing.assert_allclose(loss_fused, loss_naive, rtol=1e-4, atol=1e-4)


@given(
  batch_size=batch_sizes,
  seq_len=seq_lens,
  d_model=d_models,
  vocab_size=vocab_sizes,
  seed=seeds,
)
@settings(
  max_examples=10,
  deadline=None,
  suppress_health_check=[HealthCheck.too_slow],
  phases=[Phase.generate, Phase.target, Phase.shrink],
)
def test_fused_cross_entropy_hypothesis_backward(
  batch_size: int, seq_len: int, d_model: int, vocab_size: int, seed: int
):
  key = jax.random.PRNGKey(seed)
  key_out, key_w, key_tgt = jax.random.split(key, 3)

  outputs = jax.random.normal(
    key_out, (batch_size, seq_len, d_model), dtype=jnp.float32
  )
  w_unemb = jax.random.normal(key_w, (vocab_size, d_model), dtype=jnp.float32)
  max_valid_id = vocab_size - 1
  targets = jax.random.randint(key_tgt, (batch_size, seq_len), 0, max_valid_id)

  grad_naive = jax.grad(
    lambda o, w: naive_cross_entropy(o, w, targets, max_valid_id), argnums=(0, 1)
  )(outputs, w_unemb)

  grad_fused = jax.grad(
    lambda o, w: fused_cross_entropy(o, w, targets, max_valid_id, 128), argnums=(0, 1)
  )(outputs, w_unemb)

  np.testing.assert_allclose(grad_fused[0], grad_naive[0], rtol=2e-2, atol=2e-2)
  np.testing.assert_allclose(grad_fused[1], grad_naive[1], rtol=2e-2, atol=2e-2)


def test_fused_cross_entropy_single_block():
  key = jax.random.PRNGKey(789)
  key_out, key_w, key_tgt = jax.random.split(key, 3)

  outputs = jax.random.normal(key_out, (1, 128, 128), dtype=jnp.float32)
  w_unemb = jax.random.normal(key_w, (256, 128), dtype=jnp.float32)
  targets = jax.random.randint(key_tgt, (1, 128), 0, 255)
  max_valid_id = 255

  loss_naive = naive_cross_entropy(outputs, w_unemb, targets, max_valid_id)
  loss_fused = fused_cross_entropy(outputs, w_unemb, targets, max_valid_id, 128)

  np.testing.assert_allclose(loss_fused, loss_naive, rtol=1e-4, atol=1e-4)


def test_fused_cross_entropy_large_logits():
  key = jax.random.PRNGKey(999)
  key_out, key_w, key_tgt = jax.random.split(key, 3)

  outputs = jax.random.normal(key_out, (2, 128, 128), dtype=jnp.float32) * 5.0
  w_unemb = jax.random.normal(key_w, (512, 128), dtype=jnp.float32) * 5.0
  max_valid_id = 511
  targets = jax.random.randint(key_tgt, (2, 128), 0, max_valid_id)

  loss_naive = naive_cross_entropy(outputs, w_unemb, targets, max_valid_id)
  loss_fused = fused_cross_entropy(outputs, w_unemb, targets, max_valid_id, 128)

  assert jnp.isfinite(loss_naive)
  assert jnp.isfinite(loss_fused)

  np.testing.assert_allclose(loss_fused, loss_naive, rtol=1e-3, atol=1e-3)


def test_fused_cross_entropy_with_padding():
  key = jax.random.PRNGKey(321)
  key_out, key_w, key_tgt = jax.random.split(key, 3)

  vocab_size = 1024
  max_valid_id = 900

  outputs = jax.random.normal(key_out, (2, 128, 128), dtype=jnp.float32)
  w_unemb = jax.random.normal(key_w, (vocab_size, 128), dtype=jnp.float32)
  targets = jax.random.randint(key_tgt, (2, 128), 0, max_valid_id)

  loss_naive = naive_cross_entropy(outputs, w_unemb, targets, max_valid_id)
  loss_fused = fused_cross_entropy(outputs, w_unemb, targets, max_valid_id, 128)

  np.testing.assert_allclose(loss_fused, loss_naive, rtol=1e-4, atol=1e-4)


@pytest.mark.skipif(
  not jax.devices()[0].platform == "tpu",
  reason="TPU test - skip on non-TPU platforms",
)
@pytest.mark.parametrize(
  "batch_size,seq_len,d_model,vocab_size",
  [(8, 128, 128, 2048), (16, 256, 256, 4096)],
  ids=["B8_S128_D128_V2048", "B16_S256_D256_V4096"],
)
def test_fused_cross_entropy_on_tpu(batch_size, seq_len, d_model, vocab_size):
  key = jax.random.PRNGKey(123)
  key_out, key_w, key_tgt = jax.random.split(key, 3)

  outputs = jax.random.normal(
    key_out, (batch_size, seq_len, d_model), dtype=jnp.float32
  )
  w_unemb = jax.random.normal(key_w, (vocab_size, d_model), dtype=jnp.float32)
  max_valid_id = vocab_size - 128
  targets = jax.random.randint(key_tgt, (batch_size, seq_len), 0, max_valid_id)

  loss_naive = naive_cross_entropy(outputs, w_unemb, targets, max_valid_id)
  loss_fused = fused_cross_entropy(outputs, w_unemb, targets, max_valid_id, 128, False)

  np.testing.assert_allclose(loss_fused, loss_naive, rtol=1e-4, atol=1e-4)

  grad_naive = jax.grad(
    lambda o, w: naive_cross_entropy(o, w, targets, max_valid_id), argnums=(0, 1)
  )(outputs, w_unemb)

  grad_fused = jax.grad(
    lambda o, w: fused_cross_entropy(o, w, targets, max_valid_id, 128, False),
    argnums=(0, 1),
  )(outputs, w_unemb)

  np.testing.assert_allclose(grad_fused[0], grad_naive[0], rtol=2e-2, atol=2e-2)
  np.testing.assert_allclose(grad_fused[1], grad_naive[1], rtol=2e-2, atol=2e-2)


def fused_cross_entropy_sharded(
  outputs: jax.Array,
  w_unemb: jax.Array,
  targets: jax.Array,
  max_valid_id: int,
  mesh: jax.sharding.Mesh,
  block_size: int = 128,
  interpret: bool = True,
) -> jax.Array:
  b, s = outputs.shape[:2]
  vocab_size = w_unemb.shape[0]
  num_tokens = b * s

  data_sharding = jax.NamedSharding(mesh, jax.P("data"))
  replicated = jax.NamedSharding(mesh, jax.P())

  outputs_flat = outputs.reshape(num_tokens, -1)
  outputs_flat = jax.device_put(outputs_flat, data_sharding)
  targets_flat = targets.ravel()
  targets_flat = jax.device_put(targets_flat, data_sharding)

  w_unemb_replicated = jax.device_put(w_unemb, replicated)

  q = outputs_flat[None]
  k = w_unemb_replicated[None]

  num_devices = len(jax.devices())
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
    q_seq_shards=num_devices,
    interpret=interpret,
  )

  q_spec = jax.P(None, "data", None)
  k_spec = jax.P(None, None, None)
  lse_spec = jax.P(None, "data")
  kernel_sharding = jax.NamedSharding(mesh, jax.P(None, "data"))
  kernel_spec = kernel.manual_sharding_spec(kernel_sharding)

  @functools.partial(
    shard_map,
    mesh=mesh,
    in_specs=(kernel_spec, q_spec, k_spec),
    out_specs=lse_spec,
    check_vma=False,
  )
  def sharded_kernel(kernel, q, k):
    return kernel(q, k)

  lse = sharded_kernel(kernel, q, k).squeeze(0)

  per_token_unembs = w_unemb_replicated[targets_flat]
  per_token_unembs = jax.device_put(per_token_unembs, data_sharding)
  label_logits = jnp.sum(outputs_flat * per_token_unembs, axis=-1)

  return (lse - label_logits).mean()


@pytest.mark.skipif(
  not jax.devices()[0].platform == "tpu",
  reason="TPU test - skip on non-TPU platforms",
)
def test_fused_cross_entropy_sharded():
  batch_size = 8
  seq_len = 256
  d_model = 128
  vocab_size = 2048
  max_valid_id = vocab_size - 128
  block_size = 128

  key = jax.random.PRNGKey(42)
  key_out, key_w, key_tgt = jax.random.split(key, 3)

  outputs = jax.random.normal(
    key_out, (batch_size, seq_len, d_model), dtype=jnp.float32
  )
  w_unemb = jax.random.normal(key_w, (vocab_size, d_model), dtype=jnp.float32)
  targets = jax.random.randint(key_tgt, (batch_size, seq_len), 0, max_valid_id)

  loss_ref = naive_cross_entropy(outputs, w_unemb, targets, max_valid_id)

  num_devices = len(jax.devices())
  mesh = jax.make_mesh((num_devices,), ("data",), (jax.sharding.AxisType.Explicit,))

  with jax.set_mesh(mesh):
    loss_sharded = fused_cross_entropy_sharded(
      outputs, w_unemb, targets, max_valid_id, mesh, block_size, interpret=False
    )

  np.testing.assert_allclose(loss_sharded, loss_ref, rtol=1e-4, atol=1e-4)


@pytest.mark.skipif(
  not jax.devices()[0].platform == "tpu",
  reason="TPU test - skip on non-TPU platforms",
)
def test_fused_cross_entropy_sharded_backward():
  batch_size = 4
  seq_len = 256
  d_model = 128
  vocab_size = 1024
  max_valid_id = vocab_size - 128
  block_size = 128

  key = jax.random.PRNGKey(123)
  key_out, key_w, key_tgt = jax.random.split(key, 3)

  outputs = jax.random.normal(
    key_out, (batch_size, seq_len, d_model), dtype=jnp.float32
  )
  w_unemb = jax.random.normal(key_w, (vocab_size, d_model), dtype=jnp.float32)
  targets = jax.random.randint(key_tgt, (batch_size, seq_len), 0, max_valid_id)

  grad_ref = jax.grad(
    lambda o, w: naive_cross_entropy(o, w, targets, max_valid_id), argnums=(0, 1)
  )(outputs, w_unemb)

  num_devices = len(jax.devices())
  mesh = jax.make_mesh((num_devices,), ("data",), (jax.sharding.AxisType.Explicit,))

  with jax.set_mesh(mesh):
    grad_sharded = jax.grad(
      lambda o, w: fused_cross_entropy_sharded(
        o, w, targets, max_valid_id, mesh, block_size, interpret=False
      ),
      argnums=(0, 1),
    )(outputs, w_unemb)

  grad_sharded = jax.tree.map(
    lambda x: multihost_utils.process_allgather(x, tiled=True), grad_sharded
  )

  np.testing.assert_allclose(grad_sharded[0], grad_ref[0], rtol=2e-2, atol=2e-2)
  np.testing.assert_allclose(grad_sharded[1], grad_ref[1], rtol=2e-2, atol=2e-2)
