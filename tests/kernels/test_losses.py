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

from bmgpt.config import (
  Config,
  DatasetConfig,
  ModelConfig,
  ShardingConfig,
  DatasetName,
  TransformerType,
)
from bmgpt.losses import softmax_cross_entropy, fused_softmax_cross_entropy
from bmgpt.model import LMHead
from bmgpt.splash_helpers import make_lse_kernel_sharded

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


@pytest.mark.skipif(
  jax.devices()[0].platform == "tpu",
  reason="Hypothesis tests not designed for multi-host TPU",
)
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
  derandomize=True,  # Same examples on all TPU workers
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


@pytest.mark.skipif(
  jax.devices()[0].platform == "tpu",
  reason="Hypothesis tests not designed for multi-host TPU",
)
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
  derandomize=True,  # Same examples on all TPU workers
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


TPU_CONFIGS = [
  (8, 128, 128, 2048),
  (16, 256, 256, 4096),
  (1, 256, 64, 512),
  (2, 128, 256, 1024),
  (4, 256, 128, 2048),
  (8, 256, 64, 4096),
  (4, 128, 256, 512),
  (2, 256, 128, 1024),
]
TPU_CONFIG_IDS = [f"B{b}_S{s}_D{d}_V{v}" for b, s, d, v in TPU_CONFIGS]


@pytest.mark.skipif(
  not jax.devices()[0].platform == "tpu",
  reason="TPU test - skip on non-TPU platforms",
)
@pytest.mark.parametrize(
  "batch_size,seq_len,d_model,vocab_size",
  TPU_CONFIGS,
  ids=TPU_CONFIG_IDS,
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

  per_token_unembs = w_unemb_replicated.at[targets_flat].get(out_sharding=data_sharding)
  label_logits = jnp.sum(outputs_flat * per_token_unembs, axis=-1)

  return (lse - label_logits).mean()


@pytest.mark.skipif(
  not jax.devices()[0].platform == "tpu",
  reason="TPU test - skip on non-TPU platforms",
)
def test_fused_cross_entropy_sharded():
  num_devices = len(jax.devices())
  block_size = 128
  # num_tokens_per_shard must be >= block_size, so num_tokens >= block_size * num_devices
  min_tokens = block_size * num_devices
  seq_len = 256
  batch_size = max(8, (min_tokens + seq_len - 1) // seq_len)
  d_model = 128
  vocab_size = 2048
  max_valid_id = vocab_size - 128

  key = jax.random.PRNGKey(42)
  key_out, key_w, key_tgt = jax.random.split(key, 3)

  outputs = jax.random.normal(
    key_out, (batch_size, seq_len, d_model), dtype=jnp.float32
  )
  w_unemb = jax.random.normal(key_w, (vocab_size, d_model), dtype=jnp.float32)
  targets = jax.random.randint(key_tgt, (batch_size, seq_len), 0, max_valid_id)

  loss_ref = naive_cross_entropy(outputs, w_unemb, targets, max_valid_id)

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
  num_devices = len(jax.devices())
  block_size = 128
  # num_tokens_per_shard must be >= block_size, so num_tokens >= block_size * num_devices
  min_tokens = block_size * num_devices
  seq_len = 256
  batch_size = max(4, (min_tokens + seq_len - 1) // seq_len)
  d_model = 128
  vocab_size = 1024
  max_valid_id = vocab_size - 128

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
      wunemb=[None, None],
      data=data_sharding,
      mlp_hidden=[None],
      res_stream=[None],
      att_qkv=[None, None, None, None],
    ),
  )


@pytest.mark.skipif(
  not jax.devices()[0].platform == "tpu",
  reason="Integration tests use production kernels (no interpret mode)",
)
def test_losses_integration_forward():
  batch_size, seq_len, d_model, vocab_size = 2, 256, 128, 512
  max_valid_id = vocab_size - 128

  config = make_test_config(
    batch_size,
    seq_len,
    d_model,
    vocab_size,
    max_valid_id,
    data_sharding=[None],
    mesh_shape=[1],
    mesh_axis_names=["data"],
  )
  mesh = jax.make_mesh([1], ["data"], (jax.sharding.AxisType.Explicit,))

  key = jax.random.PRNGKey(42)
  key_out, key_w, key_tgt = jax.random.split(key, 3)

  outputs = jax.random.normal(
    key_out, (batch_size, seq_len, d_model), dtype=jnp.float32
  )
  w_unemb = jax.random.normal(key_w, (vocab_size, d_model), dtype=jnp.float32)
  bias = jnp.zeros(vocab_size, dtype=jnp.float32)
  targets = jax.random.randint(key_tgt, (batch_size, seq_len), 0, max_valid_id)

  unemb = LMHead(w=w_unemb, bias=bias)

  num_tokens = batch_size * seq_len
  shard_mapped__kernel = make_lse_kernel_sharded(
    q_seq_len=num_tokens,
    k_seq_len=vocab_size,
    mesh=mesh,
    data_sharding=config.sharding.data,
    q_seq_shards=1,
    block_size=128,
    max_valid_id=max_valid_id,
  )

  with jax.set_mesh(mesh):
    loss_naive = softmax_cross_entropy(config, unemb, outputs, targets)
    loss_fused = fused_softmax_cross_entropy(
      config, unemb, outputs, targets, shard_mapped__kernel
    )

  np.testing.assert_allclose(loss_fused, loss_naive, rtol=1e-4, atol=1e-4)


@pytest.mark.skipif(
  not jax.devices()[0].platform == "tpu",
  reason="Integration tests use production kernels (no interpret mode)",
)
def test_losses_integration_backward():
  batch_size, seq_len, d_model, vocab_size = 2, 256, 128, 512
  max_valid_id = vocab_size - 128

  config = make_test_config(
    batch_size,
    seq_len,
    d_model,
    vocab_size,
    max_valid_id,
    data_sharding=[None],
    mesh_shape=[1],
    mesh_axis_names=["data"],
  )
  mesh = jax.make_mesh([1], ["data"], (jax.sharding.AxisType.Explicit,))

  key = jax.random.PRNGKey(123)
  key_out, key_w, key_tgt = jax.random.split(key, 3)

  outputs = jax.random.normal(
    key_out, (batch_size, seq_len, d_model), dtype=jnp.float32
  )
  w_unemb = jax.random.normal(key_w, (vocab_size, d_model), dtype=jnp.float32)
  bias = jnp.zeros(vocab_size, dtype=jnp.float32)
  targets = jax.random.randint(key_tgt, (batch_size, seq_len), 0, max_valid_id)

  unemb = LMHead(w=w_unemb, bias=bias)

  num_tokens = batch_size * seq_len
  shard_mapped__kernel = make_lse_kernel_sharded(
    q_seq_len=num_tokens,
    k_seq_len=vocab_size,
    mesh=mesh,
    data_sharding=config.sharding.data,
    q_seq_shards=1,
    block_size=128,
    max_valid_id=max_valid_id,
  )

  def naive_loss(o, w):
    return softmax_cross_entropy(config, LMHead(w=w, bias=bias), o, targets)

  def fused_loss(o, w):
    return fused_softmax_cross_entropy(
      config, LMHead(w=w, bias=bias), o, targets, shard_mapped__kernel
    )

  with jax.set_mesh(mesh):
    grad_naive = jax.grad(naive_loss, argnums=(0, 1))(outputs, w_unemb)
    grad_fused = jax.grad(fused_loss, argnums=(0, 1))(outputs, w_unemb)

  np.testing.assert_allclose(grad_fused[0], grad_naive[0], rtol=2e-2, atol=2e-2)
  np.testing.assert_allclose(grad_fused[1], grad_naive[1], rtol=2e-2, atol=2e-2)


INTEGRATION_CONFIGS = [
  (2, 256, 128, 512),
  (4, 128, 64, 1024),
  (1, 256, 256, 256),
  (2, 128, 128, 2048),
]
INTEGRATION_CONFIG_IDS = [f"B{b}_S{s}_D{d}_V{v}" for b, s, d, v in INTEGRATION_CONFIGS]


@pytest.mark.skipif(
  not jax.devices()[0].platform == "tpu",
  reason="Integration tests use production kernels (no interpret mode)",
)
@pytest.mark.parametrize(
  "batch_size,seq_len,d_model,vocab_size",
  INTEGRATION_CONFIGS,
  ids=INTEGRATION_CONFIG_IDS,
)
def test_losses_integration_parametrized(batch_size, seq_len, d_model, vocab_size):
  max_valid_id = vocab_size - 128

  config = make_test_config(
    batch_size,
    seq_len,
    d_model,
    vocab_size,
    max_valid_id,
    data_sharding=[None],
    mesh_shape=[1],
    mesh_axis_names=["data"],
  )
  mesh = jax.make_mesh([1], ["data"], (jax.sharding.AxisType.Explicit,))

  key = jax.random.PRNGKey(42)
  key_out, key_w, key_tgt = jax.random.split(key, 3)

  outputs = jax.random.normal(
    key_out, (batch_size, seq_len, d_model), dtype=jnp.float32
  )
  w_unemb = jax.random.normal(key_w, (vocab_size, d_model), dtype=jnp.float32)
  bias = jnp.zeros(vocab_size, dtype=jnp.float32)
  targets = jax.random.randint(key_tgt, (batch_size, seq_len), 0, max_valid_id)

  unemb = LMHead(w=w_unemb, bias=bias)

  num_tokens = batch_size * seq_len
  shard_mapped__kernel = make_lse_kernel_sharded(
    q_seq_len=num_tokens,
    k_seq_len=vocab_size,
    mesh=mesh,
    data_sharding=config.sharding.data,
    q_seq_shards=1,
    block_size=128,
    max_valid_id=max_valid_id,
  )

  with jax.set_mesh(mesh):
    loss_naive = softmax_cross_entropy(config, unemb, outputs, targets)
    loss_fused = fused_softmax_cross_entropy(
      config, unemb, outputs, targets, shard_mapped__kernel
    )

  np.testing.assert_allclose(loss_fused, loss_naive, rtol=1e-4, atol=1e-4)


@pytest.mark.skipif(
  not jax.devices()[0].platform == "tpu",
  reason="TPU test - skip on non-TPU platforms",
)
def test_losses_integration_tpu_sharded():
  num_devices = len(jax.devices())
  block_size = 128
  min_tokens = block_size * num_devices
  seq_len = 256
  batch_size = max(8, (min_tokens + seq_len - 1) // seq_len)
  d_model = 128
  vocab_size = 2048
  max_valid_id = vocab_size - 128

  config = make_test_config(
    batch_size,
    seq_len,
    d_model,
    vocab_size,
    max_valid_id,
    data_sharding=["data"],
    mesh_shape=[num_devices],
    mesh_axis_names=["data"],
  )
  mesh = jax.make_mesh([num_devices], ["data"], (jax.sharding.AxisType.Explicit,))

  key = jax.random.PRNGKey(42)
  key_out, key_w, key_tgt = jax.random.split(key, 3)

  outputs = jax.random.normal(
    key_out, (batch_size, seq_len, d_model), dtype=jnp.float32
  )
  w_unemb = jax.random.normal(key_w, (vocab_size, d_model), dtype=jnp.float32)
  bias = jnp.zeros(vocab_size, dtype=jnp.float32)
  targets = jax.random.randint(key_tgt, (batch_size, seq_len), 0, max_valid_id)

  unemb = LMHead(w=w_unemb, bias=bias)

  num_tokens = batch_size * seq_len
  shard_mapped__kernel = make_lse_kernel_sharded(
    q_seq_len=num_tokens,
    k_seq_len=vocab_size,
    mesh=mesh,
    data_sharding=config.sharding.data,
    q_seq_shards=num_devices,
    block_size=block_size,
    max_valid_id=max_valid_id,
  )

  with jax.set_mesh(mesh):
    loss_naive = softmax_cross_entropy(config, unemb, outputs, targets)
    loss_fused = fused_softmax_cross_entropy(
      config, unemb, outputs, targets, shard_mapped__kernel
    )

  np.testing.assert_allclose(loss_fused, loss_naive, rtol=1e-4, atol=1e-4)
