from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import multihost_utils
from jax.experimental.pallas.ops.tpu.splash_attention.splash_attention_mask import (
  MultiHeadMask,
)

from bmgpt.config import (
  Config,
  DatasetConfig,
  DatasetName,
  ModelConfig,
  ShardingConfig,
  TransformerType,
)
from bmgpt.kernels.lse_kernel import BlockSizes, make_lse_kernel
from bmgpt.kernels.lse_mask import VocabMask
from bmgpt.losses import fused_softmax_cross_entropy, softmax_cross_entropy
from bmgpt.model import LMHead
from bmgpt.splash_helpers import make_lse_kernel_sharded

# block_size must be >= 128 (TPU NUM_LANES requirement) & divide (batch_size * seq_len)
CONFIGS = [
  (1, 128, 64, 256),
  (2, 128, 128, 512),
  (4, 128, 128, 1024),
  (4, 256, 256, 2048),
  (2, 256, 128, 4096),
]
CONFIG_IDS = [f"B{b}_S{s}_D{d}_V{v}" for b, s, d, v in CONFIGS]

GLOBAL_RTOL = 1e-6
GLOBAL_ATOL = 1e-6


def ref_cross_entropy(
  outputs: jax.Array,
  w_unemb: jax.Array,
  targets: jax.Array,
  max_valid_id: int | None = None,
) -> jax.Array:
  """ref cross-entropy that materializes full logits matrix."""
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

  loss_ref = jax.jit(partial(ref_cross_entropy, max_valid_id=max_valid_id))(
    outputs, w_unemb, targets
  )
  loss_fused = jax.jit(
    partial(fused_cross_entropy, max_valid_id=max_valid_id, block_size=128)
  )(outputs, w_unemb, targets)

  np.testing.assert_allclose(loss_fused, loss_ref, rtol=GLOBAL_RTOL, atol=GLOBAL_ATOL)


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

  grad_ref = jax.jit(
    jax.grad(lambda o: ref_cross_entropy(o, w_unemb, targets, max_valid_id))
  )(outputs)
  grad_fused = jax.jit(
    jax.grad(lambda o: fused_cross_entropy(o, w_unemb, targets, max_valid_id, 128))
  )(outputs)

  np.testing.assert_allclose(grad_fused, grad_ref, rtol=GLOBAL_RTOL, atol=GLOBAL_ATOL)


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

  grad_ref = jax.jit(
    jax.grad(lambda w: ref_cross_entropy(outputs, w, targets, max_valid_id), argnums=0)
  )(w_unemb)
  grad_fused = jax.jit(
    jax.grad(
      lambda w: fused_cross_entropy(outputs, w, targets, max_valid_id, 128), argnums=0
    )
  )(w_unemb)

  np.testing.assert_allclose(grad_fused, grad_ref, rtol=GLOBAL_RTOL, atol=GLOBAL_ATOL)


def test_fused_cross_entropy_single_block():
  """Edge case: single block computation."""
  key = jax.random.PRNGKey(789)
  key_out, key_w, key_tgt = jax.random.split(key, 3)

  outputs = jax.random.normal(key_out, (1, 128, 128), dtype=jnp.float32)
  w_unemb = jax.random.normal(key_w, (256, 128), dtype=jnp.float32)
  targets = jax.random.randint(key_tgt, (1, 128), 0, 255)
  max_valid_id = 255

  loss_ref = jax.jit(partial(ref_cross_entropy, max_valid_id=max_valid_id))(
    outputs, w_unemb, targets
  )
  loss_fused = jax.jit(
    partial(fused_cross_entropy, max_valid_id=max_valid_id, block_size=128)
  )(outputs, w_unemb, targets)

  np.testing.assert_allclose(loss_fused, loss_ref, rtol=GLOBAL_RTOL, atol=GLOBAL_ATOL)


def test_fused_cross_entropy_large_logits():
  """Edge case: numerical stability with large logit values."""
  key = jax.random.PRNGKey(999)
  key_out, key_w, key_tgt = jax.random.split(key, 3)

  outputs = jax.random.normal(key_out, (2, 128, 128), dtype=jnp.float32) * 5.0
  w_unemb = jax.random.normal(key_w, (512, 128), dtype=jnp.float32) * 5.0
  max_valid_id = 511
  targets = jax.random.randint(key_tgt, (2, 128), 0, max_valid_id)

  loss_ref = jax.jit(partial(ref_cross_entropy, max_valid_id=max_valid_id))(
    outputs, w_unemb, targets
  )
  loss_fused = jax.jit(
    partial(fused_cross_entropy, max_valid_id=max_valid_id, block_size=128)
  )(outputs, w_unemb, targets)

  assert jnp.isfinite(loss_ref)
  assert jnp.isfinite(loss_fused)

  np.testing.assert_allclose(loss_fused, loss_ref, rtol=GLOBAL_RTOL, atol=GLOBAL_ATOL)


def test_fused_cross_entropy_with_padding():
  """Edge case: vocab masking with max_valid_id < vocab_size."""
  key = jax.random.PRNGKey(321)
  key_out, key_w, key_tgt = jax.random.split(key, 3)

  vocab_size = 1024
  max_valid_id = 900

  outputs = jax.random.normal(key_out, (2, 128, 128), dtype=jnp.float32)
  w_unemb = jax.random.normal(key_w, (vocab_size, 128), dtype=jnp.float32)
  targets = jax.random.randint(key_tgt, (2, 128), 0, max_valid_id)

  loss_ref = jax.jit(partial(ref_cross_entropy, max_valid_id=max_valid_id))(
    outputs, w_unemb, targets
  )
  loss_fused = jax.jit(
    partial(fused_cross_entropy, max_valid_id=max_valid_id, block_size=128)
  )(outputs, w_unemb, targets)

  np.testing.assert_allclose(loss_fused, loss_ref, rtol=GLOBAL_RTOL, atol=GLOBAL_ATOL)


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
      fused_cross_entropy, max_valid_id=max_valid_id, block_size=128, interpret=False
    )
  )(outputs, w_unemb, targets)

  np.testing.assert_allclose(loss_fused, loss_ref, rtol=GLOBAL_RTOL, atol=GLOBAL_ATOL)

  grad_ref = jax.jit(
    jax.grad(
      lambda o, w: ref_cross_entropy(o, w, targets, max_valid_id), argnums=(0, 1)
    )
  )(outputs, w_unemb)

  grad_fused = jax.jit(
    jax.grad(
      lambda o, w: fused_cross_entropy(o, w, targets, max_valid_id, 128, False),
      argnums=(0, 1),
    )
  )(outputs, w_unemb)

  np.testing.assert_allclose(grad_fused[0], grad_ref[0], rtol=GLOBAL_RTOL, atol=GLOBAL_ATOL)
  np.testing.assert_allclose(grad_fused[1], grad_ref[1], rtol=GLOBAL_RTOL, atol=GLOBAL_ATOL)


# =============================================================================
# Integration Tests - test production losses.py functions
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
  wunemb_sharding: list[str | None] | None = None,
) -> Config:
  """Create a Config for testing with FSDP-style sharding."""
  # Default wunemb sharding matches FSDP config: [None, "fsdp"] for (V, D)
  if wunemb_sharding is None:
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


@pytest.mark.skipif(
  not jax.devices()[0].platform == "tpu",
  reason="Integration tests use production kernels (no interpret mode)",
)
def test_losses_integration_forward():
  """Test production loss functions match on forward pass."""
  num_devices = jax.device_count()
  block_size = 128
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

  # Pre-shard inputs matching train.py data flow
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
    block_size=block_size,
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

  print(f"\n[DEBUG] loss_ref={float(loss_ref):.8f}, loss_fused={float(loss_fused):.8f}")
  print(
    f"[DEBUG] diff={float(loss_fused - loss_ref):.8e}, "
    f"rel_diff={float((loss_fused - loss_ref) / loss_ref):.8e}"
  )

  np.testing.assert_allclose(loss_fused, loss_ref, rtol=GLOBAL_RTOL, atol=GLOBAL_ATOL)


@pytest.mark.skipif(
  not jax.devices()[0].platform == "tpu",
  reason="Integration tests use production kernels (no interpret mode)",
)
def test_losses_integration_backward():
  """Test production loss functions match on backward pass."""
  num_devices = jax.device_count()
  block_size = 128
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

  # Pre-shard inputs matching train.py data flow
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
    block_size=block_size,
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

  print(f"\n[DEBUG] grad_ref[0] norm={float(jnp.linalg.norm(grad_ref[0])):.6f}")
  print(f"[DEBUG] grad_fused[0] norm={float(jnp.linalg.norm(grad_fused[0])):.6f}")
  print(f"[DEBUG] grad_ref[1] norm={float(jnp.linalg.norm(grad_ref[1])):.6f}")
  print(f"[DEBUG] grad_fused[1] norm={float(jnp.linalg.norm(grad_fused[1])):.6f}")

  np.testing.assert_allclose(grad_fused[0], grad_ref[0], rtol=GLOBAL_RTOL, atol=GLOBAL_ATOL)
  np.testing.assert_allclose(grad_fused[1], grad_ref[1], rtol=GLOBAL_RTOL, atol=GLOBAL_ATOL)
