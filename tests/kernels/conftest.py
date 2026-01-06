"""Shared fixtures and constants for kernel tests."""

import jax
import jax.numpy as jnp
import pytest

from bmgpt.config import (
  Config,
  DatasetConfig,
  DatasetName,
  ModelConfig,
  ShardingConfig,
  TransformerType,
)

# =============================================================================
# Constants
# =============================================================================

# Tolerances
FORWARD_RTOL = 1e-4
FORWARD_ATOL = 1e-4
BACKWARD_RTOL = 2e-2  # TPU MXU precision requires looser tolerance
BACKWARD_ATOL = 2e-2
BFLOAT16_RTOL = 1e-2
BFLOAT16_ATOL = 1e-2

# Kernel defaults
DEFAULT_BLOCK_SIZE = 128
DEFAULT_MASK_VALUE = -1e10

# Common test configuration sizes
SMALL_CONFIGS = [
  # (batch_size, seq_len, d_model, vocab_size)
  (1, 128, 64, 256),
  (2, 128, 128, 512),
  (4, 128, 128, 1024),
  (4, 256, 256, 2048),
  (2, 256, 128, 4096),
]
SMALL_CONFIG_IDS = [f"B{b}_S{s}_D{d}_V{v}" for b, s, d, v in SMALL_CONFIGS]

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


# =============================================================================
# Pytest markers
# =============================================================================


def is_tpu() -> bool:
  """Check if running on TPU platform."""
  return jax.devices()[0].platform == "tpu"


requires_tpu = pytest.mark.skipif(
  not is_tpu(),
  reason="TPU test - skip on non-TPU platforms",
)


# =============================================================================
# Reference implementations
# =============================================================================


def ref_cross_entropy(
  outputs: jax.Array,
  w_unemb: jax.Array,
  targets: jax.Array,
  max_valid_id: int | None = None,
) -> jax.Array:
  """Reference cross-entropy that materializes full logits matrix.

  Args:
    outputs: Model outputs of shape (B, S, D)
    w_unemb: Unembedding weights of shape (V, D)
    targets: Target indices of shape (B, S)
    max_valid_id: Maximum valid token ID for vocab masking (optional)

  Returns:
    Scalar mean cross entropy loss
  """
  logits = jnp.einsum("bsd,vd->bsv", outputs, w_unemb)

  if max_valid_id is not None:
    vocab_ids = jnp.arange(logits.shape[-1])
    mask = vocab_ids <= max_valid_id
    logits = jnp.where(mask[None, None, :], logits, DEFAULT_MASK_VALUE)

  label_logits = jnp.take_along_axis(logits, targets[..., None], axis=-1).squeeze(-1)
  lse = jax.nn.logsumexp(logits, axis=-1)
  return (lse - label_logits).mean()


def ref_lse_forward(
  q: jax.Array,
  k: jax.Array,
  max_valid_id: int,
  vocab_size: int,
) -> jax.Array:
  """Reference LSE forward computation.

  Args:
    q: Query tensor of shape (heads, tokens, head_dim)
    k: Key tensor of shape (heads, vocab_size, head_dim)
    max_valid_id: Maximum valid vocab ID for masking
    vocab_size: Total vocabulary size

  Returns:
    LSE values of shape (heads, tokens)
  """
  logits = jnp.einsum("hsd,htd->hst", q, k)
  vocab_ids = jnp.arange(vocab_size)
  mask = vocab_ids <= max_valid_id
  logits_masked = jnp.where(mask[None, None, :], logits, DEFAULT_MASK_VALUE)
  return jax.nn.logsumexp(logits_masked, axis=-1)


def ref_backward_dq(
  q: jax.Array,
  k: jax.Array,
  do: jax.Array,
  max_valid_id: int,
  vocab_size: int,
) -> jax.Array:
  """Reference backward dQ computation: (do * S) @ K.

  Args:
    q: Query tensor of shape (heads, tokens, head_dim)
    k: Key tensor of shape (heads, vocab_size, head_dim)
    do: Upstream gradient on LSE of shape (heads, tokens)
    max_valid_id: Maximum valid vocab ID for masking
    vocab_size: Total vocabulary size

  Returns:
    dQ gradient of shape (heads, tokens, head_dim)
  """
  logits = jnp.einsum("hsd,htd->hst", q, k)
  vocab_ids = jnp.arange(vocab_size)
  mask = vocab_ids <= max_valid_id
  logits_masked = jnp.where(mask[None, None, :], logits, DEFAULT_MASK_VALUE)
  s = jax.nn.softmax(logits_masked, axis=-1)
  ds = do[:, :, None] * s
  return jnp.einsum("hst,htd->hsd", ds, k)


def ref_backward_dk(
  q: jax.Array,
  k: jax.Array,
  d_lse: jax.Array,
  max_valid_id: int,
  vocab_size: int,
) -> jax.Array:
  """Reference backward dK computation: S^T @ d_lse_q.

  Args:
    q: Query tensor of shape (heads, tokens, head_dim)
    k: Key tensor of shape (heads, vocab_size, head_dim)
    d_lse: Upstream gradient on LSE of shape (heads, tokens)
    max_valid_id: Maximum valid vocab ID for masking
    vocab_size: Total vocabulary size

  Returns:
    dK gradient of shape (heads, vocab_size, head_dim)
  """
  logits = jnp.einsum("hsd,htd->hst", q, k)
  vocab_ids = jnp.arange(vocab_size)
  mask = vocab_ids <= max_valid_id
  logits_masked = jnp.where(mask[None, None, :], logits, DEFAULT_MASK_VALUE)
  s = jax.nn.softmax(logits_masked, axis=-1)
  d_lse_q = d_lse[:, :, None] * q
  return jnp.einsum("hst,hsd->htd", s, d_lse_q)


# =============================================================================
# Config factory
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
  """Create a Config for testing with FSDP-style sharding.

  Args:
    batch_size: Global batch size
    seq_len: Sequence length
    d_model: Model dimension
    vocab_size: Vocabulary size
    max_valid_id: Maximum valid token ID
    data_sharding: Sharding spec for data dimension
    mesh_shape: TPU mesh shape
    mesh_axis_names: Names for mesh axes
    wunemb_sharding: Optional explicit sharding for unembedding weights

  Returns:
    Config object for testing
  """
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
