from functools import partial
from typing import Any, Callable

import jax
import jax.numpy as jnp
from jax import Array

from bmgpt.config import Config, sharding
from bmgpt.data import DataloaderOutputType
from bmgpt.model import (
  ClassificationHead,
  LMHead,
  Unembedding,
  embedding,
  unembedding,
)

"""Loss functions for training and evaluation."""

# API:
# Positional:
#   unembedding: output projection, shape num_vocab x d_model
#   outputs: model outputs (not logits)
#   targets: token labels (int32)
#     Both are batched as (B, S, ...)
#     Shardings according to config.py (ShardingConfig)
# Keyword:
#   shard_mapped__kernel: shard-map-compatible pallas kernel tuple (or None)
#   reduce: whether to reduce over the batch axes or not


MetricType = Callable[
  [Config, LMHead | ClassificationHead, Array, Array, Any, bool], tuple[Array, dict]
]


def softmax_cross_entropy(
  config: Config,
  params: Unembedding,
  outputs: Array,
  targets: Array,
  shard_mapped__kernel=None,
  reduce=True,
):
  """Optax-style cross entropy loss."""
  logits, aux = jax.vmap(partial(unembedding, config, params))(outputs)
  label_logits = jnp.take_along_axis(logits, targets[..., None], axis=-1).squeeze(-1)
  valid_ids_mask = (
    jnp.arange(logits.shape[-1]) <= config.train_dataset.max_valid_token_id
  )
  logits = jnp.where(valid_ids_mask, logits, -jnp.inf)
  lse = jax.nn.logsumexp(logits, axis=-1)
  loss = lse - label_logits
  if reduce:
    loss = loss.mean()
  return loss, jax.tree.map(jnp.mean, aux)


def fused_softmax_cross_entropy(
  config: Config,
  params: Unembedding,
  outputs: Array,
  targets: Array,
  shard_mapped__kernel,
  reduce=True,
):
  """HBM-efficient softmax cross entropy loss."""
  w_unemb = jax.sharding.reshard(params.w.p, out_shardings=jax.P())
  lse_sharded, lse_kernel = shard_mapped__kernel

  lse = lse_sharded(lse_kernel, outputs, w_unemb)
  w_unemb_f32 = w_unemb.astype(jnp.float32)
  target_unembs = w_unemb_f32.at[targets].get(out_sharding=sharding(config).data)
  target_unembs = target_unembs.astype(w_unemb.dtype)
  label_logits = jnp.einsum(
    "bsd,bsd->bs", outputs, target_unembs, preferred_element_type=jnp.float32
  )
  loss = lse - label_logits

  if reduce:
    loss = loss.mean()
  return loss, {}


def accuracy(
  config: Config,
  params: Unembedding,
  outputs: Array,
  targets: Array,
  shard_mapped__kernel=None,
  reduce=True,
):
  """Classification accuracy."""
  logits, aux = jax.vmap(partial(unembedding, config, params))(outputs)
  preds = logits.argmax(axis=-1)
  loss = (preds == targets).astype(jnp.int32)
  if reduce:
    loss = loss.mean()
  return loss, jax.tree.map(jnp.mean, aux)
