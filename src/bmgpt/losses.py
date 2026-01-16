from functools import partial
from typing import Any, Callable

import jax
import jax.numpy as jnp
from jax import Array

from bmgpt.config import Config
from bmgpt.data import DataloaderOutputType
from bmgpt.model import ClassificationHead, LMHead, transformer_variant_factory

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
  [Config, LMHead | ClassificationHead, Array, Array, Any, bool], Array
]


def calculate_logits(
  config: Config, unembedding: LMHead | ClassificationHead, outputs: Array
):
  """Helper to wrap _unembedding factory (expects (S, D) shape input)"""
  _, _, _, _unembedding = transformer_variant_factory(config)
  logits = _unembedding(config, unembedding, outputs)  # type: ignore
  return logits


def softmax_cross_entropy(
  config: Config,
  unembedding: LMHead | ClassificationHead,
  outputs: Array,
  targets: Array,
  shard_mapped__kernel=None,
  reduce=True,
):
  """Optax-style cross entropy loss."""
  logits = jax.vmap(partial(calculate_logits, config, unembedding))(outputs)
  label_logits = jnp.take_along_axis(logits, targets[..., None], axis=-1).squeeze(-1)
  valid_ids_mask = (
    jnp.arange(config.model.num_vocab) <= config.train_dataset.max_valid_token_id
  )
  logits = jnp.where(valid_ids_mask, logits, -jnp.inf)
  lse = jax.nn.logsumexp(logits, axis=-1)
  loss = lse - label_logits
  # loss = -label_logits
  # loss = lse
  if reduce:
    return loss.mean()
  else:
    return loss


def fused_softmax_cross_entropy(
  config: Config,
  unembedding: LMHead | ClassificationHead,
  outputs: Array,
  targets: Array,
  shard_mapped__kernel,
  reduce=True,
):
  """HBM-efficient softmax cross entropy loss."""
  # Fold batch and sequence dimensions and gather unembeddings
  b, s = outputs.shape[:2]
  outputs = outputs.reshape(b * s, -1, out_sharding=jax.P(*config.sharding.data))
  targets = targets.ravel(out_sharding=jax.P(*config.sharding.data))
  w_unemb = jax.sharding.reshard(unembedding.w, out_shardings=jax.P())

  q, k = outputs, w_unemb

  lse_sharded, lse_kernel = shard_mapped__kernel
  q, k = jax.tree.map(lambda x: x[None], (q, k))  # add singleton head dim
  # TODO: does the usual logit scaling make sense for outputs too?
  # lse = lse_sharded(lse_kernel, q / d**0.25, k / d**0.25, v, segment_ids)
  lse = lse_sharded(lse_kernel, q, k).squeeze(0)
  w_unemb = w_unemb.astype(jnp.float32)  # upcast to perform the bwd scatter-add in fp32
  per_token_unembs = w_unemb.at[targets].get(out_sharding=jax.P(*config.sharding.data))
  # targets = jax.sharding.reshard(targets, out_shardings=jax.P())
  # outputs = jax.sharding.reshard(outputs, out_shardings=jax.P())
  # per_token_unembs = w_unemb[targets]
  label_logits = jnp.sum(outputs * per_token_unembs, axis=-1)
  loss = lse - label_logits
  # loss = -label_logits
  # loss = lse
  if reduce:
    return loss.mean()
  else:
    return loss


def accuracy(
  config: Config,
  unembedding: LMHead | ClassificationHead,
  outputs: Array,
  targets: Array,
  shard_mapped__kernel=None,
  reduce=True,
):
  """Classification accuracy."""
  logits = jax.vmap(partial(calculate_logits, config, unembedding))(outputs)
  preds = logits.argmax(axis=-1)
  loss = (preds == targets).astype(jnp.int32)
  if reduce:
    return loss.mean()
  else:
    return loss
