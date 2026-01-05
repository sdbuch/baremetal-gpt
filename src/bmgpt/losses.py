from functools import partial

import jax
import jax.numpy as jnp
from jax.experimental.pallas.ops.tpu.splash_attention import (
  SegmentIds,
)

from bmgpt.config import Config
from bmgpt.model import ClassificationHead, LMHead, transformer_variant_factory

"""Loss functions for training and evaluation."""

# API: inputs = (outputs, targets)
#  outputs: model outputs (NOT logits!)
#  targets: token labels (int32)
# Both are batched as (B, S, ...)
# Shardings according to config.py (ShardingConfig)


def softmax_cross_entropy(
  config: Config,
  unembedding: LMHead | ClassificationHead,
  outputs: jax.Array,
  targets: jax.Array,
  shard_mapped__kernel=None,
):
  """Optax-style cross entropy loss."""
  _, _, _, _unembedding = transformer_variant_factory(config)
  logits = jax.remat(jax.vmap(partial(_unembedding, config, unembedding)))(outputs)
  label_logits = jnp.take_along_axis(logits, targets[..., None], axis=-1)
  lse = jax.nn.logsumexp(logits, axis=-1, keepdims=True)
  return (lse - label_logits).mean()


def fused_softmax_cross_entropy(
  config: Config,
  unembedding: LMHead | ClassificationHead,
  outputs: jax.Array,
  targets: jax.Array,
  shard_mapped__kernel,
):
  """HBM-efficient softmax cross entropy loss (with splash attention!)"""
  # Fold batch and sequence dimensions and gather unembeddings
  b, s = outputs.shape[:2]
  outputs = outputs.reshape(b * s, -1, out_sharding=jax.P(*config.sharding.data))
  w_unemb = jax.sharding.reshard(unembedding.w, out_shardings=jax.P())
  # lse from splash_attention
  q = outputs
  k = w_unemb.mT
  seq_len_q, seq_len_kv, d = q.shape[0], k.shape[0], q.shape[1]
  v = jnp.zeros((seq_len_kv, 1), dtype=jnp.bfloat16)

  q_segment_ids = jnp.zeros((seq_len_q,))
  kv_segment_ids = jnp.zeros((seq_len_kv,))
  segment_ids = SegmentIds(q=q_segment_ids, kv=kv_segment_ids)

  splash_sharded, kernel = shard_mapped__kernel
  q, k, v = jax.tree.map(lambda x: x[None], (q, k, v))  # add singleton head dim
  # TODO: does the usual logit scaling make sense for outputs too?
  # _, (lse,) = splash_sharded(kernel, q / d**0.25, k / d**0.25, v, segment_ids)
  _, (lse,) = splash_sharded(kernel, q / d**0.25, k / d**0.25, v, segment_ids)

  print(lse.shape)

  # _, _, _, _unembedding = transformer_variant_factory(config)
  # logits = jax.remat(_unembedding)(config, params.unemb, outputs)
  # label_logits = jnp.take_along_axis(logits, targets[..., None], axis=-1)
  # lse = jax.nn.logsumexp(logits, axis=-1, keepdims=True)
  # return (lse - label_logits).mean()
