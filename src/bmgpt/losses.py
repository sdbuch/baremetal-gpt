from functools import partial

import jax
import jax.numpy as jnp

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
  """HBM-efficient softmax cross entropy loss."""
  # Fold batch and sequence dimensions and gather unembeddings
  b, s = outputs.shape[:2]
  outputs = outputs.reshape(b * s, -1, out_sharding=jax.P(*config.sharding.data))
  targets = targets.ravel(out_sharding=jax.P(*config.sharding.data))
  w_unemb = jax.sharding.reshard(unembedding.w, out_shardings=jax.P())
  # lse from splash_attention
  q, k = outputs, w_unemb
  # seq_len_q, seq_len_kv, d = q.shape[0], k.shape[0], q.shape[1]

  lse_sharded, lse_kernel = shard_mapped__kernel
  q, k = jax.tree.map(lambda x: x[None], (q, k))  # add singleton head dim
  # TODO: does the usual logit scaling make sense for outputs too?
  # lse = lse_sharded(lse_kernel, q / d**0.25, k / d**0.25, v, segment_ids)
  lse = lse_sharded(lse_kernel, q, k).squeeze(0)
  # per_token_unembs = jnp.take_along_axis(w_unemb, targets, axis=0)  # B x D
  per_token_unembs = w_unemb.at[targets].get(out_sharding=jax.P(*config.sharding.data))
  label_logits = jnp.sum(outputs * per_token_unembs, axis=-1)
  return (lse - label_logits).mean()

  # _, _, _, _unembedding = transformer_variant_factory(config)
  # logits = jax.remat(_unembedding)(config, params.unemb, outputs)
  # label_logits = jnp.take_along_axis(logits, targets[..., None], axis=-1)
  # lse = jax.nn.logsumexp(logits, axis=-1, keepdims=True)
  # return (lse - label_logits).mean()
