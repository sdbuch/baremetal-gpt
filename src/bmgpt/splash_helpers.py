from functools import partial

import jax
import numpy as np
from jax.experimental.pallas.ops.tpu.splash_attention import (
  BlockSizes,
  CausalMask,
  MultiHeadMask,
  make_splash_mha,
)
from jax.experimental.pallas.ops.tpu.splash_attention.splash_attention_mask import (
  _ComputableMask,
)

from bmgpt.config import Config


class FullMask(_ComputableMask):
  """Lazy full mask, allows all tokens to attend to all other tokens."""

  def __init__(
    self,
    shape: tuple[int, int],
    shard_count: int = 1,
  ):
    def full_mask_function(q_ids, kv_ids):
      # When evaluating the mask in _process_mask we typically work with numpy
      # array views.
      # Avoid the addition when possible to avoid instantiating an actual array.
      return (q_ids >= kv_ids) | (kv_ids > q_ids)

    mask_function = full_mask_function

    super().__init__(
      shape=shape,
      mask_function=mask_function,
      shard_count=shard_count,
    )

  def __eq__(self, other: object):
    if not isinstance(other, type(self)):
      return NotImplemented

    return self.shape == other.shape and np.array_equal(
      self.q_sequence, other.q_sequence
    )

  def __hash__(self):
    return hash(
      (
        type(self),
        self.shape,
        self.q_sequence.tobytes() if self.q_sequence is not None else None,
      )
    )


def make_splash_kernel(
  config: Config,
  q_seq_len: int,
  cache_capacity: int,
  mesh,
  head_shards: int = 1,
  q_seq_shards: int = 1,
):
  # s is Q len (seq_len @ train; variable/1 at prefill/decode)
  # t is K len (s + cache_capacity)
  # see _attn
  s = q_seq_len
  t = s + cache_capacity

  if config.model.is_causal:
    mask = MultiHeadMask(
      [
        CausalMask(shape=(s, t), offset=cache_capacity)
        for _ in range(config.model.num_heads)
      ]
    )
  else:
    mask = MultiHeadMask(
      [FullMask(shape=(s, t)) for _ in range(config.model.num_heads)]
    )
  _block_size = min(config.train_dataset.seq_len, 128)
  if _block_size % 128 != 0:
    # splash attention kernel requires block size to be a multiple of 128
    return None
  block_sizes = BlockSizes(
    block_q=_block_size,
    block_kv=_block_size,
    block_kv_compute=_block_size,
    block_q_dkv=_block_size,
    block_kv_dkv=_block_size,
    block_kv_dkv_compute=_block_size,
    block_q_dq=_block_size,
    block_kv_dq=_block_size,
  )
  splash_spec = jax.P(None, None)
  splash_sharding = jax.sharding.NamedSharding(mesh, splash_spec)
  kernel = make_splash_mha(
    mask, head_shards=head_shards, q_seq_shards=q_seq_shards, block_sizes=block_sizes
  )
  kernel_spec = kernel.manual_sharding_spec(splash_sharding)

  @partial(
    jax.shard_map,
    mesh=mesh,
    in_specs=(kernel_spec, splash_spec, splash_spec, splash_spec, jax.P()),
    out_specs=splash_spec,
    check_vma=False,
  )
  def splash_sharded(kernel, q, k, v, segment_ids):
    return kernel(q, k, v, segment_ids=segment_ids)

  return (splash_sharded, kernel)
