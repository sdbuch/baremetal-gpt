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

from bmgpt.config import Config, DatasetConfig


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
  is_causal: bool,
  num_heads: int,
  seq_len: int,
  cache_capacity: int,
  mesh,
  save_residuals: bool = False,
  data_sharding: list[str | None] = [None],
  q_seq_shards: int = 1,
):
  # s is Q len (seq_len @ train; variable/1 at prefill/decode)
  # t is K len (s + cache_capacity)
  # NOTE: if save_residuals is True, assume we're doing fused cross entropy
  s = seq_len
  t = s + cache_capacity

  # mask
  if is_causal:
    mask = MultiHeadMask(
      [CausalMask(shape=(s, t), offset=cache_capacity) for _ in range(num_heads)]
    )
  else:
    mask = MultiHeadMask([FullMask(shape=(s, t)) for _ in range(num_heads)])

  # kernel
  BLOCK_SIZE = min(seq_len, 128)
  if seq_len % 128 != 0 or BLOCK_SIZE % 128 != 0:
    # splash attention kernel requires block size to be a multiple of 128
    raise NotImplementedError("Splash block size needs to be a multiple of 128")
  block_sizes = BlockSizes(
    block_q=BLOCK_SIZE,
    block_kv=BLOCK_SIZE,
    block_kv_compute=BLOCK_SIZE,
    block_q_dkv=BLOCK_SIZE,
    block_kv_dkv=BLOCK_SIZE,
    block_kv_dkv_compute=BLOCK_SIZE,
    block_q_dq=BLOCK_SIZE,
    block_kv_dq=BLOCK_SIZE,
  )
  kernel = make_splash_mha(
    mask,
    head_shards=1,
    q_seq_shards=q_seq_shards,
    block_sizes=block_sizes,
    save_residuals=save_residuals,
  )

  # sharding: can shard q_seq_len and head_dim, kv_seq_len cannot
  # q is N x S x H
  # k is N x T x H
  # v is N x T x H'
  if save_residuals:
    q_spec = jax.P(None, *data_sharding)
    kv_spec = jax.P(None, None)
    out_specs = (q_spec, (q_spec,))  # (out, (lse,))
  else:
    q_spec = jax.P(None, None)
    kv_spec = q_spec
    out_specs = q_spec
  splash_sharding = jax.sharding.NamedSharding(mesh, q_spec)
  kernel_spec = kernel.manual_sharding_spec(splash_sharding)

  # shard_mapped kernel
  @partial(
    jax.shard_map,
    mesh=mesh,
    in_specs=(kernel_spec, q_spec, kv_spec, kv_spec, jax.P()),
    out_specs=out_specs,
    check_vma=False,
  )
  def splash_sharded(kernel, q, k, v, segment_ids):
    return kernel(q, k, v, segment_ids=segment_ids)

  return (splash_sharded, kernel)
