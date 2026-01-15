from functools import partial

import jax
import numpy as np
from jax.experimental.pallas.ops.tpu.splash_attention import (
  BlockSizes as BlockSizesSplash,
)
from jax.experimental.pallas.ops.tpu.splash_attention import (
  CausalMask,
  MultiHeadMask,
  make_splash_mha,
)
from jax.experimental.pallas.ops.tpu.splash_attention.splash_attention_mask import (
  _ComputableMask,
)

from bmgpt.config import Config, DatasetConfig
from bmgpt.kernels.lse_kernel import BlockSizes, make_lse_kernel


class FullMask(_ComputableMask):
  """Lazy full mask, allows all tokens to attend to all other tokens."""

  def __init__(
    self,
    shape: tuple[int, int],
    shard_count: int = 1,
  ):
    def full_mask_function(q_ids, kv_ids):
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


class VocabMask(_ComputableMask):
  """Mask that allows attention only to valid vocabulary tokens (with fused xent)"""

  max_valid_id: int

  def __init__(
    self,
    shape: tuple[int, int],
    max_valid_id: int,
    shard_count: int = 1,
  ):
    self.max_valid_id = max_valid_id

    def vocab_mask_function(q_ids, kv_ids):
      return kv_ids <= max_valid_id

    super().__init__(
      shape=shape,
      mask_function=vocab_mask_function,
      shard_count=shard_count,
    )

  def __eq__(self, other: object):
    if not isinstance(other, type(self)):
      return NotImplemented
    return (
      self.shape == other.shape
      and self.max_valid_id == other.max_valid_id
      and np.array_equal(self.q_sequence, other.q_sequence)
    )

  def __hash__(self):
    return hash(
      (
        type(self),
        self.shape,
        self.max_valid_id,
        self.q_sequence.tobytes() if self.q_sequence is not None else None,
      )
    )


def make_lse_kernel_sharded(
  q_seq_len: int,
  k_seq_len: int,
  mesh,
  data_sharding: list[str | None] = [None],
  q_seq_shards: int = 1,
  head_shards: int = 1,
  block_size_mem_q: int = 128,
  block_size_mem_kv: int = 128,
  block_size_compute_kv: int = 128,
  max_valid_id: int | None = None,
):
  """Currently only supports causal transformer"""
  # s is Q len (seq_len @ train; variable/1 at prefill/decode)
  # t is K len (s + cache_capacity)
  # NOTE: if save_residuals is True, assume we're doing fused cross entropy
  s = q_seq_len
  t = k_seq_len

  # mask
  # BUG: a JAX bug means a simple full mask doesn't work!
  # VocabMask is a workaround -- it *requires* the vocab to have unused tokens
  # (so pad embeddings if necessary)
  if not max_valid_id:
    raise ValueError("max_valid_id needs to be set and >0 for save_residuals=True")
  mask = MultiHeadMask([VocabMask((s, t), max_valid_id=max_valid_id)])

  # kernel
  block_sizes = BlockSizes(
    block_q=block_size_mem_q,
    block_kv=block_size_mem_kv,
    block_kv_compute=block_size_compute_kv,
  )
  kernel = make_lse_kernel(
    mask, block_sizes=block_sizes, q_seq_shards=q_seq_shards, head_shards=head_shards
  )

  # sharding: can shard q_seq_len and head_dim, kv_seq_len cannot
  # q is N x S x H
  # k is N x T x H
  # v is N x T x H'
  q_spec = jax.P(None, *data_sharding)
  k_spec = jax.P(None, None)
  lse_spec = q_spec
  kernel_sharding = jax.sharding.NamedSharding(mesh, q_spec)
  kernel_spec = kernel.manual_sharding_spec(kernel_sharding)

  # shard_mapped kernel
  @partial(
    jax.shard_map,
    mesh=mesh,
    in_specs=(kernel_spec, q_spec, k_spec),
    out_specs=lse_spec,
    check_vma=False,
  )
  def sharded_kernel(kernel, q, k):
    return kernel(q, k)

  return (sharded_kernel, kernel)


def make_splash_kernel_sharded(
  num_heads: int,
  q_seq_len: int,
  cache_capacity: int,
  mesh,
  q_seq_shards: int = 1,
  head_shards: int = 1,
  block_size_mem_q: int = 128,
  block_size_mem_kv: int = 128,
  block_size_compute_kv: int = 128,
  use_fused_bwd_kernel: bool = False,
):
  """Currently only supports causal transformer"""
  # s is Q len (seq_len @ train; variable/1 at prefill/decode)
  # t is K len (s + cache_capacity)
  # NOTE: if save_residuals is True, assume we're doing fused cross entropy
  s = q_seq_len
  t = s + cache_capacity

  # mask
  mask = MultiHeadMask(
    [CausalMask(shape=(s, t), offset=cache_capacity) for _ in range(num_heads)]
  )

  # kernel
  # # Heuristic: if model is small enough, use fused splash backward
  # # This formula is (num_kv_blocks) * q.size * 2 <bf16> <= 64MB (mem per batch element)
  # if (t // block_size_mem_kv) * num_heads * s * 128 <= 2**25:
  if use_fused_bwd_kernel:
    block_xtra_args = {"use_fused_bwd_kernel": True}
  else:
    block_xtra_args = {"block_q_dq": block_size_mem_q, "block_kv_dq": block_size_mem_kv}
  block_sizes = BlockSizesSplash(
    block_q=block_size_mem_q,
    block_kv=block_size_mem_kv,
    block_kv_compute=block_size_compute_kv,
    block_q_dkv=block_size_mem_q,
    block_kv_dkv=block_size_mem_kv,
    block_kv_dkv_compute=block_size_compute_kv,
    **block_xtra_args,  # type: ignore
  )
  kernel = make_splash_mha(
    mask,
    head_shards=head_shards,
    q_seq_shards=q_seq_shards,
    block_sizes=block_sizes,
  )

  # sharding: can shard q_seq_len and head_dim, kv_seq_len cannot
  # q is N x S x H
  # k is N x T x H
  # v is N x T x H'
  q_spec = jax.P(None, None)
  kv_spec = q_spec
  out_spec = q_spec
  splash_sharding = jax.sharding.NamedSharding(mesh, q_spec)
  kernel_spec = kernel.manual_sharding_spec(splash_sharding)

  # shard_mapped kernel
  @partial(
    jax.shard_map,
    mesh=mesh,
    in_specs=(kernel_spec, q_spec, kv_spec, kv_spec, jax.P()),
    out_specs=out_spec,
    check_vma=False,
  )
  def splash_sharded(kernel, q, k, v, segment_ids):
    return kernel(q, k, v, segment_ids=segment_ids)

  return (splash_sharded, kernel)


def forward_kernels_from_config(config: Config, mesh):
  """Top-level wrapper for making the kernels we need for train.py"""

  def make_splash_kernel_wrapper(dataset: DatasetConfig):
    if not dataset.use_splash:
      # None ends up calling jax-xla attention: see _attn
      return None
    splash_args = (config.model.num_heads, dataset.seq_len)
    splash_kwargs = {
      "block_size_mem_q": dataset.splash_block_size_q,
      "block_size_mem_kv": dataset.splash_block_size_kv,
      "block_size_compute_kv": dataset.splash_block_size_kv_compute,
      "use_fused_bwd_kernel": dataset.splash_use_fused_bwd_kernel,
    }
    return make_splash_kernel_sharded(*splash_args, 0, mesh, **splash_kwargs)

  # model splash attention kernel
  train_attn_kernel = make_splash_kernel_wrapper(config.train_dataset)
  # fused cross entropy loss kernel
  if not config.use_fused_xent_loss:
    train_lse_kernel = None
  else:
    num_toks = (
      config.train_dataset.seq_len
      * config.train_dataset.global_batch_size
      // config.train_dataset.num_microbatches
    )
    if config.sharding.data and config.sharding.data[0]:
      data_axis_name = config.sharding.data[0]
      num_data_shards = config.sharding.mesh_shape[
        config.sharding.mesh_axis_names.index(data_axis_name)
      ]
    else:
      num_data_shards = 1
    lse_kernel_kwargs = {
      "data_sharding": config.sharding.data,
      "q_seq_shards": num_data_shards,
      "block_size_mem_q": config.fused_xent_block_size_T,
      "block_size_mem_kv": config.fused_xent_block_size_V,
      "block_size_compute_kv": config.fused_xent_block_size_V_compute,
      "max_valid_id": config.train_dataset.max_valid_token_id,
    }
    train_lse_kernel = make_lse_kernel_sharded(
      num_toks, config.model.num_vocab, mesh, **lse_kernel_kwargs
    )
  # val and test splash attention kernels
  val_kernels = [make_splash_kernel_wrapper(eval.dataset) for eval in config.val_list]
  eval_kernels = [make_splash_kernel_wrapper(eval.dataset) for eval in config.eval_list]
  assert len(val_kernels) == len(config.val_list)
  assert len(eval_kernels) == len(config.eval_list)
  return train_attn_kernel, train_lse_kernel, val_kernels, eval_kernels
