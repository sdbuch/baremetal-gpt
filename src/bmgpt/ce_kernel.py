# Copyright 2023 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Implementation of Sparse Flash Attention, a.k.a. "Splash" attention."""

from __future__ import annotations

import dataclasses
import enum
import functools
from collections.abc import Callable, Mapping
from typing import Any, NamedTuple, Optional

import jax
import jax.numpy as jnp
import numpy as np
from jax import ad_checkpoint, lax, tree_util
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from bmgpt import ce_mask as mask_lib
from bmgpt import ce_mask_info as mask_info_lib

partial = functools.partial
DEFAULT_MASK_VALUE = -0.7 * float(np.finfo(np.dtype("float32")).max)
NUM_LANES = 128
NUM_SUBLANES = 8
# We predefine some useful dimension numbers for dot_general
NN_DIM_NUMBERS = (((1,), (0,)), ((), ()))  # standard matmul
NT_DIM_NUMBERS = (((1,), (1,)), ((), ()))  # RHS transposed

# mypy: ignore-errors


class SegmentIds(NamedTuple):
  """SegmentIds for Q and KV sequences.

  SegmentIds are a mechanism to ensure that there is no cross-attention between
  segments (fraction of a sequence) that have been concatenated together into a
  sequence. Each array is a list of ids (integers). Only tokens with the same
  id are allowed to attend to each other.

  The static mask (e.g. causal) is "and-ed" with the segment id mask to form
  the actual attention mask. It is important that the latter does not have any
  all-zero rows (along dimension kv). Otherwise it would result in a invalid
  softmax (the denominator would be 0).
  This condition holds for causal self-attention because in this case segment
  ids form a block diagonal matrix so at least one element in each row is set.
  It is easy to break this condition with non-self-attention configurations.
  Attributes:
    q: segment ids along the Q sequence
    kv: segment ids along the KV sequence
  """

  q: jax.Array  # [q_seq_len]
  kv: jax.Array  # [kv_seq_len]


MaskFunctionType = Callable[..., jax.Array]


def get_kernel_name(
  block_metadata: Mapping[str, Any],
  is_mqa: bool,
  save_residuals: bool,
  is_segmented: bool,
  phase: str,
) -> str:
  """Returns a unique name for all SplashAttention kernel variants."""
  assert phase in ("dq", "dkv", "fwd", "ce_fwd", "ce_dq", "ce_dk")
  assert not save_residuals or phase in ("fwd", "ce_fwd")
  residuals = ""
  if save_residuals:
    residuals = "_residuals"
  elif phase in ("fwd", "ce_fwd"):
    residuals = "_no_residuals"
  attention_type = "mqa" if is_mqa else "mha"
  segments = "_segmented" if is_segmented else ""
  return f"splash_{attention_type}_{phase}{segments}{residuals}_" + "_".join(
    f"{k}={v}" for k, v in sorted(block_metadata.items())
  )


# We use an IntEnum to make it JSON serializable as regen metadata.
class QKVLayout(enum.IntEnum):
  HEAD_DIM_MINOR = enum.auto()  # [..., seq_len, head_dim]
  SEQ_MINOR = enum.auto()  # [..., head_dim, seq_len]


def from_head_minor(vals: tuple[Any, ...], layout: QKVLayout):
  if layout == QKVLayout.HEAD_DIM_MINOR:
    return vals
  return (*vals[:-2], vals[-1], vals[-2])


@dataclasses.dataclass(frozen=True, slots=True)
class BlockSizes:
  """Tile sizes parameterizing SplashAttention kernels.

  Those parameters have negligible effect on numerics, but affect performance
  greatly.

  Note that changing the layouts only influences the physical layout that the
  kernel will enforce. The logical interface to splash attention always takes
  the head dimension as the minormost one.
  """

  block_q: int
  block_kv: int
  block_kv_compute: int | None = None

  block_q_dkv: int | None = None
  block_kv_dkv: int | None = None
  block_kv_dkv_compute: int | None = None

  block_q_dq: int | None = None
  block_kv_dq: int | None = None

  use_fused_bwd_kernel: bool = False

  q_layout: QKVLayout = QKVLayout.HEAD_DIM_MINOR
  k_layout: QKVLayout = QKVLayout.HEAD_DIM_MINOR
  v_layout: QKVLayout = QKVLayout.HEAD_DIM_MINOR

  def __post_init__(self):
    if self.block_kv_compute is None:
      object.__setattr__(self, "block_kv_compute", self.block_kv)
    if self.block_kv_dkv_compute is None:
      object.__setattr__(self, "block_kv_dkv_compute", self.block_kv_dkv)
    if self.use_fused_bwd_kernel:
      if self.block_q_dq is not None or self.block_kv_dq is not None:
        raise ValueError(
          "Block sizes for dq kernel are not needed with a fused kernel."
        )

  @property
  def has_backward_blocks(self) -> bool:
    backward_blocks = (
      self.block_q_dkv,
      self.block_kv_dkv,
      self.block_kv_dkv_compute,
    )
    if not self.use_fused_bwd_kernel:
      backward_blocks += (self.block_q_dq, self.block_kv_dq)
    return all(b is not None for b in backward_blocks)

  @classmethod
  def get_default(cls):
    # TODO(apaszke,sharadmv): Select better parameters based on a heuristic.
    return BlockSizes(
      block_q=128,
      block_kv=128,
      block_kv_compute=128,
      block_q_dkv=128,
      block_kv_dkv=128,
      block_kv_dkv_compute=128,
      block_q_dq=128,
      block_kv_dq=128,
    )


def _next_nonzero(
  h,
  i,
  j,
  data_next_ref,
  block_mask_ref,
  m_next_ref,
  next_i=False,
):
  assert (data_next_ref is None) == (block_mask_ref is None)

  if data_next_ref is None and block_mask_ref is None:
    assert m_next_ref is None
    next_data = i if next_i else j
    return (
      next_data,
      None,  # next mask
      True,  # should run
      False,  # should not mask
    )

  assert data_next_ref.shape == block_mask_ref.shape
  assert m_next_ref is None or data_next_ref.shape[0] == m_next_ref.shape[0]

  if data_next_ref.shape[0] == 1:
    h = 0

  to_i32 = lambda x: x.astype(jnp.int32)

  is_nonzero = to_i32(block_mask_ref[h, i, j]) > 0
  if m_next_ref is None:
    should_not_mask = True
    next_m = None
  else:
    should_not_mask = to_i32(block_mask_ref[h, i, j]) != 1
    next_m = to_i32(m_next_ref[h, i, j])
  next_j = to_i32(data_next_ref[h, i, j])
  return next_j, next_m, is_nonzero, should_not_mask


def _apply_mask_and_soft_cap(
  qk: jax.Array,
  mask_value: float,
  should_not_mask,
  mask_ref,
  q_sequence_ref,
  q_segment_ids_ref,
  kv_segment_ids_ref,
  *,
  attn_logits_soft_cap: float,
  k_slice: pl.Slice,
  k_offset: int | jax.Array,
  bq: int,
  k_in_lanes=True,
  mask_function=None,
) -> jax.Array | tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
  assert mask_ref is None or q_sequence_ref is None
  assert (q_sequence_ref is None) == (mask_function is None)

  masks = []
  if mask_ref is not None:
    if k_in_lanes:
      mask = mask_ref[:, k_slice]
    else:
      mask = mask_ref[k_slice, :]

    masks.append(jnp.bitwise_or(mask, jnp.broadcast_to(should_not_mask, mask.shape)))
  if mask_function is not None:
    # Compute the mask using the given q_sequence indices.
    # KV indices are computed on the fly. This works because we only support Q
    # sequence sharding. If we wanted to compute Q indices too, then we would
    # need to keep into account the current shard along Q sequence.

    if k_in_lanes:
      assert q_sequence_ref.shape == (bq, NUM_LANES)

      k_sequence = k_offset + jax.lax.broadcasted_iota(jnp.int32, (bq, k_slice.size), 1)

      repeats, rem = divmod(k_slice.size, NUM_LANES)
      assert rem == 0
      q_sequence = pltpu.repeat(
        q_sequence_ref[...], repeats, axis=1
      )  # [bq, k_slice.size]
    else:
      assert q_sequence_ref.shape == (NUM_SUBLANES, bq)

      k_sequence = k_offset + jax.lax.broadcasted_iota(jnp.int32, (k_slice.size, bq), 0)
      q_sequence = q_sequence_ref[:1, :]  # [1, bq]
      q_sequence = jnp.broadcast_to(q_sequence, (k_slice.size, bq))

    assert q_sequence.shape == k_sequence.shape
    computed_mask = mask_function(
      q_sequence, k_sequence
    )  # pytype: disable=wrong-arg-count
    if computed_mask.dtype != jnp.dtype(jnp.bool_):
      raise ValueError(
        "Mask function must return a boolean-valued array, but got:"
        f" {computed_mask.dtype}"
      )
    masks.append(computed_mask)

  if q_segment_ids_ref is not None:
    if k_in_lanes:
      kv_ids = kv_segment_ids_ref[:1, k_slice]  # [1, k_slice]
      repeats, rem = divmod(kv_ids.shape[1], NUM_LANES)
      if rem:
        raise NotImplementedError(f"block_kv must be a multiple of {NUM_LANES}")
      q_ids = pltpu.repeat(q_segment_ids_ref[:], repeats, axis=1)  # [bq, bkv]
    else:
      assert bq == q_segment_ids_ref.shape[-1]
      repeats, rem = divmod(bq, NUM_LANES)
      if rem:
        raise NotImplementedError(f"block_q must be a multiple of {NUM_LANES}")
      kv_ids = pltpu.repeat(
        kv_segment_ids_ref[k_slice, :], repeats, axis=1
      )  # [k_slice, bq]
      q_ids = q_segment_ids_ref[:1, :]  # [1, bq]
    masks.append(q_ids == kv_ids)

  def cap_logits(logits):
    if attn_logits_soft_cap is not None:
      logits = jnp.tanh(qk / attn_logits_soft_cap)
      return logits * attn_logits_soft_cap
    else:
      return logits

  if masks:
    mask = functools.reduce(jnp.logical_and, masks)
    qk = cap_logits(qk)
    qk = jnp.where(mask, qk, mask_value)
  else:
    qk = cap_logits(qk)
  return qk


def ce_forward_kernel(
  # Prefetched inputs
  data_next_ref,
  block_mask_ref,
  mask_next_ref,
  # Inputs
  q_ref,
  k_ref,
  q_segment_ids_ref,
  kv_segment_ids_ref,
  sinks_ref,
  mask_ref,
  q_sequence_ref,
  # Outputs
  m_scratch_ref,
  l_scratch_ref,
  logsumexp_ref,
  *,
  mask_value: float,
  grid_width: int,
  bq: int,
  bkv: int,
  bkv_compute: int,
  q_layout: QKVLayout,
  k_layout: QKVLayout,
  attn_logits_soft_cap: float | None,
  mask_function: MaskFunctionType | None,
):
  """CE forward kernel: computes only LSE via online softmax, no V output."""
  float32 = jnp.float32
  HEAD_DIM_MINOR = QKVLayout.HEAD_DIM_MINOR

  h, i, j = pl.program_id(0), pl.program_id(1), pl.program_id(2)

  @pl.when(j == 0)
  def init():
    if sinks_ref is not None:
      sinks = sinks_ref[0, h].astype(m_scratch_ref.dtype)
      # initialize `max = sinks`, so `exp(sinks - max = 0) = 1`
      m_scratch_ref[...] = sinks * jnp.ones_like(m_scratch_ref)
      l_scratch_ref[...] = jnp.ones_like(l_scratch_ref)
    else:
      m_scratch_ref[...] = jnp.full_like(m_scratch_ref, mask_value)
      l_scratch_ref[...] = jnp.zeros_like(l_scratch_ref)

  global_kv_index, _, should_run, should_not_mask = _next_nonzero(
    h,
    i,
    j,
    data_next_ref,
    block_mask_ref,
    mask_next_ref,
  )

  def body(kv_compute_index, _):
    slice_k = pl.ds(kv_compute_index * bkv_compute, bkv_compute)
    m_prev, l_prev = m_scratch_ref[...], l_scratch_ref[...]
    assert m_prev.shape == (bq, NUM_LANES)
    assert l_prev.shape == (bq, NUM_LANES)

    q = q_ref[...] if q_layout == HEAD_DIM_MINOR else q_ref[...].T
    qk_dims = NT_DIM_NUMBERS if k_layout == HEAD_DIM_MINOR else NN_DIM_NUMBERS
    if k_layout == HEAD_DIM_MINOR:
      k = k_ref[slice_k, :]
    else:
      k = k_ref[:, slice_k]
    qk = lax.dot_general(q, k, qk_dims, preferred_element_type=float32)

    assert qk.shape == (bq, bkv_compute)
    apply_mask_and_soft_cap = functools.partial(
      _apply_mask_and_soft_cap,
      qk,
      mask_value,
      should_not_mask,
      mask_ref,
      q_sequence_ref,
      q_segment_ids_ref,
      kv_segment_ids_ref,
      attn_logits_soft_cap=attn_logits_soft_cap,
      k_slice=slice_k,
      # When the iteration space is shrunk (for local attention for example),
      # the kv_index program_id does not correspond to the actual coordinates
      # of the KV data. Make sure to use the 'unshrunk' index (coming from the
      # data_next array) when computing the mask.
      k_offset=global_kv_index * bkv + kv_compute_index * bkv_compute,
      bq=bq,
      mask_function=mask_function,
    )

    qk = apply_mask_and_soft_cap()

    m_curr = qk.max(axis=-1)[:, None]  # pytype: disable=attribute-error
    assert m_curr.shape == (bq, 1)
    m_next = jnp.maximum(m_prev, m_curr)
    assert m_next.shape == (bq, NUM_LANES)

    bkv_repeats, rem = divmod(bkv_compute, NUM_LANES)
    if rem != 0:
      raise NotImplementedError(f"{bkv_compute=} should be a multiple of {NUM_LANES}")

    s_curr = jnp.exp(qk - pltpu.repeat(m_next, bkv_repeats, axis=1))
    assert s_curr.shape == (bq, bkv_compute)

    l_curr = jax.lax.broadcast_in_dim(s_curr.sum(axis=-1), l_prev.shape, (0,))
    assert l_curr.shape == (bq, NUM_LANES)

    alpha = jnp.exp(m_prev - m_next)
    l_next = l_curr + alpha * l_prev
    m_scratch_ref[...], l_scratch_ref[...] = m_next, l_next
    # CE forward: no V @ S computation, only track m and l for LSE

  @pl.when(should_run)
  def run():
    assert bkv % bkv_compute == 0
    num_iters = k_ref.shape[0 if k_layout == HEAD_DIM_MINOR else 1] // bkv_compute
    lax.fori_loop(0, num_iters, body, None, unroll=True)

  @pl.when(j == grid_width - 1)
  def end():
    # CE forward: only compute and save logsumexp
    l = l_scratch_ref[...]
    assert logsumexp_ref.shape == (bq, NUM_LANES)
    logsumexp_ref[...] = (jnp.log(l) + m_scratch_ref[...]).astype(logsumexp_ref.dtype)

    # Reset scratch buffers
    m_scratch_ref[...] = jnp.zeros_like(m_scratch_ref)
    l_scratch_ref[...] = jnp.zeros_like(l_scratch_ref)


def _div(dividend: int, divisor: int):
  if divisor == 1:
    return dividend

  return lax.div(dividend, divisor)


def _ce_forward(
  fwd_mask_info: mask_info_lib.MaskInfo,
  q: jax.Array,
  k: jax.Array,
  segment_ids: SegmentIds | None,
  sinks: jax.Array | None,
  mask_value: float,
  is_mqa: bool,
  block_sizes: BlockSizes,
  mask_function: MaskFunctionType | None,
  attn_logits_soft_cap: float | None = None,
  interpret: bool = False,
) -> jax.Array:
  """CE forward pass: computes only LSE via online softmax.

  Args:
    fwd_mask_info: Mask info for forward pass
    q: Query tensor of shape (num_heads, q_seq_len, head_dim)
    k: Key tensor of shape (num_heads, kv_seq_len, head_dim) or (kv_seq_len, head_dim) for MQA
    segment_ids: Optional segment IDs for cross-attention masking
    sinks: Optional attention sinks
    mask_value: Value to use for masked positions
    is_mqa: Whether to use multi-query attention
    block_sizes: Block sizes for tiling
    mask_function: Optional mask function
    attn_logits_soft_cap: Optional soft cap for attention logits
    interpret: Whether to use interpret mode for debugging

  Returns:
    logsumexp: LSE values of shape (num_heads, q_seq_len)
  """
  num_q_heads, q_seq_len, head_dim_qk = q.shape
  bq, bkv = block_sizes.block_q, block_sizes.block_kv
  bkv_compute = block_sizes.block_kv_compute

  if is_mqa:
    expected_kv_rank = 2
    kv_head_dimension = 1
    kv_seq_len_dimension = 0
    num_kv_heads = 1
  else:
    expected_kv_rank = 3
    kv_head_dimension = 2
    kv_seq_len_dimension = 1
    num_kv_heads = k.shape[0]

  partial_mask_blocks = fwd_mask_info.partial_mask_blocks
  if (
    partial_mask_blocks is not None and jnp.dtype(partial_mask_blocks.dtype) != np.bool_
  ):
    raise ValueError(
      "partial_mask_blocks must be of type np.bool_ but got"
      f" {partial_mask_blocks.dtype}"
    )

  if len(k.shape) != expected_kv_rank:
    raise ValueError(
      f"Expected {expected_kv_rank}-dim 'key' tensor for MQA. Instead got a"
      f" {len(k.shape)}-dim one."
    )

  if k.shape[kv_head_dimension] != head_dim_qk:
    raise ValueError(
      f"Expected 'key' head dimension to be: {head_dim_qk}. Instead got:"
      f" {k.shape[kv_head_dimension]}."
    )

  if not is_mqa and num_q_heads % num_kv_heads != 0:
    raise ValueError(
      f"In MHA, expected number of 'key' heads ({num_kv_heads}) to be a"
      f" multiple of the number of 'query' heads ({num_q_heads})"
    )

  if bkv % bkv_compute:
    raise ValueError(f"{bkv=} must be a multiple of {bkv_compute=}.")
  if bkv_compute % NUM_LANES:
    raise ValueError(f"{bkv_compute=} must be a multiple of {NUM_LANES}.")

  kv_seq_len = k.shape[kv_seq_len_dimension]

  q_heads_per_kv_head = num_q_heads // num_kv_heads

  if segment_ids is not None:
    if segment_ids.q.shape != (q_seq_len,):
      raise ValueError(
        "Invalid shape for q segment_ids: "
        f"{segment_ids.q.shape}. Expected: {(q_seq_len,)}"
      )
    if segment_ids.kv.shape != (kv_seq_len,):
      raise ValueError(
        "Invalid shape for kv segment_ids: "
        f"{segment_ids.kv.shape}. Expected: {(kv_seq_len,)}"
      )

  q_layout = block_sizes.q_layout
  k_layout = block_sizes.k_layout

  def q_index_map(h, i, j, data_next_ref, block_mask_ref, mask_next_ref=None):
    del j, data_next_ref, mask_next_ref, block_mask_ref
    return from_head_minor((h, i, 0), q_layout)

  def k_index_map(h, i, j, data_next_ref, block_mask_ref, mask_next_ref=None):
    next_j, *_ = _next_nonzero(h, i, j, data_next_ref, block_mask_ref, mask_next_ref)
    prefix = () if is_mqa else (_div(h, q_heads_per_kv_head),)
    return from_head_minor((*prefix, next_j, 0), k_layout)

  def mask_index_map(h, i, j, data_next_ref, block_mask_ref, mask_next_ref=None):
    _, next_m, *_ = _next_nonzero(h, i, j, data_next_ref, block_mask_ref, mask_next_ref)
    return next_m, 0, 0

  def q_segment_ids_index_map(h, i, j, *_):
    del h, j  # Unused.
    return i, 0

  def kv_segment_ids_index_map(
    h, i, j, data_next_ref, block_mask_ref, mask_next_ref=None
  ):
    next_j, *_ = _next_nonzero(h, i, j, data_next_ref, block_mask_ref, mask_next_ref)
    return 0, next_j

  in_specs = [
    pl.BlockSpec(from_head_minor((None, bq, head_dim_qk), q_layout), q_index_map),
    pl.BlockSpec(
      from_head_minor(
        (bkv, head_dim_qk) if is_mqa else (None, bkv, head_dim_qk), k_layout
      ),
      k_index_map,
    ),
  ]
  if segment_ids is not None:
    in_specs += [
      pl.BlockSpec((bq, NUM_LANES), q_segment_ids_index_map),
      pl.BlockSpec((NUM_SUBLANES, bkv), kv_segment_ids_index_map),
    ]
    q_segment_ids = jax.lax.broadcast_in_dim(
      segment_ids.q, (q_seq_len, NUM_LANES), (0,)
    )
    kv_segment_ids = jax.lax.broadcast_in_dim(
      segment_ids.kv, (NUM_SUBLANES, kv_seq_len), (1,)
    )
  else:
    in_specs += [None, None]
    q_segment_ids = kv_segment_ids = None

  if sinks is not None:
    assert sinks.shape == (num_q_heads,)
    # align sinks to sublanes to allow vmap and shard_map over the kernel
    in_specs += [
      pl.BlockSpec(
        (NUM_SUBLANES, num_q_heads), lambda h, i, j, *_: (0, 0), memory_space=pltpu.SMEM
      )
    ]
    sinks = jnp.broadcast_to(
      sinks.astype(jnp.float32)[None, :], (NUM_SUBLANES, num_q_heads)
    )
  else:
    in_specs += [None]

  if fwd_mask_info.partial_mask_blocks is not None:
    in_specs.append(pl.BlockSpec((None, bq, bkv), mask_index_map))
  else:
    in_specs.append(None)

  assert fwd_mask_info.partial_mask_blocks is None or fwd_mask_info.q_sequence is None

  if fwd_mask_info.q_sequence is not None:
    q_sequence = jax.lax.broadcast_in_dim(
      fwd_mask_info.q_sequence, (q_seq_len, NUM_LANES), (0,)
    )
    in_specs.append(pl.BlockSpec((bq, NUM_LANES), q_segment_ids_index_map))
  else:
    q_sequence = None
    in_specs.append(None)

  num_scalar_prefetch = 3

  # CE forward: only output m_scratch, l_scratch, and logsumexp (no o_scratch, no out)
  def logsumexp_index_map(h, i, *_):
    return h, i, 0

  out_shapes = [
    jax.ShapeDtypeStruct((bq, NUM_LANES), jnp.float32),  # m_scratch
    jax.ShapeDtypeStruct((bq, NUM_LANES), jnp.float32),  # l_scratch
    jax.ShapeDtypeStruct((num_q_heads, q_seq_len, NUM_LANES), jnp.float32),  # logsumexp
  ]
  out_specs = [
    pl.BlockSpec((bq, NUM_LANES), lambda h, i, j, *_: (0, 0)),
    pl.BlockSpec((bq, NUM_LANES), lambda h, i, j, *_: (0, 0)),
    pl.BlockSpec((None, bq, NUM_LANES), logsumexp_index_map),
  ]

  kernel_name = get_kernel_name(
    dataclasses.asdict(block_sizes),
    is_mqa=is_mqa,
    save_residuals=True,  # CE always saves LSE
    is_segmented=segment_ids is not None,
    phase="ce_fwd",
  )

  if fwd_mask_info.data_next is not None:
    grid_width = fwd_mask_info.data_next.shape[-1]
  else:
    grid_width = kv_seq_len // bkv

  grid = (num_q_heads, q_seq_len // bq, grid_width)
  with jax.named_scope(kernel_name):
    all_out = pl.pallas_call(
      partial(
        ce_forward_kernel,
        mask_value=mask_value,
        grid_width=grid_width,
        bq=bq,
        bkv=bkv,
        bkv_compute=bkv_compute,
        q_layout=q_layout,
        k_layout=k_layout,
        attn_logits_soft_cap=attn_logits_soft_cap,
        mask_function=mask_function,
      ),
      grid_spec=pltpu.PrefetchScalarGridSpec(
        num_scalar_prefetch=num_scalar_prefetch,
        in_specs=in_specs,
        out_specs=out_specs,
        grid=grid,
      ),
      compiler_params=pltpu.CompilerParams(
        dimension_semantics=("parallel", "arbitrary", "arbitrary"),
      ),
      out_shape=out_shapes,
      name=kernel_name,
      interpret=interpret,
    )(
      fwd_mask_info.data_next,
      fwd_mask_info.block_mask,
      fwd_mask_info.mask_next,
      q if q_layout == QKVLayout.HEAD_DIM_MINOR else q.swapaxes(-1, -2),
      k if k_layout == QKVLayout.HEAD_DIM_MINOR else k.swapaxes(-1, -2),
      q_segment_ids,
      kv_segment_ids,
      sinks,
      fwd_mask_info.partial_mask_blocks,
      q_sequence,
    )

  _, _, logsumexp = all_out
  logsumexp = logsumexp[..., 0]
  return logsumexp


def _ce_backward_dq_kernel(
  # Prefetched inputs
  data_next_ref,
  block_mask_ref,
  mask_next_ref,
  # Inputs
  q_ref,
  k_ref,
  q_segment_ids_ref,
  kv_segment_ids_ref,
  sinks_ref,
  logsumexp_ref,
  d_lse_ref,
  mask_ref,
  q_sequence_ref,
  # Outputs
  dq_scratch_ref,
  dq_ref,
  *,
  mask_value: float,
  grid_width: int,
  bq: int,
  bkv: int,
  attn_logits_soft_cap: float | None = None,
  q_layout: QKVLayout,
  k_layout: QKVLayout,
  mask_function: MaskFunctionType | None,
):
  """CE backward dQ kernel: computes (do * S) @ K using pre-computed LSE.

  This computes dQ = (g_lse * softmax(Q @ K^T)) @ K with the g_lse scaling
  fused into the kernel via the do buffer. Mirrors splash attention's
  pattern where ds = (dp - di) * p, but simplified for CE (no Jacobian correction).

  Args:
    do_ref: Gradient signal (g_lse), shape (1, bq). Scales the softmax before matmul.
  """
  del sinks_ref
  float32 = jnp.float32
  HEAD_DIM_MINOR = QKVLayout.HEAD_DIM_MINOR

  h, i, j = pl.program_id(0), pl.program_id(1), pl.program_id(2)

  @pl.when(j == 0)
  def init():
    dq_scratch_ref[...] = jnp.zeros_like(dq_scratch_ref)

  global_kv_index, _, should_run, should_not_mask = _next_nonzero(
    h, i, j, data_next_ref, block_mask_ref, mask_next_ref
  )

  @pl.when(should_run)
  def run():
    q = q_ref[...] if q_layout == HEAD_DIM_MINOR else q_ref[...].T
    k = k_ref[...]
    logsumexp = jnp.expand_dims(logsumexp_ref[0], -1)
    d_lse = jnp.expand_dims(d_lse_ref[0], -1)

    qk_dims = NT_DIM_NUMBERS if k_layout == HEAD_DIM_MINOR else NN_DIM_NUMBERS
    qk_uncapped = lax.dot_general(q, k, qk_dims, preferred_element_type=float32)

    qk = _apply_mask_and_soft_cap(
      qk_uncapped,
      mask_value,
      should_not_mask,
      mask_ref,
      q_sequence_ref,
      q_segment_ids_ref,
      kv_segment_ids_ref,
      attn_logits_soft_cap=attn_logits_soft_cap,
      k_slice=pl.ds(0, bkv),
      # When the iteration space is shrunk (for local attention for example),
      # the kv_index program_id does not correspond to the actual coordinates
      # of the KV data. Make sure to use the 'unshrunk' index (coming from the
      # data_next array) when computing the mask.
      k_offset=global_kv_index * bkv,
      bq=bq,
      mask_function=mask_function,
    )
    p = jnp.exp(qk - logsumexp)
    # CE simplification: ds = d_lse * p (no Jacobian correction, just scale by d_lse)
    ds = d_lse * p

    dq_dims = NN_DIM_NUMBERS if k_layout == HEAD_DIM_MINOR else NT_DIM_NUMBERS
    dq_scratch_ref[...] += lax.dot_general(
      ds.astype(k.dtype),
      k,
      dq_dims,
      preferred_element_type=jnp.float32,
    )

  @pl.when(j == grid_width - 1)
  def end():
    dq_ref[...] = dq_scratch_ref[...].astype(dq_ref.dtype)
    dq_scratch_ref[...] = jnp.zeros_like(dq_scratch_ref)


def _ce_backward_dq(
  q,
  k,
  d_lse,
  segment_ids,
  sinks,
  logsumexp,
  *,
  bq: int,
  bkv: int,
  is_mqa: bool,
  mask_info: mask_info_lib.MaskInfo,
  mask_value: float,
  attn_logits_soft_cap: float | None,
  q_layout: QKVLayout,
  k_layout: QKVLayout,
  mask_function: MaskFunctionType | None,
  interpret: bool,
):
  """CE backward dQ: computes (do * S) @ K using pre-computed LSE.

  This computes dQ = (g_lse * softmax(Q @ K^T)) @ K with the g_lse scaling
  fused via the do buffer. Mirrors splash attention's pattern but simplified
  for CE (no Jacobian correction).

  Args:
    do: Gradient signal (g_lse), shape (num_heads, q_seq_len). Scales the softmax.
  """
  num_q_heads, q_seq_len, head_dim_qk = q.shape
  if is_mqa:
    kv_seq_len = k.shape[0]
    num_kv_heads = 1
  else:
    kv_seq_len = k.shape[1]
    num_kv_heads = k.shape[0]

  if bq > q_seq_len:
    raise ValueError(f"{bq=} should not be greater than {q_seq_len=}")
  if bkv > kv_seq_len:
    raise ValueError(f"{bkv=} should not be greater than {kv_seq_len=}")

  if not is_mqa and num_q_heads % num_kv_heads != 0:
    raise ValueError(
      f"In MHA, expected number of 'key' heads ({num_kv_heads}) to be a"
      f" multiple of the number of 'query' heads ({num_q_heads})"
    )

  if bkv % NUM_LANES:
    raise ValueError(f"{bkv=} must be a multiple of {NUM_LANES}.")

  q_heads_per_kv_head = num_q_heads // num_kv_heads

  if mask_info.data_next is not None:
    grid_width = mask_info.data_next.shape[-1]
  else:
    grid_width = kv_seq_len // bkv

  grid = (num_q_heads, q_seq_len // bq, grid_width)

  def q_index_map(h, i, *_):
    return from_head_minor((h, i, 0), q_layout)

  q_spec = pl.BlockSpec(from_head_minor((None, bq, head_dim_qk), q_layout), q_index_map)

  def k_index_map(h, i, j, data_next_ref, block_mask_ref, mask_next_ref, *_):
    next_j, *_ = _next_nonzero(h, i, j, data_next_ref, block_mask_ref, mask_next_ref)
    prefix = () if is_mqa else (_div(h, q_heads_per_kv_head),)
    return from_head_minor((*prefix, next_j, 0), k_layout)

  k_spec = pl.BlockSpec(
    from_head_minor(
      (bkv, head_dim_qk) if is_mqa else (None, bkv, head_dim_qk), k_layout
    ),
    k_index_map,
  )

  def mask_index_map(h, i, j, data_next_ref, block_mask_ref, mask_next_ref, *_):
    _, next_m, *_ = _next_nonzero(h, i, j, data_next_ref, block_mask_ref, mask_next_ref)
    return next_m, 0, 0

  mask_spec = pl.BlockSpec((None, bq, bkv), mask_index_map)

  def q_segment_ids_index_map(h, i, j, *_):
    del h, j  # Unused.
    return i, 0

  if segment_ids is not None:

    def kv_segment_ids_index_map(
      h, i, j, data_next_ref, block_mask_ref, mask_next_ref, *_
    ):
      next_j, *_ = _next_nonzero(h, i, j, data_next_ref, block_mask_ref, mask_next_ref)
      return 0, next_j

    q_segment_spec = pl.BlockSpec((bq, NUM_LANES), q_segment_ids_index_map)
    kv_segment_spec = pl.BlockSpec((NUM_SUBLANES, bkv), kv_segment_ids_index_map)
    q_segment_ids = jax.lax.broadcast_in_dim(
      segment_ids.q, (q_seq_len, NUM_LANES), (0,)
    )
    kv_segment_ids = jax.lax.broadcast_in_dim(
      segment_ids.kv, (NUM_SUBLANES, kv_seq_len), (1,)
    )
  else:
    q_segment_spec = kv_segment_spec = None
    q_segment_ids = kv_segment_ids = None

  if sinks is not None:
    assert sinks.shape == (num_q_heads,)
    # align sinks to sublanes to allow vmap and shard_map over the kernel
    sinks_spec = pl.BlockSpec(
      (NUM_SUBLANES, num_q_heads), lambda h, i, j, *_: (0, 0), memory_space=pltpu.SMEM
    )
    sinks = jnp.broadcast_to(
      sinks.astype(jnp.float32)[None, :], (NUM_SUBLANES, num_q_heads)
    )
  else:
    sinks_spec = None

  def logsumexp_index_map(h, i, *_):
    return h, 0, i

  logsumexp = jnp.expand_dims(logsumexp, axis=-2)
  logsumexp_spec = pl.BlockSpec((None, 1, bq), logsumexp_index_map)
  assert logsumexp.ndim == len(logsumexp_spec.block_shape)

  # d_lse has same shape/spec as logsumexp
  d_lse = jnp.expand_dims(d_lse, axis=-2)
  d_lse_spec = pl.BlockSpec((None, 1, bq), logsumexp_index_map)
  assert d_lse.ndim == len(d_lse_spec.block_shape)

  in_specs = [
    q_spec,
    k_spec,
    q_segment_spec,
    kv_segment_spec,
    sinks_spec,
    logsumexp_spec,
    d_lse_spec,
  ]
  if mask_info.partial_mask_blocks is not None:
    in_specs.append(mask_spec)
  else:
    in_specs.append(None)

  assert mask_info.partial_mask_blocks is None or mask_info.q_sequence is None

  if mask_info.q_sequence is not None:
    q_sequence = jax.lax.broadcast_in_dim(
      mask_info.q_sequence, (q_seq_len, NUM_LANES), (0,)
    )
    in_specs.append(pl.BlockSpec((bq, NUM_LANES), q_segment_ids_index_map))
  else:
    q_sequence = None
    in_specs.append(None)

  out_shapes = [
    jax.ShapeDtypeStruct((bq, head_dim_qk), jnp.float32),
    jax.ShapeDtypeStruct(q.shape, q.dtype),
  ]
  out_specs = [
    pl.BlockSpec((bq, head_dim_qk), lambda *_: (0, 0)),
    pl.BlockSpec((None, bq, head_dim_qk), lambda h, i, *_: (h, i, 0)),
  ]

  kernel = functools.partial(
    _ce_backward_dq_kernel,
    grid_width=grid_width,
    mask_value=mask_value,
    bq=bq,
    bkv=bkv,
    attn_logits_soft_cap=attn_logits_soft_cap,
    q_layout=q_layout,
    k_layout=k_layout,
    mask_function=mask_function,
  )
  num_scalar_prefetch = 3

  kernel_name = get_kernel_name(
    dict(
      block_q_dq=bq,
      block_kv_dq=bkv,
      q_layout=q_layout,
      k_layout=k_layout,
    ),
    is_mqa=is_mqa,
    save_residuals=False,
    is_segmented=segment_ids is not None,
    phase="ce_dq",
  )
  with jax.named_scope(kernel_name):
    _, dq = pl.pallas_call(
      kernel,
      grid_spec=pltpu.PrefetchScalarGridSpec(
        num_scalar_prefetch=num_scalar_prefetch,
        in_specs=in_specs,
        out_specs=out_specs,
        grid=grid,
      ),
      out_shape=out_shapes,
      compiler_params=pltpu.CompilerParams(
        dimension_semantics=("arbitrary", "arbitrary", "arbitrary"),
      ),
      name=kernel_name,
      interpret=interpret,
    )(
      mask_info.data_next,
      mask_info.block_mask,
      mask_info.mask_next,
      q if q_layout == QKVLayout.HEAD_DIM_MINOR else q.swapaxes(-1, -2),
      k if k_layout == QKVLayout.HEAD_DIM_MINOR else k.swapaxes(-1, -2),
      q_segment_ids,
      kv_segment_ids,
      sinks,
      logsumexp,
      d_lse,
      mask_info.partial_mask_blocks,
      q_sequence,
    )
  return dq


def _ce_backward_dk_kernel(
  # Prefetched inputs
  data_next_ref,
  block_mask_ref,
  mask_next_ref,
  # Inputs
  q_ref,
  k_ref,
  q_segment_ids_ref,
  kv_segment_ids_ref,
  sinks_ref,
  logsumexp_ref,
  d_lse_q_ref,
  mask_ref,
  q_sequence_ref,
  # Outputs
  dk_scratch_ref,
  dk_ref,
  *,
  num_q_heads: int,
  num_kv_heads: int,
  mask_value: float,
  grid_width: int,
  bq: int,
  bkv_compute: int,
  is_mqa: bool,
  attn_logits_soft_cap: float | None,
  q_layout: QKVLayout,
  k_layout: QKVLayout,
  bkv: int,
  mask_function: MaskFunctionType | None,
):
  """CE backward dK kernel: computes S^T @ d_lse_q using pre-computed LSE.

  This computes dK = softmax(Q @ K^T)^T @ d_lse_q where d_lse_q = d_lse[..., None] * q.
  The d_lse scaling is pre-multiplied into the d_lse_q buffer. Mirrors splash
  attention's pattern but simplified for CE (no Jacobian correction).

  Note: q_ref is still used to compute S via Q @ K^T. d_lse_q_ref is used for
  the final dot product S^T @ d_lse_q.

  Args:
    d_lse_q_ref: Gradient signal (d_lse * q), shape (bq, head_dim). Used in final dot product.
  """
  del sinks_ref
  HEAD_DIM_MINOR = QKVLayout.HEAD_DIM_MINOR
  kv_index, q_head_index, q_index = (
    pl.program_id(0),
    pl.program_id(1),
    pl.program_id(2),
  )
  should_initialize = q_index == 0

  q_heads_per_kv_heads = None
  q_head_index_per_kv_head = None

  # Consider this situation:
  # Q_heads:   0, 1, 2, 3, 4, 5, 6, 7
  # KV_heads:  0,    1,    2,    3
  # The gradient scratch buffers should be initialized for Q_heads 0, 2, 4, 6
  # (first Q_heads to 'see' a new KV_head).
  # The gradient output buffers should be written for Q_heads 1, 3, 5, 7 (last
  # Q_heads to 'see' the current KV_head).

  # We can use the same logic for both MQA and GA (grouped attention).
  # But for MQA there is no need for the rem instruction, so we skip it.
  if is_mqa:
    should_initialize = jnp.logical_and(should_initialize, q_head_index == 0)
  elif num_kv_heads < num_q_heads:
    q_heads_per_kv_heads = num_q_heads // num_kv_heads
    q_head_index_per_kv_head = lax.rem(q_head_index, q_heads_per_kv_heads)
    should_initialize = jnp.logical_and(
      should_initialize, q_head_index_per_kv_head == 0
    )

  @pl.when(should_initialize)
  def init():
    dk_scratch_ref[...] = jnp.zeros_like(dk_scratch_ref)

  _, _, should_run, should_not_mask = _next_nonzero(
    q_head_index,
    q_index,
    kv_index,
    data_next_ref,
    block_mask_ref,
    mask_next_ref,
    next_i=True,
  )

  def body(i, _):
    slice_k = pl.ds(i * bkv_compute, bkv_compute)
    q = q_ref[...]  # Used for computing S via Q @ K^T
    d_lse_q = d_lse_q_ref[...]  # Pre-computed d_lse * q for gradient scaling

    def _load_k(ref, layout):
      if layout == HEAD_DIM_MINOR:
        return ref[slice_k, :]
      return ref[:, slice_k].T

    k = _load_k(k_ref, k_layout)
    logsumexp = logsumexp_ref[:1, :]

    qk_dims = NT_DIM_NUMBERS if q_layout == HEAD_DIM_MINOR else NN_DIM_NUMBERS
    qk_uncapped = lax.dot_general(k, q, qk_dims, preferred_element_type=jnp.float32)

    qk = _apply_mask_and_soft_cap(
      qk_uncapped,
      mask_value,
      should_not_mask,
      mask_ref,
      q_sequence_ref,
      q_segment_ids_ref,
      kv_segment_ids_ref,
      attn_logits_soft_cap=attn_logits_soft_cap,
      k_slice=slice_k,
      k_offset=kv_index * bkv + i * bkv_compute,
      bq=bq,
      k_in_lanes=False,
      mask_function=mask_function,
    )
    # Note: qk = K @ Q^T = (Q @ K^T)^T, so p = exp(qk - lse) is S^T (transposed softmax)
    p = jnp.exp(qk - logsumexp)

    # CE: dk = S^T @ d_lse_q where d_lse_q = d_lse * q (gradient scaling fused)
    # p is S^T block (bkv_compute, bq), d_lse_q is (bq, head_dim), result is (bkv_compute, head_dim)
    dk_dims = NN_DIM_NUMBERS if q_layout == HEAD_DIM_MINOR else NT_DIM_NUMBERS
    dk = lax.dot_general(
      p.astype(d_lse_q.dtype), d_lse_q, dk_dims, preferred_element_type=jnp.float32
    )
    dk = dk.astype(dk_scratch_ref.dtype) + dk_scratch_ref[slice_k, :]
    dk_scratch_ref[slice_k, :] = dk

  @pl.when(should_run)
  def run():
    num_iters = k_ref.shape[0 if k_layout is HEAD_DIM_MINOR else 1] // bkv_compute
    lax.fori_loop(0, num_iters, body, None, unroll=True)

  should_write = q_index == grid_width - 1
  if is_mqa:
    should_write = jnp.logical_and(should_write, q_head_index == num_q_heads - 1)
  elif num_kv_heads < num_q_heads:
    should_write = jnp.logical_and(
      should_write, q_head_index_per_kv_head == q_heads_per_kv_heads - 1
    )

  @pl.when(should_write)
  def end():
    dk_ref[...] = dk_scratch_ref[...].astype(dk_ref.dtype)
    dk_scratch_ref[...] = jnp.zeros_like(dk_scratch_ref)


def _ce_backward_dk(
  q,
  k,
  d_lse_q,
  segment_ids,
  sinks,
  logsumexp,
  *,
  bq: int,
  bkv: int,
  bkv_compute: int,
  is_mqa: bool,
  mask_info: mask_info_lib.MaskInfo,
  mask_value: float,
  attn_logits_soft_cap: float | None,
  q_layout: QKVLayout,
  k_layout: QKVLayout,
  mask_function: MaskFunctionType | None,
  interpret: bool,
):
  """CE backward dK: computes S^T @ Q using pre-computed LSE.

  This is a simplified version of _splash_attention_bwd_dkv that computes
  dK = softmax(Q @ K^T)^T @ Q directly, without the Jacobian correction term.
  """
  num_q_heads, q_seq_len, head_dim_qk = q.shape
  if is_mqa:
    num_kv_heads, kv_seq_len = 1, k.shape[0]
  else:
    num_kv_heads, kv_seq_len, _ = k.shape

  if bq > q_seq_len:
    raise ValueError(f"{bq=} should not be greater than {q_seq_len=}")
  if bkv > kv_seq_len:
    raise ValueError(f"{bkv=} should not be greater than {kv_seq_len=}")
  if bkv_compute > bkv:
    raise ValueError(f"{bkv_compute=} should not be greater than {bkv=}")
  if bkv % bkv_compute:
    raise ValueError(f"{bkv=} should be a multiple of {bkv_compute=}")

  if not is_mqa and num_q_heads % num_kv_heads != 0:
    raise ValueError(
      f"In MHA, expected number of 'key' heads ({num_kv_heads}) to be a"
      f" multiple of the number of 'query' heads ({num_q_heads})"
    )

  q_heads_per_kv_head = num_q_heads // num_kv_heads

  if mask_info.data_next is not None:
    grid_width = mask_info.data_next.shape[-2]
  else:
    grid_width = q_seq_len // bq

  grid = (
    kv_seq_len // bkv,
    num_q_heads,
    grid_width,
  )

  def q_index_map(
    kv_index,
    head_index,
    q_index,
    data_next_ref,
    block_mask_ref,
    mask_next_ref=None,
  ):
    next_i, *_ = _next_nonzero(
      head_index,
      q_index,
      kv_index,
      data_next_ref,
      block_mask_ref,
      mask_next_ref,
      next_i=True,
    )
    return from_head_minor((head_index, next_i, 0), q_layout)

  q_spec = pl.BlockSpec(from_head_minor((None, bq, head_dim_qk), q_layout), q_index_map)

  def k_index_map(kv_index, head_index, *_):
    prefix = () if is_mqa else (_div(head_index, q_heads_per_kv_head),)
    return from_head_minor((*prefix, kv_index, 0), k_layout)

  k_spec = pl.BlockSpec(
    from_head_minor(
      (bkv, head_dim_qk) if is_mqa else (None, bkv, head_dim_qk),
      k_layout,
    ),
    k_index_map,
  )

  def dkv_index_map(kv_index, head_index, *_):
    prefix = () if is_mqa else (_div(head_index, q_heads_per_kv_head),)
    return (*prefix, kv_index, 0)

  dk_spec = pl.BlockSpec(
    (bkv, head_dim_qk) if is_mqa else (None, bkv, head_dim_qk),
    dkv_index_map,
  )

  def mask_index_map(
    kv_index,
    head_index,
    q_index,
    data_next_ref,
    block_mask_ref,
    mask_next_ref,
  ):
    _, next_m, *_ = _next_nonzero(
      head_index,
      q_index,
      kv_index,
      data_next_ref,
      block_mask_ref,
      mask_next_ref,
      next_i=True,
    )
    return next_m, 0, 0

  mask_spec = pl.BlockSpec((None, bkv, bq), mask_index_map)

  def q_segment_ids_index_map(
    kv_index,
    head_index,
    q_index,
    data_next_ref,
    block_mask_ref,
    mask_next_ref=None,
  ):
    next_i, *_ = _next_nonzero(
      head_index,
      q_index,
      kv_index,
      data_next_ref,
      block_mask_ref,
      mask_next_ref,
      next_i=True,
    )
    return 0, next_i

  if segment_ids is not None:

    def kv_segment_ids_index_map(kv_index, *_):
      return kv_index, 0

    q_segment_spec = pl.BlockSpec((NUM_SUBLANES, bq), q_segment_ids_index_map)
    kv_segment_spec = pl.BlockSpec((bkv, NUM_LANES), kv_segment_ids_index_map)
    q_segment_ids = jax.lax.broadcast_in_dim(
      segment_ids.q, (NUM_SUBLANES, q_seq_len), (1,)
    )
    kv_segment_ids = jax.lax.broadcast_in_dim(
      segment_ids.kv, (kv_seq_len, NUM_LANES), (0,)
    )
  else:
    q_segment_spec = kv_segment_spec = None
    q_segment_ids = kv_segment_ids = None

  if sinks is not None:
    assert sinks.shape == (num_q_heads,)
    # align sinks to sublanes to allow vmap and shard_map over the kernel
    sinks_spec = pl.BlockSpec(
      (NUM_SUBLANES, num_q_heads), lambda h, i, j, *_: (0, 0), memory_space=pltpu.SMEM
    )
    sinks = jnp.broadcast_to(
      sinks.astype(jnp.float32)[None, :], (NUM_SUBLANES, num_q_heads)
    )
  else:
    sinks_spec = None

  def logsumexp_index_map(
    kv_index,
    head_index,
    q_index,
    data_next_ref,
    block_mask_ref,
    mask_next_ref=None,
  ):
    next_i, *_ = _next_nonzero(
      head_index,
      q_index,
      kv_index,
      data_next_ref,
      block_mask_ref,
      mask_next_ref,
      next_i=True,
    )
    return head_index, 0, next_i

  assert logsumexp.shape == (num_q_heads, q_seq_len)
  # TODO(apaszke): Remove the sublane expansion once Mosaic has all retilings
  logsumexp_shape = (num_q_heads, NUM_SUBLANES, q_seq_len)
  logsumexp = jnp.broadcast_to(jnp.expand_dims(logsumexp, -2), logsumexp_shape)
  logsumexp_spec = pl.BlockSpec((None, NUM_SUBLANES, bq), logsumexp_index_map)
  assert logsumexp.ndim == len(logsumexp_spec.block_shape)

  # d_lse_q_spec matches q_spec since d_lse_q = d_lse[..., None] * q has same shape as q
  d_lse_q_spec = pl.BlockSpec(
    from_head_minor((None, bq, head_dim_qk), q_layout), q_index_map
  )

  in_specs = [
    q_spec,
    k_spec,
    q_segment_spec,
    kv_segment_spec,
    sinks_spec,
    logsumexp_spec,
    d_lse_q_spec,
  ]
  if mask_info.partial_mask_blocks is not None:
    in_specs.append(mask_spec)
  else:
    in_specs.append(None)

  if mask_info.q_sequence is not None:
    in_specs.append(pl.BlockSpec((NUM_SUBLANES, bq), q_segment_ids_index_map))
    q_sequence = jax.lax.broadcast_in_dim(
      mask_info.q_sequence, (NUM_SUBLANES, q_seq_len), (1,)
    )
  else:
    q_sequence = None
    in_specs.append(None)

  out_shapes = [
    jax.ShapeDtypeStruct((bkv, head_dim_qk), jnp.float32),
    jax.ShapeDtypeStruct(k.shape, k.dtype),
  ]
  out_specs = [
    pl.BlockSpec((bkv, head_dim_qk), lambda *_: (0, 0)),
    dk_spec,
  ]

  kernel = functools.partial(
    _ce_backward_dk_kernel,
    mask_value=mask_value,
    num_q_heads=num_q_heads,
    num_kv_heads=num_kv_heads,
    is_mqa=is_mqa,
    grid_width=grid_width,
    bq=bq,
    bkv_compute=bkv_compute,
    attn_logits_soft_cap=attn_logits_soft_cap,
    q_layout=q_layout,
    k_layout=k_layout,
    bkv=bkv,
    mask_function=mask_function,
  )
  num_scalar_prefetch = 3

  kernel_name = get_kernel_name(
    dict(
      block_q_dk=bq,
      block_kv_dk=bkv,
      block_kv_dk_compute=bkv_compute,
      q_layout=q_layout,
      k_layout=k_layout,
    ),
    is_mqa=is_mqa,
    save_residuals=False,
    is_segmented=segment_ids is not None,
    phase="ce_dk",
  )
  with jax.named_scope(kernel_name):
    _, dk = pl.pallas_call(
      kernel,
      grid_spec=pltpu.PrefetchScalarGridSpec(
        num_scalar_prefetch=num_scalar_prefetch,
        in_specs=in_specs,
        out_specs=out_specs,
        grid=grid,
      ),
      out_shape=out_shapes,
      # We set all dimensions to arbitrary because:
      # 1) for kv_seq_len, the splash attention prefetch schedule assumes no
      #    megacore
      # 2) for heads, we are reducing over heads
      # 3) for q_seq_len, we are reducing over it to compute dk
      compiler_params=pltpu.CompilerParams(
        dimension_semantics=("arbitrary", "arbitrary", "arbitrary"),
      ),
      name=kernel_name,
      interpret=interpret,
    )(
      mask_info.data_next,
      mask_info.block_mask,
      mask_info.mask_next,
      q if q_layout == QKVLayout.HEAD_DIM_MINOR else q.swapaxes(-1, -2),
      k if k_layout == QKVLayout.HEAD_DIM_MINOR else k.swapaxes(-1, -2),
      q_segment_ids,
      kv_segment_ids,
      sinks,
      logsumexp,
      d_lse_q if q_layout == QKVLayout.HEAD_DIM_MINOR else d_lse_q.swapaxes(-1, -2),
      mask_info.partial_mask_blocks,
      q_sequence,
    )
  return dk


# =============================================================================
# Custom VJP Wrappers
# =============================================================================


CEResidualsType = tuple[
  jax.Array,  # q
  jax.Array,  # k
  SegmentIds | None,  # segment_ids
  jax.Array | None,  # sinks
  jax.Array,  # logsumexp
  mask_info_lib.MaskInfo,  # dq_mask_info
  mask_info_lib.MaskInfo,  # dk_mask_info
]


@partial(
  jax.custom_vjp,
  nondiff_argnames=(
    "mask_value",
    "is_mqa",
    "block_sizes",
    "mask_function",
    "attn_logits_soft_cap",
    "interpret",
  ),
)
def _ce_custom(
  fwd_mask_info: mask_info_lib.MaskInfo,
  dq_mask_info: mask_info_lib.MaskInfo,
  dk_mask_info: mask_info_lib.MaskInfo,
  q: jax.Array,
  k: jax.Array,
  segment_ids: SegmentIds | None,
  sinks: jax.Array | None,
  mask_value: float,
  is_mqa: bool,
  block_sizes: BlockSizes,
  mask_function: MaskFunctionType | None,
  attn_logits_soft_cap: float | None = None,
  interpret: bool = False,
) -> jax.Array:
  del dq_mask_info, dk_mask_info

  return _ce_forward(
    fwd_mask_info=fwd_mask_info,
    q=q,
    k=k,
    segment_ids=segment_ids,
    sinks=sinks,
    mask_value=mask_value,
    is_mqa=is_mqa,
    block_sizes=block_sizes,
    mask_function=mask_function,
    attn_logits_soft_cap=attn_logits_soft_cap,
    interpret=interpret,
  )


def _ce_fwd(
  fwd_mask_info: mask_info_lib.MaskInfo,
  dq_mask_info: mask_info_lib.MaskInfo,
  dk_mask_info: mask_info_lib.MaskInfo,
  q: jax.Array,
  k: jax.Array,
  segment_ids: SegmentIds | None,
  sinks: jax.Array | None,
  mask_value: float,
  is_mqa: bool,
  block_sizes: BlockSizes,
  mask_function: MaskFunctionType | None,
  attn_logits_soft_cap: float | None = None,
  interpret: bool = False,
) -> tuple[jax.Array, CEResidualsType]:
  lse = _ce_forward(
    fwd_mask_info=fwd_mask_info,
    q=q,
    k=k,
    segment_ids=segment_ids,
    sinks=sinks,
    mask_value=mask_value,
    is_mqa=is_mqa,
    block_sizes=block_sizes,
    mask_function=mask_function,
    attn_logits_soft_cap=attn_logits_soft_cap,
    interpret=interpret,
  )

  residuals = (
    q,
    k,
    segment_ids,
    sinks,
    lse,
    dq_mask_info,
    dk_mask_info,
  )
  return lse, residuals


def _ce_bwd(
  mask_value: float,
  is_mqa: bool,
  block_sizes: BlockSizes,
  mask_function: MaskFunctionType | None,
  attn_logits_soft_cap: float | None,
  interpret: bool,
  residuals: CEResidualsType,
  g_lse: jax.Array,
) -> tuple[
  None,  # fwd_mask_info
  None,  # dq_mask_info
  None,  # dk_mask_info
  jax.Array,  # dq
  jax.Array,  # dk
  None,  # segment_ids
  None,  # sinks
]:
  q, k, segment_ids, sinks, logsumexp, dq_mask_info, dk_mask_info = residuals

  bq = block_sizes.block_q
  bkv = block_sizes.block_kv
  bkv_compute = block_sizes.block_kv_compute or bkv
  q_layout = block_sizes.q_layout
  k_layout = block_sizes.k_layout

  dq = _ce_backward_dq(
    q=q,
    k=k,
    d_lse=g_lse,
    segment_ids=segment_ids,
    sinks=sinks,
    logsumexp=logsumexp,
    bq=bq,
    bkv=bkv,
    is_mqa=is_mqa,
    mask_info=dq_mask_info,
    mask_value=mask_value,
    attn_logits_soft_cap=attn_logits_soft_cap,
    q_layout=q_layout,
    k_layout=k_layout,
    mask_function=mask_function,
    interpret=interpret,
  )

  d_lse_q = g_lse[:, :, None] * q
  dk = _ce_backward_dk(
    q=q,
    k=k,
    d_lse_q=d_lse_q,
    segment_ids=segment_ids,
    sinks=sinks,
    logsumexp=logsumexp,
    bq=bq,
    bkv=bkv,
    bkv_compute=bkv_compute,
    is_mqa=is_mqa,
    mask_info=dk_mask_info,
    mask_value=mask_value,
    attn_logits_soft_cap=attn_logits_soft_cap,
    q_layout=q_layout,
    k_layout=k_layout,
    mask_function=mask_function,
    interpret=interpret,
  )

  return (
    None,  # fwd_mask_info
    None,  # dq_mask_info
    None,  # dk_mask_info
    dq,  # q
    dk,  # k
    None,  # segment_ids
    None,  # sinks
  )


_ce_custom.defvjp(_ce_fwd, _ce_bwd)


@partial(
  jax.jit,
  static_argnames=[
    "is_mqa",
    "block_sizes",
    "mask_value",
    "attn_logits_soft_cap",
    "mask_function",
    "interpret",
  ],
)
def _ce(
  fwd_mask_info: mask_info_lib.MaskInfo,
  dq_mask_info: mask_info_lib.MaskInfo | None,
  dk_mask_info: mask_info_lib.MaskInfo | None,
  q: jax.Array,
  k: jax.Array,
  segment_ids: SegmentIds | None = None,
  sinks: jax.Array | None = None,
  *,
  is_mqa: bool,
  block_sizes: BlockSizes | None,
  mask_value: float,
  attn_logits_soft_cap: float | None,
  mask_function: MaskFunctionType | None,
  interpret: bool,
) -> jax.Array:
  def _collapse_partial_mask_blocks(mask_info: mask_info_lib.MaskInfo | None):
    if mask_info is None or mask_info.partial_mask_blocks is None:
      return mask_info
    return mask_info._replace(
      partial_mask_blocks=mask_info.partial_mask_blocks.reshape(
        -1, *mask_info.partial_mask_blocks.shape[-2:]
      )
    )

  fwd_mask_info = _collapse_partial_mask_blocks(fwd_mask_info)
  dq_mask_info = _collapse_partial_mask_blocks(dq_mask_info)
  dk_mask_info = _collapse_partial_mask_blocks(dk_mask_info)
  return _ce_custom(
    fwd_mask_info,
    dq_mask_info,
    dk_mask_info,
    q,
    k,
    segment_ids,
    sinks,
    mask_value=mask_value,
    is_mqa=is_mqa,
    block_sizes=block_sizes,
    mask_function=mask_function,
    attn_logits_soft_cap=attn_logits_soft_cap,
    interpret=interpret,
  )


@jax.tree_util.register_pytree_node_class
class CEKernel:
  def __init__(
    self,
    fwd_mask_info: mask_info_lib.MaskInfo,
    dq_mask_info: mask_info_lib.MaskInfo | None,
    dk_mask_info: mask_info_lib.MaskInfo | None,
    **kwargs,
  ):
    self.kwargs = kwargs
    self.fwd_mask_info = fwd_mask_info
    self.dq_mask_info = dq_mask_info
    self.dk_mask_info = dk_mask_info

  def __call__(self, *args, **kwargs) -> jax.Array:
    return _ce(
      self.fwd_mask_info,
      self.dq_mask_info,
      self.dk_mask_info,
      *args,
      **kwargs,
      **self.kwargs,
    )

  def manual_sharding_spec(self, sharding: jax.sharding.NamedSharding):
    """Returns a value that can be used as a shard_map partition spec for the kernel."""
    if self.fwd_mask_info.data_next is not None:
      block_mask_shape = self.fwd_mask_info.data_next.shape
      try:
        shard_shape = sharding.shard_shape(block_mask_shape)
      except ValueError as exc:
        raise ValueError(
          "The sharding must divide the mask blocks evenly between devices"
        ) from exc
      if block_mask_shape[-1] != shard_shape[-1]:
        raise ValueError("Sharding the kv sequence dimension is not supported")
    spec = sharding.spec
    assert len(spec) == 2
    replicated = jax.sharding.PartitionSpec()
    partial_mask_blocks_spec = (
      spec if self.fwd_mask_info.is_dynamic_mask else replicated
    )
    q_sequence_spec = jax.sharding.PartitionSpec(spec[1])
    mask_info_specs = mask_info_lib.MaskInfo(  # pytype: disable=wrong-arg-types
      data_next=spec if self.fwd_mask_info.data_next is not None else None,
      mask_next=spec if self.fwd_mask_info.mask_next is not None else None,
      block_mask=spec if self.fwd_mask_info.block_mask is not None else None,
      partial_mask_blocks=partial_mask_blocks_spec
      if self.fwd_mask_info.partial_mask_blocks is not None
      else None,
      q_sequence=q_sequence_spec if self.fwd_mask_info.q_sequence is not None else None,
    )
    return CEKernel(
      mask_info_specs,
      mask_info_specs if self.dq_mask_info is not None else None,
      mask_info_specs if self.dk_mask_info is not None else None,
      **self.kwargs,
    )

  def tree_flatten(self):
    return (
      (self.fwd_mask_info, self.dq_mask_info, self.dk_mask_info),
      self.kwargs,
    )

  @classmethod
  def tree_unflatten(cls, kwargs, values):
    fwd_mask_info, dq_mask_info, dk_mask_info = values
    dq_mask_info = (
      mask_info_lib.MaskInfo(*dq_mask_info) if dq_mask_info is not None else None
    )
    dk_mask_info = (
      mask_info_lib.MaskInfo(*dk_mask_info) if dk_mask_info is not None else None
    )
    return CEKernel(
      mask_info_lib.MaskInfo(*fwd_mask_info),
      dq_mask_info,
      dk_mask_info,
      **kwargs,
    )


def make_ce_kernel(
  mask: mask_lib.MultiHeadMask,
  *,
  block_sizes: BlockSizes | None = None,
  is_mqa: bool = False,
  mask_value: float = DEFAULT_MASK_VALUE,
  attn_logits_soft_cap: float | None = None,
  head_shards: int = 1,
  q_seq_shards: int = 1,
  interpret: bool = False,
) -> CEKernel:
  if len(mask.shape) != 3:
    raise ValueError(f"Unexpected mask shape: {mask.shape}")

  if block_sizes is None:
    block_sizes = BlockSizes.get_default()

  fwd_mask_info, mask_function = mask_info_lib._process_mask(
    mask,
    (block_sizes.block_q, block_sizes.block_kv),
    is_dkv=False,
    head_shards=head_shards,
    q_seq_shards=q_seq_shards,
  )
  fwd_mask_info = tree_util.tree_map(jnp.array, fwd_mask_info)

  dq_mask_info, _ = mask_info_lib._process_mask(
    mask,
    (block_sizes.block_q, block_sizes.block_kv),
    is_dkv=False,
    head_shards=head_shards,
    q_seq_shards=q_seq_shards,
  )
  dq_mask_info = tree_util.tree_map(jnp.array, dq_mask_info)

  dk_mask_info, _ = mask_info_lib._process_mask(
    mask,
    (block_sizes.block_q, block_sizes.block_kv),
    is_dkv=True,
    head_shards=head_shards,
    q_seq_shards=q_seq_shards,
  )
  dk_mask_info = tree_util.tree_map(jnp.array, dk_mask_info)

  return CEKernel(
    fwd_mask_info,
    dq_mask_info,
    dk_mask_info,
    block_sizes=block_sizes,
    is_mqa=is_mqa,
    mask_value=mask_value,
    attn_logits_soft_cap=attn_logits_soft_cap,
    mask_function=mask_function,
    interpret=interpret,
  )
