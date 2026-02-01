from dataclasses import dataclass, field
from functools import partial, singledispatch
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
from jax import Array
from jax.experimental.pallas.ops.tpu.splash_attention import (
  SegmentIds,
)

from bmgpt.config import Config, TransformerType

################################
#         Helpers
################################


def init_array_normal(config: Config, shape, sharding, key):
  return config.model.param_std * jax.random.normal(
    key, shape, config.model.param_dtype.value, out_sharding=jax.P(*sharding)
  )


def init_array_zeros(config: Config, shape, sharding):
  return jnp.zeros(shape, config.model.param_dtype.value, out_sharding=jax.P(*sharding))


def init_array_ones(config: Config, shape, sharding):
  return jnp.ones(shape, config.model.param_dtype.value, out_sharding=jax.P(*sharding))


################################
# Architecture: building blocks
################################


@jax.tree_util.register_dataclass
@dataclass
class ArrayWithMetadata:
  p: Array
  matmul_dims: tuple[tuple, tuple] = field(metadata=dict(static=True))

  # Common arithmetic ops (useful for optimizer updates)
  def __add__(self, other):
    other_p = other.p if isinstance(other, ArrayWithMetadata) else other
    return ArrayWithMetadata(self.p + other_p, self.matmul_dims)


@jax.tree_util.register_dataclass
@dataclass
class ParamNodeWithMetadata:
  hooks: dict = field(default_factory=dict, metadata=dict(static=True), kw_only=True)


@jax.tree_util.register_dataclass
@dataclass
class Mlp(ParamNodeWithMetadata):
  w_up: ArrayWithMetadata
  bias_up: ArrayWithMetadata
  w_down: ArrayWithMetadata
  bias_down: ArrayWithMetadata

  @classmethod
  def init(cls, config: Config, key):
    k_up, k_down = jax.random.split(key, 2)
    d_hidden = int(config.model.mlp_factor * config.model.d_model)
    shape_w_up = (d_hidden, config.model.d_model)
    shape_bias_up = (d_hidden,)
    sharding_w_up = config.sharding.wup
    sharding_bias_up = config.sharding.mlp_hidden
    if config.model.use_gating_mlp:
      shape_w_up = (2,) + shape_w_up
      shape_bias_up = (2,) + shape_bias_up
      sharding_w_up = [None] + sharding_w_up
      sharding_bias_up = [None] + sharding_bias_up
    shape_w_down = (config.model.d_model, d_hidden)
    shape_bias_down = (config.model.d_model,)

    w_up = init_array_normal(config, shape_w_up, sharding_w_up, k_up)
    bias_up = init_array_zeros(config, shape_bias_up, sharding_bias_up)
    w_down = init_array_normal(config, shape_w_down, config.sharding.wdown, k_down)
    bias_down = init_array_zeros(config, shape_bias_down, config.sharding.res_stream)

    return cls(
      w_up=ArrayWithMetadata(w_up, ((-1,), (-2,))),
      bias_up=ArrayWithMetadata(bias_up, ((), ())),
      w_down=ArrayWithMetadata(w_down, ((-1,), (-2,))),
      bias_down=ArrayWithMetadata(bias_down, ((), ())),
    )


def _mlp(config: Config, params: Mlp, x: Array):
  mlp_out_spec = jax.P(*config.sharding.res_stream)
  if config.model.use_gating_mlp:
    mlp_hidden_spec = jax.P(*([None] + config.sharding.mlp_hidden))
    preact__gate = jnp.matmul(x, params.w_up.p.mT, out_sharding=mlp_hidden_spec)
    if config.model.use_bias_mlp:
      preact__gate += params.bias_up.p
    preact, gate = preact__gate
    act = jax.nn.swish(preact) * gate
  else:
    mlp_hidden_spec = jax.P(*config.sharding.mlp_hidden)
    preact = jnp.matmul(x, params.w_up.p.mT, out_sharding=mlp_hidden_spec)
    if config.model.use_bias_mlp:
      preact += params.bias_up.p
    act = jax.nn.swish(preact)
  out = jnp.matmul(act, params.w_down.p.mT, out_sharding=mlp_out_spec)
  if config.model.use_bias_mlp:
    out += params.bias_down.p
  assert out.dtype == config.model.compute_dtype.value
  return out


class CacheParams(NamedTuple):
  enabled: bool
  size: int


@jax.tree_util.register_dataclass
@dataclass
class Attn(ParamNodeWithMetadata):
  w_qkv: ArrayWithMetadata
  w_o: ArrayWithMetadata

  @classmethod
  def init(cls, config: Config, key):
    k_qkv, k_o = jax.random.split(key, 2)
    shape_qkv = (3, config.model.num_heads, config.model.d_head, config.model.d_model)
    shape_wout = (config.model.d_model, config.model.num_heads, config.model.d_head)
    w_qkv = init_array_normal(config, shape_qkv, config.sharding.wqkv, k_qkv)
    w_out = init_array_normal(config, shape_wout, config.sharding.wo, k_o)
    return cls(
      w_qkv=ArrayWithMetadata(w_qkv, ((-1,), (-2, -3))),
      w_o=ArrayWithMetadata(w_out, ((-1, -2), (-3,))),
    )


def _precompute_rope_cossin(config: Config):
  freqs = jnp.exp(
    -jnp.log(config.model.rope_theta).astype(jnp.float32)
    * 2
    / config.model.d_head
    * jnp.arange(config.model.d_head // 2, out_sharding=jax.P())
  )
  positions = jnp.arange(config.model.max_seq_len, out_sharding=jax.P())
  cycles = positions[:, None] * freqs[None, :]
  return jnp.cos(cycles), jnp.sin(cycles)


def _apply_rope(
  config: Config,
  cos: jax.Array,
  sin: jax.Array,
  positions: jax.Array,
  x: jax.Array,  # S x H
):
  x_1, x_2 = x[:, : config.model.d_head // 2], x[:, config.model.d_head // 2 :]
  c, s = cos.at[positions].get(), sin.at[positions].get()
  return jnp.concatenate((c * x_1 - s * x_2, c * x_2 + s * x_1), axis=-1, dtype=x.dtype)


def _make_causal_mask(seq_len_q: int, seq_len_k: int, cache_size: int) -> Array:
  """(seq_len_q, seq_len_k) bool array with True for past token positions"""
  q_positions = cache_size + jnp.arange(seq_len_q)
  k_positions = jnp.arange(seq_len_k)
  return q_positions[:, None] >= k_positions[None, :]


def _make_cache_mask(seq_len_q: int, seq_len_k: int, cache_size: int) -> Array:
  """(1, seq_len_k) bool array with True for actual cache+context positions"""
  k_positions = jnp.arange(seq_len_k)
  return k_positions[None, :] < cache_size


def _attn(
  config: Config,
  shard_mapped__kernel,
  params: Attn,
  x_seq: jax.Array,
  kv_cache: jax.Array,  # 2 x cache_capacity x n x h
  cache_params: CacheParams,
):
  # s: sequence length
  # d: embedding dim (config.d_model)
  # n: attention heads (config.num_heads)
  # h: head dim (config.d_head)
  # x_seq: s x d

  q, k, v = jnp.einsum(
    "sd,3nhd->3nsh",
    x_seq,
    params.w_qkv,
    out_sharding=jax.P(*config.sharding.att_qkv),
  )
  s = q.shape[1]

  # Cache + RoPE scheme: we update the cache after applying RoPE to K,
  #  so new Q/K just needs to RoPE with positions + cache_size!
  if config.model.use_rope:
    with jax.ensure_compile_time_eval():
      rope_cos, rope_sin = _precompute_rope_cossin(config)
      positions = jnp.arange(s, out_sharding=jax.P())
    _apply_rope_one_head = partial(
      _apply_rope, config, rope_cos, rope_sin, positions + cache_params.size
    )
    _apply_rope_all_heads = jax.vmap(_apply_rope_one_head)
    q, k = _apply_rope_all_heads(q), _apply_rope_all_heads(k)

  k_cache, v_cache = kv_cache[0], kv_cache[1]
  if cache_params.enabled:
    k_cache_out = jax.lax.dynamic_update_slice(k_cache, k, (0, cache_params.size, 0))
    v_cache_out = jax.lax.dynamic_update_slice(v_cache, v, (0, cache_params.size, 0))
    kv_cache_out = jnp.concatenate((k_cache_out[None], v_cache_out[None]), axis=0)
  else:
    kv_cache_out = kv_cache
  # Cache read scheme: to enable same mask for the same s value (Q seq len),
  #  we concatenate the full cache to K, and mask empty entries
  # For efficient training, set cache size zero + cache_params.enabled=False
  cache_capacity = k_cache.shape[-2]
  k = jnp.concatenate((k_cache, k), axis=1)
  v = jnp.concatenate((v_cache, v), axis=1)

  # Attention
  t = k.shape[1]  # t = s + cache_capacity
  if shard_mapped__kernel:
    q_segment_ids = jnp.zeros((s,))
    kv_mask = _make_cache_mask(s, t, cache_params.size) | (
      ~_make_cache_mask(s, t, cache_capacity)
    )
    kv_mask = ~kv_mask[0]
    kv_segment_ids = kv_mask.astype(jnp.int32)
    segment_ids = SegmentIds(q=q_segment_ids, kv=kv_segment_ids)

    splash_sharded, kernel = shard_mapped__kernel
    attn_out = splash_sharded(
      kernel,
      q / config.model.d_head**0.25,
      k / config.model.d_head**0.25,
      v,
      segment_ids,
    )
  else:
    # Make mask
    if config.model.is_causal:
      mask = _make_causal_mask(s, t, cache_capacity)
      cache_mask = _make_cache_mask(s, t, cache_params.size) | (
        ~_make_cache_mask(s, t, cache_capacity)
      )
      mask = mask & cache_mask
    else:
      mask = ~_make_cache_mask(s, t, 0)  # full attention
    mask = mask[None, ...]  # broadcast over heads
    # Scale and causal mask
    logits = jnp.einsum("nsh,nth->nst", q, k, preferred_element_type=jnp.float32)
    logits *= 1.0 / config.model.d_head**0.5
    logits = jnp.where(mask, logits, -jnp.inf)
    probs = jax.nn.softmax(logits, axis=2)  # type: ignore[reportArgumentType]
    probs = probs.astype(q.dtype)
    attn_out = jnp.einsum("nst,nth->nsh", probs, v)
  out = jnp.einsum(
    "nsh,dnh->sd",
    attn_out,
    params.w_o,
    out_sharding=jax.P(*config.sharding.res_stream),
  )

  assert out.dtype == config.model.compute_dtype.value
  assert kv_cache_out.dtype == config.model.compute_dtype.value
  return out, kv_cache_out


@jax.tree_util.register_dataclass
@dataclass
class Embedding(ParamNodeWithMetadata):
  @classmethod
  def init(cls, config: Config, key) -> "Embedding":
    raise NotImplementedError


@singledispatch
def _embedding_interface(params, seq: Array, config: Config) -> Array:
  raise NotImplementedError(f"No embedding impl for {type(params)}")


def _embedding(config: Config, params: Embedding, seq: jax.Array):
  return _embedding_interface(params, seq, config)


@jax.tree_util.register_dataclass
@dataclass
class EmbeddingDiscrete(Embedding):
  w: ArrayWithMetadata
  bias: ArrayWithMetadata

  @classmethod
  def init(cls, config: Config, key) -> "EmbeddingDiscrete":
    shape_emb = (config.model.num_vocab, config.model.d_model)
    shape_bias = (config.model.d_model,)
    emb = init_array_normal(config, shape_emb, config.sharding.wemb, key)
    bias = init_array_zeros(config, shape_bias, config.sharding.res_stream)
    return cls(
      w=ArrayWithMetadata(emb, ((), ())), bias=ArrayWithMetadata(bias, ((), ()))
    )


@_embedding_interface.register(EmbeddingDiscrete)
def _(params: EmbeddingDiscrete, seq: Array, config: Config) -> Array:
  w_f32 = params.w.p.astype(jnp.float32)  # perform the bwd scatter-add in fp32
  emb = w_f32.at[seq].get(out_sharding=jax.P(*config.sharding.res_stream))
  emb = emb.astype(params.w.p.dtype)
  if config.model.use_bias_embeddings:
    emb += params.bias.p
  assert emb.dtype == config.model.compute_dtype.value
  return emb


@jax.tree_util.register_dataclass
@dataclass
class EmbeddingContinuous(Embedding):
  w_emb: ArrayWithMetadata
  bias: ArrayWithMetadata
  w_pos: ArrayWithMetadata
  registers: ArrayWithMetadata

  @classmethod
  def init(cls, config: Config, key) -> "EmbeddingContinuous":
    key_emb, key_pos, key_reg = jax.random.split(key, 3)
    shape_emb = (config.model.num_vocab, config.model.d_model)
    shape_reg = (config.model.num_registers, config.model.d_model)
    shape_pos = (
      config.model.num_registers + config.model.max_seq_len,
      config.model.d_model,
    )
    shape_bias = (config.model.d_model,)
    w_emb = init_array_normal(config, shape_emb, config.sharding.wemb, key_emb)
    w_reg = init_array_normal(
      config, shape_reg, [None] + config.sharding.res_stream, key_reg
    )
    w_pos = init_array_normal(
      config, shape_pos, [None] + config.sharding.res_stream, key_pos
    )
    bias = init_array_zeros(config, shape_bias, config.sharding.res_stream)
    return cls(
      w_emb=ArrayWithMetadata(w_emb, ((-2,), (-1,))),
      bias=ArrayWithMetadata(bias, ((), ())),
      w_pos=ArrayWithMetadata(w_pos, ((), ())),
      registers=ArrayWithMetadata(w_reg, ((), ())),
    )


@_embedding_interface.register(EmbeddingContinuous)
def _(params: EmbeddingContinuous, seq: Array, config: Config) -> Array:
  # seq is s x d_in
  emb_tokens = seq @ params.w_emb.p
  if config.model.use_bias_embeddings:
    emb_tokens += params.bias.p
  emb_with_regs = jnp.concatenate((params.registers.p, emb_tokens), axis=0)
  effective_seq_len = emb_with_regs.shape[0]
  emb = emb_with_regs + params.w_pos.p[:effective_seq_len]
  assert emb.dtype == config.model.compute_dtype.value
  return emb


@jax.tree_util.register_dataclass
@dataclass
class Unembedding(ParamNodeWithMetadata):
  @classmethod
  def init(cls, config: Config, key) -> "Unembedding":
    raise NotImplementedError


@singledispatch
def _unembedding_interface(params, seq: Array, config: Config) -> Array:
  raise NotImplementedError(f"No unembedding impl for {type(params)}")


def _unembedding(config: Config, params: Unembedding, seq: jax.Array):
  return _unembedding_interface(params, seq, config)


@jax.tree_util.register_dataclass
@dataclass
class LMHead(Unembedding):
  w: ArrayWithMetadata
  bias: ArrayWithMetadata

  @classmethod
  def init(cls, config: Config, key) -> "LMHead":
    shape_emb = (config.model.num_vocab, config.model.d_model)
    shape_bias = (config.model.num_vocab,)
    unemb = init_array_normal(config, shape_emb, config.sharding.wunemb, key)
    bias = init_array_zeros(config, shape_bias, [])
    return cls(
      w=ArrayWithMetadata(unemb, ((-1,), (-2,))), bias=ArrayWithMetadata(bias, ((), ()))
    )


@_unembedding_interface.register(LMHead)
def _(params: LMHead, x: Array, config: Config) -> Array:
  logits = jnp.matmul(x, params.w.p.mT, preferred_element_type=jnp.float32)
  if config.model.use_bias_embeddings:
    logits += params.bias.p
  assert logits.dtype == jnp.float32
  return logits


@jax.tree_util.register_dataclass
@dataclass
class ClassificationHead(Unembedding):
  w: ArrayWithMetadata
  bias: ArrayWithMetadata

  @classmethod
  def init(cls, config: Config, key) -> "ClassificationHead":
    shape_emb = (config.model.num_classes, config.model.d_model)
    shape_bias = (config.model.num_classes,)
    unemb = init_array_normal(config, shape_emb, config.sharding.wunemb, key)
    bias = init_array_zeros(config, shape_bias, [])
    return cls(
      w=ArrayWithMetadata(unemb, ((-1,), (-2,))), bias=ArrayWithMetadata(bias, ((), ()))
    )


@_unembedding_interface.register(ClassificationHead)
def _(params: ClassificationHead, x: Array, config: Config):
  """Input x has shape (S, D)"""
  logits = jnp.matmul(x[:1], params.w.p, preferred_element_type=jnp.float32)
  if config.model.use_bias_embeddings:
    logits += params.bias.p
  assert logits.dtype == jnp.float32
  return logits


@jax.tree_util.register_dataclass
@dataclass
class LayerNorm(ParamNodeWithMetadata):
  gamma: ArrayWithMetadata
  beta: ArrayWithMetadata

  @classmethod
  def init(cls, config: Config) -> "LayerNorm":
    shape = (config.model.d_model,)
    gamma = init_array_ones(config, shape, config.sharding.res_stream)
    beta = init_array_zeros(config, shape, config.sharding.res_stream)
    return cls(
      gamma=ArrayWithMetadata(gamma, ((), ())), beta=ArrayWithMetadata(beta, ((), ()))
    )


def _layernorm(config: Config, params: LayerNorm, x: Array):
  """Naive three-pass layernorm algorithm. (layer/RMS norm, with/without bias)"""
  out = x.astype(jnp.float32)
  if config.model.use_centering_ln:
    out = out - out.mean()
  out = out * jax.lax.rsqrt(config.model.eps_ln + (out**2).mean())
  out = params.gamma.p * out
  if config.model.use_bias_ln:
    out += params.beta.p
  out = out.astype(x.dtype)
  assert out.dtype == config.model.compute_dtype.value
  return out


##################################
# Architecture: derived components
##################################


@jax.tree_util.register_dataclass
@dataclass
class Block(ParamNodeWithMetadata):
  norm_attn: LayerNorm
  attn: Attn
  norm_mlp: LayerNorm
  mlp: Mlp

  @classmethod
  def init(cls, config: Config, key) -> "Block":
    key_attn, key_mlp = jax.random.split(key)
    return cls(
      norm_attn=LayerNorm.init(config),
      attn=Attn.init(config, key_attn),
      norm_mlp=LayerNorm.init(config),
      mlp=Mlp.init(config, key_mlp),
    )


def _block(
  config: Config,
  shard_mapped__kernel,
  params: Block,
  x_seq: Array,
  cache: jax.Array,
  cache_params: CacheParams,
):
  att_skip = x_seq
  out = jax.vmap(partial(_layernorm, config, params.norm_attn))(x_seq)
  out, cache_out = _attn(
    config,
    shard_mapped__kernel,
    params.attn,
    out,
    kv_cache=cache,
    cache_params=cache_params,
  )
  out += att_skip

  mlp_skip = out
  out = jax.vmap(partial(_layernorm, config, params.norm_mlp))(out)
  out = jax.vmap(partial(_mlp, config, params.mlp))(out)
  out += mlp_skip

  return out, cache_out


@jax.tree_util.register_dataclass
@dataclass
class Transformer(ParamNodeWithMetadata):
  emb: Embedding
  blocks: Block  # vmapped at init
  unemb: Unembedding

  @classmethod
  def init(cls, config: Config, key) -> "Transformer":
    if config.model.transformer_type == TransformerType.DISCRETE:
      embedding_cls = EmbeddingDiscrete
      unembedding_cls = LMHead
    else:
      embedding_cls = EmbeddingContinuous
      unembedding_cls = ClassificationHead
    key_emb, key_blocks, key_unemb = jax.random.split(key, 3)
    keys_blocks = jax.random.split(key_blocks, config.model.num_layers)
    init_block = lambda k: Block.init(config, k)
    model = cls(
      emb=embedding_cls.init(config, key_emb),
      blocks=jax.vmap(init_block)(keys_blocks),
      unemb=unembedding_cls.init(config, key_unemb),
    )
    return model


def _transformer(
  config: Config,
  shard_mapped__kernel,
  params: Transformer,
  tokens: Array,
  cache: jax.Array,
  cache_params: CacheParams,
):
  x_seq = _embedding(config, params.emb, tokens)

  @partial(jax.remat, policy=jax.checkpoint_policies.dots_with_no_batch_dims_saveable)
  def _block_fun(x_seq: Array, params__cache_in: tuple[Block, jax.Array]):
    params, cache_in = params__cache_in
    return _block(config, shard_mapped__kernel, params, x_seq, cache_in, cache_params)

  out, cache_out = jax.lax.scan(_block_fun, x_seq, (params.blocks, cache))

  return out, cache_out


#####################################
# Architecture: misc initialization/shapes
#####################################


def init_kv_cache(
  config: Config, global_batch_size: int, num_microbatches: int, cache_capacity: int
):
  """Cache is shape (B//num_micro, L, 2, N, max_seq_len, H)"""
  if not config.sharding.data:
    sharding_batch_layer = [None, None]
  else:
    sharding_batch_layer = config.sharding.data + [None]
  sharding = jax.P(*(sharding_batch_layer + config.sharding.att_qkv))

  return jnp.zeros(
    (
      global_batch_size // num_microbatches,
      config.model.num_layers,
      2,
      config.model.num_heads,
      cache_capacity,
      config.model.d_head,
    ),
    dtype=config.model.compute_dtype.value,
    out_sharding=sharding,
  )


#####################################
# Auxiliary
#####################################


def model_spec(model: Transformer) -> Any:
  # Make the spec (we need some way to pass metadata around)
  # HACK: not great... disadvantage of bare metal without custom pytree
  # TODO: Check and update these later... reasonable to do (in, out)?
  def _make_spec_from_str(path: str) -> tuple[int, int] | None:
    param_str = path[-1].__str__()
    matrix_axes_dict = {
      ".w_qkv": (-1, -2),
      ".w_o": (-1, -3),
      ".w_up": (-1, -2),
      ".w_gate": (-1, -2),
      ".w_down": (-1, -2),
      ".w": (-2, -1),
      ".w_emb": (-2, -1),
      ".w_pos": (-2, -1),
      ".w_reg": (-2, -1),  # TODO: weight decay registers?
    }
    return matrix_axes_dict.get(param_str, None)

  spec = jax.tree.map_with_path(lambda p, _: _make_spec_from_str(p), model)
  return spec
