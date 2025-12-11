from functools import partial
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
from jax import Array
from jax.experimental.pallas.ops.tpu.splash_attention import (
  SegmentIds,
)

from bmgpt.config import Config, TransformerType

################################
# Architecture: building blocks
################################


class Mlp(NamedTuple):
  w_up: Array
  bias_up: Array
  w_gate: Array
  bias_gate: Array
  w_down: Array
  bias_down: Array


def init_mlp(config: Config, key) -> Mlp:
  k_w_up, k_w_down = jax.random.split(key, 2)
  d_hidden = int(config.model.mlp_factor * config.model.d_model)
  w_up = config.model.param_std * jax.random.normal(
    k_w_up,
    (config.model.d_model, d_hidden),
    config.model.param_dtype.value,
    out_sharding=jax.P(*config.sharding.wup),
  )
  w_gate = config.model.param_std * jax.random.normal(
    k_w_up,
    (config.model.d_model, d_hidden),
    config.model.param_dtype.value,
    out_sharding=jax.P(*config.sharding.wup),
  )
  w_down = config.model.param_std * (
    jax.random.normal(
      k_w_down,
      (d_hidden, config.model.d_model),
      config.model.param_dtype.value,
      out_sharding=jax.P(*config.sharding.wdown),
    )
  )
  bias_up = jnp.zeros(
    (d_hidden,),
    config.model.param_dtype.value,
    out_sharding=jax.P(*config.sharding.mlp_hidden),
  )
  bias_gate = jnp.zeros(
    (d_hidden,),
    config.model.param_dtype.value,
    out_sharding=jax.P(*config.sharding.mlp_hidden),
  )
  bias_down = jnp.zeros(
    (config.model.d_model,),
    config.model.param_dtype.value,
    out_sharding=jax.P(*config.sharding.res_stream),
  )
  return Mlp(
    w_up=w_up,
    bias_up=bias_up,
    w_gate=w_gate,
    bias_gate=bias_gate,
    w_down=w_down,
    bias_down=bias_down,
  )


def _mlp(config: Config, params: Mlp, x: Array):
  preact = jnp.matmul(x, params.w_up, out_sharding=jax.P(*config.sharding.mlp_hidden))
  if config.model.use_bias_mlp:
    # PERF: check that compiler fuses these
    preact += params.bias_up
  act = jax.nn.swish(preact)
  if config.model.use_gating_mlp:
    gate = jnp.matmul(x, params.w_gate, out_sharding=jax.P(*config.sharding.mlp_hidden))
    if config.model.use_bias_mlp:
      gate += params.bias_gate
    act *= gate
  out = jnp.matmul(act, params.w_down, out_sharding=jax.P(*config.sharding.res_stream))
  if config.model.use_bias_mlp:
    out += params.bias_down
  return out


class Attn(NamedTuple):
  w_qkv: Array
  w_o: Array


class CacheParams(NamedTuple):
  enabled: bool
  size: int


def init_attn(config: Config, key) -> Attn:
  k_qkv, k_o = jax.random.split(key, 2)
  w_qkv = config.model.param_std * jax.random.normal(
    k_qkv,
    (config.model.d_model, 3, config.model.num_heads, config.model.d_head),
    config.model.param_dtype.value,
    out_sharding=jax.P(*config.sharding.wqkv),
  )
  w_out = config.model.param_std * (
    jax.random.normal(
      k_o,
      (config.model.d_head, config.model.num_heads, config.model.d_model),
      config.model.param_dtype.value,
      out_sharding=jax.P(*config.sharding.wo),
    )
  )
  return Attn(w_qkv=w_qkv, w_o=w_out)


def _precompute_rope_cossin(config: Config):
  freqs = jnp.exp(
    -jnp.log(config.model.rope_theta).astype(config.model.compute_dtype.value)
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
  return jnp.concatenate(
    (c * x_1 - s * x_2, c * x_2 + s * x_1),
    axis=-1,
    dtype=config.model.param_dtype.value,
  )


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
  kernel,
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
    "sd,d3nh->3nsh",
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
  if kernel:
    q_segment_ids = jnp.zeros((s,))
    kv_mask = _make_cache_mask(s, t, cache_params.size) | (
      ~_make_cache_mask(s, t, cache_capacity)
    )
    kv_mask = ~kv_mask[0]
    kv_segment_ids = kv_mask.astype(jnp.int32)
    segment_ids = SegmentIds(q=q_segment_ids, kv=kv_segment_ids)

    splash_sharded, kernel = kernel
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
    logits = jnp.einsum("nsh,nth->nst", q, k).astype(config.model.compute_dtype.value)
    logits *= 1.0 / config.model.d_head**0.5
    logits = jnp.where(mask, logits, -jnp.inf)
    probs = jax.nn.softmax(logits, axis=2)  # type: ignore[reportArgumentType]
    probs = probs.astype(config.model.param_dtype.value)
    attn_out = jnp.einsum("nst,nth->nsh", probs, v)
  out = jnp.einsum(
    "nsh,hnd->sd",
    attn_out,
    params.w_o,
    out_sharding=jax.P(*config.sharding.res_stream),
  )

  return out, kv_cache_out


class EmbeddingDiscrete(NamedTuple):
  w: Array
  bias: Array


def init_embedding_discrete(config: Config, key) -> EmbeddingDiscrete:
  emb = config.model.param_std * jax.random.normal(
    key,
    (config.model.num_vocab, config.model.d_model),
    config.model.param_dtype.value,
    out_sharding=jax.P(*([None] + config.sharding.res_stream)),
  )
  bias = jnp.zeros(
    (config.model.d_model,),
    config.model.param_dtype.value,
    out_sharding=jax.P(*config.sharding.res_stream),
  )
  return EmbeddingDiscrete(w=emb, bias=bias)


def _embedding_discrete(config: Config, params: EmbeddingDiscrete, tokens: jax.Array):
  emb = params.w.at[tokens].get(out_sharding=jax.P(*config.sharding.res_stream))
  if config.model.use_bias_embeddings:
    emb += params.bias
  return emb


class EmbeddingContinuous(NamedTuple):
  w_emb: Array
  bias: Array
  w_pos: Array
  registers: Array


def init_embedding_continuous(config: Config, key) -> EmbeddingContinuous:
  key_emb, key_pos, key_reg = jax.random.split(key, 3)
  w_emb = config.model.param_std * jax.random.normal(
    key_emb,
    (config.model.num_vocab, config.model.d_model),
    config.model.param_dtype.value,
    out_sharding=jax.P(*([None] + config.sharding.res_stream)),
  )
  w_reg = config.model.param_std * jax.random.normal(
    key_reg,
    (config.model.num_registers, config.model.d_model),
    config.model.param_dtype.value,
    out_sharding=jax.P(*([None] + config.sharding.res_stream)),
  )
  w_pos = config.model.param_std * jax.random.normal(
    key_pos,
    (config.model.num_registers + config.model.max_seq_len, config.model.d_model),
    config.model.param_dtype.value,
    out_sharding=jax.P(*([None] + config.sharding.res_stream)),
  )
  bias = jnp.zeros(
    (config.model.d_model,),
    config.model.param_dtype.value,
    out_sharding=jax.P(*config.sharding.res_stream),
  )
  return EmbeddingContinuous(w_emb=w_emb, w_pos=w_pos, registers=w_reg, bias=bias)


def _embedding_continuous(config: Config, params: EmbeddingContinuous, seq: jax.Array):
  # seq is s x d_in
  emb_tokens = seq @ params.w_emb
  if config.model.use_bias_embeddings:
    emb_tokens += params.bias
  emb_with_regs = jnp.concatenate((params.registers, emb_tokens), axis=0)
  effective_seq_len = emb_with_regs.shape[0]
  emb = emb_with_regs + params.w_pos[:effective_seq_len]
  return emb


class LMHead(NamedTuple):
  w: Array
  bias: Array


def init_lm_head(config: Config, key) -> LMHead:
  unemb = config.model.param_std * jax.random.normal(
    key,
    (config.model.d_model, config.model.num_vocab),
    config.model.param_dtype.value,
    out_sharding=jax.P(*config.sharding.res_stream),
  )
  bias = jnp.zeros(
    (config.model.num_vocab,),
    config.model.param_dtype.value,
    out_sharding=jax.P(*config.sharding.res_stream),
  )
  return LMHead(w=unemb, bias=bias)


def _lm_head(config: Config, params: LMHead, x: Array):
  logits = jnp.matmul(x, params.w)
  if config.model.use_bias_embeddings:
    logits += params.bias
  return logits


class ClassificationHead(NamedTuple):
  w: Array
  bias: Array


def init_classification_head(config: Config, key) -> ClassificationHead:
  unemb = config.model.param_std * jax.random.normal(
    key,
    (config.model.d_model, config.model.num_classes),
    config.model.param_dtype.value,
    out_sharding=jax.P(*config.sharding.res_stream),
  )
  bias = jnp.zeros(
    (config.model.num_classes,),
    config.model.param_dtype.value,
    out_sharding=jax.P(*config.sharding.res_stream),
  )
  return ClassificationHead(w=unemb, bias=bias)


def _classification_head(config: Config, params: ClassificationHead, x: Array):
  logits = jnp.matmul(x[:1], params.w)
  if config.model.use_bias_embeddings:
    logits += params.bias
  return logits


class LayerNorm(NamedTuple):
  gamma: Array
  beta: Array


def init_layernorm(config: Config) -> LayerNorm:
  gamma = jnp.ones(
    (config.model.d_model,),
    config.model.param_dtype.value,
    out_sharding=jax.P(*config.sharding.res_stream),
  )
  beta = jnp.zeros(
    (config.model.d_model,),
    config.model.param_dtype.value,
    out_sharding=jax.P(*config.sharding.res_stream),
  )
  return LayerNorm(gamma=gamma, beta=beta)


def _layernorm(config: Config, params: LayerNorm, x: Array):
  """Naive three-pass layernorm algorithm. (layer/RMS norm, with/without bias)"""
  x = x.astype(config.model.compute_dtype.value)
  if config.model.use_centering_ln:
    x = x - x.mean()
  x = x * jax.lax.rsqrt(config.model.eps_ln + (x**2).mean())
  out = params.gamma * x.astype(config.model.param_dtype.value)
  if config.model.use_bias_ln:
    out += params.beta
  return out


##################################
# Architecture: derived components
##################################


class Block(NamedTuple):
  norm_attn: LayerNorm
  attn: Attn
  norm_mlp: LayerNorm
  mlp: Mlp


def init_block(config: Config, key) -> Block:
  key_attn, key_mlp = jax.random.split(key)
  return Block(
    norm_attn=init_layernorm(config),
    attn=init_attn(config, key_attn),
    norm_mlp=init_layernorm(config),
    mlp=init_mlp(config, key_mlp),
  )


def _block(
  config: Config,
  kernel,
  params: Block,
  x_seq: Array,
  cache: jax.Array,
  cache_params: CacheParams,
):
  att_skip = x_seq
  out = jax.vmap(partial(_layernorm, config, params.norm_attn))(x_seq)
  out, cache_out = _attn(
    config, kernel, params.attn, out, kv_cache=cache, cache_params=cache_params
  )
  out += att_skip

  mlp_skip = out
  out = jax.vmap(partial(_layernorm, config, params.norm_mlp))(out)
  out = jax.vmap(partial(_mlp, config, params.mlp))(out)
  out += mlp_skip

  return out, cache_out


class Transformer(NamedTuple):
  emb: EmbeddingDiscrete | EmbeddingContinuous
  blocks: Block  # vmapped at init
  unemb: LMHead | ClassificationHead


def init_transformer(key, config: Config) -> Transformer:
  # Make the full network
  key_emb, key_blocks, key_unemb = jax.random.split(key, 3)
  keys_blocks = jax.random.split(key_blocks, config.model.num_layers)
  init_embedding, init_unembedding, _, __ = transformer_variant_factory(config)
  model = Transformer(
    emb=init_embedding(config, key_emb),
    blocks=jax.vmap(partial(init_block, config))(keys_blocks),
    unemb=init_unembedding(config, key_unemb),
  )

  return model


def _transformer(
  config: Config,
  kernel,
  params: Transformer,
  tokens: Array,
  cache: jax.Array,
  cache_params: CacheParams,
):
  _, __, _embedding, _unembedding = transformer_variant_factory(config)
  x_seq = _embedding(config, params.emb, tokens)

  def _block_fun(x_seq: Array, params__cache_in: tuple[Block, jax.Array]):
    params, cache_in = params__cache_in
    return _block(config, kernel, params, x_seq, cache_in, cache_params)

  out, cache_out = jax.lax.scan(_block_fun, x_seq, (params.blocks, cache))

  out = _unembedding(config, params.unemb, out)

  return out, cache_out


#####################################
# Architecture: misc initialization/shapes
#####################################


# TODO: update sharding if attention sharding is modified
def init_kv_cache(config: Config, global_batch_size: int, cache_capacity: int):
  if not config.sharding.data:
    sharding_batch_layer = [None, None]
  else:
    sharding_batch_layer = config.sharding.data + [None]
  sharding = jax.P(*(sharding_batch_layer + config.sharding.att_qkv))

  return jnp.zeros(
    (
      global_batch_size,
      config.model.num_layers,
      2,
      config.model.num_heads,
      cache_capacity,
      config.model.d_head,
    ),
    dtype=config.model.param_dtype.value,
    out_sharding=sharding,
  )


#####################################
# Auxiliary
#####################################


def model_spec(model: Transformer) -> Any:
  # Make the spec (we need some way to pass metadata around)
  # HACK: not great... disadvantage of bare metal without custom pytree
  def _make_spec_from_str(path: str) -> tuple[int, int] | None:
    param_str = path[-1].__str__()
    matrix_axes_dict = {
      ".w_qkv": (-4, -1),
      ".w_o": (-3, -1),
      ".w_up": (-2, -1),
      ".w_gate": (-2, -1),
      ".w_down": (-2, -1),
      ".w": (-2, -1),
      ".w_emb": (-2, -1),
      ".w_pos": (-2, -1),
      ".w_reg": (-2, -1),  # TODO: weight decay registers?
    }
    return matrix_axes_dict.get(param_str, None)

  spec = jax.tree.map_with_path(lambda p, _: _make_spec_from_str(p), model)
  return spec


def transformer_variant_factory(config: Config):
  if config.model.transformer_type == TransformerType.DISCRETE:
    init_embedding = init_embedding_discrete
    fun_embedding = _embedding_discrete
    init_unembedding = init_lm_head
    fun_unembedding = _lm_head
  else:
    init_embedding = init_embedding_continuous
    fun_embedding = _embedding_continuous
    init_unembedding = init_classification_head
    fun_unembedding = _classification_head

  return (init_embedding, init_unembedding, fun_embedding, fun_unembedding)
