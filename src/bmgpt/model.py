from functools import partial
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
from jax import Array

from bmgpt.config import Config, TransformerType

################################
# Architecture: building blocks
################################


class Mlp(NamedTuple):
  w_up: Array
  bias_up: Array
  w_down: Array
  bias_down: Array


def _mlp(config: Config, params: Mlp, x: Array):
  preact = jnp.matmul(x, params.w_up, out_sharding=jax.P(*config.sharding.mlp_hidden))
  if config.model.use_bias_mlp:
    # PERF: check that compiler fuses these
    preact += params.bias_up
  act = jax.nn.gelu(preact, approximate=True)
  out = jnp.matmul(act, params.w_down, out_sharding=jax.P(*config.sharding.res_stream))
  if config.model.use_bias_mlp:
    out += params.bias_down
  return out


class Attn(NamedTuple):
  w_qkv: Array
  w_o: Array


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
  return k_positions[None, :] < cache_size + seq_len_q


def _attn(
  config: Config,
  params: Attn,
  x_seq: jax.Array,
  kv_cache: jax.Array,  # 2 x config.max_seq_len x n x h (see below)
  cache_size: int,
):
  # s: sequence length
  # d: embedding dim (config.d_model)
  # n: attention heads (config.num_heads)
  # h: head dim (config.d_head)
  # x_seq: s x d

  qkv = jnp.einsum(
    "sd,d3nh->3snh",
    x_seq,
    params.w_qkv,
    out_sharding=jax.P(*config.sharding.att_qkv),
  )
  q, k, v = [qkv[i] for i in range(3)]
  s = q.shape[0]  # we save k shape later, after possibly prepending cache

  # Apply RoPE
  if config.model.use_rope:
    with jax.ensure_compile_time_eval():
      rope_cos, rope_sin = _precompute_rope_cossin(config)
      positions = jnp.arange(s, out_sharding=jax.P())
    _apply_rope_one_head = partial(
      _apply_rope, config, rope_cos, rope_sin, positions + cache_size
    )
    _apply_rope_all_heads = jax.vmap(_apply_rope_one_head, in_axes=1, out_axes=1)
    q, k = _apply_rope_all_heads(q), _apply_rope_all_heads(k)

  # Read/update/concatenate the cache
  if cache_size >= 0:
    k_cache, v_cache = kv_cache[0], kv_cache[1]
    k = jax.lax.dynamic_update_slice(k_cache, k, (cache_size, 0, 0))
    v = jax.lax.dynamic_update_slice(v_cache, v, (cache_size, 0, 0))
    kv_cache_out = jnp.concatenate((k[None], v[None]), axis=0)
  else:
    kv_cache_out = kv_cache

  # Attention computation
  t = k.shape[0]
  if config.model.is_causal:
    mask = _make_causal_mask(s, t, cache_size)
    mask = mask & _make_cache_mask(s, t, cache_size)  # need for static cache
  else:
    mask = _make_cache_mask(s, t, 0)  # full attention
  mask = mask[None, ...]  # broadcast over heads
  if config.model.use_fa:
    attn_out = jax.nn.dot_product_attention(q, k, v, scale=None, mask=mask)
  else:
    logits = jnp.einsum("snh,tnh->nst", q, k).astype(config.model.compute_dtype.value)
    # Scale and causal mask
    logits *= 1.0 / config.model.d_head**0.5
    logits = jnp.where(mask, logits, -jnp.inf)
    probs = jax.nn.softmax(logits, axis=2)  # type: ignore[reportArgumentType]
    probs = probs.astype(config.model.param_dtype.value)
    attn_out = jnp.einsum("nst,tnh->snh", probs, v)
  out = jnp.einsum(
    "snh,hnd->sd",
    attn_out,
    params.w_o,
    out_sharding=jax.P(*config.sharding.res_stream),
  )

  return out, kv_cache_out


class EmbeddingDiscrete(NamedTuple):
  w: Array
  bias: Array


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


def _lm_head(config: Config, params: LMHead, x: Array):
  logits = jnp.matmul(x, params.w)
  if config.model.use_bias_embeddings:
    logits += params.bias
  return logits


class ClassificationHead(NamedTuple):
  w: Array
  bias: Array


def _classification_head(config: Config, params: ClassificationHead, x: Array):
  logits = jnp.matmul(x[0], params.w)
  if config.model.use_bias_embeddings:
    logits += params.bias
  return logits


class LayerNorm(NamedTuple):
  gamma: Array
  beta: Array


def _layernorm(config: Config, params: LayerNorm, x: Array):
  x_std = jax.nn.standardize(
    x.astype(config.model.compute_dtype.value), epsilon=config.model.eps_ln
  )
  out = params.gamma * x_std.astype(config.model.param_dtype.value)
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


def _block(
  config: Config,
  params: Block,
  x_seq: Array,
  cache_in: jax.Array,
  cache_size: int,
):
  att_skip = x_seq
  out = jax.vmap(partial(_layernorm, config, params.norm_attn))(x_seq)
  out, cache_out = _attn(
    config, params.attn, out, kv_cache=cache_in, cache_size=cache_size
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


def _transformer(
  config: Config,
  params: Transformer,
  tokens: Array,
  cache_in: jax.Array,
  cache_size: int,
):
  _, __, _embedding, _unembedding = transformer_variant_factory(config)
  x_seq = _embedding(config, params.emb, tokens)

  def _block_fun(x_seq: Array, params__cache_in: tuple[Block, jax.Array]):
    params, cache_in = params__cache_in
    return _block(config, params, x_seq, cache_in, cache_size)

  out, cache_out = jax.lax.scan(_block_fun, x_seq, (params.blocks, cache_in))

  out = _unembedding(config, params.unemb, out)

  return out, cache_out


#####################################
# Architecture: initialization/shapes
#####################################


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


def init_mlp(config: Config, key) -> Mlp:
  k_w_up, k_w_down = jax.random.split(key, 2)
  w_up = config.model.param_std * jax.random.normal(
    k_w_up,
    (config.model.d_model, config.model.mlp_factor * config.model.d_model),
    config.model.param_dtype.value,
    out_sharding=jax.P(*config.sharding.wup),
  )
  w_down = config.model.param_std * (
    jax.random.normal(
      k_w_down,
      (config.model.mlp_factor * config.model.d_model, config.model.d_model),
      config.model.param_dtype.value,
      out_sharding=jax.P(*config.sharding.wdown),
    )
  )
  bias_up = jnp.zeros(
    (config.model.mlp_factor * config.model.d_model,),
    config.model.param_dtype.value,
    out_sharding=jax.P(*config.sharding.mlp_hidden),
  )
  bias_down = jnp.zeros(
    (config.model.d_model,),
    config.model.param_dtype.value,
    out_sharding=jax.P(*config.sharding.res_stream),
  )
  return Mlp(w_up=w_up, bias_up=bias_up, w_down=w_down, bias_down=bias_down)


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


def init_block(config: Config, key) -> Block:
  key_attn, key_mlp = jax.random.split(key)
  return Block(
    norm_attn=init_layernorm(config),
    attn=init_attn(config, key_attn),
    norm_mlp=init_layernorm(config),
    mlp=init_mlp(config, key_mlp),
  )


def init_model(key, config: Config) -> Transformer:
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


# TODO: update sharding if attention sharding is modified
def init_kv_cache(config: Config, global_batch_size: int, update_cache: bool):
  if not config.sharding.data:
    sharding_batch_layer = [None, None]
  else:
    sharding_batch_layer = config.sharding.data + [None]
  sharding = jax.P(*(sharding_batch_layer + config.sharding.att_qkv))

  if not update_cache:
    # Save memory if we aren't updating the cache
    dummy_cache = jnp.zeros(
      (global_batch_size, config.model.num_layers, 2, 1, 1, 1),
      dtype=config.model.param_dtype.value,
      out_sharding=sharding,
    )
    return dummy_cache
  return jnp.zeros(
    (
      global_batch_size,
      config.model.num_layers,
      2,
      config.model.max_seq_len,
      config.model.num_heads,
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
      ".w_down": (-2, -1),
      ".w": (-2, -1),
      ".w_emb": (-2, -1),
      ".w_pos": (-2, -1),
      ".w_reg": (-2, -1),
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
