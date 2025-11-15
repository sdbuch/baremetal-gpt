from functools import partial
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
from jax import Array
from jax.experimental.pallas.ops.tpu.splash_attention import (
    CausalMask,
    MultiHeadMask,
    make_splash_mha_single_device,
)

from bmgpt.config import Config

################################
# Architecture: building blocks
################################


class Mlp(NamedTuple):
    w_up: Array
    bias_up: Array
    w_down: Array
    bias_down: Array


def _mlp(config: Config, params: Mlp, x: Array):
    preact = jnp.dot(x, params.w_up, out_sharding=jax.P(*config.sharding_mlp_hidden))
    if config.use_bias_mlp:
        # PERF: check that compiler fuses these
        preact += params.bias_up
    act = jax.nn.gelu(preact, approximate=True)
    out = jnp.dot(act, params.w_down, out_sharding=jax.P(*config.sharding_res_stream))
    if config.use_bias_mlp:
        out += params.bias_down
    return out


class Attn(NamedTuple):
    w_qkv: Array
    w_o: Array


def _precompute_rope_cossin(config: Config):
    freqs = jnp.exp(
        -jnp.log(config.rope_theta).astype(config.compute_dtype.value)
        * 2
        / config.d_head
        * jnp.arange(config.d_head // 2, out_sharding=jax.P())
    )
    positions = jnp.arange(config.max_seq_len, out_sharding=jax.P())
    cycles = positions[:, None] * freqs[None, :]
    return jnp.cos(cycles), jnp.sin(cycles)


def _apply_rope(
    config: Config,
    cos: jax.Array,
    sin: jax.Array,
    positions: jax.Array,
    x: jax.Array,  # S x H
):
    x_1, x_2 = x[:, : config.d_head // 2], x[:, config.d_head // 2 :]
    c, s = cos.at[positions].get(), sin.at[positions].get()
    return jnp.concatenate(
        (c * x_1 - s * x_2, c * x_2 + s * x_1), axis=-1, dtype=config.param_dtype.value
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
        "sd,d3nh->3nsh",
        x_seq,
        params.w_qkv,
        out_sharding=jax.P(*config.sharding_att_qkv),
    )
    q, k, v = [qkv[i] for i in range(3)]
    s = q.shape[1]  # we save k shape later, after possibly prepending cache

    # Apply RoPE
    if config.use_rope:
        if not config.update_cache:
            cache_size = 0  # ignore passed value
        with jax.ensure_compile_time_eval():
            rope_cos, rope_sin = _precompute_rope_cossin(config)
            positions = jnp.arange(s, out_sharding=jax.P())
        _apply_rope_one_head = partial(
            _apply_rope, config, rope_cos, rope_sin, positions + cache_size
        )
        _apply_rope_all_heads = jax.vmap(_apply_rope_one_head)
        q, k = _apply_rope_all_heads(q), _apply_rope_all_heads(k)

    # Read/update/concatenate the cache
    if config.update_cache:
        k_cache, v_cache = kv_cache[0], kv_cache[1]
        k = jax.lax.dynamic_update_slice(k_cache, k, (0, cache_size, 0))
        v = jax.lax.dynamic_update_slice(v_cache, v, (0, cache_size, 0))
        kv_cache_out = jnp.concatenate((k[None], v[None]), axis=0)
    else:
        kv_cache_out = kv_cache
        cache_size = 0  # ignore passed value

    # Attention computation
    t = k.shape[1]
    if config.use_splash:
        # attn_out = jax.nn.dot_product_attention(q, k, v, scale=None, mask=mask)
        mask = MultiHeadMask([CausalMask((s, t)) for _ in range(config.num_heads)])
        attn_out = attn_fun(q, k, v)
    else:
        # Make mask
        mask = _make_causal_mask(s, t, cache_size)
        mask = mask & _make_cache_mask(s, t, cache_size)  # need for static cache
        mask = mask[None, ...]  # broadcast over heads
        # Scale and causal mask
        logits = jnp.einsum("nsh,nth->nst", q, k).astype(config.compute_dtype.value)
        logits *= 1.0 / config.d_head**0.5
        logits = jnp.where(mask, logits, -jnp.inf)
        probs = jax.nn.softmax(logits, axis=2)  # type: ignore[reportArgumentType]
        probs = probs.astype(config.param_dtype.value)
        attn_out = jnp.einsum("nst,nth->nsh", probs, v)
    out = jnp.einsum(
        "nsh,hnd->sd",
        attn_out,
        params.w_o,
        out_sharding=jax.P(*config.sharding_res_stream),
    )

    return out, kv_cache_out


class Embedding(NamedTuple):
    w: Array


def _embedding(config: Config, params: Embedding, token: jax.Array):
    emb = params.w.at[token].get(out_sharding=jax.P(*config.sharding_res_stream))
    return emb


class Unembedding(NamedTuple):
    w: Array


def _unembedding(config: Config, params: Unembedding, x: Array):
    logits = jnp.dot(x, params.w, out_sharding=jax.P(*config.sharding_res_stream))
    return logits


class LayerNorm(NamedTuple):
    gamma: Array
    beta: Array


def _layernorm(config: Config, params: LayerNorm, x: Array):
    x_std = jax.nn.standardize(
        x.astype(config.compute_dtype.value), epsilon=config.eps_ln
    )
    out = params.gamma * x_std.astype(config.param_dtype.value)
    if config.use_bias_ln:
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
    emb: Embedding
    blocks: Block  # vmapped at init
    unemb: Unembedding


def _transformer(
    config: Config,
    params: Transformer,
    tokens: Array,
    cache_in: jax.Array,
    cache_size: int = 0,
):
    x_seq = jax.vmap(partial(_embedding, config, params.emb))(tokens)

    def _block_fun(x_seq: Array, params__cache_in: tuple[Block, jax.Array]):
        params, cache_in = params__cache_in
        return _block(config, params, x_seq, cache_in, cache_size)

    out, cache_out = jax.lax.scan(_block_fun, x_seq, (params.blocks, cache_in))

    out = jax.vmap(partial(_unembedding, config, params.unemb))(out)

    return out, cache_out


#####################################
# Architecture: initialization/shapes
#####################################


def init_embedding(config: Config, key) -> Embedding:
    emb = config.param_std * jax.random.normal(
        key,
        (config.num_vocab, config.d_model),
        config.param_dtype.value,
        out_sharding=jax.P(*config.sharding_res_stream),
    )
    return Embedding(w=emb)


def init_unembedding(config: Config, key) -> Unembedding:
    unemb = config.param_std * jax.random.normal(
        key,
        (config.d_model, config.num_vocab),
        config.param_dtype.value,
        out_sharding=jax.P(*config.sharding_res_stream),
    )
    return Unembedding(w=unemb)


def init_mlp(config: Config, key) -> Mlp:
    k_w_up, k_w_down = jax.random.split(key, 2)
    w_up = config.param_std * jax.random.normal(
        k_w_up,
        (config.d_model, config.mlp_factor * config.d_model),
        config.param_dtype.value,
        out_sharding=jax.P(*config.sharding_wup),
    )
    w_down = config.param_std * (
        jax.random.normal(
            k_w_down,
            (config.mlp_factor * config.d_model, config.d_model),
            config.param_dtype.value,
            out_sharding=jax.P(*config.sharding_wdown),
        )
    )
    bias_up = jnp.zeros(
        (config.mlp_factor * config.d_model,),
        config.param_dtype.value,
        out_sharding=jax.P(*config.sharding_mlp_hidden),
    )
    bias_down = jnp.zeros(
        (config.d_model,),
        config.param_dtype.value,
        out_sharding=jax.P(*config.sharding_res_stream),
    )
    return Mlp(w_up=w_up, bias_up=bias_up, w_down=w_down, bias_down=bias_down)


def init_attn(config: Config, key) -> Attn:
    k_qkv, k_o = jax.random.split(key, 2)
    w_qkv = config.param_std * jax.random.normal(
        k_qkv,
        (config.d_model, 3, config.num_heads, config.d_head),
        config.param_dtype.value,
        out_sharding=jax.P(*config.sharding_wqkv),
    )
    w_out = config.param_std * (
        jax.random.normal(
            k_o,
            (config.d_head, config.num_heads, config.d_model),
            config.param_dtype.value,
            out_sharding=jax.P(*config.sharding_wo),
        )
    )
    return Attn(w_qkv=w_qkv, w_o=w_out)


def init_layernorm(config: Config) -> LayerNorm:
    gamma = jnp.ones(
        (config.d_model,),
        config.param_dtype.value,
        out_sharding=jax.P(*config.sharding_res_stream),
    )
    beta = jnp.zeros(
        (config.d_model,),
        config.param_dtype.value,
        out_sharding=jax.P(*config.sharding_res_stream),
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
    keys_blocks = jax.random.split(key_blocks, config.num_layers)
    model = Transformer(
        emb=init_embedding(config, key_emb),
        blocks=jax.vmap(partial(init_block, config))(keys_blocks),
        unemb=init_unembedding(config, key_unemb),
    )

    return model


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
        }
        return matrix_axes_dict.get(param_str, None)

    spec = jax.tree.map_with_path(lambda p, x: _make_spec_from_str(p), model)
    return spec


# TODO: update sharding if attention sharding is modified
def init_kv_cache(config: Config):
    if not config.sharding_data:
        sharding_batch_layer = [None, None]
    else:
        sharding_batch_layer = config.sharding_data + [None]
    sharding = jax.P(*(sharding_batch_layer + config.sharding_att_qkv))
    if config.update_cache:
        return jnp.zeros(
            (
                config.global_batch_size,
                config.num_layers,
                2,
                config.num_heads,
                config.max_seq_len,
                config.d_head,
            ),
            dtype=config.param_dtype.value,
            out_sharding=sharding,
        )
    else:
        # Save memory with a dummy cache, since its updates are no-ops
        return jnp.zeros(
            (config.global_batch_size, config.num_layers, 2, 1, 1, 1),
            dtype=config.param_dtype.value,
            out_sharding=sharding,
        )
