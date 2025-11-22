import copy

import jax
import jax.numpy as jnp
import pytest

from bmgpt.config import Config, DType, config_post_init, mesh_from_config
from bmgpt.model import (
  Transformer,
  _apply_rope,
  _attn,
  _make_cache_mask,
  _make_causal_mask,
  _precompute_rope_cossin,
  _transformer,
  init_kv_cache,
  init_transformer,
)


def test_manual_attn_matches_jax_no_kv():
  config = Config()
  config.sharding.data = []
  config.sharding.mesh_shape = [1]
  config.model.param_dtype = (
    DType.FLOAT32 if jax.default_backend() == "cpu" else DType.BFLOAT16
  )
  config.dataset.seq_len = 256
  config.dataset.num_vocab = 256
  config_no_fa = copy.deepcopy(config)
  config_no_fa.model.use_fa = False
  mesh = mesh_from_config(config)
  key = jax.random.key(config.experiment.seed)
  seq = jnp.arange(256)
  with jax.set_mesh(mesh):
    null_cache = init_kv_cache(config)[0]
    model = init_transformer(key, config)
    out, _ = _transformer(config, model, seq, null_cache, 0)
    out_no_fa, _ = _transformer(config_no_fa, model, seq, null_cache, 0)
  # print(jnp.max(jnp.abs(out - out_no_fa)))
  assert jnp.allclose(out, out_no_fa)


def test_rope():
  config = Config()
  config.sharding.data = []
  config.sharding.mesh_shape = [1]
  config.model.param_dtype = (
    DType.FLOAT32 if jax.default_backend() == "cpu" else DType.BFLOAT16
  )
  config.model.d_head = 4
  config.model.max_seq_len = 64
  mesh = mesh_from_config(config)
  with jax.set_mesh(mesh):
    cos, sin = _precompute_rope_cossin(config)
    x = jnp.arange(4)[None].astype(config.model.param_dtype.value)
    y = x[:, ::-1].astype(config.model.param_dtype.value)

    # Test: position zero does nothing
    xx = _apply_rope(config, cos, sin, jnp.array((0,)), x)
    yy = _apply_rope(config, cos, sin, jnp.array((0,)), y)
    assert x @ y.mT == xx @ yy.mT

    # Test: different position implements rotation in the way we expect
    pos0 = 3
    pos1 = 8
    xx = _apply_rope(config, cos, sin, jnp.array((pos0,)), x)
    yy = _apply_rope(config, cos, sin, jnp.array((pos1,)), y)

    def rot(c, s):
      return jnp.array(((c, -s), (s, c)))

    M00 = rot(cos[pos0, 0], sin[pos0, 0])
    M01 = rot(cos[pos0, 1], sin[pos0, 1])
    M10 = rot(cos[pos1, 0], sin[pos1, 0])
    M11 = rot(cos[pos1, 1], sin[pos1, 1])

    xxx = jnp.concatenate(
      (M00 @ jnp.array((x[0, 0], x[0, 2])), M01 @ jnp.array((x[0, 1], x[0, 3])))
    ).astype(config.model.param_dtype.value)
    yyy = jnp.concatenate(
      (M10 @ jnp.array((y[0, 0], y[0, 2])), M11 @ jnp.array((y[0, 1], y[0, 3])))
    ).astype(config.model.param_dtype.value)
  assert jnp.allclose(jnp.dot(xxx, yyy), (xx @ yy.mT)[0, 0])


def test_attention_masks():
  kv_cache_size = 16
  q_seq_len = 4
  k_seq_len = 4
  # Test vanilla causal mask
  causal_mask = _make_causal_mask(q_seq_len, k_seq_len, 0)
  # print(causal_mask)
  assert jnp.all(
    causal_mask == jnp.tril(jnp.ones((q_seq_len, k_seq_len)).astype(jnp.bool))
  )

  # Test Causal mask with used cache
  causal_mask = _make_causal_mask(q_seq_len, kv_cache_size, q_seq_len)
  stored_cache_mask = causal_mask[:, :q_seq_len]
  tril_mask = causal_mask[:, q_seq_len : 2 * q_seq_len]
  empty_cache_mask = causal_mask[:, 2 * q_seq_len :]
  assert jnp.all(stored_cache_mask == jnp.ones((q_seq_len, q_seq_len)).astype(jnp.bool))
  assert jnp.all(
    empty_cache_mask
    == jnp.zeros((q_seq_len, kv_cache_size - 2 * q_seq_len)).astype(jnp.bool)
  )
  assert jnp.all(
    tril_mask == jnp.tril(jnp.ones((q_seq_len, k_seq_len)).astype(jnp.bool))
  )

  # Test cache masking
  cache_mask = _make_cache_mask(q_seq_len, kv_cache_size, q_seq_len)
  stored_cache_mask = cache_mask[:, : 2 * q_seq_len]
  empty_cache_mask = cache_mask[:, 2 * q_seq_len :]
  assert jnp.all(
    stored_cache_mask == jnp.ones((q_seq_len, 2 * q_seq_len)).astype(jnp.bool)
  )
  assert jnp.all(
    empty_cache_mask
    == jnp.zeros((q_seq_len, kv_cache_size - 2 * q_seq_len)).astype(jnp.bool)
  )


def test_cache_correct_predictions():
  # Test we get same preds on new tokens if we have past tokens in cache vs context
  # NOTE: Test is not a great idea, as we don't have determinism from XLA-compiled...
  # see https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/
  config = Config()
  config.sharding.data = []
  config.sharding.mesh_shape = [1]
  config.model.param_dtype = (
    DType.FLOAT32 if jax.default_backend() == "cpu" else DType.BFLOAT16
  )
  config.model.num_layers = 1
  config.dataset.num_vocab = 8
  tol_args = {"atol": 1e-6 if jax.default_backend() == "cpu" else 1e-2}

  mesh = mesh_from_config(config)
  key = jax.random.key(config.experiment.seed)
  seq_len = 2
  seq = jnp.arange(seq_len)
  prefix, suffix = seq[: seq_len // 2], seq[seq_len // 2 :]
  with jax.set_mesh(mesh):
    model = init_transformer(key, config)
    null_cache = init_kv_cache(config)[0]
    out, _ = _transformer(config, model, seq, null_cache, 0)

    cache_init = init_kv_cache(config)[0]
    prefill, cache = _transformer(config, model, prefix, cache_init, 0)
    preds, cache_out = _transformer(config, model, suffix, cache, seq_len // 2)

  # print(out.shape)
  # print(prefill.shape)
  # print(preds.shape)
  # print(jnp.max(jnp.abs(prefill - out[: seq_len // 2, :])))
  # print(jnp.max(jnp.abs(preds - out[seq_len // 2 :, :])))

  assert jnp.allclose(prefill, out[: seq_len // 2, :], **tol_args)  # type: ignore[arg-type]
  assert jnp.allclose(preds, out[seq_len // 2 :, :], **tol_args)  # type: ignore[arg-type]
