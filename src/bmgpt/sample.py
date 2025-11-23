from functools import partial

import jax
import jax.numpy as jnp

from bmgpt.config import Config
from bmgpt.model import CacheParams, Transformer, _transformer


@partial(jax.jit, static_argnums=(2,), donate_argnums=(5,))
def sample_one_token(
  config: Config,
  key,
  kernel,
  params: Transformer,
  seq: jax.Array,
  cache_in: jax.Array,
  cache_params: CacheParams,
  temperature: float,
):
  """Expects seq and cache_in to have no batch axis."""
  y, cache_out = _transformer(config, kernel, params, seq, cache_in, cache_params)
  logits = y.astype(config.model.compute_dtype.value)
  cache_params = CacheParams(cache_params.enabled, cache_params.size + seq.shape[-1])
  next_token = jnp.array((jax.random.categorical(key, logits[-1] / temperature),))
  return next_token, cache_out, cache_params


@partial(jax.jit, static_argnums=(2,), donate_argnums=(5,))
def generate(
  config: Config,
  key,
  kernel,
  params: Transformer,
  prompt: jax.Array,
  cache: jax.Array,
  cache_params: CacheParams,
) -> tuple[jax.Array, jax.Array, CacheParams]:
  """Expects prompt and cache to have no batch axis."""
  # Note: next_token is a length-1 sequence throughout
  # Prefill
  key, sk = jax.random.split(key)
  next_token, cache, cache_params = sample_one_token(
    config, sk, kernel, params, prompt, cache, cache_params, config.inference.temperature
  )
  prefill = jnp.concatenate((prompt, next_token))

  # Generation loop
  def loop_fn(next_token__cache__cache_params, key):
    next_token, cache, cache_params = sample_one_token(
      config,
      key,
      params,
      *next_token__cache__cache_params,
      config.inference.temperature,
    )
    return (next_token, cache, cache_params), next_token[0]

  keys = jax.random.split(key, config.inference.max_tokens_to_generate)
  (next_token, cache, cache_params), output = jax.lax.scan(
    loop_fn, (next_token, cache, cache_params), keys
  )
  output = jnp.concatenate((prefill, output))

  return output, cache, cache_params
