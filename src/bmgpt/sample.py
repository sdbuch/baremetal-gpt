from functools import partial

import jax
import jax.numpy as jnp

from bmgpt.config import Config
from bmgpt.model import CacheParams, Transformer, _transformer


def sample_one_token(
  config: Config,
  key,
  kernel,
  params: Transformer,
  seqs: jax.Array,
  cache_in: jax.Array,
  cache_size: int,
  temperature: float,
):
  cache_params = CacheParams(enabled=True, size=cache_size)
  y, cache_out = _transformer(config, kernel, params, seqs, cache_in, cache_params)
  logits = y.astype(config.model.compute_dtype.value)
  cache_size = cache_size + seqs.shape[-1]
  next_tokens = jnp.array(jax.random.categorical(key, logits[:, -1:] / temperature))
  return next_tokens, cache_out, cache_size


@partial(jax.jit, static_argnums=(2,), donate_argnums=(5,))
def generate(
  config: Config,
  key,
  kernel,
  params: Transformer,
  prompts: jax.Array,
  cache: jax.Array,
  cache_size: int,
) -> tuple[jax.Array, jax.Array, int]:
  # Note: next_token is a length-1 sequence throughout
  # Prefill
  key, sk = jax.random.split(key)
  next_tokens, cache, cache_size = sample_one_token(
    config,
    sk,
    kernel,
    params,
    prompts,
    cache,
    cache_size,
    config.inference.temperature,
  )
  prefill = jnp.concatenate((prompts, next_tokens), axis=-1)

  # Generation loop
  def loop_fn(next_token__cache__cache_size, key):
    next_token, cache, cache_size = sample_one_token(
      config,
      key,
      kernel,
      params,
      *next_token__cache__cache_size,
      config.inference.temperature,
    )
    return (next_token, cache, cache_size), next_token[:, 0]

  keys = jax.random.split(key, config.inference.max_tokens_to_generate)
  (next_tokens, cache, cache_params), output = jax.lax.scan(
    loop_fn, (next_tokens, cache, cache_size), keys
  )
  output = jnp.concatenate((prefill, jnp.moveaxis(output, 0, 1)), axis=-1)

  return output, cache, cache_size
