from functools import partial

import jax
import jax.numpy as jnp
import tiktoken

from bmgpt.config import Config, InferenceConfig
from bmgpt.model import CacheParams, Transformer, _transformer


def sample_one_prompt(
  config: Config,
  key,
  kernel,
  params: Transformer,
  seq: jax.Array,
  cache_in: jax.Array,
  cache_size: int,
  temperature: float,
):
  """Expects seq and cache_in to have no batch axis."""
  cache_params = CacheParams(enabled=True, size=cache_size)
  y, cache_out = _transformer(config, kernel, params, seq, cache_in, cache_params)
  logits = y.astype(config.model.compute_dtype.value)
  cache_size = cache_size + seq.shape[-1]
  next_token = jnp.array((jax.random.categorical(key, logits[-1] / temperature),))
  return next_token, cache_out, cache_size


@partial(jax.jit, static_argnums=(2,), donate_argnums=(5,))
def generate(
  config: Config,
  key,
  kernel,
  params: Transformer,
  prompt: jax.Array,
  cache: jax.Array,
  cache_size: int,
) -> tuple[jax.Array, jax.Array, int]:
  """Expects prompt and cache to have no batch axis."""
  # Note: next_token is a length-1 sequence throughout
  # Prefill
  key, sk = jax.random.split(key)
  next_token, cache, cache_size = sample_one_prompt(
    config,
    sk,
    kernel,
    params,
    prompt,
    cache,
    cache_size,
    config.inference.temperature,
  )
  prefill = jnp.concatenate((prompt, next_token))

  # Generation loop
  def loop_fn(next_token__cache__cache_size, key):
    next_token, cache, cache_size = sample_one_prompt(
      config,
      key,
      kernel,
      params,
      *next_token__cache__cache_size,
      config.inference.temperature,
    )
    return (next_token, cache, cache_size), next_token[0]

  keys = jax.random.split(key, config.inference.max_tokens_to_generate)
  (next_token, cache, cache_params), output = jax.lax.scan(
    loop_fn, (next_token, cache, cache_size), keys
  )
  output = jnp.concatenate((prefill, output))

  return output, cache, cache_size
