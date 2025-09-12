from functools import partial

import jax
import jax.numpy as jnp

from config import Config
from model import Transformer, _transformer


@partial(jax.jit, donate_argnums=(4,))
def sample_one_token(
    config: Config,
    key,
    params: Transformer,
    x: jax.Array,
    cache_in: jax.Array,
    cache_size: int,
    temperature: float,
):
    y, cache_out = _transformer(config, params, x, cache_in, cache_size)
    logits = y.astype(config.compute_dtype)
    cache_size = cache_size + x.shape[-1]
    next_token = jnp.array((jax.random.categorical(key, logits[-1] / temperature),))
    return next_token, cache_out, cache_size


@jax.jit
def generate(
    config: Config,
    key,
    params: Transformer,
    prompt: jax.Array,
    cache: jax.Array,
    cache_size: int,
) -> tuple[jax.Array, jax.Array, int]:
    output = prompt

    # Prefill
    key, sk = jax.random.split(key)
    next_token, cache, cache_size = sample_one_token(
        config, sk, params, prompt, cache, cache_size, config.temperature
    )
    output = jnp.concatenate((output, next_token))
    # Generation loop
    for step in range(config.max_tokens_to_generate):
        key, sk = jax.random.split(key)
        next_token, cache, cache_size = sample_one_token(
            config, sk, params, next_token, cache, cache_size, config.temperature
        )
        output = jnp.concatenate((output, next_token))

    return output, cache, cache_size
