from functools import partial

import jax
import jax.numpy as jnp

from bmgpt.config import Config
from bmgpt.model import Transformer, _transformer


@partial(jax.jit, donate_argnums=(4,))
def sample_one_token(
    config: Config,
    key,
    params: Transformer,
    seq: jax.Array,
    cache_in: jax.Array,
    cache_size: int,
    temperature: float,
):
    y, cache_out = _transformer(config, params, seq, cache_in, cache_size)
    logits = y.astype(config.compute_dtype.value)
    cache_size = cache_size + seq.shape[-1]
    next_token = jnp.array((jax.random.categorical(key, logits[-1] / temperature),))
    return next_token, cache_out, cache_size


def generate(
    config: Config,
    key,
    params: Transformer,
    prompt: jax.Array,
    cache: jax.Array,
    cache_size: int,
) -> tuple[jax.Array, jax.Array, int]:
    # Prefill
    key, sk = jax.random.split(key)
    next_token, cache, cache_size = sample_one_token(
        config, sk, params, prompt, cache, cache_size, config.temperature
    )
    prefill = jnp.concatenate((prompt, next_token))

    # Generation loop
    def loop_fn(next_token__cache__cache_size, key):
        next_token, cache, cache_size = next_token__cache__cache_size
        next_token, cache, cache_size = sample_one_token(
            config, key, params, next_token, cache, cache_size, config.temperature
        )
        return (next_token, cache, cache_size), next_token[0]

    keys = jax.random.split(key, config.max_tokens_to_generate)

    (next_token, cache, cache_size), output = jax.lax.scan(
        loop_fn, (next_token, cache, cache_size), keys
    )

    # for step in range(config.max_tokens_to_generate):
    #     key, sk = jax.random.split(key)
    #     next_token, cache, cache_size = sample_one_token(
    #         config, sk, params, next_token, cache, cache_size, config.temperature
    #     )
    output = jnp.concatenate((prefill, output))

    return output, cache, cache_size
