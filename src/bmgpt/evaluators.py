"""Code for evaluating different types of models."""

from functools import partial

import jax

from bmgpt.config import Config
from bmgpt.model import Transformer, init_kv_cache
from bmgpt.sample import generate


def autoregressive_rollouts(
    config: Config,
    key,
    params: Transformer,
    prompts: jax.Array,
):
    """prompts should have a leading batch axis"""
    batch_size = prompts.shape[0]
    cache = init_kv_cache(config)
    cache_size = 0

    print(prompts.shape)
    print(cache.shape)

    @jax.vmap
    def batched_generate(prompt, cache):
        return generate(config, key, params, prompt, cache, cache_size)

    outputs, cache, cache_size = batched_generate(prompts, cache)
    print(f"Prompt: {prompts[0]}")
    print(f"Cache size: {cache_size}")
    print(f"Generated text: {outputs[0]}")
    return outputs
