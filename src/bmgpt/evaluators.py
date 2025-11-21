"""Code for evaluating different types of models."""

from functools import partial

import jax

from bmgpt.config import Config, EvaluationConfig, EvaluatorType
from bmgpt.data import DataloaderOutputType
from bmgpt.model import Transformer, init_kv_cache
from bmgpt.sample import generate


def evaluator_factory(evaluation_config: EvaluationConfig):
    """Returns the function to call the eval, dataloader for the eval"""
    match evaluation_config.evaluator:
        case EvaluatorType.AUTOREGRESSIVE_ROLLOUTS:
            return partial(
                autoregressive_rollouts,
                global_batch_size=evaluation_config.dataset.global_batch_size,
            )
        case EvaluatorType.ACCURACY:
            return None
        case EvaluatorType.PERPLEXITY:
            return None
        case EvaluatorType.NLL:
            return None


def autoregressive_rollouts(
    config: Config,
    key,
    mesh,
    params: Transformer,
    batch_iter: DataloaderOutputType,
    global_batch_size: int,
):
    """prompts should have a leading batch axis"""
    prompts, _ = next(batch_iter)
    cache_size = 0

    @jax.vmap
    def batched_generate(prompt: jax.Array, cache):
        return generate(config, key, params, prompt, cache, cache_size)

    with jax.set_mesh(mesh):
        cache = init_kv_cache(config, global_batch_size)
        outputs, cache, cache_size = batched_generate(prompts, cache)

    print(f"Prompt: {prompts.addressable_shards[0].data}")
    print(f"Generated text: {outputs.addressable_shards[0].data}")
    return outputs
