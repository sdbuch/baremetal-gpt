"""Code for evaluating different types of models."""

from functools import partial

import jax
import jax.numpy as jnp

from bmgpt.config import Config, EvaluationConfig, EvaluatorType
from bmgpt.data import DataloaderOutputType
from bmgpt.model import Transformer, _transformer, init_kv_cache
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
            return partial(
                calculate_metric_on_minibatches,
                metric=accuracy,
                global_batch_size=evaluation_config.dataset.global_batch_size,
            )
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


def calculate_metric_on_minibatches(
    config: Config,
    key,
    mesh,
    params: Transformer,
    batch_iter: DataloaderOutputType,
    metric,
    global_batch_size: int,
):
    prev_metric = None
    buffer = None
    num_samples_processed = 0
    with jax.set_mesh(mesh):
        cache = init_kv_cache(config, global_batch_size)
    for batch in batch_iter:
        with jax.set_mesh(mesh):
            batch_metric = metric(config, batch, params, cache)
        log_metric, prev_metric = prev_metric, batch_metric
        if log_metric is not None:
            if buffer is None:
                buffer = log_metric
            else:
                buffer += log_metric
        num_samples_processed += len(batch[0])
    acc = buffer.sum() / num_samples_processed if buffer is not None else None
    return {"accuracy": acc}


@jax.jit
def accuracy(config: Config, batch, params: Transformer, cache):
    inputs, targets = batch
    logits, _ = jax.vmap(partial(_transformer, config, params))(inputs, cache)
    preds = logits.argmax(axis=-1)
    return (preds == targets).astype(jnp.int32)
