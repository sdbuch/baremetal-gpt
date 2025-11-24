"""Code for evaluating different types of models. Online algorithms."""

from functools import partial

import jax
import jax.numpy as jnp

from bmgpt.config import Config, EvaluationConfig, EvaluatorType
from bmgpt.data import DataloaderOutputType
from bmgpt.model import CacheParams, Transformer, _transformer, init_kv_cache
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
        metric_name=evaluation_config.dataset.split.value + "/accuracy",
      )
    case EvaluatorType.NLL:
      return partial(
        calculate_metric_on_minibatches,
        metric=nll,
        global_batch_size=evaluation_config.dataset.global_batch_size,
        metric_name=evaluation_config.dataset.split.value + "/nll",
      )
    case EvaluatorType.PERPLEXITY:
      return partial(
        calculate_metric_on_minibatches,
        metric=nll,
        global_batch_size=evaluation_config.dataset.global_batch_size,
        metric_name=evaluation_config.dataset.split.value + "/perplexity",
        perplexity_flag=True,
      )


def autoregressive_rollouts(
  config: Config,
  key,
  kernel,
  mesh,
  params: Transformer,
  batch_iter: DataloaderOutputType,
  global_batch_size: int,
):
  """prompts should have a leading batch axis"""
  prompts, _ = next(batch_iter)

  @jax.vmap
  def batched_generate(prompt: jax.Array, cache):
    return generate(config, key, kernel, params, prompt, cache, 0)

  with jax.set_mesh(mesh):
    cache = init_kv_cache(config, global_batch_size, config.model.max_seq_len - 1)
    outputs, cache, cache_size = batched_generate(prompts, cache)

  print(f"Prompt: {prompts.addressable_shards[0].data}")
  print(f"Generated text: {outputs.addressable_shards[0].data}")
  return {}


def calculate_metric_on_minibatches(
  config: Config,
  key,
  kernel,
  mesh,
  params: Transformer,
  batch_iter: DataloaderOutputType,
  global_batch_size: int,
  metric,
  metric_name: str = "",
  perplexity_flag: bool = False,
):
  num_samples_processed = 0
  with jax.set_mesh(mesh):
    cache = init_kv_cache(config, global_batch_size, 0)

  # Process first batch (to get on-device buffer shape)
  batch = next(batch_iter)
  with jax.set_mesh(mesh):
    batch_metric = metric(config, kernel, batch, params, cache)
  buffer = batch_metric
  num_samples_processed += len(batch[0])

  # Process remaining batches
  for batch in batch_iter:
    with jax.set_mesh(mesh):
      batch_metric = metric(config, batch, params, cache)
      buffer += batch_metric
    num_samples_processed += len(batch[0])
  metric = buffer.sum() / num_samples_processed
  if perplexity_flag:
    # there is an online algorithm for perplexity with a product reduction
    # but it is not numerically stable... easier to just do this hack
    metric = jnp.exp(metric)
  return {metric_name: metric}


@jax.jit
def accuracy(config: Config, kernel, batch, params: Transformer, cache):
  inputs, targets = batch
  logits, _ = jax.vmap(
    partial(
      _transformer,
      config,
      kernel,
      params,
      cache_params=CacheParams(enabled=False, size=0),
    )
  )(inputs, cache)
  preds = logits.argmax(axis=-1)
  return (preds == targets).astype(jnp.int32)


@jax.jit
def nll(config: Config, kernel, batch, params: Transformer, cache):
  """Negative log likelihood, calculated in nats."""
  inputs, targets = batch
  logits, _ = jax.vmap(
    partial(
      _transformer,
      config,
      kernel,
      params,
      cache_params=CacheParams(enabled=False, size=0),
    )
  )(inputs, cache)
  logits = logits.astype(config.model.compute_dtype.value)
  logprobs = jax.nn.log_softmax(logits, axis=-1)
  return -jnp.take_along_axis(logprobs, targets[..., None], axis=-1).squeeze(-1)
