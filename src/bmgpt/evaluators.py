"""Code for evaluating different types of models. Online algorithms."""

from functools import partial

import jax
import jax.numpy as jnp
from jax.experimental.multihost_utils import process_allgather

from bmgpt.config import Config, EvaluationConfig, EvaluatorType
from bmgpt.data import DataloaderOutputType
from bmgpt.losses import MetricType, accuracy, softmax_cross_entropy
from bmgpt.model import CacheParams, Transformer, _transformer, init_kv_cache
from bmgpt.sample import generate
from bmgpt.tokenizers import get_tokenizer_factory


def evaluator_factory(evaluation_config: EvaluationConfig):
  """Returns the function to call the eval, dataloader for the eval"""
  match evaluation_config.evaluator:
    case EvaluatorType.AUTOREGRESSIVE_ROLLOUTS:
      return partial(
        autoregressive_rollouts,
        global_batch_size__num_microbatches=(
          evaluation_config.dataset.global_batch_size,
          evaluation_config.dataset.num_microbatches,
        ),
        prompt_size=evaluation_config.dataset.seq_len,
      )
    case EvaluatorType.ACCURACY:
      return partial(
        calculate_metric_on_minibatches,
        metric_fun=accuracy,
        global_batch_size__num_microbatches=(
          evaluation_config.dataset.global_batch_size,
          evaluation_config.dataset.num_microbatches,
        ),
        metric_name=evaluation_config.dataset.split.value + "/accuracy",
        num_steps=evaluation_config.dataset.num_steps,
      )
    case EvaluatorType.NLL:
      return partial(
        calculate_metric_on_minibatches,
        metric_fun=softmax_cross_entropy,
        global_batch_size__num_microbatches=(
          evaluation_config.dataset.global_batch_size,
          evaluation_config.dataset.num_microbatches,
        ),
        metric_name=evaluation_config.dataset.split.value + "/nll",
        num_steps=evaluation_config.dataset.num_steps,
      )
    case EvaluatorType.PERPLEXITY:
      return partial(
        calculate_metric_on_minibatches,
        metric_fun=softmax_cross_entropy,
        global_batch_size__num_microbatches=(
          evaluation_config.dataset.global_batch_size,
          evaluation_config.dataset.num_microbatches,
        ),
        metric_name=evaluation_config.dataset.split.value + "/perplexity",
        perplexity_flag=True,
        num_steps=evaluation_config.dataset.num_steps,
      )


def autoregressive_rollouts(
  config: Config,
  key,
  kernel,
  mesh,
  params: Transformer,
  batch_iter: DataloaderOutputType,
  global_batch_size__num_microbatches: tuple[int, int],
  prompt_size: int,
):
  prompts, _ = next(batch_iter)

  @jax.vmap
  def batched_generate(prompt: jax.Array, cache):
    return generate(config, key, kernel, params, prompt, cache, 0)

  with jax.set_mesh(mesh):
    cache = init_kv_cache(
      config, *global_batch_size__num_microbatches, config.model.max_seq_len - 1
    )
    outputs, cache, cache_size = batched_generate(prompts, cache)

  prompts, outputs = process_allgather((prompts, outputs), tiled=True)
  tokenizer = get_tokenizer_factory(config.inference)
  str_prompts = [tokenizer.decode(ids[:prompt_size]) for ids in outputs]
  str_outputs = [tokenizer.decode(ids[prompt_size:]) for ids in outputs]
  print("#" * 32 + "\nPrompt:\n" + "#" * 32)
  print(f"{str_prompts[jax.process_index()]}")
  print("#" * 32 + "\nGenerated text:\n" + "#" * 32)
  print(f"{str_outputs[jax.process_index()]}")
  return {}


def calculate_metric_on_minibatches(
  config: Config,
  key,
  kernel,
  mesh,
  params: Transformer,
  batch_iter: DataloaderOutputType,
  global_batch_size__num_microbatches: tuple[int, int],
  metric_fun: MetricType,
  metric_name: str = "",
  perplexity_flag: bool = False,
  num_steps: int = 0,
):
  with jax.set_mesh(mesh):
    cache = init_kv_cache(config, *global_batch_size__num_microbatches, 0)

  # Loss accumulation function (avoid communication until we're done)
  @jax.jit
  def forward_and_calc_metric(inputs, targets, params, cache):
    cache_params = CacheParams(enabled=False, size=0)
    model = partial(_transformer, config, kernel, params, cache_params=cache_params)
    outputs, _ = jax.vmap(model)(inputs, cache)
    return metric_fun(
      config, params.unemb, outputs, targets, kernel, False
    )  # don't reduce

  # Process first batch (to get on-device buffer shape)
  batch = next(batch_iter)
  with jax.set_mesh(mesh):
    batch_metric = forward_and_calc_metric(*batch, params, cache)
  batch_metric_buffer = batch_metric
  tokens_per_batch = batch_metric_buffer.size
  num_batches_processed = 1

  # Process remaining batches
  step = -1
  if num_steps != 1:  # executed step 1 above
    for step, batch in enumerate(batch_iter):
      with jax.set_mesh(mesh):
        batch_metric = forward_and_calc_metric(*batch, params, cache)
        batch_metric_buffer += batch_metric
      if step == num_steps - 1:
        break
  num_batches_processed += step + 1
  num_tokens_processed = tokens_per_batch * num_batches_processed
  metric = batch_metric_buffer.sum() / num_tokens_processed
  if perplexity_flag:
    metric = jnp.exp(metric)
  return {metric_name: metric}
