import copy
import math
from functools import partial
from pathlib import Path
from time import sleep
from typing import Any, NamedTuple

import hydra
import jax
import jax.numpy as jnp

from bmgpt.config import (
  Config,
  EvaluationConfig,
  config_post_init,
  mesh_from_config,
  register_configs,
)
from bmgpt.data import get_distributed_batch_iter
from bmgpt.evaluators import evaluator_factory
from bmgpt.loggers import Logger, logger_factory
from bmgpt.model import (
  Transformer,
  _transformer,
  init_kv_cache,
  init_transformer,
  model_spec,
)
from bmgpt.optimizers import (
  grad_norm_and_clip,
  init_adam_state,
  opt_update_factory,
)

register_configs()


# Setup for training loop
class TrainState(NamedTuple):
  params: Transformer
  opt_state: Any
  kv_cache: jax.Array


@jax.jit
def init_train_state(key, config: Config) -> TrainState:
  model_params = init_transformer(key, config)
  adam_state = jax.tree.map(partial(init_adam_state, config), model_params)
  cache = init_kv_cache(
    config, config.train_dataset.global_batch_size, update_cache=False
  )
  return TrainState(params=model_params, opt_state=adam_state, kv_cache=cache)


@hydra.main(
  version_base=None,
  config_path=str(Path("configs").absolute().resolve()),
  config_name="base_config",
)
def main(config: Config):
  try:
    # Launch distributed and register configs
    jax.distributed.initialize()
    jax.tree_util.register_static(type(config))
  except RuntimeError:
    # This implies the distributed backend has already been initialized
    pass
  config_post_init(config)
  mesh = mesh_from_config(config)
  Logger = logger_factory(config.logger_type)

  # Randomness
  key = jax.random.key(config.seed)
  key_model, key_train, key_val, key_eval = jax.random.split(key, 4)

  # Data
  batch_iter = get_distributed_batch_iter(config, config.train_dataset, key_train, mesh)

  # Initialize state, configure optimization
  with jax.set_mesh(mesh):
    train_state = init_train_state(key_model, config)
  spec = model_spec(train_state.params)
  opt_update = opt_update_factory(config.optimizer.type)
  weight_decay_mask = jax.tree.map(lambda _, s: bool(s), train_state.params, spec)

  @partial(jax.jit, donate_argnums=2)
  def train_step(config: Config, batch, train_state: TrainState):
    def loss_fn(params: Transformer):
      inputs, targets = batch
      logits, _ = jax.vmap(partial(_transformer, config, params, cache_size=-1))(
        inputs, train_state.kv_cache
      )
      logits = logits.astype(config.model.compute_dtype.value)
      logprobs = jax.nn.log_softmax(logits, axis=-1)
      return -jnp.take_along_axis(logprobs, targets[..., None], axis=-1).mean()

    loss, grad = jax.value_and_grad(loss_fn)(train_state.params)
    grad_clipped, _, global_grad_norm = grad_norm_and_clip(config, grad)
    update__opt_state = jax.tree.map(
      partial(opt_update, config),
      train_state.params,
      grad_clipped,
      train_state.opt_state,
      weight_decay_mask,
    )
    # Transpose the output tree to get update tree and state tree
    update, opt_state = map(
      lambda i: jax.tree.map(lambda x, y: y[i], grad, update__opt_state), range(2)
    )
    params = jax.tree.map(lambda x, y: x + y, train_state.params, update)
    new_state = TrainState(
      params=params, opt_state=opt_state, kv_cache=train_state.kv_cache
    )

    metrics = {"batch_loss": loss, "grad_norm": global_grad_norm}
    return metrics, new_state

  # Training loop
  with Logger(config) as logger:
    for step, batch in enumerate(batch_iter):
      with jax.set_mesh(mesh):
        metrics, train_state = train_step(config, batch, train_state)
      logger.log(metrics | {"step": step})
      if (step + 1) % config.val_log_interval == 0:
        # Calculate val metrics
        key_val = eval_loop(
          config, key_val, config.val_list, train_state.params, logger, mesh, step
        )
      if step == config.optimizer.num_steps - 1:
        break

    # Run evals (testing)
    key_eval = eval_loop(
      config, key_eval, config.eval_list, train_state.params, logger, mesh, step
    )


def eval_loop(
  config: Config,
  key,
  eval_list: list[EvaluationConfig],
  params: Transformer,
  logger: Logger,
  mesh,
  step: int,
):
  logger.flush_buffer()
  for evaluation in eval_list:
    key, key_d, key_e = jax.random.split(key, 3)
    batch_iter = get_distributed_batch_iter(config, evaluation.dataset, key_d, mesh)
    evaluation_fn = evaluator_factory(evaluation)
    metrics = evaluation_fn(config, key_e, mesh, params, batch_iter)
    logger.log(metrics | {"step": step})
  logger.flush_buffer()
  return key


if __name__ == "__main__":
  main()
