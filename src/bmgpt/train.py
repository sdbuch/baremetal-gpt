from functools import partial
from pathlib import Path
from typing import Any, Iterable, NamedTuple

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
from bmgpt.losses import fused_softmax_cross_entropy, softmax_cross_entropy
from bmgpt.model import (
  CacheParams,
  Transformer,
  _transformer,
  init_kv_cache,
  init_transformer,
  model_spec,
)
from bmgpt.optimizers import (
  cosine_with_warmup,
  get_lr,
  grad_norm_and_clip,
  init_adam_state,
  opt_update_factory,
)
from bmgpt.splash_helpers import forward_kernels_from_config

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
    config,
    config.train_dataset.global_batch_size,
    config.train_dataset.num_microbatches,
    0,
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

  # Initialize state
  with jax.set_mesh(mesh):
    train_state = init_train_state(key_model, config)
  cache_params = CacheParams(enabled=False, size=0)

  # Configure forward pass (attention kernels)
  train_attn_kernel, train_lse_kernel, val_kernels, eval_kernels = (
    forward_kernels_from_config(config, mesh)
  )

  # Configure optimization
  spec = model_spec(train_state.params)
  weight_decay_mask = jax.tree.map(lambda _, s: bool(s), train_state.params, spec)
  opt_update = opt_update_factory(config.optimizer.type)
  opt_update = partial(opt_update, config, weight_decay_mask)

  @partial(jax.jit, donate_argnums=2)
  def train_step(config: Config, batch, state: TrainState):
    def loss_fn(params: Transformer, microbatch: tuple[jax.Array, jax.Array]):
      inputs, targets = microbatch
      to_compute_dtype = lambda p: p.astype(config.model.compute_dtype.value)
      params = jax.tree.map(to_compute_dtype, params)
      model = partial(
        _transformer, config, train_attn_kernel, params, cache_params=cache_params
      )
      outputs, _ = jax.vmap(model)(inputs, state.kv_cache)
      if config.use_fused_xent_loss:
        loss = fused_softmax_cross_entropy(
          config, params.unemb, outputs, targets, train_lse_kernel
        )
      else:
        loss = softmax_cross_entropy(config, params.unemb, outputs, targets)
      return loss

    # Calculate gradients: use a scan for gradient accumulation
    def gradient_accum(loss__grad, microbatch):
      loss_accum, grad_accum = loss__grad
      loss, grad = jax.value_and_grad(loss_fn)(state.params, microbatch)
      return (loss_accum + loss, jax.tree.map(jnp.add, grad_accum, grad)), loss

    zeros_like_f32 = partial(jnp.zeros_like, dtype=jnp.float32)
    carry = (jnp.zeros(()), jax.tree.map(zeros_like_f32, state.params))
    (loss, grad), raw_losses = jax.lax.scan(gradient_accum, carry, batch)
    assert raw_losses.dtype == jnp.float32
    # NOTE: breaks if per-token loss masking introduced (see unsloth blog)
    loss = loss / config.train_dataset.num_microbatches
    grad = jax.tree.map(lambda x: x / config.train_dataset.num_microbatches, grad)

    # Update parameters
    grad_clipped, _, global_grad_norm = grad_norm_and_clip(config, grad)
    update_tree = jax.tree.map(opt_update, state.params, grad_clipped, state.opt_state)
    update, opt_state, lr = [
      jax.tree.map(lambda _, y: y[i], state.params, update_tree) for i in range(3)
    ]
    params = jax.tree.map(jnp.add, state.params, update)
    new_state = TrainState(params=params, opt_state=opt_state, kv_cache=state.kv_cache)

    metrics = {
      "lr": jax.tree.leaves(lr)[0],
      "batch_loss": loss,
      "grad_norm": global_grad_norm,
    }
    return metrics, new_state

  # Training loop
  with Logger(config) as logger:
    do_evals = partial(eval_loop, config, mesh=mesh, logger=logger)
    for step, batch in enumerate(batch_iter):
      with jax.set_mesh(mesh):
        metrics, train_state = train_step(config, batch, train_state)
      logger.log(metrics | {"step": step})
      if (step + 1) % config.val_log_interval == 0:
        # Calculate val metrics
        key_val = do_evals(
          key_val, zip(config.val_list, val_kernels), train_state.params, step
        )
      if step == config.train_dataset.num_steps - 1:
        break

    # Run evals (testing)
    key_eval = do_evals(
      key_eval, zip(config.eval_list, eval_kernels), train_state.params, step
    )


def eval_loop(
  config: Config,
  key,
  evals_and_kernels: Iterable[tuple[EvaluationConfig, Any]],
  params: Transformer,
  step: int,
  logger: Logger,
  mesh,
):
  logger.flush_buffer()
  for evaluation, shard_mapped__kernel in evals_and_kernels:
    key, key_d, key_e = jax.random.split(key, 3)
    batch_iter = get_distributed_batch_iter(config, evaluation.dataset, key_d, mesh)
    assert evaluation.dataset.num_microbatches == 1, (
      "Microbatching for evaluators not supported"
    )
    batch_iter = map(lambda b: jax.tree.map(lambda x: x.squeeze(0), b), batch_iter)
    evaluation_fn = evaluator_factory(evaluation)
    metrics = evaluation_fn(
      config, key_e, shard_mapped__kernel, mesh, params, batch_iter
    )
    logger.log(metrics | {"step": step})
  logger.flush_buffer()
  return key


if __name__ == "__main__":
  main()
