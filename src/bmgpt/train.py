import operator
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
  ArrayWithMetadata,
  CacheParams,
  Transformer,
  init_kv_cache,
  transformer,
)
from bmgpt.optimizers import (
  grad_norm_and_clip,
  init_adam_state,
  opt_update_factory,
)
from bmgpt.splash_helpers import forward_kernels_from_config

register_configs()


class TrainState(NamedTuple):
  params: Transformer
  opt_state: Any
  kv_cache: jax.Array


def is_ann(x):
  return isinstance(x, ArrayWithMetadata)


@jax.jit
def init_train_state(key, config: Config) -> TrainState:
  i2l = config.model.intermediates_to_log
  params = Transformer.init(
    config, key, name="model", intermediates_to_log=i2l.transformer
  )
  adam_state = jax.tree.map(partial(init_adam_state, config), params, is_leaf=is_ann)
  cache = init_kv_cache(
    config,
    config.train_dataset.global_batch_size,
    config.train_dataset.num_microbatches,
    0,
  )
  return TrainState(params=params, opt_state=adam_state, kv_cache=cache)


@hydra.main(
  version_base=None,
  config_path=str(Path("configs").absolute().resolve()),
  config_name="base_config",
)
def main(config: Config):
  try:
    jax.distributed.initialize()
    jax.tree_util.register_static(type(config))
  except RuntimeError:
    # This implies the distributed backend has already been initialized
    pass
  config_post_init(config)
  mesh = mesh_from_config(config)
  Logger = logger_factory(config.logger_type)

  key = jax.random.key(config.seed)
  key_model, key_train, key_val, key_eval = jax.random.split(key, 4)
  batch_iter = get_distributed_batch_iter(config, config.train_dataset, key_train, mesh)
  with jax.set_mesh(mesh):
    train_state = init_train_state(key_model, config)
  cache_params = CacheParams(enabled=False, size=0)
  train_attn_kernel, train_lse_kernel, val_kernels, eval_kernels = (
    forward_kernels_from_config(config, mesh)
  )
  opt_update = opt_update_factory(config.optimizer.type)
  opt_update = partial(opt_update, config)

  @partial(jax.jit, donate_argnums=2)
  def train_step(config: Config, batch, state: TrainState):
    def loss_fn(params: Transformer, microbatch: tuple[jax.Array, jax.Array]):
      inputs, targets = microbatch
      to_compute_dtype = lambda p: p.astype(config.model.compute_dtype.value)
      params = jax.tree.map(to_compute_dtype, params)
      model = partial(
        transformer, config, train_attn_kernel, params, cache_params=cache_params
      )
      outputs, _, aux = jax.vmap(model)(inputs, state.kv_cache)
      if config.use_fused_xent_loss:
        loss, aux_loss = fused_softmax_cross_entropy(
          config, params.unemb, outputs, targets, train_lse_kernel
        )
      else:
        loss, aux_loss = softmax_cross_entropy(config, params.unemb, outputs, targets)
      return loss, jax.tree.map(jnp.mean, aux) | aux_loss

    def gradient_accum(loss__grad, microbatch):
      loss_accum, grad_accum = loss__grad
      grad_fn = partial(jax.value_and_grad, has_aux=True)
      (loss, aux), grad = grad_fn(loss_fn)(state.params, microbatch)
      return (loss_accum + loss, jax.tree.map(jnp.add, grad_accum, grad)), aux

    zeros_like_f32 = partial(jnp.zeros_like, dtype=jnp.float32)
    carry = (jnp.zeros(()), jax.tree.map(zeros_like_f32, state.params))
    (loss, grad), aux = jax.lax.scan(gradient_accum, carry, batch)
    # NOTE: breaks if per-token loss masking introduced (see unsloth blog)
    loss = loss / config.train_dataset.num_microbatches
    grad = jax.tree.map(lambda x: x / config.train_dataset.num_microbatches, grad)

    grad_clip, global_grad_norm = grad_norm_and_clip(config, grad)
    update_tree = jax.tree.map(
      opt_update, state.params, grad_clip, state.opt_state, is_leaf=is_ann
    )
    update, opt_state, lr = [
      jax.tree.map(lambda _, y: y[i], state.params, update_tree, is_leaf=is_ann)
      for i in range(3)
    ]
    params = jax.tree.map(operator.add, state.params, update, is_leaf=is_ann)
    new_state = TrainState(params=params, opt_state=opt_state, kv_cache=state.kv_cache)

    if config.log_aux_metrics:
      aux = jax.tree.map(jnp.mean, aux)
      aux = compute_aux_metrics(aux, weight=state.params, grad=grad, update=update)
    else:
      aux = {}
    metrics = {
      "train/lr": jax.tree.leaves(lr)[0],
      "train/batch_loss": loss,
      "train/grad_norm": global_grad_norm,
      **aux,
    }
    return metrics, new_state

  with Logger(config) as logger:
    do_evals = partial(eval_loop, config, mesh=mesh, logger=logger)
    step = 0
    for step, batch in enumerate(batch_iter, start=1):
      with jax.set_mesh(mesh):
        metrics, train_state = train_step(config, batch, train_state)
      logger.log(metrics | {"step": step})
      if step % config.val_log_interval == 0:
        key_val = do_evals(
          key_val, zip(config.val_list, val_kernels), train_state.params, step
        )
      if step == config.train_dataset.num_steps:
        break
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


def compute_aux_metrics(aux: dict, **kwargs):
  add_key_prefix = lambda d, prefix: {f"{prefix}/{k}": v for k, v in d.items()}
  rms = lambda x: jnp.sqrt(jnp.mean(x.astype(jnp.float32) ** 2))
  reduce_fn = lambda x: x.reduce(rms, "rms")
  aux_out = add_key_prefix(aux, "fwd")
  for k, v in kwargs.items():
    aux_out |= add_key_prefix(reduce_fn(v), k)
  return aux_out


if __name__ == "__main__":
  main()
