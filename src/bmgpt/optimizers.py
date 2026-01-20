import operator
from typing import NamedTuple

import jax
import jax.numpy as jnp

from bmgpt.config import Config, LRScheduleType, OptType
from bmgpt.model import Transformer

"""Optimizer init/updates operate on Arrays, not pytrees (tree.map them)"""


##############################
#         Utility
##############################


def grad_norm_and_clip(
  config: Config, model: Transformer
) -> tuple[Transformer, Transformer, float]:
  # Gradient norms in fp32
  norm_sq = lambda x: jnp.einsum(
    "...,...->", x, x, preferred_element_type=jnp.float32, out_sharding=jax.P()
  )
  grad_norms_squared = jax.tree.map(norm_sq, model)
  global_grad_norm = jax.tree.reduce(operator.add, grad_norms_squared) ** 0.5
  truncated_norm = jax.lax.select(
    global_grad_norm >= config.optimizer.clip_grad,
    global_grad_norm,
    jnp.ones_like(global_grad_norm),
  ).astype(config.model.param_dtype.value)
  return (
    jax.tree.map(lambda grad: grad / truncated_norm, model),
    grad_norms_squared,
    global_grad_norm,
  )


##############################
#      Optimizer State
##############################


class OptState(NamedTuple):
  mu: jax.Array  # 1st moment EMA
  nu: jax.Array  # 2nd moment EMA
  step: jax.Array  # step number


def init_adam_state(config: Config, param: jax.Array) -> OptState:
  return OptState(
    mu=jnp.zeros_like(param, dtype=config.model.opt_dtype.value),
    nu=jnp.zeros_like(param, dtype=config.model.opt_dtype.value),
    step=jnp.array(0, dtype=jnp.int32),
  )


def init_sgd_state(config: Config, param: jax.Array) -> OptState:
  return OptState(
    mu=jnp.zeros_like(param, dtype=config.model.opt_dtype.value),
    nu=jnp.array(()),
    step=jnp.array(0, dtype=jnp.int32),
  )


##############################
#   Learning Rate Schedules
##############################


def lr_schedule_factory(lr_schedule_type: LRScheduleType):
  match lr_schedule_type:
    case LRScheduleType.WARMUP_STABLE:
      return warmup_stable
    case LRScheduleType.COSINE_WITH_WARMUP:
      return cosine_with_warmup


def warmup_stable(config: Config, step: jax.Array):
  """Range from base_lr to peak_lr over num_warmup_steps, then cap."""
  base_lr = config.optimizer.base_lr
  peak_lr = config.optimizer.peak_lr
  max_step = config.optimizer.num_warmup_steps - 1
  if max_step <= 0:
    return jnp.array(peak_lr)
  linear_ramp = lambda eta: base_lr + eta * (peak_lr - base_lr) / max_step
  return jnp.minimum(linear_ramp(step), peak_lr)


def cosine_decay(config: Config, step: jax.Array, num_warmup_steps: int = 0):
  """Cosine decay schedule from peak_lr to base_lr over num_steps."""
  peak_lr = config.optimizer.peak_lr
  min_lr = config.optimizer.min_lr
  # NOTE: doesn't support pure epoch-based currently
  max_step = config.train_dataset.num_steps - num_warmup_steps - 1

  progress = jnp.array(1.0) if max_step <= 0 else jnp.clip(step / max_step, 0, 1)
  cosine_decay = 0.5 * (1 + jnp.cos(jnp.pi * progress))
  return min_lr + (peak_lr - min_lr) * cosine_decay


def cosine_with_warmup(config: Config, step: jax.Array):
  num_warmup_steps = max(config.optimizer.num_warmup_steps, 1)
  return jnp.where(
    step < num_warmup_steps,
    warmup_stable(config, step),
    cosine_decay(
      config, step - num_warmup_steps + 1, num_warmup_steps=num_warmup_steps - 1
    ),
  )


def get_lr(config: Config, schedule_type: LRScheduleType, state: OptState):
  lr_schedule_fn = lr_schedule_factory(schedule_type)
  lr = lr_schedule_fn(config, state.step)
  return lr


##############################
#          Updates
##############################

# TODO: Expose configuration options for LR schedulers


def opt_update_factory(opt_type: OptType):
  match opt_type:
    case OptType.ADAMW:
      return adamw_update
    case OptType.SGD:
      return sgd_update


def adamw_update(
  config: Config, wd_mask: bool, param: jax.Array, grad: jax.Array, state: OptState
):
  beta1 = config.optimizer.beta1
  beta2 = config.optimizer.beta2
  eps = config.optimizer.eps_adam
  weight_decay = config.optimizer.weight_decay
  lr = get_lr(config, config.optimizer.schedule_type, state)

  mu = beta1 * state.mu + (1 - beta1) * grad
  nu = beta2 * state.nu + (1 - beta2) * grad**2
  new_state = OptState(mu=mu, nu=nu, step=state.step + 1)

  # simulate initializing the buffers with initial gradients via pre-incrementing step
  mu_debias = mu / (1 - beta1**new_state.step)
  nu_debias = nu / (1 - beta2**new_state.step)
  update = -lr * mu_debias / (eps + jnp.sqrt(nu_debias))
  if wd_mask:
    update = update - lr * weight_decay * param
  return update.astype(config.model.param_dtype.value), lr, new_state


def sgd_update(
  config: Config, wd_mask: bool, param: jax.Array, grad: jax.Array, state: OptState
):
  beta1 = config.optimizer.beta1
  weight_decay = config.optimizer.weight_decay
  lr = get_lr(config, config.optimizer.schedule_type, state)

  mu = beta1 * state.mu + (1 - beta1) * grad
  new_state = OptState(mu=mu, nu=state.nu, step=state.step + 1)

  # simulate initializing the buffers with initial gradients via pre-incrementing step
  mu_debias = mu / (1 - beta1**new_state.step)
  update = -mu_debias
  if wd_mask:
    update = update - weight_decay * param
  return update.astype(config.model.param_dtype.value), lr, state
