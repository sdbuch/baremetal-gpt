import operator
from typing import NamedTuple

import jax
import jax.numpy as jnp

from bmgpt.config import Config, OptType
from bmgpt.model import Transformer

"""Optimizer init/updates operate on Arrays, not pytrees (tree.map them)"""


def opt_update_factory(opt_type: OptType):
  match opt_type:
    case OptType.ADAMW:
      return adamw_update
    case OptType.SGD:
      return sgd_update


def grad_norm_and_clip(
  config: Config, model: Transformer
) -> tuple[Transformer, Transformer, float]:
  # Gradient norms in param precision (NOTE: might want fp32?)
  grad_norms_squared = jax.tree.map(lambda grad: jnp.sum(grad**2), model)
  global_grad_norm = jax.tree.reduce(operator.add, grad_norms_squared) ** 0.5
  truncated_norm = jax.lax.select(
    global_grad_norm >= config.optimizer.clip_grad,
    global_grad_norm,
    jnp.ones_like(global_grad_norm),
  )
  return (
    jax.tree.map(lambda grad: grad / truncated_norm, model),
    grad_norms_squared,
    global_grad_norm,
  )


##############################
#      State Management
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


##############################
#          Updates
##############################


def adamw_update(
  config: Config, wd_mask: bool, param: jax.Array, grad: jax.Array, state: OptState
):
  beta1 = config.optimizer.beta1
  beta2 = config.optimizer.beta2
  lr = config.optimizer.lr
  eps = config.optimizer.eps_adam
  weight_decay = config.optimizer.weight_decay

  mu = beta1 * state.mu + (1 - beta1) * grad
  nu = beta2 * state.nu + (1 - beta2) * grad**2
  new_state = OptState(mu=mu, nu=nu, step=state.step + 1)

  # simulate initializing the buffers with initial gradients via pre-incrementing step
  mu_debias = mu / (1 - beta1**new_state.step)
  nu_debias = nu / (1 - beta2**new_state.step)
  update = -lr * mu_debias / (eps + jnp.sqrt(nu_debias))
  if wd_mask:
    update = update - lr * weight_decay * param
  return update.astype(config.model.param_dtype.value), new_state


def sgd_update(
  config: Config, wd_mask: bool, param: jax.Array, grad: jax.Array, state: OptState
):
  update = -config.optimizer.lr * grad
  if wd_mask:
    update = update - config.optimizer.lr * config.optimizer.weight_decay * param
  return update.astype(config.model.param_dtype.value), state
