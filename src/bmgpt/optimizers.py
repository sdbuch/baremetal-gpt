import operator
from typing import NamedTuple

import jax
import jax.numpy as jnp

from bmgpt.config import Config, OptType
from bmgpt.model import Transformer


def get_opt_update_fn_from_enum(opt_type: OptType):
    match opt_type:
        case OptType.ADAMW:
            return adamw_update
        case OptType.SGD:
            return sgd_update


def grad_norm_and_clip(
    config: Config, model: Transformer
) -> tuple[Transformer, Transformer]:
    # Gradient norms in param precision (NOTE: might want fp32?)
    grad_norms_squared = jax.tree.map(lambda grad: jnp.sum(grad**2), model)
    global_grad_norm = jax.tree.reduce(operator.add, grad_norms_squared) ** 0.5
    truncated_norm = jax.lax.select(
        global_grad_norm >= config.clip_grad,
        global_grad_norm,
        jnp.ones_like(global_grad_norm),
    )
    # truncated_norm = jnp.maximum(global_grad_norm - config.clip_grad, 0.0) + 1.0
    return jax.tree.map(lambda grad: grad / truncated_norm, model), grad_norms_squared


class OptState(NamedTuple):
    mu: jax.Array  # 1st moment EMA
    nu: jax.Array  # 2nd moment EMA
    step: jax.Array  # step number


def init_adam_state(config: Config, param: jax.Array) -> OptState:
    return OptState(
        mu=jnp.zeros_like(param, dtype=config.optimizer_dtype.value),
        nu=jnp.zeros_like(param, dtype=config.optimizer_dtype.value),
        step=jnp.array(0, dtype=jnp.int32),
    )


def adamw_update(
    config: Config, param: jax.Array, grad: jax.Array, state: OptState, mask: bool
):
    beta1 = config.beta1
    beta2 = config.beta2
    lr = config.lr
    eps = config.eps_adam
    weight_decay = config.weight_decay

    mu = beta1 * state.mu + (1 - beta1) * grad
    nu = beta2 * state.nu + (1 - beta2) * grad.astype(config.optimizer_dtype.value) ** 2
    new_state = OptState(mu=mu, nu=nu, step=state.step + 1)

    mu_debias = mu / (1 - beta1**new_state.step)
    nu_debias = nu / (1 - beta2**new_state.step)
    update = -lr * mu_debias / (eps + jnp.sqrt(nu_debias))
    if mask:
        # Apply weight decay
        update = update - lr * weight_decay * param
    return param + update.astype(config.param_dtype.value), new_state


def sgd_update(
    config: Config, param: jax.Array, grad: jax.Array, state: OptState, mask: bool
):
    update = -config.lr * grad
    if mask:
        # Apply weight decay
        update = update - config.lr * config.weight_decay * param
    return param + update, state
