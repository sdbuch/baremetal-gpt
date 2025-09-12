from types import NoneType
from typing import NamedTuple

import jax
import jax.numpy as jnp

from config import Config


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


def adam_update(config: Config, param: jax.Array, grad: jax.Array, state: OptState):
    beta1 = config.beta1
    beta2 = config.beta2
    lr = config.lr
    eps = config.eps_adam

    mu = beta1 * state.mu + (1 - beta1) * grad
    nu = beta2 * state.nu + (1 - beta2) * grad**2
    new_state = OptState(mu=mu, nu=nu, step=state.step + 1)

    mu_debias = mu / (1 - beta1**new_state.step)
    nu_debias = nu / (1 - beta2**new_state.step)
    update = -lr * mu_debias / (eps + jnp.sqrt(nu_debias))
    return param + update, new_state


def sgd_update(config: Config, param: jax.Array, grad: jax.Array, state: OptState):
    update = -config.lr * grad
    return param + update, state
