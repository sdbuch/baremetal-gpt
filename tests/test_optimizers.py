import jax
import jax.numpy as jnp
import optax
import pytest

from bmgpt.config import Config, DType, config_post_init, mesh_from_config
from bmgpt.model import (
  Transformer,
  _apply_rope,
  _attn,
  _make_cache_mask,
  _make_causal_mask,
  _precompute_rope_cossin,
  _transformer,
  init_kv_cache,
  init_transformer,
  model_spec,
)
from bmgpt.optimizers import (
  adamw_update,
  grad_norm_and_clip,
  init_adam_state,
  sgd_update,
)
from bmgpt.train import init_train_state


def test_sgd_update():
  config = Config()
  config.optimizer.weight_decay = 1000.0
  # Simple least squares
  key = jax.random.key(config.experiment.seed)
  k, sk = jax.random.split(key)
  A, X = jax.random.normal(k, (2, 100, 200))
  Y = jax.random.normal(sk, (100, 100))

  def loss(X):
    return jnp.sum((A @ X.T - Y) ** 2)

  loss_val, grad = jax.value_and_grad(loss)(X)

  # Compare updates: no wd
  upd, state = sgd_update(config, X, grad, None, False)
  upd_optax_fn = optax.sgd(config.optimizer.lr)
  state_optax = upd_optax_fn.init(X)
  upd_optax, state_optax = upd_optax_fn.update(grad, state_optax, X)
  assert jnp.allclose(upd, upd_optax)

  # compare updated parameters
  X_new = X + upd
  X_new_optax = optax.apply_updates(X, upd_optax)
  assert jnp.allclose(X_new, X_new_optax)

  # Compare updates: wd
  upd, state = sgd_update(config, X, grad, None, True)
  upd_optax_fn = optax.chain(
    optax.sgd(config.optimizer.lr),
    optax.add_decayed_weights(-config.optimizer.weight_decay * config.optimizer.lr),
  )
  state_optax = upd_optax_fn.init(X)
  upd_optax, state_optax = upd_optax_fn.update(grad, state_optax, X)
  assert jnp.allclose(upd, upd_optax)

  # compare updated parameters
  X_new = X + upd
  X_new_optax = optax.apply_updates(X, upd_optax)
  assert jnp.allclose(X_new, X_new_optax)


def test_adam_update():
  dtype = DType.FLOAT32
  config = Config()
  config.model.param_dtype = dtype
  config.optimizer.weight_decay = 100.0
  # Simple least squares
  key = jax.random.key(config.experiment.seed)
  k, sk = jax.random.split(key)
  A, X = jax.random.normal(k, (2, 100, 200), dtype=dtype.value)
  Y = jax.random.normal(sk, (100, 100), dtype=dtype.value)

  def loss(X):
    return jnp.sum((A @ X.T - Y) ** 2)

  loss_val, grad = jax.value_and_grad(loss)(X)

  # Compare updates: no wd
  state = init_adam_state(config, X)
  upd, state_new = adamw_update(config, X, grad, state, False)
  upd_optax_fn = optax.adamw(
    config.optimizer.lr,
    config.optimizer.beta1,
    config.optimizer.beta2,
    config.optimizer.eps_adam,
    weight_decay=0.0,
    # weight_decay=config.weight_decay,
  )
  state_optax = upd_optax_fn.init(X)
  upd_optax, state_optax_new = upd_optax_fn.update(grad, state_optax, X)
  assert all(
    jax.tree.map(
      jnp.allclose,
      (state_optax[0].count, state_optax[0].mu, state_optax[0].nu),
      (state.step, state.mu, state.nu),
    )
  )
  assert jnp.allclose(upd, upd_optax)

  # compare updated parameters
  X_new = X + upd
  X_new_optax = optax.apply_updates(X, upd_optax)
  assert jnp.allclose(X_new, X_new_optax)

  # Compare updates: wd
  upd, state_new = adamw_update(config, X, grad, state, True)
  upd_optax_fn = optax.adamw(
    config.optimizer.lr,
    config.optimizer.beta1,
    config.optimizer.beta2,
    config.optimizer.eps_adam,
    weight_decay=config.optimizer.weight_decay,
  )
  state_optax = upd_optax_fn.init(X)
  upd_optax, state_optax_new = upd_optax_fn.update(grad, state_optax, X)
  assert jnp.allclose(upd, upd_optax)

  # compare updated parameters
  X_new = X + upd
  X_new_optax = optax.apply_updates(X, upd_optax)
  assert jnp.allclose(X_new, X_new_optax)


def test_update_mask():
  config = Config()
  config.sharding.mesh_shape = [1]
  config.dataset.seq_len = 256
  config.dataset.num_vocab = 256
  mesh = mesh_from_config(config)
  key = jax.random.key(config.experiment.seed)
  with jax.set_mesh(mesh):
    state = init_train_state(key, config)
  spec = model_spec(state.params)
  weight_decay_mask = jax.tree.map(lambda x, s: bool(s), state.params, spec)

  def check_correct(p, x: bool):
    # LN and biases should be false, else true
    path_strs = list(k.__str__() for k in p)
    param_str = path_strs[-1]
    parent_str = path_strs[-2]
    if "norm" in parent_str:
      assert not x
    elif "bias" in param_str:
      assert not x
    else:
      assert x

  _ = jax.tree.map_with_path(check_correct, weight_decay_mask)
