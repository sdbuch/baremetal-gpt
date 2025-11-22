from functools import partial
from pathlib import Path

import hydra
import jax
import jax.numpy as jnp

from bmgpt.config import Config, config_post_init, mesh_from_config
from bmgpt.data import get_distributed_batch_iter
from bmgpt.model import Transformer, _transformer, model_spec
from bmgpt.optimizers import grad_norm_and_clip, opt_update_factory
from bmgpt.train import TrainState, init_train_state


@hydra.main(
  version_base=None,
  config_path=str(Path("configs").absolute().resolve()),
  config_name="base_config",
)
def main(config: Config):
  # Launch distributed and profile
  jax.distributed.initialize()
  # Config
  config_post_init(config)
  mesh = mesh_from_config(config)

  # Randomness
  key = jax.random.key(config.seed)
  key_model, key_train, key_eval = jax.random.split(key, 3)

  # Data
  batch_iter = get_distributed_batch_iter(config, config.train_dataset, key_train, mesh)

  # Initialize state, configure optimization
  init_compiled = init_train_state.lower(key_model, config).compile()
  with jax.set_mesh(mesh):
    train_state = init_compiled(key_model)
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

  # Simple training loop
  for step, batch in enumerate(batch_iter):
    step_compiled = train_step.lower(config, batch, train_state).compile()
    with jax.set_mesh(mesh):
      jax.profiler.start_trace("/tmp/profile-train")
      cur_metrics, train_state = step_compiled(batch, train_state)
      cur_metrics['grad_norm'].block_until_ready()
      jax.profiler.stop_trace()
    if step == config.optimizer.num_steps - 1:
      break


if __name__ == "__main__":
  main()
