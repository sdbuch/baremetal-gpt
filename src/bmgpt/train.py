import copy
from functools import partial
from pathlib import Path
from typing import Any, NamedTuple

import hydra
import jax
import jax.numpy as jnp

from bmgpt.config import Config, config_post_init, mesh_from_config, register_configs
from bmgpt.data import (
    dataset_dataloader_factory,
    get_dataset_on_device,
)
from bmgpt.loggers import logger_factory
from bmgpt.model import Transformer, _transformer, init_kv_cache, init_model, model_spec
from bmgpt.optimizers import (
    grad_norm_and_clip,
    init_adam_state,
    opt_update_factory,
)
from bmgpt.sample import generate

register_configs()


# Setup for training loop
class TrainState(NamedTuple):
    params: Transformer
    opt_state: Any
    kv_cache: jax.Array


@jax.jit
def init_train_state(key, config: Config) -> TrainState:
    model_params = init_model(key, config)
    adam_state = jax.tree.map(partial(init_adam_state, config), model_params)
    cache = init_kv_cache(config)
    return TrainState(params=model_params, opt_state=adam_state, kv_cache=cache)


@hydra.main(
    version_base=None,
    config_path=str(Path("configs").absolute().resolve()),
    config_name="config",
)
def main(config: Config):
    # Config
    jax.distributed.initialize()
    config_post_init(config)
    mesh = mesh_from_config(config)
    Logger = logger_factory(config.experiment.logger_type)
    # TODO: Expose these somehow, parameter groups?
    config_sampling = copy.deepcopy(config)
    config_sampling.dataset.global_batch_size = 1
    config_sampling.sharding.data = []

    # Randomness
    key = jax.random.key(config.experiment.seed)
    key_params, key_data, key_sampling = jax.random.split(key, 3)

    # Data
    data, dataloader = dataset_dataloader_factory(config)
    batch_iter = get_dataset_on_device(config, dataloader(key_data, config, data), mesh)

    # Initialize state, configure optimization
    with jax.set_mesh(mesh):
        train_state = init_train_state(key_params, config)
    spec = model_spec(train_state.params)
    opt_update = opt_update_factory(config.optimizer.type)
    weight_decay_mask = jax.tree.map(lambda x, s: bool(s), train_state.params, spec)

    @partial(jax.jit, donate_argnums=2)
    def train_step(config: Config, batch, train_state: TrainState):
        def loss_fn(params: Transformer):
            inputs, targets = batch
            logits, _ = jax.vmap(partial(_transformer, config, params))(
                inputs, train_state.kv_cache
            )
            logits = logits.astype(config.model.compute_dtype.value)
            logprobs = jax.nn.log_softmax(logits, axis=-1)
            print(logits.shape)
            print(targets.shape)
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
            lambda i: jax.tree.map(lambda x, y: y[i], grad, update__opt_state),
            range(2),
        )
        params = jax.tree.map(lambda x, y: x + y, train_state.params, update)
        new_state = TrainState(
            params=params, opt_state=opt_state, kv_cache=train_state.kv_cache
        )

        metrics = {"loss": loss, "grad_norm": global_grad_norm}
        return metrics, new_state

    # Simple training loop
    prev_metrics = None
    with Logger(config) as logger:
        for step in range(config.optimizer.num_steps):
            batch = next(batch_iter)
            with jax.set_mesh(mesh):
                cur_metrics, train_state = train_step(config, batch, train_state)
            log_metrics, prev_metrics = prev_metrics, cur_metrics
            if log_metrics:
                log_metrics |= {"step": step}
                logger.log(log_metrics)

        # Evaluate
        # TODO: currently just samples 1 rollout
        with jax.set_mesh(mesh):
            prompt = jnp.array((1,))
            cache = init_kv_cache(config_sampling)[0]
            cache_size = 0

            output, cache, cache_size = generate(
                config_sampling,
                key_sampling,
                train_state.params,
                prompt,
                cache,
                cache_size,
            )
            print(f"Prompt: {prompt}")
            print(f"Cache size: {cache_size}")
            print(f"Generated text: {output}")


if __name__ == "__main__":
    main()
