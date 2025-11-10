import copy
import operator
from functools import partial
from pathlib import Path
from typing import Any, NamedTuple

import hydra
import jax
import jax.numpy as jnp

from bmgpt.config import Config, config_post_init, register_configs
from bmgpt.data import (
    dataloader,
    get_dataset_on_device,
    make_number_staircase_data,
    split_data,
)
from bmgpt.loggers import get_logger_class_from_enum
from bmgpt.model import Transformer, _transformer, init_kv_cache, init_model, model_spec
from bmgpt.optimizers import (
    get_opt_update_fn_from_enum,
    grad_norm_and_clip,
    init_adam_state,
)
from bmgpt.sample import generate

register_configs()


# Setup for training loop
class TrainState(NamedTuple):
    params: Transformer
    opt_state: Any


@jax.jit
def init_train_state(key, config: Config) -> TrainState:
    model_params = init_model(key, config)
    adam_state = jax.tree.map(partial(init_adam_state, config), model_params)
    return TrainState(params=model_params, opt_state=adam_state)


@hydra.main(
    version_base=None,
    config_path=str(Path("configs").absolute().resolve()),
    config_name="config",
)
def main(config: Config):
    # Config
    jax.distributed.initialize()
    kk = jax.random.key(1337)
    print(kk.is_fully_addressable)
    config_post_init(config)
    kkk = jax.random.key(1337)
    print(kkk.is_fully_addressable)
    kkkk = jax.random.fold_in(kkk, jax.process_index())
    print(kkkk.is_fully_addressable)
    Logger = get_logger_class_from_enum(config.logger_type)
    # TODO: Expose these somehow, parameter groups?
    config_sampling_args = {
        "global_batch_size": 1,  # one prompt
        "update_cache": True,  # inference mode
        "sharding_data": [],  # No parallelism atm
    }
    config_sampling = copy.deepcopy(config)
    for k, v in config_sampling_args.items():
        config_sampling.__setattr__(k, v)

    # Randomness
    key = jax.random.key(config.seed)
    print(key.is_fully_addressable)
    key_params, key_data, key_sampling = jax.random.split(key, 3)

    # Data
    data = make_number_staircase_data(config)
    key_data, sk = jax.random.split(key_data)
    data = jax.random.permutation(sk, data, axis=0)
    Xtr, Xdev, Xte = split_data(data, 0.8, 0.1)
    print(key_data.is_fully_addressable)
    batch = get_dataset_on_device(config, dataloader(key_data, config, Xtr))

    # Initialize state, configure optimization
    cache = init_kv_cache(config)
    train_state = init_train_state(key_params, config)
    spec = model_spec(train_state.params)
    opt_update = get_opt_update_fn_from_enum(config.optimizer_type)
    weight_decay_mask = jax.tree.map(lambda x, s: bool(s), train_state.params, spec)

    @partial(jax.jit, donate_argnums=2)
    def train_step(config: Config, batch, train_state: TrainState):
        def loss_fn(params: Transformer):
            inputs, targets = batch
            logits, cache_out = jax.vmap(partial(_transformer, config, params))(
                inputs, cache
            )
            logits = logits.astype(config.compute_dtype.value)
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
            lambda i: jax.tree.map(lambda x, y: y[i], grad, update__opt_state),
            range(2),
        )
        params = jax.tree.map(lambda x, y: x + y, train_state.params, update)
        new_state = TrainState(params=params, opt_state=opt_state)

        metrics = {"loss": loss, "grad_norm": global_grad_norm}
        return metrics, new_state

    # Simple training loop
    prev_metrics = None
    with Logger(config) as logger:
        for step in range(config.num_steps):
            cur_metrics, train_state = train_step(config, next(batch), train_state)
            log_metrics, prev_metrics = prev_metrics, cur_metrics
            if log_metrics:
                log_metrics |= {"step": step}
                logger.log(log_metrics)

        # Perform sampling
        prompt = jnp.array((1,))
        cache = init_kv_cache(config_sampling)[0]
        cache_size = 0

        output, cache, cache_size = generate(
            config_sampling, key_sampling, train_state.params, prompt, cache, cache_size
        )
        print(f"Prompt: {prompt}")
        print(f"Cache size: {cache_size}")
        print(f"Generated text: {output}")


if __name__ == "__main__":
    main()
