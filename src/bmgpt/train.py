import copy
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
from bmgpt.model import Transformer, _transformer, init_kv_cache, init_model_params
from bmgpt.optimizers import get_opt_update_fn_from_enum, init_adam_state
from bmgpt.sample import generate

register_configs()


# Setup for training loop
class TrainState(NamedTuple):
    params: Transformer
    opt: Any


@jax.jit
def init_train_state(key, config: Config) -> TrainState:
    model_params = init_model_params(key, config)
    adam_state = jax.tree.map(partial(init_adam_state, config), model_params)
    return TrainState(params=model_params, opt=adam_state)


@hydra.main(
    version_base=None,
    config_path=str(Path("configs").absolute().resolve()),
    config_name="config",
)
def main(config: Config):
    # Config
    jax.distributed.initialize()
    config_post_init(config)
    opt_update = get_opt_update_fn_from_enum(config.optimizer_type)
    # TODO: Expose these somehow, parameter groups?
    config_sampling_args = {
        "global_batch_size": 1,  # one prompt
        "update_cache": True,  # inference mode
        "sharding_data": [],  # No parallelism atm
    }
    config_sampling = copy.deepcopy(config)
    for key, value in config_sampling_args.items():
        config_sampling.__setattr__(key, value)

    # Randomness
    key = jax.random.key(config.seed)
    key_params, key_data, key_sampling = jax.random.split(key, 3)

    # Data
    data = make_number_staircase_data(config)
    key_data, sk = jax.random.split(key_data)
    data = jax.random.permutation(sk, data, axis=0)
    Xtr, Xdev, Xte = split_data(data, 0.8, 0.1)
    batch = get_dataset_on_device(config, dataloader(key_data, config, Xtr))

    # Initialize state
    cache = init_kv_cache(config)
    train_state = init_train_state(key_params, config)

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
        new_params_and_opt = jax.tree.map(
            partial(opt_update, config), train_state.params, grad, train_state.opt
        )
        # Transpose the output tree to get param tree and state tree
        new_params, new_opt = map(
            lambda i: jax.tree.map(lambda x, y: y[i], grad, new_params_and_opt),
            range(2),
        )
        new_state = TrainState(params=new_params, opt=new_opt)

        metrics = {"loss": loss}
        return metrics, new_state

    # Simple training loop
    prev_metrics = None
    for step in range(config.num_steps):
        cur_metrics, train_state = train_step(config, next(batch), train_state)
        log_metrics, prev_metrics = prev_metrics, cur_metrics
        if log_metrics:
            log_metrics |= {"step": step}
            log_metrics |= {"pid": jax.process_index()}
            print(
                *[f"{metric}: {val}" for metric, val in log_metrics.items()], sep="\t"
            )

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
