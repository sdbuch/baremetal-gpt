from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import tyro

from config import Config
from data.number_staircase import dataloader, make_data
from data.utils import get_dataset_on_device, split_data
from model import Transformer, _transformer, init_kv_cache, init_model_params


# Setup for training loop
class TrainState(NamedTuple):
    params: Transformer
    opt: None


@jax.jit
def init_train_state(key, config: Config) -> TrainState:
    return TrainState(params=init_model_params(key, config), opt=None)


def main():
    # Config
    config_args = {
        "num_vocab": 10,
        "num_layers": 4,
        "num_steps": 10**3,
        "lr": 1e-2,
    }
    config_sampling_args = config_args | {
        "global_batch_size": 1,  # one prompt
        "update_cache": True,  # inference mode
        "sharding_data": jax.P(),  # No parallelism atm
    }
    config = Config(**config_args)
    config_sampling = Config(**config_sampling_args)

    # Randomness
    key = jax.random.key(config.seed)
    key_params, key_data, key_sampling = jax.random.split(key, 3)

    # Data
    data = make_data(config)
    key_data, sk = jax.random.split(key_data)
    data = jax.random.permutation(sk, data, axis=0)
    Xtr, Xdev, Xte = split_data(data, 0.8, 0.1)
    batch = get_dataset_on_device(config, dataloader(key_data, config, Xtr))

    # Initialize state
    cache = init_kv_cache(config)
    train_state = init_train_state(key_params, config)

    # Start with SGD then adam (state)... then adamw (selective)... each challenges!
    @partial(jax.jit, donate_argnums=2)
    def train_step(config: Config, batch, train_state: TrainState):
        def loss_fn(params: Transformer):
            inputs, targets = batch
            logits, cache_out = jax.vmap(partial(_transformer, config, params))(
                inputs, cache
            )
            logits = logits.astype(config.compute_dtype)
            logprobs = jax.nn.log_softmax(logits, axis=-1)
            return -jnp.take_along_axis(logprobs, targets[..., None], axis=-1).mean()

        loss, grad = jax.value_and_grad(loss_fn)(train_state.params)
        new_state = TrainState(
            params=jax.tree.map(
                lambda x, y: x - config.lr * y, train_state.params, grad
            ),
            opt=None,
        )

        metrics = {"loss": loss}
        return metrics, new_state

    # Simple training loop
    prev_metrics = None
    for step in range(config.num_steps):
        cur_metrics, train_state = train_step(config, next(batch), train_state)
        log_metrics, prev_metrics = prev_metrics, cur_metrics
        if log_metrics:
            log_metrics |= {"step": step}
            print(
                *[f"{metric}: {val}" for metric, val in log_metrics.items()], sep="\t"
            )

    # Perform sampling!
    # Setup
    prompt = jnp.array((1, 2, 3, 4))
    cache = init_kv_cache(config_sampling)[0]

    @partial(jax.jit, static_argnums=(5,))
    def sample_one_token(config, key, x, cache_in, cache_size, temperature=1.0):
        y, cache_out = _transformer(config, train_state.params, x, cache_in, cache_size)
        logits = y.astype(config.compute_dtype)
        cache_size = cache_size + x.shape[-1]
        next_token = jnp.array((jax.random.categorical(key, logits[-1] / temperature),))
        return next_token, cache_out, cache_size

    # Prefill and generation loop
    cache_size = 0
    tokens_to_generate = 64
    temperature = 0.7
    output = prompt

    # Prefill
    key_sampling, sk = jax.random.split(key_sampling)
    next_token, cache, cache_size = sample_one_token(
        config_sampling, sk, prompt, cache, 0, temperature
    )
    output = jnp.concatenate((output, next_token))
    # Generation loop
    for step in range(tokens_to_generate):
        key_sampling, sk = jax.random.split(key_sampling)
        next_token, cache, cache_size = sample_one_token(
            config_sampling, sk, next_token, cache, cache_size, temperature
        )
        output = jnp.concatenate((output, next_token))

    print(output)


if __name__ == "__main__":
    tyro.cli(main)
