import itertools as it
from typing import Iterator

import jax
import jax.numpy as jnp

from config import Config


def make_data(config: Config):
    # Simple data sequence!
    # Small, so mega replicate for ease
    text = "012345678987654321" * 1024
    seqs = [
        [int(c) for c in text[i : i + config.seq_len + 1]]
        for i in range(len(text) - config.seq_len - 1)
    ]
    data = jnp.array(seqs, dtype=jnp.int32)
    return data


# Simple dataloader: each host has the whole dataset
def dataloader(
    key, config: Config, data: jax.Array
) -> Iterator[tuple[jax.Array, jax.Array]]:
    key = jax.random.fold_in(key, jax.process_index())
    num_data = len(data)
    for step in it.count():
        key = jax.random.fold_in(key, step)
        offsets = jax.random.randint(key, (config.global_batch_size,), 0, num_data)
        yield (data.at[offsets, :-1].get(), data.at[offsets, 1:].get())
