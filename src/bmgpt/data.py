import itertools as it
from typing import Iterator

import jax
import jax.numpy as jnp

# from jax._src.mesh import get_concrete_mesh
from jax.sharding import Mesh, NamedSharding

from bmgpt.config import Config


def make_number_staircase_data(config: Config):
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
    num_data = len(data)
    key = jax.random.fold_in(key, jax.process_index())
    for step in it.count():
        key = jax.random.fold_in(key, step)
        offsets = jax.random.randint(
            key, (config.global_batch_size // jax.process_count(),), 0, num_data
        )
        yield (data.at[offsets, :-1].get(), data.at[offsets, 1:].get())


def split_data(data: jax.Array, train_fraction: float, dev_fraction: float):
    num_data = len(data)
    Xtr = data[: int(train_fraction * num_data)]
    Xdev = data[
        int(train_fraction * num_data) : int((train_fraction + dev_fraction) * num_data)
    ]
    Xte = data[int((train_fraction + dev_fraction) * num_data) :]
    return Xtr, Xdev, Xte


def get_dataset_on_device(
    config: Config, dataloader: Iterator[tuple[jax.Array, jax.Array]], mesh: Mesh
) -> Iterator[tuple[jax.Array, jax.Array]]:
    return map(
        lambda batch: jax.make_array_from_process_local_data(
            NamedSharding(mesh, jax.P(*config.sharding_data)), batch
        ),
        dataloader,
    )
