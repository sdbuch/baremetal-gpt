import itertools as it
from typing import Iterator

import jax
import jax.numpy as jnp
from jax._src import xla_bridge
from jax._src.mesh import get_concrete_mesh
from jax.sharding import NamedSharding

from bmgpt.config import Config


def make_number_staircase_data(config: Config):
    # Simple data sequence!
    # Small, so mega replicate for ease
    text = "012345678987654321" * 1024
    seqs = [
        [int(c) for c in text[i : i + config.seq_len + 1]]
        for i in range(len(text) - config.seq_len - 1)
    ]
    data = jnp.array(
        seqs,
        dtype=jnp.int32,
        device=NamedSharding(get_concrete_mesh(), jax.P()),
    )
    return data


# Simple dataloader: each host has the whole dataset
def dataloader(
    key, config: Config, data: jax.Array
) -> Iterator[tuple[jax.Array, jax.Array]]:
    key = jax.random.fold_in(key, jax.process_index())
    num_data = len(data)
    for step in it.count():
        key = jax.random.fold_in(key, step)
        # offsets = jax.random.randint(key, (config.global_batch_size,), 0, num_data)
        offsets = jax.random.randint(
            key, (config.global_batch_size // jax.process_count(),), 0, num_data
        )
        print(jax.local_devices())
        print(jax.typeof(offsets))
        print(offsets.devices())
        print(offsets.is_fully_addressable)
        print(offsets.is_fully_replicated)
        print(offsets)
        # print(xla_bridge.process_count())
        # print(
        #     NamedSharding(
        #         get_concrete_mesh(), jax.P(*config.sharding_data)
        #     )._internal_device_list
        # )
        # print(
        #     NamedSharding(
        #         get_concrete_mesh(), jax.P(*config.sharding_data)
        #     )._internal_device_list.process_indices
        # )
        # print(
        #     NamedSharding(
        #         get_concrete_mesh(), jax.P(*config.sharding_data)
        #     ).is_fully_addressable
        # )
        # print(
        #     NamedSharding(get_concrete_mesh(), jax.P(*config.sharding_data)).device_set
        # )
        # print(data.at[offsets, :-1].get().sharding._internal_device_list)
        # print(data.at[offsets, :-1].get().is_fully_addressable)
        # print(data.at[offsets, :-1].get().sharding.device_set)
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
    config: Config, dataloader: Iterator[tuple[jax.Array, jax.Array]]
) -> Iterator[tuple[jax.Array, jax.Array]]:
    return map(
        # lambda batch: jax.device_put(
        #     batch, NamedSharding(get_concrete_mesh(), jax.P(*config.sharding_data))
        # ),
        # dataloader,
        lambda batch: jax.make_array_from_process_local_data(
            NamedSharding(get_concrete_mesh(), jax.P(*config.sharding_data)), batch
        ),
        dataloader,
    )
