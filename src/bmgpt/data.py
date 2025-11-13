import itertools as it
from typing import Iterator

import jax
import jax.numpy as jnp

# from jax._src.mesh import get_concrete_mesh
from jax.sharding import Mesh, NamedSharding

from bmgpt.config import Config, DatasetName


def dataset_dataloader_factory(config: Config):
    match config.dataset.name:
        case DatasetName.MNIST:
            return (None, None)
        case DatasetName.NUMBER_STAIRCASE:
            return (make_number_staircase_data(config), dataloader_ntp_no_replacement)


##################################
##        (Toy) Datasets
##################################


def make_number_staircase_data(config: Config):
    # Simple data sequence!
    # Small, so mega replicate for ease
    text = "012345678987654321" * 1024
    seqs = [
        [int(c) for c in text[i : i + config.dataset.seq_len + 1]]
        for i in range(len(text) - config.dataset.seq_len - 1)
    ]
    data = jnp.array(seqs, dtype=jnp.int32)
    Xtr, Xdev, Xte = split_data(data, 0.8, 0.1)
    return Xtr


def split_data(data: jax.Array, train_fraction: float, dev_fraction: float):
    num_data = len(data)
    Xtr = data[: int(train_fraction * num_data)]
    Xdev = data[
        int(train_fraction * num_data) : int((train_fraction + dev_fraction) * num_data)
    ]
    Xte = data[int((train_fraction + dev_fraction) * num_data) :]
    return Xtr, Xdev, Xte


##################################
##        Dataloaders
##################################


# Simple next-token prediction dataloader:
# - Each host has the whole dataset (data input)
# - Random sampling without replacement to draw batches (consumes key)
# - Targets are the next token (expect data to be num_data x (T+1))
def dataloader_ntp_no_replacement(
    key, config: Config, data: jax.Array
) -> Iterator[tuple[jax.Array, jax.Array]]:
    num_data = len(data)
    key = jax.random.fold_in(key, jax.process_index())
    for step in it.count():
        key = jax.random.fold_in(key, step)
        offsets = jax.random.randint(
            key, (config.dataset.global_batch_size // jax.process_count(),), 0, num_data
        )
        yield (data.at[offsets, :-1].get(), data.at[offsets, 1:].get())


# Simple classification dataloader:
# - Each host has the whole dataset (data and labels inputs)
# - Random sampling without replacement to draw batches (consumes key)
# - "drop_last" behavior (batches are always the same size, even last batch of epoch)
def dataloader_classification(
    key, config: Config, data: jax.Array, labels: jax.Array
) -> Iterator[tuple[jax.Array, jax.Array]]:
    data = data[
        : (len(data) // config.dataset.global_batch_size)
        * config.dataset.global_batch_size
    ]
    num_data = len(data)
    local_batch_size = config.dataset.global_batch_size // jax.process_count()
    # key = jax.random.fold_in(key, jax.process_index())
    for epoch in it.count():
        key = jax.random.fold_in(key, epoch)
        perm = jax.random.permutation(key, num_data)
        data, labels = data[perm], labels[perm]
        for step in range(num_data // config.dataset.global_batch_size):
            offset = jax.process_index() * num_data // jax.process_count()
            index = step + offset
            yield (
                data.at[index : index + local_batch_size].get(),
                labels.at[index : index + local_batch_size].get(),
            )


##################################
##     Distributed utils
##################################


# Helper to map a dataloader with make_array_from_process_local_data
def get_dataset_on_device(
    config: Config, dataloader: Iterator[tuple[jax.Array, jax.Array]], mesh: Mesh
) -> Iterator[tuple[jax.Array, jax.Array]]:
    return map(
        lambda batch: jax.make_array_from_process_local_data(
            NamedSharding(mesh, jax.P(*config.sharding.data)), batch
        ),
        dataloader,
    )
