import itertools as it
from enum import Enum
from pathlib import Path
from typing import Iterator

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding

from bmgpt.config import Config, DatasetConfig, DatasetName, SplitType


def dataset_dataloader_factory(config: DatasetConfig):
    match config.name:
        case DatasetName.MNIST:
            return (load_mnist(config), dataloader_without_replacement)
        case DatasetName.NUMBER_STAIRCASE:
            return (
                make_number_staircase_data(config),
                dataloader_with_replacement,
            )


def get_range_iterable(config: DatasetConfig):
    return range(config.epochs_to_loop) if config.epochs_to_loop >= 0 else it.count()


##################################
##           Datasets
##################################


def load_mnist(config: DatasetConfig):
    path = Path(config.path)
    data = jnp.load(path / ("mnist_" + config.split.value + ".npz"))
    return jnp.array(data["images"]), jnp.array(data["labels"])


def make_number_staircase_data(config: DatasetConfig):
    # Simple data sequence!
    # Small, so mega replicate for ease
    text = "012345678987654321" * 1024
    seqs = [
        [int(c) for c in text[i : i + config.seq_len + 1]]
        for i in range(len(text) - config.seq_len - 1)
    ]
    data = jnp.array(seqs, dtype=jnp.int32)
    Xtr, Xdev, Xte = split_data(data, 0.8, 0.1)
    match config.split:
        case SplitType.TRAIN:
            return Xtr, jnp.array(0)
        case SplitType.TEST:
            return Xte, jnp.array(0)
        case SplitType.VAL:
            return Xdev, jnp.array(0)


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


# Simple next-token prediction dataloader for text training:
# - Each host has the whole dataset (data input)
# - Random sampling without replacement to draw batches (consumes key)
# - Targets are the next token (expect data to be num_data x (T+1))
def dataloader_with_replacement(
    key, config: DatasetConfig, data: tuple[jax.Array, jax.Array]
) -> Iterator[tuple[jax.Array, jax.Array]]:
    inputs, _ = data
    num_data = len(inputs)
    key = jax.random.fold_in(key, jax.process_index())
    for step in it.count():
        key = jax.random.fold_in(key, step)
        offsets = jax.random.randint(
            key, (config.global_batch_size // jax.process_count(),), 0, num_data
        )
        yield (inputs.at[offsets, :-1].get(), inputs.at[offsets, 1:].get())


# General "shuffle, drop last, sample without replacement" distributed dataloader
# - Each host has the whole dataset (data and labels inputs)
# - Random sampling without replacement to draw batches (consumes key)
# - "drop_last" behavior (batches are always the same size, even last batch of epoch)
def dataloader_without_replacement(
    key, config: DatasetConfig, data: tuple[jax.Array, jax.Array]
) -> Iterator[tuple[jax.Array, jax.Array]]:
    inputs, labels = data
    inputs = inputs[
        : (len(inputs) // config.global_batch_size) * config.global_batch_size
    ]
    num_data = len(inputs)
    local_batch_size = config.global_batch_size // jax.process_count()
    range_iterable = get_range_iterable(config)
    for epoch in range_iterable:
        key = jax.random.fold_in(key, epoch)
        perm = jax.random.permutation(key, num_data)
        inputs, labels = inputs[perm], labels[perm]
        for step in range(num_data // config.global_batch_size):
            offset = jax.process_index() * num_data // jax.process_count()
            index = step + offset
            yield (
                inputs.at[index : index + local_batch_size].get(),
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


# Helper to just get the global batch iterator, if we don't need the local data/loader
def get_distributed_batch_iter(
    config: Config, dataset_config: DatasetConfig, key, mesh
):
    data, dataloader = dataset_dataloader_factory(dataset_config)
    print(data[0].sharding.is_fully_addressable)
    print(data[0].sharding.is_fully_replicated)
    return get_dataset_on_device(config, dataloader(key, dataset_config, data), mesh)
