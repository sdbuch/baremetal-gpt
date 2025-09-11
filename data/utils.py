from functools import partial
from typing import Iterator

import jax

from config import Config


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
):
    return map(
        partial(jax.make_array_from_process_local_data, config.sharding_data),
        dataloader,
    )
