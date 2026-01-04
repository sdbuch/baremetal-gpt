import itertools as it
from pathlib import Path
from typing import Any, Callable, Iterator

import jax
import jax.numpy as jnp
import webdataset as wds
from jax import Array
from jax.sharding import Mesh, NamedSharding

from bmgpt.config import Config, DatasetConfig, DatasetName, SplitType

##################################
##           Helpers
##################################


def dataset_dataloader_factory(config: DatasetConfig):
  match config.name:
    case DatasetName.MNIST:
      return (load_mnist(config), dataloader_without_replacement)
    case DatasetName.NUMBER_STAIRCASE:
      return (make_number_staircase_data(config), dataloader_with_replacement)
    case DatasetName.SHAKESPEARE:
      return (load_shakespeare(config), dataloader_with_replacement)
    case DatasetName.DCLM:
      return (load_dclm(config), dataloader_dclm)


def get_range_iterable(config: DatasetConfig):
  return range(config.epochs_to_loop) if config.epochs_to_loop >= 0 else it.count()


##################################
##           APIs/Types
##################################

DataloaderOutputType = Iterator[tuple[Array, Array]]
DataloaderType = Callable[[Any, DatasetConfig, Array], DataloaderOutputType]


##################################
##           Datasets
##################################


def load_mnist(config: DatasetConfig):
  path = Path(config.path)
  if config.split == SplitType.VAL:
    load_str = "test"
    start = 0
    size = 4000
  else:
    load_str = config.split.value
    if config.split == SplitType.TEST:
      start = 4000
      size = 6000
    else:
      start = 0
      size = 60000
  data = jnp.load(path / ("mnist_" + load_str + ".npz"))
  inputs = jnp.array(data["images"][start : start + size]).astype(jnp.bfloat16)
  labels = jnp.array(data["labels"][start : start + size])
  return inputs, labels[:, None]


def load_shakespeare(config: DatasetConfig):
  path = Path(config.path)
  data = jnp.load(path / (config.split.value + ".npy"))
  return data, jnp.array(0)


def load_dclm(config: DatasetConfig):
  # For streaming support, just returns the list of local shards
  if config.split == SplitType.TRAIN:
    data_dir = Path(config.path) / "train"
  elif config.split == SplitType.VAL:
    data_dir = Path(config.path) / "val"
  else:
    raise NotImplementedError
  shard_files = sorted(data_dir.glob("shard_*.tar"))
  shard_urls = [str(f) for f in shard_files]
  return shard_urls


def make_number_staircase_data(config: DatasetConfig):
  # Simple data sequence!
  # Small, so mega replicate for ease
  text = "012345678987654321" * 1024
  ids = [int(c) for c in text]
  data = jnp.array(ids, dtype=jnp.int32)
  Xtr, Xdev, Xte = split_data(data, 0.8, 0.1)
  match config.split:
    case SplitType.TRAIN:
      return Xtr, jnp.array(0)
    case SplitType.TEST:
      return Xte, jnp.array(0)
    case SplitType.VAL:
      return Xdev, jnp.array(0)


def split_data(data: Array, train_fraction: float, dev_fraction: float):
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


# DCLM-specific dataloader
# - We use webdataset for this, as DCLM preprocessing outputs in wds format!
# - Data is pre-shuffled and we currently don't re-shuffle it (doesn't use key)
# - Can't load all data into RAM, so doesn't follow the same API as other loaders
# - Data is sharded across hosts in advance (see load_dclm)
# - DCLM Llama3 tokenizer is used and is weird about the pad token (handle it here)
def dataloader_dclm(
  key, config: DatasetConfig, data: list[str]
) -> DataloaderOutputType:
  PAD_TOKEN = 128258
  local_batch_size = config.global_batch_size // jax.process_count()
  s = config.seq_len

  to_jax_array = lambda x: jnp.array(x, dtype=jnp.int32)

  dataloader = (
    wds.WebDataset(data, shardshuffle=False)  # type: ignore[attr-defined]
    .decode()  # Auto-decompress gzip and decode JSON
    .to_tuple("json.gz")  # Extract json.gz key-value, no metadata, as (1,) shape tuple
    .select(lambda ctx: all(token != PAD_TOKEN for token in ctx[0]))  # no PAD tokens
    .map(lambda x: map(to_jax_array, (x[0][:s], x[0][1 : s + 1])))  # to JAX array
    .batched(
      local_batch_size,
      partial=False,
      collation_fn=lambda batches: [jnp.stack(x) for x in zip(*batches)],
    )
  )
  yield from dataloader


# Simple next-token prediction dataloader for text training:
# - Each host has the whole dataset (data input)
# - Random sampling without replacement to draw batches (consumes key)
# - Targets are the next token (expect data to be (num_data,) shape)
def dataloader_with_replacement(
  key, config: DatasetConfig, data: tuple[Array, Array]
) -> DataloaderOutputType:
  inputs, _ = data
  num_data = len(inputs)
  key = jax.random.fold_in(key, jax.process_index())
  for step in it.count():
    key = jax.random.fold_in(key, step)
    offsets = jax.random.randint(
      key,
      (config.global_batch_size // jax.process_count(),),
      0,
      num_data - config.seq_len - 1,
    )
    seqs = inputs.at[offsets[:, None] + jnp.arange(config.seq_len)].get()
    targets = inputs.at[1 + offsets[:, None] + jnp.arange(config.seq_len)].get()
    yield seqs, targets


# General "shuffle, drop last, sample without replacement" distributed dataloader
# - Each host has the whole dataset (data and labels inputs)
# - Random sampling without replacement to draw batches (consumes key)
# - "drop_last" behavior (batches are always the same size, even last batch of epoch)
def dataloader_without_replacement(
  key, config: DatasetConfig, data: tuple[Array, Array]
) -> DataloaderOutputType:
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
      index = step * local_batch_size + offset
      yield (
        inputs.at[index : index + local_batch_size].get(),
        labels.at[index : index + local_batch_size].get(),
      )


##################################
##     Distributed utils
##################################


# Helper to map a dataloader with make_array_from_process_local_data
def get_dataset_on_device(
  dataloader: DataloaderOutputType, mesh: Mesh, sharding_data: list[str | None]
) -> DataloaderOutputType:
  return map(
    lambda batch: jax.make_array_from_process_local_data(
      NamedSharding(mesh, jax.P(sharding_data)), batch
    ),
    dataloader,
  )


# Helper to just get the global batch iterator, if we don't need the local data/loader
def get_distributed_batch_iter(
  config: Config, dataset_config: DatasetConfig, key, mesh
):
  data, dataloader_factory = dataset_dataloader_factory(dataset_config)
  dataloader = dataloader_factory(key, dataset_config, data)
  sharding_data = config.sharding.data
  if dataset_config.num_microbatches > 0:
    # Microbatch the batch axis for gradient accumulation
    num_microbatches = dataset_config.num_microbatches
    microbatch_size = dataset_config.global_batch_size // num_microbatches

    def make_microbatch(batch):
      return jax.tree.map(
        lambda x: x.reshape(num_microbatches, microbatch_size, -1), batch
      )

    dataloader = map(make_microbatch, dataloader)
    sharding_data = [None] + sharding_data
  return get_dataset_on_device(dataloader, mesh, sharding_data)
