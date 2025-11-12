#!/bin/bash

TPU_NAME="$1"
SSH_FLAGS='-A -o ForwardAgent=yes'

PYTHON_SCRIPT='
import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding, Mesh, AxisType, PartitionSpec as P

# Initialize JAX distributed
jax.distributed.initialize()

process_index = jax.process_index()
num_processes = jax.process_count()

print(f"Process {process_index}/{num_processes}")
print(f"Mesh devices: {len(jax.devices())}")

# Create mesh with all devices
devices = jax.devices()
mesh = jax.make_mesh((16,), ("data",), (AxisType.Explicit,))

seed = 42

arr = jnp.arange(1024).reshape(256, 4)
arr = jax.device_put(arr, NamedSharding(mesh, P()))

# Test 1: WITHOUT folding in process index
print("=" * 60)
print(f"Process {process_index}: WITHOUT fold_in")
print("=" * 60)
key = jax.random.key(seed)
local_array_idxs = jax.random.randint(key, (64,), 0, arr.shape[0])
local_array = arr.at[local_array_idxs].get()
print(f"Process {process_index} array (first 10): {local_array[:10]}")

sharding = NamedSharding(mesh, P("data"))
sharded_array = jax.device_put(local_array, sharding)
print("Arr addressable:", arr.is_fully_addressable)
print("Local arr addressable:", local_array.is_fully_addressable)
print("Sharded arr addressable:", sharded_array.is_fully_addressable)
print("Sharding addressable:", sharding.is_fully_addressable)

# Test 2: WITH folding in process index
print("\n" + "=" * 60)
print(f"Process {process_index}: WITH fold_in")
print("=" * 60)
key = jax.random.key(seed)
print(key.is_fully_addressable)
key = jax.random.fold_in(key, process_index)
print(key.is_fully_addressable)
local_array_idxs = jax.random.randint(key, (64,), 0, arr.shape[0])
local_array = arr.at[local_array_idxs].get()
print(f"Process {process_index} array (first 10): {local_array[:10]}")
print(f"Idxs addressable:", local_array_idxs.is_fully_addressable)

sharding = NamedSharding(mesh, P("data"))
sharded_array = jax.device_put(local_array, sharding)
print("Arr addressable:", arr.is_fully_addressable)
print("Local arr addressable:", local_array.is_fully_addressable)
print("Sharded arr addressable:", sharded_array.is_fully_addressable)
print("Sharding addressable:", sharding.is_fully_addressable)

# Output
arr_out = jax.device_put(sharded_array, NamedSharding(mesh, jax.P()))
print(f"Post-Sharded array {process_index} (first 10): {arr_out[:10]}")
'

COMMANDS="mkdir -p tmp && cd tmp && uv run python -c '$PYTHON_SCRIPT'"

gcloud compute tpus tpu-vm ssh "$TPU_NAME" \
  --ssh-flag="$SSH_FLAGS" \
  --command="$COMMANDS" \
  --worker=all
