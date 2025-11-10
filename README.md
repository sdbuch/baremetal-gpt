# Installation

With `uv` [(download)](https://docs.astral.sh/uv/getting-started/installation/),
installation is a cinch: for TPU support,

```
uv sync --extra tpu
```

and for CPU (debugging)

```
uv sync --extra cpu
```


## TPU VM setup

We include a bare-bones startup script in `/deploy/startup.sh` to provide as an
argument when creating Google Cloud TPU VMs (etc.) for the first time. It works
with the `run.sh` script mentioned below.

# Use

The project is configured to have `uv` install a project script for easy
training:
```
uv run train {Hydra overrides}
```

We also include an example launch script, `/deploy/run.sh`, for use on
Google Cloud TPU VMs.
For example, from the repository root, to train on a 16-chip (4 host) TPU v4 VM
with data parallel:

```
./deploy/run.sh tpu-v4-32 num_vocab=10 num_layers=4 num_steps=300 lr=3e-4 'mesh_shape=[16]'
```

# For More Information

Check out the associated series of blogs. Currently two are published:

- [SPMD in JAX #1: Sharding](https://sdbuchanan.com/blog/jax-1/)
- [SPMD in JAX #2: Transformers in Bare-Metal JAX](https://sdbuchanan.com/blog/jax-2/)
