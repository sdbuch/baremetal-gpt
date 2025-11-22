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

## TPU VM spinup

Code runs most directly in Google Cloud TPU VMs. You will need to set this up on
your own:

1. Create a project in the Google Cloud console
2. Install the `gcloud` CLI and associate it with your project and SSH keys
3. Create a TPU VM instance, e.g. using a command like
   ```
   gcloud compute tpus tpu-vm create tpu-v4-8 --zone=us-central2-b
   --accelerator-type=v4-8 --version=tpu-ubuntu2204-base
   --metadata-from-file="startup-script=$STARTUP_SCRIPT"
   ```
   Replace with your project's zone / desired TPU VM configuration.

See [here](https://docs.cloud.google.com/tpu/docs/managing-tpus-tpu-vm) for
setup documentation (and more generally
[here](https://docs.cloud.google.com/tpu/docs/quick-starts)).

We include a bare-bones startup script in `/deploy/startup.sh` to provide as an
argument when creating Google Cloud TPU VMs via `gcloud` for the first time,
like in the example above.
It enables the `run.sh` script mentioned below.

# Use

The project is configured to have `uv` install a project script for easy
training:

```
uv run train {Hydra overrides}
```

We also include an example launch script, `/deploy/run.sh`, for use on
Google Cloud TPU VMs.
For example, from the repository root, to train on a 16-chip (4 host) TPU v4 VM
`tpu-v4-32` with data parallel:

```
./deploy/run.sh tpu-v4-32 num_vocab=10 num_layers=4 num_steps=300 lr=3e-4 'mesh_shape=[16]'
```

# For More Information

Check out the associated series of blogs. Currently two are published:

- [SPMD in JAX #1: Sharding](https://sdbuchanan.com/blog/jax-1/)
- [SPMD in JAX #2: Transformers in Bare-Metal JAX](https://sdbuchanan.com/blog/jax-2/)
