#!/bin/bash
# USAGE: TPU VM name is first arg, second arg is --patch_size=<2, 4, or 7>

TPU_NAME="$1"
shift
SSH_FLAGS='-A -o ForwardAgent=yes'
COMMANDS="if [ ! -d \"baremetal-gpt\" ]; then git clone git@github.com:sdbuch/baremetal-gpt; fi \
    && cd baremetal-gpt \
    && git pull \
    && git checkout vision \
    && uv sync --extra tpu \
    && uv run data/download_and_preprocess_mnist.py $@"

gcloud compute tpus tpu-vm ssh "$TPU_NAME" \
  --ssh-flag="$SSH_FLAGS" \
  --command="$COMMANDS" \
  --worker=all
