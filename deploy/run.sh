#!/bin/bash

TPU_NAME="$1"
shift
SSH_FLAGS='-A -o ForwardAgent=yes'
COMMANDS="if [ ! -d \"baremetal-gpt\" ]; then git clone git@github.com:sdbuch/baremetal-gpt; fi \
    && export HYDRA_FULL_ERROR=1 \
    && cd baremetal-gpt \
    && git fetch \
    && git checkout train-loop-first-cut \
    && uv sync --extra tpu \
    && uv run train $@"

gcloud compute tpus tpu-vm ssh "$TPU_NAME" \
  --ssh-flag="$SSH_FLAGS" \
  --command="$COMMANDS" \
  --worker=all
