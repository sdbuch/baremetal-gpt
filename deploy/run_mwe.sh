#!/bin/bash

TPU_NAME="$1"
shift
SSH_FLAGS='-A -o ForwardAgent=yes'
COMMANDS="if [ ! -d \"baremetal-gpt\" ]; then git clone git@github.com:sdbuch/baremetal-gpt; fi \
    && cd baremetal-gpt \
    && git fetch \
    && git checkout -f splash \
    && git pull \
    && uv sync --extra tpu \
    && uv run python tests/splash_mwe.py $@"

gcloud compute tpus tpu-vm ssh "$TPU_NAME" \
  --ssh-flag="$SSH_FLAGS" \
  --command="$COMMANDS" \
  --worker=all
