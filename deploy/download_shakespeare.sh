#!/bin/bash
# USAGE: TPU VM name is first arg

TPU_NAME="$1"
shift
SSH_FLAGS='-A -o ForwardAgent=yes'
COMMANDS="if [ ! -d \"baremetal-gpt\" ]; then git clone git@github.com:sdbuch/baremetal-gpt; fi \
    && export HF_TOKEN='$HF_TOKEN' \
    && cd baremetal-gpt \
    && git fetch \
    && git checkout -f main \
    && git pull \
    && uv sync --extra tpu \
    && uv run data/tiny-shakespeare/download_and_tokenize_tiny_shakespeare.py $@"

gcloud compute tpus tpu-vm ssh "$TPU_NAME" \
  --ssh-flag="$SSH_FLAGS" \
  --command="$COMMANDS" \
  --worker=all
