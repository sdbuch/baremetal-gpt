#!/bin/bash

TPU_NAME="$1"
shift
SSH_FLAGS='-A -o ForwardAgent=yes'
COMMANDS="if [ ! -d \"nanocrate\" ]; then git clone git@github.com:sdbuch/nanocrate; fi \
    && source .local/bin/env \
    && cd nanocrate \
    && git pull \
    && git checkout train-loop-first-cut \
    && uv run --extra tpu train.py $@"

gcloud compute tpus tpu-vm ssh "$TPU_NAME" \
  --ssh-flag="$SSH_FLAGS" \
  --command="$COMMANDS" \
  --worker=all
