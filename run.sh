#!/bin/bash

TPU_NAME="$1"
SSH_FLAGS='-A -o ForwardAgent=yes'
COMMANDS="cd nanocrate && git pull && git checkout train-loop-first-cut && uv run --extra tpu train.py"

gcloud compute tpus tpu-vm ssh "$TPU_NAME" \
  --ssh-flag="$SSH_FLAGS" \
  --command="$COMMANDS" \
  --worker=all
