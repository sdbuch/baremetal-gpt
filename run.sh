#!/bin/bash

TPU_NAME="$1"
SSH_FLAGS='-A -o ForwardAgent=yes'
COMMANDS="cd nanocrate && git pull && git checkout trc-infra-setup && uv run --extra tpu main.py"

gcloud compute tpus tpu-vm ssh "$TPU_NAME" \
  --ssh-flag="$SSH_FLAGS" \
  --command="$COMMANDS" \
  --worker=all
