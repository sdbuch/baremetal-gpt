#!/bin/bash
# Install AWS CLI on TPU VM workers
# USAGE: ./setup_aws_cli.sh <TPU_NAME>

TPU_NAME="$1"
SSH_FLAGS='-A -o ForwardAgent=yes'

COMMANDS='
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip -q awscliv2.zip
sudo ./aws/install
rm -rf aws awscliv2.zip
aws --version
'

gcloud compute tpus tpu-vm ssh "$TPU_NAME" \
  --ssh-flag="$SSH_FLAGS" \
  --command="$COMMANDS" \
  --worker=all
