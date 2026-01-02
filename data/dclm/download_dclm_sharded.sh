#!/bin/bash
# Download from private R2 bucket using credentials
# USAGE: ./download_dclm_sharded_private.sh <TPU_NAME>
# NOTE: We assume deterministic execution of jax.process_index() in this downloading scheme
#   Dataloading should fail explicitly if deterministic execution does not hold

TPU_NAME="$1"
SSH_FLAGS='-A -o ForwardAgent=yes'

COMMANDS="
export AWS_ACCESS_KEY_ID='$R2_ACCESS_KEY'
export AWS_SECRET_ACCESS_KEY='$R2_SECRET_KEY'

R2_ENDPOINT='https://6b9007962db3da17afc26054ded653eb.r2.cloudflarestorage.com'
BUCKET='dclm'
DATASET='dclm_tokshuf_1b'
OUTPUT_DIR='data/dclm'
TRAIN_SHARDS=2176
VAL_SHARDS=16

cd baremetal-gpt
PROCESS_INDEX=\$(uv run python -c 'import jax; jax.distributed.initialize(); print(jax.process_index())')
PROCESS_COUNT=\$(uv run python -c 'import jax; jax.distributed.initialize(); print(jax.process_count())')

echo \"Process \$PROCESS_INDEX of \$PROCESS_COUNT\"

mkdir -p \"\$OUTPUT_DIR/train\"
mkdir -p \"\$OUTPUT_DIR/val\"

# Download manifest
aws s3 cp --endpoint-url \"\$R2_ENDPOINT\" \
    \"s3://\${BUCKET}/\${DATASET}/manifest.jsonl\" \
    \"\$OUTPUT_DIR/manifest.jsonl\"

# Download training shards (strided across hosts)
echo \"Downloading training shards...\"
for ((i=PROCESS_INDEX; i<TRAIN_SHARDS; i+=PROCESS_COUNT)); do
    SHARD=\$(printf 'shard_%08d.tar' \$i)
    [ -f \"\$OUTPUT_DIR/train/\$SHARD\" ] && continue
    aws s3 cp --endpoint-url \"\$R2_ENDPOINT\" \
        \"s3://\${BUCKET}/\${DATASET}/\${SHARD}\" \
        \"\$OUTPUT_DIR/train/\${SHARD}\"
done

# Download validation shards (strided across hosts)
echo \"Downloading validation shards...\"
for ((i=TRAIN_SHARDS+PROCESS_INDEX; i<TRAIN_SHARDS+VAL_SHARDS; i+=PROCESS_COUNT)); do
    SHARD=\$(printf 'shard_%08d.tar' \$i)
    [ -f \"\$OUTPUT_DIR/val/\$SHARD\" ] && continue
    aws s3 cp --endpoint-url \"\$R2_ENDPOINT\" \
        \"s3://\${BUCKET}/\${DATASET}/\${SHARD}\" \
        \"\$OUTPUT_DIR/val/\${SHARD}\"
done

echo \"Downloaded \$((TRAIN_SHARDS / PROCESS_COUNT)) training shards to \$OUTPUT_DIR/train\"
echo \"Downloaded \$((VAL_SHARDS / PROCESS_COUNT)) validation shards to \$OUTPUT_DIR/val\"
"

gcloud compute tpus tpu-vm ssh "$TPU_NAME" \
  --ssh-flag="$SSH_FLAGS" \
  --command="$COMMANDS" \
  --worker=all
