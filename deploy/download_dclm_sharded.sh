#!/bin/bash
# Download from private R2 bucket using credentials
# USAGE: ./download_dclm_sharded_private.sh <TPU_NAME>

TPU_NAME="$1"
SSH_FLAGS='-A -o ForwardAgent=yes'

COMMANDS="
export AWS_ACCESS_KEY_ID='$R2_ACCESS_KEY'
export AWS_SECRET_ACCESS_KEY='$R2_SECRET_KEY'

R2_ENDPOINT='https://6b9007962db3da17afc26054ded653eb.r2.cloudflarestorage.com'
BUCKET='dclm'
DATASET='dclm_tokshuf_1b'
OUTPUT_DIR='data/dclm'
TOTAL_SHARDS=2192

cd baremetal-gpt
PROCESS_INDEX=\$(uv run python -c 'import jax; print(jax.process_index())')
PROCESS_COUNT=\$(uv run python -c 'import jax; print(jax.process_count())')

echo \"Process \$PROCESS_INDEX of \$PROCESS_COUNT\"

mkdir -p \"\$OUTPUT_DIR\"

if [ \"\$PROCESS_INDEX\" -eq 0 ]; then
    aws s3 cp --endpoint-url \"\$R2_ENDPOINT\" \
        \"s3://\${BUCKET}/\${DATASET}/manifest.jsonl\" \
        \"\$OUTPUT_DIR/manifest.jsonl\"
fi

for ((i=PROCESS_INDEX; i<TOTAL_SHARDS; i+=PROCESS_COUNT)); do
    SHARD=\$(printf 'shard_%08d.tar' \$i)
    [ -f \"\$OUTPUT_DIR/\$SHARD\" ] && continue
    aws s3 cp --endpoint-url \"\$R2_ENDPOINT\" \
        \"s3://\${BUCKET}/\${DATASET}/\${SHARD}\" \
        \"\$OUTPUT_DIR/\${SHARD}\"
done

echo \"Downloaded \$((TOTAL_SHARDS / PROCESS_COUNT)) shards to \$OUTPUT_DIR\"
"

gcloud compute tpus tpu-vm ssh "$TPU_NAME" \
  --ssh-flag="$SSH_FLAGS" \
  --command="$COMMANDS" \
  --worker=all
