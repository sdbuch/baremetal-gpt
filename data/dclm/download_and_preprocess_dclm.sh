#!/bin/bash
# DCLM-Baseline 1B-1x: Download and tokenization commands
# Run from: rust_processing/tokshuf-rs/
#
# See the README.md at https://github.com/mlfoundations/dclm
# Requires AWS cli setup (download), rust (preprocessing)

# Download preprocessed data from CommonCrawl S3
aws s3 cp --recursive s3://commoncrawl/contrib/datacomp/DCLM-baseline/global-shard_03_of_10/local-shard_1_of_10/ dclm_local

# Build tokenizer
cargo build --release

# Create temp/output directories
mkdir -p tokshuf_tmp dclm_tokshuf

# Tokenize and shuffle with Llama 3 tokenizer
cargo run --release -- \
    --input dclm_local \
    --local-cell-dir tokshuf_tmp \
    --output dclm_tokshuf \
    --tokenizer "meta-llama/Meta-Llama-3-8B" \
    --use_tiktoken \
    --seqlen 2049 \
    --wds-chunk-size 8192 \
    --num-local-cells 512
