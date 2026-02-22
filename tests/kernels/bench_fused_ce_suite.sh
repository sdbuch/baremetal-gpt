#!/bin/bash
# Fused CE loss benchmark suite for TPU v4-64
#
# Runs a matrix of configs to compare bmgpt Pallas, Marin XLA streaming,
# and naive reference across model sizes. Captures xprof traces for each.
#
# Usage:
#   # From local machine (launches on all workers):
#   TPU_NAME=tpu-v4-64 ./tests/kernels/bench_fused_ce_suite.sh
#
#   # Or via gcloud directly:
#   gcloud compute tpus tpu-vm ssh tpu-v4-64 --worker=all \
#     --command='cd baremetal-gpt && bash tests/kernels/bench_fused_ce_suite.sh'

set -euo pipefail

BENCH="uv run python -u tests/kernels/bench_fused_ce.py"
TRACE_BASE="/tmp/jax-trace/bench_fused_ce"

echo "============================================================"
echo "Fused CE Loss Benchmark Suite"
echo "============================================================"

# ── 1. Production 740M config (default) ──────────────────────────────
# B=16, S=2048, D=1536, V=129024 — what we actually train with
echo ""
echo ">>> [1/5] 740M production config (B=16, S=2048, D=1536, V=129024)"
$BENCH --trace "$TRACE_BASE/740M"

# ── 2. Marin's small shape ───────────────────────────────────────────
# B=512, H=512, V=128256 (batch=1, pos=512)
# This is the primary shape in Marin PR #2951 benchmarks.
# Their v4 results: XLA fwd ~567k tok/s, custom VJP bwd ~210k tok/s
echo ""
echo ">>> [2/5] Marin small shape (B=1, S=512, D=512, V=128256)"
$BENCH --batch 1 --pos 512 --embed 512 --vocab 128256 \
  --trace "$TRACE_BASE/marin_small"

# ── 3. 7B config (H=4096) ───────────────────────────────────────────
# B=4, S=2048, D=4096, V=129024 (8192 tokens)
# Matches Marin's large shape (B=8192, H=4096, V=128256).
# Their v4 results: XLA value+grad ~89k tok/s
echo ""
echo ">>> [3/5] 7B config (B=4, S=2048, D=4096, V=129024)"
$BENCH --batch 4 --pos 2048 --embed 4096 --vocab 129024 \
  --trace "$TRACE_BASE/7B_small_batch"

# ── 4. 7B config, full batch ─────────────────────────────────────────
# B=16, S=2048, D=4096, V=129024 (32768 tokens)
# Matches Marin's XL shape (B=32768, H=4096, V=128256).
# Their v5p results: custom VJP ~84k tok/s
echo ""
echo ">>> [4/5] 7B full batch (B=16, S=2048, D=4096, V=129024)"
$BENCH --batch 16 --pos 2048 --embed 4096 --vocab 129024 \
  --trace "$TRACE_BASE/7B_full_batch"

# ── 5. Block size sweep on 740M config ───────────────────────────────
# Sweeps block_q in [128, 256, 512], block_kv in [128, 256, 384, 512].
# Tests whether 384 (noted as viable in dclm.yaml) beats the default 256.
echo ""
echo ">>> [5/5] Block size sweep (740M config)"
$BENCH --sweep

echo ""
echo "============================================================"
echo "Suite complete. Traces saved under $TRACE_BASE/"
echo "============================================================"
