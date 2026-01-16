# Fused Cross-Entropy Backward Pass Debug Session

## High-Level Goal
Debug a training quality degradation issue when using `use_fused_xent_loss=True` with `num_microbatches=2`. The symptom is that forward pass (batch_loss) matches between fused and non-fused at step 0, but gradients differ (grad_norm differs by ~0.2), leading to training divergence.

## Hypothesis
The bug is in the backward pass of the fused softmax cross-entropy kernel (`src/bmgpt/kernels/lse_kernel.py`). Key suspects:
1. Precision mismatch in label_logits backward path (einsum backward may accumulate in bfloat16)
2. Issues with how `d_lse_q = g_lse[:, :, None] * q` is computed for dK
3. Microbatch-specific tensor sizes hitting kernel edge cases

## Progress So Far

### 1. Initial Analysis
- Explored codebase to understand fused loss implementation
- Identified that forward matches but backward differs → bug is in backward pass
- Found recent precision fix in `losses.py` (einsum with preferred_element_type=float32)

### 2. Debug Infrastructure (Current)
Modified `src/bmgpt/train.py` to:
- Disable JIT on `train_step` for debugging
- Save loss function inputs at step 0:
  - Full batch (inputs, targets) with microbatch axis
  - Unembedding weights
  - Per-microbatch outputs/targets via `jax.debug.callback`
- Verify reconstruction from saved shards matches original arrays

Key functions added:
- `debug_save_batch_and_weights()` - saves before scan (outside traced context)
- `debug_save_microbatch()` - saves inside scan via `jax.debug.callback`
- `debug_verify_reconstruction()` - verifies roundtrip after scan

### 3. Issues Encountered & Fixed
- `jax.lax.scan` traces function body even without JIT → used `jax.debug.callback`
- bfloat16 not supported by numpy (becomes void dtype) → convert to float32 for saving, store original dtype, convert back on load

## Current Task
**Fix the verification bug** - there's still an error in the reconstruction/verification code. Need to:
1. Run the debug job and capture the error
2. Fix the issue
3. Verify that saved data can be correctly reconstructed and matches originals

## Next Steps (After Verification Works)
1. Create offline debug script to:
   - Load saved data
   - Step through fused vs non-fused loss functions
   - Compare gradients at each stage (dQ, dK, label_logits grad)
2. Identify exactly where gradients diverge
3. Fix the root cause in the kernel backward pass

## How to Run Debug Job
```bash
./deploy/run.sh tpu-v4-64 --multirun +deploy=v4-32 +sharding=fsdp +model=290M +experiment=dclm seed=420 train_dataset.num_microbatches=2 use_fused_xent_loss=True train_dataset.num_steps=1 'eval_list=[]'
```

To capture output for debugging:
```bash
./deploy/run.sh tpu-v4-64 ... 2>&1 | tee /tmp/run_output.txt
# Then grep for errors:
grep -A5 "Error\|Traceback" /tmp/run_output.txt | head -50
tail -100 /tmp/run_output.txt
```

## Important Files
- `src/bmgpt/train.py` - debug instrumentation added here
- `src/bmgpt/losses.py` - fused_softmax_cross_entropy function
- `src/bmgpt/kernels/lse_kernel.py` - LSE kernel with custom VJP
- `deploy/run.sh` - deployment script (pulls from fused-ce-debug-bwd branch)

## Workflow for Debugging
1. Make changes locally
2. Commit and push to `fused-ce-debug-bwd` branch
3. Run `./deploy/run.sh ...` which pulls code on remote TPU hosts
4. Check output for errors
5. Iterate

The run.sh script does:
```bash
git fetch && git checkout -f fused-ce-debug-bwd && git pull && uv sync --extra tpu && uv run train $@
```

## Session Context
- Running on 8-host TPU v4-32 setup
- Debug output prefixed with `[DEBUG][H0]` etc for easy grep
- Saved files go to `/tmp/debug_loss_inputs/proc_N/` on each host
- Global counter `_debug_microbatch_counter` persists across Hydra multirun jobs

## Permissions Needed for Autonomous Debugging
1. Run bash commands (for git commit/push, running deploy script)
2. Edit files in the repo
3. Read output files
