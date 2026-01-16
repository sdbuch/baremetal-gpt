# Plan: Fused dQ/dK Backward Kernel for LSE

## Executive Summary

**Discovery**: The original splash attention already has fused backward support via `use_fused_bwd_kernel=True`. We deleted this when porting to the LSE kernel. We can reintegrate it.

**Key tradeoff**: Fused backward saves compute (one P computation instead of two) but costs memory (intermediate dQ storage proportional to `kv_seq_len // bkv`).

## What Splash Attention Does

### Fused Backward in `_flash_attention_dkv_kernel` (lines 1673-1858)

When `use_fused_bwd_kernel=True`, the kernel computes **all three gradients from one P**:

```python
# P computed ONCE (line 1791)
p = jnp.exp(qk - logsumexp)

# dV (line 1792)
dv = lax.dot(p.astype(do.dtype), do, ...)

# ds for dK and dQ (lines 1796-1800)
dp = lax.dot_general(v, do, NT_DIM_NUMBERS, ...)
ds = (dp - di) * p

# dK (lines 1806-1811)
dk = lax.dot_general(ds.astype(do.dtype), q, dk_dims, ...)

# dQ - CONDITIONALLY in fused mode (lines 1812-1823)
if dq_scratch_ref is not None or dq_ref is not None:
    dq = lax.dot_general(ds.T.astype(k.dtype), k, NN_DIM_NUMBERS, ...)
    dq_ref[...] = dq.astype(dq_ref.dtype)
```

### The Output Shape Trick (lines 1998-2002, 2235-2237)

The fused kernel solves the accumulation asymmetry by **writing dQ partials indexed by kv_block**:

```python
# Output shape includes kv_block dimension
if use_fused_bwd_kernel:
    dq_spec = pl.BlockSpec((None, None, bq, head_dim_qk), dq_index_map)
    dq_shape = jax.ShapeDtypeStruct((kv_seq_len // bkv, *q.shape), q.dtype)
    #                                ^^^^^^^^^^^^^^^^^ extra leading dim

# After kernel: reduce over kv_blocks
if use_fused_bwd_kernel:
    dq = dq_unreduced.sum(axis=0)
```

## Why It's Not the Default

`use_fused_bwd_kernel` defaults to `False` in splash attention (line 516). The tradeoffs:

| Aspect | Fused (`True`) | Separate (`False`) |
|--------|---------------|-------------------|
| P computation | 1× | 2× |
| Q, K loads | 1× per pair | 2× per pair |
| dQ intermediate memory | O(kv_blocks × heads × q_seq × head_dim) | None |
| Block size flexibility | Single block size | Separate dQ vs dKV tuning |
| Sparse attention | Can't shrink grid | Can shrink grid |

**Memory example**: seq_len=8192, bkv=128, heads=32, head_dim=128
- dQ intermediate: `64 × 32 × 8192 × 128 × 4 bytes = 8.5 GB`

For long sequences, this memory overhead is prohibitive, explaining the default.

## What We Deleted in LSE Kernel

When porting splash → LSE kernel, we:

1. ✗ Removed the conditional dQ computation from dK kernel (splash lines 1812-1823)
2. ✗ Removed the `dq_spec` with kv_block indexing (splash lines 1998-2009)
3. ✗ Removed the `dq_unreduced.sum(axis=0)` reduction (splash line 2237)
4. ✗ Created completely separate `_lse_backward_dq_kernel` and `_lse_backward_dk_kernel`

**Result**: Our LSE kernel always computes P twice, even though the fused option exists upstream.

## Reintegration Plan

### Step 1: Modify `_lse_backward_dk_kernel` (lines 1013-1161)

Add optional dQ computation. The key difference from splash: our `ds = d_lse * p` (no Jacobian correction like splash's `ds = (dp - di) * p`).

```python
def _lse_backward_dk_kernel(
    ...
    # Add new outputs
    dq_scratch_ref,  # NEW: (bq, head_dim) or None
    dq_ref,          # NEW: (None, None, bq, head_dim) indexed by (kv_idx, h, q_idx)
    ...
):
    # ... existing dk computation ...

    # After computing p (around line 1134):
    # p = jnp.exp(qk - logsumexp)  # This is P^T (bkv, bq)

    # Existing dK computation (lines 1136-1143):
    # dk = p @ d_lse_q

    # NEW: Compute dQ contribution if fused mode
    if dq_ref is not None:
        # ds = d_lse * p for LSE (simplified)
        # p is P^T (bkv, bq), we need ds with shape (bq, bkv) for dQ = ds @ K
        # Actually: dQ = (d_lse * P) @ K = (d_lse[:, None] * P) @ K
        # P = p.T, so: dQ = (d_lse[:, None] * p.T) @ K

        p_T = p.T  # (bq, bkv)
        ds = d_lse[:, None] * p_T  # (bq, bkv)

        dq_contrib = lax.dot_general(
            ds.astype(k.dtype), k, NN_DIM_NUMBERS,
            preferred_element_type=jnp.float32,
        )  # (bq, head_dim)

        if dq_scratch_ref is not None:
            # bkv != bkv_compute case
            dq_scratch_ref[...] += dq_contrib
        else:
            dq_ref[...] = dq_contrib.astype(dq_ref.dtype)

    # ... rest of kernel (write dk at end) ...
```

### Step 2: Modify `_lse_backward_dk` wrapper (lines 1164-1461)

Add dQ output specs matching splash pattern:

```python
def _lse_backward_dk(
    ...
    use_fused_bwd_kernel: bool,  # NEW parameter
):
    # ... existing setup ...

    # NEW: dQ output specs (mirror splash lines 1998-2009)
    if use_fused_bwd_kernel:
        def dq_index_map(kv_idx, h, q_idx, *_):
            return (kv_idx, h, q_idx, 0)
        dq_spec = pl.BlockSpec((None, None, bq, head_dim_qk), dq_index_map)
        dq_shape = jax.ShapeDtypeStruct((kv_seq_len // bkv, *q.shape), q.dtype)

        if bkv == bkv_compute:
            dq_scratch_spec = dq_scratch_shape = None
        else:
            dq_scratch_spec = pl.BlockSpec((bq, head_dim_qk), lambda *_: (0, 0))
            dq_scratch_shape = jax.ShapeDtypeStruct((bq, head_dim_qk), jnp.float32)
    else:
        dq_spec = dq_shape = dq_scratch_spec = dq_scratch_shape = None

    # Update out_shapes and out_specs to include dQ
    out_shapes = [
        dq_scratch_shape,  # NEW
        jax.ShapeDtypeStruct((bkv, head_dim_qk), jnp.float32),  # dk_scratch
        dq_shape,  # NEW
        jax.ShapeDtypeStruct(k.shape, k.dtype),  # dk
    ]
    out_specs = [
        dq_scratch_spec,  # NEW
        pl.BlockSpec((bkv, head_dim_qk), lambda *_: (0, 0)),
        dq_spec,  # NEW
        dk_spec,
    ]

    # ... kernel call ...

    # NEW: Return dQ (reduced if fused)
    if use_fused_bwd_kernel:
        dq = dq_unreduced.sum(axis=0)
    else:
        dq = None

    return dq, dk
```

### Step 3: Modify `_lse_bwd` (lines 1564-1638)

Conditionally use fused vs separate backward:

```python
def _lse_bwd(..., residuals, g_lse):
    q, k, segment_ids, sinks, logsumexp, dq_mask_info, dk_mask_info = residuals

    use_fused_bwd_kernel = block_sizes.use_fused_bwd_kernel

    # Always call the dk kernel (now optionally computes dq too)
    dq_fused, dk = _lse_backward_dk(
        q=q,
        k=k,
        d_lse_q=g_lse[:, :, None] * q,
        ...,
        use_fused_bwd_kernel=use_fused_bwd_kernel,  # NEW
    )

    if use_fused_bwd_kernel:
        dq = dq_fused
    else:
        # Separate dQ kernel (existing code)
        dq = _lse_backward_dq(
            q=q, k=k, d_lse=g_lse, ...
        )

    return (None, None, None, dq, dk, None, None)
```

### Step 4: Update `BlockSizes` validation (if needed)

Our `BlockSizes` already has `use_fused_bwd_kernel` at line 132. Verify the validation logic at lines 143-147 works for our case:

```python
if self.use_fused_bwd_kernel:
    if self.block_q_dq is not None or self.block_kv_dq is not None:
        raise ValueError(
            "Block sizes for dq kernel are not needed with a fused kernel."
        )
```

## Memory Considerations for LSE vs Splash

Our LSE kernel might be more amenable to fused backward because:

1. **No V**: We don't have value tensors, so total memory footprint is smaller
2. **Simpler gradient**: `ds = d_lse * p` vs `ds = (dp - di) * p`, fewer intermediates
3. **Use case**: LSE is often used for cross-entropy loss where sequences may be shorter

### Memory budget comparison

For seq_len=2048, bkv=128, heads=8, head_dim=128:
- dQ intermediate: `16 × 8 × 2048 × 128 × 4 bytes = 128 MB` ← Reasonable!

For seq_len=8192, bkv=128, heads=32, head_dim=128:
- dQ intermediate: `64 × 32 × 8192 × 128 × 4 bytes = 8.5 GB` ← Too large

**Recommendation**: Make `use_fused_bwd_kernel` configurable, default to `False` for compatibility, but document that `True` can be faster for shorter sequences.

## Testing Strategy

1. **Unit test**: Compare fused vs separate outputs
   ```python
   # In tests/kernels/test_lse_kernel.py
   @parameterized.product(use_fused_bwd_kernel=[False, True])
   def test_lse_backward(self, use_fused_bwd_kernel):
       block_sizes = BlockSizes(..., use_fused_bwd_kernel=use_fused_bwd_kernel)
       # ... run backward, compare to reference ...
   ```

2. **Gradient check**: Verify correctness with `jax.test_util.check_grads`

3. **Performance benchmark**: Measure wall time for both modes across sequence lengths

## Files to Modify

1. `src/bmgpt/kernels/lse_kernel.py`:
   - `_lse_backward_dk_kernel`: Add dQ computation (lines ~1102-1143)
   - `_lse_backward_dk`: Add dQ output specs and reduction (lines ~1164-1461)
   - `_lse_bwd`: Conditional fused/separate logic (lines ~1564-1638)

2. `tests/kernels/test_lse_kernel.py` (or equivalent):
   - Add `use_fused_bwd_kernel` parameter to backward tests

## Appendix: Line Number Reference

### Splash Attention (upstream)
- `_flash_attention_dkv_kernel`: lines 1673-1858
- P computation: line 1791
- dQ computation (fused): lines 1812-1823
- dQ output spec: lines 1998-2009
- dQ reduction: line 2237
- `use_fused_bwd_kernel` default: line 516

### LSE Kernel (our code)
- `_lse_backward_dq_kernel`: lines 695-790
- `_lse_backward_dk_kernel`: lines 1013-1161
- `_lse_backward_dk` wrapper: lines 1164-1461
- `_lse_bwd`: lines 1564-1638
- `BlockSizes.use_fused_bwd_kernel`: line 132
