"""Debug script for comparing fused vs non-fused xent gradients.

Loads saved step 0 data from training run and computes gradients locally
for comparison.

Usage:
  1. Run training with use_fused_xent_loss=False (saves to /tmp/debug_loss_inputs/nonfused/)
  2. Run training with use_fused_xent_loss=True (saves to /tmp/debug_loss_inputs/fused/)
  3. Run this script to load data and compare gradients:
     uv run python tests/test_debug_gradients.py
"""

import jax
import jax.numpy as jnp

from bmgpt.config import Config, mesh_from_config
from bmgpt.losses import fused_softmax_cross_entropy
from bmgpt.model import LMHead
from bmgpt.splash_helpers import forward_kernels_from_config
from bmgpt.train import (
  _debug_log,
  debug_get_save_dir,
  debug_load_and_reconstruct,
  debug_load_sharding_info,
)


def _log(msg):
  """Print debug message only on host 0."""
  if jax.process_index() == 0:
    _debug_log(msg, proc_idx=0)


def load_variant_data(variant: str, mesh):
  """Load all debug data for a variant using the tested train.py functions.

  Args:
    variant: "fused" or "nonfused"
    mesh: JAX mesh for sharding

  Returns:
    Tuple of (arrays dict, Config object)
  """
  dtypes, specs, config = debug_load_sharding_info(variant)

  arrays = {}
  for name in ["batch_inputs", "batch_targets", "unemb_w", "all_outputs"]:
    arrays[name] = debug_load_and_reconstruct(name, variant, mesh, dtypes, specs)
    _log(f"loaded {name}: {arrays[name].shape} {arrays[name].dtype}")

  return arrays, config


def main():
  """Main debug script entry point."""
  # Initialize distributed backend
  try:
    jax.distributed.initialize()
  except RuntimeError:
    pass  # Already initialized

  _log("=" * 60)
  _log("DEBUG GRADIENT COMPARISON SCRIPT")
  _log("=" * 60)

  # Check which variants are available
  available_variants = []
  for variant in ["fused", "nonfused"]:
    save_dir = debug_get_save_dir(variant)
    if save_dir.exists():
      available_variants.append(variant)
  _log(f"Available variants: {available_variants}")

  if not available_variants:
    _log("ERROR: No debug data found. Run training first.")
    return

  # Load sharding info from first variant to get config for mesh
  first_variant = available_variants[0]
  _, _, config = debug_load_sharding_info(first_variant)

  # Create mesh using the standard helper
  mesh = mesh_from_config(config)
  _log(f"Mesh: {mesh}")

  # Everything below runs inside the mesh context for proper sharding
  with jax.set_mesh(mesh):
    # Load data for each variant
    variant_data = {}
    for variant in available_variants:
      _log(f"--- Loading {variant} data ---")
      arrays, _ = load_variant_data(variant, mesh)
      variant_data[variant] = arrays

    # Print summary
    _log("=" * 60)
    _log("SUMMARY")
    _log("=" * 60)
    for variant in available_variants:
      _log(f"{variant}:")
      for name, arr in variant_data[variant].items():
        _log(f"  {name}: shape={arr.shape}, dtype={arr.dtype}")

    # Verify inputs match between fused and nonfused (they should be identical)
    if len(available_variants) == 2:
      _log("=" * 60)
      _log("VERIFYING INPUTS MATCH BETWEEN VARIANTS")
      _log("=" * 60)
      for name in ["batch_inputs", "batch_targets", "unemb_w", "all_outputs"]:
        fused_arr = variant_data["fused"][name]
        nonfused_arr = variant_data["nonfused"][name]
        if jnp.array_equal(fused_arr, nonfused_arr):
          _log(f"  {name}: MATCH")
        else:
          diff_count = jnp.sum(fused_arr != nonfused_arr)
          _log(f"  {name}: DIFFER ({diff_count} elements)")

    # Create LSE kernel using the standard helper
    _log("=" * 60)
    _log("CREATING LSE KERNEL")
    _log("=" * 60)

    _, lse_kernel, _, _ = forward_kernels_from_config(config, mesh)
    _log("LSE kernel created via forward_kernels_from_config")

    # Use nonfused data (inputs are same between variants)
    data = variant_data["nonfused"]
    all_outputs = data["all_outputs"]  # (num_mb, batch_per_mb, seq, hidden)
    batch_targets = data["batch_targets"]  # (num_mb, batch_per_mb, seq)
    unemb_w = data["unemb_w"]  # (num_vocab, hidden)

    # Create LMHead with zero bias
    unemb = LMHead(
      w=unemb_w, bias=jnp.zeros(config.model.num_vocab, dtype=unemb_w.dtype)
    )

    num_microbatches = all_outputs.shape[0]
    _log(f"Processing {num_microbatches} microbatches")

    # Log input statistics
    _log("=" * 60)
    _log("INPUT STATISTICS")
    _log("=" * 60)
    _log(f"  unemb.w: shape={unemb_w.shape}, dtype={unemb_w.dtype}")
    _log(
      f"    mean={float(jnp.mean(unemb_w)):.6f}, std={float(jnp.std(unemb_w)):.6f}, "
      f"min={float(jnp.min(unemb_w)):.6f}, max={float(jnp.max(unemb_w)):.6f}"
    )
    _log(f"  all_outputs: shape={all_outputs.shape}, dtype={all_outputs.dtype}")
    _log(
      f"    mean={float(jnp.mean(all_outputs)):.6f}, std={float(jnp.std(all_outputs)):.6f}, "
      f"min={float(jnp.min(all_outputs)):.6f}, max={float(jnp.max(all_outputs)):.6f}"
    )
    # Check for extreme values that might cause precision issues
    outputs_absmax = float(jnp.max(jnp.abs(all_outputs)))
    unemb_absmax = float(jnp.max(jnp.abs(unemb_w)))
    _log(f"  |outputs|_max={outputs_absmax:.4f}, |unemb.w|_max={unemb_absmax:.4f}")
    _log(f"  outputs * unemb scale ~ {outputs_absmax * unemb_absmax:.4f}")

    # Compare gradients per-microbatch
    _log("=" * 60)
    _log("GRADIENT COMPARISON (per microbatch)")
    _log("=" * 60)

    for mb_idx in range(num_microbatches):
      _log(f"--- Microbatch {mb_idx} ---")

      # Get microbatch data
      mb_outputs = all_outputs[mb_idx]  # (batch_per_mb, seq, hidden)
      mb_targets = batch_targets[mb_idx]  # (batch_per_mb, seq)

      # Define fused loss function for gradient computation
      def fused_loss(outputs, unemb_head):
        return fused_softmax_cross_entropy(
          config, unemb_head, outputs, mb_targets, lse_kernel
        )

      # Compute fused gradients only (nonfused OOMs due to full logits materialization)
      loss_f, (grad_outputs_f, grad_unemb_f) = jax.value_and_grad(
        fused_loss, argnums=(0, 1)
      )(mb_outputs, unemb)

      # Log fused loss and gradient statistics
      # Note: use sum of squares instead of ravel().norm() to work with sharded arrays
      _log(f"  Fused loss: {float(loss_f):.6f}")

      grad_out_f_sumsq = float(jnp.sum(grad_outputs_f**2))
      grad_out_f_norm = float(jnp.sqrt(grad_out_f_sumsq))
      grad_out_f_mean = float(jnp.mean(grad_outputs_f))
      grad_out_f_std = float(jnp.std(grad_outputs_f))
      grad_out_f_min = float(jnp.min(grad_outputs_f))
      grad_out_f_max = float(jnp.max(grad_outputs_f))
      _log(
        f"  Grad outputs: norm={grad_out_f_norm:.4f}, mean={grad_out_f_mean:.6e}, "
        f"std={grad_out_f_std:.6e}, min={grad_out_f_min:.6e}, max={grad_out_f_max:.6e}"
      )

      grad_w_f_sumsq = float(jnp.sum(grad_unemb_f.w**2))
      grad_w_f_norm = float(jnp.sqrt(grad_w_f_sumsq))
      grad_w_f_mean = float(jnp.mean(grad_unemb_f.w))
      grad_w_f_std = float(jnp.std(grad_unemb_f.w))
      grad_w_f_min = float(jnp.min(grad_unemb_f.w))
      grad_w_f_max = float(jnp.max(grad_unemb_f.w))
      _log(
        f"  Grad unemb.w: norm={grad_w_f_norm:.4f}, mean={grad_w_f_mean:.6e}, "
        f"std={grad_w_f_std:.6e}, min={grad_w_f_min:.6e}, max={grad_w_f_max:.6e}"
      )

  _log("=" * 60)
  _log("DONE")
  _log("=" * 60)


if __name__ == "__main__":
  main()
