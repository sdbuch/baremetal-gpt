"""Debug script for comparing fused vs non-fused xent gradients.

Loads saved step 0 data from training run and computes gradients locally
for comparison.

Usage:
  1. Run training with use_fused_xent_loss=False (saves to /tmp/debug_loss_inputs/nonfused/)
  2. Run training with use_fused_xent_loss=True (saves to /tmp/debug_loss_inputs/fused/)
  3. Run this script to load data and compare gradients:
     uv run python tests/test_debug_gradients.py
"""

from pathlib import Path

import jax

from bmgpt.config import mesh_from_config
from bmgpt.train import (
  _debug_log,
  debug_get_save_dir,
  debug_load_and_reconstruct,
  debug_load_sharding_info,
)


def load_variant_data(variant: str, mesh):
  """Load all debug data for a variant using the tested train.py functions.

  Args:
    variant: "fused" or "nonfused"
    mesh: JAX mesh for sharding

  Returns:
    dict with reconstructed JAX arrays
  """
  dtypes, specs, config_dict = debug_load_sharding_info(variant)

  arrays = {}
  for name in ["batch_inputs", "batch_targets", "unemb_w", "all_outputs"]:
    arrays[name] = debug_load_and_reconstruct(name, variant, mesh, dtypes, specs)
    _debug_log(f"loaded {name}: {arrays[name].shape} {arrays[name].dtype}")

  return arrays, config_dict


def main():
  """Main debug script entry point."""
  # Initialize distributed backend
  try:
    jax.distributed.initialize()
  except RuntimeError:
    pass  # Already initialized

  proc_idx = jax.process_index()
  _debug_log("=" * 60)
  _debug_log("DEBUG GRADIENT COMPARISON SCRIPT")
  _debug_log("=" * 60)

  # Check which variants are available
  available_variants = []
  for variant in ["fused", "nonfused"]:
    save_dir = debug_get_save_dir(variant)
    if save_dir.exists():
      available_variants.append(variant)
  _debug_log(f"Available variants: {available_variants}")

  if not available_variants:
    _debug_log("ERROR: No debug data found. Run training first.")
    return

  # Load sharding info from first variant to get config for mesh
  from omegaconf import OmegaConf

  first_variant = available_variants[0]
  _, _, config_dict = debug_load_sharding_info(first_variant)
  config = OmegaConf.create(config_dict)
  config = OmegaConf.to_object(config)

  # Create mesh
  mesh = mesh_from_config(config)
  _debug_log(f"Mesh: {mesh}")

  # Load data for each variant
  variant_data = {}
  with jax.set_mesh(mesh):
    for variant in available_variants:
      _debug_log(f"--- Loading {variant} data ---")
      arrays, _ = load_variant_data(variant, mesh)
      variant_data[variant] = arrays

  # Print summary
  _debug_log("=" * 60)
  _debug_log("SUMMARY")
  _debug_log("=" * 60)
  for variant in available_variants:
    _debug_log(f"{variant}:")
    for name, arr in variant_data[variant].items():
      _debug_log(f"  {name}: shape={arr.shape}, dtype={arr.dtype}")

  # TODO: Add gradient computation here
  _debug_log("[Next step: implement gradient comparison]")


if __name__ == "__main__":
  main()
