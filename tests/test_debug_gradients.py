"""Debug script for comparing fused vs non-fused xent gradients.

Loads saved step 0 data from training run and computes gradients locally
for comparison.

Usage:
  1. Run training with use_fused_xent_loss=False (saves to /tmp/debug_loss_inputs/nonfused/)
  2. Run training with use_fused_xent_loss=True (saves to /tmp/debug_loss_inputs/fused/)
  3. Run this script to load data and compare gradients:
     uv run python tests/test_debug_gradients.py
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import Array

from bmgpt.losses import fused_softmax_cross_entropy, softmax_cross_entropy
from bmgpt.splash_helpers import make_lse_kernel_sharded
from bmgpt.train import (
  _debug_log,
  debug_get_save_dir,
  debug_load_and_reconstruct,
  debug_load_sharding_info,
)


class LMHead(NamedTuple):
  """Mock LMHead for gradient computation."""

  w: Array
  bias: Array


class MockConfig:
  """Mock config with just the fields needed for loss computation."""

  def __init__(self, config_dict):
    self.config_dict = config_dict

    # Sharding config
    class ShardingConfig:
      pass

    self.sharding = ShardingConfig()
    self.sharding.data = config_dict["sharding"]["data"]
    self.sharding.mesh_shape = config_dict["sharding"]["mesh_shape"]
    self.sharding.mesh_axis_names = config_dict["sharding"]["mesh_axis_names"]

    # Model config
    class ModelConfig:
      pass

    self.model = ModelConfig()
    self.model.num_vocab = config_dict["model"]["num_vocab"]
    self.model.d_model = config_dict["model"]["d_model"]

    # Train dataset config
    class DatasetConfig:
      pass

    self.train_dataset = DatasetConfig()
    self.train_dataset.max_valid_token_id = config_dict["train_dataset"][
      "max_valid_token_id"
    ]
    self.train_dataset.seq_len = config_dict["train_dataset"]["seq_len"]
    self.train_dataset.global_batch_size = config_dict["train_dataset"][
      "global_batch_size"
    ]
    self.train_dataset.num_microbatches = config_dict["train_dataset"][
      "num_microbatches"
    ]

    # Fused xent config
    self.fused_xent_block_size_T = config_dict.get("fused_xent_block_size_T", 512)
    self.fused_xent_block_size_V = config_dict.get("fused_xent_block_size_V", 512)
    self.fused_xent_block_size_V_compute = config_dict.get(
      "fused_xent_block_size_V_compute", 512
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
  first_variant = available_variants[0]
  _, _, config_dict = debug_load_sharding_info(first_variant)

  # Create mesh directly from config_dict (avoid OmegaConf.to_object which fails on None values)
  sharding = config_dict["sharding"]
  mesh = jax.make_mesh(
    sharding["mesh_shape"],
    sharding["mesh_axis_names"],
    len(sharding["mesh_shape"]) * (jax.sharding.AxisType.Explicit,),
  )
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

  # Verify inputs match between fused and nonfused (they should be identical)
  if len(available_variants) == 2:
    _debug_log("=" * 60)
    _debug_log("VERIFYING INPUTS MATCH BETWEEN VARIANTS")
    _debug_log("=" * 60)
    for name in ["batch_inputs", "batch_targets", "unemb_w", "all_outputs"]:
      fused_arr = variant_data["fused"][name]
      nonfused_arr = variant_data["nonfused"][name]
      if jnp.array_equal(fused_arr, nonfused_arr):
        _debug_log(f"  {name}: MATCH")
      else:
        diff_count = jnp.sum(fused_arr != nonfused_arr)
        _debug_log(f"  {name}: DIFFER ({diff_count} elements)")

  # Create mock config for loss functions
  config = MockConfig(config_dict)

  # Create LSE kernel for fused loss
  _debug_log("=" * 60)
  _debug_log("CREATING LSE KERNEL")
  _debug_log("=" * 60)
  num_toks = (
    config.train_dataset.seq_len
    * config.train_dataset.global_batch_size
    // config.train_dataset.num_microbatches
  )
  if config.sharding.data and config.sharding.data[0]:
    data_axis_name = config.sharding.data[0]
    num_data_shards = config.sharding.mesh_shape[
      config.sharding.mesh_axis_names.index(data_axis_name)
    ]
  else:
    num_data_shards = 1

  lse_kernel = make_lse_kernel_sharded(
    num_toks,
    config.model.num_vocab,
    mesh,
    data_sharding=config.sharding.data,
    q_seq_shards=num_data_shards,
    block_size_mem_q=config.fused_xent_block_size_T,
    block_size_mem_kv=config.fused_xent_block_size_V,
    block_size_compute_kv=config.fused_xent_block_size_V_compute,
    max_valid_id=config.train_dataset.max_valid_token_id,
  )
  _debug_log(
    f"LSE kernel created: num_toks={num_toks}, num_vocab={config.model.num_vocab}"
  )

  # Use nonfused data (inputs are same between variants)
  data = variant_data["nonfused"]
  all_outputs = data["all_outputs"]  # (num_mb, batch_per_mb, seq, hidden)
  batch_targets = data["batch_targets"]  # (num_mb, batch_per_mb, seq)
  unemb_w = data["unemb_w"]  # (num_vocab, hidden)

  # Create LMHead with zero bias
  unemb = LMHead(w=unemb_w, bias=jnp.zeros(config.model.num_vocab, dtype=unemb_w.dtype))

  num_microbatches = all_outputs.shape[0]
  _debug_log(f"Processing {num_microbatches} microbatches")

  # Log input statistics
  _debug_log("=" * 60)
  _debug_log("INPUT STATISTICS")
  _debug_log("=" * 60)
  _debug_log(f"  unemb.w: shape={unemb_w.shape}, dtype={unemb_w.dtype}")
  _debug_log(
    f"    mean={float(jnp.mean(unemb_w)):.6f}, std={float(jnp.std(unemb_w)):.6f}, "
    f"min={float(jnp.min(unemb_w)):.6f}, max={float(jnp.max(unemb_w)):.6f}"
  )
  _debug_log(f"  all_outputs: shape={all_outputs.shape}, dtype={all_outputs.dtype}")
  _debug_log(
    f"    mean={float(jnp.mean(all_outputs)):.6f}, std={float(jnp.std(all_outputs)):.6f}, "
    f"min={float(jnp.min(all_outputs)):.6f}, max={float(jnp.max(all_outputs)):.6f}"
  )
  # Check for extreme values that might cause precision issues
  outputs_absmax = float(jnp.max(jnp.abs(all_outputs)))
  unemb_absmax = float(jnp.max(jnp.abs(unemb_w)))
  _debug_log(f"  |outputs|_max={outputs_absmax:.4f}, |unemb.w|_max={unemb_absmax:.4f}")
  _debug_log(f"  outputs * unemb scale ~ {outputs_absmax * unemb_absmax:.4f}")

  # Compare gradients per-microbatch
  _debug_log("=" * 60)
  _debug_log("GRADIENT COMPARISON (per microbatch)")
  _debug_log("=" * 60)

  for mb_idx in range(num_microbatches):
    _debug_log(f"--- Microbatch {mb_idx} ---")

    # Get microbatch data
    mb_outputs = all_outputs[mb_idx]  # (batch_per_mb, seq, hidden)
    mb_targets = batch_targets[mb_idx]  # (batch_per_mb, seq)

    # Define loss functions for gradient computation
    def nonfused_loss(outputs, unemb_head):
      return softmax_cross_entropy(config, unemb_head, outputs, mb_targets)

    def fused_loss(outputs, unemb_head):
      return fused_softmax_cross_entropy(
        config, unemb_head, outputs, mb_targets, lse_kernel
      )

    # Compute gradients
    loss_nf, (grad_outputs_nf, grad_unemb_nf) = jax.value_and_grad(
      nonfused_loss, argnums=(0, 1)
    )(mb_outputs, unemb)
    loss_f, (grad_outputs_f, grad_unemb_f) = jax.value_and_grad(
      fused_loss, argnums=(0, 1)
    )(mb_outputs, unemb)

    # Compare losses
    loss_diff = float(jnp.abs(loss_nf - loss_f))
    _debug_log(
      f"  Loss: nonfused={float(loss_nf):.6f}, fused={float(loss_f):.6f}, diff={loss_diff:.6e}"
    )

    # Compare output gradients
    grad_out_diff = jnp.abs(grad_outputs_nf - grad_outputs_f)
    grad_out_max_diff = float(jnp.max(grad_out_diff))
    grad_out_mean_diff = float(jnp.mean(grad_out_diff))
    grad_out_nf_norm = float(jnp.linalg.norm(grad_outputs_nf.ravel()))
    grad_out_f_norm = float(jnp.linalg.norm(grad_outputs_f.ravel()))
    _debug_log(
      f"  Grad outputs: nf_norm={grad_out_nf_norm:.4f}, f_norm={grad_out_f_norm:.4f}, "
      f"max_diff={grad_out_max_diff:.6e}, mean_diff={grad_out_mean_diff:.6e}"
    )

    # Compare unemb.w gradients
    grad_w_diff = jnp.abs(grad_unemb_nf.w - grad_unemb_f.w)
    grad_w_max_diff = float(jnp.max(grad_w_diff))
    grad_w_mean_diff = float(jnp.mean(grad_w_diff))
    grad_w_nf_norm = float(jnp.linalg.norm(grad_unemb_nf.w.ravel()))
    grad_w_f_norm = float(jnp.linalg.norm(grad_unemb_f.w.ravel()))
    _debug_log(
      f"  Grad unemb.w: nf_norm={grad_w_nf_norm:.4f}, f_norm={grad_w_f_norm:.4f}, "
      f"max_diff={grad_w_max_diff:.6e}, mean_diff={grad_w_mean_diff:.6e}"
    )

  _debug_log("=" * 60)
  _debug_log("DONE")
  _debug_log("=" * 60)


if __name__ == "__main__":
  main()
