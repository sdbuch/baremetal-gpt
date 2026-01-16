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
from omegaconf import OmegaConf

from bmgpt.config import Config
from bmgpt.losses import fused_softmax_cross_entropy, softmax_cross_entropy
from bmgpt.model import LMHead
from bmgpt.splash_helpers import make_lse_kernel_sharded
from bmgpt.train import (
  _debug_log,
  debug_get_save_dir,
  debug_load_and_reconstruct,
  debug_load_sharding_info,
)


def config_from_dict(config_dict: dict) -> Config:
  """Create a Config object from a saved config dict.

  Creates a default Config and updates fields from the saved dict.
  This avoids issues with MISSING values and enum conversion.
  """
  from bmgpt.config import (
    DatasetConfig,
    DatasetName,
    DType,
    ModelConfig,
    ShardingConfig,
    TokenizerType,
    TransformerType,
  )

  # Create base config with defaults
  config = Config()

  # Update sharding
  s = config_dict["sharding"]
  config.sharding = ShardingConfig(
    mesh_shape=s["mesh_shape"],
    mesh_axis_names=s["mesh_axis_names"],
    data=s["data"],
  )

  # Update model config (use .get() with defaults for fields that might be missing)
  m = config_dict["model"]
  config.model = ModelConfig(
    num_heads=m.get("num_heads", 12),
    num_layers=m.get("num_layers", 12),
    d_model=m.get("d_model", 768),
    d_head=m.get("d_head", 64),
    d_ff=m.get("d_ff", m.get("d_model", 768) * 4),  # Default to 4x d_model
    num_vocab=m["num_vocab"],  # Required
    max_seq_len=m.get("max_seq_len", 2048),
    param_dtype=DType[m.get("param_dtype", "bfloat16").upper().replace(".", "_")],
    transformer_type=TransformerType(m.get("transformer_type", "discrete")),
  )

  # Update train_dataset config
  td = config_dict["train_dataset"]
  config.train_dataset = DatasetConfig(
    name=DatasetName(td["name"]),
    path=td.get("path", ""),
    seq_len=td["seq_len"],
    global_batch_size=td["global_batch_size"],
    num_microbatches=td["num_microbatches"],
    max_valid_token_id=td["max_valid_token_id"],
    tokenizer=TokenizerType(td.get("tokenizer", 0)),
  )

  return config


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

  # Create config from saved dict
  try:
    config = config_from_dict(config_dict)
  except Exception as e:
    _debug_log(f"Failed to create Config from dict: {e}")
    _debug_log("Falling back to direct dict access")
    config = None

  # Create LSE kernel for fused loss (using same logic as forward_kernels_from_config)
  _debug_log("=" * 60)
  _debug_log("CREATING LSE KERNEL")
  _debug_log("=" * 60)

  # Extract needed values from config_dict directly for kernel creation
  sharding = config_dict["sharding"]
  train_ds = config_dict["train_dataset"]
  model = config_dict["model"]

  num_toks = (
    train_ds["seq_len"] * train_ds["global_batch_size"] // train_ds["num_microbatches"]
  )
  if sharding["data"] and sharding["data"][0]:
    data_axis_name = sharding["data"][0]
    num_data_shards = sharding["mesh_shape"][
      sharding["mesh_axis_names"].index(data_axis_name)
    ]
  else:
    num_data_shards = 1

  lse_kernel = make_lse_kernel_sharded(
    num_toks,
    model["num_vocab"],
    mesh,
    data_sharding=sharding["data"],
    q_seq_shards=num_data_shards,
    block_size_mem_q=config_dict.get("fused_xent_block_size_T", 512),
    block_size_mem_kv=config_dict.get("fused_xent_block_size_V", 512),
    block_size_compute_kv=config_dict.get("fused_xent_block_size_V_compute", 512),
    max_valid_id=train_ds["max_valid_token_id"],
  )
  _debug_log(f"LSE kernel created: num_toks={num_toks}, num_vocab={model['num_vocab']}")

  # Use nonfused data (inputs are same between variants)
  data = variant_data["nonfused"]
  all_outputs = data["all_outputs"]  # (num_mb, batch_per_mb, seq, hidden)
  batch_targets = data["batch_targets"]  # (num_mb, batch_per_mb, seq)
  unemb_w = data["unemb_w"]  # (num_vocab, hidden)

  # Create LMHead with zero bias
  num_vocab = model["num_vocab"]
  unemb = LMHead(w=unemb_w, bias=jnp.zeros(num_vocab, dtype=unemb_w.dtype))

  # Ensure we have a valid config for loss functions
  if config is None:
    _debug_log("ERROR: Config creation failed, cannot compute gradients")
    return

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
