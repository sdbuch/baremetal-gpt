"""Debug script for comparing fused vs non-fused xent gradients.

Loads saved step 0 data from training run and computes gradients locally
for comparison.

Usage:
  1. Run training with use_fused_xent_loss=False (saves to /tmp/debug_loss_inputs/nonfused/)
  2. Run training with use_fused_xent_loss=True (saves to /tmp/debug_loss_inputs/fused/)
  3. Run this script to load data and compare gradients:
     uv run python tests/test_debug_gradients.py
"""

from collections import defaultdict
from itertools import combinations

import jax
import jax.numpy as jnp

from bmgpt.config import mesh_from_config
from bmgpt.losses import fused_softmax_cross_entropy, softmax_cross_entropy
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
    _log(f"loaded {name}: {jax.typeof(arrays[name])}")

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
        _log(f"  {name}: {jax.typeof(arr)}")

    # Create LSE kernel using the standard helper (only needed for fused)
    _, lse_kernel, _, _ = forward_kernels_from_config(config, mesh)

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
    _log(f"  unemb.w: {jax.typeof(unemb_w)}")
    _log(f"  all_outputs: {jax.typeof(all_outputs)}")

    # Compute gradients for all microbatches using scan (matches train.py structure)
    _log("=" * 60)
    _log("COMPUTING GRADIENTS VIA SCAN")
    _log("=" * 60)

    def compute_gradients(carry, microbatch):
      mb_outputs, mb_targets = microbatch

      def nonfused_loss(outputs, unemb_head):
        return softmax_cross_entropy(config, unemb_head, outputs, mb_targets)

      def fused_loss(outputs, unemb_head):
        return fused_softmax_cross_entropy(
          config, unemb_head, outputs, mb_targets, lse_kernel
        )

      loss_nf, (grad_outputs_nf, grad_unemb_nf) = jax.value_and_grad(
        nonfused_loss, argnums=(0, 1)
      )(mb_outputs, unemb)

      loss_f, (grad_outputs_f, grad_unemb_f) = jax.value_and_grad(
        fused_loss, argnums=(0, 1)
      )(mb_outputs, unemb)

      return carry, {
        "loss_nf": loss_nf,
        "grad_outputs_nf": grad_outputs_nf,
        "grad_unemb_nf_w": grad_unemb_nf.w,
        "loss_f": loss_f,
        "grad_outputs_f": grad_outputs_f,
        "grad_unemb_f_w": grad_unemb_f.w,
      }

    _log("  Running scan over microbatches...")
    _, results = jax.lax.scan(compute_gradients, None, (all_outputs, batch_targets))
    _log("  Scan complete.")

    # Compare gradients per-microbatch
    _log("=" * 60)
    _log("GRADIENT COMPARISON (per microbatch)")
    _log("=" * 60)

    for mb_idx in range(num_microbatches):
      _log(f"--- Microbatch {mb_idx} ---")

      # Extract results for this microbatch
      loss_nf = results["loss_nf"][mb_idx]
      grad_outputs_nf = results["grad_outputs_nf"][mb_idx]
      grad_unemb_nf_w = results["grad_unemb_nf_w"][mb_idx]
      loss_f = results["loss_f"][mb_idx]
      grad_outputs_f = results["grad_outputs_f"][mb_idx]
      grad_unemb_f_w = results["grad_unemb_f_w"][mb_idx]

      nonfused_stats = {
        "loss": float(loss_nf),
        "grad_out_norm": float(jnp.sqrt(jnp.sum(grad_outputs_nf**2))),
        "grad_out_mean": float(jnp.mean(grad_outputs_nf)),
        "grad_out_std": float(jnp.std(grad_outputs_nf)),
        "grad_out_min": float(jnp.min(grad_outputs_nf)),
        "grad_out_max": float(jnp.max(grad_outputs_nf)),
        "grad_w_norm": float(jnp.sqrt(jnp.sum(grad_unemb_nf_w**2))),
        "grad_w_mean": float(jnp.mean(grad_unemb_nf_w)),
        "grad_w_std": float(jnp.std(grad_unemb_nf_w)),
        "grad_w_min": float(jnp.min(grad_unemb_nf_w)),
        "grad_w_max": float(jnp.max(grad_unemb_nf_w)),
      }

      fused_stats = {
        "loss": float(loss_f),
        "grad_out_norm": float(jnp.sqrt(jnp.sum(grad_outputs_f**2))),
        "grad_out_mean": float(jnp.mean(grad_outputs_f)),
        "grad_out_std": float(jnp.std(grad_outputs_f)),
        "grad_out_min": float(jnp.min(grad_outputs_f)),
        "grad_out_max": float(jnp.max(grad_outputs_f)),
        "grad_w_norm": float(jnp.sqrt(jnp.sum(grad_unemb_f_w**2))),
        "grad_w_mean": float(jnp.mean(grad_unemb_f_w)),
        "grad_w_std": float(jnp.std(grad_unemb_f_w)),
        "grad_w_min": float(jnp.min(grad_unemb_f_w)),
        "grad_w_max": float(jnp.max(grad_unemb_f_w)),
      }

      fs, nfs = fused_stats, nonfused_stats
      _log(f"  Loss:       fused={fs['loss']:.6f}, nonfused={nfs['loss']:.6f}")
      _log(
        f"  Grad out (norm): fused={fs['grad_out_norm']:.6f}, nf={nfs['grad_out_norm']:.6f}"
      )
      _log(
        f"  Grad out (mean): fused={fs['grad_out_mean']:.6e}, nf={nfs['grad_out_mean']:.6e}"
      )
      _log(
        f"  Grad out (std):  fused={fs['grad_out_std']:.6e}, nf={nfs['grad_out_std']:.6e}"
      )
      _log(
        f"  Grad out (min):  fused={fs['grad_out_min']:.6e}, nf={nfs['grad_out_min']:.6e}"
      )
      _log(
        f"  Grad out (max):  fused={fs['grad_out_max']:.6e}, nf={nfs['grad_out_max']:.6e}"
      )
      _log(
        f"  Grad w (norm):   fused={fs['grad_w_norm']:.6f}, nf={nfs['grad_w_norm']:.6f}"
      )
      _log(
        f"  Grad w (mean):   fused={fs['grad_w_mean']:.6e}, nf={nfs['grad_w_mean']:.6e}"
      )
      _log(
        f"  Grad w (std):    fused={fs['grad_w_std']:.6e}, nf={nfs['grad_w_std']:.6e}"
      )
      _log(
        f"  Grad w (min):    fused={fs['grad_w_min']:.6e}, nf={nfs['grad_w_min']:.6e}"
      )
      _log(
        f"  Grad w (max):    fused={fs['grad_w_max']:.6e}, nf={nfs['grad_w_max']:.6e}"
      )

  _log("=" * 60)
  _log("DONE")
  _log("=" * 60)


def mwe():
  T = 256
  V = 1024
  D = 3 * 64
  key = jax.random.key(42)
  kx, kw, kt = jax.random.split(key, 3)
  mesh = jax.make_mesh((32,), ("x",), (jax.sharding.AxisType.Explicit,))

  with jax.set_mesh(mesh):
    x = jax.random.normal(kx, (T, D), dtype=jnp.float32, out_sharding=jax.P("x"))
    t = jax.random.randint(kt, (T,), 0, V, out_sharding=jax.P("x"))
    w = jax.random.normal(kw, (V, D), dtype=jnp.float32, out_sharding=jax.P(None, "x"))

  def loss(w: jax.Array, x, t):
    w_rep = jax.sharding.reshard(w, jax.P())
    gathered = w_rep.at[t].get(out_sharding=jax.P("x"))
    loss = (gathered * x).sum()
    return loss

  def loss_replicated(w, x, t):
    w_rep, x_rep, t_rep = jax.tree.map(
      lambda z: jax.sharding.reshard(z, jax.P()), (w, x, t)
    )
    gathered = w_rep[t_rep]
    loss = (gathered * x_rep).sum()
    return loss

  results = defaultdict(dict)

  for dtype in [jnp.bfloat16, jnp.float32]:
    for loss_fn in [loss, loss_replicated]:
      w_dtype, x_dtype, t_dtype = jax.tree.map(lambda z: z.astype(dtype), (w, x, t))
      with jax.set_mesh(mesh):
        loss, grad = jax.value_and_grad(loss_fn)(w_dtype, x_dtype, t_dtype)
      results[loss_fn.__name__][dtype.__name__] = (loss, grad)

  for dtype_str, d in results.items():
    for loss_str, (loss, grad) in d.items():
      print(dtype_str, loss_str, loss)
    errs = [g0 - g1 for (g0, g1) in combinations([v[-1] for v in d.values()], 2)]
    for err in errs:
      print(dtype_str, err)


if __name__ == "__main__":
  # main()
  mwe()
