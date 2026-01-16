"""
CLEAN DIAGNOSTIC TEST - Run on your TPU v4

This will tell us:
1. Does error correlate with number of index duplicates per row?
2. What's the error distribution?
3. Is this purely from all-reduce accumulation or something else?
"""

import jax
import jax.numpy as jnp


def run_diagnostic():
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

  def loss(w, x, t):
    w_rep = jax.sharding.reshard(w, jax.P())
    gathered = w_rep.at[t].get(out_sharding=jax.P("x"))
    return jnp.einsum(
      "td,td->", gathered, x, preferred_element_type=jnp.float32, out_sharding=jax.P()
    )

  def loss_replicated(w, x, t):
    w_rep, x_rep, t_rep = jax.tree.map(
      lambda z: jax.sharding.reshard(z, jax.P()), (w, x, t)
    )
    gathered = w_rep[t_rep]
    return jnp.einsum(
      "td,td->",
      gathered,
      x_rep,
      preferred_element_type=jnp.float32,
      out_sharding=jax.P(),
    )

  dtype = jnp.bfloat16

  with jax.set_mesh(mesh):
    w_bf16, x_bf16 = w.astype(dtype), x.astype(dtype)

    g_sharded = jax.jit(jax.grad(loss))(w_bf16, x_bf16, t)
    g_replicated = jax.jit(jax.grad(loss_replicated))(w_bf16, x_bf16, t)

    # Gather everything for analysis - ALL devices must participate
    g_sharded_full = jax.sharding.reshard(g_sharded, jax.P())
    g_replicated_full = jax.sharding.reshard(g_replicated, jax.P())
    t_full = jax.sharding.reshard(t, jax.P())

    # Compute all metrics BEFORE the process_index check (all devices participate)
    diff = g_sharded_full - g_replicated_full

    l1_err = float(jnp.abs(diff).sum())
    max_err = float(jnp.abs(diff).max())
    grad_norm = float(jnp.linalg.norm(g_replicated_full))

    # Per-row analysis
    row_errors = jnp.abs(diff).sum(axis=1)  # [V]
    row_hits = jnp.zeros(V, dtype=jnp.int32).at[t_full].add(1)
    max_hits = int(row_hits.max())

    # Pre-compute stats for each hit count
    hit_stats = []
    for n_hits in range(0, max_hits + 1):
      mask = row_hits == n_hits
      n_rows = int(mask.sum())
      if n_rows > 0:
        mean_err = float(row_errors[mask].mean())
        max_err_row = float(row_errors[mask].max())
        hit_stats.append((n_hits, n_rows, mean_err, max_err_row))

    # Now only process 0 prints
    if jax.process_index() == 0:
      print("=" * 60)
      print("ERROR ANALYSIS")
      print("=" * 60)
      print(f"L1 error: {l1_err:.4f}")
      print(f"Max element error: {max_err:.6f}")
      print(f"Grad norm: {grad_norm:.2f}")
      print()

      print("=" * 60)
      print("ERROR vs INDEX DUPLICATES")
      print("=" * 60)
      print(f"{'Hits':<6} {'Rows':<8} {'Mean Err':<12} {'Max Err':<12}")
      print("-" * 40)

      for n_hits, n_rows, mean_err, max_err_row in hit_stats:
        print(f"{n_hits:<6} {n_rows:<8} {mean_err:<12.6f} {max_err_row:<12.6f}")

      print()
      print("=" * 60)
      print("INTERPRETATION")
      print("=" * 60)
      print("If error is 0 for rows with 0-1 hits and increases with hits:")
      print("  -> Error is from different accumulation ORDER (associativity)")
      print()
      print("If error is ~uniform across all rows regardless of hits:")
      print("  -> Error is from all-reduce precision loss")
      print()
      print("If error is only in rows with >0 hits but doesn't scale with hits:")
      print("  -> Error is from scatter implementation difference")


if __name__ == "__main__":
  run_diagnostic()
