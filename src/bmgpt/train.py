import pickle
from functools import partial
from pathlib import Path
from typing import Any, Iterable, NamedTuple

import hydra
import jax
import jax.numpy as jnp
import numpy as np

from bmgpt.config import (
  Config,
  EvaluationConfig,
  config_post_init,
  mesh_from_config,
  register_configs,
)
from bmgpt.data import get_distributed_batch_iter
from bmgpt.evaluators import evaluator_factory
from bmgpt.loggers import Logger, logger_factory
from bmgpt.losses import fused_softmax_cross_entropy, softmax_cross_entropy
from bmgpt.model import (
  CacheParams,
  Transformer,
  _transformer,
  init_kv_cache,
  init_transformer,
  model_spec,
)
from bmgpt.optimizers import grad_norm_and_clip, init_adam_state, opt_update_factory
from bmgpt.splash_helpers import forward_kernels_from_config

register_configs()


# DEBUG: Save loss inputs for debugging fused xent backward
_debug_microbatch_counter = 0  # Track which microbatch we're on


def _debug_log(msg, proc_idx=None, host0_only=False):
  """Print debug message with consistent formatting."""
  if proc_idx is None:
    proc_idx = jax.process_index()
  if host0_only and proc_idx != 0:
    return
  print(f"[DEBUG][H{proc_idx}] {msg}", flush=True)


def _debug_save_microbatch_callback(outputs, targets, microbatch_idx):
  """Callback that runs with concrete values inside scan - saves microbatch outputs."""
  proc_idx = jax.process_index()
  save_dir = Path(f"/tmp/debug_loss_inputs/proc_{proc_idx}")
  save_dir.mkdir(parents=True, exist_ok=True)
  suffix = f"_mb{int(microbatch_idx)}"

  # Save arrays (concrete from callback)
  # In callback, arrays are already concrete numpy-compatible
  # bfloat16 shows as float32 after going through callback
  np.save(save_dir / f"outputs{suffix}.npy", np.asarray(outputs, dtype=np.float32))
  np.save(save_dir / f"targets{suffix}.npy", np.asarray(targets))

  _debug_log(f"saved microbatch {int(microbatch_idx)} outputs/targets", proc_idx)


def debug_save_microbatch(outputs, targets, microbatch_idx):
  """Save microbatch outputs/targets from inside scan using jax.debug.callback."""
  jax.debug.callback(_debug_save_microbatch_callback, outputs, targets, microbatch_idx)


def debug_save_batch_and_weights(config, batch, unemb, mesh):
  """Save full batch and weights BEFORE the scan (outside traced context).

  Saves per-process local data (not per-device) following the pattern in data.py.
  This allows reconstruction via NamedSharding(mesh, P(*spec)).
  """
  from omegaconf import OmegaConf

  proc_idx = jax.process_index()
  save_dir = Path(f"/tmp/debug_loss_inputs/proc_{proc_idx}")
  save_dir.mkdir(parents=True, exist_ok=True)

  dtypes = {}  # Track original dtypes for bfloat16 handling
  specs = {}  # Track partition specs for reconstruction

  def get_sharded_axis(spec):
    """Find which axis is sharded (has non-None value in PartitionSpec)."""
    for i, s in enumerate(spec):
      if s is not None:
        return i
    return None

  def save_sharded(arr, name):
    """Save per-process local data by concatenating device shards along sharded axis.

    For bfloat16: view as uint16 to preserve exact bits (numpy doesn't support bfloat16).
    """
    orig_dtype = arr.dtype
    dtypes[name] = str(orig_dtype)

    # Get partition spec from NamedSharding
    if hasattr(arr, "sharding") and hasattr(arr.sharding, "spec"):
      spec = tuple(arr.sharding.spec)
      specs[name] = spec
      sharded_axis = get_sharded_axis(spec)
    else:
      specs[name] = None
      sharded_axis = None

    if hasattr(arr, "addressable_shards"):
      local_shards = []
      for s in arr.addressable_shards:
        shard_np = jax.device_get(s.data)
        shard_np = np.asarray(shard_np)
        # For bfloat16: view as uint16 to preserve exact bits
        if orig_dtype == jnp.bfloat16:
          shard_np = shard_np.view(np.uint16)
        local_shards.append(shard_np)
      # Concatenate along sharded axis to get per-process data (like dataloader produces)
      if sharded_axis is not None and len(local_shards) > 1:
        data = np.concatenate(local_shards, axis=sharded_axis)
      else:
        # No sharding or single shard - just use the first shard
        data = (
          local_shards[0] if len(local_shards) == 1 else np.stack(local_shards, axis=0)
        )
    else:
      data = jax.device_get(arr)
      data = np.asarray(data)
      if orig_dtype == jnp.bfloat16:
        data = data.view(np.uint16)
    np.save(save_dir / f"{name}.npy", data)
    _debug_log(
      f"save {name}: orig_dtype={orig_dtype}, saved_dtype={data.dtype}, shape={data.shape}",
      proc_idx,
    )
    return data.shape

  # batch is (inputs, targets), each with shape (num_microbatches, batch_per_mb, seq_len, ...)
  inputs, targets = batch
  inputs_local_shape = save_sharded(inputs, "batch_inputs")
  targets_local_shape = save_sharded(targets, "batch_targets")
  unemb_local_shape = save_sharded(unemb.w, "unemb_w")

  # Save config and sharding info - include partition specs for reconstruction
  config_dict = OmegaConf.to_container(config, resolve=True)
  sharding_info = {
    "config": config_dict,
    "specs": specs,  # Partition specs as tuples
    "dtypes": dtypes,
  }
  with open(save_dir / "sharding_info.pkl", "wb") as f:
    pickle.dump(sharding_info, f)

  _debug_log(
    f"saved batch inputs={inputs_local_shape} targets={targets_local_shape} "
    f"unemb={unemb_local_shape} | specs={specs}",
    proc_idx,
  )


def debug_verify_reconstruction(batch, unemb, mesh):
  """Verify that saved data can be reconstructed and matches original.

  Called AFTER the scan completes (outside traced context).
  Uses NamedSharding(mesh, P(*spec)) pattern from data.py.
  """
  from jax.sharding import NamedSharding

  proc_idx = jax.process_index()
  save_dir = Path(f"/tmp/debug_loss_inputs/proc_{proc_idx}")

  _debug_log("=== VERIFICATION START ===", proc_idx)

  # Load sharding info
  with open(save_dir / "sharding_info.pkl", "rb") as f:
    sharding_info = pickle.load(f)
  dtypes = sharding_info.get("dtypes", {})
  specs = sharding_info.get("specs", {})

  def get_sharded_axis(spec):
    """Find which axis is sharded (has non-None value in PartitionSpec)."""
    if spec is None:
      return None
    for i, s in enumerate(spec):
      if s is not None:
        return i
    return None

  def load_and_reconstruct(name, orig_dtype):
    """Load saved local data and reconstruct sharded array using NamedSharding.

    For bfloat16: data was saved as uint16, convert via bytes for exact bit preservation.
    """
    import ml_dtypes

    data_uint16 = np.load(save_dir / f"{name}.npy")
    saved_dtype = data_uint16.dtype
    saved_shape = data_uint16.shape
    # For bfloat16: convert uint16 -> bytes -> bfloat16 for exact bit preservation
    if "bfloat16" in dtypes.get(name, ""):
      # Use tobytes/frombuffer for exact bit-level conversion
      raw_bytes = data_uint16.tobytes()
      data = np.frombuffer(raw_bytes, dtype=ml_dtypes.bfloat16).reshape(saved_shape)
      # Debug: print first few uint16 values from file
      _debug_log(
        f"load {name}: first uint16 from file: {data_uint16.flat[:3]}",
        proc_idx,
      )
    else:
      data = data_uint16
    spec = specs.get(name)
    if spec is not None:
      # Reconstruct using NamedSharding(mesh, P(*spec)) pattern from data.py
      sharding = NamedSharding(mesh, jax.sharding.PartitionSpec(*spec))
      reconstructed = jax.make_array_from_process_local_data(sharding, data)
    else:
      # No sharding info - just convert to jax array
      reconstructed = jnp.array(data)
    _debug_log(
      f"load {name}: saved_dtype={saved_dtype}, loaded_dtype={data.dtype}, "
      f"recon_dtype={reconstructed.dtype}, shape={data.shape}",
      proc_idx,
    )
    return reconstructed

  def to_local_numpy(arr, spec=None):
    """Convert sharded array to per-process numpy by concatenating along sharded axis.

    For bfloat16: view as uint16 for exact bit comparison.
    """
    is_bf16 = hasattr(arr, "dtype") and arr.dtype == jnp.bfloat16
    sharded_axis = get_sharded_axis(spec)

    if hasattr(arr, "addressable_shards"):
      local_shards = []
      for s in arr.addressable_shards:
        shard_np = jax.device_get(s.data)
        shard_np = np.asarray(shard_np)
        # For bfloat16: view as uint16 for exact bit comparison
        if is_bf16:
          shard_np = shard_np.view(np.uint16)
        local_shards.append(shard_np)
      # Concatenate along sharded axis (same as save)
      if sharded_axis is not None and len(local_shards) > 1:
        return np.concatenate(local_shards, axis=sharded_axis)
      return (
        local_shards[0] if len(local_shards) == 1 else np.stack(local_shards, axis=0)
      )
    data = jax.device_get(arr)
    data = np.asarray(data)
    if is_bf16:
      data = data.view(np.uint16)
    return data

  results = {}

  def check_equal(name, orig, reconstructed, spec=None):
    orig_local = to_local_numpy(orig, spec)
    recon_local = to_local_numpy(reconstructed, spec)
    if orig_local.shape != recon_local.shape:
      results[name] = f"SHAPE_MISMATCH({orig_local.shape} vs {recon_local.shape})"
      return False
    if np.array_equal(orig_local, recon_local):
      results[name] = "EXACT"
      return True
    # For uint16 views (bfloat16), array_equal is the only valid check
    # Don't try float conversion on uint16 data
    if orig_local.dtype == np.uint16:
      num_diff = np.sum(orig_local != recon_local)
      # Debug: print first few values where they differ
      diff_idx = np.where(orig_local != recon_local)
      if len(diff_idx[0]) > 0:
        first_diff = tuple(idx[0] for idx in diff_idx)
        _debug_log(
          f"{name} first diff at {first_diff}: "
          f"orig={orig_local[first_diff]} recon={recon_local[first_diff]}",
          proc_idx,
        )
      results[name] = f"MISMATCH({num_diff} bits differ)"
      return False
    # For other dtypes, try float comparison
    if np.allclose(orig_local.astype(np.float32), recon_local.astype(np.float32)):
      results[name] = "CLOSE"
      return True
    max_diff = np.max(
      np.abs(orig_local.astype(np.float32) - recon_local.astype(np.float32))
    )
    results[name] = f"MISMATCH(diff={max_diff:.2e})"
    return False

  inputs, targets = batch

  # Test 1: Reconstruct batch from saved local data using NamedSharding pattern
  inputs_recon = load_and_reconstruct("batch_inputs", inputs.dtype)
  targets_recon = load_and_reconstruct("batch_targets", targets.dtype)
  unemb_w_recon = load_and_reconstruct("unemb_w", unemb.w.dtype)

  # Debug: compare first few uint16 values from original and reconstructed unemb_w
  if hasattr(unemb.w, "addressable_shards"):
    orig_first_shard = jax.device_get(list(unemb.w.addressable_shards)[0].data)
    orig_first_uint16 = np.asarray(orig_first_shard).view(np.uint16).flat[:3]
    recon_first_shard = jax.device_get(list(unemb_w_recon.addressable_shards)[0].data)
    recon_first_uint16 = np.asarray(recon_first_shard).view(np.uint16).flat[:3]
    _debug_log(f"unemb_w orig first shard uint16: {orig_first_uint16}", proc_idx)
    _debug_log(f"unemb_w recon first shard uint16: {recon_first_uint16}", proc_idx)

  check_equal("batch_inputs", inputs, inputs_recon, specs.get("batch_inputs"))
  check_equal("batch_targets", targets, targets_recon, specs.get("batch_targets"))
  check_equal("unemb_w", unemb.w, unemb_w_recon, specs.get("unemb_w"))

  # Test 2: Check that saved microbatches match slices of the full batch
  # Note: jax.debug.callback gathers data globally, so saved microbatches have global shape
  # Only H0 has the files (callback runs once per step)
  # Use process_allgather to get global data in multi-host setup
  from jax.experimental.multihost_utils import process_allgather

  # Gather targets globally - must run on ALL processes (collective op)
  batch_targets_global = np.asarray(process_allgather(targets, tiled=True))

  for mb_idx in range(2):
    suffix = f"_mb{mb_idx}"
    if not (save_dir / f"targets{suffix}.npy").exists():
      results[f"mb{mb_idx}_targets"] = "MISSING"
      continue

    # Load microbatch targets saved from inside scan (global shape from callback)
    mb_targets_saved = np.load(save_dir / f"targets{suffix}.npy")

    # Get corresponding slice from full batch
    mb_targets_from_batch = batch_targets_global[mb_idx, :, :]

    if np.array_equal(mb_targets_saved, mb_targets_from_batch):
      results[f"mb{mb_idx}_targets"] = "EXACT"
    else:
      results[f"mb{mb_idx}_targets"] = (
        f"MISMATCH(shapes:{mb_targets_saved.shape}vs{mb_targets_from_batch.shape})"
      )

  # Print compact summary
  summary = " | ".join(f"{k}:{v}" for k, v in results.items())
  _debug_log(f"VERIFY: {summary}", proc_idx)
  _debug_log("=== VERIFICATION DONE ===", proc_idx)


# Setup for training loop
class TrainState(NamedTuple):
  params: Transformer
  opt_state: Any
  kv_cache: jax.Array


@jax.jit
def init_train_state(key, config: Config) -> TrainState:
  model_params = init_transformer(key, config)
  adam_state = jax.tree.map(partial(init_adam_state, config), model_params)
  cache = init_kv_cache(
    config,
    config.train_dataset.global_batch_size,
    config.train_dataset.num_microbatches,
    0,
  )
  return TrainState(params=model_params, opt_state=adam_state, kv_cache=cache)


@hydra.main(
  version_base=None,
  config_path=str(Path("configs").absolute().resolve()),
  config_name="base_config",
)
def main(config: Config):
  try:
    # Launch distributed and register configs
    jax.distributed.initialize()
    jax.tree_util.register_static(type(config))
  except RuntimeError:
    # This implies the distributed backend has already been initialized
    pass
  config_post_init(config)
  mesh = mesh_from_config(config)
  Logger = logger_factory(config.logger_type)

  # Randomness
  key = jax.random.key(config.seed)
  key_model, key_train, key_val, key_eval = jax.random.split(key, 4)

  # Data
  batch_iter = get_distributed_batch_iter(config, config.train_dataset, key_train, mesh)

  # Initialize state
  with jax.set_mesh(mesh):
    train_state = init_train_state(key_model, config)
  cache_params = CacheParams(enabled=False, size=0)

  # Configure forward pass (attention kernels)
  train_attn_kernel, train_lse_kernel, val_kernels, eval_kernels = (
    forward_kernels_from_config(config, mesh)
  )

  # Configure optimization
  spec = model_spec(train_state.params)
  weight_decay_mask = jax.tree.map(lambda _, s: bool(s), train_state.params, spec)
  opt_update = opt_update_factory(config.optimizer.type)
  opt_update = partial(opt_update, config, weight_decay_mask)

  # DEBUG: disabled JIT for debugging fused xent backward
  # @partial(jax.jit, donate_argnums=2)
  def train_step(config: Config, batch, state: TrainState, step: int = -1):
    # DEBUG: save full batch and weights BEFORE scan (step 0 only)
    if step == 0:
      debug_save_batch_and_weights(config, batch, state.params.unemb, mesh)

    def loss_fn(params: Transformer, microbatch_with_idx):
      microbatch_idx, inputs, targets = microbatch_with_idx
      outputs, _ = jax.vmap(
        partial(
          _transformer, config, train_attn_kernel, params, cache_params=cache_params
        )
      )(inputs, state.kv_cache)

      # DEBUG: save microbatch outputs/targets at step 0
      if step == 0:
        debug_save_microbatch(outputs, targets, microbatch_idx)

      if config.use_fused_xent_loss:
        loss = fused_softmax_cross_entropy(
          config, params.unemb, outputs, targets, train_lse_kernel
        )
      else:
        loss = softmax_cross_entropy(config, params.unemb, outputs, targets)
      return loss

    # Calculate gradients: use a scan for gradient accumulation
    def gradient_accum(loss__grad, microbatch_with_idx):
      loss_accum, grad_accum = loss__grad
      loss, grad = jax.value_and_grad(loss_fn)(state.params, microbatch_with_idx)
      return (loss_accum + loss, jax.tree.map(jnp.add, grad_accum, grad)), loss

    zeros_like_fp32 = partial(jnp.zeros_like, dtype=jnp.float32)
    carry = (jnp.zeros(()), jax.tree.map(zeros_like_fp32, state.params))
    # Add microbatch indices to scan inputs
    num_microbatches = config.train_dataset.num_microbatches
    indexed_batch = (jnp.arange(num_microbatches), batch[0], batch[1])
    (loss, grad), raw_losses = jax.lax.scan(gradient_accum, carry, indexed_batch)
    assert raw_losses.dtype == jnp.float32

    # NOTE: breaks if per-token loss masking introduced (see unsloth blog)
    loss = loss / config.train_dataset.num_microbatches
    grad = jax.tree.map(lambda x: x / config.train_dataset.num_microbatches, grad)

    # Update parameters
    grad_clipped, _, global_grad_norm = grad_norm_and_clip(config, grad)
    update_tree = jax.tree.map(opt_update, state.params, grad_clipped, state.opt_state)
    update = jax.tree.map(lambda _, y: y[0], state.params, update_tree)
    opt_state = jax.tree.map(lambda _, y: y[1], state.params, update_tree)
    params = jax.tree.map(jnp.add, state.params, update)
    new_state = TrainState(params=params, opt_state=opt_state, kv_cache=state.kv_cache)

    metrics = {"batch_loss": loss, "grad_norm": global_grad_norm}
    return metrics, new_state

  # Training loop
  with Logger(config) as logger:
    do_evals = partial(eval_loop, config, mesh=mesh, logger=logger)
    for step, batch in enumerate(batch_iter):
      with jax.set_mesh(mesh):
        metrics, train_state = train_step(config, batch, train_state, step=step)
      # DEBUG: verify reconstruction OUTSIDE mesh context (step 0 only)
      if step == 0:
        debug_verify_reconstruction(batch, train_state.params.unemb, mesh)
      logger.log(metrics | {"step": step})
      if (step + 1) % config.val_log_interval == 0:
        # Calculate val metrics
        key_val = do_evals(
          key_val, zip(config.val_list, val_kernels), train_state.params, step
        )
      if step == config.train_dataset.num_steps - 1:
        break

    # Run evals (testing)
    key_eval = do_evals(
      key_eval, zip(config.eval_list, eval_kernels), train_state.params, step
    )


def eval_loop(
  config: Config,
  key,
  evals_and_kernels: Iterable[tuple[EvaluationConfig, Any]],
  params: Transformer,
  step: int,
  logger: Logger,
  mesh,
):
  logger.flush_buffer()
  for evaluation, shard_mapped__kernel in evals_and_kernels:
    key, key_d, key_e = jax.random.split(key, 3)
    batch_iter = get_distributed_batch_iter(config, evaluation.dataset, key_d, mesh)
    assert evaluation.dataset.num_microbatches == 1, (
      "Microbatching for evaluators not supported"
    )
    batch_iter = map(lambda b: jax.tree.map(lambda x: x.squeeze(0), b), batch_iter)
    evaluation_fn = evaluator_factory(evaluation)
    metrics = evaluation_fn(
      config, key_e, shard_mapped__kernel, mesh, params, batch_iter
    )
    logger.log(metrics | {"step": step})
  logger.flush_buffer()
  return key


if __name__ == "__main__":
  main()
