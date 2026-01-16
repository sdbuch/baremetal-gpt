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


def _debug_log(msg, proc_idx=None):
  """Print debug message with host index prefix."""
  if proc_idx is None:
    proc_idx = jax.process_index()
  print(f"[DEBUG][H{proc_idx}] {msg}", flush=True)


def _get_sharded_axis(spec):
  """Find which axis is sharded (has non-None value in PartitionSpec)."""
  if spec is None:
    return None
  for i, s in enumerate(spec):
    if s is not None:
      return i
  return None


def _debug_save_microbatch_callback(outputs, targets, microbatch_idx):
  """Callback that runs with concrete values inside scan - saves microbatch outputs."""
  import ml_dtypes

  proc_idx = jax.process_index()
  save_dir = Path(f"/tmp/debug_loss_inputs/proc_{proc_idx}")
  save_dir.mkdir(parents=True, exist_ok=True)
  suffix = f"_mb{int(microbatch_idx)}"

  # Save arrays (concrete from callback)
  # For bfloat16: save as uint16 view to preserve exact bits
  outputs_np = np.asarray(outputs)
  targets_np = np.asarray(targets)

  dtypes = {"outputs": str(outputs_np.dtype), "targets": str(targets_np.dtype)}

  # Handle bfloat16 by viewing as uint16
  if outputs_np.dtype == ml_dtypes.bfloat16:
    outputs_np = outputs_np.view(np.uint16)
  if targets_np.dtype == ml_dtypes.bfloat16:
    targets_np = targets_np.view(np.uint16)

  np.save(save_dir / f"outputs{suffix}.npy", outputs_np)
  np.save(save_dir / f"targets{suffix}.npy", targets_np)

  # Save dtype info for this microbatch
  with open(save_dir / f"dtypes{suffix}.pkl", "wb") as f:
    pickle.dump(dtypes, f)

  _debug_log(
    f"saved microbatch {int(microbatch_idx)} outputs({dtypes['outputs']})/targets({dtypes['targets']})",
    proc_idx,
  )


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

  dtypes = {}
  specs = {}

  def save_sharded(arr, name):
    """Save per-process local data by concatenating device shards along sharded axis.

    For bfloat16: view as uint16 to preserve exact bits (numpy doesn't support bfloat16).
    """
    orig_dtype = arr.dtype
    dtypes[name] = str(orig_dtype)

    if hasattr(arr, "sharding") and hasattr(arr.sharding, "spec"):
      spec = tuple(arr.sharding.spec)
      specs[name] = spec
      sharded_axis = _get_sharded_axis(spec)
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


def debug_verify_reconstruction(batch, unemb, mesh, all_outputs):
  """Verify that saved data can be reconstructed and matches original.

  Called AFTER the scan completes (outside traced context).
  Uses NamedSharding(mesh, P(*spec)) pattern from data.py.

  Args:
    batch: (inputs, targets) tuple from dataloader
    unemb: original unemb params from before train_step
    mesh: JAX mesh for sharding
    all_outputs: outputs from train_step scan, shape (num_mb, batch_per_mb, seq, hidden)
  """
  import ml_dtypes
  from jax.sharding import NamedSharding

  proc_idx = jax.process_index()
  save_dir = Path(f"/tmp/debug_loss_inputs/proc_{proc_idx}")

  _debug_log("=== VERIFICATION START ===", proc_idx)

  with open(save_dir / "sharding_info.pkl", "rb") as f:
    sharding_info = pickle.load(f)
  dtypes = sharding_info.get("dtypes", {})
  specs = sharding_info.get("specs", {})

  def load_and_reconstruct(name):
    """Load saved local data and reconstruct sharded array using NamedSharding.

    For bfloat16: data was saved as uint16, convert via bytes for exact bit preservation.
    """
    data_uint16 = np.load(save_dir / f"{name}.npy")
    saved_shape = data_uint16.shape
    # For bfloat16: convert uint16 -> bytes -> bfloat16 for exact bit preservation
    if "bfloat16" in dtypes.get(name, ""):
      raw_bytes = data_uint16.tobytes()
      data = np.frombuffer(raw_bytes, dtype=ml_dtypes.bfloat16).reshape(saved_shape)
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
    return reconstructed

  results = {}

  def check_equal(name, orig, reconstructed):
    """Compare JAX arrays: check type (dtype+shape+sharding) and values."""
    # Check full type including dtype, shape, and sharding via jax.typeof
    orig_type = jax.typeof(orig)
    recon_type = jax.typeof(reconstructed)
    if orig_type != recon_type:
      results[name] = f"TYPE_MISMATCH({orig_type} vs {recon_type})"
      return False
    if jnp.array_equal(orig, reconstructed):
      results[name] = "EXACT"
      return True
    num_diff = jnp.sum(orig != reconstructed)
    results[name] = f"MISMATCH({num_diff} values differ)"
    return False

  inputs, targets = batch

  # Test 1: Reconstruct batch from saved local data using NamedSharding pattern
  inputs_recon = load_and_reconstruct("batch_inputs")
  targets_recon = load_and_reconstruct("batch_targets")
  unemb_w_recon = load_and_reconstruct("unemb_w")

  check_equal("batch_inputs", inputs, inputs_recon)
  check_equal("batch_targets", targets, targets_recon)
  # unemb is the ORIGINAL params from before train_step (passed in explicitly)
  check_equal("unemb_w", unemb.w, unemb_w_recon)

  # Test 2: Check that saved microbatches match outputs from train_step
  # Note: jax.debug.callback gathers data globally, so saved microbatches have global shape
  # Only H0 has the files (callback runs once per step)
  # Use process_allgather to get global data in multi-host setup
  from jax.experimental.multihost_utils import process_allgather

  # Gather targets and outputs globally - must run on ALL processes (collective op)
  batch_targets_global = process_allgather(targets, tiled=True)
  all_outputs_global = process_allgather(all_outputs, tiled=True)

  num_microbatches = all_outputs.shape[0]
  for mb_idx in range(num_microbatches):
    suffix = f"_mb{mb_idx}"
    if not (save_dir / f"targets{suffix}.npy").exists():
      results[f"mb{mb_idx}_targets"] = "MISSING"
      results[f"mb{mb_idx}_outputs"] = "MISSING"
      continue

    # Load dtype info for this microbatch
    with open(save_dir / f"dtypes{suffix}.pkl", "rb") as f:
      mb_dtypes = pickle.load(f)

    # Load microbatch targets saved from inside scan (global shape from callback)
    mb_targets_saved_np = np.load(save_dir / f"targets{suffix}.npy")
    # Convert back to bfloat16 if needed, then to JAX array
    if "bfloat16" in mb_dtypes.get("targets", ""):
      mb_targets_saved_np = np.frombuffer(
        mb_targets_saved_np.tobytes(), dtype=ml_dtypes.bfloat16
      ).reshape(mb_targets_saved_np.shape)
    mb_targets_saved = jnp.array(mb_targets_saved_np)

    # Compare against slice from full batch
    check_equal(f"mb{mb_idx}_targets", batch_targets_global[mb_idx], mb_targets_saved)

    # Load microbatch outputs and compare against train_step outputs
    mb_outputs_saved_np = np.load(save_dir / f"outputs{suffix}.npy")
    if "bfloat16" in mb_dtypes.get("outputs", ""):
      mb_outputs_saved_np = np.frombuffer(
        mb_outputs_saved_np.tobytes(), dtype=ml_dtypes.bfloat16
      ).reshape(mb_outputs_saved_np.shape)
    mb_outputs_saved = jnp.array(mb_outputs_saved_np)

    # Compare against outputs from train_step
    check_equal(f"mb{mb_idx}_outputs", all_outputs_global[mb_idx], mb_outputs_saved)

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
      # Return outputs as aux for verification
      return loss, outputs

    # Calculate gradients: use a scan for gradient accumulation
    def gradient_accum(loss__grad, microbatch_with_idx):
      loss_accum, grad_accum = loss__grad
      (loss, outputs), grad = jax.value_and_grad(loss_fn, has_aux=True)(
        state.params, microbatch_with_idx
      )
      return (loss_accum + loss, jax.tree.map(jnp.add, grad_accum, grad)), (
        loss,
        outputs,
      )

    zeros_like_fp32 = partial(jnp.zeros_like, dtype=jnp.float32)
    carry = (jnp.zeros(()), jax.tree.map(zeros_like_fp32, state.params))
    # Add microbatch indices to scan inputs
    num_microbatches = config.train_dataset.num_microbatches
    indexed_batch = (jnp.arange(num_microbatches), batch[0], batch[1])
    (loss, grad), (raw_losses, all_outputs) = jax.lax.scan(
      gradient_accum, carry, indexed_batch
    )
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
    # Return all_outputs for verification (shape: num_microbatches, batch_per_mb, seq_len, hidden)
    return metrics, new_state, all_outputs

  # Training loop
  with Logger(config) as logger:
    do_evals = partial(eval_loop, config, mesh=mesh, logger=logger)
    for step, batch in enumerate(batch_iter):
      # DEBUG: save original params BEFORE train_step modifies them
      if step == 0:
        original_unemb = train_state.params.unemb
      with jax.set_mesh(mesh):
        metrics, train_state, all_outputs = train_step(
          config, batch, train_state, step=step
        )
      # DEBUG: verify reconstruction OUTSIDE mesh context (step 0 only)
      if step == 0:
        debug_verify_reconstruction(batch, original_unemb, mesh, all_outputs)
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
