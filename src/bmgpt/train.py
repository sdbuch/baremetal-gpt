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

  This saves the sharded arrays properly so we can verify reconstruction.
  """
  from omegaconf import OmegaConf

  proc_idx = jax.process_index()
  save_dir = Path(f"/tmp/debug_loss_inputs/proc_{proc_idx}")
  save_dir.mkdir(parents=True, exist_ok=True)

  dtypes = {}  # Track original dtypes for bfloat16 handling

  def save_sharded(arr, name):
    """Save sharded array - stack local shards. Handle bfloat16 via device_get."""
    orig_dtype = arr.dtype
    dtypes[name] = str(orig_dtype)

    if hasattr(arr, "addressable_shards"):
      local_shards = []
      for s in arr.addressable_shards:
        # Use device_get to safely extract data, then handle bfloat16
        shard_np = jax.device_get(s.data)
        # bfloat16 arrays from device_get can be viewed as uint16 and converted
        if orig_dtype == jnp.bfloat16:
          # JAX device_get returns a numpy array that we can convert
          shard_np = np.array(shard_np, dtype=np.float32)
        else:
          shard_np = np.asarray(shard_np)
        local_shards.append(shard_np)
      data = np.stack(local_shards, axis=0)
    else:
      data = jax.device_get(arr)
      if orig_dtype == jnp.bfloat16:
        data = np.array(data, dtype=np.float32)
      else:
        data = np.asarray(data)
    np.save(save_dir / f"{name}.npy", data)
    return data.shape

  # batch is (inputs, targets), each with shape (num_microbatches, batch_per_mb, seq_len, ...)
  inputs, targets = batch
  inputs_shape = save_sharded(inputs, "batch_inputs")
  targets_shape = save_sharded(targets, "batch_targets")
  unemb_shape = save_sharded(unemb.w, "unemb_w")

  # Save config and sharding info
  config_dict = OmegaConf.to_container(config, resolve=True)
  sharding_info = {
    "config": config_dict,
    "inputs_sharding": str(inputs.sharding) if hasattr(inputs, "sharding") else None,
    "targets_sharding": str(targets.sharding) if hasattr(targets, "sharding") else None,
    "unemb_w_sharding": str(unemb.w.sharding) if hasattr(unemb.w, "sharding") else None,
    "dtypes": dtypes,  # Store original dtypes
  }
  with open(save_dir / "sharding_info.pkl", "wb") as f:
    pickle.dump(sharding_info, f)

  _debug_log(
    f"saved batch inputs={inputs_shape} targets={targets_shape} unemb={unemb_shape}",
    proc_idx,
  )


def debug_verify_reconstruction(batch, unemb, mesh):
  """Verify that saved data can be reconstructed and matches original.

  Called AFTER the scan completes (outside traced context).
  """
  proc_idx = jax.process_index()
  save_dir = Path(f"/tmp/debug_loss_inputs/proc_{proc_idx}")

  _debug_log("=== VERIFICATION START ===", proc_idx)

  # Load sharding info to get original dtypes
  with open(save_dir / "sharding_info.pkl", "rb") as f:
    sharding_info = pickle.load(f)
  dtypes = sharding_info.get("dtypes", {})

  def load_and_reconstruct(name, orig_sharding, orig_dtype):
    """Load saved shards and reconstruct sharded array."""
    data = np.load(save_dir / f"{name}.npy")
    # data is (num_local_shards, *shard_shape)
    local_shards = [data[i] for i in range(data.shape[0])]
    reconstructed = jax.make_array_from_process_local_data(orig_sharding, local_shards)
    # make_array_from_process_local_data may return a list in some JAX versions
    if isinstance(reconstructed, list):
      reconstructed = reconstructed[0]
    # Convert back to original dtype if it was bfloat16
    if "bfloat16" in dtypes.get(name, ""):
      reconstructed = reconstructed.astype(jnp.bfloat16)
    return reconstructed

  def to_local_numpy(arr):
    # Use device_get to safely extract data
    is_bf16 = hasattr(arr, "dtype") and arr.dtype == jnp.bfloat16

    if hasattr(arr, "addressable_shards"):
      local_shards = []
      for s in arr.addressable_shards:
        shard_np = jax.device_get(s.data)
        if is_bf16:
          shard_np = np.array(shard_np, dtype=np.float32)
        local_shards.append(shard_np)
      return np.stack(local_shards, axis=0)
    data = jax.device_get(arr)
    if is_bf16:
      data = np.array(data, dtype=np.float32)
    return data

  results = {}

  def check_equal(name, orig, reconstructed):
    orig_local = to_local_numpy(orig)
    recon_local = to_local_numpy(reconstructed)
    if np.array_equal(orig_local, recon_local):
      results[name] = "EXACT"
      return True
    elif np.allclose(orig_local.astype(np.float32), recon_local.astype(np.float32)):
      results[name] = "CLOSE"
      return True
    else:
      max_diff = np.max(
        np.abs(orig_local.astype(np.float32) - recon_local.astype(np.float32))
      )
      results[name] = f"MISMATCH(diff={max_diff:.2e})"
      return False

  inputs, targets = batch

  # Test 1: Reconstruct batch from saved shards
  inputs_recon = load_and_reconstruct("batch_inputs", inputs.sharding, inputs.dtype)
  targets_recon = load_and_reconstruct("batch_targets", targets.sharding, targets.dtype)
  unemb_w_recon = load_and_reconstruct("unemb_w", unemb.w.sharding, unemb.w.dtype)

  check_equal("batch_inputs", inputs, inputs_recon)
  check_equal("batch_targets", targets, targets_recon)
  check_equal("unemb_w", unemb.w, unemb_w_recon)

  # Test 2: Check that saved microbatches match slices of the full batch
  for mb_idx in range(2):
    suffix = f"_mb{mb_idx}"
    if not (save_dir / f"targets{suffix}.npy").exists():
      results[f"mb{mb_idx}_targets"] = "MISSING"
      continue

    # Load microbatch targets saved from inside scan
    mb_targets_saved = np.load(save_dir / f"targets{suffix}.npy")

    # Get corresponding slice from full batch
    # batch_targets has shape (num_local_shards, num_microbatches, batch_per_mb, seq_len)
    batch_targets_local = to_local_numpy(targets)
    mb_targets_from_batch = batch_targets_local[:, mb_idx, :, :]

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

    # DEBUG: verify reconstruction AFTER scan (step 0 only)
    if step == 0:
      debug_verify_reconstruction(batch, state.params.unemb, mesh)
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
