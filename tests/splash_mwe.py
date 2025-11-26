from functools import partial

import jax
import jax.numpy as jnp
from jax.experimental.pallas.ops.tpu.splash_attention import (
  BlockSizes,
  CausalMask,
  MultiHeadMask,
  SegmentIds,
  make_splash_mha,
)

NUM_HEADS = 4
SEQ_LEN = 256
HEAD_DIM = 64
DTYPE = jnp.bfloat16


def make_splash_kernel_with_shard_map(mesh):
  mask = MultiHeadMask([CausalMask(shape=(SEQ_LEN, SEQ_LEN)) for _ in range(NUM_HEADS)])
  block_sizes = BlockSizes(
    block_q=128,
    block_kv=128,
    block_kv_compute=128,
    block_q_dkv=128,
    block_kv_dkv=128,
    block_kv_dkv_compute=128,
    block_q_dq=128,
    block_kv_dq=128,
  )

  splash_spec = jax.P(None, None)
  sspec = jax.sharding.NamedSharding(mesh, splash_spec)

  kernel = make_splash_mha(
    mask,
    head_shards=1,
    q_seq_shards=1,
    block_sizes=block_sizes,
  )
  kspec = kernel.manual_sharding_spec(sspec)

  @partial(
    jax.shard_map,
    mesh=mesh,
    in_specs=(kspec, splash_spec, splash_spec, splash_spec, jax.P()),
    out_specs=splash_spec,
    check_vma=False,
  )
  def splash_sharded(kernel, q, k, v, segment_ids):
    return kernel(q, k, v, segment_ids=segment_ids)

  return splash_sharded, kernel


def attention_fn_with_internal_shard_map(splash_sharded, kernel, x_seq):
  s = x_seq.shape[0]

  q, k, v = jnp.ones((3, NUM_HEADS, s, HEAD_DIM), out_sharding=jax.P())
  segment_ids = SegmentIds(q=jnp.zeros((s,)), kv=jnp.zeros((s,)))

  scale = HEAD_DIM**-0.25
  out = splash_sharded(kernel, q * scale, k * scale, v, segment_ids)
  return out


def test_case_fails_vmap_outside_shard_map(mesh, batch_size):
  splash_sharded, kernel = make_splash_kernel_with_shard_map(mesh)

  d_model = NUM_HEADS * HEAD_DIM
  key = jax.random.key(0)
  input_sharding = jax.sharding.NamedSharding(mesh, jax.P("dp", None, None))
  x_seq = jax.random.normal(key, (batch_size, SEQ_LEN, d_model), dtype=DTYPE)
  x_seq = jax.device_put(x_seq, input_sharding)

  @jax.jit
  def step(x_seq):
    def loss_fn(x_seq):
      attn_fn = partial(attention_fn_with_internal_shard_map, splash_sharded, kernel)
      out = jax.vmap(attn_fn)(x_seq)
      return out.sum()

    loss, grads = jax.value_and_grad(loss_fn)(x_seq)
    return loss, grads

  with jax.set_mesh(mesh):
    jax.profiler.start_trace("/tmp/profile-step")
    loss, grads = step(x_seq)
    jax.block_until_ready((loss, grads))
    jax.profiler.stop_trace()
  return True


def main():
  jax.distributed.initialize()
  devices = jax.devices()
  mesh = jax.sharding.Mesh(devices, ("dp",), (jax.sharding.AxisType.Explicit,))

  batch_size = jax.device_count()
  test_case_fails_vmap_outside_shard_map(mesh, batch_size)
  print("passed")


if __name__ == "__main__":
  main()
