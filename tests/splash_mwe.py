from functools import partial

import jax
import jax.numpy as jnp
from jax.experimental.pallas.ops.tpu.splash_attention import (
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
  splash_spec = jax.P(None, None)
  sspec = jax.sharding.NamedSharding(mesh, splash_spec)

  kernel = make_splash_mha(mask, head_shards=1, q_seq_shards=1)
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


def attention_fn_with_internal_shard_map(splash_sharded, kernel, qkv_proj, x_seq):
  s = x_seq.shape[0]
  qkv = jnp.einsum("sd,dhne->nhse", x_seq, qkv_proj)
  qkv = qkv.astype(x_seq.dtype)
  q, k, v = qkv[0], qkv[1], qkv[2]

  segment_ids = SegmentIds(q=jnp.zeros((s,)), kv=jnp.zeros((s,)))

  scale = HEAD_DIM**-0.25
  out = splash_sharded(kernel, q * scale, k * scale, v, segment_ids)
  return out


def test_case_fails_vmap_outside_shard_map(mesh, batch_size):
  splash_sharded, kernel = make_splash_kernel_with_shard_map(mesh)

  d_model = NUM_HEADS * HEAD_DIM
  key = jax.random.key(0)
  key_x, key_qkv = jax.random.split(key)

  input_sharding = jax.sharding.NamedSharding(mesh, jax.P("dp", None, None))
  x_seq = jax.random.normal(key_x, (batch_size, SEQ_LEN, d_model), dtype=DTYPE)
  x_seq = jax.device_put(x_seq, input_sharding)

  param_sharding = jax.sharding.NamedSharding(mesh, jax.P(None, None, None, None))
  qkv_proj = jax.random.normal(key_qkv, (d_model, 3, NUM_HEADS, HEAD_DIM), dtype=DTYPE)
  qkv_proj = jax.device_put(qkv_proj, param_sharding)

  @jax.jit
  def step(qkv_proj, x_seq):
    def loss_fn(qkv_proj, x_seq):
      attn_fn = partial(
        attention_fn_with_internal_shard_map, splash_sharded, kernel, qkv_proj
      )
      out = jax.vmap(attn_fn)(x_seq)
      return out.sum()

    loss, grads = jax.value_and_grad(loss_fn, argnums=(0, 1))(qkv_proj, x_seq)
    return loss, grads

  with jax.set_mesh(mesh):
    loss, grads = step(qkv_proj, x_seq)
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
