from dataclasses import dataclass

import jax
import jax.numpy as jnp


@jax.tree_util.register_static
@dataclass(kw_only=True, frozen=True)
class Config:
    # Experiment orchestration params
    mesh_axis_names: tuple[str, ...] = ("dp",)
    mesh_shape: tuple[int, ...] = (4,)
    seed: int = 1337

    # Data and training params
    seq_len: int = 256
    global_batch_size: int = 128
    num_steps: int = 10**3
    lr: float = 1e-3

    # Model architecture params
    num_vocab: int = 2**8
    d_model: int = 768
    num_heads: int = 12
    d_head: int = 64
    mlp_factor: int = 4
    num_layers: int = 12
    param_std: float = 0.02
    rope_theta: float = 10000.0
    max_seq_len: int = 1024

    # Model dtypes
    param_dtype = jnp.bfloat16  # weights, activations
    compute_dtype = jnp.float32  # layernorm, attn logits, rope
    optimizer_dtype = jnp.float32  # optimizer state

    # Model call-time params
    eps_ln: float = 1e-6
    use_bias_ln: bool = False
    use_fa: bool = True
    use_bias_mlp: bool = False
    use_rope: bool = True

    # Inference params
    update_cache: bool = False  # default training
    max_tokens_to_generate: int = 64
    temperature: float = 0.7

    # Model sharding params
    # TODO: Currently no support for pipeline parallel
    # TODO: test!
    sharding_data: jax.sharding.PartitionSpec = jax.P("dp")
    sharding_wqkv: jax.sharding.PartitionSpec = jax.P()
    sharding_wo: jax.sharding.PartitionSpec = jax.P()
    sharding_wup: jax.sharding.PartitionSpec = jax.P()
    sharding_wdown: jax.sharding.PartitionSpec = jax.P()
    sharding_mlp_hidden: jax.sharding.PartitionSpec = jax.P()
    sharding_res_stream: jax.sharding.PartitionSpec = jax.P()
    sharding_att_qkv: jax.sharding.PartitionSpec = jax.P()

    def __post_init__(self):
        # Set up and register mesh
        mesh = jax.make_mesh(
            self.mesh_shape,
            self.mesh_axis_names,
            len(self.mesh_shape) * (jax.sharding.AxisType.Explicit,),
        )
        jax.sharding.set_mesh(mesh)

        # Checks
        assert self.d_head % 2 == 0, (
            "Head dimension needs to be divisible by 2 for RoPE"
        )
