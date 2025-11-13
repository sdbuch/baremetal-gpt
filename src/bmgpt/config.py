from dataclasses import dataclass, field
from enum import Enum

import jax
import jax.numpy as jnp
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING


class DType(Enum):
    FLOAT32 = jnp.float32
    FLOAT16 = jnp.float16
    BFLOAT16 = jnp.bfloat16
    INT32 = jnp.int32
    INT16 = jnp.int16


class OptType(Enum):
    ADAMW = "adamw"
    SGD = "sgd"


class LoggerType(Enum):
    PRINT = "print"
    WANDB = "wandb"


class TransformerType(Enum):
    """Are inputs tokens or vectors (ViT-like)?"""

    DISCRETE = "discrete"
    CONTINUOUS = "continuous"


class DatasetName(Enum):
    """Supported datasets we train on. See data.py for factory"""

    NUMBER_STAIRCASE = "number_staircase"
    MNIST = "mnist"


@dataclass(kw_only=True, unsafe_hash=True)
class Config:
    ## Experiment orchestration params
    mesh_axis_names: list[str] = field(default_factory=lambda: ["dp"])
    mesh_shape: list[int] = MISSING
    seed: int = 1337
    logger_type: LoggerType = LoggerType.WANDB
    project_name: str = "bmgpt-debug"
    run_name: str = ""

    ## Data params
    dataset_name: DatasetName = MISSING
    seq_len: int = MISSING
    num_vocab: int = MISSING
    global_batch_size: int = 128

    ## Optimizer params
    num_steps: int = 10**3
    optimizer_type: OptType = OptType.ADAMW
    lr: float = 1e-3
    beta1: float = 0.9
    beta2: float = 0.999
    eps_adam: float = 1e-8
    weight_decay: float = 1e-2
    clip_grad: float = 1.0  # global ell^2 norm

    ## Model architecture params
    # Overarching
    transformer_type: TransformerType = MISSING

    # Transformer-type-agnostic params
    is_causal: bool = True
    d_model: int = 768
    num_heads: int = 12
    d_head: int = 64
    mlp_factor: int = 4
    num_layers: int = 12
    param_std: float = 0.02
    rope_theta: float = 10000.0
    max_seq_len: int = 1024

    # Model dtypes
    param_dtype: DType = DType.BFLOAT16  # weights, activations
    compute_dtype: DType = DType.FLOAT32  # layernorm, attn logits, rope
    optimizer_dtype: DType = DType.FLOAT32  # optimizer state

    # Model call-time params
    eps_ln: float = 1e-6  # epsilon for layer norm
    use_bias_ln: bool = False  # layer norm or RMS norm
    use_fa: bool = True  # use JAX's dot_product_attention or not
    use_bias_mlp: bool = False  # bias in MLPs
    use_rope: bool = True  # RoPE or not

    # Discrete-specific model parameters
    # Continuous-specific model parameters

    ## Autoregressive inference params
    max_tokens_to_generate: int = 64
    temperature: float = 0.7

    ## Model sharding params (args to jax.P)-- list of mesh_axis_names els or None
    # NOTE: technically jax.P can merge axes, e.g. (('x', 'y')), but we reject this
    sharding_data: list[str | None] = field(default_factory=lambda: ["dp"])
    sharding_wqkv: list[str | None] = field(default_factory=list)  # D x 3 x N x H
    sharding_wo: list[str | None] = field(default_factory=list)  # D x N x H
    sharding_wup: list[str | None] = field(default_factory=list)  # D x 4D
    sharding_wdown: list[str | None] = field(default_factory=list)  # 4D x D
    sharding_mlp_hidden: list[str | None] = field(default_factory=list)  # S x 4D
    sharding_res_stream: list[str | None] = field(default_factory=list)  # S x D
    sharding_att_qkv: list[str | None] = field(default_factory=list)  # 3 x S x N x H


def register_configs():
    cs = ConfigStore.instance()
    cs.store(name="config", node=Config)


def mesh_from_config(config: Config):
    mesh = jax.make_mesh(
        config.mesh_shape,
        config.mesh_axis_names,
        len(config.mesh_shape) * (jax.sharding.AxisType.Explicit,),
    )
    return mesh


def config_post_init(config: Config):
    """Call after jax.distributed.initialize()"""
    # Register the argument's type as static (since hydra wraps Config)
    jax.tree_util.register_static(type(config))
    # Check arguments
    assert config.d_head % 2 == 0, "Head dimension needs to be divisible by 2 for RoPE"
    assert config.global_batch_size % jax.process_count() == 0, (
        "Number of hosts needs to divide the global batch size"
    )
