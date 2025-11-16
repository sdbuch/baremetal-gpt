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
class ExperimentConfig:
    ## Experiment orchestration params
    seed: int = 1337
    logger_type: LoggerType = LoggerType.WANDB
    project_name: str = "bmgpt-debug"
    run_name: str = ""


@dataclass(kw_only=True, unsafe_hash=True)
class DatasetConfig:
    ## Data params
    name: DatasetName = MISSING
    path: str = MISSING
    seq_len: int = MISSING
    num_vocab: int = MISSING
    global_batch_size: int = 128


@dataclass(kw_only=True, unsafe_hash=True)
class OptimizerConfig:
    ## Optimizer params
    num_steps: int = 10**3
    type: OptType = OptType.ADAMW
    lr: float = 3e-4
    beta1: float = 0.9
    beta2: float = 0.999
    eps_adam: float = 1e-8
    weight_decay: float = 1e-2
    clip_grad: float = 1.0  # global ell^2 norm


@dataclass(kw_only=True, unsafe_hash=True)
class ModelConfig:
    ## Model architecture params
    # Overarching
    transformer_type: TransformerType = MISSING
    max_seq_len: int = 1024  # should be larger than train sequence length

    # Transformer-type-agnostic params
    is_causal: bool = True
    d_model: int = 768
    num_heads: int = 12
    d_head: int = 64
    mlp_factor: int = 4
    num_layers: int = 12
    param_std: float = 0.02
    rope_theta: float = 10000.0

    # Model dtypes
    param_dtype: DType = DType.BFLOAT16  # weights, activations
    compute_dtype: DType = DType.FLOAT32  # layernorm, attn logits, rope
    optimizer_dtype: DType = DType.FLOAT32  # optimizer state

    # Model call-time params
    eps_ln: float = 1e-6
    use_bias_ln: bool = False
    use_splash: bool = True
    use_bias_mlp: bool = False
    use_rope: bool = True

    # Discrete-specific model parameters
    # Continuous-specific model parameters


@dataclass(kw_only=True, unsafe_hash=True)
class InferenceConfig:
    ## Autoregressive inference params
    max_tokens_to_generate: int = 64
    temperature: float = 0.7


@dataclass(kw_only=True, unsafe_hash=True)
class ShardingConfig:
    ## Model sharding params (args to jax.P)-- list of mesh_axis_names els or None
    # NOTE: technically jax.P can merge axes, e.g. (('x', 'y')), but we reject this
    mesh_shape: list[int] = MISSING
    mesh_axis_names: list[str] = field(default_factory=lambda: ["dp"])
    data: list[str | None] = field(default_factory=lambda: ["dp"])
    wqkv: list[str | None] = field(default_factory=list)  # D x 3 x N x H
    wo: list[str | None] = field(default_factory=list)  # D x N x H
    wup: list[str | None] = field(default_factory=list)  # D x 4D
    wdown: list[str | None] = field(default_factory=list)  # 4D x D
    mlp_hidden: list[str | None] = field(default_factory=list)  # S x 4D
    res_stream: list[str | None] = field(default_factory=list)  # S x D
    att_qkv: list[str | None] = field(default_factory=list)  # 3 x S x N x H


@dataclass(kw_only=True, unsafe_hash=True)
class Config:
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    sharding: ShardingConfig = field(default_factory=ShardingConfig)


def register_configs():
    cs = ConfigStore.instance()
    cs.store(group="experiment", name="base_experiment", node=ExperimentConfig)
    cs.store(group="dataset", name="base_dataset", node=DatasetConfig)
    cs.store(group="optimizer", name="base_optimizer", node=OptimizerConfig)
    cs.store(group="model", name="base_model", node=ModelConfig)
    cs.store(group="inference", name="base_inference", node=InferenceConfig)
    cs.store(group="sharding", name="base_sharding", node=ShardingConfig)
    cs.store(name="config", node=Config)


def mesh_from_config(config: Config):
    mesh = jax.make_mesh(
        config.sharding.mesh_shape,
        config.sharding.mesh_axis_names,
        len(config.sharding.mesh_shape) * (jax.sharding.AxisType.Explicit,),
    )
    return mesh


def config_post_init(config: Config):
    """Call after jax.distributed.initialize()"""
    # Register the argument's type as static (since hydra wraps Config)
    # We make everything above unsafe_hash=True to allow this!
    jax.tree_util.register_static(type(config))
    # Check arguments
    assert config.model.d_head % 2 == 0, (
        "Head dimension needs to be divisible by 2 for RoPE"
    )
    assert config.dataset.global_batch_size % jax.process_count() == 0, (
        "Number of hosts needs to divide the global batch size"
    )
