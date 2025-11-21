from dataclasses import dataclass, field
from enum import Enum

import jax
import jax.numpy as jnp
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING


class DType(Enum):
    """Different data types we can use. models.py"""

    FLOAT32 = jnp.float32
    FLOAT16 = jnp.float16
    BFLOAT16 = jnp.bfloat16
    INT32 = jnp.int32
    INT16 = jnp.int16


class SplitType(Enum):
    """Different splits for datasets. data.py"""

    TRAIN = "train"
    TEST = "test"
    VAL = "val"


class OptType(Enum):
    """Different optimizers we can use. optimizers.py"""

    ADAMW = "adamw"
    SGD = "sgd"


class LoggerType(Enum):
    """Different loggers we can use. loggers.py"""

    PRINT = "print"
    WANDB = "wandb"


class EvaluatorType(Enum):
    """Different evals we can run. evaluators.py"""

    AUTOREGRESSIVE_ROLLOUTS = 0
    ACCURACY = 1
    PERPLEXITY = 2
    NLL = 3


class TransformerType(Enum):
    """Are inputs tokens or vectors (ViT-like)? models.py"""

    DISCRETE = "discrete"
    CONTINUOUS = "continuous"


class DatasetName(Enum):
    """Supported datasets we train on. data.py"""

    NUMBER_STAIRCASE = "number_staircase"
    MNIST = "mnist"


@dataclass(kw_only=True, unsafe_hash=True)
class DatasetConfig:
    """Params for a single dataset"""

    name: DatasetName = MISSING
    path: str = MISSING
    split: SplitType = SplitType.VAL
    seq_len: int = MISSING
    global_batch_size: int = MISSING
    epochs_to_loop: int = -1  # -1 means indefinite; otherwise, fixed num epochs


@dataclass(kw_only=True, unsafe_hash=True)
class EvaluationConfig:
    """Params for a single evaluation"""

    dataset: DatasetConfig = MISSING
    evaluator: EvaluatorType = MISSING


@dataclass(kw_only=True, unsafe_hash=True)
class OptimizerConfig:
    """Optimizer params"""

    num_steps: int = 10**3  # if 0 or less, will train until dataloader exhausted
    type: OptType = OptType.ADAMW
    lr: float = 3e-4
    beta1: float = 0.9
    beta2: float = 0.999
    eps_adam: float = 1e-8
    weight_decay: float = 1e-2
    clip_grad: float = 1.0  # global ell^2 norm


@dataclass(kw_only=True, unsafe_hash=True)
class ModelConfig:
    """Model architecture params"""

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
    num_registers: int = 1  # Currently only used for EmbeddingContinuous
    num_vocab: int = MISSING  # input dim
    num_classes: int = MISSING  # output dim (equals input dim for text tf)

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


@dataclass(kw_only=True, unsafe_hash=True)
class InferenceConfig:
    """Autoregressive inference params"""

    max_tokens_to_generate: int = 64
    temperature: float = 0.7


@dataclass(kw_only=True, unsafe_hash=True)
class ShardingConfig:
    """Model sharding params (args to jax.P)-- list of mesh_axis_names els or None"""

    # NOTE: technically jax.P can merge axes, e.g. (('x', 'y')), but we reject this
    mesh_shape: list[int] = MISSING
    mesh_axis_names: list[str] = field(default_factory=lambda: ["dp"])
    data: list[str | None] = field(default_factory=lambda: ["dp"])
    wqkv: list[str | None] = field(default_factory=list)  # D x 3 x N x H
    wo: list[str | None] = field(default_factory=list)  # D x N x H
    wup: list[str | None] = field(default_factory=list)  # D x 4D
    wdown: list[str | None] = field(default_factory=list)  # 4D x D
    mlp_hidden: list[str | None] = field(default_factory=list)  # 4D
    res_stream: list[str | None] = field(default_factory=list)  # D
    att_qkv: list[str | None] = field(default_factory=list)  # 3 x S x N x H


@dataclass(kw_only=True, unsafe_hash=True)
class Config:
    """Overall config class, containing top-level orchestration parameters"""

    seed: int = 1337
    logger_type: LoggerType = LoggerType.WANDB
    project_name: str = "bmgpt-debug"
    run_name: str = ""

    train_dataset: DatasetConfig = MISSING
    eval_list: list[EvaluationConfig] = MISSING

    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    sharding: ShardingConfig = field(default_factory=ShardingConfig)


def register_configs():
    cs = ConfigStore.instance()
    cs.store(group="optimizer", name="base_optimizer", node=OptimizerConfig)
    cs.store(group="model", name="base_model", node=ModelConfig)
    cs.store(group="inference", name="base_inference", node=InferenceConfig)
    cs.store(group="sharding", name="base_sharding", node=ShardingConfig)
    cs.store(name="base_config", node=Config)


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
    assert config.train_dataset.global_batch_size % jax.process_count() == 0 and all(
        eval.dataset.global_batch_size % jax.process_count() == 0
        for eval in config.eval_list
    ), "Number of hosts needs to divide the global batch size for all data"
