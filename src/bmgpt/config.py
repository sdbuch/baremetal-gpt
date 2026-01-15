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
  SHAKESPEARE = "tiny-shakespeare"
  DCLM = "dclm"


class TokenizerType(Enum):
  IDENTITY = 0
  GPT2 = 1
  LLAMA3 = 2


@dataclass(kw_only=True, unsafe_hash=True)
class DatasetConfig:
  """Params for a single dataset. data.py"""

  name: DatasetName = MISSING
  path: str = MISSING
  split: SplitType = SplitType.VAL
  seq_len: int = MISSING
  max_valid_token_id: int = MISSING  # For padded embeddings, mask invalid token ids

  # microbatch_size = global_batch_size / num_microbatches
  global_batch_size: int = MISSING
  num_microbatches: int = 1

  ## Duration parameters
  # For some datasets, epochs makes sense; others are too large.
  # We can run for a specified number of epochs, or a specified number of batches
  #  (or both.)
  # But the dataloader being used might limit some possibilities; see data.py
  epochs_to_loop: int = -1  # -1 means indefinite; otherwise, fixed num epochs
  num_steps: int = 10**3  # if 0 or less, will train until dataloader exhausted

  ## Splash Attention kernel parameters (dataset-specific)
  use_splash: bool = True  # use splash attention pallas kernel or jax-xla attention
  splash_block_size_q: int = 128
  splash_block_size_kv: int = 128
  splash_block_size_kv_compute: int = 128


@dataclass(kw_only=True, unsafe_hash=True)
class EvaluationConfig:
  """Params for a single evaluation. evaluators.py"""

  dataset: DatasetConfig = MISSING
  evaluator: EvaluatorType = MISSING


@dataclass(kw_only=True, unsafe_hash=True)
class OptimizerConfig:
  """Optimizer params. optimizers.py"""

  type: OptType = OptType.ADAMW
  lr: float = 3e-4
  beta1: float = 0.9
  beta2: float = 0.999
  eps_adam: float = 1e-8
  weight_decay: float = 1e-2
  clip_grad: float = 1.0  # global ell^2 norm


@dataclass(kw_only=True, unsafe_hash=True)
class ModelConfig:
  """Model architecture params. model.py"""

  # Overarching
  transformer_type: TransformerType = MISSING

  # Transformer-type-agnostic params
  is_causal: bool = True
  d_model: int = 768
  num_heads: int = 12
  d_head: int = 64
  mlp_factor: float = 8 / 3
  num_layers: int = 12
  param_std: float = 0.02
  rope_theta: float = 10000.0
  max_seq_len: int = 1024
  num_registers: int = 1  # Currently only used for EmbeddingContinuous
  num_vocab: int = MISSING  # input dim
  num_classes: int = MISSING  # output dim (equals input dim for text tf)

  # Model dtypes
  # Select operations (layernorm, logits, rope) always done in FP32
  param_dtype: DType = DType.BFLOAT16  # master weights dtype
  opt_dtype: DType = DType.FLOAT32  # optimizer state dtype

  # Model call-time params
  use_bias_embeddings: bool = False  # bias in emb / unemb
  eps_ln: float = 1e-6  # epsilon for layer norm
  use_centering_ln: bool = False  # layer norm or RMS norm
  use_bias_ln: bool = False  # bias in layernorm/RMSnorm
  use_gating_mlp: bool = True  # Gated MLP or standard
  use_bias_mlp: bool = False  # bias in MLPs
  use_rope: bool = True  # RoPE or not

  # Discrete-specific model parameters
  # Continuous-specific model parameters


@dataclass(kw_only=True, unsafe_hash=True)
class InferenceConfig:
  """Autoregressive inference params. sample.py"""

  max_tokens_to_generate: int = 64
  temperature: float = 0.7
  tokenizer: TokenizerType = TokenizerType.IDENTITY


@dataclass(kw_only=True, unsafe_hash=True)
class ShardingConfig:
  """Model sharding params (args to jax.P)-- list of mesh_axis_names els or None.

  Note: activation shardings go as deep as possible (no batch/seq for MLPs).
  See configs/ directory for examples (dp, fsdp).
  """

  # NOTE: technically jax.P can merge axes, e.g. (('x', 'y')), but hydra rejects this
  mesh_shape: list[int] = MISSING
  mesh_axis_names: list[str] = MISSING
  # Parameter sharding specs
  wqkv: list[str | None] = MISSING  # D x 3 x N x H
  wo: list[str | None] = MISSING  # H x N x D
  wup: list[str | None] = MISSING  # D x F
  wdown: list[str | None] = MISSING  # F x D
  wemb: list[str | None] = MISSING  # V x D
  wunemb: list[str | None] = MISSING  # D x V
  # Activation sharding specs
  data: list[str | None] = MISSING  # M (accum_steps x micro_bs = global_bs)
  mlp_hidden: list[str | None] = MISSING  # F
  res_stream: list[str | None] = MISSING  # D
  att_qkv: list[str | None] = MISSING  # 3 x S x N x H


@dataclass(kw_only=True, unsafe_hash=True)
class Config:
  """Overall config class, containing top-level orchestration parameters"""

  seed: int = 1337
  logger_type: LoggerType = LoggerType.WANDB
  project_name: str = "bmgpt-debug"
  run_name: str = ""
  val_log_interval: int = 1000  # log validation metrics every <this many> batches
  use_fused_xent_loss: bool = True  # requires vocab size divisible by 128
  fused_xent_block_size_T: int = 512  # block size along batch axis
  fused_xent_block_size_V: int = 512  # block size along vocab axis
  fused_xent_block_size_V_compute: int = 512  # block size along vocab axis for dots

  train_dataset: DatasetConfig = MISSING
  val_list: list[EvaluationConfig] = field(default_factory=list)  # validation metrics
  eval_list: list[EvaluationConfig] = field(default_factory=list)

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
  """Input validation. Call after jax.distributed.initialize()"""
  # model arguments
  assert config.model.d_head % 2 == 0, (
    "Head dimension needs to be divisible by 2 for RoPE"
  )
  # batch size checking
  num_hosts = jax.process_count()

  def check_batch_size_division(dataset: DatasetConfig):
    assert dataset.global_batch_size % dataset.num_microbatches == 0, (
      f"Number of gradient accumulation steps must divide global batch size {dataset}"
    )
    assert (dataset.global_batch_size // dataset.num_microbatches) % num_hosts == 0, (
      f"Number of hosts needs to divide microbatch size {dataset}"
    )

  check_batch_size_division(config.train_dataset)
  for eval in config.val_list:
    check_batch_size_division(eval.dataset)
  for eval in config.eval_list:
    check_batch_size_division(eval.dataset)
