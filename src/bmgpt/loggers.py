from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any, cast

import jax
import wandb
from omegaconf import DictConfig, OmegaConf

from bmgpt.config import Config, LoggerType


def logger_factory(logger_type: LoggerType):
  match logger_type:
    case LoggerType.PRINT:
      return PrintLogger
    case LoggerType.WANDB:
      return WandbLogger


def get_run_name(base: str) -> str:
  """Prepend current UTC time to config-specified run name"""
  timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
  run_name = f"{timestamp}-{base}"
  return run_name


class Logger:
  """Logger base class. Implements a depth-1 buffer for device-host pipelining"""

  def __init__(self, config: Config):
    self.project_name = config.project_name
    self.run_name = get_run_name(config.run_name)
    if isinstance(config, DictConfig):
      config_dict = cast(dict[str, Any], OmegaConf.to_container(config, resolve=True))
    else:
      config_dict = asdict(config)
    self.config = config_dict
    self.prev_log_data = None

  def __enter__(self):
    return self

  def __exit__(self, exc_type, exc_value, traceback):
    return False

  def log(self, log_dict: dict):
    pass

  def warn(self, message: str):
    pass

  def buffer(self, log_dict: dict) -> dict | None:
    self.prev_log_data, buffered_metrics = log_dict, self.prev_log_data
    return buffered_metrics

  def flush_buffer(self):
    if self.prev_log_data is not None:
      self.log({})
    self.prev_log_data = None


class PrintLogger(Logger):
  def __init__(self, config: Config):
    super().__init__(config)

  def __enter__(self):
    print(f"Project: {self.project_name}")
    print(f"Run: {self.run_name}")
    return super().__enter__()

  def log(self, log_dict: dict):
    if (buffered_dict := self.buffer(log_dict)) is None:
      return
    print(*[f"{metric}: {val}" for metric, val in buffered_dict.items()], sep="\t")

  def warn(self, message: str):
    print(message)


class WandbLogger(Logger):
  def __init__(self, config: Config):
    super().__init__(config)
    self.is_master = jax.process_index() == 0

  def __enter__(self):
    if self.is_master:
      wandb.init(project=self.project_name, name=self.run_name, config=self.config)
    return super().__enter__()

  def __exit__(self, exc_type, exc_value, traceback):
    if self.is_master:
      wandb.finish()
    return super().__exit__(exc_type, exc_value, traceback)

  def log(self, log_dict: dict):
    if self.is_master:
      if (buffered_dict := self.buffer(log_dict)) is None:
        return
      wandb.log(buffered_dict)

  def warn(self, message: str):
    if self.is_master:
      print(message)
